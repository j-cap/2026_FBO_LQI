"""Read-only audit and consolidation of the frozen D256 confirmation campaign.

Checks each paid policy separately, records cross-runtime decision differences,
recomputes held-out scores, and verifies every saved numerical table. No new
rollout or alternative policy campaign is run. Source results are never rewritten.
"""
import argparse
from pathlib import Path
import platform
import subprocess
import tempfile

import tracking_confirmation as runner
import numpy as np
import pandas as pd
import scipy

from analyze_tracking_hierarchical import compare_saved
from src.tracking_confirmation import METHODS, evaluate_job, paired_tables
from src.tracking_confirmation_report import write_report
from src.tracking_budget import Posterior,feasibility_probability
from src.tracking_online import LocalCostModel,features
from src.tracking_online_lcb import acquisition_values
from src.tracking_peer_cost import OBSERVATION_FIELDS
from src.tracking_spectral import run_fleet

OUT=runner.REPO/'docs/analysis/spectral_confirmation'


class DecisionDifference(Exception):
    pass


def replay_audit_worker(job):
    """Audit independent arms so one backend-sensitive GP does not hide others."""
    state=runner.WORKER;result=runner.read(runner.run_path(state,job));clients=result['clients']
    checks,differences=[],[]
    for methods in ([METHODS[0]],[METHODS[1]],METHODS[2:]):
        current=None
        def checkpoint(data):
            nonlocal current
            current=data
        def recorded(plans,cache):
            rows=[]
            for plan in plans:
                event=result['state']['methods'][plan['method']]['events'][(plan['round']-1)*len(clients)+plan['client']]
                diff={k:[plan[k],event[k]] for k in plan if plan[k]!=event[k]}
                if diff:
                    if (set(diff)-{'action','candidate'} or plan['method'] not in METHODS[:2] or
                        plan['mode']!='feasibility_penalized_LCB' or current is None):
                        raise RuntimeError('Non-numerical confirmation replay/protocol difference: '+str(diff))
                    i=plan['client'];data=current['methods'][plan['method']];history=data['histories'][i]
                    prior=np.zeros(len(state['domain'].actions))
                    if plan['method']==METHODS[1]:
                        phi=features(state['domain'],state['cfg']['model'])
                        prior=phi@LocalCostModel(history,phi,state['cfg']['model']['ridge']).fit()
                    posterior=Posterior(state['domain'],history,prior,state['cfg']['gp'],data['gp_parameters'][i],False)
                    mean,variance=posterior.predict()
                    probability=feasibility_probability(history,state['domain'],np.full(len(mean),.5),state['cfg']['gp'])
                    values=acquisition_values(mean,variance,probability,state['cfg']['acquisition'])
                    differences.append(dict(**job,method=plan['method'],round=plan['round'],client=i,
                        computed_candidate=plan['candidate'],recorded_candidate=event['candidate'],
                        computed_score=float(values[plan['action']]),recorded_score=float(values[event['action']]),
                        recorded_minus_computed_score=float(values[event['action']]-values[plan['action']]),
                        declared_tie_tolerance=1e-12*max(1.,abs(float(values[plan['action']]))),
                        eligible_training_count=sum(o['eligible'] for o in history),
                        fitted_log_parameters=data['gp_parameters'][i]))
                    raise DecisionDifference()
                rows.append({k:event[k] for k in OBSERVATION_FIELDS if k!='action'})
            return rows
        exact=True
        try:
            actual=run_fleet(state['domain'],dict(state['cfg'],methods=methods),clients,job['seed'],
                recorded,state['phis'],checkpoint=checkpoint)
            for method in methods:
                if actual['methods'][method]['events']!=result['state']['methods'][method]['events']:
                    raise RuntimeError('Recorded confirmation event replay differs.')
                if method in METHODS[2:]:
                    for a,b in zip(actual['methods'][method]['learning'],result['state']['methods'][method]['learning']):
                        for k in ('sharing_weights','component_log_evidence','expected_sharing_fraction'):
                            np.testing.assert_allclose(a[k],b[k],rtol=1e-10,atol=1e-10)
        except DecisionDifference:
            exact=False
        checks.extend(dict(**job,method=method,exact_decision_replay=exact) for method in methods)
    runner.save(Path(state['replay_audit_out'])/runner.run_path(state,job).name,
        dict(checks=checks,first_differences=differences))


def verify(workers):
    state=runner.setup()
    manifest=runner.read(state['out']/'run_manifest.json')
    if (manifest['signature']!=state['signature'] or
        runner.read(state['out']/'completion.json')!=dict(signature=state['signature'],jobs=40)):
        raise RuntimeError('Completed confirmation signature or job count differs.')
    if (manifest['mode']!='full' or manifest['historical_training_records']!=0 or
        manifest['worker_reference_access'] or manifest['worker_evaluation_access'] or
        manifest['configuration']['methods']!=METHODS or len(manifest['physical_draws'])!=90):
        raise RuntimeError('Confirmation information or physical-draw contract differs.')
    results=runner.completed_live(state,replay=False,workers=workers)
    replay_checks,replay_differences=[],[]
    with tempfile.TemporaryDirectory(prefix='confirmation_per_arm_replay_') as directory:
        payload=dict(runner.payload(state),replay_audit_out=directory)
        runner.parallel(replay_audit_worker,state['jobs'],payload,workers,'Independent paid-arm replay')
        for job in state['jobs']:
            record=runner.read(Path(directory)/runner.run_path(state,job).name)
            replay_checks.extend(record['checks']);replay_differences.extend(record['first_differences'])
    replay_checks=pd.DataFrame(replay_checks);replay_differences=pd.DataFrame(replay_differences)
    print('Exact arm/job replays: '+str(int(replay_checks.exact_decision_replay.sum()))+'/'+str(len(replay_checks)),flush=True)
    # All evaluator reads occur after the live decisions have passed replay.
    chosen=runner.load_calibration(state)
    selection=runner.validation_selection(state,results,chosen)
    validation_tasks=runner.evaluation_tasks(state,'validation',selection)
    references,validation=runner.load_validation(state,validation_tasks,chosen)
    frames=[[],[],[],[]]
    for job,result in zip(state['jobs'],results):
        group=job['fleet_id']
        refs={(p,c):v for (g,p,c),v in references.items() if g==group}
        vals={(p,c,k):v for (g,p,c,k),v in validation.items() if g==group}
        for target,frame in zip(frames,evaluate_job(job,result,state['domain'],state['cfg'],refs,vals)):
            target.append(frame)
    trials,cells,queries,learning=[pd.concat(items,ignore_index=True) for items in frames]
    paired,units=paired_tables(trials,state['cfg'],state['requested'],'full')
    reference_rows=pd.DataFrame([dict(group=g,plant_id=p,context=c,**value) for (g,p,c),value in references.items()])
    comparisons=[
        (trials,'budget_trials',['condition','fleet_id','seed','method','fleet_budget','client']),
        (cells,'budget_contexts',['condition','fleet_id','seed','method','fleet_budget','client','context']),
        (queries,'query_observations',['condition','fleet_id','seed','method','fleet_evaluation']),
        (learning,'learning_diagnostics',['condition','fleet_id','seed','method','round']),
        (paired,'paired_comparisons',['condition','fleet_budget','method','baseline']),
        (units,'replication_unit_comparisons',['condition','fleet_budget','method','baseline','replication_unit']),
        (reference_rows,'empirical_references',['group','plant_id','context'])]
    for frame,name,keys in comparisons:compare_saved(frame,state['out']/(name+'.csv'),keys)
    with tempfile.TemporaryDirectory(prefix='confirmation_numeric_audit_') as directory:
        temporary=Path(directory)
        write_report(temporary,trials,cells,queries,paired,units,state['cfg'],'full')
        regenerated=sorted(temporary.glob('*.csv'))
        for path in regenerated:
            frame=pd.read_csv(path,float_precision='round_trip')
            keys=[k for k in ('condition','fleet_id','seed','method','baseline','fleet_budget','replication_unit',
                'family','context','mode') if k in frame.columns]
            compare_saved(frame,state['out']/path.name,keys)
    if (len(trials),len(cells),len(queries),len(learning))!=(8640,51840,34560,3840):
        raise RuntimeError('Confirmation denominators differ.')
    counts=queries.groupby(['condition','fleet_id','seed','method','client']).size()
    if not (counts==24).all() or len(counts)!=1440:
        raise RuntimeError('A paid local cap differs.')
    if not (paired.replication_units==10).all():
        raise RuntimeError('Confirmation resampled clients or runs instead of ten units.')
    hetero=units[units.condition=='heterogeneous']
    homo=units[units.condition=='homogeneous']
    if not (hetero.programme_runs==3).all() or not (homo.programme_runs==1).all():
        raise RuntimeError('Replication-unit nesting differs.')
    cal_tasks=runner.evaluation_tasks(state,'calibration')
    expected=dict(signature=state['signature'],all_four_method_decisions_replayed=True,
        online_requests=len(queries),all_local_caps_respected=True,
        unavailable_reference_contexts=sum(not r['eligible'] for r in references.values()),
        unresolved_online_requests=int((~queries.integration_resolved).sum()),
        calibration_rollouts=sum(len(t['candidates']) for t in cal_tasks),
        validation_rollouts=sum(len(t['candidates']) for t in validation_tasks),
        old_rollouts_used_for_learning_or_evaluation=0,public_bases=state['public_bases'])
    if runner.read(state['out']/'verification.json')!=expected:
        raise RuntimeError('Saved completion verification differs.')
    print(f'Checked {len(comparisons)+len(regenerated)} saved numeric tables and '
        f'{len(cal_tasks)+len(validation_tasks)} evaluator batches.',flush=True)
    return state,results,trials,cells,queries,learning,paired,units,reference_rows,cal_tasks,validation_tasks,len(comparisons)+len(regenerated),replay_checks,replay_differences


def descriptive_tables(trials,cells,queries,learning,paired,units):
    final=trials.fleet_budget.max()
    policies=trials[trials.fleet_budget==final]
    contexts=cells[cells.fleet_budget==final]
    outcomes=policies.groupby(['condition','method'],sort=False).agg(
        client_runs=('client','size'),eligible_policies=('fully_feasible','sum'),
        context_success_rate=('context_success','mean'),conditional_mean_J=('J','mean'),
        mean_ineligible_tests_per_client=('ineligible_queries','mean')).reset_index()
    coverage=contexts.groupby(['condition','method'],sort=False).agg(
        context_trials=('success','size'),successes=('success','sum'),missing_local_gains=('has_local_eligible_gain',lambda x:int((~x).sum())),
        validation_eligible_contexts=('eligible','sum'),reference_contexts=('reference_available','sum')).reset_index()
    outcomes=outcomes.merge(coverage,on=['condition','method'],validate='one_to_one')
    by_context=contexts.groupby(['condition','method','context'],sort=False).agg(
        context_trials=('success','size'),successes=('success','sum'),eligible_contexts=('eligible','sum'),
        missing_local_gains=('has_local_eligible_gain',lambda x:int((~x).sum())),conditional_mean_J=('J','mean')).reset_index()
    family=policies.groupby(['condition','family','method'],sort=False).agg(
        client_runs=('client','size'),eligible_policies=('fully_feasible','sum'),context_success=('context_success','mean'),
        conditional_mean_J=('J','mean')).reset_index()
    rows=[]
    for (condition,budget),group in trials.groupby(['condition','fleet_budget'],sort=False):
        for baseline in METHODS[:2]+[METHODS[3]]:
            a=group[group.method==METHODS[2]].set_index(['fleet_id','seed','client']).sort_index()
            b=group[group.method==baseline].set_index(['fleet_id','seed','client']).sort_index()
            if not a.index.equals(b.index):raise RuntimeError('Unpaired policy coverage analysis.')
            good=a.fully_feasible & b.fully_feasible
            per_run=[]
            for (fleet,seed),g in a.groupby(level=['fleet_id','seed'],sort=False):
                av=a.loc[g.index];bv=b.loc[g.index];common=av.fully_feasible & bv.fully_feasible
                per_run.append(dict(condition=condition,fleet_budget=budget,fleet_id=fleet,seed=seed,
                    method=METHODS[2],baseline=baseline,context_success_delta=float((av.context_success-bv.context_success).mean()),
                    eligible_method=int(av.fully_feasible.sum()),eligible_baseline=int(bv.fully_feasible.sum()),
                    jointly_eligible_pairs=int(common.sum()),
                    mean_J_delta_joint=float((av.loc[common,'J']-bv.loc[common,'J']).mean()) if common.any() else None))
            rows.extend(per_run)
    per_run=pd.DataFrame(rows)
    transitions=[]
    for condition,group in policies.groupby('condition',sort=False):
        for baseline in METHODS[:2]+[METHODS[3]]:
            a=group[group.method==METHODS[2]].set_index(['fleet_id','seed','client']).sort_index()
            b=group[group.method==baseline].set_index(['fleet_id','seed','client']).sort_index()
            transitions.append(dict(condition=condition,method=METHODS[2],baseline=baseline,
                both_eligible=int((a.fully_feasible & b.fully_feasible).sum()),
                method_only_eligible=int((a.fully_feasible & ~b.fully_feasible).sum()),
                baseline_only_eligible=int((~a.fully_feasible & b.fully_feasible).sum()),
                both_ineligible=int((~a.fully_feasible & ~b.fully_feasible).sum())))
    modes=queries.groupby(['condition','method','mode'],sort=False).agg(
        requests=('eligible','size'),eligible_requests=('eligible','sum'),new_incumbents=('became_incumbent','sum')).reset_index()
    sharing=learning[(learning.method==METHODS[2]) & (learning['round']==23)][
        ['condition','fleet_id','replicate','seed','round','expected_sharing_fraction','sharing_weights']].copy()
    return dict(final_outcomes=outcomes,final_comparisons=paired[paired.fleet_budget==final],
        final_replication_units=units[units.fleet_budget==final],budget_comparisons=paired,
        per_run_final_comparisons=per_run[per_run.fleet_budget==final],
        final_policy_coverage_transitions=pd.DataFrame(transitions),final_context_outcomes=by_context,
        final_family_outcomes=family,query_modes=modes,sharing_before_final_query=sharing,
        final_missing_contexts=contexts[~contexts.has_local_eligible_gain],
        final_validation_failures=contexts[contexts.has_local_eligible_gain & ~contexts.eligible],
        unresolved_online_requests=queries[~queries.integration_resolved])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    if args.workers<1:parser.error('--workers must be positive')
    state,results,trials,cells,queries,learning,paired,units,refs,cal_tasks,val_tasks,checked,replay_checks,replay_differences=verify(args.workers)
    outputs=descriptive_tables(trials,cells,queries,learning,paired,units)
    OUT.mkdir(parents=True,exist_ok=True)
    for name,frame in outputs.items():frame.to_csv(OUT/(name+'.csv'),index=False)
    replay_checks.to_csv(OUT/'per_arm_replay_checks.csv',index=False)
    replay_differences.to_csv(OUT/'first_replay_differences.csv',index=False)
    paths=[state['out']/n for n in ('run_manifest.json','completion.json','public_bases.npz','verification.json','preflight.json')]
    paths += [runner.run_path(state,j) for j in state['jobs']]
    paths += [runner.evaluation_path(state,t) for t in cal_tasks+val_tasks]
    paths += sorted(state['out'].glob('*.csv'))
    final=cells[cells.fleet_budget==216]
    runner.save(OUT/'verification.json',dict(result_signature=state['signature'],jobs=40,
        result_commit=subprocess.check_output(['git','log','-1','--format=%H','--',state['out'].relative_to(runner.REPO).as_posix()],cwd=runner.REPO,text=True).strip(),
        all_protocol_contracts_validated=True,all_four_paid_policies_exactly_replayed=bool(replay_checks.exact_decision_replay.all()),
        exact_arm_job_replays=int(replay_checks.exact_decision_replay.sum()),arm_job_replays=len(replay_checks),
        original_runtime_replay_record=runner.read(state['out']/'verification.json')['all_four_method_decisions_replayed'],
        audit_runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,system=platform.system()),
        recorded_runtime=runner.read(state['out']/'run_manifest.json')['runtime'],
        paid_online_requests=len(queries),local_query_cap=24,
        evaluator_batches_verified=len(cal_tasks)+len(val_tasks),
        calibration_rollouts=sum(len(t['candidates']) for t in cal_tasks),validation_rollouts=sum(len(t['candidates']) for t in val_tasks),
        numeric_tables_recomputed=checked,physical_reference_contexts=len(refs),unavailable_references=int((~refs.eligible).sum()),
        unresolved_online_requests=int((~queries.integration_resolved).sum()),
        selected_final_gains_failing_validation=int((final.has_local_eligible_gain & ~final.eligible).sum()),
        homogeneous_units=10,heterogeneous_units=10,heterogeneous_runs_per_unit=3,
        new_rollouts=0,new_model_variants=0,new_deployed_controllers=0,
        analysis_source_sha256=runner.digest(Path(__file__)),
        input_hashes={p.relative_to(runner.REPO).as_posix():runner.digest(p) for p in paths}))
    print(outputs['final_outcomes'].to_string(index=False))
    print('Audit and analysis: '+str(OUT),flush=True)


if __name__=='__main__':main()
