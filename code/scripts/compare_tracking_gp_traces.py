"""Compare original/Linux GP fits and replay their recorded states without refits.

Only paid prior observations and logged fit parameters are used. Experimental
controllers, outcomes, budgets and frozen analysis tables are never changed.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from unittest.mock import patch

import tracking_confirmation as runner
import numpy as np
import pandas as pd
import scipy

from src.tracking_budget import Posterior,feasibility_probability
from src.tracking_online import seed_for
from src.tracking_online_lcb import acquisition_values,choose

SOURCE=runner.REPO/'docs/analysis/spectral_confirmation'
OUT=runner.REPO/'docs/analysis/gp_replay_closure'
KEYS=['condition','fleet_id','replicate','seed','method','round','client','context']


def logged_state_prediction(history,domain,context,cfg,params,seed,client,round_id):
    """Frozen acquisition code with the recorded hyperparameters, no optimizer."""
    count=sum(o['eligible'] for o in history)
    prior=np.zeros(len(domain.actions))
    action,used,_,mode=choose(history,domain,context,prior,cfg['gp'],cfg['acquisition'],params,count,
        np.random.default_rng(seed_for('acquisition',seed,client,round_id-1)))
    np.testing.assert_array_equal(used,params)
    if mode!='feasibility_penalized_LCB':raise RuntimeError('Acquisition mode differs.')
    posterior=Posterior(domain,history,prior,cfg['gp'],params,False)
    mean,variance=posterior.predict()
    probability=feasibility_probability(history,domain,np.full(len(mean),.5),cfg['gp'])
    scores=acquisition_values(mean,variance,probability,cfg['acquisition'])
    return dict(action=action,mean=mean,variance=variance,probability=probability,scores=scores,
        log_likelihood=float(-posterior._loss(np.asarray(params))))


def main():
    state=runner.setup();cfg=state['cfg']
    om=runner.read(SOURCE/'gp_trace_original_runtime.json')
    lm=runner.read(SOURCE/'gp_trace_linux_audit_runtime.json')
    trace_code=runner.REPO/'code/scripts/trace_tracking_confirmation_gp.py'
    for metadata in (om,lm):
        if (metadata['result_signature']!=state['signature'] or not metadata['recorded_path_only'] or
            metadata['new_rollouts']!=0 or metadata['source_sha256']!=runner.digest(trace_code)):
            raise RuntimeError('GP trace provenance differs.')
    if om['programme_seeds']!=lm['programme_seeds']:
        raise RuntimeError('Trace job selection differs.')
    original=pd.read_csv(SOURCE/'gp_trace_original.csv',float_precision='round_trip').sort_values(KEYS).reset_index(drop=True)
    linux=pd.read_csv(SOURCE/'gp_trace_linux_audit.csv',float_precision='round_trip').sort_values(KEYS).reset_index(drop=True)
    if original.duplicated(KEYS).any() or linux.duplicated(KEYS).any() or not original[KEYS].equals(linux[KEYS]):
        raise RuntimeError('GP trace decisions are not uniquely paired.')
    for column in ['recorded_candidate','eligible_training_count','last_fit_eligible_count','fitted_this_query','training_fingerprint']:
        if not original[column].equals(linux[column]):raise RuntimeError('Prior-history trace mismatch: '+column)
    if len(original)!=om['bo_decisions'] or len(linux)!=lm['bo_decisions']:
        raise RuntimeError('Trace row count differs.')
    if int((~original.exact_candidate_match).sum())!=om['candidate_differences'] or int((~linux.exact_candidate_match).sum())!=lm['candidate_differences']:
        raise RuntimeError('Trace candidate-difference count differs.')
    jobs=[j for j in state['jobs'] if j['seed'] in om['programme_seeds']]
    results={}
    for job in jobs:
        result=runner.read(runner.run_path(state,job))
        runner.validate_job(state,job,result,complete=True)
        results[job['seed']]=result
    expected_keys=set()
    for job in jobs:
        for event in results[job['seed']]['state']['methods']['peer_gp_bo']['events']:
            if event['mode']=='feasibility_penalized_LCB':
                expected_keys.add(tuple(dict(**job,method='peer_gp_bo',round=event['round'],
                    client=event['client'],context=event['context'])[key] for key in KEYS))
    if set(original[KEYS].itertuples(index=False,name=None))!=expected_keys:
        raise RuntimeError('The traces omit or add a paid BO decision in the selected jobs.')
    with np.load(state['out']/'public_bases.npz',allow_pickle=False) as saved:
        pinned_X=saved['domain_X'].copy()
    max_input_roundoff=float(np.max(np.abs(state['domain'].X-pinned_X)))
    domain=deepcopy(state['domain']);domain.X=pinned_X
    fits={};rows=[]
    # This rejects any accidental hyperparameter refit during state verification.
    with patch('src.tracking_budget.minimize',side_effect=RuntimeError('Refitting is forbidden during logged-state replay')):
        for index,record in original.iterrows():
            other=linux.iloc[index];result=results[int(record.seed)];i=int(record.client);round_id=int(record['round'])
            data=result['state']['methods'][record.method]
            event=data['events'][(round_id-1)*len(result['clients'])+i]
            if (event['mode']!='feasibility_penalized_LCB' or event['candidate']!=record.recorded_candidate or
                event['context']!=record.context or record.method!='peer_gp_bo'):
                raise RuntimeError('Trace differs from the frozen paid event.')
            history=data['histories'][i][:round_id-1]
            training=[(o['action'],o['J']) for o in history if o['eligible']]
            fingerprint=hashlib.sha256(json.dumps(training).encode()).hexdigest()
            if fingerprint!=record.training_fingerprint or len(training)!=record.eligible_training_count:
                raise RuntimeError('Trace contains different or later cost observations.')
            params=np.asarray(json.loads(record.fitted_log_parameters),float)
            linux_params=np.asarray(json.loads(other.fitted_log_parameters),float)
            if params.shape!=(domain.X.shape[1]+1,) or not np.isfinite(params).all():
                raise RuntimeError('Invalid logged GP parameters.')
            bounds=np.log([cfg['gp']['length_bounds']]*domain.X.shape[1]+[cfg['gp']['amplitude_bounds']])
            if np.any(params<bounds[:,0]-1e-12) or np.any(params>bounds[:,1]+1e-12):
                raise RuntimeError('Logged GP fit violates the frozen bounds.')
            fit_key=(int(record.seed),i)
            previous_count,previous_params=fits.get(fit_key,(0,np.log([*cfg['gp']['length_scales'],cfg['gp']['amplitude']])))
            refit=len(training)>=4 and len(training)-previous_count>=cfg['gp']['optimize_every']
            expected_count=len(training) if refit else previous_count
            if bool(record.fitted_this_query)!=refit or int(record.last_fit_eligible_count)!=expected_count:
                raise RuntimeError('Logged GP refit schedule differs.')
            if not refit:np.testing.assert_allclose(params,previous_params,rtol=0,atol=1e-13)
            fits[fit_key]=(expected_count,params)
            client=result['clients'][i]['id']
            op=logged_state_prediction(history,domain,record.context,cfg,params,int(record.seed),client,round_id)
            lp=logged_state_prediction(history,domain,record.context,cfg,linux_params,int(record.seed),client,round_id)
            action=event['action']
            row={k:record[k] for k in KEYS}
            row.update(recorded_candidate=record.recorded_candidate,original_computed_candidate=record.computed_candidate,
                linux_computed_candidate=other.computed_candidate,
                original_trace_matches_recorded=bool(record.exact_candidate_match),linux_refit_matches_recorded=bool(other.exact_candidate_match),
                logged_original_state_candidate=domain.actions[op['action']][1],
                logged_original_state_matches_recorded=domain.actions[op['action']][1]==record.recorded_candidate,
                logged_linux_state_candidate=domain.actions[lp['action']][1],
                logged_linux_state_matches_linux_trace=domain.actions[lp['action']][1]==other.computed_candidate,
                max_log_parameter_difference=float(np.max(np.abs(params-linux_params))),
                fitted_this_query=bool(record.fitted_this_query),eligible_training_count=len(training),
                original_state_score_error=float(op['scores'][action]-record.recorded_score),
                original_state_mean_error=float(op['mean'][action]-record.recorded_mean_log_J),
                original_state_variance_error=float(op['variance'][action]-record.recorded_latent_variance),
                original_state_eligibility_error=float(op['probability'][action]-record.recorded_eligibility_probability),
                original_state_likelihood_error=float(op['log_likelihood']-record.log_likelihood),
                original_state_log_likelihood=op['log_likelihood'],linux_state_log_likelihood=lp['log_likelihood'],
                original_minus_linux_log_likelihood=op['log_likelihood']-lp['log_likelihood'],
                linux_state_recorded_score_error=float(lp['scores'][action]-other.recorded_score))
            rows.append(row)
    comparison=pd.DataFrame(rows)
    OUT.mkdir(parents=True,exist_ok=True)
    comparison.to_csv(OUT/'trace_comparison.csv',index=False)
    comparison[~comparison.linux_refit_matches_recorded].to_csv(OUT/'changed_decisions.csv',index=False)
    grouped=comparison.groupby('seed',sort=True).agg(
        bo_decisions=('round','size'),original_trace_matches=('original_trace_matches_recorded','sum'),
        linux_refit_differences=('linux_refit_matches_recorded',lambda x:int((~x).sum())),
        original_state_matches=('logged_original_state_matches_recorded','sum'),
        linux_state_matches=('logged_linux_state_matches_linux_trace','sum'),
        max_parameter_difference=('max_log_parameter_difference','max')).reset_index()
    grouped.to_csv(OUT/'job_summary.csv',index=False)
    sources=[SOURCE/n for n in ('gp_trace_original.csv','gp_trace_original_runtime.json',
        'gp_trace_linux_audit.csv','gp_trace_linux_audit_runtime.json','verification.json')]
    sources += [runner.run_path(state,j) for j in jobs]
    sources += [state['out']/'public_bases.npz',trace_code]
    errors={c:float(comparison[c].abs().max()) for c in comparison if c.endswith('_error')}
    verification=dict(result_signature=state['signature'],original_trace_commit=subprocess.check_output(
        ['git','log','-1','--format=%H','--',(SOURCE/'gp_trace_original.csv').relative_to(runner.REPO).as_posix()],cwd=runner.REPO,text=True).strip(),
        jobs=len(jobs),paired_bo_decisions=len(comparison),all_prior_histories_and_refit_counters_match=True,
        all_paid_bo_decisions_in_selected_jobs_present=True,
        original_trace_differences=int((~comparison.original_trace_matches_recorded).sum()),
        linux_refit_differences=int((~comparison.linux_refit_matches_recorded).sum()),
        logged_original_state_differences=int((~comparison.logged_original_state_matches_recorded).sum()),
        logged_linux_state_differences=int((~comparison.logged_linux_state_matches_linux_trace).sum()),
        materially_different_fitted_states=int((comparison.max_log_parameter_difference>1e-8).sum()),
        changed_decisions_with_different_parameters=int(((~comparison.linux_refit_matches_recorded)&(comparison.max_log_parameter_difference>1e-8)).sum()),
        max_log_parameter_difference=float(comparison.max_log_parameter_difference.max()),max_prediction_errors=errors,
        public_input_roundoff_before_loading_pinned_values=max_input_roundoff,
        source_runtime=om['runtime'],previous_linux_runtime=lm['runtime'],
        verification_runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,platform=platform.platform()),
        gp_optimizations_during_state_replay=0,new_physical_rollouts=0,new_policy_variants=0,
        closure_scope='Original-run validity and logged-state replay; fresh cross-platform GP refitting remains non-identical.',
        source_sha256=runner.digest(Path(__file__)),
        input_hashes={p.relative_to(runner.REPO).as_posix():runner.digest(p) for p in sources})
    runner.save(OUT/'verification.json',verification)
    print(grouped.to_string(index=False))
    print(json.dumps({k:v for k,v in verification.items() if k not in ('input_hashes',)},indent=2),flush=True)


if __name__=='__main__':main()
