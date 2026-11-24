"""Assess D kernel representations against its exact reference on saved histories.

No physical simulation, new controller policy or validation-cost lookup. Bases
depend only on the public inputs and frozen kernel. Only recorded source labels
advance histories; alternative rankings never supply future observations.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='1'

import tracking_hierarchical as runner
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src.tracking_budget import feasibility_probability
from src.tracking_hierarchical import SHARED,cost_posterior,model_payload_bytes
from src.tracking_kernel_features import basis_bank,query_posterior
from src.tracking_online_lcb import acquisition_values
from src.tracking_peer_cost import OBSERVATION_FIELDS

CONFIG=runner.ROOT/'config/runs/tracking_kernel_assessment.json'
ARMS={'A':'peer_gp_bo','D':SHARED}
WORKER=None


def accuracy_metrics(mean,std,exact_mean,exact_std,probability,acquisition):
    approximate=acquisition_values(mean,std**2,probability,acquisition)
    exact=acquisition_values(exact_mean,exact_std**2,probability,acquisition)
    tolerance=1e-12*max(1.,float(abs(approximate.min())))
    ties=np.flatnonzero(approximate <= approximate.min()+tolerance)
    # Conservative over numerical ties. This is an exact-score difference,
    # not physical controller regret or an alternative observed rollout.
    regret=max(0.,float(exact[ties].max()-exact.min()))
    best=int(ties[0])
    exact_rank=1+int(np.sum(exact < exact[best]-1e-12*max(1.,abs(float(exact.min())))))
    correlation=(float(spearmanr(approximate,exact).statistic)
        if np.ptp(approximate)>0 and np.ptp(exact)>0 else None)
    return dict(mean_MSE=float(np.mean((mean-exact_mean)**2)),std_MSE=float(np.mean((std-exact_std)**2)),
        max_abs_mean_error=float(np.max(np.abs(mean-exact_mean))),
        max_abs_std_error=float(np.max(np.abs(std-exact_std))),
        score_MSE=float(np.mean((approximate-exact)**2)),exact_score_regret=regret,
        approximate_best_exact_rank=exact_rank,
        top_one_agreement=bool(np.any(exact[ties] <= exact.min()+1e-10)),
        score_rank_correlation=correlation),approximate,exact


def init_worker(payload):
    global WORKER
    WORKER=payload


def assess_job(job):
    state=WORKER
    name=f'{job["condition"]}__{job["seed"]}__{job["source"]}'
    cache=state['cache']/(name+'.json')
    if cache.exists():
        saved=runner.read(cache)
        if saved['signature'] != state['signature']:
            raise RuntimeError('Assessment cache identity differs.')
        return saved
    cfg,domain=state['cfg'],state['domain']
    histories=[[] for _ in job['clients']]
    rows,focus=[],[]
    payloads={name:0 for name in state['bank']}
    factual_checked,max_factual_gap=0,0.
    for round_id in range(1,cfg['queries_per_client']+1):
        events=job['events'][(round_id-1)*len(histories):round_id*len(histories)]
        if round_id > 1:
            counts=[sum(o['eligible'] for o in h) for h in histories]
            for representation,basis in state['bank'].items():
                payloads[representation]+=model_payload_bytes(counts,basis['feature_width'],len(cfg['hierarchical']['sharing_fractions']))
        cases=[e for e in events if e['mode']=='feasibility_penalized_LCB']
        query_clients,query_actions,groups=[],[],[]
        for event in cases:
            i=event['client']
            allowed=np.asarray([c==event['context'] for c,_ in domain.actions])
            allowed[[o['action'] for o in histories[i]]]=False
            indices=np.flatnonzero(allowed)
            first=len(query_actions)
            query_clients.extend([i]*len(indices));query_actions.extend(indices.tolist())
            probability=feasibility_probability(histories[i],domain,np.full(len(domain.actions),.5),cfg['gp'])[indices]
            groups.append((event,indices,probability,slice(first,len(query_actions))))
        if groups:
            exact=query_posterior(histories,state['exact'],cfg['hierarchical'],cfg['gp'],query_clients,query_actions)
            exact_std=np.sqrt(exact['variance'])
            for representation,basis in state['bank'].items():
                predicted=query_posterior(histories,basis['kernel'],cfg['hierarchical'],cfg['gp'],query_clients,query_actions)
                std=np.sqrt(predicted['variance'])
                for event,indices,probability,sl in groups:
                    metrics,scores,exact_scores=accuracy_metrics(predicted['mean'][sl],std[sl],
                        exact['mean'][sl],exact_std[sl],probability,cfg['acquisition'])
                    row=dict(condition=job['condition'],seed=job['seed'],source=job['source'],
                        round=round_id,client=job['clients'][event['client']],context=event['context'],
                        representation=representation,available_candidates=len(indices),
                        eligible_cost_records=sum(len([o for o in h if o['eligible']]) for h in histories),
                        rho_difference=abs(predicted['expected_sharing_fraction']-exact['expected_sharing_fraction']),**metrics)
                    rows.append(row)
                    if representation=='rff_64' and job['source']=='D':
                        at=int(np.flatnonzero(indices==event['action'])[0])
                        gap=float(scores[at]-scores.min())
                        if gap > 1e-8:
                            raise RuntimeError('Observation-space baseline fails factual D choice check.')
                        max_factual_gap=max(max_factual_gap,gap);factual_checked+=1
                    if job['condition']=='homogeneous' and event['context']=='s30_f0.1':
                        reference=domain.action(event['context'],'g012')
                        present=np.flatnonzero(indices==reference)
                        if len(present):
                            at=int(present[0])
                            focus.append(dict(condition=job['condition'],seed=job['seed'],source=job['source'],
                                round=round_id,client=job['clients'][event['client']],representation=representation,
                                reference_score_rank=1+int(np.sum(scores < scores[at]-1e-12)),
                                exact_reference_score_rank=1+int(np.sum(exact_scores < exact_scores[at]-1e-12))))
        for event in events:
            histories[event['client']].append({k:event[k] for k in OBSERVATION_FIELDS})
    if job['source']=='D' and payloads['rff_64'] != job['baseline_model_payload_bytes']:
        raise RuntimeError('Baseline communication estimate differs from frozen D snapshot.')
    result=dict(signature=state['signature'],condition=job['condition'],seed=job['seed'],source=job['source'],
        rows=rows,focus=focus,payloads=payloads,factual_D_choices_checked=factual_checked,
        max_factual_D_score_gap=max_factual_gap)
    runner.save(cache,result)
    print('Assessed '+name,flush=True)
    return result


def summarize(rows):
    def group(keys):
        data=rows.groupby(keys,sort=False).agg(
            decisions=('mean_MSE','size'),mean_MSE=('mean_MSE','mean'),std_MSE=('std_MSE','mean'),
            score_MSE=('score_MSE','mean'),mean_exact_score_regret=('exact_score_regret','mean'),
            p95_exact_score_regret=('exact_score_regret',lambda x:float(x.quantile(.95))),
            max_exact_score_regret=('exact_score_regret','max'),
            top_one_agreement=('top_one_agreement','mean'),
            mean_exact_rank_of_approximate_best=('approximate_best_exact_rank','mean'),
            median_rank_correlation=('score_rank_correlation','median'),
            max_rho_difference=('rho_difference','max')).reset_index()
        for name in ('mean','std','score'):data[name+'_RMSE']=np.sqrt(data.pop(name+'_MSE'))
        return data
    return (group(['representation']),group(['representation','condition','source']),
            group(['representation','condition','source','context']),
            group(['representation','condition','source','seed']))


def selection_table(prior,overall,groups,contexts,targets):
    out=prior.merge(overall,on='representation',validate='one_to_one')
    worst=contexts.groupby('representation').agg(worst_context_mean_RMSE=('mean_RMSE','max'),
        worst_context_std_RMSE=('std_RMSE','max')).reset_index()
    group_worst=groups.groupby('representation').agg(worst_group_mean_exact_score_regret=('mean_exact_score_regret','max'),
        worst_group_p95_exact_score_regret=('p95_exact_score_regret','max')).reset_index()
    out=out.merge(worst,on='representation').merge(group_worst,on='representation')
    out['prior_accuracy_pass']=(out.relative_kernel_Frobenius_error<=targets['relative_kernel_Frobenius_error']) & (
        out.max_relative_diagonal_error<=targets['max_relative_diagonal_error'])
    out['posterior_accuracy_pass']=(out.worst_context_mean_RMSE<=targets['worst_context_mean_RMSE']) & (
        out.worst_context_std_RMSE<=targets['worst_context_std_RMSE'])
    out['score_accuracy_pass']=(out.worst_group_mean_exact_score_regret<=targets['mean_exact_score_regret']) & (
        out.worst_group_p95_exact_score_regret<=targets['p95_exact_score_regret'])
    out['all_accuracy_targets_pass']=out.prior_accuracy_pass & out.posterior_accuracy_pass & out.score_accuracy_pass
    return out.sort_values(['feature_width','worst_context_mean_RMSE','representation'])


def benchmark(state,bank,jobs):
    rows=[]
    for job in jobs:
        if job['source']!='D' or job['seed']!=1101:continue
        histories=[[{k:e[k] for k in OBSERVATION_FIELDS} for e in job['events'] if e['client']==i and e['round']<=23]
                   for i in range(len(job['clients']))]
        clients=np.repeat(np.arange(len(histories)),len(state['domain'].actions))
        actions=np.tile(np.arange(len(state['domain'].actions)),len(histories))
        for name in ['rff_64','spectral_64','spectral_256','spectral_384','spectral_426']:
            basis=bank[name];F=basis['F'];width=F.shape[1]
            scale=np.r_[np.sqrt(state['cfg']['gp']['constant_variance']),np.full(width-1,state['cfg']['gp']['amplitude'])]
            started=time.perf_counter()
            feature=cost_posterior(histories,F/scale,state['cfg']['gp'],state['cfg']['hierarchical'])
            seconds=time.perf_counter()-started
            dual=query_posterior(histories,basis['kernel'],state['cfg']['hierarchical'],state['cfg']['gp'],clients,actions)
            np.testing.assert_allclose(np.asarray(feature['mean']).ravel(),dual['mean'],rtol=1e-7,atol=1e-8)
            np.testing.assert_allclose(np.asarray(feature['variance']).ravel(),dual['variance'],rtol=1e-7,atol=1e-8)
            rows.append(dict(condition=job['condition'],representation=name,feature_width=width,
                eligible_records=sum(sum(o['eligible'] for o in h) for h in histories),
                feature_message_fit_and_all_predictions_seconds=seconds,
                model_payload_this_training_round=feature['model_payload_bytes'],dual_feature_predictions_match=True))
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=min(6,os.cpu_count() or 1))
    args=parser.parse_args()
    if args.workers < 1:parser.error('--workers must be positive')
    specification=runner.read(CONFIG)
    state=runner.setup()
    if runner.read(state['out']/'completion.json') != dict(signature=state['signature'],jobs=6):
        raise RuntimeError('Completed D campaign identity differs.')
    exact,bank=basis_bank(state['domain'],state['cfg']['gp'],state['cfg']['model'],specification['candidates'])
    jobs=[];input_paths=[CONFIG,state['out']/'run_manifest.json',state['out']/'completion.json']
    for item in runner.jobs(state):
        result=runner.read(runner.path_for(state,item))
        runner.validate_job(state,item,result,complete=True)
        combined=runner.comparison_result(state,item,result)
        input_paths.extend([runner.path_for(state,item),runner.path_for(state['reference_state'],item)])
        for source in specification['history_sources']:
            data=combined['state']['methods'][ARMS[source]]
            jobs.append(dict(**item,source=source,clients=[c['id'] for c in combined['clients']],
                events=data['events'],baseline_model_payload_bytes=data['snapshots']['216']['model_payload_bytes']))
    source_paths=[Path(__file__),runner.ROOT/'src/tracking_kernel_features.py']
    hashes={p.relative_to(runner.REPO).as_posix():runner.digest(p) for p in input_paths+source_paths}
    identity=dict(configuration=specification,campaign_signature=state['signature'],hashes=hashes)
    signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    cache=runner.ROOT/'.cache/kernel_assessment'/signature;cache.mkdir(parents=True,exist_ok=True)
    payload=dict(cfg=state['cfg'],domain=state['domain'],bank=bank,exact=exact,signature=signature,cache=cache)
    print(f'Assessing {len(bank)} representations on {len(jobs)} saved-history jobs; no new rollouts.',flush=True)
    if args.workers==1:
        init_worker(payload);results=[assess_job(job) for job in jobs]
    else:
        results=[]
        with ProcessPoolExecutor(max_workers=min(args.workers,len(jobs)),mp_context=multiprocessing.get_context('spawn'),
                initializer=init_worker,initargs=(payload,)) as pool:
            for future in as_completed([pool.submit(assess_job,job) for job in jobs]):results.append(future.result())
    results.sort(key=lambda r:(r['condition'],r['seed'],r['source']))
    rows=pd.DataFrame([row for result in results for row in result['rows']])
    focus=pd.DataFrame([row for result in results for row in result['focus']])
    overall,groups,contexts,seeds=summarize(rows)
    prior=pd.DataFrame([dict(representation=name,**{k:v for k,v in basis.items() if k not in ('F','kernel','pivots')})
                        for name,basis in bank.items()])
    selected=selection_table(prior,overall,groups,contexts,specification['accuracy_targets'])
    passing=selected[selected.all_accuracy_targets_pass]
    choice=None if passing.empty else str(passing.iloc[0].representation)
    payload_rows=[]
    for result in results:
        for name,value in result['payloads'].items():
            payload_rows.append(dict(condition=result['condition'],seed=result['seed'],source=result['source'],
                representation=name,model_payload_at_final_decision=value,
                public_basis_broadcast_bytes=0 if bank[name]['family']=='rff' else len(jobs[0]['clients'])*bank[name]['public_basis_bytes']))
    communication=pd.DataFrame(payload_rows)
    bench=benchmark(state,bank,jobs)
    out=runner.REPO/'docs/analysis'/specification['tag'];out.mkdir(parents=True,exist_ok=True)
    for name,frame in [('prior_kernel_errors',prior),('representation_comparison',selected),
        ('by_condition_and_history',groups),('by_context',contexts),('by_seed',seeds),
        ('focused_reference_rankings',focus),('communication_estimates',communication),('feature_inference_benchmark',bench)]:
        frame.to_csv(out/(name+'.csv'),index=False)
    focus_summary=focus.groupby(['representation','source'],sort=False).agg(
        decisions=('reference_score_rank','size'),median_reference_score_rank=('reference_score_rank','median'),
        approximate_reference_rank_one=('reference_score_rank',lambda x:int((x==1).sum())),
        exact_reference_rank_one=('exact_reference_score_rank',lambda x:int((x==1).sum()))).reset_index()
    focus_summary.to_csv(out/'focused_reference_summary.csv',index=False)
    runner.save(out/'verification.json',dict(signature=signature,**identity,selected_representation=choice,
        accuracy_targets_fixed_before_conditioning=True,public_grid_size=len(state['domain'].actions),
        saved_history_jobs=len(results),actual_BO_decision_cases=len(rows)//len(bank),representation_cases=len(rows),
        original_D_choices_checked=sum(r['factual_D_choices_checked'] for r in results),
        max_original_D_numerical_score_gap=max(r['max_factual_D_score_gap'] for r in results),
        new_physical_rollouts=0,new_full_policy_variants=0,validation_cost_labels_used=False,
        equivalent_observation_space_inference_used_for_assessment=True,
        feature_message_algebra_checked_at_round_23=True,
        basis_hashes={name:hashlib.sha256(b['F'].tobytes()).hexdigest() for name,b in bank.items()}))
    intermediate=runner.ROOT/'figures/kernel_assessment_decision_cases.csv'
    intermediate.parent.mkdir(parents=True,exist_ok=True);rows.to_csv(intermediate,index=False)
    print(selected[['representation','feature_width','relative_kernel_Frobenius_error',
        'worst_context_mean_RMSE','worst_context_std_RMSE','worst_group_mean_exact_score_regret',
        'worst_group_p95_exact_score_regret','all_accuracy_targets_pass']].to_string(index=False),flush=True)
    print('Selected by the predeclared accuracy targets: '+str(choice),flush=True)
    print('Assessment: '+str(out),flush=True)


if __name__=='__main__':main()
