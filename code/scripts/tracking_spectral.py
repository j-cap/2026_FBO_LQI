"""Matched online spectral D/E experiment. Protocol: docs/tracking_spectral.md."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import time

for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='1'

import tracking_hierarchical as reference_runner
from src.tracking_budget import dense_references
from src.tracking_kernel_features import basis_bank
from src.tracking_spectral import (LEGACY,basis_descriptor,load_bases,replay_methods,run_fleet,
    save_bases,shared,local,spectral_methods,validate_state)
from src.tracking_peer_cost_report import evaluate,eligibility_times
from src.tracking_spectral_report import write_report

ROOT,REPO=reference_runner.ROOT,reference_runner.REPO
read,save,digest=reference_runner.read,reference_runner.save,reference_runner.digest
jobs,path_for,clients_for=reference_runner.jobs,reference_runner.path_for,reference_runner.clients_for
observe=reference_runner.observe
np,pd=reference_runner.np,reference_runner.pd
CONFIG=ROOT/'config/runs/tracking_spectral.json'
WORKER=None


def setup(path=CONFIG,smoke=False):
    requested=read(path)
    if (requested['spectral_ranks']!=[256,384] or requested['primary_rank']!=256 or
        requested['basis_family']!='spectral'):
        raise ValueError('This matched protocol fixes 256 as primary and 384 as the accuracy control.')
    tag=requested['tag']
    if Path(tag).name!=tag or tag in ('.','..',requested['reference_tag']):
        raise ValueError('Use a new single-directory result tag.')
    reference=reference_runner.setup(ROOT/requested['reference_config'])
    if requested['reference_tag']!=reference['cfg']['tag']:
        raise ValueError('Completed D/E reference tag differs.')
    if (read(reference['out']/'run_manifest.json')['signature']!=reference['signature'] or
        read(reference['out']/'completion.json')!=dict(signature=reference['signature'],jobs=len(jobs(reference)))):
        raise RuntimeError('The original D/E reference changed or is incomplete.')
    cfg=deepcopy(reference['cfg'])
    cfg.update(tag=tag,reference_config=requested['reference_config'],reference_tag=requested['reference_tag'],
        spectral_ranks=requested['spectral_ranks'],primary_rank=requested['primary_rank'],basis_family='spectral',
        methods=spectral_methods(requested['spectral_ranks']),
        comparison_methods=LEGACY+spectral_methods(requested['spectral_ranks']))
    vehicles=reference['vehicles']
    if smoke:
        for key in ('seeds','queries_per_client','fleet_budgets'):cfg[key]=cfg['smoke'][key]
        vehicles=[v for v in vehicles if v['id'] in cfg['smoke']['client_plant_ids']]
        cfg['methods']=cfg['comparison_methods']
    out=ROOT/'results'/(tag+('_smoke' if smoke else ''))
    basis_path=out/'public_bases.npz'
    manifest_path=out/'run_manifest.json'
    if basis_path.exists():
        if not manifest_path.exists():raise RuntimeError('Public basis artifact has no signed manifest.')
        phis=load_bases(basis_path,cfg['spectral_ranks'],reference['domain'],read(manifest_path)['public_bases'])
    else:
        _,bank=basis_bank(reference['domain'],cfg['gp'],cfg['model'],
            [dict(family='spectral',rank=r) for r in cfg['spectral_ranks']])
        phis={r:bank[f'spectral_{r}']['F']/np.r_[np.sqrt(cfg['gp']['constant_variance']),
            np.full(r,cfg['gp']['amplitude'])] for r in cfg['spectral_ranks']}
    descriptors={str(r):basis_descriptor(phi) for r,phi in phis.items()}
    source_paths=[Path(__file__),ROOT/'src/tracking_spectral.py',ROOT/'src/tracking_spectral_report.py',
        ROOT/'src/tracking_kernel_features.py']
    source_hashes=dict(reference['specification']['source_hashes'])
    source_hashes.update({p.relative_to(REPO).as_posix():digest(p) for p in source_paths})
    reference_paths=[reference['out']/name for name in ('run_manifest.json','completion.json')]
    reference_paths += [path_for(reference,job) for job in jobs(reference)]
    assessment=REPO/'docs/analysis/kernel_representation_assessment/verification.json'
    if read(assessment)['selected_representation']!='spectral_256':
        raise RuntimeError('The numerical representation selection record differs.')
    reference_paths.append(assessment)
    spec=dict(configuration=cfg,mode='smoke' if smoke else 'full',source_hashes=source_hashes,
        evaluation_hashes=reference['specification']['evaluation_hashes'],
        reference_hashes={p.relative_to(REPO).as_posix():digest(p) for p in reference_paths},
        physical_vehicles=vehicles,public_bases=descriptors,historical_training_records=0,
        measurement_pairing=reference['specification']['measurement_pairing'],
        peer_rule=reference['specification']['peer_rule'],
        primary_contrast=shared(256)+' minus peer_gp_bo at final budget',
        secondary_contrast=shared(256)+' minus peer_local_cost_bo at final budget',
        accuracy_control_contrasts=[shared(384)+' minus '+m for m in LEGACY[:2]],
        sharing_controls=[shared(r)+' minus '+local(r) for r in cfg['spectral_ranks']],
        inference='Original D/E Gaussian messages and rho mixture; public spectral bases replace only the feature map.',
        evaluation_only_baselines=[] if smoke else LEGACY,
        worker_reference_access=False,worker_evaluation_access=False,
        basis_rule='Leading eigenvectors of public signal covariance plus exact intercept; no rollout labels.')
    signature=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    return dict(cfg=cfg,physics=reference['physics'],vehicles=vehicles,all_vehicles=reference['all_vehicles'],
        refs=reference['refs'],pool=reference['pool'],domain=reference['domain'],archive=reference['archive'],
        reference_state=reference,reference_out=reference['out'],specification=spec,
        signature=signature,out=out,phis=phis,public_bases=descriptors)


def initialize(state):
    reference_runner.initialize(state)
    path=state['out']/'public_bases.npz'
    if not path.exists():save_bases(path,state['phis'],state['domain'])
    load_bases(path,state['cfg']['spectral_ranks'],state['domain'],state['public_bases'])


def validate_job(state,job,result,complete=False,replay=False):
    if result.get('signature')!=state['signature'] or any(result.get(k)!=v for k,v in job.items()):
        raise RuntimeError('Spectral checkpoint identity differs.')
    clients=clients_for(state,job['condition'])
    if result['clients']!=clients or result.get('public_bases')!=state['public_bases']:
        raise RuntimeError('Spectral fleet or public basis differs.')
    data=result['state']
    validate_state(data,state['domain'],state['cfg'],clients,job['seed'],state['phis'],complete)
    for record in data['methods'].values():
        for event in record['events']:
            key=f'{event["client"]}__{event["local_evaluation"]}__{event["context"]}__{event["candidate"]}'
            entry=data['oracle_cache'].get(key)
            if entry is None or any(entry[k]!=event[k] for k in ('noise_seed','client','local_evaluation','context')):
                raise RuntimeError('Missing or mismatched spectral simulator record.')
            raw=entry['result']
            if raw['candidate']!=event['candidate'] or any(event[k]!=raw[k] for k in
                ('complete','feasible','integration_resolved','substeps')):
                raise RuntimeError('Spectral observation differs from simulator record.')
            if event['eligible'] and not np.isclose(event['J'],raw['J'],rtol=1e-12,atol=1e-12):
                raise RuntimeError('Spectral objective differs from simulator record.')
    if replay:replay_methods(data,state['domain'],state['cfg'],clients,job['seed'],state['phis'])


def init_worker(payload):
    global WORKER
    WORKER=payload


def worker(job):
    state=WORKER;clients=clients_for(state,job['condition']);path=path_for(state,job)
    previous=read(path) if path.exists() else None
    if previous:validate_job(state,job,previous)
    elapsed=previous.get('elapsed_seconds',0.) if previous else 0.;started=time.perf_counter()
    def checkpoint(data):
        save(path,dict(**job,signature=state['signature'],clients=clients,public_bases=state['public_bases'],
            state=data,elapsed_seconds=elapsed+time.perf_counter()-started))
        if data['round']%3==0:
            print(f'{job["condition"]}/{job["seed"]}: round {data["round"]}/{state["cfg"]["queries_per_client"]}, '
                f'{len(data["oracle_cache"])} unique simulations cached',flush=True)
    run_fleet(state['domain'],state['cfg'],clients,job['seed'],
        lambda plans,cache:observe(plans,cache,state,clients),state['phis'],checkpoint,
        previous['state'] if previous else None)
    validate_job(state,job,read(path),complete=True)


def comparison_result(state,job,result):
    if state['specification']['mode']=='smoke':return result
    old=read(path_for(dict(out=state['reference_out']),job))
    reference_runner.validate_job(state['reference_state'],job,old,complete=True)
    previous=reference_runner.comparison_result(state['reference_state'],job,old)
    if result['clients']!=previous['clients']:raise RuntimeError('Spectral/comparator fleets differ.')
    baselines={m:previous['state']['methods'][m] for m in LEGACY}
    n=len(result['clients']);initial=baselines[LEGACY[0]]['events'][:n]
    for record in result['state']['methods'].values():
        for a,b in zip(record['events'][:n],initial):
            for key in ('action','context','candidate','noise_seed','eligible','J'):
                equal=np.isclose(a[key],b[key],rtol=1e-10,atol=1e-12) if key=='J' and a[key] is not None else a[key]==b[key]
                if not equal:raise RuntimeError('Spectral/comparator factory test differs: '+key)
    return dict(result,state=dict(result['state'],methods={**baselines,**result['state']['methods']}))


def preflight(state):
    initialize(state)
    per_method=len(jobs(state))*len(state['vehicles'])*state['cfg']['queries_per_client']
    save(state['out']/'preflight.json',dict(jobs=len(jobs(state)),new_charged_requests=per_method*len(state['cfg']['methods']),
        reused_baseline_requests=0 if state['specification']['mode']=='smoke' else per_method*len(LEGACY),
        primary_rank=256,accuracy_control_rank=384,primary_baseline='A',secondary_baseline='B',
        worker_reference_access=False,worker_evaluation_access=False,historical_training_records=0,
        first_adaptive_round=2,public_basis_is_pinned=True))
    print(f'Preflight: {len(jobs(state))} jobs; {per_method*len(state["cfg"]["methods"]):,} newly charged requests. '
        f'Output: {state["out"]}',flush=True)


def analyze(state):
    initialize(state);results=[]
    for job in jobs(state):
        path=path_for(state,job)
        if not path.exists():raise RuntimeError('Missing spectral checkpoint: '+path.name)
        result=read(path);validate_job(state,job,result,complete=True,replay=True)
        results.append(comparison_result(state,job,result))
        print(f'Replayed {job["condition"]}/{job["seed"]}',flush=True)
    raw=pd.read_csv(state['archive']/'controlled_evaluations.csv',float_precision='round_trip')
    tables=dense_references(raw[raw.role=='calibration'],raw[raw.role=='validation'],state['domain'])
    if not all(v['eligible'] for v in tables[0].values()):raise RuntimeError('A dense empirical reference fails validation.')
    trials,contexts,queries,learning=evaluate(results,state['domain'],state['cfg'],tables)
    times,context_times=eligibility_times(results,state['domain'],state['cfg'])
    for name,frame in [('budget_trials',trials),('budget_contexts',contexts),('query_observations',queries),
        ('learning_diagnostics',learning),('eligibility_times',times),('context_eligibility_times',context_times)]:
        frame.to_csv(state['out']/(name+'.csv'),index=False)
    write_report(state['out'],trials,contexts,queries,learning,times,state['cfg'],state['specification']['mode'],len(state['domain'].actions))
    save(state['out']/'reference_control.json',dict(reused_for_evaluation_only=state['specification']['mode']=='full',
        methods=LEGACY,exact_fleet_and_factory_pairing=True,worker_reference_access=False))
    save(state['out']/'analysis_provenance.json',dict(all_new_choices_replayed=True,public_bases=state['public_bases'],
        report_sha256=digest(ROOT/'src/tracking_spectral_report.py')))
    save(state['out']/'completion.json',dict(signature=state['signature'],jobs=len(results)))
    print('Report: '+str(state['out']/'report.html'),flush=True)


def run(state,workers):
    preflight(state);pending=[]
    for job in jobs(state):
        path=path_for(state,job)
        if path.exists():
            result=read(path);validate_job(state,job,result)
            if result['state']['round']==state['cfg']['queries_per_client']:continue
        pending.append(job)
    print(f'{len(jobs(state))-len(pending)}/{len(jobs(state))} complete; {len(pending)} pending.',flush=True)
    payload={k:state[k] for k in ('cfg','physics','vehicles','all_vehicles','refs','pool','domain','out','signature','phis','public_bases')}
    if workers==1:
        init_worker(payload)
        for job in pending:worker(job)
    elif pending:
        with ProcessPoolExecutor(max_workers=min(workers,len(pending)),mp_context=multiprocessing.get_context('spawn'),
                initializer=init_worker,initargs=(payload,)) as pool:
            for future in as_completed([pool.submit(worker,job) for job in pending]):future.result()
    analyze(state)


def collect(state):
    if state['specification']['mode']!='full':raise RuntimeError('Smoke results cannot be collected as experimental evidence.')
    analyze(state)
    relative=state['out'].relative_to(REPO).as_posix()
    subprocess.run(['git','add','-f','--',relative],cwd=REPO,check=True)
    print('Staged only '+relative)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['preflight','smoke','run','status','analyze','collect'])
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--workers',type=int,default=min(6,os.cpu_count() or 1))
    parser.add_argument('--smoke-results',action='store_true')
    args=parser.parse_args()
    if args.workers<1:parser.error('--workers must be positive')
    state=setup(args.config,args.command=='smoke' or args.smoke_results)
    if args.command in ('run','smoke'):run(state,args.workers)
    elif args.command=='preflight':preflight(state)
    elif args.command=='analyze':analyze(state)
    elif args.command=='collect':collect(state)
    else:
        for job in jobs(state):
            path=path_for(state,job)
            round_id=read(path)['state']['round'] if path.exists() else 0
            print(f'{job["condition"]}/{job["seed"]}: {round_id}/{state["cfg"]["queries_per_client"]} rounds')


if __name__=='__main__':main()
