"""Frozen D256 confirmation on fresh seeds/fleets. See docs/tracking_confirmation.md."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import time

for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='1'

import tracking_spectral as reference_runner
import numpy as np
import pandas as pd
import scipy

from src.tracking_confirmation import METHODS, aggregate_candidates, calibration_reference, design, evaluate_job, paired_tables, pinned_design
from src.tracking_online import seed_for
from src.tracking_peer_cost import OBSERVATION_FIELDS
from src.tracking_spectral import basis_descriptor, load_bases, run_fleet, save_bases, validate_state
from src.tracking_task import resolved_rollout
from src.tracking_confirmation_report import write_report

ROOT,REPO=reference_runner.ROOT,reference_runner.REPO
read,save,digest=reference_runner.read,reference_runner.save,reference_runner.digest
CONFIG=ROOT/'config/runs/tracking_confirmation.json'
WORKER=None


def setup(path=CONFIG,smoke=False):
    requested=read(path)
    tag=requested['tag']
    if Path(tag).name!=tag or tag in ('.','..','tracking_spectral_v1'):
        raise ValueError('Use a new single-directory confirmation tag.')
    if (len(requested['homogeneous_seeds'])!=10 or len(set(requested['homogeneous_seeds']))!=10 or
        len(requested['fleet_seeds'])!=10 or len(set(requested['fleet_seeds']))!=10 or
        requested['runs_per_fleet']!=3 or requested['calibration_repeats']!=2 or
        requested['validation_repeats']!=2 or requested['bootstrap_repeats']!=10000):
        raise ValueError('The confirmation fixes ten units, three runs/fleet and two evaluator repeats.')
    if any(not np.isfinite(x) or x<=0 for x in requested['scatter'].values()):
        raise ValueError('Scatter bounds must be finite and positive.')
    previous=reference_runner.setup(ROOT/requested['reference_config'])
    if read(previous['out']/'completion.json')!=dict(signature=previous['signature'],jobs=6):
        raise RuntimeError('The completed D256 development reference differs.')
    cfg=deepcopy(previous['cfg'])
    cfg.update(tag=tag,methods=METHODS.copy(),comparison_methods=METHODS.copy(),spectral_ranks=[256])
    if smoke:
        cfg.update(queries_per_client=requested['smoke']['queries_per_client'],fleet_budgets=requested['smoke']['fleet_budgets'])
    groups,jobs,draws=design(previous['vehicles'],requested,smoke)
    out=ROOT/'results'/(tag+('_smoke' if smoke else ''))
    if (out/'run_manifest.json').exists():
        manifest=read(out/'run_manifest.json')
        pinned=pinned_design(dict(groups=groups,jobs=jobs,physical_draws=draws),
            {k:manifest[k] for k in ('groups','jobs','physical_draws')})
        groups,jobs,draws=(pinned[k] for k in ('groups','jobs','physical_draws'))
    cfg.update(seeds=[j['seed'] for j in jobs],reference_config=requested['reference_config'],reference_tag=previous['cfg']['tag'])
    if any(j['seed'] in previous['cfg']['seeds'] for j in jobs):
        raise ValueError('Confirmation reuses a development programme/noise seed.')
    if any(not set(c['params']).issubset(previous['vehicles'][0]['params']) for g in groups.values() for c in g):
        raise RuntimeError('Unexpected plant parameter schema.')
    n=len(groups['homogeneous'])
    if cfg['fleet_budgets'][-1]!=n*cfg['queries_per_client']:
        raise ValueError('Confirmation cap and snapshots differ.')
    descriptors={'256':basis_descriptor(previous['phis'][256])}
    phis={256:previous['phis'][256]}
    if (out/'public_bases.npz').exists():
        phis=load_bases(out/'public_bases.npz',[256],previous['domain'],descriptors)
    sources=dict(previous['specification']['source_hashes'])
    for p in [Path(__file__),ROOT/'src/tracking_confirmation.py',ROOT/'src/tracking_confirmation_report.py']:
        sources[p.relative_to(REPO).as_posix()]=digest(p)
    specification=dict(configuration=cfg,confirmation_design=requested,mode='smoke' if smoke else 'full',
        source_hashes=sources,public_bases=descriptors,groups=groups,jobs=jobs,physical_draws=draws,
        development_signature=previous['signature'],historical_training_records=0,
        worker_reference_access=False,worker_evaluation_access=False,
        peer_rule=previous['specification']['peer_rule'],measurement_pairing=previous['specification']['measurement_pairing'],
        evaluator_rule='Fresh dense calibration; held-out validation of calibration winners and all budget recommendations only.',
        missing_reference_rule='Retain all cells; unavailable reference means unsuccessful with missing gap; flag incomplete reference coverage.',
        replication_unit='Heterogeneous fleet averaged over three runs; homogeneous fresh programme/noise seed.',
        confirmation_rule='Final 5%-context success bootstrap lower > 0, complete references, no loss of eligible-policy count, lower jointly eligible mean J; A first, B second.')
    signature=hashlib.sha256(json.dumps(specification,sort_keys=True).encode()).hexdigest()
    return dict(cfg=cfg,requested=requested,physics=previous['physics'],refs=previous['refs'],pool=previous['pool'],
        domain=previous['domain'],phis=phis,public_bases=descriptors,groups=groups,jobs=jobs,
        specification=specification,signature=signature,out=out)


def initialize(state):
    out=state['out'];path=out/'run_manifest.json'
    if path.exists():
        if read(path)['signature']!=state['signature']:
            raise RuntimeError('Confirmation inputs changed; use a new result tag.')
    else:
        save(path,dict(signature=state['signature'],**state['specification'],
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
            runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__)))
    if not (out/'public_bases.npz').exists():
        save_bases(out/'public_bases.npz',state['phis'],state['domain'])
    load_bases(out/'public_bases.npz',[256],state['domain'],state['public_bases'])


def run_path(state,job):
    return state['out']/'runs'/f'{job["condition"]}__{job["fleet_id"]}__{job["seed"]}.json'


def validate_job(state,job,result,complete=False,replay=False):
    clients=state['groups'][job['fleet_id']]
    if (result.get('signature')!=state['signature'] or result.get('clients')!=clients or
        result.get('public_bases')!=state['public_bases'] or any(result.get(k)!=v for k,v in job.items())):
        raise RuntimeError('Confirmation checkpoint identity differs.')
    data=result['state']
    validate_state(data,state['domain'],state['cfg'],clients,job['seed'],state['phis'],complete)
    for record in data['methods'].values():
        for event in record['events']:
            key=f'{event["client"]}__{event["local_evaluation"]}__{event["context"]}__{event["candidate"]}'
            entry=data['oracle_cache'].get(key)
            if entry is None or any(entry[k]!=event[k] for k in ('noise_seed','client','local_evaluation','context')):
                raise RuntimeError('Confirmation observation lacks its requested simulator record.')
            raw=entry['result']
            if raw['candidate']!=event['candidate'] or any(raw[k]!=event[k] for k in
                ('complete','feasible','integration_resolved','substeps')):
                raise RuntimeError('Confirmation observation differs from simulator record.')
            if event['eligible'] and not np.isclose(event['J'],raw['J'],rtol=1e-12,atol=1e-12):
                raise RuntimeError('Confirmation cost differs from simulator record.')
    if replay:
        def recorded(plans,cache):
            rows=[]
            for plan in plans:
                event=data['methods'][plan['method']]['events'][(plan['round']-1)*len(clients)+plan['client']]
                if any(plan[k]!=event[k] for k in plan):
                    raise RuntimeError('Confirmation paid decision replay differs.')
                rows.append({k:event[k] for k in OBSERVATION_FIELDS if k!='action'})
            return rows
        actual=run_fleet(state['domain'],state['cfg'],clients,job['seed'],recorded,state['phis'],stop_round=data['round'])
        for method in state['cfg']['methods']:
            if actual['methods'][method]['events']!=data['methods'][method]['events']:
                raise RuntimeError('Confirmation event replay differs.')


def payload(state):
    # No development histories, completed checkpoints or evaluator tables.
    return {k:state[k] for k in ('cfg','physics','refs','pool','domain','phis','public_bases','groups','signature','out')}


def init_worker(state):
    global WORKER
    WORKER=state


def live_worker(job):
    state=WORKER;clients=state['groups'][job['fleet_id']];path=run_path(state,job)
    previous=read(path) if path.exists() else None
    if previous:
        validate_job(state,job,previous)
    elapsed=previous.get('elapsed_seconds',0.) if previous else 0.;started=time.perf_counter()
    def checkpoint(data):
        save(path,dict(**job,signature=state['signature'],public_bases=state['public_bases'],clients=clients,
            state=data,elapsed_seconds=elapsed+time.perf_counter()-started))
        if data['round']%6==0:
            print(f'{job["fleet_id"]}/{job["seed"]}: {data["round"]}/{state["cfg"]["queries_per_client"]} rounds',flush=True)
    run_fleet(state['domain'],state['cfg'],clients,job['seed'],
        lambda plans,cache:reference_runner.observe(plans,cache,state,clients),state['phis'],checkpoint,
        previous['state'] if previous else None)
    validate_job(state,job,read(path),complete=True)


def replay_worker(job):
    validate_job(WORKER,job,read(run_path(WORKER,job)),complete=True,replay=True)


def parallel(function,tasks,state,workers,label):
    if not tasks:
        return
    if workers==1:
        init_worker(state)
        for i,task in enumerate(tasks,1):
            function(task)
            if i==len(tasks) or i%25==0:print(f'{label}: {i}/{len(tasks)}',flush=True)
    else:
        with ProcessPoolExecutor(max_workers=min(workers,len(tasks)),mp_context=multiprocessing.get_context('spawn'),
                initializer=init_worker,initargs=(state,)) as executor:
            futures=[executor.submit(function,task) for task in tasks]
            for i,future in enumerate(as_completed(futures),1):
                future.result()
                if i==len(tasks) or i%25==0:print(f'{label}: {i}/{len(tasks)}',flush=True)


def plants(state,group):
    return {c['plant_id']:c for c in state['groups'][group]}


def evaluation_tasks(state,role,selection=None):
    tasks=[]
    repeats=1 if state['specification']['mode']=='smoke' else state['requested'][role+'_repeats']
    for group in state['groups']:
        for plant_id in plants(state,group):
            for context in state['domain'].contexts:
                gains=[c['id'] for c in state['pool']] if role=='calibration' else sorted(selection[group,plant_id,context])
                if not gains:
                    continue
                for rep in range(repeats):
                    tasks.append(dict(group=group,plant_id=plant_id,context=context,role=role,repeat=rep,
                        noise_seed=seed_for('confirmation_evaluator',group,role,rep),candidates=gains))
    return tasks


def evaluation_path(state,task):
    return state['out']/'evaluation'/task['role']/f'{task["plant_id"]}__{task["context"]}__{task["repeat"]}.json'


def validate_batch(state,task,batch):
    if batch.get('signature')!=state['signature'] or any(batch.get(k)!=v for k,v in task.items()):
        raise RuntimeError('Confirmation evaluator cache identity differs.')
    if [r['candidate'] for r in batch['rows']]!=task['candidates']:
        raise RuntimeError('Confirmation evaluator candidate set differs.')
    for row in batch['rows']:
        if row['complete'] and (row['J'] is None or not np.isfinite(row['J']) or row['J']<=0):
            raise RuntimeError('Invalid evaluator objective.')


def evaluation_worker(task):
    state=WORKER;path=evaluation_path(state,task)
    if path.exists():
        validate_batch(state,task,read(path));return
    plant=plants(state,task['group'])[task['plant_id']]
    ref=next(r for r in state['refs'] if r['id']==task['context'])
    lookup={c['id']:c for c in state['pool']}
    rows,_=resolved_rollout(plant['params'],ref,[lookup[c] for c in task['candidates']],state['physics'],task['noise_seed'])
    batch=dict(**task,signature=state['signature'],rows=rows)
    validate_batch(state,task,batch);save(path,batch)


def completed_live(state,replay=False,workers=1):
    for job in state['jobs']:
        path=run_path(state,job)
        if not path.exists():raise RuntimeError('Missing live checkpoint: '+path.name)
        validate_job(state,job,read(path),complete=True)
    if replay:parallel(replay_worker,state['jobs'],payload(state),workers,'Paid-decision replay')
    return [read(run_path(state,job)) for job in state['jobs']]


def load_calibration(state):
    buckets={}
    for task in evaluation_tasks(state,'calibration'):
        path=evaluation_path(state,task)
        if not path.exists():raise RuntimeError('Missing dense calibration batch: '+path.name)
        batch=read(path);validate_batch(state,task,batch)
        buckets.setdefault((task['group'],task['plant_id'],task['context']),[]).append(batch)
    aggregated={key:aggregate_candidates(batches) for key,batches in buckets.items()}
    return {key:calibration_reference(value) for key,value in aggregated.items()}


def validation_selection(state,results,chosen):
    selection={key:({candidate} if candidate is not None else set()) for key,candidate in chosen.items()}
    for result in results:
        for data in result['state']['methods'].values():
            for snapshot in data['snapshots'].values():
                for client,recommendations in zip(result['clients'],snapshot['recommendations']):
                    for context,candidate in zip(state['domain'].contexts,recommendations):
                        if candidate is not None:
                            selection[result['fleet_id'],client['plant_id'],context].add(candidate)
    return selection


def load_validation(state,tasks,chosen):
    buckets={}
    for task in tasks:
        path=evaluation_path(state,task)
        if not path.exists():raise RuntimeError('Missing holdout validation batch: '+path.name)
        batch=read(path);validate_batch(state,task,batch)
        buckets.setdefault((task['group'],task['plant_id'],task['context']),[]).append(batch)
    refs={};validation={}
    for key,candidate in chosen.items():
        group,plant,context=key
        rows=aggregate_candidates(buckets.get(key,[]))
        refs[key]=dict(candidate=candidate,**rows[candidate]) if candidate is not None else dict(candidate=None,eligible=False,J=None)
        validation.update({(group,plant,context,c):value for c,value in rows.items()})
    return refs,validation


def preflight(state):
    initialize(state)
    online=sum(len(state['groups'][j['fleet_id']])*state['cfg']['queries_per_client']*len(METHODS) for j in state['jobs'])
    cal=evaluation_tasks(state,'calibration')
    record=dict(signature=state['signature'],live_jobs=len(state['jobs']),new_charged_online_requests=online,
        reused_online_requests=0,dense_calibration_batches=len(cal),dense_calibration_rollouts=sum(len(t['candidates']) for t in cal),
        validation_rollout_upper_bound=sum(len(t['candidates']) for t in cal),
        heterogeneous_fleets=len(state['groups'])-1,homogeneous_runs=sum(j['condition']=='homogeneous' for j in state['jobs']),
        physical_reference_contexts=sum(len(plants(state,g)) for g in state['groups'])*len(state['domain'].contexts),
        methods=METHODS,historical_training_records=0,worker_reference_access=False,worker_evaluation_access=False,
        evaluator_selection='Union of calibration-only winners and all frozen budget recommendations; same held-out noise for all arms.')
    save(state['out']/'preflight.json',record)
    print(f'Preflight: {len(state["jobs"])} live jobs; {online:,} charged online requests; '
        f'{record["dense_calibration_rollouts"]:,} offline reference rollouts.',flush=True)


def analyze(state,workers):
    initialize(state)
    results=completed_live(state,replay=True,workers=workers)
    chosen=load_calibration(state)
    selection=validation_selection(state,results,chosen)
    tasks=evaluation_tasks(state,'validation',selection)
    refs,validation=load_validation(state,tasks,chosen)
    frames=[[],[],[],[]]
    for job,result in zip(state['jobs'],results):
        group=job['fleet_id']
        group_refs={(p,c):v for (g,p,c),v in refs.items() if g==group}
        group_validation={(p,c,k):v for (g,p,c,k),v in validation.items() if g==group}
        for target,frame in zip(frames,evaluate_job(job,result,state['domain'],state['cfg'],group_refs,group_validation)):
            target.append(frame)
    trials,cells,queries,learning=[pd.concat(items,ignore_index=True) for items in frames]
    paired,units=paired_tables(trials,state['cfg'],state['requested'],state['specification']['mode'])
    for name,frame in [('budget_trials',trials),('budget_contexts',cells),('query_observations',queries),
        ('learning_diagnostics',learning),('paired_comparisons',paired),('replication_unit_comparisons',units)]:
        frame.to_csv(state['out']/(name+'.csv'),index=False)
    ref_rows=[dict(group=g,plant_id=p,context=c,**value) for (g,p,c),value in refs.items()]
    pd.DataFrame(ref_rows).to_csv(state['out']/'empirical_references.csv',index=False)
    write_report(state['out'],trials,cells,queries,paired,units,state['cfg'],state['specification']['mode'])
    save(state['out']/'completion.json',dict(signature=state['signature'],jobs=len(results)))
    cal_tasks=evaluation_tasks(state,'calibration')
    save(state['out']/'verification.json',dict(signature=state['signature'],all_four_method_decisions_replayed=True,
        online_requests=len(queries),all_local_caps_respected=bool((queries.groupby(['fleet_id','seed','method','client']).size()==state['cfg']['queries_per_client']).all()),
        unavailable_reference_contexts=sum(not r['eligible'] for r in refs.values()),
        unresolved_online_requests=int((~queries.integration_resolved).sum()),
        calibration_rollouts=sum(len(t['candidates']) for t in cal_tasks),validation_rollouts=sum(len(t['candidates']) for t in tasks),
        old_rollouts_used_for_learning_or_evaluation=0,public_bases=state['public_bases']))
    print('Report: '+str(state['out']/'report.html'),flush=True)


def run(state,workers):
    preflight(state)
    pending=[]
    for job in state['jobs']:
        path=run_path(state,job)
        if path.exists():
            result=read(path);validate_job(state,job,result)
            if result['state']['round']==state['cfg']['queries_per_client']:continue
        pending.append(job)
    print(f'Online: {len(state["jobs"])-len(pending)}/{len(state["jobs"])} complete.',flush=True)
    parallel(live_worker,pending,payload(state),workers,'Online jobs')
    results=completed_live(state)
    cal=evaluation_tasks(state,'calibration')
    parallel(evaluation_worker,cal,payload(state),workers,'Offline dense calibration')
    chosen=load_calibration(state)
    val=evaluation_tasks(state,'validation',validation_selection(state,results,chosen))
    parallel(evaluation_worker,val,payload(state),workers,'Offline holdout validation')
    analyze(state,workers)


def collect(state,workers):
    if state['specification']['mode']!='full':raise RuntimeError('Smoke is software verification only.')
    analyze(state,workers)
    path=state['out'].relative_to(REPO).as_posix()
    subprocess.run(['git','add','-f','--',path],cwd=REPO,check=True)
    print('Staged only '+path,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['preflight','run','smoke','status','analyze','collect'])
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--workers',type=int,default=min(6,os.cpu_count() or 1))
    parser.add_argument('--smoke-results',action='store_true')
    args=parser.parse_args()
    if args.workers<1:parser.error('--workers must be positive')
    state=setup(args.config,args.command=='smoke' or args.smoke_results)
    if args.command in ('run','smoke'):run(state,args.workers)
    elif args.command=='preflight':preflight(state)
    elif args.command=='analyze':analyze(state,args.workers)
    elif args.command=='collect':collect(state,args.workers)
    else:
        complete=0
        for job in state['jobs']:
            path=run_path(state,job)
            if path.exists() and read(path)['state']['round']==state['cfg']['queries_per_client']:complete+=1
        print(f'Online: {complete}/{len(state["jobs"])} complete; report: {state["out"]}/report.html')


if __name__=='__main__':main()
