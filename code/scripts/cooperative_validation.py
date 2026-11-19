"""Fixed-sample fresh-noise validation of frozen common/reference controllers."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from joblib import Parallel, delayed, parallel_config
from actuator_bandwidth_experiment import context, write_json
from actuator_corrected import digest, Progress
from actuator_diagnostics import integration_check
from src.actuator_mechanism import episode
from src.episodes import build_multi_maneuver_episode_bank

P=ROOT/'results'
ARMS=['target_redesign','reference']
N=1000
ALPHA=.05
FAMILY=120  # 6 vehicles x 5 bandwidths x 2 maneuvers x 2 controllers
BLOCK=25
TAG='cooperative_validation_v1'


def noise_seeds(seed,client,maneuver,n,smoke=False):
    # The old episode builder draws below 2**31. Reserve the upper uint32 half.
    # Include realization; common random numbers pair only arms and bandwidths.
    key=f'cooperative_validation_v1/{seed}/{client}/{maneuver:g}/{int(smoke)}'
    entropy=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')
    rng=np.random.default_rng(entropy)
    values=[];seen=set()
    while len(values)<n:
        v=int(rng.integers(2**31,2**32))
        if v not in seen:values.append(v);seen.add(v)
    return values


def mean_interval(values,threshold,family=FAMILY,alpha=ALPHA):
    x=np.asarray(values,dtype=float)
    if len(x)<2 or not np.isfinite(x).all():raise ValueError('Invalid mean sample')
    mean=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(len(x)))
    half=float(student_t.ppf(1-alpha/(2*family),len(x)-1)*se)
    low,high=mean-half,mean+half
    status='supported' if high<=threshold else 'exceeds' if low>threshold else 'unresolved'
    return dict(mean=mean,se=se,lower=low,upper=high,mean_status=status)


def inputs_and_protocol(smoke):
    oldpath=P/'actuator_diagnostics_v2/protocol.json'
    old=json.loads(oldpath.read_text())
    cfgpath=ROOT/'config/runs/cooperative_acceptance.json'
    cfg=json.loads(cfgpath.read_text())
    if not np.isclose(cfg['threshold_rad_per_s'],np.deg2rad(.92),rtol=0,atol=1e-15):
        raise ValueError('This frozen runner is for 0.92 deg/s')
    priorpath=P/'actuator_mechanism_v7/protocol.json'
    prior=json.loads(priorpath.read_text())
    # Preserve v7 numerical implementation and inputs, including corrected anti-windup.
    for rel,h in {**prior['code'],**prior['inputs']}.items():
        if digest(ROOT/rel)!=h:raise ValueError(f'Frozen v7 dependency changed: {rel}')
    targets=prior['targets'];taus=prior['taus']
    if smoke:targets=targets[:1];taus=[.2,.4]
    n=2 if smoke else N
    files=[oldpath,cfgpath,priorpath,Path(__file__)]
    models={};frozen={};noise={}
    for seed,client in targets:
        modelpath=P/f'actuator_bandwidth_v1_pilot/seeds/{seed}/sources.json'
        files.append(modelpath)
        m=json.loads(modelpath.read_text())['models'][client]
        models[(seed,client)]=(np.array(m['A']),np.array(m['B']))
        for tau in taus:
            f=P/f'actuator_mechanism_v7/seeds/{seed}/{client}/{tau:g}/gains.json'
            files.append(f)
            d=json.loads(f.read_text())
            frozen[(seed,client,tau)]={a:d['arms'][a] for a in ARMS}
        for maneuver in old['fleet']['lane_change_time']:
            noise[f'{seed}/{client}/{maneuver:g}']=noise_seeds(seed,client,maneuver,n,smoke)
    # Different maneuvers/vehicles must not accidentally share noise seeds.
    allnoise=[v for values in noise.values() for v in values]
    if len(set(allnoise))!=len(allnoise):raise ValueError('Noise seed collision')
    protocol=dict(version=1,smoke=smoke,targets=targets,taus=taus,arms=ARMS,
        episodes_per_maneuver=n,threshold_rad_per_s=cfg['threshold_rad_per_s'],alpha=ALPHA,
        simultaneous_means=FAMILY,interval='two-sided Bonferroni Student-t; approximate large-sample',
        sample_size_origin='fixed development-based planning: SD about 0.062 deg/s, simultaneous halfwidth about 0.007 deg/s at n=1000',
        noise_seeds=noise,hash_convention='SHA256 after CRLF->LF',
        files={f.relative_to(ROOT).as_posix():digest(f) for f in sorted(set(files))})
    return old,models,frozen,protocol


def run_target(target,old,models,frozen,protocol,out):
    seed,client_id=target
    fc,cc=old['fleet'],old['controller']
    progress=Progress(f'{seed}/{client_id}')
    progress.say('reconstructing fixed vehicle; no optimization')
    fleet,rebuilt=context(seed,old['base'],fc,cc)
    client=next(c for c in fleet if c.client_id==client_id)
    a,b=models[(seed,client_id)]
    for saved,check in zip((a,b),rebuilt[client_id]):np.testing.assert_allclose(saved,check,rtol=1e-10,atol=1e-10)
    n=protocol['episodes_per_maneuver']
    bank=build_multi_maneuver_episode_bank(client,Ts=fc['Ts'],T_total=cc['T_total'],T0=cc['T0'],r_max=cc['r_max'],
        lane_change_time=fc['lane_change_time'],n_calibration_per_maneuver=n,n_test_per_maneuver=0,
        base_seed=2900000,tfilter=cc.get('tfilter'))
    for maneuver in fc['lane_change_time']:
        eps=[e for e in bank.episodes if e.T_lc==maneuver]
        seeds=protocol['noise_seeds'][f'{seed}/{client_id}/{maneuver:g}']
        for ep,s in zip(eps,seeds):ep.seed=s;ep.role='test'
    for tau in protocol['taus']:
        for arm in ARMS:
            g=frozen[(seed,client_id,tau)][arm]
            for maneuver in fc['lane_change_time']:
                eps=[e for e in bank.episodes if e.T_lc==maneuver]
                for start in range(0,n,BLOCK):
                    file=out/f'seeds/{seed}/{client_id}/{tau:g}/{arm}/{maneuver:g}/{start:04d}.json'
                    expected=[e.seed for e in eps[start:start+BLOCK]]
                    if file.exists():
                        cached=json.loads(file.read_text())
                        if [r['noise_seed'] for r in cached]!=expected:raise ValueError(f'Invalid checkpoint: {file}')
                        progress.hits+=len(cached)
                        continue
                    rows=[]
                    for ep in eps[start:start+BLOCK]:
                        coarse=episode(client,a,b,tau,g,bank,ep,cc,fc['Ts'],5)
                        fine=episode(client,a,b,tau,g,bank,ep,cc,fc['Ts'],10)
                        rows.append(dict(seed=seed,client=client_id,tau=tau,arm=arm,maneuver=maneuver,
                            noise_seed=ep.seed,coarse=coarse,fine=fine,integration=integration_check(coarse,fine,.001)))
                        progress.new+=1
                        progress.say(f'{arm}, tau={tau:g}, maneuver={maneuver:g}, {progress.new} new episode pairs',False)
                    write_json(file,rows)
                progress.say(f'{arm}, tau={tau:g}, maneuver={maneuver:g}: {n} episodes complete')


def analyze(out,protocol):
    rows=[];summaries=[]
    n=protocol['episodes_per_maneuver'];threshold=protocol['threshold_rad_per_s']
    for seed,client in protocol['targets']:
        for tau in protocol['taus']:
            for arm in ARMS:
                for maneuver in [1.,2.]:
                    directory=out/f'seeds/{seed}/{client}/{tau:g}/{arm}/{maneuver:g}'
                    group=[]
                    for start in range(0,n,BLOCK):
                        f=directory/f'{start:04d}.json'
                        if not f.exists():raise ValueError(f'Incomplete run: missing {f}')
                        group.extend(json.loads(f.read_text()))
                    expected=protocol['noise_seeds'][f'{seed}/{client}/{maneuver:g}']
                    if [r['noise_seed'] for r in group]!=expected:raise ValueError(f'Wrong episode bank: {directory}')
                    identity=dict(seed=seed,client=client,tau=tau,arm=arm,maneuver=maneuver)
                    if any(any(r[k]!=v for k,v in identity.items()) for r in group):raise ValueError('Checkpoint identity mismatch')
                    failures=sum(not r['coarse']['feasible'] and not r['fine']['feasible'] for r in group)
                    unresolved=sum(r['integration']['status']!='pass' for r in group)
                    complete=all(r[z]['complete'] and np.isfinite(r[z]['cost']) for r in group for z in ['coarse','fine'])
                    for r in group:
                        rows.append(dict(**identity,noise_seed=r['noise_seed'],cost=r['coarse']['cost'],fine_cost=r['fine']['cost'],
                            true_cost=r['coarse']['true_cost'],feasible=r['coarse']['feasible'],fine_feasible=r['fine']['feasible'],
                            integration=r['integration']['status']))
                    result=dict(**identity,n=n,physical_failures=failures,numerically_unresolved=unresolved)
                    if complete:
                        # Include coarse/fine discrepancy conservatively in interval endpoints.
                        stats=mean_interval([r['coarse']['cost'] for r in group],threshold)
                        sensitivity=max(abs(r['coarse']['cost']-r['fine']['cost']) for r in group)
                        stats['lower']-=sensitivity;stats['upper']+=sensitivity
                        stats['mean_status']='supported' if stats['upper']<=threshold else 'exceeds' if stats['lower']>threshold else 'unresolved'
                        result.update(stats,numerical_margin=sensitivity)
                    else:result['mean_status']='unavailable'
                    result['status']='physical_failure' if failures else 'numerically_unresolved' if unresolved or not complete else result['mean_status']
                    if protocol['smoke']:result['status']='smoke_only'
                    summaries.append(result)
    processed=out/'processed';processed.mkdir(exist_ok=True)
    df=pd.DataFrame(summaries);df.to_csv(processed/'conditions.csv',index=False)
    pd.DataFrame(rows).to_csv(processed/'episodes.csv',index=False)
    arm_results={}
    for arm,g in df.groupby('arm'):
        status='supported' if g.status.eq('supported').all() else 'physical_failure' if g.status.eq('physical_failure').any() else 'exceeds' if g.status.eq('exceeds').any() else 'unresolved'
        if protocol['smoke']:status='smoke_only'
        arm_results[arm]=dict(status=status,condition_counts=g.status.value_counts().to_dict())
    write_json(processed/'summary.json',dict(arms=arm_results,smoke=protocol['smoke'],episode_pairs=len(rows),numerical_rollouts=2*len(rows),
        note='Approximate simultaneous mean inference on fixed simulated systems. No distribution-free safety certificate or new-vehicle guarantee. No adaptive sample extension.'))
    print(json.dumps(arm_results,indent=2),flush=True)
    return [processed/'conditions.csv',processed/'episodes.csv',processed/'summary.json']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['preflight','smoke','run','analyze','collect'])
    p.add_argument('--n-jobs',type=int,default=2)
    args=p.parse_args()
    if args.n_jobs<1:p.error('--n-jobs must be positive')
    smoke=args.stage=='smoke'
    old,models,frozen,protocol=inputs_and_protocol(smoke)
    print(f"Protocol: {len(protocol['targets'])} vehicles x {len(protocol['taus'])} bandwidths x 2 arms x 2 maneuvers x {protocol['episodes_per_maneuver']} episodes; 5/10 RK4 substeps.",flush=True)
    if args.stage=='preflight':
        print('Preflight passed: saved gains/dependencies verified; fresh noise bank fixed. No simulation.',flush=True);return
    out=P/(TAG+('_smoke' if smoke else ''))
    manifest=out/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text())!=protocol:raise ValueError('Protocol mismatch; do not mix runs')
    if args.stage in ['analyze','collect'] and not manifest.exists():raise ValueError('Run missing')
    write_json(manifest,protocol)
    if args.stage in ['run','smoke']:
        with parallel_config(backend='loky',inner_max_num_threads=1):
            Parallel(n_jobs=args.n_jobs)(delayed(run_target)(t,old,models,frozen,protocol,out) for t in protocol['targets'])
    outputs=analyze(out,protocol)
    if args.stage=='collect':
        files=[manifest]+outputs+sorted(out.glob('seeds/*/*/*/*/*/*.json'))
        for i in range(0,len(files),40):
            subprocess.run(['git','add','-f','--']+[f.relative_to(ROOT.parent).as_posix() for f in files[i:i+40]],cwd=ROOT.parent,check=True)
        print(f'Staged {len(files)} files. Commit and push.',flush=True)

if __name__=='__main__':main()
