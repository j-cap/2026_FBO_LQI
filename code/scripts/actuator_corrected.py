"""Corrected-controller calibration and reference rebuild. No legacy scores reused."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from scipy.stats import qmc
from joblib import Parallel, delayed, parallel_config
from actuator_bandwidth_experiment import context, valid, write_json
from actuator_diagnostics import make_banks, integration_check
from src.actuator_bandwidth import augment_discrete, design, step
from src.actuator_antiwindup import evaluate, integral_update
from src.episodes import build_multi_maneuver_episode_bank
from src.io_utils import load_yaml
from src.optimization.gp_surrogate import fit_weighted_gp, expected_improvement
from phase2_fbo_comparison import _resolve


def digest(path):
    return hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()


def sobol(n,seed):
    return qmc.Sobol(3,scramble=True,seed=seed).random_base2(math.ceil(math.log2(max(2,n))))[:n]


class Progress:
    def __init__(self,label,seconds=30):
        self.label=label;self.start=self.last=time.monotonic();self.hits=0;self.new=0;self.seconds=seconds
    def say(self,message,force=True):
        now=time.monotonic()
        if force or now-self.last>=self.seconds:
            print(f'[{self.label}] {message}; elapsed={(now-self.start)/60:.1f} min; cache={self.hits}, new={self.new}',flush=True)
            self.last=now


def optimize(score,method,budget,lower,upper,seed,n_candidates,scale,records=None,checkpoint=None,progress=None):
    """Ablations share the original initial pair and candidate pool; mask repeats."""
    records=list(records or [])
    initial=sobol(2,seed);grid=sobol(n_candidates,seed+1)
    for n in range(len(records),budget):
        if n<2:
            x=initial[n]
        else:
            observed=np.array([(np.array(r['xi'])-lower)/(upper-lower) for r in records])
            available=np.flatnonzero(~np.any(np.max(abs(grid[:,None,:]-observed[None,:,:]),axis=2)<1e-12,axis=1))
            if not len(available): raise ValueError('Candidate pool exhausted')
            candidates=grid[available]
            if method['kind']=='sobol': idx=0
            else:
                y=np.array([r['training_y'] for r in records])
                gp=fit_weighted_gp(observed,y,np.ones(n),seed=seed+n).gp
                good=[r['training_y'] for r in records if valid(r['score'])]
                if gp is None: idx=0
                else:
                    ei=expected_improvement(gp,candidates,min(good) if good else min(y),xi=method['margin'])
                    idx=int(np.argmax(ei))
            x=candidates[idx]
        xi=lower+x*(upper-lower);ev=score(xi)
        records.append(dict(xi=xi.tolist(),score=ev,
            training_y=ev['cost']/scale if valid(ev) else method.get('penalty',10.)))
        if checkpoint: checkpoint(records)
        if progress: progress.say(f'optimizer {n+1}/{budget}',force=(n+1==budget or (n+1)%10==0))
    return records


def incumbent(records):
    good=[r for r in records if valid(r['score'])]
    return min(good,key=lambda r:r['score']['cost'])['xi'] if good else None


def diverse_shortlist(records,n,lower,upper):
    """Half lowest cost, half farthest-point coverage over all feasible weights."""
    good=sorted((r for r in records if valid(r['calibration'])),key=lambda r:r['calibration']['cost'])
    chosen=good[:max(1,n//2)];remaining=good[len(chosen):]
    while remaining and len(chosen)<n:
        xs=np.array([r['xi'] for r in chosen])
        distances=[np.min(np.sum(((xs-r['xi'])/(upper-lower))**2,axis=1)) for r in remaining]
        chosen.append(remaining.pop(int(np.argmax(distances))))
    return chosen


def trace_episode(client,ac,bc,tau,xi,bank,ep,cc,ts,substeps):
    """Replay with corrected tracking back-calculation; return time history up to first constraint abort."""
    a,b=augment_discrete(ac,bc,tau,ts)
    try: K,kr,_=design(a,b,xi,cc['state_scale'],ts,cc['fixed_R'])
    except (ValueError,np.linalg.LinAlgError): return []
    _,ref=bank.refs[ep.T_lc];rng=np.random.RandomState(ep.seed)
    x=np.zeros(3);eta=kr*float(ref[0])/K[3];rows=[]
    for k,rr in enumerate(ref[:-1]):
        raw=-K[:3]@x-K[3]*eta+kr*rr;u=float(np.clip(raw,cc['u_min'],cc['u_max']))
        y=x[1]+cc['noise_std']*rng.randn()
        rows.append(dict(time=k*ts,beta=float(x[0]),yaw_rate=float(x[1]),delta=float(x[2]),
            command=u,raw_command=float(raw),reference=float(rr),measured_error=float(y-rr),
            actuator_rate=float((u-x[2])/tau)))
        eta=integral_update(eta,y-rr,raw,u,K[3],ts)
        for j in range(substeps):
            x=step(x,u,client.params,tau,ts/substeps,cc.get('tire_mu',1.))
            if not np.isfinite(x).all() or abs(x[0])>np.deg2rad(cc['beta_max_deg']):
                rows.append(dict(time=k*ts+(j+1)*ts/substeps,beta=float(x[0]),yaw_rate=float(x[1]),
                    delta=float(x[2]),command=u,raw_command=float(raw),reference=float(rr),
                    measured_error=None,actuator_rate=float((u-x[2])/tau)))
                return rows
        x[:2]+=rng.multivariate_normal(np.zeros(2),np.diag(cc['process_noise']))
    return rows


def run_vehicle(seed,cid,stage,cfg,old,out,smoke=False):
    base,fc,cc,acfg=old['base'],old['fleet'],old['controller'],old['actuator']
    path=out/'seeds'/str(seed)/cid
    path.mkdir(parents=True,exist_ok=True)
    progress=Progress(f'{stage} {seed}/{cid}',cfg['progress_seconds']);progress.say('starting')
    fleet,models=context(seed,base,fc,cc);client=next(c for c in fleet if c.client_id==cid);ac,bc=models[cid]
    banks=make_banks(client,old['config'],fc,cc)
    # Old test episodes are now DEVELOPMENT data. Never report them as fresh tests.
    for name,count,bankseed in [('validation_new',cfg['validation_per_maneuver'],cfg['validation_seed']),
                               ('test_new',cfg['test_per_maneuver'],cfg['test_seed'])]:
        banks[name]=build_multi_maneuver_episode_bank(client,Ts=fc['Ts'],T_total=cc['T_total'],T0=cc['T0'],
            r_max=cc['r_max'],lane_change_time=fc['lane_change_time'],n_calibration_per_maneuver=count,
            n_test_per_maneuver=0,base_seed=bankseed,tfilter=cc.get('tfilter'))
    banksets=[{e.seed for e in b.episodes} for b in banks.values()]
    if any(a&b for i,a in enumerate(banksets) for b in banksets[i+1:]): raise ValueError('Episode-bank overlap')
    if smoke:
        banks={k:replace(b,episodes=[next(e for e in b.episodes if e.T_lc==t) for t in fc['lane_change_time']]) for k,b in banks.items()}
    tau0=acfg['source_tau'];steps=acfg['substeps'];lower=np.array(cc['xi_lower']);upper=np.array(cc['xi_upper'])
    previous=ROOT/cfg['previous_results']/'seeds'/str(seed)
    def score(x,role='calibration',tau=tau0,substeps=steps):
        if role=='test_new' and stage!='test': raise RuntimeError('Final test access outside test stage')
        key=json.dumps([list(map(float,x)),role,float(tau),substeps]);f=path/'cache'/(hashlib.sha256(key.encode()).hexdigest()+'.json')
        if f.exists(): progress.hits+=1;ev=json.loads(f.read_text())
        else:
            ev=asdict(evaluate(client,ac,bc,tau,x,banks[role],'calibration',cc,fc['Ts'],substeps));write_json(f,ev);progress.new+=1
        progress.say(f'evaluating {role}, tau={tau:g}',False)
        return ev
    def episode_report(x,tau,role,label):
        f=path/'episodes'/f'{label}.json'
        if f.exists(): return json.loads(f.read_text())
        rows=[];bank=banks[role]
        for i,ep in enumerate(bank.episodes):
            one=replace(bank,episodes=[ep]);coarse=asdict(evaluate(client,ac,bc,tau,x,one,'calibration',cc,fc['Ts'],steps))
            fine=asdict(evaluate(client,ac,bc,tau,x,one,'calibration',cc,fc['Ts'],2*steps))
            rows.append(dict(index=i,noise_seed=ep.seed,maneuver=ep.T_lc,coarse=coarse,fine=fine,
                integration=integration_check(coarse,fine,.001)))
            progress.say(f'{label} episode {i+1}/{len(bank.episodes)}',False)
        # Full summary for every episode; trajectories for first failure and two
        # largest finite costs. Selection is for explanation, never calibration.
        failures=[r for r in rows if not valid(r['coarse'])]
        worst=sorted((r for r in rows if valid(r['coarse'])),key=lambda r:r['coarse']['cost'],reverse=True)[:2]
        chosen={r['index'] for r in failures[:1]+worst}
        traces={str(i):{str(n):trace_episode(client,ac,bc,tau,x,bank,bank.episodes[i],cc,fc['Ts'],n)
                        for n in [steps,2*steps]} for i in sorted(chosen)}
        write_json(f,dict(label=label,role=role,tau=tau,xi=x,episodes=rows,traces=traces))
        progress.say(f'{label}: {len(failures)}/{len(rows)} episodes infeasible')
        return json.loads(f.read_text())
    if stage=='audit':
        for optseed in cfg['optimizer_seeds']:
            for method,spec in cfg['methods'].items():
                f=path/'audit'/f'{method}_{optseed}.json'
                if f.exists(): progress.say(f'{method}/{optseed}: checkpoint complete');continue
                partial=path/'partial'/f'{method}_{optseed}.json'
                history=json.loads(partial.read_text()) if partial.exists() else []
                progress.say(f'{method}/{optseed}: resume at {len(history)}')
                records=optimize(score,spec,max(cfg['budgets']),lower,upper,optseed,acfg['n_candidates'],cc['r_max'],
                    history,lambda r:write_json(partial,r),progress)
                write_json(f,dict(method=method,optimizer_seed=optseed,records=records))
    elif stage=='references':
        for tau in ([tau0] if smoke else acfg['target_taus']):
            f=path/'references'/f'{tau:g}.json'
            if f.exists():progress.say(f'tau={tau:g}: checkpoint complete');continue
            prior=json.loads((previous/'references'/f'{cid}_{tau:g}.json').read_text())
            pool={}
            def add(x,origin):
                key=tuple(float(v) for v in x)
                if key not in pool:pool[key]=dict(xi=list(key),origins=[])
                pool[key]['origins'].append(origin)
            for x in lower+sobol(cfg['reference_grid'],seed+1900)*(upper-lower):add(x,'new_sobol')
            for r in prior['shortlist']:add(r['xi'],'v2_shortlist')
            repaired=ROOT/cfg['repair_results']/'seeds'/str(seed)/cid/'references'/f'{tau:g}.json'
            for r in json.loads(repaired.read_text())['shortlist']:add(r['xi'],'v3_shortlist')
            # Source incumbents from ALL methods/seeds/budgets, not test winners.
            for optseed in cfg['optimizer_seeds']:
                histories=[]
                for method in cfg['methods']:
                    af=path/'audit'/f'{method}_{optseed}.json'
                    if not af.exists():raise FileNotFoundError('Complete audit before references')
                    histories.append(json.loads(af.read_text())['records'])
                for history in histories:
                    for b in cfg['budgets']:
                        x=incumbent(history[:b])
                        if x is not None:add(x,'source_incumbent')
            # Low integral weights are conservative candidates, not presumed safe.
            for qi in [-3.,-2.,-1.,0.,1.]:add(np.clip([0.,0.,qi],lower,upper),'anchor')
            records=list(pool.values());progress.say(f'tau={tau:g}: screen {len(records)} candidates')
            for i,r in enumerate(records):
                r['calibration']=score(r['xi'],'calibration',tau)
                if (i+1)%32==0:progress.say(f'tau={tau:g}: screen {i+1}/{len(records)}')
            shortlist=diverse_shortlist(records,cfg['reference_shortlist'],lower,upper)
            progress.say(f'tau={tau:g}: validate {len(shortlist)} candidates')
            for i,r in enumerate(shortlist):
                r['validation']=score(r['xi'],'validation_new',tau)
                fine=score(r['xi'],'validation_new',tau,2*steps)
                r['validation_integration']=integration_check(r['validation'],fine,.001)
                progress.say(f'tau={tau:g}: validation {i+1}/{len(shortlist)}')
            eligible=[r for r in shortlist if valid(r['validation']) and r['validation_integration']['status']=='pass']
            selected=min(eligible,key=lambda r:r['validation']['cost']) if eligible else None
            write_json(f,dict(tau=tau,candidates=records,shortlist=shortlist,selected=selected,
                status='validated_benchmark' if selected else 'unresolved'))
            progress.say(f'tau={tau:g}: reference {"selected" if selected else "unresolved"}')
    elif stage=='test':
        for tau in ([tau0] if smoke else acfg['target_taus']):
            f=path/'assessment'/f'reference_{tau:g}.json'
            if f.exists():continue
            ref=json.loads((path/'references'/f'{tau:g}.json').read_text());r=ref['selected']
            ev=score(r['xi'],'test_new',tau) if r else None
            fine=score(r['xi'],'test_new',tau,2*steps) if r else None
            ic=integration_check(ev,fine,.001) if r else None
            report=episode_report(r['xi'],tau,'test_new',f'new_reference_{tau:g}') if r else None
            episodes_pass=bool(report and all(valid(e['coarse']) and valid(e['fine']) and e['integration']['status']=='pass' for e in report['episodes']))
            write_json(f,dict(kind='reference',tau=tau,xi=r['xi'] if r else None,test=ev,integration=ic,
                episodes_pass=episodes_pass,usable=bool(ev and valid(ev) and ic['status']=='pass' and episodes_pass)))
        for optseed in cfg['optimizer_seeds']:
            for method in list(cfg['methods']):
                f=path/'assessment'/f'{method}_{optseed}.json'
                if f.exists():continue
                hf=path/'audit'/f'{method}_{optseed}.json'
                records=json.loads(hf.read_text())['records'];rows=[]
                for b in cfg['budgets']:
                    x=incumbent(records[:b]);ev=score(x,'test_new') if x is not None else None
                    fine=score(x,'test_new',substeps=2*steps) if x is not None else None
                    rows.append(dict(budget=b,xi=x,test=ev,integration=integration_check(ev,fine,.001) if ev else None,
                        calibration_best=min((r['score']['cost'] for r in records[:b] if valid(r['score'])),default=None),
                        unique_evaluations=len({tuple(r['xi']) for r in records[:b]}),
                        infeasible_evaluations=sum(not valid(r['score']) for r in records[:b])))
                write_json(f,dict(kind='optimizer',method=method,optimizer_seed=optseed,rows=rows))
    progress.say('complete')


def analyze(out,targets,cfg,old,smoke=False):
    refs=[];rows=[];episodes=[]
    taus=[old['actuator']['source_tau']] if smoke else old['actuator']['target_taus']
    for seed,cid in targets:
        path=out/'seeds'/str(seed)/cid
        for tau in taus:
            r=json.loads((path/'assessment'/f'reference_{tau:g}.json').read_text());ev=r['test'] or {}
            refs.append(dict(seed=seed,client_id=cid,tau=tau,usable=r['usable'],cost=ev.get('cost'),
                reason=ev.get('reason','no_validated_benchmark'),integration=(r['integration'] or {}).get('status')))
        matched=json.loads((path/'assessment'/f'reference_{old["actuator"]["source_tau"]:g}.json').read_text())
        for method in list(cfg['methods']):
            for optseed in cfg['optimizer_seeds']:
                d=json.loads((path/'assessment'/f'{method}_{optseed}.json').read_text())
                for r in d['rows']:
                    ev=r['test'];regret=None
                    if ev is None or not valid(ev):status='infeasible'
                    elif r['integration']['status']!='pass':status='unresolved_integration'
                    elif not matched['usable']:status='unresolved_reference'
                    else:
                        regret=ev['cost']/matched['test']['cost']-1
                        status='success' if regret<=.05 else 'inaccurate'
                    rows.append(dict(seed=seed,client_id=cid,method=method,optimizer_seed=optseed,budget=r['budget'],
                        status=status,regret=regret,cost=ev['cost'] if ev else None,
                        feasible=bool(ev and valid(ev)),calibration_best=r['calibration_best'],
                        unique_evaluations=r['unique_evaluations'],infeasible_evaluations=r['infeasible_evaluations']))
        for f in (path/'episodes').glob('*.json'):
            d=json.loads(f.read_text())
            for r in d['episodes']:
                episodes.append(dict(seed=seed,client_id=cid,label=d['label'],role=d['role'],tau=d['tau'],
                    noise_seed=r['noise_seed'],maneuver=r['maneuver'],**r['coarse'],integration_status=r['integration']['status']))
    processed=out/'processed';processed.mkdir(exist_ok=True)
    for name,data in [('reference_assessment',refs),('optimizer_assessment',rows),('episode_assessment',episodes)]:
        pd.DataFrame(data).to_csv(processed/f'{name}.csv',index=False)
    counts=pd.DataFrame(rows).groupby(['method','budget','status']).size().rename('count').reset_index()
    counts.to_csv(processed/'status_counts.csv',index=False);print(counts.to_string(index=False))
    print('Benchmarks are validated candidates, not global optima. Retain negative gaps and unresolved cases.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['preflight','smoke','audit','references','test','analyze','collect'])
    p.add_argument('--config',default='config/runs/actuator_corrected.yaml');p.add_argument('--n-jobs',type=int,default=2)
    a=p.parse_args();cfg=load_yaml(_resolve(a.config));previous=ROOT/cfg['previous_results']
    old=json.loads((previous/'protocol.json').read_text())
    targets=[(s,f'{f}_{slot}') for s in old['config']['fleet_seeds'] for f in ['nominal','payload','tire_degraded'] for slot in old['config']['target_slots']]
    smoke=a.stage=='smoke'
    if smoke:
        targets=targets[:1];cfg.update(budgets=[3],optimizer_seeds=[101],reference_grid=8,reference_shortlist=4,
            validation_per_maneuver=1,test_per_maneuver=1)
    # Validate all original hashed code; new scripts do not change the v2 evaluator.
    for f,h in old['code'].items():
        if digest(ROOT/f)!=h:raise ValueError(f'Original code changed: {f}')
    prior_files=[previous/'protocol.json']+sorted(previous.glob('seeds/*/sources/*.json'))+sorted(previous.glob('seeds/*/references/*.json'))+sorted(previous.glob('seeds/*/tests/*.json'))
    prior_files+=sorted((ROOT/cfg['repair_results']).glob('seeds/*/*/references/*.json'))
    files=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),ROOT/'scripts/actuator_diagnostics.py',ROOT/'scripts/actuator_bandwidth_experiment.py',ROOT/'scripts/phase2_fbo_comparison.py']
    protocol=dict(antiwindup=dict(mode='tracking',tracking_time=.2),config=cfg,targets=[list(t) for t in targets],smoke=smoke,
        prior_files={f.relative_to(ROOT).as_posix():digest(f) for f in prior_files},code={f.relative_to(ROOT).as_posix():digest(f) for f in files})
    out=ROOT/'results'/(cfg['run_tag']+('_smoke' if smoke else ''))
    if a.stage=='preflight':
        for seed,cid in targets:
            for tau in old['actuator']['target_taus']:
                for f in [previous/'seeds'/str(seed)/'references'/f'{cid}_{tau:g}.json', ROOT/cfg['repair_results']/'seeds'/str(seed)/cid/'references'/f'{tau:g}.json']:
                    if not f.exists():raise FileNotFoundError(f)
        print(f'Preflight passed: {len(targets)} fixed vehicles, {len(cfg["methods"])} new optimizer arms; v2 hashes verified.');return
    out.mkdir(parents=True,exist_ok=True);manifest=out/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text())!=protocol:raise ValueError('Protocol changed; do not overwrite existing results')
    write_json(manifest,protocol)
    if a.stage=='collect':
        analyze(out,targets,cfg,old)
        # Explicit files, Windows-safe, force-add only deliverables; never caches.
        import subprocess
        selected=[manifest]+sorted((out/'processed').glob('*.csv'))
        for folder in ['audit','references','assessment','episodes']:
            selected+=sorted(out.glob(f'seeds/*/*/{folder}/*.json'))
        if not (out/'processed/optimizer_assessment.csv').exists():raise ValueError('Run test/analyze before collect')
        for start in range(0,len(selected),30):
            subprocess.run(['git','add','-f','--']+[f.relative_to(ROOT.parent).as_posix() for f in selected[start:start+30]],cwd=ROOT.parent,check=True)
        print(f'Staged {len(selected)} result files. Review with git diff --cached --stat, then commit and push.');return
    if a.stage=='analyze':analyze(out,targets,cfg,old);return
    stages=['audit','references','test'] if smoke else [a.stage]
    with parallel_config(backend='loky',inner_max_num_threads=1):
        for stage in stages:
            Parallel(n_jobs=a.n_jobs)(delayed(run_vehicle)(s,c,stage,cfg,old,out,smoke) for s,c in targets)
    if a.stage in ['test','smoke']:analyze(out,targets,cfg,old,smoke)


if __name__=='__main__':main()
