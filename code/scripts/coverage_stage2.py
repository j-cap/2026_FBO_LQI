"""Stage 2: imperfect coverage, equal budgets, three held-out vehicle roles."""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import pandas as pd
import yaml
from scipy.stats import qmc
from mismatch_pilot import ROOT,save,identify,evaluate
from coverage_transfer import valid,local_summary
from correction_transfer import choose,predicted,observed
from src.ifac_bridge import nominal_params


def merge_summaries(parts):
    merged={}
    for part in parts:
        for d,s in part.items():
            z=merged.setdefault(d,dict(n=0,sums=dict(E=0.,S=0.)))
            z['n']+=s['n']
            for m in ['E','S']:z['sums'][m]+=s['sums'][m]
    return merged


def interpolate(summary,d):
    # No extrapolation: hold nearest available correction outside observed demand range.
    good=sorted(k for k,s in summary.items() if s['n']>=3)
    if not good:return None
    return {m:float(np.exp(np.interp(d*d,np.square(good),[summary[k]['sums'][m]/summary[k]['n'] for k in good]))) for m in ['E','S']}


def summarize(rows):
    return {d:local_summary([r for r in rows if r['demand']==d]) for d in sorted({r['demand'] for r in rows})}


def allocate(lookup,vehicle,counts,demands):
    return [lookup[vehicle,k,d] for d,n in zip(demands,counts) for k in range(n)]


def library(rows,d):
    # Fixed direct-weight baseline: geometric means of observed E/d,S/d per weight/context.
    result={}
    for k in sorted({r['candidate'] for r in rows}):
        good=[r for r in rows if r['candidate']==k and valid(r)]
        ds=sorted({r['demand'] for r in good})
        if ds:
            result[k]={m:float(d*np.exp(np.interp(d*d,np.square(ds),
                [np.mean([np.log(r['plant'][m]/dd) for r in good if r['demand']==dd]) for dd in ds]))) for m in ['E','S']}
    return result


def case_keys(cfg):
    return {(v,k,d) for v in ['nominal','payload','tire'] for k in range(cfg['calibration_count']+cfg['evaluation_count']) for d in cfg['evaluation_demands']}


def analyze(out,cfg,base,fractions):
    rs=[json.loads(p.read_text()) for p in (out/'cases').glob('*.json')]
    lookup={(r['vehicle'],r['candidate'],r['demand']):r for r in rs}
    if set(lookup)!=case_keys(cfg) or len(rs)!=len(lookup):raise RuntimeError('Incomplete or duplicate cases')
    decisions=[];errors=[];ledger=[];shared=[];max_equivalence=0.
    vehicles=['nominal','payload','tire']
    for i,target in enumerate(vehicles):
        s1,s2=vehicles[(i+1)%3],vehicles[(i+2)%3]
        for pattern,(a1,a2) in cfg['patterns'].items():
            r1=allocate(lookup,s1,a1,cfg['training_demands']);r2=allocate(lookup,s2,a2,cfg['training_demands'])
            pools={'source1_only':allocate(lookup,s1,[2*n for n in a1],cfg['training_demands']),
                   'source2_only':allocate(lookup,s2,[2*n for n in a2],cfg['training_demands']),
                   'source1_full':allocate(lookup,s1,cfg['single_full_allocation'],cfg['training_demands']),
                   'source2_full':allocate(lookup,s2,cfg['single_full_allocation'],cfg['training_demands'])}
            budget=sum(a1)+sum(a2)
            assert all(len(rows)==budget for rows in pools.values())
            assert all(r['vehicle']!=target for r in r1+r2+sum(pools.values(),[]))
            summaries={name:summarize(rows) for name,rows in pools.items()}
            summaries['pooled']=merge_summaries([summarize(r1),summarize(r2)])
            central=summarize(r1+r2)
            for name,rows in {**pools,'pooled':r1+r2,'direct_library':r1+r2}.items():
                ledger.append(dict(target=target,pattern=pattern,method=name,budget=len(rows),
                    records=[dict(vehicle=r['vehicle'],candidate=r['candidate'],demand=r['demand']) for r in rows]))
            shared.append(dict(target=target,pattern=pattern,source1=s1,source2=s2,summaries=summaries))
            for d in cfg['evaluation_demands']:
                tgt=[r for r in rs if r['vehicle']==target and r['demand']==d];actual=observed(tgt)
                policies={name:predicted(tgt,interpolate(s,d)) for name,s in summaries.items()}
                f=interpolate(summaries['pooled'],d);c=interpolate(central,d)
                assert (f is None)==(c is None)
                if f is not None:
                    diff=max(abs(f[m]-c[m]) for m in ['E','S']);max_equivalence=max(max_equivalence,diff)
                    if diff>1e-10:raise RuntimeError('Distributed and centralized fits disagree')
                policies.update(model_only=predicted(tgt,dict(E=1.,S=1.)),fixed_margin=predicted(tgt,dict(E=cfg['fixed_margin'],S=1.)),direct_library=library(r1+r2,d))
                by_id={r['candidate']:r for r in tgt}
                for method,pred in policies.items():
                    for frac in fractions:
                        limit=float(np.rad2deg(d/15)*frac);selected=choose(pred,limit);oracle=choose(actual,limit)
                        status='abstains';excess=None
                        if selected is not None:
                            r=by_id[selected]
                            if 'plant' not in r:status='synthesis_failure'
                            elif not r['plant']['physical']:status='physical_failure'
                            elif not r['numerical_resolved']:status='numerical_unknown'
                            else:
                                e=r['plant']['E'];tol=r['differences']['E']
                                status='boundary_unknown' if abs(e-limit)<=tol else 'passes' if e<=limit else 'tracking_violation'
                                if status=='passes' and oracle is not None:excess=100*(r['plant']['S']/actual[oracle]['S']-1)
                        decisions.append(dict(target=target,source1=s1,source2=s2,pattern=pattern,demand=d,
                            unseen_demand=d not in cfg['training_demands'],fraction=frac,limit_deg_s=limit,method=method,
                            selected=selected,status=status,reference=oracle,sampled_feasible=oracle is not None,steering_excess_percent=excess))
                    for k,y in pred.items():
                        if k<cfg['calibration_count'] or k not in actual:continue
                        for metric in ['E','S']:errors.append(dict(target=target,pattern=pattern,demand=d,method=method,candidate=k,metric=metric,absolute_error=abs(y[metric]-actual[k][metric])))
    save(out/'calibration_ledger.json',ledger);save(out/'shared_summaries.json',shared)
    df=pd.DataFrame(decisions);df.to_csv(out/'decisions.csv',index=False);pd.DataFrame(errors).to_csv(out/'heldout_prediction_errors.csv',index=False)
    tables=[]
    for label,sub in [('all',df),('unseen',df[df.unseen_demand])]:
        counts=sub.groupby(['target','pattern','method','status']).size().unstack(fill_value=0)
        counts.to_csv(out/f'{label}_outcomes.csv')
        quality=sub[sub.status=='passes'].groupby(['target','pattern','method']).steering_excess_percent.agg(['mean','median','max'])
        quality.to_csv(out/f'{label}_steering_excess.csv')
        tables.append((label,counts.join(quality)))
    save(out/'summary.json',dict(cases=len(rs),synthesis_failures=sum('plant' not in r for r in rs),
        physical_failures=sum('plant' in r and not r['plant']['physical'] for r in rs),
        numerical_unknown=sum('plant' in r and r['plant']['physical'] and not r['numerical_resolved'] for r in rs),
        centralized_equivalence_max_abs=max_equivalence,calibration_budget=budget,requests=len(df),
        identification_seconds_per_vehicle=base['identification_duration']))
    html='<html><meta charset="utf-8"><title>Coverage Stage 2</title><style>body{font:15px sans-serif;max-width:1300px;margin:35px auto}td,th{padding:7px}details{margin:20px 0}table{border-collapse:collapse}</style><h1>Uneven coverage: three target rotations</h1><p>Equal24-experiment calibration budgets; three source-coverage patterns; two unseen demands. Counts are correlated requests, not success probabilities. Mean/median/max are steering excess [%] on passing requests only. Inspect violations, abstention and tail excess jointly.</p>'
    for label,tab in tables:
        html+=f'<h2>{label.title()} demands</h2>'
        for target in vehicles:
            for pattern in cfg['patterns']:
                html+=f'<details open><summary>{target} target — {pattern}</summary>'+tab.loc[(target,pattern)].to_html()+'</details>'
    html+='<p>Pooling beating restricted sources but matching a full-coverage source supports complementary access, not an intrinsic multi-vehicle advantage. Narrow redundant coverage is a negative control. Source roles are cyclic, not optimized. Centralized and distributed fits use identical records and are checked numerically. No formal privacy or safety guarantee. All variants reuse simulations offline; each policy receives only its recorded budget.</p></html>'
    (out/'report.html').write_text(html,encoding='utf-8');print(f'Report: {out / "report.html"}',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['smoke','run','analyze','collect']);args=ap.parse_args()
    cfg=json.loads((ROOT/'config/runs/coverage_stage2.json').read_text());base=json.loads((ROOT/'config/runs/mismatch_pilot.json').read_text());prior=json.loads((ROOT/'config/runs/correction_transfer.json').read_text())
    if args.command=='smoke':cfg.update(tag='coverage_stage2_smoke',evaluation_count=2)
    out=ROOT/'results'/cfg['tag'];out.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),*[ROOT/'scripts'/f for f in ['mismatch_pilot.py','correction_transfer.py','coverage_transfer.py']],*[ROOT/'config/runs'/f for f in ['coverage_stage2.json','mismatch_pilot.json','correction_transfer.json']],ROOT/'config/controller.yaml',*[ROOT/'src'/f for f in ['actuator_bandwidth.py','actuator_antiwindup.py','nonlinear_plant.py','ifac/utils.py']]]
    sig=hashlib.sha256(json.dumps([cfg,base,prior],sort_keys=True).encode()+b''.join(p.read_bytes().replace(b'\r\n',b'\n') for p in files)).hexdigest();mp=out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())['signature']!=sig:raise RuntimeError('Changed inputs: preserve results and choose a new tag')
    if args.command in ['run','smoke']:
        p=nominal_params();vehicles=dict(nominal=p,payload=replace(p,m=p.m*1.2,Iz=p.Iz*1.25),tire=replace(p,Cf=p.Cf*.75))
        def sample(n,seed):return qmc.scale(qmc.Sobol(3,scramble=True,seed=seed).random_base2(int(np.ceil(np.log2(n))))[:n],base['xi_lower'],base['xi_upper'])
        X=np.vstack([sample(cfg['calibration_count'],cfg['calibration_seed']),sample(cfg['evaluation_count'],cfg['evaluation_seed'])]);assert len(np.unique(X,axis=0))==len(X)
        save(mp,dict(signature=sig,config=cfg,base=base,vehicles={k:asdict(v) for k,v in vehicles.items()}));save(out/'candidates.json',X.tolist())
        cc=yaml.safe_load((ROOT/'config/controller.yaml').read_text());start=time.monotonic();done=0;total=len(case_keys(cfg))
        for name,p in vehicles.items():
            print(f'Identifying {name}...',flush=True);a,b,tau,fit,u,y=identify(base,p)
            save(out/f'{name}_identification.json',dict(**fit,A=a.tolist(),B=b.tolist()));np.savez_compressed(out/f'{name}_identification.npz',u=u,x=y)
            for k,xi in enumerate(X):
                for j,d in enumerate(cfg['evaluation_demands']):
                    path=out/'cases'/f'{name}_{k:03d}_{j}.json'
                    if not path.exists():save(path,dict(vehicle=name,**evaluate(k,xi,d,p,base,cc,a,b,tau)))
                    done+=1
                print(f'{done}/{total} cases | {time.monotonic()-start:.1f}s',flush=True)
        analyze(out,cfg,base,prior['fractions'])
    else:
        if not mp.exists():raise RuntimeError('Run first')
        analyze(out,cfg,base,prior['fractions'])
        if args.command=='collect':subprocess.run(['git','add','-f','--',str(out.relative_to(ROOT.parent))],cwd=ROOT.parent,check=True)

if __name__=='__main__':main()
