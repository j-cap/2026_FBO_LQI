"""Stage 3: fixed fleet prior versus local evidence at equal target budgets."""
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
from coverage_transfer import valid
from coverage_stage2 import summarize,merge_summaries,interpolate,allocate,library
from correction_transfer import choose,observed
from src.ifac_bridge import nominal_params


def schedule(budget,demands):
    return [(j//len(demands),demands[j%len(demands)]) for j in range(budget)]


def feature(d):return np.array([1.,(d/7.)**2])


def adjust(rows,prior,ridge):
    """Same two-parameter ridge adjustment for local and fleet-prior methods."""
    good=[r for r in rows if valid(r)]
    X=np.array([feature(r['demand']) for r in good]).reshape(-1,2)
    coefficients={}
    for m in ['E','S']:
        y=np.array([np.log(r['plant'][m]/r['prediction'][m])-np.log(prior(r['demand'])[m]) for r in good])
        coefficients[m]=np.linalg.solve(X.T@X+ridge*np.eye(2),X.T@y)
    return coefficients


def predict(rows,prior,coef):
    return {r['candidate']:{m:float(r['prediction'][m]*prior(r['demand'])[m]*np.exp(feature(r['demand'])@coef[m])) for m in ['E','S']}
            for r in rows if 'prediction' in r and r['prediction']['physical']}


def source_prior(source_records,target):
    names=['nominal','payload','tire'];i=names.index(target);s1,s2=names[(i+1)%3],names[(i+2)%3]
    lookup={(r['vehicle'],r['candidate'],r['demand']):r for r in source_records}
    a=allocate(lookup,s1,[8,4,0],[1.5,4.,7.]);b=allocate(lookup,s2,[0,4,8],[1.5,4.,7.])
    assert len(a+b)==24 and all(r['vehicle']!=target for r in a+b)
    summary=merge_summaries([summarize(a),summarize(b)])
    return summary,[dict(vehicle=r['vehicle'],candidate=r['candidate'],demand=r['demand']) for r in a+b]


def analyze(out,cfg,base,fractions,source_records):
    rs=[json.loads(p.read_text()) for p in (out/'cases').glob('*.json')]
    names=['nominal','payload','tire'];expected={(v,k,d) for v in names for k in range(cfg['calibration_count']+cfg['evaluation_count']) for d in cfg['evaluation_demands']}
    lookup={(r['vehicle'],r['candidate'],r['demand']):r for r in rs}
    if set(lookup)!=expected or len(rs)!=len(expected):raise RuntimeError('Incomplete or duplicate cases')
    decisions=[];fits=[];ledger=[];errors=[]
    unit=lambda d:dict(E=1.,S=1.)
    for target in names:
        summary,source_ids=source_prior(source_records,target)
        prior=lambda d:interpolate(summary,d)
        if any(prior(d) is None for d in cfg['evaluation_demands']):raise RuntimeError('Frozen fleet prior unavailable')
        for budget in cfg['budgets']:
            chosen=[lookup[target,k,d] for k,d in schedule(budget,cfg['training_demands'])]
            ledger.append(dict(target=target,budget=budget,target_records=[dict(candidate=r['candidate'],demand=r['demand']) for r in chosen],source_records=source_ids))
            local=adjust(chosen,unit,cfg['ridge']);fleet=adjust(chosen,prior,cfg['ridge']);zero={m:np.zeros(2) for m in ['E','S']}
            fits.append(dict(target=target,budget=budget,usable=sum(valid(r) for r in chosen),local={m:v.tolist() for m,v in local.items()},fleet={m:v.tolist() for m,v in fleet.items()}))
            for d in cfg['evaluation_demands']:
                tgt=[r for r in rs if r['vehicle']==target and r['demand']==d];actual=observed(tgt);by_id={r['candidate']:r for r in tgt}
                policies=dict(local_correction=predict(tgt,unit,local),fleet_adjusted=predict(tgt,prior,fleet),
                    fleet_frozen=predict(tgt,prior,zero),model_only=predict(tgt,unit,zero),
                    fixed_margin=predict(tgt,lambda dd:dict(E=cfg['fixed_margin'],S=1.),zero),
                    direct_local_library=library(chosen,d))
                for method,pred in policies.items():
                    used_budget=budget if method in ['local_correction','fleet_adjusted','direct_local_library'] else 0
                    for frac in fractions:
                        limit=float(np.rad2deg(d/15)*frac);selected=choose(pred,limit);oracle=choose(actual,limit);status='abstains';excess=None
                        if selected is not None:
                            r=by_id[selected]
                            if 'plant' not in r:status='synthesis_failure'
                            elif not r['plant']['physical']:status='physical_failure'
                            elif not r['numerical_resolved']:status='numerical_unknown'
                            else:
                                e=r['plant']['E'];tol=r['differences']['E'];status='boundary_unknown' if abs(e-limit)<=tol else 'passes' if e<=limit else 'tracking_violation'
                                if status=='passes' and oracle is not None:excess=100*(r['plant']['S']/actual[oracle]['S']-1)
                        decisions.append(dict(target=target,budget_axis=budget,target_budget_used=used_budget,source_budget=24 if method.startswith('fleet') else 0,
                            demand=d,unseen_demand=d not in cfg['training_demands'],fraction=frac,limit_deg_s=limit,method=method,selected=selected,status=status,
                            reference=oracle,sampled_feasible=oracle is not None,steering_excess_percent=excess))
                    for k,y in pred.items():
                        if k<cfg['calibration_count'] or k not in actual:continue
                        for metric in ['E','S']:errors.append(dict(target=target,budget=budget,demand=d,method=method,candidate=k,metric=metric,absolute_error=abs(y[metric]-actual[k][metric])))
    df=pd.DataFrame(decisions);df.to_csv(out/'decisions.csv',index=False);pd.DataFrame(errors).to_csv(out/'heldout_prediction_errors.csv',index=False)
    save(out/'calibration_ledger.json',ledger);save(out/'fits.json',fits)
    metrics=[];gate=cfg['quality_gate']
    for (t,b,m),g in df.groupby(['target','budget_axis','method']):
        feasible=int(g.sampled_feasible.sum());success=g[g.status=='passes'];viol=int((g.status=='tracking_violation').sum());other=int(g.status.isin(['physical_failure','synthesis_failure','numerical_unknown','boundary_unknown']).sum())
        mean=float(success.steering_excess_percent.mean()) if len(success) else None;worst=float(success.steering_excess_percent.max()) if len(success) else None
        coverage=len(success)/feasible if feasible else None
        passed=bool(feasible and viol<=gate['max_violations'] and other==0 and coverage>=gate['min_feasible_coverage'] and mean is not None and mean<=gate['mean_steering_excess_max'] and worst<=gate['max_steering_excess_max'])
        metrics.append(dict(target=t,budget=b,method=m,passes=len(success),violations=viol,abstains=int((g.status=='abstains').sum()),other_failures=other,sampled_feasible=feasible,coverage=coverage,mean_excess=mean,max_excess=worst,quality_gate=passed))
    mf=pd.DataFrame(metrics);mf.to_csv(out/'budget_metrics.csv',index=False)
    first=mf[mf.quality_gate].groupby(['target','method']).budget.min().rename('first_passing_budget');first.to_csv(out/'first_passing_budget.csv')
    save(out/'summary.json',dict(cases=len(rs),synthesis_failures=sum('plant' not in r for r in rs),physical_failures=sum('plant' in r and not r['plant']['physical'] for r in rs),
        numerical_unknown=sum('plant' in r and r['plant']['physical'] and not r['numerical_resolved'] for r in rs),quality_gate=gate))
    html='<html><meta charset="utf-8"><title>Stage3 target budgets</title><style>body{font:15px sans-serif;max-width:1300px;margin:35px auto}td,th{padding:7px}</style><h1>Does fleet knowledge save target calibration?</h1><p>Three targets; fresh weights; nested0/4/8/16 target episodes. Historical fleet prior is frozen from Stage2 complementary coverage. Fixed baselines are repeated along the budget axis but consume zero target episodes. Correlated requests are not statistical trials.</p>'
    html+='<p>Developmental quality gate: zero violations/other failures; at least95% coverage of sampled-feasible requests; mean steering excess≤2%, worst≤10%. This is a declared comparison criterion, not an engineering guarantee. Missing entries below mean no tested budget met the gate.</p>'+first.to_frame().to_html()
    for target in names:html+=f'<h2>{target}</h2>'+mf[mf.target==target].drop(columns='target').to_html(index=False)
    html+='<p>Local and fleet-adjusted methods use the same two-feature ridge adjustment and fixed regularization. Positive results establish value relative to this local learner, not optimal local calibration. Direct local library is a simple fixed-sequence baseline, not BO. Better local identification is a separate outstanding comparison. Source history costs24 episodes per target prior plus source identification; target identification is20 seconds for every method. Frozen baselines must not be counted four times.</p></html>'
    (out/'report.html').write_text(html,encoding='utf-8');print(f'Report: {out / "report.html"}',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['smoke','run','analyze','collect']);ap.add_argument('--source-root',type=Path,default=ROOT/'results');args=ap.parse_args()
    cfg=json.loads((ROOT/'config/runs/coverage_stage3.json').read_text());base=json.loads((ROOT/'config/runs/mismatch_pilot.json').read_text());prior=json.loads((ROOT/'config/runs/correction_transfer.json').read_text())
    if args.command=='smoke':cfg.update(tag='coverage_stage3_smoke',evaluation_count=2)
    src=args.source_root/cfg['stage2_tag'];source_files=sorted((src/'cases').glob('*.json'))
    if len(source_files)!=720:raise RuntimeError('Need committed complete coverage_stage2_v1 results; git pull first')
    source_records=[json.loads(p.read_text()) for p in source_files]
    out=ROOT/'results'/cfg['tag'];out.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),*[ROOT/'scripts'/f for f in ['mismatch_pilot.py','correction_transfer.py','coverage_transfer.py','coverage_stage2.py']],*[ROOT/'config/runs'/f for f in ['coverage_stage3.json','mismatch_pilot.json','correction_transfer.json']],ROOT/'config/controller.yaml',*[ROOT/'src'/f for f in ['actuator_bandwidth.py','actuator_antiwindup.py','nonlinear_plant.py','ifac/utils.py']]]
    source_hash=hashlib.sha256(json.dumps(source_records,sort_keys=True).encode()).hexdigest()
    sig=hashlib.sha256(json.dumps([cfg,base,prior,source_hash],sort_keys=True).encode()+b''.join(p.read_bytes().replace(b'\r\n',b'\n') for p in files)).hexdigest();mp=out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())['signature']!=sig:raise RuntimeError('Changed inputs: preserve outputs and use new tag')
    if args.command in ['run','smoke']:
        p=nominal_params();vehicles=dict(nominal=p,payload=replace(p,m=p.m*1.2,Iz=p.Iz*1.25),tire=replace(p,Cf=p.Cf*.75))
        def sample(n,seed):return qmc.scale(qmc.Sobol(3,scramble=True,seed=seed).random_base2(int(np.ceil(np.log2(n))))[:n],base['xi_lower'],base['xi_upper'])
        X=np.vstack([sample(cfg['calibration_count'],cfg['calibration_seed']),sample(cfg['evaluation_count'],cfg['evaluation_seed'])]);assert len(np.unique(X,axis=0))==len(X)
        save(mp,dict(signature=sig,source_hash=source_hash,config=cfg,base=base,vehicles={k:asdict(v) for k,v in vehicles.items()}));save(out/'candidates.json',X.tolist())
        cc=yaml.safe_load((ROOT/'config/controller.yaml').read_text());start=time.monotonic();done=0;total=len(X)*15
        for name,p in vehicles.items():
            print(f'Identifying {name}...',flush=True);a,b,tau,fit,u,y=identify(base,p)
            save(out/f'{name}_identification.json',dict(**fit,A=a.tolist(),B=b.tolist()));np.savez_compressed(out/f'{name}_identification.npz',u=u,x=y)
            for k,xi in enumerate(X):
                for j,d in enumerate(cfg['evaluation_demands']):
                    path=out/'cases'/f'{name}_{k:03d}_{j}.json'
                    if not path.exists():save(path,dict(vehicle=name,**evaluate(k,xi,d,p,base,cc,a,b,tau)))
                    done+=1
                print(f'{done}/{total} cases | {time.monotonic()-start:.1f}s',flush=True)
        analyze(out,cfg,base,prior['fractions'],source_records)
    else:
        if not mp.exists():raise RuntimeError('Run first')
        analyze(out,cfg,base,prior['fractions'],source_records)
        if args.command=='collect':subprocess.run(['git','add','-f','--',str(out.relative_to(ROOT.parent))],cwd=ROOT.parent,check=True)

if __name__=='__main__':main()
