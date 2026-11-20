"""Calibration-only margin replay and weight-dependence diagnosis; no simulation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from mismatch_pilot import ROOT,save
from coverage_transfer import valid
from coverage_stage2 import interpolate
from coverage_stage3 import schedule,adjust,predict,source_prior
from correction_transfer import choose,observed


def calibration_margin(rows,prior,ridge):
    """Leave all observations of one weight out together. Evaluation data never enter."""
    folds=[]
    for k in sorted({r['candidate'] for r in rows}):
        train=[r for r in rows if r['candidate']!=k]
        test=[r for r in rows if r['candidate']==k and valid(r)]
        fit=adjust(train,prior,ridge)
        for r in test:
            prediction=predict([r],prior,fit)[k]['E']
            ratio=(r['plant']['E']+r['differences']['E'])/prediction
            folds.append(dict(candidate=k,demand=r['demand'],training_episodes=len(train),
                training_candidates=sorted({x['candidate'] for x in train}),prediction_E=prediction,upper_actual_E=r['plant']['E']+r['differences']['E'],ratio=float(ratio)))
    return max([1.]+[r['ratio'] for r in folds]),folds


def load_cases(path,n):
    rows=[json.loads(p.read_text()) for p in sorted((path/'cases').glob('*.json'))]
    if len(rows)!=n or len({(r['vehicle'],r['candidate'],r['demand']) for r in rows})!=n:
        raise RuntimeError(f'Incomplete/duplicate input {path}: expected {n} cases')
    return rows


def replay(out,cfg,rs,sources,targets,budgets):
    lookup={(r['vehicle'],r['candidate'],r['demand']):r for r in rs}
    fractions=np.arange(1,31)/100.;decisions=[];margins=[];fold_rows=[];diagnosis=[]
    for target in targets:
        summary,source_ids=source_prior(sources,target)
        fleet=lambda d:interpolate(summary,d)
        for budget in budgets:
            cal=[lookup[target,k,d] for k,d in schedule(budget,cfg['training_demands'])]
            for kind,prior in [('local',lambda d:dict(E=1.,S=1.)),('fleet',fleet)]:
                fit=adjust(cal,prior,cfg['ridge']);margin,folds=calibration_margin(cal,prior,cfg['ridge'])
                margins.append(dict(target=target,budget=budget,kind=kind,multiplier=margin,fold_observations=len(folds),
                    distinct_weights=len({r['candidate'] for r in cal}),empirically_calibrated=bool(folds)))
                fold_rows.extend(dict(target=target,budget=budget,kind=kind,**f) for f in folds)
                for demand in cfg['evaluation_demands']:
                    rows=[r for r in rs if r['vehicle']==target and r['demand']==demand]
                    predictions=predict(rows,prior,fit);actual=observed(rows);by_id={r['candidate']:r for r in rows}
                    # Diagnostic only: held-out outcomes describe discrepancy, never select a margin.
                    held=[r for r in rows if r['candidate']>=cfg['calibration_count'] and valid(r)]
                    ratios=[r['plant']['E']/predictions[r['candidate']]['E'] for r in held if r['candidate'] in predictions]
                    rawratios=[r['plant']['E']/r['prediction']['E'] for r in held]
                    if ratios:
                        diagnosis.append(dict(target=target,budget=budget,kind=kind,demand=demand,n=len(ratios),
                            ratio_min=min(ratios),ratio_median=float(np.median(ratios)),ratio_max=max(ratios),
                            raw_ratio_min=min(rawratios),raw_ratio_max=max(rawratios),
                            max_integration_E=max(r['differences']['E'] for r in held),calibration_margin=margin,
                            outside_margin=sum(r>margin for r in ratios)))
                    policies={f'{kind}_original':predictions,f'{kind}_margin':{k:dict(E=y['E']*margin,S=y['S']) for k,y in predictions.items()}}
                    if kind=='local':policies['fixed_25pct']={r['candidate']:dict(E=r['prediction']['E']*cfg['fixed_margin'],S=r['prediction']['S']) for r in rows if 'prediction' in r and r['prediction']['physical']}
                    for method,ys in policies.items():
                        for frac in fractions:
                            limit=float(np.rad2deg(demand/15)*frac);selected=choose(ys,limit);oracle=choose(actual,limit)
                            status='abstains';excess=None;error=None
                            if selected is not None:
                                r=by_id[selected]
                                if 'plant' not in r:status='synthesis_failure'
                                elif not r['plant']['physical']:status='physical_failure'
                                elif not r['numerical_resolved']:status='numerical_unknown'
                                else:
                                    error=r['plant']['E'];tol=r['differences']['E'];status='boundary_unknown' if abs(error-limit)<=tol else 'passes' if error<=limit else 'tracking_violation'
                                    if status=='passes' and oracle is not None:excess=100*(r['plant']['S']/actual[oracle]['S']-1)
                            decisions.append(dict(target=target,budget=budget,method=method,demand=demand,fraction=float(frac),limit_deg_s=limit,
                                selected=selected,status=status,actual_E=error,reference=oracle,sampled_feasible=oracle is not None,steering_excess_percent=excess,
                                target_budget_used=0 if method=='fixed_25pct' else budget,source_budget=24 if method.startswith('fleet') else 0))
    df=pd.DataFrame(decisions);df.to_csv(out/'decisions.csv',index=False)
    pd.DataFrame(margins).to_csv(out/'margins.csv',index=False);save(out/'crossfit_folds.json',fold_rows)
    pd.DataFrame(diagnosis).to_csv(out/'weight_dependence.csv',index=False)
    metrics=[];gate=cfg['quality_gate']
    for (target,budget,method),g in df.groupby(['target','budget','method']):
        passed=g[g.status=='passes'];n=int(g.sampled_feasible.sum());coverage=len(passed)/n if n else 0
        violations=int((g.status=='tracking_violation').sum());other=int(g.status.isin(['synthesis_failure','physical_failure','numerical_unknown','boundary_unknown']).sum())
        mean=float(passed.steering_excess_percent.mean()) if len(passed) else None;worst=float(passed.steering_excess_percent.max()) if len(passed) else None
        ok=bool(n and violations<=gate['max_violations'] and other==0 and coverage>=gate['min_feasible_coverage'] and mean is not None and mean<=gate['mean_steering_excess_max'] and worst<=gate['max_steering_excess_max'])
        metrics.append(dict(target=target,budget=budget,method=method,passes=len(passed),violations=violations,abstains=int((g.status=='abstains').sum()),other_failures=other,sampled_feasible=n,coverage=coverage,mean_excess=mean,max_excess=worst,quality_gate=ok))
    mf=pd.DataFrame(metrics);mf.to_csv(out/'budget_metrics.csv',index=False)
    first=mf[mf.quality_gate].groupby(['target','method']).budget.min().rename('first_passing_budget');first.to_csv(out/'first_passing_budget.csv')
    html='<html><meta charset="utf-8"><title>Stage3 margin replay</title><style>body{font:15px sans-serif;max-width:1300px;margin:35px auto}td,th{padding:7px}</style><h1>Does the fleet advantage survive calibration-only margins?</h1><p>No new simulations. Both learners use the same leave-one-weight-out margin rule and original target budgets. Margin=max(1, held-out actual/predicted tracking ratios), then refit on all calibration records. Steering predictions are unchanged. At budget0 margin=1: no empirical robustness information.</p>'
    html+='<p>Same quality criterion: zero violations/other failures, ≥95% feasible coverage, mean steering excess≤2%, worst≤10%. Missing rows mean no budget passes. This is a retrospective developmental replay, not independent confirmation or a statistical coverage guarantee.</p>'+first.to_frame().to_html()
    html+='<h2>Calibration-derived multipliers</h2>'+pd.DataFrame(margins).to_html(index=False)
    for target in targets:html+=f'<h2>{target}</h2>'+mf[mf.target==target].drop(columns='target').to_html(index=False)
    html+='<p>weight_dependence.csv diagnoses held-out ratio spread at fixed demand. Variation cannot be captured exactly by one demand-only multiplicative factor. This diagnostic does NOT tune margins, and variation alone does not establish a need for a richer learner. Cross-fit folds are stored for checking that the held-out weight never enters its fitted model. The fixed25% comparator uses no target calibration. Tiny calibration sets do not support a formal guarantee.</p></html>'
    (out/'report.html').write_text(html,encoding='utf-8');print(f'Report: {out / "report.html"}',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['run','smoke','collect'])
    ap.add_argument('--stage2-path',type=Path,default=ROOT/'results/coverage_stage2_v1');ap.add_argument('--stage3-path',type=Path,default=ROOT/'results/coverage_stage3_v1');args=ap.parse_args()
    cfg=json.loads((ROOT/'config/runs/coverage_stage3.json').read_text());sources=load_cases(args.stage2_path,720);rs=load_cases(args.stage3_path,570)
    targets=['nominal','payload','tire'];budgets=cfg['budgets'];tag='coverage_stage3_margin_v1'
    if args.command=='smoke':targets=targets[:1];budgets=[0,4];tag='coverage_stage3_margin_smoke'
    out=ROOT/'results'/tag;out.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),*[ROOT/'scripts'/f for f in ['coverage_stage3.py','coverage_stage2.py','coverage_transfer.py','correction_transfer.py','mismatch_pilot.py']],ROOT/'config/runs/coverage_stage3.json']
    hashes={name:hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest() for name,rows in [('stage2',sources),('stage3',rs)]}
    sig=hashlib.sha256(json.dumps([cfg,hashes,targets,budgets],sort_keys=True).encode()+b''.join(p.read_bytes().replace(b'\r\n',b'\n') for p in files)).hexdigest();mp=out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())['signature']!=sig:raise RuntimeError('Changed inputs; preserve results and use new tag')
    save(mp,dict(signature=sig,input_hashes=hashes,config=cfg,targets=targets,budgets=budgets,rule='leave-one-weight-out maximum upper actual/predicted ratio, minimum1'))
    replay(out,cfg,rs,sources,targets,budgets)
    if args.command=='collect':subprocess.run(['git','add','-f','--',str(out.relative_to(ROOT.parent))],cwd=ROOT.parent,check=True)

if __name__=='__main__':main()
