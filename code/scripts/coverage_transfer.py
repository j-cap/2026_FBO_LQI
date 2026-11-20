"""Equal-budget complementary coverage pilot with a held-out tire-change target."""
import argparse
import base64
from dataclasses import asdict, replace
import hashlib
import io
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.stats import qmc
from mismatch_pilot import ROOT, save, identify, evaluate
from correction_transfer import choose, predicted, observed
from src.ifac_bridge import nominal_params


def valid(r):
    return 'plant' in r and r['plant']['physical'] and r['prediction']['physical'] and r['numerical_resolved']


def local_summary(records):
    """Only counts and summed log performance ratios leave this simulated client."""
    good=[r for r in records if valid(r)]
    return dict(n=len(good), sums={m:float(sum(np.log(r['plant'][m]/max(r['prediction'][m],1e-12)) for r in good)) for m in ['E','S']})


def factors(summary_by_demand, demand):
    ds=sorted(summary_by_demand)
    if any(summary_by_demand[d]['n']<3 for d in ds):return None
    if len(ds)==1:
        s=summary_by_demand[ds[0]]
        return {m:float(np.exp(s['sums'][m]/s['n'])) for m in ['E','S']}
    lo,hi=ds[0],ds[-1]
    if not lo<=demand<=hi:raise ValueError('No correction extrapolation outside coverage endpoints')
    w=(demand**2-lo**2)/(hi**2-lo**2)
    return {m:float(np.exp((1-w)*summary_by_demand[lo]['sums'][m]/summary_by_demand[lo]['n']+
                         w*summary_by_demand[hi]['sums'][m]/summary_by_demand[hi]['n'])) for m in ['E','S']}


def training_sets(rs,cfg):
    lo,hi=cfg['endpoints'];h=cfg['per_endpoint'];n=cfg['calibration_count']
    def pick(v,d,k):return [r for r in rs if r['vehicle']==v and r['demand']==d and r['candidate']<k]
    return {
        'A_low_only':{lo:pick('A',lo,n)},
        'B_high_only':{hi:pick('B',hi,n)},
        'pooled_A_low_B_high':{lo:pick('A',lo,h),hi:pick('B',hi,h)},
        'pooled_B_low_A_high':{lo:pick('B',lo,h),hi:pick('A',hi,h)},
        'A_full_coverage':{lo:pick('A',lo,h),hi:pick('A',hi,h)},
        'B_full_coverage':{lo:pick('B',lo,h),hi:pick('B',hi,h)},
        'target_local_diagnostic':{lo:pick('target',lo,h),hi:pick('target',hi,h)}}


def direct_library(groups,demand):
    """Interpolate measured source E/d and S/d; transfer weights, no target outcomes."""
    lo,hi=sorted(groups);left={r['candidate']:r for r in groups[lo] if valid(r)}
    right={r['candidate']:r for r in groups[hi] if valid(r)}
    w=(demand**2-lo**2)/(hi**2-lo**2)
    return {k:{m:float(demand*np.exp((1-w)*np.log(left[k]['plant'][m]/lo)+w*np.log(right[k]['plant'][m]/hi)))
               for m in ['E','S']} for k in left.keys() & right.keys()}


def expected_cases(cfg):
    n=cfg['calibration_count'];total=n+cfg['evaluation_count']
    return {(v,k,d) for v in ['A','B','target'] for k in range(total if v=='target' else n)
            for d in (cfg['target_demands'] if v=='target' else cfg['endpoints'])}


def analyze(out,cfg,base,fractions):
    rs=[json.loads(p.read_text()) for p in (out/'cases').glob('*.json')]
    if len(rs)!=len(expected_cases(cfg)) or {(r['vehicle'],r['candidate'],r['demand']) for r in rs}!=expected_cases(cfg):
        raise RuntimeError('Incomplete or duplicate case table')
    train=training_sets(rs,cfg)
    assert all(sum(map(len,g.values()))==cfg['calibration_count'] for g in train.values())
    summaries={name:{d:local_summary(rows) for d,rows in g.items()} for name,g in train.items()}
    save(out/'shared_summaries.json',summaries)
    decisions=[];errors=[];factor_rows=[]
    for d in cfg['target_demands']:
        target=[r for r in rs if r['vehicle']=='target' and r['demand']==d]
        by_id={r['candidate']:r for r in target};actual=observed(target)
        fs={name:factors(g,d) for name,g in summaries.items()}
        policies={name:predicted(target,f) for name,f in fs.items()}
        policies.update(model_only=predicted(target,dict(E=1.,S=1.)),fixed_margin=predicted(target,dict(E=cfg['fixed_margin'],S=1.)),
                        direct_pooled_library=direct_library(train['pooled_A_low_B_high'],d))
        for name,f in fs.items():factor_rows.append(dict(method=name,demand=d,factors=f))
        for method,pred in policies.items():
            for frac in fractions:
                limit=float(np.rad2deg(d/15)*frac);selected=choose(pred,limit);oracle=choose(actual,limit)
                status='abstains';excess=None;error=None
                if selected is not None:
                    r=by_id[selected]
                    if 'plant' not in r:status='synthesis_failure'
                    elif not r['plant']['physical']:status='physical_failure'
                    elif not r['numerical_resolved']:status='numerical_unknown'
                    else:
                        error=r['plant']['E'];tol=r['differences']['E']
                        status='boundary_unknown' if abs(error-limit)<=tol else 'passes' if error<=limit else 'tracking_violation'
                        if status=='passes' and oracle is not None:excess=100*(r['plant']['S']/actual[oracle]['S']-1)
                decisions.append(dict(method=method,demand=d,interior=d not in cfg['endpoints'],fraction=frac,
                    limit_deg_s=limit,selected=selected,status=status,actual_E=error,reference=oracle,
                    sampled_feasible=oracle is not None,steering_excess_percent=excess))
            for k,y in pred.items():
                if k<cfg['calibration_count'] or k not in actual:continue
                for metric in ['E','S']:errors.append(dict(method=method,demand=d,candidate=k,metric=metric,
                    absolute_error=abs(y[metric]-actual[k][metric])))
    df=pd.DataFrame(decisions);df.to_csv(out/'decisions.csv',index=False)
    pd.DataFrame(errors).to_csv(out/'heldout_prediction_errors.csv',index=False);save(out/'correction_factors.json',factor_rows)
    counts=df.groupby(['method','status']).size().unstack(fill_value=0);counts.to_csv(out/'outcome_counts.csv')
    interior=df[df.interior].groupby(['method','status']).size().unstack(fill_value=0);interior.to_csv(out/'interior_outcomes.csv')
    quality=df[df.status=='passes'].groupby('method').steering_excess_percent.agg(['mean','median','max']);quality.to_csv(out/'steering_excess.csv')
    save(out/'summary.json',dict(cases=len(rs),synthesis_failures=sum('plant' not in r for r in rs),
        physical_failures=sum('plant' in r and not r['plant']['physical'] for r in rs),
        numerical_unknown=sum('plant' in r and r['plant']['physical'] and not r['numerical_resolved'] for r in rs),
        calibration_budget_per_method=cfg['calibration_count'],identification_seconds_per_vehicle=base['identification_duration'],
        requests_per_method=len(fractions)*len(cfg['target_demands'])))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    methods=list(df.method.unique());fig,axs=plt.subplots(len(cfg['target_demands']),1,figsize=(12,15))
    codes={'passes':0,'tracking_violation':1,'abstains':2,'boundary_unknown':3,'numerical_unknown':3,'physical_failure':4,'synthesis_failure':4}
    for ax,d in zip(axs,cfg['target_demands']):
        sub=df[df.demand==d];mat=[[codes[s] for s in sub[sub.method==m].status] for m in methods]
        ax.imshow(mat,aspect='auto',vmin=0,vmax=4,cmap=ListedColormap(['#4daf4a','#e41a1c','#bbbbbb','#ffbf00','#984ea3']))
        ax.set(title=f'Demand {d:g} m/s²'+(' (unseen interior)' if d not in cfg['endpoints'] else ' (endpoint)'),
               yticks=range(len(methods)),yticklabels=methods,xticks=[0,9,19,29],xticklabels=[1,10,20,30],xlabel='Tracking limit / peak reference [%]')
    fig.tight_layout();buf=io.BytesIO();fig.savefig(buf,format='png',dpi=120);plt.close(fig)
    html='<html><meta charset="utf-8"><title>Complementary coverage</title><style>body{font:15px sans-serif;max-width:1200px;margin:35px auto}td,th{padding:7px}img{width:100%}</style><h1>Does complementary source coverage help?</h1><p>Two source vehicles, held-out tire-change target. Equal 24-rollout calibration budget per data-based method. Three target demands are unseen in calibration. Threshold counts are correlated diagnostic requests, not probabilities.</p>'
    html+='<h2>All requests</h2>'+counts.to_html()+'<h2>Unseen interior demands only</h2>'+interior.to_html()+'<h2>Steering excess on passing requests [%]</h2>'+quality.to_html()
    html+='<p>Green passes; red tracking violation; gray abstains; amber unresolved; purple physical/synthesis failure.</p><img src="data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()+'">'
    html+='<p>Compare pooled against restricted sources AND full-coverage single-source baselines. If pooling only beats restricted sources, evidence concerns access to complementary coverage, not inherently superior multi-vehicle learning. Coverage-role swap checks confounding by vehicle identity. No target outcomes enter zero-shot selection. Aggregation exchanges counts and summed log ratios; this is not a formal privacy guarantee. All results are exploratory.</p></html>'
    (out/'report.html').write_text(html,encoding='utf-8');print(f'Report: {out / "report.html"}',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['smoke','run','analyze','collect']);args=ap.parse_args()
    cfg=json.loads((ROOT/'config/runs/coverage_transfer.json').read_text());base=json.loads((ROOT/'config/runs/mismatch_pilot.json').read_text())
    prior=json.loads((ROOT/'config/runs/correction_transfer.json').read_text());fractions=prior['fractions']
    if args.command=='smoke':cfg.update(tag='coverage_transfer_smoke',calibration_count=6,per_endpoint=3,evaluation_count=4)
    out=ROOT/'results'/cfg['tag'];out.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),ROOT/'scripts/mismatch_pilot.py',ROOT/'scripts/correction_transfer.py',ROOT/'config/runs/coverage_transfer.json',ROOT/'config/runs/mismatch_pilot.json',ROOT/'config/runs/correction_transfer.json',ROOT/'config/controller.yaml',ROOT/'src/actuator_bandwidth.py',ROOT/'src/actuator_antiwindup.py',ROOT/'src/nonlinear_plant.py',ROOT/'src/ifac/utils.py']
    sig=hashlib.sha256(json.dumps([cfg,base,prior],sort_keys=True).encode()+b''.join(p.read_bytes().replace(b'\r\n',b'\n') for p in files)).hexdigest();mp=out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())['signature']!=sig:raise RuntimeError('Inputs changed: preserve old outputs and use new tag')
    if args.command in ['run','smoke']:
        p=nominal_params();vehicles=dict(A=p,B=replace(p,m=p.m*1.2,Iz=p.Iz*1.25),target=replace(p,Cf=p.Cf*.75))
        cc=yaml.safe_load((ROOT/'config/controller.yaml').read_text())
        def sample(n,seed):return qmc.scale(qmc.Sobol(3,scramble=True,seed=seed).random_base2(int(np.ceil(np.log2(n))))[:n],base['xi_lower'],base['xi_upper'])
        X=np.vstack([sample(cfg['calibration_count'],cfg['calibration_seed']),sample(cfg['evaluation_count'],cfg['evaluation_seed'])]);assert len(np.unique(X,axis=0))==len(X)
        save(mp,dict(signature=sig,config=cfg,base=base,vehicles={k:asdict(v) for k,v in vehicles.items()}));save(out/'candidates.json',X.tolist())
        start=time.monotonic();done=0;total=len(expected_cases(cfg))
        for name,p in vehicles.items():
            print(f'Identifying {name}...',flush=True);a,b,tau,fit,u,y=identify(base,p)
            save(out/f'{name}_identification.json',dict(**fit,A=a.tolist(),B=b.tolist()));np.savez_compressed(out/f'{name}_identification.npz',u=u,x=y)
            demands=cfg['target_demands'] if name=='target' else cfg['endpoints']
            for k,xi in enumerate(X if name=='target' else X[:cfg['calibration_count']]):
                for j,d in enumerate(demands):
                    path=out/'cases'/f'{name}_{k:03d}_{j}.json'
                    if not path.exists():save(path,dict(vehicle=name,**evaluate(k,xi,d,p,base,cc,a,b,tau)))
                    done+=1
                print(f'{done}/{total} cases | {time.monotonic()-start:.1f}s',flush=True)
        analyze(out,cfg,base,fractions)
    else:
        if not mp.exists():raise RuntimeError('Run first')
        analyze(out,cfg,base,fractions)
        if args.command=='collect':subprocess.run(['git','add','-f','--',str(out.relative_to(ROOT.parent))],cwd=ROOT.parent,check=True)

if __name__=='__main__':main()
