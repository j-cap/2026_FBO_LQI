"""Two-vehicle correction-transfer diagnostic, no target-outcome access in selection."""
import argparse
import base64
from dataclasses import asdict, replace
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import yaml
from scipy.stats import qmc
from mismatch_pilot import ROOT, save, identify, evaluate
from src.ifac_bridge import nominal_params


def correction(records):
    good=[r for r in records if 'prediction' in r and r['prediction']['physical'] and
          r['plant']['physical'] and r['numerical_resolved']]
    if len(good)<3: return None
    return {m:float(np.median([r['plant'][m]/max(r['prediction'][m],1e-12) for r in good])) for m in ['E','S']}


def choose(predictions, limit):
    feasible=[(k,y) for k,y in predictions.items() if y is not None and y['E']<=limit]
    return min(feasible,key=lambda z:(z[1]['S'],z[0]))[0] if feasible else None


def predicted(records, factors):
    if factors is None:return {}
    return {r['candidate']:{m:r['prediction'][m]*factors[m] for m in ['E','S']}
            for r in records if 'prediction' in r and r['prediction']['physical']}


def observed(records):
    return {r['candidate']:{m:r['plant'][m] for m in ['E','S']}
            for r in records if 'plant' in r and r['plant']['physical'] and r['numerical_resolved']}


def analyze(out,cfg,base):
    records=[json.loads(p.read_text()) for p in (out/'cases').glob('*.json')]
    n=cfg['candidate_count']+cfg['calibration_count']; demands=base['demands_m_s2']
    expected={(v,k,d) for v in ['source','target'] for k in range(n) for d in demands}
    if {(r['vehicle'],r['candidate'],r['demand']) for r in records}!=expected or len(records)!=len(expected):
        raise RuntimeError('Missing or duplicate cases; complete run first')
    decisions=[]; factors=[]; errors=[]
    for d in demands:
        src=[r for r in records if r['vehicle']=='source' and r['demand']==d]
        tgt=[r for r in records if r['vehicle']=='target' and r['demand']==d]
        cal=lambda rs:[r for r in rs if r['candidate']<cfg['calibration_count']]
        sf=correction(cal(src));tf=correction(cal(tgt))
        factors.append(dict(demand=d,source=sf,target=tf))
        # Selection inputs: only target model predictions and explicitly budgeted calibration data.
        policies={
            'model_only':predicted(tgt,dict(E=1.,S=1.)),
            'fixed_margin_25pct':predicted(tgt,dict(E=cfg['fixed_margin'],S=1.)),
            'source_correction':predicted(tgt,sf),
            'target_correction_diagnostic':predicted(tgt,tf),
            'direct_transfer_8':observed(cal(src)),
            'source_library_72_diagnostic':observed(src)}
        actual=observed(tgt);target_by_id={r['candidate']:r for r in tgt}
        for method, predictions in policies.items():
            for frac in cfg['fractions']:
                limit=float(np.rad2deg(d/nominal_params().U)*frac)
                selected=choose(predictions,limit); oracle=choose(actual,limit)
                status='abstains';excess=None;tracking=None
                if selected is not None:
                    r=target_by_id[selected]
                    if 'synthesis_error' in r:status='synthesis_failure'
                    elif not r['plant']['physical']:status='physical_failure'
                    elif not r['numerical_resolved']:status='numerical_unknown'
                    else:
                        tracking=r['plant']['E'];margin=r['differences']['E']
                        status='boundary_unknown' if abs(tracking-limit)<=margin else 'passes' if tracking<=limit else 'tracking_violation'
                        if status=='passes' and oracle is not None:excess=100*(r['plant']['S']/actual[oracle]['S']-1)
                decisions.append(dict(demand=d,tracking_fraction=frac,limit_deg_s=limit,method=method,
                    selected=selected,status=status,actual_E=tracking,reference=oracle,
                    sampled_feasible=oracle is not None,steering_excess_percent=excess))
        for r in tgt:
            if r['candidate']<cfg['calibration_count'] or r['candidate'] not in actual:continue
            for method in ['model_only','fixed_margin_25pct','source_correction','target_correction_diagnostic']:
                if r['candidate'] not in policies[method]:continue
                for metric in ['E','S']:
                    errors.append(dict(demand=d,candidate=r['candidate'],method=method,metric=metric,
                        absolute_error=abs(policies[method][r['candidate']][metric]-actual[r['candidate']][metric])))
    df=pd.DataFrame(decisions);df.to_csv(out/'decisions.csv',index=False)
    ef=pd.DataFrame(errors);ef.to_csv(out/'heldout_prediction_errors.csv',index=False)
    save(out/'correction_factors.json',factors)
    counts=df.groupby(['method','status']).size().unstack(fill_value=0)
    counts.to_csv(out/'outcome_counts.csv')
    quality=df[df.status=='passes'].groupby('method').steering_excess_percent.agg(['median','max'])
    quality.to_csv(out/'steering_excess.csv')
    save(out/'summary.json',dict(cases=len(records),physical_failures=sum('plant' in r and not r['plant']['physical'] for r in records),
        numerical_unknown=sum('plant' in r and r['plant']['physical'] and not r['numerical_resolved'] for r in records),
        synthesis_failures=sum('synthesis_error' in r for r in records),
        calibration_rollouts_per_vehicle=cfg['calibration_count']*len(demands),
        reference_evaluation_rollouts_per_vehicle=n*len(demands)))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=list(df.method.unique());fig,axes=plt.subplots(1,3,figsize=(14,4))
    codes={'passes':0,'tracking_violation':1,'abstains':2,'numerical_unknown':3,'boundary_unknown':3,'physical_failure':4,'synthesis_failure':4}
    from matplotlib.colors import ListedColormap
    for ax,d in zip(axes,demands):
        sub=df[df.demand==d]; mat=np.array([[codes[s] for s in sub[sub.method==m].status] for m in methods])
        ax.imshow(mat,aspect='auto',vmin=0,vmax=4,cmap=ListedColormap(['#4daf4a','#e41a1c','#bbbbbb','#ffbf00','#984ea3']))
        ax.set(title=f'Demand {d:g} m/s²',xlabel='Tracking limit / peak reference [%]',yticks=range(len(methods)),yticklabels=methods)
        ax.set_xticks([0,9,19,29],[1,10,20,30])
    fig.tight_layout();buf=io.BytesIO();fig.savefig(buf,format='png',dpi=140);plt.close(fig)
    html='<html><meta charset="utf-8"><title>Correction transfer pilot</title><style>body{font:15px sans-serif;margin:35px}img{width:100%}td,th{padding:7px}</style><h1>Does a source performance correction transfer?</h1><p>Nominal source → payload target; three demands; 8 calibration weights and 64 separate evaluation weights. Noise-free developmental study. Threshold counts are correlated design requests, not success probabilities.</p>'
    html+='<p>Green: passes; red: tracking violation; gray: abstains; amber: numerical/boundary unknown; purple: physical/synthesis failure.</p><img src="data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()+'">'
    html+=counts.to_html()+'<h2>Steering excess on passing requests only [%]</h2>'+quality.to_html()
    html+='<p>Abstentions and violations must be considered alongside steering quality. Source correction and direct transfer each use 24 source plant rollouts; the dense source-library diagnostic uses 216. Target correction uses 24 target rollouts. Model-only and fixed margin use none. All methods require local identification; the large plant table is offline evaluation, not available to selection. The fixed 25% margin was motivated by the preceding pilot and is not independently tuned. Source correction is useful only if it improves target decisions beyond these simple baselines; no FL benefit is established here.</p></html>'
    (out/'report.html').write_text(html,encoding='utf-8');print(f'Report: {out / "report.html"}',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['smoke','run','analyze','collect']);args=ap.parse_args()
    cfg=json.loads((ROOT/'config/runs/correction_transfer.json').read_text());base=json.loads((ROOT/'config/runs/mismatch_pilot.json').read_text())
    if args.command=='smoke':cfg.update(tag='correction_transfer_smoke',calibration_count=4,candidate_count=4)
    out=ROOT/'results'/cfg['tag'];out.mkdir(parents=True,exist_ok=True)
    paths=[Path(__file__),ROOT/'scripts/mismatch_pilot.py',ROOT/'config/runs/correction_transfer.json',ROOT/'config/runs/mismatch_pilot.json',ROOT/'config/controller.yaml',ROOT/'src/actuator_bandwidth.py',ROOT/'src/actuator_antiwindup.py',ROOT/'src/nonlinear_plant.py',ROOT/'src/ifac/utils.py']
    sig=hashlib.sha256(json.dumps([cfg,base],sort_keys=True).encode()+b''.join(p.read_bytes().replace(b'\r\n',b'\n') for p in paths)).hexdigest()
    manifest=out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['signature']!=sig:raise RuntimeError('Inputs changed; preserve results and use a new tag')
    if args.command in ['run','smoke']:
        cc=yaml.safe_load((ROOT/'config/controller.yaml').read_text());p=nominal_params()
        vehicles=dict(source=p,target=replace(p,m=p.m*cfg['target_mass_factor'],Iz=p.Iz*cfg['target_inertia_factor']))
        def sample(n,seed):return qmc.scale(qmc.Sobol(3,scramble=True,seed=seed).random_base2(int(np.log2(n))),base['xi_lower'],base['xi_upper'])
        X=np.vstack([sample(cfg['calibration_count'],cfg['calibration_seed']),sample(cfg['candidate_count'],cfg['candidate_seed'])])
        assert len(np.unique(X,axis=0))==len(X)
        save(manifest,dict(signature=sig,config=cfg,base=base,vehicles={k:asdict(v) for k,v in vehicles.items()}));save(out/'candidates.json',X.tolist())
        start=time.monotonic();done=0;total=len(X)*len(vehicles)*len(base['demands_m_s2'])
        for name,p in vehicles.items():
            print(f'Identifying {name}...',flush=True);a,b,tau,fit,u,y=identify(base,p)
            save(out/f'{name}_identification.json',dict(**fit,A=a.tolist(),B=b.tolist()));np.savez_compressed(out/f'{name}_identification.npz',u=u,x=y)
            for k,xi in enumerate(X):
                for j,d in enumerate(base['demands_m_s2']):
                    path=out/'cases'/f'{name}_{k:03d}_{j}.json'
                    if not path.exists():save(path,dict(vehicle=name,**evaluate(k,xi,d,p,base,cc,a,b,tau)))
                    done+=1
                print(f'{done}/{total} paired cases | {time.monotonic()-start:.1f}s',flush=True)
        analyze(out,cfg,base)
    else:
        if not manifest.exists():raise RuntimeError('Run first')
        analyze(out,cfg,base)
        if args.command=='collect':subprocess.run(['git','add','-f','--',str(out.relative_to(ROOT.parent))],cwd=ROOT.parent,check=True)

if __name__=='__main__':main()
