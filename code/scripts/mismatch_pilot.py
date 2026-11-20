"""Small deterministic opportunity diagnostic. No learning or fleet claim."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import least_squares
from scipy.stats import qmc
from src.ifac_bridge import nominal_params, build_bicycle_beta_r, lc_yaw_rate_ref
from src.actuator_bandwidth import augment_discrete, design, step
from src.actuator_antiwindup import integral_update


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def identify(cfg, p):
    """Noise-free output-error fit of Cf, Cr, tau; geometry/inertia known locally."""
    ts = cfg['ts']
    t = np.arange(0, cfg['identification_duration'], ts)
    u = np.deg2rad(cfg['identification_amplitude_deg']) * (
        np.sin(2*np.pi*.23*t) + .5*np.sin(2*np.pi*.71*t) + .25*np.sin(2*np.pi*1.3*t))/1.75
    def truth(n):
        x = np.zeros(3); hist = [x.copy()]
        for uk in u:
            for _ in range(n): x = step(x, uk, p, cfg['tau_true'], ts/n, 1.)
            hist.append(x.copy())
        return np.array(hist)
    y = truth(20); finer = truth(40)
    if np.max(abs(y-finer)) > 1e-6: raise RuntimeError('Identification integration unresolved')
    scales = np.maximum(np.std(y, axis=0), 1e-5)
    def predict(z):
        cf, cr, tau = np.exp(z)
        a, b, _, _ = build_bicycle_beta_r(replace(p, Cf=cf, Cr=cr))
        ad, bd = augment_discrete(a, b, tau, ts)
        x = np.zeros(3); hist = [x.copy()]
        for uk in u:
            x = ad@x + bd[:, 0]*uk; hist.append(x.copy())
        return np.array(hist)
    fit = least_squares(lambda z: ((predict(z)-y)/scales).ravel(),
        np.log([60000., 60000., .3]), bounds=(np.log([10000.,10000.,.03]), np.log([200000.,200000.,1.])))
    if not fit.success: raise RuntimeError(fit.message)
    cf, cr, tau = np.exp(fit.x)
    a, b, _, _ = build_bicycle_beta_r(replace(p, Cf=cf, Cr=cr))
    return a, b, float(tau), dict(Cf=float(cf), Cr=float(cr), tau=float(tau),
        normalized_fit_rmse=float(np.sqrt(np.mean(((predict(fit.x)-y)/scales)**2))),
        observed_peak_yaw_equivalent_ay=float(np.max(abs(y[:,1]))*p.U),
        integration_max_difference=float(np.max(abs(y-finer)))), u, y


def rollout(p, cfg, cc, a, b, tau, K, kr, ref, nonlinear, n):
    ts = cfg['ts']; x = np.zeros(3); eta = 0.; err = []; rate2 = []
    peaks = np.zeros(3); saturated = 0
    ad, bd = augment_discrete(a, b, tau, ts/n)
    actual_tau = cfg['tau_true'] if nonlinear else tau
    for rr in ref:
        raw = float(-K[:3]@x-K[3]*eta+kr*rr)
        u = float(np.clip(raw, cc['u_min'], cc['u_max']))
        err.append((x[1]-rr)**2)
        eta = integral_update(eta, x[1]-rr, raw, u, K[3], ts)
        saturated += abs(raw-u)>1e-10
        # Exact integral of squared first-order actuator rate over one held-input interval.
        rate2.append((u-x[2])**2/(2*actual_tau*ts)*(1-np.exp(-2*ts/actual_tau)))
        peaks[2] = max(peaks[2], abs((u-x[2])/actual_tau))
        for _ in range(n):
            x = step(x,u,p,actual_tau,ts/n,1.) if nonlinear else ad@x+bd[:,0]*u
            peaks[:2] = np.maximum(peaks[:2], abs(x[[0,2]]))
            if not np.isfinite(x).all() or peaks[0]>np.deg2rad(cc['beta_max_deg']):
                return dict(physical=False, E=None, S=None, reason='sideslip_or_divergence')
    physical = bool(peaks[1] <= max(abs(cc['u_min']),abs(cc['u_max']))+1e-8 and peaks[2]<=cc['u_rate_max']/ts)
    return dict(physical=physical, E=float(np.rad2deg(np.sqrt(np.mean(err)))),
        S=float(np.rad2deg(np.sqrt(np.mean(rate2)))), reason='' if physical else 'actuator_limit',
        beta_peak_deg=float(np.rad2deg(peaks[0])), delta_peak_deg=float(np.rad2deg(peaks[1])),
        rate_peak_deg_s=float(np.rad2deg(peaks[2])), saturation_fraction=float(saturated/len(ref)))


def evaluate(k, xi, demand, p, cfg, cc, a, b, tau):
    ad, bd = augment_discrete(a,b,tau,cfg['ts'])
    base = dict(candidate=k, demand=demand, xi=list(map(float,xi)))
    try: K, kr, radius = design(ad,bd,xi,cc['state_scale'],cfg['ts'],cc['fixed_R'])
    except (ValueError, np.linalg.LinAlgError) as e: return dict(**base, synthesis_error=str(e))
    _, ref = lc_yaw_rate_ref(cfg['ts'], cfg['duration'], 1., cfg['maneuver_half_duration'], demand/p.U, tfilter=.1)
    # Normalize filtered shape to the specified peak yaw-equivalent acceleration demand.
    ref *= (demand/p.U)/max(abs(ref))
    pred = rollout(p,cfg,cc,a,b,tau,K,kr,ref,False,10)
    previous = None; resolved = False; differences = None
    for n in cfg['substeps']:
        actual = rollout(p,cfg,cc,a,b,tau,K,kr,ref,True,n)
        if previous is not None:
            if actual['physical'] and previous['physical']:
                differences = {m:abs(actual[m]-previous[m]) for m in ['E','S']}
                resolved = all(differences[m] <= 1e-5+cfg['relative_tolerance']*max(abs(actual[m]),1e-3) for m in ['E','S'])
            # Early physical exits have incomplete metrics, never claim numerical convergence.
            if resolved: break
        previous = actual
    return dict(**base, prediction=pred, plant=actual, numerical_resolved=resolved,
        differences=differences, substeps=n, gains=K.tolist(), kr=kr, radius=radius)


def analyze(out, cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = [json.loads(p.read_text()) for p in sorted((out/'cases').glob('*.json'))]
    expected=cfg['candidate_count']*len(cfg['demands_m_s2'])
    if len(records)!=expected: raise RuntimeError(f'Incomplete run: {len(records)}/{expected}')
    rows=[]; decisions=[]
    for r in records:
        if 'synthesis_error' in r: continue
        rows.append(dict(candidate=r['candidate'], demand=r['demand'],
            pred_E=r['prediction']['E'],pred_S=r['prediction']['S'],pred_ok=r['prediction']['physical'],
            plant_E=r['plant']['E'],plant_S=r['plant']['S'],plant_ok=r['plant']['physical'],
            resolved=r['numerical_resolved']))
    df=pd.DataFrame(rows); df.to_csv(out/'metrics.csv',index=False)
    fig, axes=plt.subplots(1,len(cfg['demands_m_s2']),figsize=(13,4),squeeze=False)
    for ax,demand in zip(axes[0],cfg['demands_m_s2']):
        sub=df[df.demand==demand]
        usable=sub[sub.plant_ok & sub.resolved]
        ax.scatter(sub.pred_E,sub.pred_S,label='Linear prediction',s=15)
        ax.scatter(usable.plant_E,usable.plant_S,label='Nonlinear plant',s=15)
        for frac in cfg['tracking_fractions']:
            limit=float(np.rad2deg(demand/nominal_params().U)*frac)
            pred=sub[sub.pred_ok & (sub.pred_E<=limit)]
            oracle=usable[usable.plant_E<=limit]
            chosen=None if pred.empty else pred.sort_values(['pred_S','candidate']).iloc[0]
            best=None if oracle.empty else oracle.sort_values(['plant_S','candidate']).iloc[0]
            status='no_predicted_solution'
            if chosen is not None:
                status=('physical_failure' if not chosen.plant_ok else 'numerical_unknown' if not chosen.resolved else
                    'tracking_violation' if chosen.plant_E>limit else 'passes')
            decisions.append(dict(demand=demand,tracking_fraction=frac,limit_deg_s=limit,
                selected=None if chosen is None else int(chosen.candidate),status=status,
                selected_actual_E=None if chosen is None or pd.isna(chosen.plant_E) else float(chosen.plant_E),
                reference=None if best is None else int(best.candidate),
                steering_excess_percent=None if status!='passes' or best is None else float(100*(chosen.plant_S/best.plant_S-1)),
                feasible_candidates=len(oracle)))
        ax.set(title=f'Demand {demand:g} m/s²',xlabel='Yaw RMSE [deg/s]',ylabel='Steering-rate RMS [deg/s]'); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); buf=io.BytesIO(); fig.savefig(buf,format='png',dpi=140); plt.close(fig)
    pd.DataFrame(decisions).to_csv(out/'decisions.csv',index=False)
    save(out/'summary.json',dict(total_cases=len(records),synthesis_failures=len(records)-len(rows),
        physical_failures=int((~df.plant_ok).sum()),numerical_unknown=int((df.plant_ok & ~df.resolved).sum()),decisions=decisions))
    html='<html><meta charset="utf-8"><title>Model mismatch pilot</title><style>body{font:16px sans-serif;max-width:1200px;margin:40px auto}td,th{padding:8px;text-align:left}img{width:100%}</style><h1>Does model mismatch change weight selection?</h1><p>One nominal vehicle; noise-free mechanism diagnostic. Same identified-model gains on both systems. Reference is best among 64 sampled weights, not a global optimum. Limits are exploratory fractions of reference amplitude, not engineering standards.</p>'
    html+='<img src="data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()+'">'
    html+=pd.DataFrame(decisions).to_html(index=False)+'<p>Promising: model-selected designs violate a requirement despite a feasible sampled alternative, or have meaningful steering excess. No feasible alternative means a task/controller/search limitation, not evidence for learning. This does not establish fleet benefit.</p></html>'
    (out/'report.html').write_text(html,encoding='utf-8')
    print(f'Report: {out / "report.html"}',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['run','smoke','analyze','collect']);args=ap.parse_args()
    cfg=json.loads((ROOT/'config/runs/mismatch_pilot.json').read_text())
    cc=yaml.safe_load((ROOT/'config/controller.yaml').read_text());p=nominal_params()
    if args.command=='smoke':cfg.update(candidate_count=4,tag='mismatch_pilot_smoke')
    out=ROOT/'results'/cfg['tag'];out.mkdir(parents=True,exist_ok=True)
    paths=[Path(__file__),ROOT/'config/runs/mismatch_pilot.json',ROOT/'config/controller.yaml',
           ROOT/'src/actuator_bandwidth.py',ROOT/'src/actuator_antiwindup.py',ROOT/'src/nonlinear_plant.py',ROOT/'src/ifac/utils.py']
    signature=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()+b''.join(x.read_bytes().replace(b'\r\n',b'\n') for x in paths)).hexdigest()
    manifest=out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['signature']!=signature:
        raise RuntimeError('Inputs changed: preserve existing outputs and use a new run tag')
    if args.command in ['run','smoke']:
        started=time.monotonic(); print('Identifying local linear model (20-second gentle multisine)...',flush=True)
        a,b,tau,fit,u,y=identify(cfg,p)
        save(manifest,dict(signature=signature,config=cfg,plant=asdict(p),identification=fit,A=a.tolist(),B=b.tolist()))
        np.savez_compressed(out/'identification.npz',u=u,x=y)
        X=qmc.scale(qmc.Sobol(3,scramble=True,seed=cfg['sobol_seed']).random_base2(int(np.log2(cfg['candidate_count']))),cfg['xi_lower'],cfg['xi_upper'])
        save(out/'candidates.json',X.tolist())
        total=len(X)*len(cfg['demands_m_s2']); done=0
        for k,xi in enumerate(X):
            for j,demand in enumerate(cfg['demands_m_s2']):
                path=out/'cases'/f'{k:03d}_{j}.json'
                if not path.exists():save(path,evaluate(k,xi,demand,p,cfg,cc,a,b,tau))
                done+=1
            print(f'{done}/{total} cases | elapsed {time.monotonic()-started:.1f}s',flush=True)
        analyze(out,cfg)
    else:
        if not manifest.exists():raise RuntimeError('Run first')
        analyze(out,cfg)
        if args.command=='collect':
            subprocess.run(['git','add','-f','--',str(out.relative_to(ROOT.parent))],cwd=ROOT.parent,check=True)
            print('Staged this pilot only. Commit and push next.',flush=True)

if __name__=='__main__':main()
