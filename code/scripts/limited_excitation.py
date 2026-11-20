"""Gate 1: limited-excitation simulation, informed by test-standard scopes.

No standard compliance, fleet algorithm, deployment safety, or real-world sensor
specification is claimed. Isolated from previous evaluators and result directories.
"""
import argparse
import base64
from dataclasses import asdict
import hashlib
import io
import itertools
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
from scipy.special import ndtr
from scipy.stats import qmc
from src.ifac_bridge import nominal_params, build_bicycle_beta_r, lc_yaw_rate_ref
from src.actuator_bandwidth import augment_discrete, design, step
from src.actuator_antiwindup import integral_update
from src.nonlinear_plant import nonlinear_bicycle_derivatives
from mismatch_pilot import save


def actual_ay(x, p, mu):
    return p.U * (nonlinear_bicycle_derivatives(x[:2], x[2], p, mu=mu)[0] + x[1])


def measurement(x, p, mu):
    # No direct sideslip measurement in the information diagnostic.
    return np.array([np.rad2deg(x[1]), actual_ay(x, p, mu), np.rad2deg(x[2])])


def whiten_ar1(z, sigma, rho):
    """Whiten stationary AR(1) errors; sigma denotes marginal channel SD."""
    z = np.asarray(z, float) / np.asarray(sigma)
    return np.vstack([z[:1], (z[1:] - rho*z[:-1])/np.sqrt(1-rho*rho)])


def separation(y1, y2, sigma, rho):
    """Exact two-simple-hypothesis Gaussian distance, no estimated nuisance terms."""
    distance = float(np.linalg.norm(whiten_ar1(y1-y2, sigma, rho)))
    return dict(mahalanobis_distance=distance, expected_log_likelihood_ratio=distance**2/2,
                ideal_equal_prior_error=float(ndtr(-distance/2)))


def excitation_input(cfg, p, cap):
    # Common input selected from the known tangent model only, never plant outputs.
    ts = cfg['ts']; t = np.arange(0, cfg['excitation_duration'], ts)
    shape = (np.sin(2*np.pi*.23*t) + .5*np.sin(2*np.pi*.71*t)
             + .25*np.sin(2*np.pi*1.3*t))/1.75
    a, b, _, _ = build_bicycle_beta_r(p)
    ad, bd = augment_discrete(a, b, cfg['tau_true'], ts/20)
    x = np.zeros(3); peak = 0.
    for uk in shape:
        for _ in range(20):
            x = ad@x + bd[:, 0]*uk
            peak = max(peak, abs(p.U*((a@x[:2]+b[:, 0]*x[2])[0]+x[1])))
    # Bound command angle too. Rate/actual acceleration are monitored on each plant.
    amplitude = min(cfg['excitation_headroom']*cap/max(peak, 1e-12),
                    np.deg2rad(cfg['excitation_wheel_angle_cap_deg'])/max(abs(shape)))
    return amplitude*shape


def excitation_rollout(u, p, cfg, mu, cap, n):
    x = np.zeros(3); y = []; peak_ay = peak_angle = peak_rate = 0.
    for uk in u:
        y.append(measurement(x, p, mu))
        peak_rate = max(peak_rate, abs(np.rad2deg((uk-x[2])/cfg['tau_true'])))
        for _ in range(n):
            x = step(x, uk, p, cfg['tau_true'], cfg['ts']/n, mu)
            peak_ay = max(peak_ay, abs(actual_ay(x, p, mu)))
            peak_angle = max(peak_angle, abs(np.rad2deg(x[2])))
            if (not np.isfinite(x).all() or abs(x[0])>np.deg2rad(6.) or
                peak_ay > cap or peak_angle > cfg['excitation_wheel_angle_cap_deg'] or
                peak_rate > cfg['excitation_wheel_rate_cap_deg_s']):
                return dict(complete=False, peak_ay=peak_ay, peak_wheel_deg=peak_angle,
                            peak_rate_deg_s=peak_rate, samples=len(y)), np.asarray(y)
    return dict(complete=True, peak_ay=peak_ay, peak_wheel_deg=peak_angle,
                peak_rate_deg_s=peak_rate, samples=len(y)), np.asarray(y)


def characterize(out, cfg, p):
    """Archive exact mean trajectories, compare with stipulated noisy observation model.

    Truth and integration comparisons are offline diagnostics, not calibration data.
    No random noise seed is needed: pairwise Gaussian overlap is evaluated analytically.
    """
    sigma = list(cfg['measurement_sigma'].values())
    rows = []; pairs = []
    for ci, cap in enumerate(cfg['excitation_caps_m_s2']):
        u = excitation_input(cfg, p, cap); means = {}; metadata = {}
        for vi, mu in enumerate(cfg['mu_variants']):
            coarse, yc = excitation_rollout(u, p, cfg, mu, cap, 20)
            fine, yf = excitation_rollout(u, p, cfg, mu, cap, 40)
            resolved = coarse['complete'] and fine['complete']
            diff = None
            if resolved:
                diff = float(np.max(np.abs(yc-yf)/sigma))
                resolved = diff <= 0.001
            row = dict(cap=cap, mu=mu, **fine, integration_resolved=bool(resolved),
                       integration_difference_in_sigma=diff,
                       command_peak_deg=float(np.rad2deg(max(abs(u)))))
            rows.append(row); metadata[vi] = row
            np.savez_compressed(out/f'excitation_{ci}_{vi}.npz', u=u, mean_measurements=yf)
            if resolved: means[vi] = yf
            print(f'Excitation cap={cap:g}, mu={mu:g}: complete={fine["complete"]}, '
                  f'peak ay={fine["peak_ay"]:.3f}, resolved={resolved}', flush=True)
        for i, j in itertools.combinations(range(len(cfg['mu_variants'])), 2):
            row = dict(cap=cap, mu_a=cfg['mu_variants'][i], mu_b=cfg['mu_variants'][j],
                       usable=i in means and j in means)
            if row['usable']:
                row.update(separation(means[i], means[j], sigma, cfg['measurement_ar1_rho']))
                row['maximum_channel_difference_in_sigma'] = float(np.max(abs(means[i]-means[j])/sigma))
            pairs.append(row)
    save(out/'excitation.json', dict(envelopes=rows, pairs=pairs))


def controller_rollout(p, cfg, cc, K, kr, ref, mu, n):
    ts=cfg['ts']; x=np.zeros(3); eta=0.; errors=[]; rate2=[]
    peaks=np.zeros(4); saturated=0
    for rr in ref:
        raw=float(-K[:3]@x-K[3]*eta+kr*rr)
        u=float(np.clip(raw, cc['u_min'], cc['u_max']))
        errors.append((x[1]-rr)**2)
        eta=integral_update(eta,x[1]-rr,raw,u,K[3],ts)
        saturated += abs(raw-u)>1e-10
        tau=cfg['tau_true']
        rate2.append((u-x[2])**2/(2*tau*ts)*(1-np.exp(-2*ts/tau)))
        peaks[2]=max(peaks[2],abs((u-x[2])/tau))
        for _ in range(n):
            x=step(x,u,p,tau,ts/n,mu)
            peaks[:2]=np.maximum(peaks[:2],abs(x[[0,2]]))
            peaks[3]=max(peaks[3],abs(actual_ay(x,p,mu)))
            if not np.isfinite(x).all() or peaks[0]>np.deg2rad(cc['beta_max_deg']):
                return dict(physical=False,E=None,S=None,reason='sideslip_or_divergence')
    physical=bool(peaks[1]<=max(abs(cc['u_min']),abs(cc['u_max']))+1e-8 and peaks[2]<=cc['u_rate_max']/ts)
    return dict(physical=physical,E=float(np.rad2deg(np.sqrt(np.mean(errors)))),
                S=float(np.rad2deg(np.sqrt(np.mean(rate2)))),reason='' if physical else 'actuator_limit',
                peak_ay=float(peaks[3]),peak_beta_deg=float(np.rad2deg(peaks[0])),
                peak_wheel_deg=float(np.rad2deg(peaks[1])),peak_rate_deg_s=float(np.rad2deg(peaks[2])),
                saturation_fraction=float(saturated/len(ref)))


def evaluate(k, xi, demand, mu, p, cfg, cc):
    base=dict(candidate=k,xi=list(map(float,xi)),demand=demand,mu=mu)
    a,b,_,_=build_bicycle_beta_r(p)
    ad,bd=augment_discrete(a,b,cfg['tau_true'],cfg['ts'])
    try: K,kr,radius=design(ad,bd,xi,cc['state_scale'],cfg['ts'],cc['fixed_R'])
    except (ValueError,np.linalg.LinAlgError) as e: return dict(**base,synthesis_error=str(e))
    _,ref=lc_yaw_rate_ref(cfg['ts'],cfg['duration'],1.,cfg['maneuver_half_duration'],demand/p.U,tfilter=.1)
    ref *= (demand/p.U)/max(abs(ref))
    prev=None; resolved=False; diff=None
    for n in cfg['substeps']:
        actual=controller_rollout(p,cfg,cc,K,kr,ref,mu,n)
        if prev is not None and actual['physical'] and prev['physical']:
            diff={key:abs(actual[key]-prev[key]) for key in ['E','S']}
            resolved=all(diff[key]<=1e-5+cfg['relative_tolerance']*max(abs(actual[key]),1e-3) for key in ['E','S'])
        if resolved: break
        prev=actual
    return dict(**base,plant=actual,numerical_resolved=resolved,differences=diff,
                substeps=n,gains=K.tolist(),kr=kr,radius=radius)


def acceptable_sets(frame, mus, limit, excess):
    """Finite-pool hindsight sets; unresolved points excluded, never called infeasible."""
    sets={}; feasible={}; best={}
    for mu in mus:
        sub=frame[(frame.mu==mu)&frame.resolved&frame.physical&(frame.E<=limit)].copy()
        feasible[mu]=set(sub.candidate.astype(int))
        best[mu]=None if sub.empty else sub.sort_values(['S','candidate']).iloc[0]
        sets[mu]=set() if sub.empty else set(sub[sub.S<=sub.S.min()*(1+excess/100)].candidate.astype(int))
    return sets,feasible,best


def analyze(out,cfg):
    X=json.loads((out/'candidates.json').read_text())
    expected={(vi,k,di) for vi in range(len(cfg['mu_variants'])) for k in range(len(X)) for di in range(len(cfg['demands_m_s2']))}
    files=list((out/'cases').glob('*.json'))
    if {tuple(map(int,p.stem.split('_'))) for p in files}!=expected:
        raise RuntimeError('Incomplete or unexpected case files; run first')
    rows=[]
    for path in sorted(files):
        r=json.loads(path.read_text());v=r.get('plant',{})
        rows.append(dict(mu=r['mu'],candidate=r['candidate'],demand=r['demand'],
                         E=v.get('E'),S=v.get('S'),physical=v.get('physical',False),
                         resolved=r.get('numerical_resolved',False),
                         synthesis_failure='synthesis_error' in r,reason=v.get('reason','synthesis')))
    df=pd.DataFrame(rows);df.to_csv(out/'metrics.csv',index=False)
    requests=[]; overlaps=[]; transfers=[]; fixed=[]
    fractions=sorted(set(cfg['tracking_fractions']+list(np.round(np.arange(.01,.301,cfg['diagnostic_fraction_step']),8))))
    for demand,group in df.groupby('demand'):
        for fraction in fractions:
            limit=float(np.rad2deg(demand/nominal_params().U)*fraction)
            aset,fset,best=acceptable_sets(group,cfg['mu_variants'],limit,cfg['acceptable_excess_percent'])
            shared=set.intersection(*aset.values());feasible_shared=set.intersection(*fset.values())
            all_have=all(b is not None for b in best.values())
            common=None; worst=None
            if feasible_shared:
                candidates=[]
                for k in feasible_shared:
                    ss=group[group.candidate==k].set_index('mu').S
                    gap=max(100*(ss[mu]/best[mu].S-1) for mu in cfg['mu_variants'])
                    candidates.append((gap,k))
                worst,common=min(candidates)
            tag=dict(demand=float(demand),fraction=float(fraction),limit_deg_s=limit,
                     primary=any(abs(fraction-f)<1e-8 for f in cfg['tracking_fractions']))
            requests.append(dict(**tag,each_plant_has_sampled_solution=all_have,
                                 shared_feasible_count=len(feasible_shared),shared_acceptable_count=len(shared),
                                 common_oracle_candidate=common,common_oracle_worst_excess=worst,
                                 physical_failure_count=int((~group.physical & ~group.synthesis_failure).sum()),
                                 numerical_unknown_count=int((group.physical & ~group.resolved).sum())))
            for ma,mb in itertools.combinations(cfg['mu_variants'],2):
                union=aset[ma]|aset[mb]
                overlaps.append(dict(**tag,mu_a=ma,mu_b=mb,n_a=len(aset[ma]),n_b=len(aset[mb]),
                                     intersection=len(aset[ma]&aset[mb]),jaccard=None if not union else len(aset[ma]&aset[mb])/len(union)))
            for ma,mb in itertools.permutations(cfg['mu_variants'],2):
                donor=best[ma]
                result=dict(**tag,source_mu=ma,target_mu=mb,status='no_source_solution')
                if donor is not None:
                    target=group[(group.mu==mb)&(group.candidate==int(donor.candidate))].iloc[0]
                    result['candidate']=int(donor.candidate)
                    result['status']=('synthesis_failure' if target.synthesis_failure else 'physical_failure' if not target.physical
                                      else 'numerical_unknown' if not target.resolved else 'tracking_violation' if target.E>limit else 'passes')
                    if result['status']=='passes' and best[mb] is not None:
                        result['excess_percent']=float(100*(target.S/best[mb].S-1))
                transfers.append(result)
            for i,name in enumerate(cfg['fixed_weights']):
                k=cfg['candidate_count']+i
                for mu in cfg['mu_variants']:
                    r=group[(group.mu==mu)&(group.candidate==k)].iloc[0]
                    status=('synthesis_failure' if r.synthesis_failure else 'physical_failure' if not r.physical else
                            'numerical_unknown' if not r.resolved else 'tracking_violation' if r.E>limit else 'passes')
                    fixed.append(dict(**tag,name=name,mu=mu,status=status,
                                      excess_percent=None if status!='passes' else float(100*(r.S/best[mu].S-1))))
    q=pd.DataFrame(requests);q.to_csv(out/'requests.csv',index=False)
    pd.DataFrame(overlaps).to_csv(out/'overlap.csv',index=False)
    pd.DataFrame(transfers).to_csv(out/'direct_transfer_oracle.csv',index=False)
    pd.DataFrame(fixed).to_csv(out/'fixed_weights.csv',index=False)
    ex=json.loads((out/'excitation.json').read_text())
    pairs=pd.DataFrame(ex['pairs']);pairs.to_csv(out/'distinguishability.csv',index=False)
    pd.DataFrame(ex['envelopes']).to_csv(out/'excitation_envelopes.csv',index=False)
    primary=q[q.primary]; eligible=primary[primary.each_plant_has_sampled_solution]
    summary=dict(cases=len(df),synthesis_failures=int(df.synthesis_failure.sum()),
                 physical_failures=int((~df.physical & ~df.synthesis_failure).sum()),
                 numerical_unknown=int((df.physical & ~df.resolved).sum()),
                 primary_requests=len(primary),primary_all_plants_sampled_feasible=len(eligible),
                 primary_shared_acceptable=int((eligible.shared_acceptable_count>0).sum()),
                 primary_no_shared_acceptable=int((eligible.shared_acceptable_count==0).sum()))
    save(out/'summary.json',summary)
    report(out,cfg,df,q,pairs,summary)


def report(out,cfg,df,requests,pairs,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots=[]
    def embed(fig):
        fig.tight_layout();buf=io.BytesIO();fig.savefig(buf,format='png',dpi=140);plt.close(fig)
        return '<img src="data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()+'">'
    fig,axes=plt.subplots(1,len(cfg['demands_m_s2']),figsize=(12,3.8),squeeze=False)
    for ax,d in zip(axes[0],cfg['demands_m_s2']):
        for mu in cfg['mu_variants']:
            z=df[(df.demand==d)&(df.mu==mu)&df.resolved&df.physical]
            ax.scatter(z.E,z.S,s=16,label=f'mu={mu:g}')
        ax.set(title=f'd={d:g} m/s²',xlabel='True yaw RMSE [deg/s]',ylabel='Steering-rate RMS [deg/s]');ax.legend()
    plots.append(embed(fig))
    fig,axes=plt.subplots(1,2,figsize=(11,3.7))
    if 'ideal_equal_prior_error' in pairs:
        for (a,b),z in pairs[pairs.usable].groupby(['mu_a','mu_b']):
            axes[0].plot(z.cap,z.ideal_equal_prior_error,'o-',label=f'{a:g} vs {b:g}')
        axes[0].legend()
    axes[0].set(xlabel='Permitted actual ay cap [m/s²]',ylabel='Ideal binary decision error',ylim=(-.01,.51))
    for demand,z in requests.groupby('demand'):
        axes[1].plot(z.fraction*100,z.shared_acceptable_count,'o-',markersize=3,label=f'd={demand:g}')
    axes[1].set(xlabel='Tracking limit [% reference peak]',ylabel='Shared acceptable sampled weights');axes[1].legend()
    plots.append(embed(fig))
    text='''<h1>Limited-excitation simulation study — Gate 1</h1>
<p><b>Research assumptions informed by vehicle-test standard scopes; no ISO compliance or industrial commissioning claim.</b></p>
<p>Question: do limited observations leave decision-relevant uncertainty? All variants have identical small-slip stiffness, mass, geometry and actuator. Only effective tire saturation coefficient varies. LQI synthesis is given the exact common tangent model and actuator: an optimistic isolation of nonlinear uncertainty, not an identification result.</p>
<h2>1. Does the choice of weights matter?</h2><p>Full-state, noise-free closed-loop evaluation is an offline diagnostic. Same candidate pool and gains across variants. Reference is best sampled feasible weight, not a global optimum.</p>'''
    text+=plots[0]+'<h2>2. Is the uncertainty visible in permitted measurements?</h2>'+plots[1]
    text+='''<p>Observation channels: yaw rate, actual lateral acceleration and actual wheel angle. Stationary Gaussian AR(1) measurement noise is stipulated, with known marginal standard deviations and correlation. Pairwise error is Phi(-D/2), where D is the whitened distance between exact mean trajectories, under equal priors and two known plant hypotheses. This is an optimistic local diagnostic with known nuisance parameters. It is not the error of a fitted estimator, a three-class error, or a sensor guarantee. Near-zero error means low excitation alone does not hide these variants under this noise model. Large error does not imply different weights are required.</p>
<p>Actual ay/angle/rate limits are monitored at integration substeps; aborted tests are excluded from the separation calculation and reported. Input amplitudes are determined from the common tangent model, never scaled using target nonlinear outcomes. The cap is a permission envelope, not a demand to reach its boundary.</p>
<h2>Primary requests</h2><p>Three fixed tracking fractions (10%, 20%, 30% of reference peak). Acceptable weights satisfy physical/tracking constraints and lie within 10% of each plant's best sampled steering RMS. These are inherited-style developmental choices, not external requirements. The full 1–30% grid is descriptive only; do not select a favorable threshold after inspection.</p>'''
    text+=requests[requests.primary].to_html(index=False)
    text+='<h2>Observation separation</h2>'+pairs.to_html(index=False)
    text+='<h2>Completion and failure counts</h2><pre>'+json.dumps(summary,indent=2)+'</pre>'
    text+='''<h2>Interpretation and stopping rule</h2><ul>
<li>Large acceptable-set overlap: common weights remain a strong answer, even with nonlinear mismatch.</li>
<li>Different choices but easily distinguished plants: allow a strong local estimator before claiming fleet value.</li>
<li>Different choices and poorly distinguished plants: a candidate information gap; still needs a practical restriction and an observable source–target relationship.</li>
<li>No sampled feasible solution: distinguish controller/task/candidate-pool limitations from transfer failure. Empty intersections in a small pool are not proof of disjoint continuous sets.</li></ul>
<p>The hindsight common controller and donor-specific best weights use evaluation truth and are diagnostics, not deployable baselines. No fleet learner, source budget, trained local estimator, or calibrated reliability bound is implemented in Gate 1. No new noise enters controller feedback; the observation-noise model applies only to the characterization information diagnostic.</p>
<h2>Sources and protocol</h2><p>ISO scopes: <a href="https://www.iso.org/standard/70164.html">15037-1</a> (general conditions), <a href="https://www.iso.org/standard/54144.html">7401</a> (transient response), <a href="https://www.iso.org/standard/81710.html">4138</a> (steady-state response). The multisine, amplitudes, noise, duration and requirements here are our own research choices. See docs/limited_excitation.md and manifest.json.</p>'''
    (out/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Limited excitation Gate 1</title><style>body{font:16px/1.6 system-ui;max-width:1150px;margin:35px auto;padding:15px;color:#183047}img{max-width:100%}table{font-size:12px;display:block;overflow:auto;border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd}h2{margin-top:35px}</style>'+text+'</html>',encoding='utf-8')
    print(f'Report: {out / "report.html"}',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['smoke','run','analyze','collect']);args=ap.parse_args()
    cfg=json.loads((ROOT/'config/runs/limited_excitation.json').read_text())
    if args.command=='smoke':cfg.update(candidate_count=4,tag='limited_excitation_smoke')
    cc=yaml.safe_load((ROOT/'config/controller.yaml').read_text());p=nominal_params()
    out=ROOT/'results'/cfg['tag'];out.mkdir(parents=True,exist_ok=True)
    dependencies=[Path(__file__),ROOT/'scripts/mismatch_pilot.py',ROOT/'config/runs/limited_excitation.json',ROOT/'config/controller.yaml',
                  ROOT/'src/actuator_bandwidth.py',ROOT/'src/actuator_antiwindup.py',ROOT/'src/nonlinear_plant.py',ROOT/'src/ifac_bridge.py',ROOT/'src/ifac/utils.py']
    hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for f in dependencies}
    signature=hashlib.sha256(json.dumps(dict(config=cfg,files=hashes),sort_keys=True).encode()).hexdigest()
    manifest=out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['signature']!=signature:
        raise RuntimeError('Inputs changed; preserve existing outputs and use a new run tag')
    if args.command in ['smoke','run']:
        save(manifest,dict(signature=signature,config=cfg,files=hashes,plant=asdict(p),
                           tangent_model='known exactly; no identification claim',
                           protocol='limited-excitation simulation; no standard compliance'))
        X=qmc.scale(qmc.Sobol(3,scramble=True,seed=cfg['sobol_seed']).random_base2(int(np.log2(cfg['candidate_count']))),cfg['xi_lower'],cfg['xi_upper'])
        X=np.vstack([X,list(cfg['fixed_weights'].values())]);save(out/'candidates.json',X.tolist())
        start=time.monotonic()
        if not (out/'excitation.json').exists():characterize(out,cfg,p)
        total=len(X)*len(cfg['mu_variants'])*len(cfg['demands_m_s2']);done=0
        for vi,mu in enumerate(cfg['mu_variants']):
            for k,xi in enumerate(X):
                for di,demand in enumerate(cfg['demands_m_s2']):
                    path=out/'cases'/f'{vi}_{k:03d}_{di}.json'
                    if not path.exists():save(path,evaluate(k,xi,demand,mu,p,cfg,cc))
                    done+=1
                print(f'{done}/{total} cases | mu={mu:g} | elapsed {time.monotonic()-start:.1f}s',flush=True)
    elif not manifest.exists():raise RuntimeError('Run first')
    analyze(out,cfg)
    if args.command=='collect':
        subprocess.run(['git','add','-f','--',str(out.relative_to(ROOT.parent))],cwd=ROOT.parent,check=True)
        print('Staged only the full Gate 1 result directory. Commit and push next.',flush=True)

if __name__=='__main__':main()
