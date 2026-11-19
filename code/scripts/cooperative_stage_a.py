"""Read-only opportunity audit. No optimization, simulation or threshold selection."""
import argparse
import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / 'results'
OUT = ROOT.parent / 'docs/analysis/cooperative_stage_a'


def aligned_curves(maps):
    """Exact-vector alignment; unknown/failed outcomes cannot pass."""
    keys = sorted(set.intersection(*(set(m) for m in maps.values())))
    if not keys:
        return [], []
    names = list(maps)
    a = np.array([[maps[n][k] for n in names] for k in keys], dtype=float)
    finite = a[np.isfinite(a)]
    if not len(finite):
        return [], []
    thresholds = np.unique(np.r_[0., finite])
    curves, donors = [], []
    for b in thresholds:
        ok = np.isfinite(a) & (a <= b)
        joint = ok.all(axis=1)
        curves.append(dict(threshold=b, n_shared=len(keys), joint=int(joint.sum()),
                           joint_fraction=float(joint.mean())))
        for j, n in enumerate(names):
            count = int(ok[:, j].sum())
            donors.append(dict(threshold=b, donor=n, donor_acceptable=count,
                               joint=int(joint.sum()), conditional_transfer=float(joint.sum()/count) if count else np.nan))
    return curves, donors


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    inputs = {}
    def read(path):
        raw = path.read_bytes()
        inputs[path.relative_to(ROOT.parent).as_posix()] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    def csv(path):
        raw = path.read_bytes()
        inputs[path.relative_to(ROOT.parent).as_posix()] = hashlib.sha256(raw).hexdigest()
        return pd.read_csv(io.BytesIO(raw))
    print('Reading corrected matched grids...', flush=True)
    records = []
    files = sorted((P/'actuator_corrected_v5/seeds').glob('*/*/references/*.json'))
    if len(files) != 30:
        raise ValueError(f'Expected 30 reference files, found {len(files)}')
    for f in files:
        d = read(f)
        seed, client, tau = int(f.parts[-4]), f.parts[-3], float(f.stem)
        seen = set()
        for c in d['candidates']:
            if 'new_sobol' not in c['origins']:
                continue
            key = tuple(c['xi'])
            if key in seen:
                raise ValueError(f'Duplicate grid vector: {f}')
            seen.add(key)
            v = c['calibration']
            records.append(dict(seed=seed, client=client, tau=tau, xi=json.dumps(key),
                cost=v['cost'], feasible=v['feasible'],
                validation_integration=c.get('validation_integration', {}).get('status', 'not_checked')))
        if len(seen) != 256:
            raise ValueError(f'Expected 256 new Sobol points: {f}')
    grid = pd.DataFrame(records)
    grid.to_csv(out/'grid_scores.csv', index=False)
    curves, donors, coverage, bottlenecks = [], [], [], []
    groups = []
    for seed, s in grid.groupby('seed'):
        for tau, g in s.groupby('tau'):
            groups.append((f'{seed}/tau={tau:g}', g))
        for client, g in s.groupby('client'):
            groups.append((f'{seed}/{client}/all_tau', g))
        groups.append((f'{seed}/all_tau', s))
    groups.append(('both_realizations', grid))
    for label, g in groups:
        maps = {}
        for (seed, client, tau), h in g.groupby(['seed', 'client', 'tau']):
            maps[f'{seed}/{client}/{tau:g}'] = {tuple(json.loads(r.xi)): r.cost if r.feasible and np.isfinite(r.cost) else np.inf for r in h.itertuples()}
        c, d = aligned_curves(maps)
        coverage.append(dict(group=label, systems=len(maps), shared_points=c[0]['n_shared'] if c else 0))
        curves.extend(dict(group=label, **x) for x in c)
        donors.extend(dict(group=label, **x) for x in d)
        keys = sorted(set.intersection(*(set(m) for m in maps.values())))
        names = list(maps)
        for k in keys:
            vals = np.array([maps[n][k] for n in names])
            for n, v in zip(names, vals):
                if v == max(vals):
                    bottlenecks.append(dict(group=label, xi=json.dumps(k), limiting_system=n, worst_cost=v))
    curves = pd.DataFrame(curves)
    curves.to_csv(out/'threshold_curves.csv', index=False)
    pd.DataFrame(donors).to_csv(out/'conditional_transfer.csv', index=False)
    pd.DataFrame(coverage).to_csv(out/'alignment_coverage.csv', index=False)
    pd.DataFrame(bottlenecks).to_csv(out/'bottlenecks.csv', index=False)
    grid.groupby(['seed','client','tau']).cost.quantile([0,.25,.5,.75,1]).rename('cost').to_csv(out/'grid_quantiles.csv')

    print('Reading unique transfer evaluations (no repeated donor-label counting)...', flush=True)
    transfers = csv(P/'actuator_transfer_v6/processed/transfers.csv')
    if len(transfers) != 1800:
        raise ValueError('Expected 1800 transfer labels')
    cache, episodes = {}, []
    for idx, r in enumerate(transfers.itertuples()):
        key = (r.target_seed, r.target_client, r.evaluation_key)
        if key in cache:
            continue
        directory = P/f'actuator_transfer_v6/seeds/{key[0]}/{key[1]}/evaluations/{key[2]}'
        fs = sorted(directory.glob('[0-9][0-9][0-9].json'))
        if len(fs) != 80:
            raise ValueError(f'Expected 80 episodes: {directory}')
        local = []
        for f in fs:
            d = read(f)
            a, b = d['coarse'], d['fine']
            local.append(dict(target_seed=key[0], target_client=key[1], evaluation_key=key[2],
                maneuver=d['maneuver'], noise_seed=d['noise_seed'], cost=a['cost'],
                feasible=bool(a['feasible'] and b['feasible']),
                integration=d['integration']['status']))
        episodes.extend(local)
        e = pd.DataFrame(local)
        good = bool(e.feasible.all() and e.integration.eq('pass').all() and np.isfinite(e.cost).all())
        means = e.groupby('maneuver').cost.mean()
        if set(means.index) != {1.,2.}:
            raise ValueError('Unexpected maneuvers')
        cache[key] = dict(valid=good, combined_cost=float(e.cost.mean()),
                          cost_1=float(means.loc[1.]), cost_2=float(means.loc[2.]),
                          required_threshold=float(means.max()) if good else np.inf)
        if len(cache) % 100 == 0:
            print(f'  {len(cache)} unique evaluations read', flush=True)
    targets = []
    for r in transfers.to_dict('records'):
        d = cache[(r['target_seed'], r['target_client'], r['evaluation_key'])]
        if not np.isclose(r['cost'],d['combined_cost'],rtol=1e-10,atol=1e-12):
            raise ValueError('Saved aggregate does not match episodes')
        targets.append(dict(**r, **d))
    target = pd.DataFrame(targets)
    target.to_csv(out/'transfer_targets.csv', index=False)
    ep = pd.DataFrame(episodes)
    ep.to_csv(out/'unique_transfer_episodes.csv', index=False)
    source_cols = ['donor_seed','donor_client','method','optimizer_seed','budget']
    runs = []
    for keys, g in target.groupby(source_cols):
        if len(g) != 25 or len(g[['target_seed','target_client']].drop_duplicates()) != 5:
            raise ValueError('Expected five other vehicles x five bandwidths per source run')
        runs.append(dict(zip(source_cols, keys), required_threshold=g.required_threshold.max(),
                         all_valid=bool(g.valid.all()), n_target_conditions=len(g)))
    runs = pd.DataFrame(runs)
    runs.to_csv(out/'source_run_requirements.csv', index=False)
    success = []
    for (method,budget), g in runs.groupby(['method','budget']):
        for b in np.unique(np.r_[0., runs.required_threshold[np.isfinite(runs.required_threshold)]]):
            success.append(dict(method=method,budget=budget,threshold=b,runs=len(g),
                                accepted=int((g.required_threshold <= b).sum())))
    pd.DataFrame(success).to_csv(out/'source_run_threshold_curves.csv',index=False)
    print('Summarizing paired measured/true mechanism evidence...', flush=True)
    mech = csv(P/'actuator_mechanism_v7/processed/episodes.csv')
    if len(mech) != 9600:
        raise ValueError('Expected 9600 mechanism episode records')
    mech.groupby(['arm','maneuver']).agg(episodes=('cost','size'),
        physical_failures=('feasible',lambda x:int((~x).sum())),
        unresolved=('integration',lambda x:int(x.ne('pass').sum())),
        measured_rmse=('cost','mean'), true_rmse=('true_cost','mean')).to_csv(out/'mechanism_descriptive.csv')
    # Only valid target redesign/reference records form the score comparison figure.
    valid = mech[mech.arm.isin(['target_redesign','reference']) & mech.feasible & mech.fine_feasible & mech.integration.eq('pass') & mech.true_integration.eq('pass')]
    valid.groupby(['seed','client','tau','arm','maneuver'])[['cost','true_cost']].mean().to_csv(out/'mechanism_maneuvers.csv')

    plt.rcParams.update({'font.size':10})
    fig, ax = plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for label,g in curves.groupby('group',sort=False):
        if '/tau=' in label:
            ax[0].step(np.rad2deg(g.threshold),g.joint_fraction,where='post',label=label)
    ax[0].set(xlabel='Combined measured RMSE threshold (deg/s)',ylabel='Fraction of grid jointly acceptable',xlim=(0.7,1.5),title='Screening evidence: three vehicles per realization')
    ax[0].legend(fontsize=6,ncol=2)
    for (method,budget),g in pd.DataFrame(success).groupby(['method','budget']):
        ax[1].step(np.rad2deg(g.threshold),g.accepted/g.runs,where='post',label=f'{method}, B={budget}')
    ax[1].set(xlabel='Per-maneuver RMSE threshold (deg/s)',ylabel='Fraction of source runs passing all targets',title='Five other vehicles × five bandwidths',xlim=(0.89,0.95))
    ax[1].legend(fontsize=8)
    fig.savefig(out/'opportunity.svg');plt.close(fig)
    summary = dict(grid_rows=len(grid),grid_feasible=int(grid.feasible.sum()),
        grid_numerically_checked_on_validation=int(grid.validation_integration.eq('pass').sum()),
        unique_transfer_evaluations=len(cache),unique_transfer_episodes=len(ep),
        transfer_physical_failures=int((~ep.feasible).sum()),transfer_unresolved=int(ep.integration.ne('pass').sum()),
        source_runs=len(runs),required_threshold_deg_range=list(np.rad2deg([runs.required_threshold.min(),runs.required_threshold.max()])),
        decision='No new cooperative optimization campaign justified yet: quantify thresholds, obtain engineering requirement; current grids are screening-only.',
        limitations=['Grid costs combine maneuvers; no grid true-state scores.',
                     'Most grid candidates have no saved coarse/fine validation check.',
                     'No common grid points across the two fleet realizations.',
                     'Direct transfers cover five OTHER vehicles; donor at all bandwidths was not tested.',
                     'Bandwidth variants and optimizer repetitions are not independent fleets.',
                     'Threshold curves are retrospective descriptions, not a policy or sample-efficiency experiment.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    body = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Cooperative calibration: Stage A</title>
<style>body{{max-width:1100px;margin:40px auto;padding:0 24px;font:17px/1.55 system-ui;color:#182534}}h1{{line-height:1.2}}svg{{width:100%;height:auto}}table{{border-collapse:collapse}}td,th{{padding:8px;border-bottom:1px solid #ddd}}.note{{background:#eef3f8;padding:18px}}</style>
<h1>Does fleet cooperation have room to help?</h1><p>Stage A · corrected saved evidence · no new simulation or optimization</p>
<p class="note">These curves show how acceptance changes with the requirement. They do not select a justified engineering threshold or demonstrate an experiment saving.</p>
<svg{(out/'opportunity.svg').read_text().split('<svg',1)[1]}
<h2>What the evidence covers</h2><p>{len(grid):,} grid evaluations; {len(cache)} unique donor-transfer evaluations; {len(ep):,} transfer episodes; {len(runs)} source calibration runs. Repeated donor labels are not independent evidence.</p>
<p>All tested target conditions pass the per-maneuver threshold for a source run once it exceeds that run's worst maneuver mean. Across the saved source runs these thresholds range from {summary['required_threshold_deg_range'][0]:.4f} to {summary['required_threshold_deg_range'][1]:.4f} deg/s. This concerns five other vehicles at five bandwidths, not a complete donor-inclusive fleet test.</p>
<h2>Strong transfer baseline</h2><p>At an illustrative 0.95 deg/s per-maneuver threshold, {int((runs.required_threshold <= np.deg2rad(.95)).sum())}/{len(runs)} source runs pass all their tested target conditions. This threshold is not a selected engineering requirement. The table shows each method's worst required threshold across source runs, in deg/s.</p>
{runs.assign(required_deg=np.rad2deg(runs.required_threshold)).groupby(['method','budget']).required_deg.agg(['min','median','max']).round(4).to_html()}
<p>At the same numeric threshold applied to the different, combined-maneuver grid score, {', '.join(f'{v:.1%}' for v in curves[curves.group.isin(['355238/all_tau','987946/all_tau']) & (curves.threshold <= np.deg2rad(.95))].groupby('group').tail(1).joint_fraction)} of the sampled weights pass all 15 vehicle/bandwidth conditions within the respective realizations. These are screening results; they cannot substitute for per-maneuver acceptance. Broad shared regions remain plausible.</p>
<p>The payload vehicle at tau=0.4 s has the largest grid cost for {sum(x['group']=='355238/all_tau' and x['limiting_system']=='355238/payload_00/0.4' for x in bottlenecks)}/256 candidates in one realization and {sum(x['group']=='987946/all_tau' and x['limiting_system']=='987946/payload_00/0.4' for x in bottlenecks)}/256 in the other. This suggests a demanding-condition calibration baseline worth testing before a sophisticated allocation algorithm. It does not prove that this vehicle is always sufficient.</p>
<h2>Numerical evidence</h2><p>Transfer episodes: {summary['transfer_physical_failures']} physical failures and {summary['transfer_unresolved']} unresolved checks. Grid: {summary['grid_numerically_checked_on_validation']}/{len(grid)} candidates have a saved passing validation integration check; that different-bank check does not certify every grid score.</p>
<h2>Interpretation and next decision</h2><p>A broad region at an acceptable engineering threshold favors simple transfer. A narrow region with vehicle-dependent exclusions motivates further study, but these descriptive curves cannot establish that cooperation saves experiments. Obtain the engineering requirement before Stage B; do not choose a threshold because it makes transfer fail.</p>
<h2>Coverage limits</h2><ul>{''.join('<li>'+x+'</li>' for x in summary['limitations'])}</ul>
<p>CSV outputs retain exact threshold breakpoints and aligned weights. The left plot is cropped to 0.7–1.5 deg/s for readability; the CSV covers the complete observed range. Measured and true-state diagnostics are in mechanism_maneuvers.csv and concern only common/reference controllers.</p></html>'''
    (out/'report.html').write_text(body,encoding='utf-8')
    outputs = {f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.iterdir()) if f.is_file() and f.name!='manifest.json'}
    (out/'manifest.json').write_text(json.dumps(dict(version=1,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs=inputs,outputs=outputs),indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    print(f'Report: {out / "report.html"}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['run','collect'])
    a=p.parse_args()
    if a.stage=='run':
        run(OUT)
    else:
        m=json.loads((OUT/'manifest.json').read_text())
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=m['script_sha256']:
            raise ValueError('Analysis code changed; rerun before collecting')
        for name,h in m['outputs'].items():
            if hashlib.sha256((OUT/name).read_bytes()).hexdigest()!=h:
                raise ValueError(f'Output changed: {name}; rerun')
        files=[OUT/'manifest.json']+[OUT/n for n in m['outputs']]
        subprocess.run(['git','add','--']+[str(f) for f in files],cwd=ROOT.parent,check=True)
        print('Stage A outputs staged. Commit and push.',flush=True)

if __name__=='__main__':
    main()
