"""Retrospective requirement sensitivity; no tuning or engineering-limit selection."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'docs/analysis/cooperative_requirements'


def analyze():
    OUT.mkdir(parents=True,exist_ok=True)
    episode_path=ROOT/'results/actuator_mechanism_v7/processed/episodes.csv'
    transfer_path=ROOT.parent/'docs/analysis/cooperative_stage_a/transfer_targets.csv'
    d=pd.read_csv(episode_path)
    d=d[d.arm.isin(['target_redesign','reference'])].copy()
    if len(d)!=4800 or not (d.feasible & d.fine_feasible & d.integration.eq('pass') & d.true_integration.eq('pass')).all():
        raise ValueError('Expected 4800 valid common/reference episode records')
    ids=['seed','client','tau','maneuver']
    means=d.groupby(ids+['arm'])[['cost','true_cost']].mean().reset_index()
    means['measured_deg']=np.rad2deg(means.cost)
    means['true_deg']=np.rad2deg(means.true_cost)
    means.to_csv(OUT/'condition_means.csv',index=False)
    common=means[means.arm.eq('target_redesign')]
    ref=means[means.arm.eq('reference')]
    worst=common.loc[common.measured_deg.idxmax()]
    match=np.ones(len(d),dtype=bool)
    for k in ids:match &= d[k].eq(worst[k])
    paired=d[match].pivot(index='noise_seed',columns='arm',values='cost')
    if len(paired)!=40 or paired.isna().any().any():raise ValueError('Expected 40 paired worst-condition episodes')
    x=np.rad2deg(paired.target_redesign.to_numpy())
    delta=np.rad2deg((paired.target_redesign-paired.reference).to_numpy())
    rng=np.random.default_rng(20261119)
    indices=rng.integers(0,len(x),(20000,len(x)))
    interval=np.quantile(x[indices].mean(axis=1),[.025,.975])
    delta_interval=np.quantile(delta[indices].mean(axis=1),[.025,.975])
    t=pd.read_csv(transfer_path)
    keys=['donor_seed','donor_client','method','optimizer_seed','budget']
    runs=[]
    for identity,g in t.groupby(keys):
        if len(g)!=25 or not g.valid.all():raise ValueError('Incomplete/invalid donor coverage')
        targets=g[['target_seed','target_client','tau']].rename(columns={'target_seed':'seed','target_client':'client'})
        matched=common.merge(targets,on=['seed','client','tau'],validate='many_to_one')
        if len(matched)!=50:raise ValueError('Mismatched common/donor target coverage')
        runs.append(dict(zip(keys,identity),donor_required_deg=float(np.rad2deg(g.required_threshold.max())),
                         common_required_deg=float(matched.measured_deg.max())))
    runs=pd.DataFrame(runs);runs.to_csv(OUT/'matched_runs.csv',index=False)
    rows=[]
    for b in [.90,.91,.92,.93,.94,.95]:
        for (method,budget),g in runs.groupby(['method','budget']):
            cp=g.common_required_deg<=b;dp=g.donor_required_deg<=b
            rows.append(dict(threshold_deg=b,method=method,budget=budget,runs=len(g),donor_pass=int(dp.sum()),
                common_pass_matched=int(cp.sum()),common_pass_donor_fail=int((cp&~dp).sum()),
                common_full_conditions=int((common.measured_deg<=b).sum()),total_full_conditions=len(common)))
    table=pd.DataFrame(rows);table.to_csv(OUT/'threshold_comparison.csv',index=False)
    summary=dict(common_max_deg=float(common.measured_deg.max()),reference_max_deg=float(ref.measured_deg.max()),
        worst_condition={k:(int(worst[k]) if k=='seed' else float(worst[k]) if k in ['tau','maneuver'] else worst[k]) for k in ids},
        worst_condition_episodes=len(x),worst_condition_sd_deg=float(x.std(ddof=1)),
        pointwise_bootstrap_95_percent_interval_deg=interval.tolist(),
        paired_common_minus_reference_deg=float(delta.mean()),paired_difference_interval_deg=delta_interval.tolist(),
        bootstrap_seed=20261119,bootstrap_replicates=20000,
        interpretation='Retrospective sensitivity only; interval ignores worst-condition selection and is not a simultaneous fleet certificate.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    text=f'''# Requirement sensitivity: common weights versus donor calibration

No new optimization or simulation. All thresholds below are illustrative, not externally justified requirements.

## Result

The retrospective common weights have worst per-maneuver mean measured RMSE
**{summary['common_max_deg']:.6f} deg/s** across 6 vehicles x 5 bandwidths x 2 maneuvers.
The target-specific references have worst mean **{summary['reference_max_deg']:.6f} deg/s**.
References were optimized for the combined measured objective; neither reference
nor common weights establish the minimax optimum or an infeasibility bound.

At 0.92 deg/s, common weights satisfy all 60 condition/maneuver sample means.
The donor comparison uses the SAME five-other-vehicle target coverage for both
common and donor weights. Full donor-inclusive calibration success remains unknown.

| Threshold (deg/s) | BO25 | BO75 | Sobol25 | Sobol75 | Common full conditions |
|---|---:|---:|---:|---:|---:|
'''
    for b,g in table.groupby('threshold_deg'):
        vals=[]
        for method,budget in [('bo_lower_penalty',25),('bo_lower_penalty',75),('sobol',25),('sobol',75)]:
            z=g[(g.method==method)&(g.budget==budget)].iloc[0];vals.append(f'{int(z.donor_pass)}/{int(z.runs)}')
        text+=f"| {b:.2f} | "+' | '.join(vals)+f" | {int(g.iloc[0].common_full_conditions)}/60 |\n"
    text+=f'''
## Why 0.92 deg/s is not yet a defensible requirement

The worst common-weight case is seed {int(worst['seed'])}, {worst['client']},
tau={worst['tau']:g} s, maneuver={worst['maneuver']:g} s.
Its 40 episode RMSEs have standard deviation {summary['worst_condition_sd_deg']:.4f} deg/s.
An exploratory pointwise percentile bootstrap interval for the mean is
**[{interval[0]:.4f}, {interval[1]:.4f}] deg/s**. The 0.92 threshold lies inside it.
This interval ignores selection of the worst condition and is not a simultaneous
fleet guarantee. Episodes provide noise replication, not independent fleet replication.

The paired common-minus-reference difference on this condition is
{delta.mean():.6f} deg/s, with exploratory interval
[{delta_interval[0]:.6f}, {delta_interval[1]:.6f}] deg/s.
Similar common/reference performance does not remove uncertainty about an absolute limit.

## Corrected interpretation of the demanding condition

Payload/tau=0.4 often has the largest cost over the FULL screening grid.
For the good common weights, tire_degraded/tau=0.4 and the faster maneuver limit
performance. A grid-wide worst-case frequency is therefore not enough to choose
one representative vehicle for calibration. Any demanding-condition rule selected
from these results is exploratory and needs independent confirmation.

## Requirement specification to complete before Stage B

1. Meaning: expected episode yaw-rate RMSE separately for each prescribed maneuver,
   or a specified probability of exceeding a limit? These are different requirements.
2. Signal: measured RMSE includes sensor noise; true-state RMSE is currently only a
   simulation diagnostic. Use a common measurement convention for any deployment claim.
3. Value: supply an absolute limit from an engineering acceptance rule or partner
   specification. Alternatively declare an illustrative benchmark requirement explicitly.
4. Evidence: specify accepted uncertainty and physical-constraint violation tolerance.
   Sample means below the limit alone are not a reliability certificate.
5. Scope: finite evaluated vehicles and maneuvers initially; unseen-vehicle claims need
   independent fleets. Preserve model-specific gain redesign and the physical limits.

## Next decision

Do not launch fleet BO based on the 0.92 threshold alone. First settle the acceptance
meaning and engineering limit. If a tight limit is justified, use fresh noise episodes
to validate frozen common/reference candidates before spending on optimization.
Use a sample-size plan and one fixed final analysis, or a sequentially valid rule;
do not repeatedly sample until an ordinary confidence interval passes.
If no practical limit is available, this can only be an exploratory benchmark study.
The existing 0.92/0.93/0.95 sensitivity grid must not be sold as a preregistered
engineering requirement. No current result demonstrates experiment-count savings.
'''
    (OUT/'report.md').write_text(text,encoding='utf-8')
    def digest(p):return hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    manifest=dict(hash_convention='SHA256 of UTF-8 file bytes after CRLF to LF normalization',
                  inputs={str(p.relative_to(ROOT.parent)):digest(p) for p in [episode_path,transfer_path,Path(__file__)]})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(summary,indent=2));print('Saved',OUT/'report.md')

if __name__=='__main__':analyze()
