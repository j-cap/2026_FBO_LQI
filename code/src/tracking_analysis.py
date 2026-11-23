"""Calibration-only selection, independent replay, and Q1/Q2/Q3 diagnostics."""
import base64
import io
import itertools
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .tracking_task import descriptor_distances, oracle_features, reference_metadata


def aggregate(batches):
    rows = []
    for b in batches:
        for r in b['rows']:
            rows.append(dict(vehicle=b['vehicle'], context=b['reference'],
                             role=b['role'], seed=b['seed'], **r))
    frame = pd.DataFrame(rows)
    records = []
    for keys, group in frame.groupby(['vehicle', 'context', 'role', 'candidate'], sort=True):
        usable = bool((group['complete'] & group['feasible'] & group['integration_resolved']).all())
        records.append(dict(zip(('vehicle', 'context', 'role', 'candidate'), keys),
                            usable=usable, seeds=len(group),
                            J=float(group.J.mean()) if group.complete.all() else None,
                            rmse_y=float(group.rmse_y.mean()) if group.complete.all() else None,
                            rms_rate_deg_s=float(group.rms_rate_deg_s.mean()) if group.complete.all() else None,
                            failed_seeds=int((~group.feasible).sum()),
                            unresolved_seeds=int((~group.integration_resolved).sum())))
    return frame, pd.DataFrame(records)


def _best(frame, ids=None):
    if ids is not None:
        frame = frame[frame.candidate.isin(ids)]
    valid = frame.groupby('candidate', sort=True).agg(
        all_usable=('usable', 'all'), J=('J', 'mean'))
    valid = valid[valid.all_usable & valid.J.notna()]
    return str(valid.sort_values(['J', 'candidate']).index[0]) if len(valid) else None


def select_policies(aggregated, vehicles, refs, pool):
    """Never inspect validation rows when choosing a controller."""
    cal = aggregated[aggregated.role == 'calibration']
    scale_ids = [p['id'] for p in pool if p['kind'] == 'scaling']
    selected = dict(baseline=pool[0]['id'], common=_best(cal),
                    common_scaling=_best(cal, scale_ids), vehicle_fixed={}, context_tuned={})
    for v in vehicles:
        frame = cal[cal.vehicle == v['id']]
        selected['vehicle_fixed'][v['id']] = _best(frame)
        selected['context_tuned'][v['id']] = {
            r['id']: _best(frame[frame.context == r['id']]) for r in refs}
    return selected


def _lookup(agg, vehicle, context, candidate, role='validation'):
    if candidate is None:
        return None
    row = agg[(agg.vehicle == vehicle) & (agg.context == context) &
              (agg.candidate == candidate) & (agg.role == role)]
    if len(row) != 1:
        return None
    return row.iloc[0]


def policy_row(agg, policy, vehicle, context, candidate, **extra):
    r = _lookup(agg, vehicle, context, candidate)
    usable = r is not None and bool(r.usable)
    return dict(policy=policy, vehicle=vehicle, context=context, candidate=candidate,
                usable=usable, J=float(r.J) if usable else None,
                rmse_y=float(r.rmse_y) if usable else None,
                rms_rate_deg_s=float(r.rms_rate_deg_s) if usable else None, **extra)


def policy_summary(frame):
    output = []
    for name, group in frame.groupby('policy', sort=False):
        valid = group[group.usable]
        all_ok = bool(group.usable.all())
        output.append(dict(policy=name, total_cells=len(group), usable_cells=len(valid),
                           coverage=float(group.usable.mean()),
                           fleet_mean_J=float(valid.J.mean()) if all_ok else None,
                           mean_J_on_usable_cells=float(valid.J.mean()) if len(valid) else None,
                           rmse_y_on_usable_cells=float(valid.rmse_y.mean()) if len(valid) else None,
                           rms_rate_on_usable_cells=float(valid.rms_rate_deg_s.mean()) if len(valid) else None))
    return pd.DataFrame(output)


def q1(agg, selected, vehicles, refs):
    rows = []
    for v in vehicles:
        for r in refs:
            vv, rr = v['id'], r['id']
            choices = {k: selected[k] for k in ('baseline', 'common_scaling', 'common')}
            choices['vehicle_fixed'] = selected['vehicle_fixed'][vv]
            choices['context_tuned'] = selected['context_tuned'][vv][rr]
            rows.extend(policy_row(agg, name, vv, rr, candidate)
                        for name, candidate in choices.items())
    frame = pd.DataFrame(rows)
    return frame, policy_summary(frame)


def near_optimal_sets(agg, cfg):
    sets, rows = {}, []
    cal = agg[agg.role == 'calibration']
    for (vehicle, context), group in cal.groupby(['vehicle', 'context']):
        valid = group[group.usable]
        best = float(valid.J.min()) if len(valid) else None
        good = (set(valid.loc[valid.J <= best*(1+cfg['near_optimal_relative_tolerance']), 'candidate'])
                if best is not None else set())
        sets[vehicle, context] = good
        rows.append(dict(vehicle=vehicle, context=context, best_J=best,
                         eligible_candidates=len(valid), near_optimal_candidates=len(good),
                         candidate_ids=','.join(sorted(good))))
    return sets, pd.DataFrame(rows)


def q2(agg, cfg, vehicles, refs):
    sets, cells = near_optimal_sets(agg, cfg)
    rows = []
    for v in vehicles:
        for a, b in itertools.combinations(refs, 2):
            aa, bb = sets[v['id'], a['id']], sets[v['id'], b['id']]
            available = bool(aa and bb)
            rows.append(dict(vehicle=v['id'], context_a=a['id'], context_b=b['id'],
                             size_a=len(aa), size_b=len(bb), available=available,
                             intersection=len(aa & bb),
                             jaccard=len(aa & bb)/len(aa | bb) if available else None))
    return cells, pd.DataFrame(rows)


def rank_donors(target, ref, vehicles, selected, feature_map, cfg, context_id=None):
    """Distances are fixed a priori; neither target cost nor target optimum is read."""
    context_id = context_id or ref['id']
    target_features = feature_map[target['id'], ref['id']]
    rows = []
    for donor in vehicles:
        if donor['id'] == target['id']:
            continue
        gain = selected['context_tuned'][donor['id']][context_id]
        distance = descriptor_distances(target, donor, target_features,
                                        feature_map[donor['id'], ref['id']], ref, cfg)
        rows.append(dict(donor=donor['id'], candidate=gain, **distance))
    choices = {}
    for metric in ('family', 'physical', 'steady', 'transient', 'closed_loop'):
        valid = [r for r in rows if r['candidate'] is not None and r[metric] is not None
                 and np.isfinite(r[metric])]
        choices[metric] = min(valid, key=lambda r: (r[metric], r['donor'])) if valid else None
    return rows, choices


def q3(agg, selected, vehicles, refs, cfg):
    features = {(v['id'], r['id']): oracle_features(v, r, cfg) for v in vehicles for r in refs}
    pairs, policies, correlations = [], [], []
    for target in vehicles:
        for r in refs:
            vv, rr = target['id'], r['id']
            distances, chosen = rank_donors(target, r, vehicles, selected, features, cfg)
            local = _lookup(agg, vv, rr, selected['context_tuned'][vv][rr])
            case_pairs = []
            for row in distances:
                val = _lookup(agg, vv, rr, row['candidate'])
                good = val is not None and bool(val.usable)
                comparable = good and local is not None and bool(local.usable)
                gap = float((val.J-local.J)/max(abs(local.J), 1e-12)) if comparable else None
                case_pairs.append(dict(vehicle=vv, context=rr, **row, usable=good,
                                       transfer_J=float(val.J) if good else None,
                                       relative_gap_to_local=gap))
            pairs.extend(case_pairs)
            for metric, donor in chosen.items():
                policies.append(policy_row(agg, 'donor_'+metric, vv, rr,
                                           donor['candidate'] if donor else None,
                                           donor=donor['donor'] if donor else None))
            # Descriptive within-target ranks; donor pairs are not independent trials.
            for metric in chosen:
                valid = [p for p in case_pairs if p[metric] is not None
                         and p['relative_gap_to_local'] is not None]
                xx = [p[metric] for p in valid]
                yy = [p['relative_gap_to_local'] for p in valid]
                rho = (float(spearmanr(xx, yy).statistic) if len(valid) >= 3
                       and np.ptp(xx) > 0 and np.ptp(yy) > 0 else None)
                correlations.append(dict(vehicle=vv, context=rr, metric=metric,
                                         donors=len(valid), spearman_rho=rho))
    serial_features = [{k: val for k, val in f.items() if k not in ('hy', 'hd_rate')}
                       for f in features.values()]
    return (pd.DataFrame(pairs), pd.DataFrame(policies), pd.DataFrame(correlations),
            serial_features)


def nearest_context(ref, controlled_refs, dt):
    """Predeclared reference-only assignment, including the unseen 25 m/s speed."""
    ff = reference_metadata(ref, dt)['spectral_centroid_hz']
    def distance(r):
        f = reference_metadata(r, dt)['spectral_centroid_hz']
        return ((ref['speed']-r['speed'])/10.)**2 + np.log(max(ff, 1e-6)/max(f, 1e-6))**2
    return min(controlled_refs, key=lambda r: (distance(r), r['id']))['id']


def realism_policies(selected, vehicles, controlled_refs, realism_refs, cfg):
    features = {(v['id'], r['id']): oracle_features(v, r, cfg)
                for v in vehicles for r in realism_refs}
    rows = []
    for v in vehicles:
        for r in realism_refs:
            context = nearest_context(r, controlled_refs, cfg['dt'])
            choices = {k: selected[k] for k in ('baseline', 'common_scaling', 'common')}
            choices['vehicle_fixed'] = selected['vehicle_fixed'][v['id']]
            choices['context_nearest'] = selected['context_tuned'][v['id']][context]
            _, donors = rank_donors(v, r, vehicles, selected, features, cfg, context)
            for metric, donor in donors.items():
                choices['donor_'+metric] = donor['candidate'] if donor else None
            for policy, gain in choices.items():
                rows.append(dict(policy=policy, vehicle=v['id'], context=r['id'],
                                 candidate=gain, prototype_context=context,
                                 speed=r['speed'],
                                 generalization='interior_speed' if r['speed'] not in
                                 cfg['references']['speeds_m_s'] else 'shape'))
    return rows


def analyze_controlled(batches, selected, vehicles, refs, pool, cfg, out):
    raw, agg = aggregate(batches)
    raw.to_csv(out/'controlled_evaluations.csv', index=False)
    agg.to_csv(out/'controlled_seed_means.csv', index=False)
    rows, summary = q1(agg, selected, vehicles, refs)
    rows.to_csv(out/'q1_validation.csv', index=False)
    summary.to_csv(out/'q1_summary.csv', index=False)
    cells, overlap = q2(agg, cfg, vehicles, refs)
    cells.to_csv(out/'q2_near_optimal.csv', index=False)
    overlap.to_csv(out/'q2_context_overlap.csv', index=False)
    pairs, policies, correlation, features = q3(agg, selected, vehicles, refs, cfg)
    pairs.to_csv(out/'q3_transfer_pairs.csv', index=False)
    policies.to_csv(out/'q3_donor_selection.csv', index=False)
    policy_summary(policies).to_csv(out/'q3_donor_summary.csv', index=False)
    correlation.to_csv(out/'q3_rank_correlations.csv', index=False)
    random_rows = []
    for (vehicle, context), group in pairs.groupby(['vehicle', 'context']):
        available = group[group.candidate.notna()]
        good = available[available.usable]
        random_rows.append(dict(vehicle=vehicle, context=context, donors=len(available),
                                probability_feasible=float(available.usable.mean()) if len(available) else None,
                                expected_J=float(good.transfer_J.mean()) if len(good) == len(available)
                                and len(good) else None,
                                mean_J_conditional_on_feasible=float(good.transfer_J.mean()) if len(good) else None))
    uniform = pd.DataFrame(random_rows)
    uniform.to_csv(out/'q3_uniform_donor_reference.csv', index=False)
    boundary = []
    for c in pool:
        lo, hi = cfg['gains']['log10_multiplier_bounds']
        edge = any(abs(np.log10(m)-b) <= .1*(hi-lo)
                   for m in c['multipliers'] for b in (lo, hi))
        boundary.append(dict(candidate=c['id'], near_search_boundary=edge, **{
            f'K{j+1}': gain for j, gain in enumerate(c['gains'])}))
    pd.DataFrame(boundary).to_csv(out/'candidate_boundary_flags.csv', index=False)
    return dict(raw=raw, agg=agg, q1=summary, q1_rows=rows, q2=cells,
                overlap=overlap, pairs=pairs, donor_policies=policies,
                correlations=correlation, features=features, uniform_donor=uniform)


def analyze_realism(batches, schedule, out):
    raw, agg = aggregate(batches)
    raw.to_csv(out/'realism_evaluations.csv', index=False)
    rows = [policy_row(agg, row['policy'], row['vehicle'], row['context'], row['candidate'],
                       prototype_context=row['prototype_context'], speed=row['speed'],
                       generalization=row['generalization']) for row in schedule]
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'realism_validation.csv', index=False)
    summary = pd.concat([policy_summary(group).assign(generalization=name)
                         for name, group in frame.groupby('generalization')], ignore_index=True)
    summary.to_csv(out/'realism_summary.csv', index=False)
    return frame, summary


def write_report(out, controlled, realism_summary, refs, cfg, manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10})

    def img(fig, name):
        fig.savefig(out/name, dpi=150, bbox_inches='tight')
        stream = io.BytesIO(); fig.savefig(stream, format='png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        return '<img alt="'+name+'" src="data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()+'">'

    figures = []
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True)
    for r in refs:
        axes[0].plot(r['t'], r['ay_ref'], label=r['id'])
        ff = np.fft.rfftfreq(len(r['t']), cfg['dt'])
        axes[1].plot(ff, abs(np.fft.rfft(r['ay_ref']))**2/len(ff)**2)
    axes[0].set(ylabel='Reference lateral demand [m/s²]', xlabel='Time [s]')
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].set(xlabel='Frequency [Hz]', ylabel='Squared FFT amplitude', xlim=(0, 1.5))
    figures.append(img(fig, 'controlled_references.png'))

    pivot = controlled['agg'].query("role == 'calibration'")
    vehicles = sorted(pivot.vehicle.unique())
    contexts = [r['id'] for r in refs]
    fig, ax = plt.subplots(figsize=(9, max(3, len(vehicles)*.43)), constrained_layout=True)
    matrix = np.array([[float(pivot[(pivot.vehicle == v) & (pivot.context == c) & pivot.usable].J.min())
                        for c in contexts] for v in vehicles])
    im = ax.imshow(matrix, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(contexts)), contexts, rotation=30, ha='right')
    ax.set_yticks(range(len(vehicles)), vehicles)
    ax.set_title('Best eligible calibration cost in the finite gain pool')
    fig.colorbar(im, ax=ax, label='J')
    figures.append(img(fig, 'finite_pool_landscapes.png'))

    # Choose one plant per family and the last controlled context by protocol
    # order, without looking for especially favorable gain landscapes.
    family_examples = [next(v for v in manifest['vehicles'] if v['family'] == family)['id']
                       for family in ('compact', 'sport', 'suv')
                       if any(v['family'] == family for v in manifest['vehicles'])]
    fig, axes = plt.subplots(1, len(family_examples), figsize=(4*len(family_examples), 3.8),
                             squeeze=False, constrained_layout=True)
    context = refs[-1]['id']
    for ax, vehicle in zip(axes[0], family_examples):
        frame = pivot[(pivot.vehicle == vehicle) & (pivot.context == context)]
        good = frame[frame.usable]
        ax.scatter(good.rmse_y, good.rms_rate_deg_s, s=22, label='Eligible candidates')
        for name, marker, color in [('baseline', 'x', '#e07a17'), ('common', 's', '#8d43ad'),
                                    ('vehicle_fixed', '^', '#bb4040'), ('context_tuned', '*', '#27833c')]:
            selection = controlled['q1_rows']
            gain = selection[(selection.vehicle == vehicle) & (selection.context == context) &
                             (selection.policy == name)].candidate.iloc[0]
            row = frame[frame.candidate == gain]
            if len(row) and row.J.notna().all():
                ax.plot(row.rmse_y, row.rms_rate_deg_s, linestyle='None', marker=marker,
                        markersize=8, markerfacecolor='none' if marker in ('s', '^') else color,
                        markeredgecolor=color, markeredgewidth=1.5, label=name)
        ax.set(title=vehicle+' / '+context, xlabel='Lateral RMSE [m]', ylabel='Steering-rate RMS [deg/s]')
        ax.set_yscale('log')
        ax.legend(fontsize=7)
    figures.append(img(fig, 'gain_tradeoffs.png'))

    q1rows = controlled['q1_rows']
    common = q1rows[q1rows.policy == 'common'].set_index(['vehicle', 'context'])
    tuned = q1rows[q1rows.policy == 'context_tuned'].set_index(['vehicle', 'context'])
    paired = common[['J', 'usable']].join(tuned[['J', 'usable']], lsuffix='_common', rsuffix='_local')
    usable = paired[paired.usable_common & paired.usable_local]
    paired_gain = (float((usable.J_common-usable.J_local).mean()) if len(usable) else None)
    overall = dict(smoke=manifest['mode'] == 'smoke', controlled_batches=manifest['expected_controlled_batches'],
                   candidates=len(manifest['candidates']), validation_cells=len(paired),
                   paired_usable_cells=len(usable), mean_J_common_minus_context_local=paired_gain,
                   failed_evaluations=int((~controlled['raw'].feasible).sum()),
                   unresolved_evaluations=int((~controlled['raw'].integration_resolved).sum()),
                   q1=controlled['q1'].replace({np.nan: None}).to_dict('records'),
                   realism=realism_summary.replace({np.nan: None}).to_dict('records'))
    (out/'summary.json').write_text(json.dumps(overall, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    def table(df):
        return df.to_html(index=False, float_format=lambda x: f'{x:.4g}', na_rep='unavailable', escape=True)
    title = 'SMOKE: pipeline verification only' if overall['smoke'] else 'Trajectory-tracking diagnostic'
    correlation_rows = []
    for metric, group in controlled['correlations'].groupby('metric'):
        valid = group.spearman_rho.dropna()
        correlation_rows.append(dict(metric=metric, valid_cases=len(valid),
                                     median_within_case_rho=float(valid.median()) if len(valid) else None))
    correlations = pd.DataFrame(correlation_rows)
    html = '''<!doctype html><html><head><meta charset="utf-8"><title>Tracking diagnostic</title>
<style>body{font:16px system-ui;max-width:1180px;margin:36px auto;padding:0 20px;color:#192736}
table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #ccd5df}img{max-width:100%}
code{background:#eef2f5}h2{margin-top:32px}.note{background:#edf3f7;padding:16px}</style></head><body>'''
    html += '<h1>'+title+'</h1><p class="note">Finite-pool diagnostic with true nonlinear plants. '
    html += 'All controller selections use calibration seeds only; tables use independent validation noise. '
    html += 'Exact-model similarity descriptors are oracle diagnostics. No BO or federated benefit is claimed.</p>'
    html += '<p>J = RMSE(e_y)/0.25 m + RMS(applied steering rate)/(10 deg/s). '
    html += 'Coverage is shown before cost. Fleet mean is unavailable if any cell is ineligible; conditional means '
    html += 'must not be compared across unequal coverage. Failed and numerically unresolved trials remain in the CSV files.</p>'
    html += '<h2>Q1: Is there useful tuning and vehicle specialization?</h2>'+table(controlled['q1'])
    html += '<p>The common-scaling baseline allows one multiplier of all three gains. '
    html += 'Vehicle-fixed gains are shared across contexts within a vehicle; context-tuned gains vary by both.</p>'
    html += '<h2>Q2: Do useful gains depend on the reference context?</h2>'+table(controlled['q2'])
    overlaps = controlled['overlap']
    html += '<p>Median near-optimal-set Jaccard: '+str(overlaps.jaccard.median())
    html += '. Empty sets are unavailable, rather than perfect agreement.</p>'
    html += '<h2>Q3: Which donors supply useful gains?</h2>'+table(policy_summary(controlled['donor_policies']))
    html += table(correlations)
    uniform = controlled['uniform_donor']
    html += '<p>Uniform random donor reference: mean feasibility probability '
    html += f'{uniform.probability_feasible.mean():.3f}. '
    html += 'Per-case expected costs and coverage are in q3_uniform_donor_reference.csv; '
    html += 'expected cost is unavailable if any candidate donor fails.</p>'
    html += '<p>Donors are selected by predeclared distance and then evaluated on the target. '
    html += 'Family-distance ties use donor ID. Correlations are computed within each target and context; '
    html += 'donor pairs are dependent and no pooled significance claim is made.</p>'
    html += '<h2>Held-out CFL trajectories</h2>'+table(realism_summary)
    html += '<p>Shape generalization uses 20/30 m/s. Interior-speed generalization uses 25 m/s. '
    html += 'Context selection uses reference speed and spectrum only. The realism bank also changes demand magnitude, '
    html += 'so its outcomes do not isolate frequency effects.</p>'
    html += '<h2>Reference and landscape checks</h2>'+''.join(figures)
    html += '<p>Inspect candidate_boundary_flags.csv before interpreting selected gains as well localized. '
    html += 'The pool gives comparative benchmarks, not globally optimal controllers. See docs/tracking_diagnostic.md '
    html += 'for the frozen design and decision rules.</p></body></html>'
    (out/'report.html').write_text(html, encoding='utf-8')
    return overall
