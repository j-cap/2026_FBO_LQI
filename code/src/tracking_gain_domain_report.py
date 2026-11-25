"""Calibration-selected gain-domain comparisons on fresh held-out repeats."""
import base64
from html import escape
import io
import json

import numpy as np
import pandas as pd

from .tracking_gain_domain import STAGES, boundary_flags

STAGE_LABELS = dict(original='Original 71', dense_original='Dense original box', expanded='Expanded box')


def tables(state, calibration, choices, validation, buckets):
    lookup = {c['id']: c for c in state['pool']}
    original_hi = state['physics']['gains']['log10_multiplier_bounds'][1]
    lo = state['physics']['gains']['log10_multiplier_bounds'][0]
    expanded_hi = np.log10(state['requested']['expanded_multiplier_upper'])
    rows = []
    for context in state['contexts']:
        cid = context['id']
        for stage in STAGES:
            candidate = choices[cid][stage]
            cal = calibration[cid].get(candidate, dict(eligible=False, J=None))
            val = validation[cid].get(candidate, dict(eligible=False, J=None))
            point = lookup.get(candidate)
            flags = boundary_flags(point['multipliers'], lo, expanded_hi if stage == 'expanded' else original_hi,
                state['requested']['boundary_fraction']) if point else dict(near_lower_boundary=None, near_upper_boundary=None)
            rows.append(dict(context=cid, speed=context['speed'], frequency=context['frequency'], stage=stage,
                candidate=candidate, origin=state['labels'][candidate]['origin'] if candidate else None,
                calibration_eligible=bool(cal['eligible']), calibration_J=cal['J'] if cal['eligible'] else None,
                validation_eligible=bool(val['eligible']), validation_J=val['J'] if val['eligible'] else None,
                K_y=point['gains'][0] if point else None, K_psi=point['gains'][1] if point else None,
                K_r=point['gains'][2] if point else None,
                outside_original=state['labels'][candidate]['outside_original'] if candidate else None, **flags))
    selected = pd.DataFrame(rows)
    row_index = {(r['context'], r['stage']): r for r in rows}
    raw = {}
    for cid, batches in buckets.items():
        for batch in batches:
            for row in batch['rows']:
                raw[cid, batch['repeat'], row['candidate']] = row
    comparisons = []
    for baseline, stage, effect in [('original', 'dense_original', 'density'), ('dense_original', 'expanded', 'extension')]:
        for context in state['contexts']:
            cid = context['id']
            before, after = row_index[cid, baseline], row_index[cid, stage]
            common = before['validation_eligible'] and after['validation_eligible']
            change = 100 * (1 - after['validation_J'] / before['validation_J']) if common else None
            positive = tied = negative = 0
            if common:
                for repeat in range(len(state['validation_seeds'])):
                    difference = raw[cid, repeat, before['candidate']]['J'] - raw[cid, repeat, after['candidate']]['J']
                    positive += difference > 1e-12
                    negative += difference < -1e-12
                    tied += abs(difference) <= 1e-12
            comparisons.append(dict(context=cid, speed=context['speed'], frequency=context['frequency'],
                effect=effect, baseline=baseline, stage=stage, baseline_candidate=before['candidate'],
                candidate=after['candidate'], candidate_outside_original=after['outside_original'],
                calibration_improvement_percent=100 * (1 - after['calibration_J'] / before['calibration_J'])
                    if before['calibration_eligible'] and after['calibration_eligible'] else None,
                baseline_validation_eligible=before['validation_eligible'], validation_eligible=after['validation_eligible'],
                common_validation_eligible=common, validation_improvement_percent=change,
                material_improvement=bool(common and change >= state['requested']['material_improvement_percent']),
                positive_validation_repeats=int(positive), negative_validation_repeats=int(negative),
                tied_validation_repeats=int(tied)))
    comparisons = pd.DataFrame(comparisons)
    common_contexts = [c['id'] for c in state['contexts'] if all(row_index[c['id'], s]['validation_eligible'] for s in STAGES)]
    summaries = []
    for stage in STAGES:
        frame = selected[selected.stage == stage]
        summaries.append(dict(stage=stage, candidates=len(state['members'][stage]), contexts=len(frame),
            calibration_eligible_winners=int(frame.calibration_eligible.sum()),
            validation_eligible_winners=int(frame.validation_eligible.sum()),
            selected_outside_original=int(frame.outside_original.fillna(False).sum()),
            selected_near_lower_boundary=int(frame.near_lower_boundary.fillna(False).sum()),
            selected_near_upper_boundary=int(frame.near_upper_boundary.fillna(False).sum()),
            common_all_stage_contexts=len(common_contexts),
            mean_validation_J_on_common_contexts=float(frame[frame.context.isin(common_contexts)].validation_J.mean())
                if common_contexts else None))
    profiles = []
    unique_sources = {(p['candidate'], p['coordinate'], p['frequency_hz']) for p in state['profiles']}
    for source, j, frequency in sorted(unique_sources):
        points = [source] + [p['candidate_id'] for p in state['profiles']
            if p['candidate'] == source and p['coordinate'] == j and p['frequency_hz'] == frequency]
        for context in state['contexts']:
            if context['frequency'] != frequency:
                continue
            for candidate in points:
                val = validation[context['id']][candidate]
                profiles.append(dict(context=context['id'], speed=context['speed'], frequency=frequency,
                    source_candidate=source, coordinate=j, candidate=candidate,
                    multiplier=lookup[candidate]['multipliers'][j], actual_gain=lookup[candidate]['gains'][j],
                    outside_original=state['labels'][candidate]['outside_original'],
                    validation_eligible=bool(val['eligible']), validation_J=val['J'] if val['eligible'] else None))
    return selected, comparisons, pd.DataFrame(summaries), pd.DataFrame(profiles)


def write_report(state, calibration, choices, validation, buckets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = state['out']
    selected, comparisons, summary, profiles = tables(state, calibration, choices, validation, buckets)
    pool = pd.DataFrame([dict(candidate=c['id'], origin=state['labels'][c['id']]['origin'],
        original=state['labels'][c['id']]['original'], dense_original=state['labels'][c['id']]['dense_original'],
        outside_original=state['labels'][c['id']]['outside_original'],
        K_y=c['gains'][0], K_psi=c['gains'][1], K_r=c['gains'][2],
        multiplier_y=c['multipliers'][0], multiplier_psi=c['multipliers'][1], multiplier_r=c['multipliers'][2]) for c in state['pool']])
    availability = pd.DataFrame([dict(context=cid, stage=stage, candidates=len(members),
        calibration_eligible=sum(rows[k]['eligible'] for k in members))
        for cid, rows in calibration.items() for stage, members in state['members'].items()])
    frames = dict(candidate_pool=pool, selected_controllers=selected, paired_improvements=comparisons,
        stage_summary=summary, boundary_profiles=profiles, calibration_availability=availability)
    for name, frame in frames.items():
        frame.to_csv(out / (name + '.csv'), index=False)
    readout = dict(mode=state['specification']['mode'], new_online_requests=0,
        spectral_basis_constructed=False, confirmation_claim=False,
        interpretation='Offline finite-pool best-found gains; calibration selection and fresh validation are separate.')
    for effect, frame in comparisons.groupby('effect', sort=False):
        readout[effect] = dict(contexts=len(frame), common_validation_eligible=int(frame.common_validation_eligible.sum()),
            improved_contexts=int((frame.validation_improvement_percent > 1e-10).sum()),
            materially_improved_contexts=int(frame.material_improvement.sum()),
            lost_validation_eligibility=int((frame.baseline_validation_eligible & ~frame.validation_eligible).sum()))
    (out / 'readout.json').write_text(json.dumps(readout, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, (source, j, frequency) in zip(axes, sorted({(p['candidate'], p['coordinate'], p['frequency_hz']) for p in state['profiles']})):
        frame = profiles[(profiles.source_candidate == source) & (profiles.coordinate == j)]
        for speed, points in frame.groupby('speed', sort=True):
            points = points.sort_values('actual_gain')
            line, = ax.plot(points.actual_gain, points.validation_J, marker='o', label=f'{speed:g} m/s')
            invalid = points[~points.validation_eligible]
            ax.scatter(invalid.actual_gain, np.full(len(invalid), .04), transform=ax.get_xaxis_transform(),
                marker='x', color=line.get_color())
        original_upper = state['physics']['gains']['nominal'][j] * 10
        ax.axvline(original_upper, color='#68747c', linestyle='--', label='Old upper bound')
        # Mixed data/axes coordinates for failure markers do not expand x limits.
        # Include every tested gain, even when its cost is unavailable.
        xmin, xmax = frame.actual_gain.min(), frame.actual_gain.max()
        margin = .05 * (xmax - xmin)
        ax.set_xlim(xmin - margin, xmax + margin)
        ax.set(xlabel=('K_y [rad/m]' if j == 0 else 'K_psi [rad/rad]'), ylabel='Held-out mean J',
            title=f'{frequency:g} Hz: one gain varied from {source}')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, ncol=2)
    fig.savefig(out / 'boundary_profiles.png', dpi=150)
    profile_image = io.BytesIO()
    fig.savefig(profile_image, format='png', dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 3.8), constrained_layout=True)
    contexts = [c['id'] for c in state['contexts']]
    for stage, color in zip(STAGES, ['#66748a', '#3478b0', '#007d75']):
        frame = selected[selected.stage == stage].set_index('context').loc[contexts]
        ax.plot(range(len(contexts)), frame.validation_J, marker='o', label=STAGE_LABELS[stage], color=color)
    ax.set(xticks=range(len(contexts)), xticklabels=contexts, ylabel='Held-out mean J', title='Calibration-selected controllers, fresh validation')
    ax.tick_params(axis='x', labelrotation=55, labelsize=8)
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    fig.savefig(out / 'selected_costs.png', dpi=150)
    cost_image = io.BytesIO()
    fig.savefig(cost_image, format='png', dpi=120)
    plt.close(fig)
    def table(frame):
        frame = frame.copy()
        for column in ('stage', 'baseline'):
            if column in frame:
                frame[column] = frame[column].map(STAGE_LABELS)
        return '<div class="table">' + frame.to_html(index=False, float_format=lambda x: f'{x:.5g}', na_rep='unavailable') + '</div>'
    nominal = np.asarray(state['physics']['gains']['nominal'])
    bounds = pd.DataFrame(dict(gain=['K_y', 'K_psi', 'K_r'], nominal=nominal,
        original_lower=nominal * .1, original_upper=nominal * 10,
        expanded_lower=nominal * .1, expanded_upper=nominal * state['requested']['expanded_multiplier_upper']))
    title = 'Gain-domain audit: sampling density and genuine boundary extension'
    if state['specification']['mode'] == 'smoke':
        title = 'Gain-domain SMOKE: software verification only'
    html = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    html += '<title>' + title + '</title><style>body{font:16px/1.6 system-ui;max-width:1250px;margin:35px auto;padding:0 22px;color:#203747}'
    html += 'table{border-collapse:collapse;font-size:12px}td,th{padding:8px;border:1px solid #d0dce2}.table{overflow-x:auto;margin:18px 0}'
    html += 'h2{margin-top:35px}img{max-width:100%}.note{background:#edf6f2;padding:18px;border-left:4px solid #007d75}</style></head><body>'
    html += '<h1>' + title + '</h1><p>This offline audit asks whether the original 71-point controller pool was too coarse '
    html += 'or its physical upper bounds too restrictive. The nominal controller, plant, trajectories, objective and eligibility rules remain fixed.</p>'
    html += '<h2>Actual physical gain bounds</h2>' + table(bounds)
    html += '<p>Original: the frozen 71 gain vectors. Dense original: the nested 1,024 Sobol draws plus original scaling '
    html += 'anchors and two old-upper-face probes (reduced Sobol counts in smoke). Expanded: the entire dense-original '
    html += 'set plus only points outside the old box, sampled in the 0.1–30 multiplier box and supplemented by fixed '
    html += 'boundary profiles and common scalings. Expanded Sobol points still inside the old box are discarded. '
    html += 'Thus an old-boundary or interior-resolution improvement cannot be counted as evidence of a boundary extension.</p>'
    html += '<h2>Stage summary</h2>' + table(summary)
    html += '<p class="note">Original calibration rows are reused only after the source simulator parity checks pass. '
    html += 'Each stage\'s winner is selected using the same two calibration noise repeats; all three choices then receive '
    html += f'{len(state["validation_seeds"])} fresh common validation repeats. Held-out failure does not trigger fallback '
    html += 'or reselection. Mean costs use the common eligible context set shown in the table. These are finite-pool '
    html += 'best-found controllers on one plant, not continuous global optima or population confirmation.</p>'
    html += '<h2>Density and extension effects</h2>' + table(comparisons)
    html += '<p>Positive improvement means lower J. Material improvement uses the predeclared '
    html += f'{state["requested"]["material_improvement_percent"]:g}% threshold. Per-repeat signs and eligibility remain explicit.</p>'
    html += '<img alt="Held-out costs of the calibration-selected controllers" src="data:image/png;base64,' + base64.b64encode(cost_image.getvalue()).decode() + '">'
    html += '<h2>Controlled boundary profiles</h2><p>The profiles were selected retrospectively from the earlier '
    html += 'g012/K_y and g030/K_psi boundary observations. Other gains remain exactly fixed. The point at multiplier 10 '
    html += 'belongs to the dense original box; 12, 20 and 30 are genuinely outside. Crosses at the bottom mark '
    html += 'validation-ineligible profile points; they are not low-cost outcomes.</p>'
    html += '<img alt="Physical gain boundary profiles with old upper limits marked" src="data:image/png;base64,' + base64.b64encode(profile_image.getvalue()).decode() + '">'
    html += '<h2>Selected controllers</h2>' + table(selected)
    html += '<h2>Interpretation</h2><p>No new online BO requests, kernel training or public spectral-basis construction '
    html += 'are performed. The old normalized gain transform is retained, so outside points have coordinates above 1. '
    html += 'All failed and unresolved offline attempts remain in the availability counts. The original online studies '
    html += 'and their reference labels are preserved. This audit does not rerank their recorded policies under a new benchmark.</p>'
    html += '<p>Upper limits are extended for all three gains; lower limits remain unchanged. A selected point near '
    html += 'an expanded boundary still leaves that direction unresolved. Future larger-domain online BO would need '
    html += 'a scalable feature representation and a separately fixed protocol.</p>'
    html += '<p>Configuration: <code>' + escape(state['requested']['tag']) + '</code>; signature <code>' + escape(state['signature']) + '</code>.</p></body></html>'
    (out / 'report.html').write_text(html, encoding='utf-8')
    return frames
