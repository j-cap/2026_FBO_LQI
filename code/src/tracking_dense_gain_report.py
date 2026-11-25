"""Matched larger-domain outcomes, with sharing and practical tradeoffs explicit."""
import base64
from html import escape
import io
import json

import numpy as np
import pandas as pd

from . import tracking_six_speed as old
from .tracking_six_speed_report import paired_comparisons as legacy_pairs, visit_tables
from .tracking_dense_gain import A, B, D, E, METHODS

LABELS = {A: 'A: local GP + peer gains', B: 'B: local cost mean + peer gains',
    D: 'D: shared/local exact GP', E: 'E: independent exact GP'}


def paired_comparisons(trials):
    aliases = dict(zip(METHODS, old.METHODS))
    inverse = {v: k for k, v in aliases.items()}
    paired, seeds = legacy_pairs(trials.assign(method=trials.method.map(aliases)))
    for frame in (paired, seeds):
        frame['method'] = frame.method.map(inverse)
        frame['baseline'] = frame.baseline.map(inverse)
        frame['role'] = frame.role.replace({'sharing_control': 'sharing'})
    return paired, seeds


def write_report(state, trials, cells, visits, learning, refs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out, cfg = state['out'], state['cfg']
    final = cfg['fleet_budgets'][-1]
    paired, seed_pairs = paired_comparisons(trials)
    summary = trials.groupby(['method', 'fleet_budget'], sort=False).agg(
        client_runs=('client', 'size'), seeds=('seed', 'nunique'),
        context_success_percent=('context_success', lambda v: 100 * v.mean()),
        validation_eligible_coverage_percent=('validation_eligible_context_coverage', lambda v: 100 * v.mean()),
        complete_eligible_policies=('complete_eligible_on_encountered', 'sum'),
        failed_paid_requests=('ineligible_queries', 'sum'),
        conditional_mean_J=('J', 'mean'), reference_coverage=('reference_coverage', 'mean')).reset_index()
    metadata = pd.DataFrame(cfg['context_metadata'])[['id', 'speed', 'frequency']]
    context = cells.merge(metadata, left_on='context', right_on='id', validate='many_to_one')
    frequency_units = context.groupby(['method', 'seed', 'client', 'fleet_budget', 'frequency'], sort=False).agg(
        context_success=('success', 'mean'), encountered_contexts=('success', 'size'),
        validation_eligible_coverage=('eligible', 'mean')).reset_index()
    frequency = frequency_units.groupby(['method', 'fleet_budget', 'frequency'], sort=False).agg(
        context_success_percent=('context_success', lambda v: 100 * v.mean()),
        encountered_contexts=('encountered_contexts', 'sum'),
        validation_eligible_coverage_percent=('validation_eligible_coverage', lambda v: 100 * v.mean())).reset_index()
    modes = visits.groupby(['method', 'seed', 'mode'], sort=False).agg(requests=('eligible', 'size'),
        eligible=('eligible', 'sum'), new_incumbents=('became_incumbent', 'sum')).reset_index()
    modes['ineligible'] = modes.requests - modes.eligible
    per_seed = trials[trials.fleet_budget == final].groupby(['method', 'seed'], sort=False).agg(
        context_success_percent=('context_success', lambda v: 100 * v.mean()),
        complete_eligible_policies=('complete_eligible_on_encountered', 'sum'),
        failed_paid_requests=('ineligible_queries', 'sum')).reset_index()
    frames = dict(budget_trials=trials, budget_contexts=cells, paid_visits=visits, learning_diagnostics=learning,
        online_summary=summary, paired_comparisons=paired, per_seed_comparisons=seed_pairs,
        frequency_units=frequency_units, frequency_outcomes=frequency, query_modes=modes,
        per_seed_outcomes=per_seed, visit_summary=visit_tables(visits),
        empirical_references=pd.DataFrame([dict(context=c, **row) for c, row in refs.items()]))
    for name, frame in frames.items():
        frame.to_csv(out / (name + '.csv'), index=False)
    outcomes = summary[summary.fleet_budget == final].set_index('method')
    contrasts = paired[paired.fleet_budget == final].set_index('role')
    readout = dict(confirmation_claim=False, mode=state['specification']['mode'],
        primary_metric='Encountered-context near-reference success, equal weights over clients and seeds.',
        success_above_A_B_E=bool((contrasts.success_delta_pp > 0).all()),
        positive_effect_in_every_seed_against_A_B_E=bool((contrasts.positive_seeds == len(state['jobs'])).all()),
        no_more_failed_requests_than_A_B_E=bool(all(outcomes.loc[D, 'failed_paid_requests'] <= outcomes.loc[b, 'failed_paid_requests'] for b in (A, B, E))),
        no_fewer_complete_policies_than_A_B_E=bool(all(outcomes.loc[D, 'complete_eligible_policies'] >= outcomes.loc[b, 'complete_eligible_policies'] for b in (A, B, E))),
        common_complete_policy_cost_not_higher=bool((contrasts.common_eligible_pairs > 0).all() and (contrasts.paired_J_change_percent <= 0).all()),
        next_step='Audit and consolidate these outcomes and their tradeoffs before deciding whether the evidence is sufficient for communication work.')
    (out / 'pilot_readout.json').write_text(json.dumps(readout, indent=2) + '\n', encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for method, color in zip(METHODS, ['#66748a', '#aa7935', '#007d75', '#7864ad']):
        rows = summary[summary.method == method]
        axes[0].plot(rows.fleet_budget, rows.context_success_percent, marker='o', color=color, label=LABELS[method])
    axes[0].set(xlabel='Paid fleet requests per arm', ylabel='Encountered-context success [%]', ylim=(-2, 102))
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=.2)
    matrix = frequency[frequency.fleet_budget == final].pivot(index='method', columns='frequency', values='context_success_percent').loc[METHODS]
    axes[1].imshow(matrix, vmin=0, vmax=100, cmap='YlGnBu', aspect='auto')
    axes[1].set_xticks(range(len(matrix.columns)), [f'{f:g} Hz' for f in matrix.columns])
    axes[1].set_yticks(range(4), ['A', 'B', 'D', 'E'])
    axes[1].set(title='Final success by frequency')
    for i, row in enumerate(matrix.to_numpy()):
        for j, value in enumerate(row):
            axes[1].text(j, i, f'{value:.1f}%', ha='center', va='center', color='white' if value > 65 else '#203747')
    fig.savefig(out / 'budget_and_frequency.png', dpi=150)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120)
    plt.close(fig)
    def table(frame):
        frame = frame.copy()
        for col in ('method', 'baseline'):
            if col in frame:
                frame[col] = frame[col].map(LABELS)
        return '<div class="table">' + frame.to_html(index=False, float_format=lambda x: f'{x:.4g}', na_rep='unavailable') + '</div>'
    title = 'Denser K_y-extended online A/B/D/E comparison'
    if state['specification']['mode'] == 'smoke':
        title = 'Dense-gain SMOKE: software verification only'
    html = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    html += '<title>' + title + '</title><style>body{font:16px/1.6 system-ui;max-width:1250px;margin:35px auto;padding:0 22px;color:#203747}'
    html += 'table{border-collapse:collapse;font-size:12px}td,th{padding:8px;border:1px solid #d0dce2}.table{overflow-x:auto;margin:18px 0}img{max-width:100%}</style></head><body>'
    html += '<h1>' + title + '</h1><p>Question: does shared cost information beat A and then B, and add value over matched independent E, '
    html += 'when the public controller pool is denser and the lateral-error gain upper bound is genuinely extended?</p>'
    html += f'<p>{len(state["pool"]):,} public gains; upper bounds [0.15, 1, 1]. Six speeds, three frequencies, '
    html += f'{len(state["clients"])} homogeneous clients, {cfg["queries_per_client"]} paid requests per client/arm. '
    html += 'One paid factory test, the same paid peer-first rule, empty initial histories and the same local eligibility penalty. A/B beta=2; D/E beta=0.5.</p>'
    html += '<p>D/E use the exact target Matern covariance and the same shared/local Gaussian information-message model. '
    html += 'The active-input dictionary retains the unobserved conditional GP variance. No rank-256 or full-public-grid kernel approximation is used. '
    html += 'B keeps its original 65-coefficient local mean. This is a representation and search-domain follow-up, not a communication result.</p>'
    html += '<h2>Final matched comparisons</h2>' + table(paired[paired.fleet_budget == final])
    html += '<p>Success effects are percentage-point changes. Paired cost effects use only common complete eligible policies; '
    html += 'their denominators are explicit. A positive success result does not erase a failure or coverage tradeoff.</p>'
    html += '<h2>Final outcomes</h2>' + table(summary[summary.fleet_budget == final])
    html += '<img alt="Budget and frequency outcomes" src="data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode() + '">'
    html += '<h2>Frequency outcomes</h2>' + table(frequency[frequency.fleet_budget == final])
    html += '<h2>Per-seed outcomes</h2>' + table(per_seed)
    html += '<h2>Failed requests and allocation</h2>' + table(modes)
    html += '<h2>Descriptive checks</h2>' + table(pd.DataFrame([dict(criterion=k, met=v) for k, v in readout.items() if isinstance(v, bool)]))
    html += '<p>Three programme/noise seeds on one plant are exploratory evidence. No statistical confirmation or vehicle-population claim. '
    html += 'Success averages encountered contexts within clients, then clients and seeds equally; unvisited contexts are separate.</p>'
    html += '<p>Archived two-repeat calibration labels select the reference only. Every paid controller and reference receives fresh common validation. '
    html += 'Reference labels never enter learners. The 5% target remains a finite-pool reference criterion.</p>'
    html += '<h2>Next decision</h2><p>' + escape(readout['next_step']) + '</p>'
    html += '<p>Signature: <code>' + escape(state['signature']) + '</code>.</p></body></html>'
    (out / 'report.html').write_text(html, encoding='utf-8')
    return frames
