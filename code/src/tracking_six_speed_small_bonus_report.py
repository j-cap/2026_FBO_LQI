"""Fixed small-bonus readout and evidence bundle for the next consolidation."""
import base64
from html import escape
import io
import json

import numpy as np
import pandas as pd

from . import tracking_six_speed_zero_bonus as zero
from .tracking_six_speed_zero_bonus_report import paired_comparisons as previous_comparisons
from .tracking_six_speed_report import visit_tables
from .tracking_six_speed_small_bonus import A, B, D0, E0, D_SMALL, E_SMALL, METHODS, BONUSES

LABELS = {A: 'A', B: 'B', D0: 'D256 (beta=0)', E0: 'E256 (beta=0)',
          D_SMALL: 'D256 (beta=0.5)', E_SMALL: 'E256 (beta=0.5)'}
CONTRASTS = [(D_SMALL, A, 'primary'), (D_SMALL, B, 'secondary'),
    (D_SMALL, E_SMALL, 'sharing_small_bonus'), (D_SMALL, D0, 'shared_exploration_effect'),
    (E_SMALL, E0, 'local_exploration_effect'), (D0, E0, 'sharing_zero_bonus'),
    (D0, A, 'zero_A'), (D0, B, 'zero_B')]


def paired_comparisons(trials):
    # Reuse the audited paired arithmetic with a bijective, evaluation-only
    # relabeling. No learner histories, method identities or settings are aliased.
    aliases = {A: A, B: B, D0: zero.D2, E0: zero.E2, D_SMALL: zero.D0, E_SMALL: zero.E0}
    inverse = {v: k for k, v in aliases.items()}
    roles = dict(primary='primary', secondary='secondary', sharing_zero_bonus='sharing_small_bonus',
        shared_bonus_effect='shared_exploration_effect', local_bonus_effect='local_exploration_effect',
        sharing_reference='sharing_zero_bonus', reference_A='zero_A', reference_B='zero_B')
    paired, units = previous_comparisons(trials.assign(method=trials.method.map(aliases)))
    for frame in (paired, units):
        for col in ('method', 'baseline'):
            frame[col] = frame[col].map(inverse)
        frame['role'] = frame.role.map(roles)
    return paired, units


def supplemental_tables(state, trials, cells, visits):
    metadata = pd.DataFrame(state['cfg']['context_metadata'])[['id', 'speed', 'frequency']]
    context = cells.merge(metadata, left_on='context', right_on='id', validate='many_to_one')
    frequency_trials = context.groupby(['method', 'seed', 'fleet_budget', 'client', 'frequency'], sort=False).agg(
        encountered_contexts=('success', 'size'), context_success=('success', 'mean'),
        eligible_context_coverage=('eligible', 'mean')).reset_index()
    frequency = frequency_trials.groupby(['method', 'fleet_budget', 'frequency'], sort=False).agg(
        client_runs=('client', 'size'), seeds=('seed', 'nunique'),
        context_success_percent=('context_success', lambda x: 100 * x.mean()),
        eligible_context_coverage_percent=('eligible_context_coverage', lambda x: 100 * x.mean())).reset_index()
    counts = context.groupby(['method', 'fleet_budget', 'frequency'], sort=False).agg(
        encountered_contexts=('success', 'size'), successes=('success', 'sum'),
        pooled_context_success_percent=('success', lambda x: 100 * x.mean()),
        median_eligible_relative_gap=('relative_gap', 'median')).reset_index()
    frequency = frequency.merge(counts, on=['method', 'fleet_budget', 'frequency'], validate='one_to_one')
    discoveries = []
    for (method, seed, freq), group in visits.groupby(['method', 'seed', 'frequency'], sort=False):
        successful = group[group.requested_success].sort_values('fleet_evaluation')
        first = successful.iloc[0] if len(successful) else None
        discoveries.append(dict(method=method, seed=int(seed), frequency=float(freq),
            requests=len(group), near_reference_requests=len(successful),
            BO_near_reference_requests=int((successful['mode'] == 'feasibility_penalized_LCB').sum()),
            first_round=int(first['round']) if first is not None else None,
            first_context=first['context'] if first is not None else None,
            first_client=first['client'] if first is not None else None,
            first_candidate=first['candidate'] if first is not None else None,
            first_mode=first['mode'] if first is not None else None))
    final = state['cfg']['fleet_budgets'][-1]
    per_seed = trials[trials.fleet_budget == final].groupby(['method', 'seed'], sort=False).agg(
        context_success_percent=('context_success', lambda x: 100 * x.mean()),
        complete_eligible_policies=('complete_eligible_on_encountered', 'sum'),
        failed_paid_requests=('ineligible_queries', 'sum')).reset_index()
    return dict(frequency_trials=frequency_trials, frequency_outcomes=frequency,
        frequency_discoveries=pd.DataFrame(discoveries), per_seed_outcomes=per_seed)


def pilot_readout(summary, paired, supplements, final):
    outcomes = summary[summary.fleet_budget == final].set_index('method')
    comparisons = paired[paired.fleet_budget == final].set_index('role')
    middle = supplements['frequency_outcomes'].query('fleet_budget == @final and frequency == 0.3').set_index('method')
    discoveries = supplements['frequency_discoveries']
    discoveries = discoveries[(discoveries.method == D_SMALL) & (discoveries.frequency == .3)]
    delta_middle = float(middle.loc[D_SMALL, 'context_success_percent'] - middle.loc[D0, 'context_success_percent'])
    delta_failures = int(outcomes.loc[D_SMALL, 'failed_paid_requests'] - outcomes.loc[D0, 'failed_paid_requests'])
    delta_complete = int(outcomes.loc[D_SMALL, 'complete_eligible_on_encountered'] - outcomes.loc[D0, 'complete_eligible_on_encountered'])
    criteria = dict(
        primary_success_above_A=bool(comparisons.loc['primary', 'success_delta_pp'] > 0),
        secondary_success_above_B=bool(comparisons.loc['secondary', 'success_delta_pp'] > 0),
        sharing_success_above_matched_E=bool(comparisons.loc['sharing_small_bonus', 'success_delta_pp'] > 0),
        overall_success_above_zero_D=bool(comparisons.loc['shared_exploration_effect', 'success_delta_pp'] > 0),
        middle_frequency_success_above_zero_D=delta_middle > 0,
        middle_frequency_discovery_in_every_seed=bool(discoveries.seed.nunique() == outcomes.loc[D_SMALL, 'seeds']
            and (discoveries.near_reference_requests > 0).all()),
        failed_paid_requests_not_above_zero_D=delta_failures <= 0,
        complete_eligible_policies_not_below_zero_D=delta_complete >= 0)
    return dict(interpretation='Descriptive pilot criteria fixed before the run; no statistical confirmation claim.',
        confirmation_claim=False, criteria=criteria, all_descriptive_criteria_met=all(criteria.values()),
        clean_improvement_over_zero_D=all(criteria[k] for k in (
            'overall_success_above_zero_D', 'middle_frequency_success_above_zero_D',
            'failed_paid_requests_not_above_zero_D', 'complete_eligible_policies_not_below_zero_D')),
        middle_frequency_success_change_pp=delta_middle,
        failed_paid_request_change_vs_zero_D=delta_failures,
        complete_eligible_policy_change_vs_zero_D=delta_complete,
        complete_eligible_policy_change_vs_B=int(outcomes.loc[D_SMALL, 'complete_eligible_on_encountered'] -
            outcomes.loc[B, 'complete_eligible_on_encountered']),
        programme_seeds=int(outcomes.loc[D_SMALL, 'seeds']),
        middle_frequency_seeds_with_discovery=int((discoveries.near_reference_requests > 0).sum()),
        next_step='Consolidate this experiment with both preceding six-speed pilots and review the bigger picture before choosing another variant.')


def write_report(state, trials, cells, visits, learning, refs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out, cfg = state['out'], state['cfg']
    final = cfg['fleet_budgets'][-1]
    paired, units = paired_comparisons(trials)
    summary = trials.groupby(['method', 'fleet_budget'], sort=False).agg(
        client_runs=('client', 'size'), seeds=('seed', 'nunique'),
        context_success_percent=('context_success', lambda x: 100 * x.mean()),
        eligible_context_coverage_percent=('validation_eligible_context_coverage', lambda x: 100 * x.mean()),
        complete_eligible_on_encountered=('complete_eligible_on_encountered', 'sum'),
        complete_eligible_on_full_grid=('complete_eligible_on_full_grid', 'sum'),
        conditional_mean_J=('J', 'mean'), mean_encountered_contexts=('encountered_contexts', 'mean'),
        mean_unvisited_contexts=('unvisited_contexts', 'mean'),
        failed_paid_requests=('ineligible_queries', 'sum'), reference_coverage=('reference_coverage', 'mean')).reset_index()
    modes = visits.groupby(['method', 'seed', 'mode'], sort=False).agg(
        requests=('eligible', 'size'), online_eligible=('eligible', 'sum'),
        new_incumbents=('became_incumbent', 'sum')).reset_index()
    modes['online_ineligible'] = modes.requests - modes.online_eligible
    supplements = supplemental_tables(state, trials, cells, visits)
    readout = pilot_readout(summary, paired, supplements, final)
    frames = dict(budget_trials=trials, budget_contexts=cells, paid_visits=visits, learning_diagnostics=learning,
        online_summary=summary, paired_comparisons=paired, per_seed_comparisons=units,
        visit_summary=visit_tables(visits), query_modes=modes,
        empirical_references=pd.DataFrame([dict(context=c, **row) for c, row in refs.items()]), **supplements)
    for name, frame in frames.items():
        frame.to_csv(out / (name + '.csv'), index=False)
    (out / 'pilot_readout.json').write_text(json.dumps(readout, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for method, color in zip(METHODS, ['#66748a', '#a77a39', '#007d75', '#7362ad', '#007d75', '#7362ad']):
        rows = summary[summary.method == method]
        axes[0].plot(rows.fleet_budget, rows.context_success_percent, marker='o', color=color,
            linestyle='--' if method in (D0, E0) else '-', label=LABELS[method])
    axes[0].set(xlabel='Paid fleet requests per arm', ylabel='Context success [%]', ylim=(-2, 102), xticks=cfg['fleet_budgets'])
    axes[0].grid(alpha=.2)
    axes[0].legend(fontsize=8, ncol=2)
    frequency = supplements['frequency_outcomes'].query('fleet_budget == @final').pivot(
        index='method', columns='frequency', values='context_success_percent').loc[METHODS]
    axes[1].imshow(frequency.to_numpy(), vmin=0, vmax=100, cmap='YlGnBu', aspect='auto')
    axes[1].set_xticks(range(len(frequency.columns)), [f'{f:.1f} Hz' for f in frequency.columns])
    axes[1].set_yticks(range(len(METHODS)), [LABELS[m] for m in METHODS])
    axes[1].set(title='Final success by frequency', xlabel='Equal client/seed weights within frequency')
    for i, row in enumerate(frequency.to_numpy()):
        for j, value in enumerate(row):
            axes[1].text(j, i, f'{value:.1f}%', ha='center', va='center', color='white' if value > 65 else '#203747')
    fig.savefig(out / 'budget_and_frequency.png', dpi=160)
    image = io.BytesIO()
    fig.savefig(image, format='png', dpi=120)
    plt.close(fig)
    def table(frame):
        frame = frame.copy()
        for col in ('method', 'baseline'):
            if col in frame:
                frame[col] = frame[col].map(LABELS)
        return '<div class="table">' + frame.to_html(index=False, float_format=lambda x: f'{x:.4g}', na_rep='unavailable') + '</div>'
    title = 'Six-speed fixed small-bonus pilot'
    if state['specification']['mode'] == 'smoke':
        title = 'Six-speed small-bonus SMOKE: software verification only'
    html = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    html += '<title>' + title + '</title><style>body{font:16px/1.6 system-ui;max-width:1250px;margin:35px auto;padding:0 22px;color:#203747}'
    html += 'table{border-collapse:collapse;font-size:12px}td,th{padding:8px;border:1px solid #d0dce2}.table{overflow-x:auto;margin:18px 0}'
    html += 'h2{margin-top:35px}img{max-width:100%}.note{background:#edf6f2;padding:18px;border-left:4px solid #007d75}</style></head><body>'
    html += '<h1>' + title + '</h1><p>Question: does one fixed, restrained cost uncertainty bonus recover useful discoveries '
    html += 'without losing the zero-bonus shared model\'s failure profile? A/B retain beta=2; D/E use beta=0 or beta=0.5 '
    html += 'in every context. No frequency-specific rule, coefficient sweep or adaptive retuning is used.</p>'
    html += '<p>Six speeds, three frequencies, nine homogeneous clients, 24 paid requests each (three clients/eight requests '
    html += 'in smoke). The factory test, exact-context peer-first rule, local eligibility penalty, kernel, Bayesian messages '
    html += 'and priors are fixed. Histories and peer catalogues are separate per arm. Failed tests remain charged.</p>'
    html += '<p class="note">Exploratory evidence from three programme/noise seeds on one plant. Primary success first '
    html += 'averages encountered contexts within each client, then clients and seeds equally. Unvisited contexts are separate. '
    html += 'Missing or ineligible recommendations remain failures. No confidence interval or confirmation claim.</p>'
    html += '<h2>Primary and secondary benchmarks</h2>' + table(paired[(paired.fleet_budget == final) & paired.role.isin(['primary', 'secondary'])])
    html += '<h2>Sharing and exploration effects</h2>' + table(paired[(paired.fleet_budget == final) & paired.role.isin(
        ['sharing_small_bonus', 'shared_exploration_effect', 'local_exploration_effect'])])
    html += '<p>Success changes are method minus baseline in percentage points. Paired J changes use only common '
    html += 'complete eligible client policies; those counts remain explicit. The peer rule is common, but catalogue '
    html += 'changes caused by earlier proposals contribute to each method\'s full online outcome.</p>'
    html += '<h2>Final outcome and eligibility</h2>' + table(summary[summary.fleet_budget == final])
    html += '<img alt="Overall budget success and frequency-specific final success" src="data:image/png;base64,' + base64.b64encode(image.getvalue()).decode() + '">'
    html += '<p>The encountered-context denominator grows with budget. Frequency rates use equal client/seed weights '
    html += 'within that frequency; pooled counts and rates are exported separately.</p>'
    html += '<h2>Frequency-specific outcomes</h2>' + table(supplements['frequency_outcomes'].query('fleet_budget == @final'))
    html += '<h2>Near-reference discoveries at 0.3 Hz</h2>' + table(supplements['frequency_discoveries'].query('frequency == 0.3'))
    html += '<p>A discovery requires eligibility in the paid trial and all held-out repeats and a held-out cost within '
    html += f'{cfg["relative_tolerance"]:.0%} of the independent calibration-selected reference. Evaluator labels never guide requests.</p>'
    html += '<h2>Paid failures and allocation</h2>' + table(modes)
    html += '<h2>Per-seed outcomes</h2>' + table(supplements['per_seed_outcomes'])
    html += '<h2>Predeclared descriptive readout</h2>' + table(pd.DataFrame(
        [dict(criterion=k, met=v) for k, v in readout['criteria'].items()]))
    html += '<p>A clean improvement over zero-bonus D requires higher overall and 0.3 Hz success, no more failed paid '
    html += 'requests and no fewer complete eligible policies. A/B benchmarks, sharing and discoveries in every seed '
    html += 'are reported separately. These are descriptive checks; a valid run completes whether they pass or fail.</p>'
    html += '<h2>Consolidation next</h2><p>' + escape(readout['next_step']) + '</p>'
    html += '<p>The pinned rank-256 basis and its slight maximum-diagonal accuracy miss are retained. Gaussian covariance '
    html += 'and communication accounting remain active at both bonuses. This is not a communication or heterogeneous-fleet study. '
    html += 'All paid decisions are replayed before evaluator labels are opened; A/B restore logged GP fits, with fresh optimization prohibited.</p>'
    html += '<p>Configuration: <code>' + escape(cfg['tag']) + '</code>; signature <code>' + escape(state['signature']) + '</code>.</p></body></html>'
    (out / 'report.html').write_text(html, encoding='utf-8')
    lines = ['# Small-bonus experiment: consolidation inputs', '',
        'These are exploratory pilot outcomes. Use the complete report and per-seed tables to assess tradeoffs.', '',
        '| Method | Context success | Complete eligible policies | Failed paid requests |',
        '|---|---:|---:|---:|']
    for row in summary[summary.fleet_budget == final].itertuples():
        lines.append(f'| {LABELS[row.method]} | {row.context_success_percent:.2f}% | '
            f'{row.complete_eligible_on_encountered}/{row.client_runs} | {row.failed_paid_requests} |')
    lines += ['', 'Next: ' + readout['next_step'], '',
        'Detailed evidence: report.html, paired_comparisons.csv, per_seed_outcomes.csv, frequency_outcomes.csv,',
        'frequency_discoveries.csv, query_modes.csv, pilot_readout.json and verification.json.', '']
    (out / 'consolidation_inputs.md').write_text('\n'.join(lines), encoding='utf-8')
    return frames
