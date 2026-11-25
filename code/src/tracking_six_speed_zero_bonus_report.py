"""Six-arm evaluation: benchmark priority, sharing effect and bonus ablation."""
import base64
from html import escape
import io

import numpy as np
import pandas as pd

from .tracking_six_speed_report import visit_tables
from .tracking_six_speed_zero_bonus import A, B, D2, E2, D0, E0, METHODS, BONUSES

LABELS = {A: 'A', B: 'B', D2: 'D256 (beta=2)', E2: 'E256 (beta=2)',
          D0: 'D256 (beta=0)', E0: 'E256 (beta=0)'}
CONTRASTS = [(D0, A, 'primary'), (D0, B, 'secondary'), (D0, E0, 'sharing_zero_bonus'),
    (D0, D2, 'shared_bonus_effect'), (E0, E2, 'local_bonus_effect'),
    (D2, E2, 'sharing_reference'), (D2, A, 'reference_A'), (D2, B, 'reference_B')]


def paired_comparisons(trials):
    comparisons, units = [], []
    for budget, frame in trials.groupby('fleet_budget', sort=True):
        for method, baseline, role in CONTRASTS:
            a = frame[frame.method == method].set_index(['seed', 'client']).sort_index()
            b = frame[frame.method == baseline].set_index(['seed', 'client']).sort_index()
            if (not a.index.is_unique or not b.index.is_unique or not a.index.equals(b.index) or
                    not a.encountered_contexts.equals(b.encountered_contexts)):
                raise RuntimeError('Counterfactual client/context denominators differ.')
            delta = a.context_success - b.context_success
            effects = delta.groupby(level='seed').mean()
            common = a.complete_eligible_on_encountered & b.complete_eligible_on_encountered
            ma = float(a.loc[common, 'J'].mean()) if common.any() else None
            mb = float(b.loc[common, 'J'].mean()) if common.any() else None
            comparisons.append(dict(fleet_budget=int(budget), method=method, baseline=baseline, role=role,
                programme_seeds=len(effects), matched_client_runs=len(a),
                success_delta_pp=100 * float(effects.mean()),
                positive_seeds=int((effects > 1e-12).sum()), negative_seeds=int((effects < -1e-12).sum()),
                tied_seeds=int((abs(effects) <= 1e-12).sum()), worst_seed_delta_pp=100 * float(effects.min()),
                eligible_method=int(a.complete_eligible_on_encountered.sum()),
                eligible_baseline=int(b.complete_eligible_on_encountered.sum()), common_eligible_pairs=int(common.sum()),
                paired_mean_J_method=ma, paired_mean_J_baseline=mb,
                paired_J_change_percent=100 * (ma / mb - 1) if ma is not None else None))
            units.extend(dict(fleet_budget=int(budget), method=method, baseline=baseline, role=role,
                seed=int(seed), success_delta_pp=100 * float(effect)) for seed, effect in effects.items())
    return pd.DataFrame(comparisons), pd.DataFrame(units)


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
        validation_eligible_context_coverage_percent=('validation_eligible_context_coverage', lambda x: 100 * x.mean()),
        complete_eligible_on_encountered=('complete_eligible_on_encountered', 'sum'),
        complete_eligible_on_full_grid=('complete_eligible_on_full_grid', 'sum'),
        conditional_mean_J=('J', 'mean'), mean_encountered_contexts=('encountered_contexts', 'mean'),
        mean_unvisited_contexts=('unvisited_contexts', 'mean'), mean_ineligible_requests=('ineligible_queries', 'mean'),
        reference_coverage=('reference_coverage', 'mean')).reset_index()
    counts = cells.groupby(['method', 'fleet_budget'], sort=False).agg(
        encountered_context_outcomes=('success', 'size'), successes=('success', 'sum'),
        pooled_context_success_percent=('success', lambda x: 100 * x.mean())).reset_index()
    summary = summary.merge(counts, on=['method', 'fleet_budget'], validate='one_to_one')
    summary['confidence_multiplier'] = summary.method.map(BONUSES)
    modes = visits.groupby(['method', 'seed', 'mode'], sort=False).agg(
        requests=('eligible', 'size'), online_eligible=('eligible', 'sum'),
        new_incumbents=('became_incumbent', 'sum')).reset_index()
    modes['online_ineligible'] = modes.requests - modes.online_eligible
    context = cells[cells.fleet_budget == final].groupby(['method', 'context'], sort=False).agg(
        encounters=('success', 'size'), success_percent=('success', lambda x: 100 * x.mean()),
        eligible=('eligible', 'sum'), mean_local_visits=('local_visits', 'mean')).reset_index()
    visit_summary = visit_tables(visits)
    frames = dict(budget_trials=trials, budget_contexts=cells, paid_visits=visits,
        learning_diagnostics=learning, online_summary=summary, paired_comparisons=paired,
        per_seed_comparisons=units, visit_summary=visit_summary, query_modes=modes,
        final_context_outcomes=context,
        empirical_references=pd.DataFrame([dict(context=c, **row) for c, row in refs.items()]))
    for name, frame in frames.items():
        frame.to_csv(out / (name + '.csv'), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for method, color in zip(METHODS, ['#66748a', '#a77a39', '#007d75', '#7362ad', '#007d75', '#7362ad']):
        data = summary[summary.method == method]
        style = '-' if method in (D0, E0, A, B) else '--'
        for ax, column in zip(axes, ['context_success_percent', 'validation_eligible_context_coverage_percent']):
            ax.plot(data.fleet_budget, data[column], marker='o', linestyle=style, color=color, label=LABELS[method])
    axes[0].set(ylabel='Success over encountered contexts [%]')
    axes[1].set(ylabel='Eligible recommendations / encountered contexts [%]')
    for ax in axes:
        ax.set(xlabel='Paid fleet requests per arm', ylim=(-2, 102), xticks=cfg['fleet_budgets'])
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8, ncol=2)
    fig.savefig(out / 'budget_curves.png', dpi=160)
    image = io.BytesIO()
    fig.savefig(image, format='png', dpi=120)
    plt.close(fig)
    def table(frame):
        frame = frame.copy()
        for col in ('method', 'baseline'):
            if col in frame:
                frame[col] = frame[col].map(LABELS)
        return '<div class="table">' + frame.to_html(index=False, float_format=lambda x: f'{x:.4g}', na_rep='unavailable') + '</div>'
    title = 'Six-speed matched zero-bonus pilot'
    if state['specification']['mode'] == 'smoke':
        title = 'Six-speed zero-bonus SMOKE: software verification only'
    html = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    html += '<title>' + title + '</title><style>body{font:16px/1.6 system-ui;max-width:1250px;margin:35px auto;padding:0 22px;color:#203747}'
    html += 'table{border-collapse:collapse;font-size:12px}td,th{padding:8px;border:1px solid #d0dce2}.table{overflow-x:auto;margin:18px 0}'
    html += 'h2{margin-top:35px}img{max-width:100%}.note{background:#edf6f2;padding:18px;border-left:4px solid #007d75}</style></head><body>'
    html += '<h1>' + title + '</h1><p>Goal: beat A, then B, and measure whether sharing the contextual cost model adds value '
    html += 'when D256 and E256 both omit the cost uncertainty bonus. Fresh programme/noise seeds; cold histories; '
    html += 'six discrete speeds 20–30 m/s; three frequencies; one externally assigned speed/frequency context per trial.</p>'
    design = pd.DataFrame([dict(method=m, bonus=BONUSES[m], cost_information=(
        'Local adaptive GP' if m == A else 'Local RFF mean + residual GP' if m == B else
        'Shared/local spectral Gaussian model' if m in (D2, D0) else 'Local spectral Gaussian model')) for m in METHODS])
    html += '<h2>Matched methods</h2>' + table(design)
    html += '<p>All methods use the same paid exact-context peer-first rule, one charged factory test, a 24-test local cap '
    html += '(8 in smoke), and distinct untested gain/context pairs. Peer catalogues and histories evolve independently per method. '
    html += 'The zero-bonus score is predicted log cost plus the unchanged negative-log eligibility penalty: '
    html += '<code>mean(log J) - lambda * log(max(p_eligible, floor))</code>. The beta=2 controls additionally subtract '
    html += '<code>2 * sqrt(latent variance)</code>. Full Bayesian covariance, evidence mixture and communication accounting are retained.</p>'
    html += '<p class="note"><strong>Exploratory pilot.</strong> Three seeds on one homogeneous compact plant. '
    html += 'Primary success averages encountered contexts within each vehicle, then vehicles and seeds equally. '
    html += 'Failed or missing recommendations remain failures; unvisited contexts are separate. A success is a held-out cost '
    html += f'within {cfg["relative_tolerance"]:.0%} of the independent calibration-selected reference. No confidence interval or confirmation claim.</p>'
    html += '<h2>Primary and secondary: zero-bonus D256 versus A and B</h2>'
    html += table(paired[(paired.fleet_budget == final) & paired.role.isin(['primary', 'secondary'])])
    html += '<h2>Sharing with zero bonus and the effect of removing the bonus</h2>'
    html += table(paired[(paired.fleet_budget == final) & paired.role.isin(
        ['sharing_zero_bonus', 'shared_bonus_effect', 'local_bonus_effect'])])
    html += '<p>Success differences are method minus baseline in percentage points; positive favors the method. '
    html += 'Cost changes are negative when better, calculated only over common complete eligible client policies. '
    html += 'Eligibility counts and common-pair denominators are shown to expose selection effects. Cost sharing at beta=0 '
    html += 'is supported only if D256 improves over its matched E256 control without an adverse eligibility tradeoff.</p>'
    html += '<h2>Final success and eligibility</h2>' + table(summary[summary.fleet_budget == final])
    html += '<img alt="Success and recommendation eligibility by paid budget" src="data:image/png;base64,' + base64.b64encode(image.getvalue()).decode() + '">'
    html += '<p>The encountered-context denominator grows with budget. These are not fixed-grid learning curves. '
    html += 'Pooled context success is also exported, with a different weighting from the primary metric.</p>'
    html += '<h2>First and subsequent visits</h2>' + table(visit_summary)
    html += '<p>Requested-controller success requires the paid test and all held-out repeats to be eligible. Incumbent success '
    html += 'evaluates the locally tested recommendation after that visit. Factory tests are included unless explicitly excluded.</p>'
    html += '<h2>Peer tests and BO failures</h2>' + table(modes)
    html += '<h2>All final comparisons</h2>' + table(paired[paired.fleet_budget == final])
    html += '<h2>Per-seed effects</h2>' + table(units[units.fleet_budget == final])
    html += '<h2>Final context outcomes</h2>' + table(context)
    accuracy = state['specification']['basis_accuracy']
    html += '<h2>Reproducibility and scope</h2><p>The public feature matrix is exactly the frozen rank-256 matrix '
    html += '(1,278 inputs, 257 coefficients). Relative kernel Frobenius error '
    html += f'{accuracy["relative_kernel_Frobenius_error"]:.3%}; maximum relative diagonal error '
    html += f'{accuracy["max_relative_diagonal_error"]:.3%}. The earlier 2% / 5% targets are unchanged; the slight diagonal-target miss is retained.</p>'
    html += '<p>Every paid decision is replayed before evaluator labels are opened. A/B restore logged GP fit states; '
    html += 'cross-platform fresh optimizer identity is not claimed. The existing bounds-recovery path is retained. '
    html += 'Calibration and validation use a fresh common bank, including every gain paid for by any arm. '
    html += 'No old histories or evaluator labels enter a learner. Cached numerical work does not reduce paid query counts.</p>'
    html += '<p>Zero bonus does not remove the Bayesian model or its messages. This experiment does not measure a communication improvement. '
    html += 'A/B and beta=2 spectral controls are rerun on the same fresh programmes as the zero-bonus pair; old pilot scores are not mixed into comparisons.</p>'
    html += '<p>Configuration: <code>' + escape(cfg['tag']) + '</code>; signature <code>' + escape(state['signature']) + '</code>.</p></body></html>'
    (out / 'report.html').write_text(html, encoding='utf-8')
    return frames
