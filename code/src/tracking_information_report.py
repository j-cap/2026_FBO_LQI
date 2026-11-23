"""Descriptive paired effects for the four historical-information arms."""
import base64
import io
import json

import numpy as np
import pandas as pd

from .tracking_budget_report import summarize
from .tracking_information import ARMS, factorial_effects


LABELS = {'catalogue': 'Fixed catalogue', 'no_extra_prior': 'No extra prior (00)',
          'cost_only': 'Cost prior (10)', 'feasibility_only': 'Feasibility prior (01)',
          'both_priors': 'Both priors (11)'}


def write_report(out, frame, contexts, controls, cfg, mode):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = summarize(frame)
    trials, effects, per_vehicle = factorial_effects(frame)
    summary.to_csv(out/'information_summary.csv', index=False)
    trials.to_csv(out/'per_trial_effects.csv', index=False)
    effects.to_csv(out/'factorial_effects.csv', index=False)
    per_vehicle.to_csv(out/'per_vehicle_effects.csv', index=False)
    context_summary = contexts.groupby(['protocol', 'target', 'context', 'method', 'budget']).agg(
        success_rate=('success', 'mean'), feasible_rate=('eligible', 'mean'),
        conditional_relative_gap=('relative_gap', 'mean')).reset_index()
    context_summary.to_csv(out/'per_context_summary.csv', index=False)
    protocols = cfg['protocols']
    fig, axes = plt.subplots(len(protocols), 2, figsize=(11, 3.6*len(protocols)),
                             squeeze=False, constrained_layout=True)
    colors = {'catalogue': '#333333', 'no_extra_prior': '#888888', 'cost_only': '#c27c1e',
              'feasibility_only': '#286fa1', 'both_priors': '#287c50'}
    markers = {'catalogue': 's', 'no_extra_prior': 'x', 'cost_only': '^',
               'feasibility_only': 'v', 'both_priors': 'o'}
    for row, protocol in enumerate(protocols):
        for method in cfg['methods']:
            g = summary[(summary.protocol == protocol) & (summary.method == method)]
            style = '--' if method == 'catalogue' else '-'
            args = dict(color=colors[method], marker=markers[method], markersize=5, label=LABELS[method])
            axes[row, 0].plot(g.budget, 100*g.context_success_rate, style, **args)
            axes[row, 1].plot(g.budget, g.mean_failed_queries, style, **args)
        axes[row, 0].set(title=protocol.replace('_', ' '), ylabel='Contexts within 5% [%]',
                         xlabel='Target rollout allowance per vehicle', ylim=(-2, 102), xticks=cfg['budgets'])
        axes[row, 1].set(ylabel='Mean failed target queries', xlabel='Target rollout allowance per vehicle',
                         xticks=cfg['budgets'])
        for ax in axes[row]:
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1.065), ncol=3, fontsize=9)
    fig.savefig(out/'information_curves.png', dpi=150, bbox_inches='tight')
    stream = io.BytesIO(); fig.savefig(stream, format='png', dpi=120, bbox_inches='tight'); plt.close(fig)
    encoded = base64.b64encode(stream.getvalue()).decode()
    def records(df):
        return df.astype(object).where(pd.notna(df), None).to_dict('records')
    payload = dict(mode=mode, factor_levels={a: dict(cost_prior=c, feasibility_prior=f) for a, (c, f) in ARMS.items()},
                   summary=records(summary), effects=records(effects),
                   previous_control_action_match_fraction=float(controls.action_trace_match.mean()))
    (out/'summary.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    def table(df):
        return '<div class="table">'+df.to_html(index=False, float_format=lambda x: f'{x:.4g}',
                                                 na_rep='unavailable', escape=True)+'</div>'
    last = effects[(effects.budget == max(cfg['budgets'])) &
                   effects.metric.isin(['success_fraction', 'failed_queries', 'J']) &
                   ~effects.contrast.str.endswith('main_effect')]
    audit = controls.groupby('method').agg(trials=('seed', 'count'),
        action_match_fraction=('action_trace_match', 'mean'),
        recommendation_match_fraction=('recommendation_trace_match', 'mean')).reset_index()
    title = 'SMOKE: verification only' if mode == 'smoke' else 'Which historical information helps calibration?'
    html = '''<!doctype html><html><head><meta charset="utf-8"><title>Historical information ablation</title>
<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#192736}
table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border:1px solid #ccd5df}
.table{overflow-x:auto}img{max-width:100%}.note{background:#edf3f7;padding:16px}</style></head><body>'''
    html += '<h1>'+title+'</h1><p class="note">A paired 2×2 ablation of two additional historical '
    html += 'information channels: the objective GP prior mean and the feasibility prior used by acquisition. '
    html += 'All four BO arms share the same catalogue, six initial target observations, target-learning rules, '
    html += 'hyperparameter policy, budgets, source histories, and measurement-noise realization.</p>'
    html += '<p>00: no additional historical prior. 10: cost mean only. 01: feasibility only. '
    html += '11: both. Target objective and target feasibility learning remain active in every arm. '
    html += 'The fixed catalogue is a separate six-query benchmark.</p>'
    html += '<img alt="Information ablation curves" src="data:image/png;base64,'+encoded+'">'
    html += '<h2>Budgeted results</h2>'+table(summary)
    html += '<p>Success uses the unchanged 5% finite-pool reference threshold on held-out noise. '
    html += 'Missing or ineligible recommendations remain unsuccessful. Costs conditional on feasibility '
    html += 'must not be compared across different subsets. Failed tests refer to developmental rollout '
    html += 'constraints, not a certified safety probability.</p>'
    html += '<h2>Paired contrasts at the largest budget</h2>'+table(last)
    html += '<p>Cost with feasibility off = 10−00; cost with feasibility on = 11−01. '
    html += 'Feasibility with cost off = 01−00; feasibility with cost on = 11−10. '
    html += 'Interaction = 11−10−01+00. Positive success effects are better; negative failed-query and J effects '
    html += 'are better. Rate differences are fractions: .05 means five percentage points. '
    html += 'Main effects in the CSV average the two conditional effects.</p>'
    html += '<p>Inspect conditional effects first. Interaction signs alone do not establish complementarity, '
    html += 'especially for bounded success rates and failure counts with a zero floor. J contrasts use '
    html += 'only trials where every included arm has a feasible six-context policy; matched and eligible '
    html += 'trial counts are retained. There are no pooled-sample significance tests.</p>'
    html += '<h2>Reproduction diagnostics</h2>'+table(audit)
    html += '<p>00 and 11 reuse the unchanged optimizer and correspond to catalogue-initialized local BO '
    html += 'and pooled BO in the previous campaign. All four arms are rerun together under the current '
    html += 'runtime. Exact agreement with old traces is an audit, not a condition for selecting results; '
    html += 'numerical-library differences can change acquisition paths. Unit tests verify control equivalence '
    html += 'under one runtime.</p>'
    html += '<h2>Interpretation</h2><p>The catalogue itself still uses historical costs and feasibility, '
    html += 'and the historical cost GP was fitted to eligible source objectives. This test identifies the '
    html += 'usefulness of the two algorithmic channels after a common initialization, not the value of '
    html += 'raw cost data without any historical eligibility information. The cost channel includes both '
    html += 'the overall historical mean level and its gain-dependent variation.</p>'
    html += '<p>The full source archive, nine deterministic vehicles, two target-noise realizations, and '
    html += '71-candidate pool remain fixed. Source-history sparsity, new physical rollouts and new reference '
    html += 'shapes are not tested here. Twenty optimizer seeds are not twenty independent fleets. '
    html += 'See docs/tracking_information.md for the frozen attribution protocol.</p></body></html>'
    (out/'report.html').write_text(html, encoding='utf-8')
    return summary
