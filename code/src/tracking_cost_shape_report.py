"""Descriptive reporting for the final cost-prior attribution check."""
import base64
import io
import json

import pandas as pd

from .tracking_budget_report import summarize
from .tracking_cost_shape import ARMS, paired_effects


LABELS = {'catalogue': 'Fixed catalogue', 'no_extra_prior': 'No additional prior',
          'context_level': 'Context cost level', 'full_cost_prior': 'Full cost prior'}


def write_report(out, frame, contexts, controls, cfg, mode):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = summarize(frame)
    trials, effects, per_vehicle = paired_effects(frame)
    summary.to_csv(out/'cost_shape_summary.csv', index=False)
    trials.to_csv(out/'per_trial_effects.csv', index=False)
    effects.to_csv(out/'paired_effects.csv', index=False)
    per_vehicle.to_csv(out/'per_vehicle_effects.csv', index=False)
    contexts.groupby(['protocol', 'target', 'context', 'method', 'budget']).agg(
        success_rate=('success', 'mean'), feasible_rate=('eligible', 'mean'),
        conditional_relative_gap=('relative_gap', 'mean')).reset_index().to_csv(
            out/'per_context_summary.csv', index=False)
    frame.groupby(['protocol', 'target', 'method', 'budget']).agg(
        context_success_rate=('success_fraction', 'mean'),
        all_contexts_success_rate=('joint_success', 'mean'),
        feasible_policy_rate=('fully_feasible', 'mean'),
        mean_failed_queries=('failed_queries', 'mean'),
        conditional_mean_J=('J', 'mean')).reset_index().to_csv(out/'per_vehicle_summary.csv', index=False)

    fig, axes = plt.subplots(len(cfg['protocols']), 2, figsize=(11, 3.6*len(cfg['protocols'])),
                             squeeze=False, constrained_layout=True)
    colors = {'catalogue': '#333333', 'no_extra_prior': '#888888',
              'context_level': '#286fa1', 'full_cost_prior': '#287c50'}
    markers = {'catalogue': 's', 'no_extra_prior': 'x', 'context_level': '^', 'full_cost_prior': 'o'}
    for row, protocol in enumerate(cfg['protocols']):
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
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1.065), ncol=2, fontsize=9)
    fig.savefig(out/'cost_shape_curves.png', dpi=150, bbox_inches='tight')
    stream = io.BytesIO(); fig.savefig(stream, format='png', dpi=120, bbox_inches='tight'); plt.close(fig)
    encoded = base64.b64encode(stream.getvalue()).decode()

    def records(df):
        return df.astype(object).where(pd.notna(df), None).to_dict('records')
    payload = dict(mode=mode, prior_modes=ARMS, prior_definition=cfg['prior_definition'],
                   historical_feasibility_prior=False, summary=records(summary), effects=records(effects),
                   previous_control_action_match_fraction=float(controls.action_trace_match.mean()),
                   next_step='Consolidate the complete experimental evidence before choosing further experiments.')
    (out/'summary.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n', encoding='utf-8')

    def table(df):
        return '<div class="table">'+df.to_html(index=False, float_format=lambda x: f'{x:.4g}',
                                                 na_rep='unavailable', escape=True)+'</div>'
    last = effects[effects.budget == max(cfg['budgets'])]
    audit = controls.groupby('method').agg(trials=('seed', 'count'),
        action_match_fraction=('action_trace_match', 'mean'),
        recommendation_match_fraction=('recommendation_trace_match', 'mean')).reset_index()
    title = 'SMOKE: verification only' if mode == 'smoke' else 'Does historical cost shape help beyond its level?'
    html = '''<!doctype html><html><head><meta charset="utf-8"><title>Historical cost level and shape</title>
<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#192736}
table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border:1px solid #ccd5df}
.table{overflow-x:auto}img{max-width:100%}.note{background:#edf3f7;padding:16px}</style></head><body>'''
    html += '<h1>'+title+'</h1><p class="note">Three paired BO arms share the source-derived catalogue, '
    html += 'six initial target tests, frozen source histories, target learning, GP settings and rollout budgets. '
    html += 'Only the historical log-cost prior mean changes. Historical feasibility rates are neutral (.5); '
    html += 'target feasibility learning remains active in every BO arm.</p>'
    html += '<p>No additional prior: zero log-cost mean. Context level: the uniform average of the frozen '
    html += 'historical GP mean across all 71 gains in each context, repeated for every gain in that context. '
    html += 'Full prior: the unchanged gain-dependent historical mean. The context-level arm preserves each '
    html += 'context\'s average log-cost exactly. This is a geometric cost level, not an arithmetic mean of J. '
    html += 'No target data or validation outcome enters that average.</p>'
    html += '<img alt="Cost prior comparison" src="data:image/png;base64,'+encoded+'">'
    html += '<h2>Budgeted outcomes</h2>'+table(summary)
    html += '<p>Success uses the unchanged 5% finite-pool reference threshold on held-out noise. '
    html += 'Failed or missing recommendations remain unsuccessful. The fixed catalogue uses six queries '
    html += 'at every allowance; BO budgets are total target rollouts per vehicle across six contexts.</p>'
    html += '<h2>Paired effects at the largest budget</h2>'+table(last)
    html += '<p>Level minus none measures the effect of context-specific centering. Full minus level measures '
    html += 'the additional effect of gain dependence while preserving each context\'s mean. Full minus none '
    html += 'is the complete cost-prior effect. Higher success differences are better; lower J and failure '
    html += 'differences are better. A success-rate difference of .05 is five percentage points.</p>'
    html += '<p>Success and failure contrasts retain all matched trials. Each J contrast uses only pairs '
    html += 'where both final six-context policies are eligible; its own denominator is reported. Inspect '
    html += 'per-vehicle and per-context results alongside fleet averages. Similar averages alone do not '
    html += 'establish equivalence. Optimizer seeds are repeated searches on nine fixed models, not '
    html += 'independent fleets, and no pooled-row significance claim is made.</p>'
    html += '<h2>Reproduction diagnostics</h2>'+table(audit)
    html += '<p>No additional prior corresponds to the previous no-extra-prior arm; full prior corresponds '
    html += 'to the previous cost-only arm. Both controls are rerun alongside the context-level arm. '
    html += 'Old-trace agreement is diagnostic; numerical-library differences can change acquisition paths. '
    html += 'Source-level audit CSV records the preserved mean and removed variation for each fold/context.</p>'
    html += '<h2>Interpretation and stopping point</h2><p>A benefit of full over level supports useful '
    html += 'gain-dependent historical information under this optimizer. If level reproduces the benefit, '
    html += 'the evidence supports centering rather than a need for detailed historical cost shape. '
    html += 'Mixed outcomes should be described by vehicle, quality and failed-query tradeoffs.</p>'
    html += '<p>The catalogue still uses historical costs and eligibility, and source cost GPs were fitted '
    html += 'to eligible observations. This is a conditional attribution after common initialization. '
    html += 'History is data-rich, the 71-point gain pool and references are fixed, and eligibility is not '
    html += 'a stability certificate. No new physical rollouts, sparse-history study or networked federated '
    html += 'protocol is introduced.</p><p><strong>After these results, consolidate the complete evidence: '
    html += 'what was learned, which claims survive, what remains unsupported, and whether the task '
    html += 'justifies further federated controller-tuning work. Do not automatically expand the experiment '
    html += 'chain.</strong> See docs/tracking_cost_shape.md.</p></body></html>'
    (out/'report.html').write_text(html, encoding='utf-8')
    return summary
