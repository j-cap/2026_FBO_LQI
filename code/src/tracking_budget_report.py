"""Presentation and descriptive summaries; does not select BO queries."""
import base64
import io
import json

import numpy as np
import pandas as pd


LABELS = {'random': 'Random search', 'independent_bo': 'Independent BO',
          'catalogue': 'Fixed source catalogue', 'catalogue_warmstart_bo': 'Catalogue + local BO',
          'pooled_bo': 'Fleet-prior BO'}


def summarize(frame):
    rows = []
    for (protocol, method, budget), g in frame.groupby(['protocol', 'method', 'budget'], sort=False):
        eligible = g[g.fully_feasible]
        rows.append(dict(protocol=protocol, method=method, budget=int(budget),
                         trials=len(g), vehicles=g.target.nunique(),
                         context_success_rate=float(g.success_fraction.mean()),
                         all_contexts_success_rate=float(g.joint_success.mean()),
                         fully_feasible_policy_rate=float(g.fully_feasible.mean()),
                         mean_J_all_trials=float(g.J.mean()) if g.fully_feasible.all() else None,
                         mean_J_on_fully_feasible_trials=float(eligible.J.mean()) if len(eligible) else None,
                         fully_feasible_trials=len(eligible),
                         mean_failed_queries=float(g.failed_queries.mean()),
                         mean_unresolved_queries=float(g.unresolved_queries.mean()),
                         mean_actual_target_rollouts=float(g.actual_rollouts.mean()),
                         restricted_mean_evals_to_joint_success=float(g.restricted_evals_to_success.mean())))
    return pd.DataFrame(rows)


def paired_comparisons(frame):
    rows = []
    comparisons = [('independent_bo', 'random'), ('catalogue', 'independent_bo'),
                   ('catalogue_warmstart_bo', 'independent_bo'),
                   ('catalogue_warmstart_bo', 'catalogue'), ('pooled_bo', 'catalogue_warmstart_bo'),
                   ('pooled_bo', 'catalogue')]
    for (protocol, budget), block in frame.groupby(['protocol', 'budget']):
        for method, baseline in comparisons:
            if method not in set(block.method) or baseline not in set(block.method):
                continue
            a = block[block.method == method].set_index(['target', 'seed'])
            b = block[block.method == baseline].set_index(['target', 'seed'])
            merged = a.join(b, lsuffix='_method', rsuffix='_baseline', how='inner')
            if len(merged) != len(a) or len(merged) != len(b):
                raise ValueError('Paired comparison has unmatched trials.')
            good = merged[merged.fully_feasible_method & merged.fully_feasible_baseline]
            rows.append(dict(protocol=protocol, budget=int(budget), method=method, baseline=baseline,
                             paired_trials=len(merged),
                             context_success_delta=float((merged.success_fraction_method-
                                                          merged.success_fraction_baseline).mean()),
                             failed_queries_delta=float((merged.failed_queries_method-
                                                         merged.failed_queries_baseline).mean()),
                             jointly_feasible_pairs=len(good),
                             mean_relative_J_improvement_on_jointly_feasible_pairs=float(
                                 (1-good.J_method/good.J_baseline).mean()) if len(good) else None))
    return pd.DataFrame(rows)


def write_report(out, frame, contexts, cfg, mode):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = summarize(frame)
    paired = paired_comparisons(frame)
    summary.to_csv(out/'budget_summary.csv', index=False)
    paired.to_csv(out/'paired_comparisons.csv', index=False)
    per_vehicle = frame.groupby(['protocol', 'target', 'method', 'budget']).agg(
        context_success_rate=('success_fraction', 'mean'),
        all_contexts_success_rate=('joint_success', 'mean'),
        feasible_policy_rate=('fully_feasible', 'mean'),
        failed_queries=('failed_queries', 'mean'),
        actual_rollouts=('actual_rollouts', 'mean')).reset_index()
    per_vehicle.to_csv(out/'per_vehicle_summary.csv', index=False)
    protocols = cfg['protocols']
    fig, axes = plt.subplots(len(protocols), 2, figsize=(11, 3.7*len(protocols)),
                             squeeze=False, constrained_layout=True)
    colors = ['#999999', '#2564a3', '#292929', '#d7821b', '#287d57']
    for row, protocol in enumerate(protocols):
        for method, color in zip(cfg['methods'], colors):
            g = summary[(summary.protocol == protocol) & (summary.method == method)]
            style = '--' if method == 'catalogue' else '-'
            axes[row, 0].plot(g.budget, 100*g.context_success_rate, style, marker='o',
                              color=color, label=LABELS[method])
            axes[row, 1].plot(g.budget, g.mean_failed_queries, style, marker='o', color=color)
        axes[row, 0].set(title=protocol.replace('_', ' '), ylabel='Contexts within 5% [%]',
                         xlabel='Target rollout allowance per vehicle', ylim=(-2, 102),
                         xticks=cfg['budgets'])
        axes[row, 1].set(ylabel='Mean failed target rollouts', xlabel='Target rollout allowance per vehicle',
                         xticks=cfg['budgets'])
        for ax in axes[row]:
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, loc='upper center',
               bbox_to_anchor=(.5, 1.065), ncol=3)
    fig.savefig(out/'budget_curves.png', dpi=150, bbox_inches='tight')
    stream = io.BytesIO(); fig.savefig(stream, format='png', dpi=120, bbox_inches='tight'); plt.close(fig)
    encoded = base64.b64encode(stream.getvalue()).decode()
    def table(df):
        return df.to_html(index=False, float_format=lambda x: f'{x:.4g}', na_rep='unavailable', escape=True)
    def clean_records(df):
        return df.astype(object).where(pd.notna(df), None).to_dict('records')
    (out/'summary.json').write_text(json.dumps(dict(mode=mode, retrospective_replay=True,
        budgets=cfg['budgets'], summary=clean_records(summary), paired=clean_records(paired)),
        indent=2, allow_nan=False)+'\n', encoding='utf-8')
    title = 'SMOKE: software verification only' if mode == 'smoke' else 'Budgeted trajectory-tracking calibration'
    html = '''<!doctype html><html><head><meta charset="utf-8"><title>Tracking budget comparison</title>
<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#192736}
table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border:1px solid #ccd5df}
.table{overflow-x:auto}img{max-width:100%}.note{background:#eef3f7;padding:16px}</style></head><body>'''
    html += '<h1>'+title+'</h1><p class="note">Retrospective finite-pool replay: all physics rollouts already '
    html += 'exist. Learners reveal only requested target calibration observations. Validation noise and dense '
    html += 'reference values are read after query traces have been frozen. This is a historical-transfer '
    html += 'benchmark; it does not implement networked federation or a privacy guarantee.</p>'
    html += '<p>Budgets count individual 25-second target rollouts across all six reference contexts. '
    html += 'One archived calibration-noise realization is used per optimizer seed. '
    html += 'The fixed catalogue spends six verification rollouts and stops; its curve stays flat at larger allowances. '
    html += 'Each historical source has 852 calibration rollouts. This is an explicitly data-rich historical scenario.</p>'
    html += '<p>Context success requires a feasible recommendation within 5% of its calibration-selected '
    html += 'finite-pool reference on independent validation noise. Recommendations must have been observed '
    html += 'feasible during target calibration. Missing incumbents and failed validations count as unsuccessful '
    html += 'contexts. All-context success requires all six contexts to succeed. Rollout feasibility is not '
    html += 'a stability certificate.</p><img alt="Budget curves" src="data:image/png;base64,'+encoded+'">'
    html += '<h2>Primary comparison</h2><div class="table">'+table(summary)+'</div>'
    html += '<p>Coverage precedes cost. A mean across all trials is unavailable if any policy is infeasible. '
    html += 'Conditional costs must not be compared across different feasible subsets. First-success time is '
    html += 'evaluated offline; unsuccessful trials remain censored at the budget in the restricted mean.</p>'
    html += '<h2>Paired attribution checks</h2><div class="table">'+table(paired)+'</div>'
    html += '<p>Catalogue + local BO versus fixed catalogue tests the value of additional target search. '
    html += 'Fleet-prior BO versus catalogue + local BO tests the additional historical surrogate with the same '
    html += 'initialization. Success-rate and failed-rollout differences use every matched target/seed pair. '
    html += 'Cost differences are labeled with the jointly feasible pair count.</p>'
    html += '<h2>Interpretation limits</h2><p>These nine deterministic plants, two calibration-noise realizations, '
    html += 'and optimizer seeds are not independent population samples. No pooled-pair significance claim is made. '
    html += 'The 71-candidate pool and original search-boundary limitations remain. New shapes, 25 m/s performance '
    html += 'and sparse source histories are not evaluated by this replay. The deterministic v2 reference '
    html += 'assignment audit is saved separately; the original v1 trajectories and results are unchanged.</p>'
    html += '<p>Before designing a more complex method, inspect whether either BO variant provides useful '
    html += 'improvement over the source catalogue and whether pooling increases failed commissioning trials. '
    html += 'See docs/tracking_budget.md for the complete frozen protocol.</p></body></html>'
    (out/'report.html').write_text(html, encoding='utf-8')
    return summary
