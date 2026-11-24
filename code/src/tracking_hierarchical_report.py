"""A-first evaluation of joint shared/local cost BO and its local control."""
import base64
import io
import json

import pandas as pd

from .tracking_hierarchical import LOCAL, SHARED
from .tracking_peer_cost import METHODS as BASELINES
from .tracking_peer_cost_report import LABELS as OLD_LABELS, mechanism_tables, summarize
from .tracking_online_lcb_report import summarize_times


LABELS = dict(OLD_LABELS, **{SHARED: 'D: Joint shared/local Bayes', LOCAL: 'E: Local Bayes control'})
PAIRS = [(SHARED, BASELINES[0]), (SHARED, BASELINES[1]), (SHARED, LOCAL), (SHARED, BASELINES[2])]


def paired_comparisons(trials, per_seed=False):
    rows = []
    keys = ['condition', 'fleet_budget']+(['seed'] if per_seed else [])
    for values, group in trials.groupby(keys, sort=False):
        common = dict(zip(keys, values))
        for method, baseline in PAIRS:
            a = group[group.method == method].set_index(['seed', 'client']).sort_index()
            b = group[group.method == baseline].set_index(['seed', 'client']).sort_index()
            if not a.index.equals(b.index) or a.empty:
                raise RuntimeError('Missing or unmatched comparison trials.')
            good = a.fully_feasible & b.fully_feasible
            mean_a = float(a.loc[good, 'J'].mean()) if good.any() else None
            mean_b = float(b.loc[good, 'J'].mean()) if good.any() else None
            rows.append(dict(**common, method=method, baseline=baseline,
                contrast_priority=PAIRS.index((method, baseline))+1,
                matched_client_trials=len(a), jointly_feasible_pairs=int(good.sum()),
                eligible_policies_method=int(a.fully_feasible.sum()),
                eligible_policies_baseline=int(b.fully_feasible.sum()),
                context_success_delta=float((a.context_success-b.context_success).mean()),
                eligible_policy_rate_delta=float((a.fully_feasible.astype(int)-b.fully_feasible.astype(int)).mean()),
                mean_J_method_jointly_feasible=mean_a, mean_J_baseline_jointly_feasible=mean_b,
                mean_J_delta_jointly_feasible=mean_a-mean_b if good.any() else None,
                relative_reduction_of_joint_mean_J=1-mean_a/mean_b if good.any() else None,
                mean_ineligible_tests_per_client_delta=float((a.ineligible_queries-b.ineligible_queries).mean())))
    return pd.DataFrame(rows)


def benchmark_readout(paired, per_seed, final_budget):
    final = paired[(paired.fleet_budget == final_budget) & paired.baseline.isin(BASELINES[:2])].copy()
    rows = []
    for _, row in final.iterrows():
        seeds = per_seed[(per_seed.fleet_budget == final_budget) & (per_seed.condition == row.condition) &
                        (per_seed.baseline == row.baseline)]
        primary_improved = row.context_success_delta > 1e-12
        eligibility_preserved = row.eligible_policies_method >= row.eligible_policies_baseline
        cost_improved = pd.notna(row.mean_J_delta_jointly_feasible) and row.mean_J_delta_jointly_feasible < 0
        rows.append(dict(condition=row.condition, baseline=row.baseline,
            contrast_priority=int(row.contrast_priority), context_success_delta=row.context_success_delta,
            eligible_policies_method=int(row.eligible_policies_method),
            eligible_policies_baseline=int(row.eligible_policies_baseline),
            jointly_feasible_pairs=int(row.jointly_feasible_pairs),
            relative_reduction_of_joint_mean_J=row.relative_reduction_of_joint_mean_J,
            primary_context_metric_improved=primary_improved,
            complete_eligibility_not_reduced=eligibility_preserved,
            conditional_cost_improved=bool(cost_improved),
            seeds_with_positive_primary=int((seeds.context_success_delta > 1e-12).sum()),
            seeds_with_negative_primary=int((seeds.context_success_delta < -1e-12).sum()),
            seeds_with_equal_primary=int((seeds.context_success_delta.abs() <= 1e-12).sum()),
            all_three_aggregate_metrics_support_improvement=bool(primary_improved and eligibility_preserved and cost_improved)))
    return pd.DataFrame(rows).sort_values(['contrast_priority', 'condition'])


def sharing_table(learning, cfg):
    rows = []
    for _, row in learning[learning.method == SHARED].iterrows():
        for fraction, weight, evidence in zip(cfg['hierarchical']['sharing_fractions'],
                row.sharing_weights, row.component_log_evidence):
            rows.append(dict(condition=row.condition, seed=int(row.seed), round=int(row['round']),
                used_by_later_query=row['round'] < cfg['queries_per_client'], sharing_fraction=fraction,
                posterior_weight=weight, log_evidence=evidence,
                expected_sharing_fraction=row.expected_sharing_fraction))
    return pd.DataFrame(rows)


def write_report(out, trials, contexts, queries, learning, times, cfg, mode):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    summary = summarize(trials)
    paired, seeds = paired_comparisons(trials), paired_comparisons(trials, per_seed=True)
    final_budget = cfg['fleet_budgets'][-1]
    primary = paired[(paired.fleet_budget == final_budget) & (paired.baseline == BASELINES[0])]
    secondary = paired[(paired.fleet_budget == final_budget) & (paired.baseline == BASELINES[1])]
    primary_seeds = seeds[(seeds.fleet_budget == final_budget) & (seeds.baseline == BASELINES[0])]
    readout = benchmark_readout(paired, seeds, final_budget)
    timing, sharing = summarize_times(times), sharing_table(learning, cfg)
    modes, peers, transfers = mechanism_tables(queries)
    final = trials[trials.fleet_budget == final_budget]
    family = final.groupby(['condition', 'family', 'method'], sort=False).agg(
        client_trials=('client', 'size'), eligible_policies=('fully_feasible', 'sum'),
        context_success_rate=('context_success', 'mean'), conditional_mean_policy_J=('J', 'mean')).reset_index()
    cells = contexts[contexts.fleet_budget == final_budget]
    by_context = cells.groupby(['condition', 'method', 'context'], sort=False).agg(
        client_context_trials=('client', 'size'), eligible_contexts=('eligible', 'sum'),
        successes=('success', 'sum'), context_success_rate=('success', 'mean'),
        conditional_mean_context_J=('J', 'mean')).reset_index()
    for name, frame in [('online_summary', summary), ('paired_comparisons', paired),
        ('per_seed_comparisons', seeds), ('primary_comparison', primary), ('secondary_comparison', secondary),
        ('primary_per_seed', primary_seeds), ('benchmark_readout', readout),
        ('eligibility_timing_summary', timing), ('sharing_strength', sharing),
        ('query_modes', modes), ('peer_verifications', peers), ('bo_origin_transfers', transfers),
        ('final_family_outcomes', family), ('final_context_outcomes', by_context)]:
        frame.to_csv(out/(name+'.csv'), index=False)

    fig, axes = plt.subplots(len(cfg['conditions']), 3, figsize=(15, 7.5), squeeze=False, constrained_layout=True)
    colors = ['#a35b39', '#3376b0', '#a0a4a8', '#248363', '#8c61a8']
    for row, condition in enumerate(cfg['conditions']):
        for method, color in zip(cfg['comparison_methods'], colors):
            g = summary[(summary.condition == condition) & (summary.method == method)]
            axes[row, 0].plot(g.fleet_budget, 100*g.eligible_context_coverage, marker='o', color=color, label=LABELS[method])
            axes[row, 1].plot(g.fleet_budget, 100*g.context_success_rate, marker='o', color=color)
            axes[row, 2].plot(g.fleet_budget, g.mean_fleet_ineligible_tests, marker='o', color=color)
        axes[row, 0].set(title=condition, ylabel='Contexts with a locally eligible gain [%]', ylim=(-2, 102))
        axes[row, 1].set(ylabel='Validation contexts within 5% [%]', ylim=(-2, 102))
        axes[row, 2].set(ylabel='Mean ineligible fleet tests')
        for ax in axes[row]:
            ax.set(xlabel='Fleet calibration tests (log scale)', xscale='log', xticks=cfg['fleet_budgets'])
            ax.set_xticklabels([str(b) for b in cfg['fleet_budgets']]); ax.minorticks_off(); ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1.09), ncol=3, fontsize=9)
    fig.savefig(out/'hierarchical_curves.png', dpi=150, bbox_inches='tight')
    stream = io.BytesIO(); fig.savefig(stream, format='png', dpi=110, bbox_inches='tight'); plt.close(fig)
    encoded = base64.b64encode(stream.getvalue()).decode()
    def records(frame):
        return frame.astype(object).where(pd.notna(frame), None).to_dict('records')
    (out/'summary.json').write_text(json.dumps(dict(mode=mode, development_pilot=True,
        historical_training_records=0, primary_baseline='A', secondary_baseline='B',
        inference='Exact federated Gaussian information messages; not FedAvg',
        summary=records(summary), benchmarks=records(readout)), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    def table(frame):
        return '<div class="table">'+frame.to_html(index=False, float_format=lambda x: f'{x:.4g}', na_rep='unavailable')+'</div>'
    title = 'SMOKE: software verification only' if mode == 'smoke' else 'Joint shared/local cost BO: A first, then B'
    html = '''<!doctype html><html><head><meta charset="utf-8"><title>Joint cost BO</title>
<style>body{font:16px system-ui;max-width:1450px;margin:30px auto;padding:0 20px;color:#192736}
table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccd5df;padding:6px}
.table{overflow-x:auto}img{max-width:100%}.note{background:#edf3f7;padding:16px}</style></head><body>'''
    html += '<h1>'+title+'</h1><p class="note">All methods use one factory test, the same peer-first '
    html += 'verification rule and the existing feasibility-penalized LCB. D jointly infers shared '
    html += 'and vehicle-local cost effects and integrates a public prior over sharing strength. '
    html += 'E has the same marginal cost prior but independent vehicle models. A/B/C are unchanged. '
    html += 'The full run reuses their completed traces for evaluation only; workers cannot access them. '
    html += 'The reduced smoke run simulates all five methods and is not evidence of performance.</p>'
    html += '<h2>Primary: D minus A at the declared final budget</h2>'+table(primary)
    html += '<h2>Primary by seed</h2>'+table(primary_seeds)
    html += '<h2>Secondary: D minus B</h2>'+table(secondary)
    html += '<h2>Benchmark readout</h2>'+table(readout)
    html += '<p>A positive context-success delta favors D. Retain all missing/ineligible contexts in '
    html += 'its denominator. Read complete-policy counts alongside it. Cost comparisons use jointly '
    html += 'eligible policies only. The metric flags are descriptive, not statistical tests or a new '
    html += 'composite score. Improvement over B/C/E alone does not meet the primary goal of beating A. '
    html += 'These are three programme/noise seeds on fixed plants, not independent physical fleets.</p>'
    html += '<img alt="Eligibility, near-reference success and failed tests" src="data:image/png;base64,'+encoded+'">'
    html += '<h2>Model-sharing and previous-architecture controls</h2>'+table(
        paired[(paired.fleet_budget == final_budget) & paired.baseline.isin([LOCAL, BASELINES[2]])])
    html += '<p>D versus E isolates cost-model dependence across vehicles under the same Bayesian '
    html += 'marginal prior. D versus C changes the statistical model and inference, so it is not a '
    html += 'pure sharing ablation. Each arm owns its queries and catalogue; adaptive peer feedback '
    html += 'remains part of the full-policy effect.</p>'
    html += '<h2>Final outcomes and payload</h2>'+table(summary[summary.fleet_budget == final_budget])
    html += '<p>Payload estimates count packed information matrices, vectors and evidence scalars for '
    html += 'D and the original vector iterations for C, separately from peer publications. Neither '
    html += 'formal privacy nor a communication advantage is presumed. End-of-final-round training '
    html += 'is logged but excluded from the final decision snapshot because no later query uses it.</p>'
    html += '<h2>Sharing strength before the last decision</h2>'+table(
        sharing[sharing['round'] == cfg['queries_per_client']-1])
    html += '<p>Weights use only this arm\'s eligible earlier-round labels. Kernel lengths and noise '
    html += 'remain public and fixed for D/E; their latent predictive variance includes uncertainty '
    html += 'within and between sharing-strength components. The LCB uses its mean and standard '
    html += 'deviation; it is not a certified confidence bound for the Gaussian mixture.</p>'
    html += '<h2>Time to eligibility</h2>'+table(timing)
    html += '<h2>Family outcomes</h2>'+table(family)
    html += '<h2>Context outcomes</h2>'+table(by_context)
    html += '<h2>Query modes</h2>'+table(modes)
    html += '<p>All new choices are replayed against their recorded outcomes before evaluator tables '
    html += 'are opened. Repeated computations and simulator caching never create free learner labels. '
    html += 'All budgets and per-seed contrasts are retained in the CSV files.</p></body></html>'
    (out/'report.html').write_text(html, encoding='utf-8')
