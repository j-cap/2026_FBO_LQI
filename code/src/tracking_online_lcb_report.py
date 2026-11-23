"""Evaluation of one-test calibration, retaining early missing recommendations."""
import base64
import io
import json

import numpy as np
import pandas as pd


LABELS = {'independent_bo': 'Independent GP BO', 'local_cost_bo': 'Local cost model + BO',
    'online_gain_sharing': 'Online gain sharing', 'single_scout': 'Single scout + verification',
    'federated_cost_bo': 'Federated cost model + BO', 'centralized_cost_bo': 'Centralized cost model + BO'}


def evaluate(results, domain, cfg, tables):
    references, validation = tables
    rows, cells, queries, learning = [], [], [], []
    for job in results:
        base = dict(condition=job['condition'], seed=job['seed'])
        for method, data in job['state']['methods'].items():
            for event in data['events']:
                queries.append(dict(**base, **{k: v for k, v in event.items()
                    if k != 'recommendations'}))
            learning.extend(dict(**base, method=method, **r) for r in data['learning'])
            for budget in cfg['fleet_budgets']:
                snapshot = data['snapshots'][str(budget)]
                prefix = data['events'][:budget]
                for i, client in enumerate(job['clients']):
                    common = dict(**base, method=method, fleet_budget=budget, client=client['id'],
                        plant_id=client['plant_id'], family=client['family'],
                        calibration_rounds=snapshot['round'], payload_bytes=snapshot['payload_bytes'])
                    local = [e for e in prefix if e['client'] == i]
                    outcomes = []
                    for context, candidate in zip(domain.contexts, snapshot['recommendations'][i]):
                        ref = references[client['plant_id'], context]
                        val = validation[client['plant_id'], context, candidate] if candidate is not None else None
                        eligible = val is not None and bool(val['eligible'])
                        gap = float(val['J']/ref['J']-1) if eligible else None
                        cell = dict(**common, context=context, candidate=candidate, eligible=eligible,
                            has_local_eligible_gain=candidate is not None,
                            J=float(val['J']) if eligible else None, relative_gap=gap,
                            success=gap is not None and gap <= cfg['relative_tolerance'])
                        cells.append(cell); outcomes.append(cell)
                    fully = all(c['eligible'] for c in outcomes)
                    rows.append(dict(**common, actual_client_queries=len(local),
                        fully_feasible=fully, context_success=float(np.mean([c['success'] for c in outcomes])),
                        eligible_context_coverage=float(np.mean([c['has_local_eligible_gain'] for c in outcomes])),
                        validation_eligible_context_coverage=float(np.mean([c['eligible'] for c in outcomes])),
                        all_contexts_success=all(c['success'] for c in outcomes),
                        J=float(np.mean([c['J'] for c in outcomes])) if fully else None,
                        failed_queries=sum(not e['complete'] or not e['feasible'] for e in local),
                        unresolved_queries=sum(not e['integration_resolved'] for e in local),
                        ineligible_queries=sum(not e['eligible'] for e in local)))
    return tuple(pd.DataFrame(x) for x in (rows, cells, queries, learning))


def eligibility_times(results, domain, cfg):
    """First local eligible observations, with unobserved events explicitly censored.

    These are online eligibility times, not validation or near-reference times.
    The clock is reported as both local queries and elapsed calibration rounds.
    """
    clients, contexts = [], []
    for job in results:
        for method, data in job['state']['methods'].items():
            for i, client in enumerate(job['clients']):
                base = dict(condition=job['condition'], seed=job['seed'], method=method,
                    client=client['id'], plant_id=client['plant_id'], family=client['family'])
                events = [e for e in data['events'] if e['client'] == i]
                first = {}
                for event in events:
                    if event['eligible']:
                        first.setdefault(event['context'], event)
                for context in domain.contexts:
                    event = first.get(context)
                    contexts.append(dict(**base, context=context, observed=event is not None,
                        first_eligible_query=event['local_evaluation'] if event else None,
                        first_eligible_round=event['round'] if event else None,
                        censor_query=cfg['queries_per_client'] if event is None else None,
                        censor_round=events[-1]['round'] if event is None else None))
                any_event = min(first.values(), key=lambda e: e['local_evaluation']) if first else None
                all_event = (max(first.values(), key=lambda e: e['local_evaluation'])
                             if len(first) == len(domain.contexts) else None)
                clients.append(dict(**base, any_eligible_observed=any_event is not None,
                    first_eligible_query=any_event['local_evaluation'] if any_event else None,
                    first_eligible_round=any_event['round'] if any_event else None,
                    all_contexts_observed=all_event is not None,
                    first_all_contexts_query=all_event['local_evaluation'] if all_event else None,
                    first_all_contexts_round=all_event['round'] if all_event else None,
                    eligible_contexts_at_cap=len(first), total_contexts=len(domain.contexts),
                    local_query_cap=cfg['queries_per_client'], final_round=events[-1]['round']))
    return pd.DataFrame(clients), pd.DataFrame(contexts)


def summarize_times(times):
    rows = []
    for (condition, method), g in times.groupby(['condition', 'method'], sort=False):
        any_good, all_good = g.any_eligible_observed, g.all_contexts_observed
        rows.append(dict(condition=condition, method=method, client_trials=len(g),
            first_eligible_observed=int(any_good.sum()), first_eligible_censored=int((~any_good).sum()),
            mean_first_eligible_query_observed=float(g.loc[any_good, 'first_eligible_query'].mean()) if any_good.any() else None,
            mean_first_eligible_round_observed=float(g.loc[any_good, 'first_eligible_round'].mean()) if any_good.any() else None,
            all_contexts_observed=int(all_good.sum()), all_contexts_censored=int((~all_good).sum()),
            mean_first_all_contexts_query_observed=float(g.loc[all_good, 'first_all_contexts_query'].mean()) if all_good.any() else None,
            mean_first_all_contexts_round_observed=float(g.loc[all_good, 'first_all_contexts_round'].mean()) if all_good.any() else None))
    return pd.DataFrame(rows)


def summarize(frame):
    rows = []
    for (condition, method, budget), g in frame.groupby(['condition', 'method', 'fleet_budget'], sort=False):
        seeds = g.groupby('seed')
        all_good = bool(g.fully_feasible.all())
        rows.append(dict(condition=condition, method=method, fleet_budget=int(budget),
            runs=g.seed.nunique(), physical_configurations=g.plant_id.nunique(),
            context_success_rate=float(g.context_success.mean()),
            eligible_context_coverage=float(g.eligible_context_coverage.mean()),
            validation_eligible_context_coverage=float(g.validation_eligible_context_coverage.mean()),
            all_contexts_success_rate=float(g.all_contexts_success.mean()),
            whole_fleet_success_rate=float(seeds.all_contexts_success.all().mean()),
            fully_feasible_policy_rate=float(g.fully_feasible.mean()),
            mean_J_all_trials=float(g.J.mean()) if all_good else None,
            conditional_mean_J=float(g[g.fully_feasible].J.mean()) if g.fully_feasible.any() else None,
            mean_fleet_failed_tests=float(seeds.failed_queries.sum().mean()),
            mean_fleet_unresolved_tests=float(seeds.unresolved_queries.sum().mean()),
            mean_fleet_ineligible_tests=float(seeds.ineligible_queries.sum().mean()),
            mean_calibration_rounds=float(seeds.calibration_rounds.first().mean()),
            mean_payload_bytes=float(seeds.payload_bytes.first().mean()),
            min_client_queries=int(g.actual_client_queries.min()), max_client_queries=int(g.actual_client_queries.max())))
    return pd.DataFrame(rows)


def write_report(out, frame, contexts, queries, learning, cfg, mode, times):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = summarize(frame)
    summary.to_csv(out/'online_summary.csv', index=False)
    timing = summarize_times(times)
    timing.to_csv(out/'eligibility_timing_summary.csv', index=False)
    paired = []
    for condition, g in frame.groupby('condition'):
        for method, baseline in [('federated_cost_bo', 'independent_bo'),
            ('federated_cost_bo', 'local_cost_bo'), ('federated_cost_bo', 'online_gain_sharing'),
            ('federated_cost_bo', 'single_scout'), ('federated_cost_bo', 'centralized_cost_bo')]:
            a = g[g.method == method].set_index(['seed', 'fleet_budget', 'client'])
            b = g[g.method == baseline].set_index(['seed', 'fleet_budget', 'client']).loc[a.index]
            for budget in cfg['fleet_budgets']:
                av, bv = a.xs(budget, level='fleet_budget'), b.xs(budget, level='fleet_budget')
                good = av.fully_feasible & bv.fully_feasible
                paired.append(dict(condition=condition, fleet_budget=budget, method=method, baseline=baseline,
                    matched_client_trials=len(av), jointly_feasible_pairs=int(good.sum()),
                    context_success_delta=float((av.context_success-bv.context_success).mean()),
                    mean_J_delta_on_jointly_feasible_pairs=float((av.loc[good, 'J']-bv.loc[good, 'J']).mean()) if good.any() else None,
                    mean_failed_tests_per_client_delta=float((av.failed_queries-bv.failed_queries).mean())))
    pd.DataFrame(paired).to_csv(out/'paired_comparisons.csv', index=False)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    colors = ['#8d9298', '#b0a15c', '#aa6946', '#72529b', '#238365', '#2b70a2']
    for row, condition in enumerate(cfg['conditions']):
        for method, color in zip(cfg['methods'], colors):
            g = summary[(summary.condition == condition) & (summary.method == method)]
            axes[row, 0].plot(g.fleet_budget, 100*g.eligible_context_coverage, marker='o', color=color, label=LABELS[method])
            axes[row, 1].plot(g.fleet_budget, 100*g.context_success_rate, marker='o', color=color)
            axes[row, 2].plot(g.fleet_budget, g.mean_fleet_ineligible_tests, marker='o', color=color)
        axes[row, 0].set(title=condition, ylabel='Contexts with a locally eligible gain [%]', ylim=(-2, 102))
        axes[row, 1].set(ylabel='Contexts within 5% on validation [%]', ylim=(-2, 102))
        axes[row, 2].set(ylabel='Mean ineligible fleet tests')
        for ax in axes[row]:
            ax.set(xlabel='Total fleet calibration tests (log scale)', xscale='log', xticks=cfg['fleet_budgets'])
            ax.set_xticklabels([str(b) for b in cfg['fleet_budgets']])
            ax.minorticks_off()
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1.07), ncol=3, fontsize=9)
    fig.savefig(out/'online_curves.png', dpi=150, bbox_inches='tight')
    stream = io.BytesIO(); fig.savefig(stream, format='png', dpi=110, bbox_inches='tight'); plt.close(fig)
    encoded = base64.b64encode(stream.getvalue()).decode()
    def records(df):
        return df.astype(object).where(pd.notna(df), None).to_dict('records')
    (out/'summary.json').write_text(json.dumps(dict(mode=mode, development_pilot=True,
        historical_training_records=0, initialization_tests=cfg['initialization_tests'],
        acquisition=cfg['acquisition'], summary=records(summary), eligibility_timing=records(timing)),
        indent=2, allow_nan=False)+'\n', encoding='utf-8')
    def table(df):
        return '<div class="table">'+df.to_html(index=False, float_format=lambda x: f'{x:.4g}',
                                                 escape=True, na_rep='unavailable')+'</div>'
    title = 'SMOKE: software verification only' if mode == 'smoke' else 'Online fleet calibration from empty histories'
    html = '''<!doctype html><html><head><meta charset="utf-8"><title>Online contextual calibration</title>
<style>body{font:16px system-ui;max-width:1300px;margin:30px auto;padding:0 20px;color:#192736}
table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccd5df;padding:6px}
.table{overflow-x:auto}img{max-width:100%}.note{background:#edf3f7;padding:16px}</style></head><body>'''
    html += '<h1>'+title+'</h1><p class="note">All methods start without calibration observations or a '
    html += 'learned catalogue. Fresh nonlinear rollouts are generated only when queried. Every query '
    html += 'counts, including initialization, failures, and scout/follower verification. Physical dynamics, '
    html += 'controller, objective and the 71-point gain pool match the frozen tracking diagnostic.</p>'
    html += '<p>The five concurrent methods share one factory-gain test per client. They adapt from round 2, '
    html += 'even if no locally eligible incumbent exists. All BO decisions minimize '
    html += 'mu - 2 sigma - log(max(1e-6, p_eligible)) on log J. Gain sharing can request a peer incumbent '
    html += 'from round 2; its BO fallback uses the same acquisition. The scout starts with one factory '
    html += 'test, adapts immediately, and preserves its separate waiting/verification procedure. Context schedules '
    html += 'and client/query noise are paired across methods. Each complete context cycle visits all six '
    html += 'references once. Round-t outcomes become shareable only after all round-t actions are frozen.</p>'
    html += '<img alt="Online fleet comparison" src="data:image/png;base64,'+encoded+'">'
    html += '<h2>Matched fleet budgets</h2>'+table(summary)
    html += '<h2>Time to local eligibility</h2>'+table(timing)
    html += '<p>Times refer to first eligible online tests, not validation or near-reference success. '
    html += 'Unobserved first events are censored at the local cap and retained in the counts. Conditional '
    html += 'means use only observed events and are shown with their denominators; methods with more '
    html += 'censoring cannot be ranked by those means alone. Local queries and calibration rounds are '
    html += 'separate clocks. Per-client and per-context timing tables are included.</p>'
    html += '<p>Missing or ineligible recommendations remain unsuccessful. Mean J is available across all '
    html += 'trials only when every policy is validation-eligible; conditional means use different subsets '
    html += 'and require care. Paired cost differences retain their own jointly feasible denominator. '
    html += 'Final recommendations always come from eligible experiments on the target client.</p>'
    html += '<h2>What each comparison identifies</h2><p>Independent GP BO versus local-cost BO checks '
    html += 'the local learned-mean pipeline. Federated versus local-cost BO tests sharing beyond the same '
    html += 'cost model trained locally. Gain sharing tests a simpler online library. Single scout tests '
    html += 'learning on one vehicle followed by target verification and repair. Centralized uses an exact '
    html += 'ridge fit of the same fixed-feature regression objective; it is an information/optimization '
    html += 'reference, not a required empirical upper bound.</p>'
    html += '<h2>Federated optimization diagnostics</h2>'+table(learning[learning.method == 'federated_cost_bo'][
        ['condition', 'seed', 'round', 'optimization_gap', 'cumulative_payload_bytes']])
    html += '<p>The shared mean uses fixed random features and conventional full-participation FedAvg '
    html += '(one full-batch local gradient step by default). The optimization gap is computed offline '
    html += 'against the exact centralized ridge optimum on the federated arm\'s own adaptively collected '
    html += 'records; it never changes acquisition. A large gap identifies learning error before judging '
    html += 'the value of collaboration. Local GP residuals and local feasibility learning remain active.</p>'
    html += '<h2>Time and communication accounting</h2><p>Parallel calibration rounds and total experiments '
    html += 'are separate metrics. Scout followers wait during scout search, then verify its frozen catalogue '
    html += 'and adapt locally. They eventually receive the same local query cap as other methods. Payload '
    html += 'bytes are float64 model-vector estimates or declared record/incumbent payloads; they exclude '
    html += 'transport overhead. No bandwidth advantage or formal privacy guarantee follows from this simulation.</p>'
    html += '<h2>Scope</h2><p>This is a three-seed development pilot on fixed plants, with independent fresh '
    html += 'client/query noise streams. It is not independent-fleet confirmation. Homogeneous clients have '
    html += 'identical physical parameters, but different noise and context programmes. The validation bank '
    html += 'uses the two frozen independent noise repeats; homogeneous clones share that evaluation bank. '
    html += 'The dense empirical reference and validation records are evaluator-only. No held-out shapes, '
    html += 'continuous gain optima, certified safety, or real-world commissioning protocol is established.</p>'
    html += '<p>The learned mean is empirical Bayes: it can include the target\'s own queried records. '
    html += 'Shared predictions are not inserted as extra observed trials. Residual GP uncertainty and '
    html += 'feasibility weights are working surrogates, not calibrated certificates. This changes both '
    html += 'initialization and acquisition relative to the frozen six-test pilot, so an across-pilot '
    html += 'difference cannot isolate either change alone. See docs/tracking_online_lcb.md.</p>'
    html += '</body></html>'
    (out/'report.html').write_text(html, encoding='utf-8')
