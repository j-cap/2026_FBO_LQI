"""Independent validation and matched comparisons of frozen peer-assisted traces."""
import base64
import io
import json

import pandas as pd

from .tracking_peer_cost import METHODS
from .tracking_online_lcb_report import (eligibility_times, evaluate as original_evaluate,
    summarize as original_summarize, summarize_times)


LABELS = dict(zip(METHODS, ['A: Peer gains + local GP', 'B: Peer gains + local cost model',
                           'C: Peer gains + federated cost model']))
PAIRS = [(METHODS[2], METHODS[1]), (METHODS[2], METHODS[0]), (METHODS[1], METHODS[0])]


def evaluate(results, domain, cfg, tables):
    trials, contexts, queries, learning = original_evaluate(results, domain, cfg, tables)
    payloads = []
    for job in results:
        for method, data in job['state']['methods'].items():
            for budget in cfg['fleet_budgets']:
                snapshot = data['snapshots'][str(budget)]
                payloads.append(dict(condition=job['condition'], seed=job['seed'], method=method,
                    fleet_budget=budget, **{k: snapshot[k] for k in ('peer_payload_bytes', 'model_payload_bytes')}))
    keys = ['condition', 'seed', 'method', 'fleet_budget']
    payload = pd.DataFrame(payloads)
    return (trials.merge(payload, on=keys, validate='many_to_one'),
            contexts.merge(payload, on=keys, validate='many_to_one'), queries, learning)


def summarize(trials):
    summary = original_summarize(trials)
    keys = ['condition', 'method', 'fleet_budget']
    payload = trials.groupby(keys, sort=False)[['peer_payload_bytes', 'model_payload_bytes']].mean().reset_index()
    payload = payload.rename(columns={k: 'mean_'+k for k in ('peer_payload_bytes', 'model_payload_bytes')})
    return summary.merge(payload, on=keys, validate='one_to_one')


def paired_comparisons(trials, per_seed=False):
    rows = []
    keys = ['condition', 'fleet_budget']+(['seed'] if per_seed else [])
    for values, group in trials.groupby(keys, sort=False):
        common = dict(zip(keys, values))
        for method, baseline in PAIRS:
            a = group[group.method == method].set_index(['seed', 'client']).sort_index()
            b = group[group.method == baseline].set_index(['seed', 'client']).sort_index()
            if not a.index.equals(b.index):
                raise RuntimeError('Unmatched client trials in the peer-assisted comparison.')
            good = a.fully_feasible & b.fully_feasible
            mean_a = float(a.loc[good, 'J'].mean()) if good.any() else None
            mean_b = float(b.loc[good, 'J'].mean()) if good.any() else None
            rows.append(dict(**common, method=method, baseline=baseline, matched_client_trials=len(a),
                jointly_feasible_pairs=int(good.sum()), eligible_policies_method=int(a.fully_feasible.sum()),
                eligible_policies_baseline=int(b.fully_feasible.sum()),
                context_success_delta=float((a.context_success-b.context_success).mean()),
                eligible_policy_rate_delta=float((a.fully_feasible.astype(int)-b.fully_feasible.astype(int)).mean()),
                mean_J_method_jointly_feasible=mean_a, mean_J_baseline_jointly_feasible=mean_b,
                mean_J_delta_jointly_feasible=mean_a-mean_b if good.any() else None,
                relative_reduction_of_joint_mean_J=1-mean_a/mean_b if good.any() else None,
                mean_failed_tests_per_client_delta=float((a.failed_queries-b.failed_queries).mean())))
    return pd.DataFrame(rows)


def mechanism_tables(queries):
    modes = queries.groupby(['condition', 'seed', 'method', 'mode'], sort=False).agg(
        queries=('eligible', 'size'), eligible_queries=('eligible', 'sum'),
        became_incumbent=('became_incumbent', 'sum'), conditional_mean_observed_J=('J', 'mean')).reset_index()
    peers = queries[queries['mode'] == 'peer_gain_verification'].copy()
    for key in ('peer_donor', 'peer_donor_query'):
        peers[key] = peers[key].astype(int)
    originated = peers[peers.peer_source_mode == 'feasibility_penalized_LCB']
    transfers = originated.groupby(['condition', 'seed', 'method', 'peer_donor',
        'peer_donor_query', 'context', 'candidate'], sort=False).agg(
        donor_J=('peer_donor_J', 'first'), target_tests=('eligible', 'size'),
        eligible_targets=('eligible', 'sum'), target_incumbents=('became_incumbent', 'sum'),
        first_target_round=('round', 'min')).reset_index()
    return modes, peers, transfers


def write_report(out, trials, contexts, queries, learning, times, cfg, mode):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    summary = summarize(trials)
    paired = paired_comparisons(trials)
    per_seed = paired_comparisons(trials, per_seed=True)
    final_budget = cfg['fleet_budgets'][-1]
    primary = paired[(paired.method == METHODS[2]) & (paired.baseline == METHODS[1]) &
                     (paired.fleet_budget == final_budget)]
    primary_seeds = per_seed[(per_seed.method == METHODS[2]) & (per_seed.baseline == METHODS[1]) &
                            (per_seed.fleet_budget == final_budget)]
    timing = summarize_times(times)
    modes, peers, transfers = mechanism_tables(queries)
    final = trials[trials.fleet_budget == final_budget]
    family = final.groupby(['condition', 'family', 'method'], sort=False).agg(
        client_trials=('client', 'size'), eligible_policies=('fully_feasible', 'sum'),
        context_success_rate=('context_success', 'mean'),
        conditional_mean_policy_J=('J', 'mean')).reset_index()
    cells = contexts[contexts.fleet_budget == final_budget]
    by_context = cells.groupby(['condition', 'method', 'context'], sort=False).agg(
        client_context_trials=('client', 'size'), eligible_contexts=('eligible', 'sum'),
        successes=('success', 'sum'), context_success_rate=('success', 'mean'),
        conditional_mean_context_J=('J', 'mean')).reset_index()
    for name, frame in [('online_summary', summary), ('paired_comparisons', paired),
        ('per_seed_comparisons', per_seed), ('primary_comparison', primary),
        ('primary_per_seed', primary_seeds), ('eligibility_timing_summary', timing),
        ('query_modes', modes), ('peer_verifications', peers), ('bo_origin_transfers', transfers),
        ('final_family_outcomes', family), ('final_context_outcomes', by_context)]:
        frame.to_csv(out/(name+'.csv'), index=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), constrained_layout=True)
    for row, condition in enumerate(cfg['conditions']):
        for method, color in zip(METHODS, ['#a35b39', '#3376b0', '#248363']):
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
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1.055), ncol=3, fontsize=9)
    fig.savefig(out/'peer_cost_curves.png', dpi=150, bbox_inches='tight')
    stream = io.BytesIO(); fig.savefig(stream, format='png', dpi=110, bbox_inches='tight'); plt.close(fig)
    encoded = base64.b64encode(stream.getvalue()).decode()

    def records(frame):
        return frame.astype(object).where(pd.notna(frame), None).to_dict('records')
    (out/'summary.json').write_text(json.dumps(dict(mode=mode, development_pilot=True,
        historical_training_records=0, common_peer_priority=True,
        primary_contrast='peer_federated_cost_bo minus peer_local_cost_bo at final budget',
        summary=records(summary), primary=records(primary), primary_per_seed=records(primary_seeds)),
        indent=2, allow_nan=False)+'\n', encoding='utf-8')

    def table(frame):
        return '<div class="table">'+frame.to_html(index=False, escape=True,
            float_format=lambda x: f'{x:.4g}', na_rep='unavailable')+'</div>'
    title = 'SMOKE: software verification' if mode == 'smoke' else 'Peer-assisted online cost-model comparison'
    html = '''<!doctype html><html><head><meta charset="utf-8"><title>Peer-assisted cost model</title>
<style>body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:0 20px;color:#192736}
table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccd5df;padding:6px}
.table{overflow-x:auto}img{max-width:100%}.note{background:#edf3f7;padding:16px}</style></head><body>'''
    html += '<h1>'+title+'</h1><p class="note">All three methods start with one paid factory-gain test, '
    html += 'then use the same peer-first rule: test the lowest-donor-cost eligible context incumbent '
    html += 'not yet tried locally. If none exists, use the common feasibility-penalized LCB BO rule. '
    html += 'A uses a zero additional mean, B a locally fitted mean, and C a federated mean. Every '
    html += 'verification, failure and BO test consumes the same rollout budget.</p>'
    html += '<h2>Primary: C minus B at the final fleet budget</h2>'+table(primary)
    html += '<p>Positive context-success deltas favor the federated mean. Read them alongside complete '
    html += 'eligible policy counts. J deltas are method minus baseline on jointly eligible pairs only. '
    html += 'Missing/ineligible recommendations remain unsuccessful in all coverage metrics.</p>'
    html += '<h2>Primary comparison by seed</h2>'+table(primary_seeds)
    html += '<p>These are programme/noise seeds on fixed plants, not independent fleet draws. '
    html += 'Homogeneous clients also share the archived validation bank. Seed-level results expose '
    html += 'variation without treating clients and contexts as independent fleets.</p>'
    html += '<img alt="Eligibility, quality and failures across matched budgets" src="data:image/png;base64,'+encoded+'">'
    html += '<h2>Final policies and payload</h2>'+table(summary[summary.fleet_budget == final_budget])
    html += '<p>Each method owns its online observations and catalogue. Different BO discoveries can '
    html += 'produce different later peer proposals and fallback counts. This is part of the total '
    html += 'calibration effect. Peer and model payloads are separate estimates; transport overhead and '
    html += 'offline provenance fields are excluded. No privacy or communication advantage is presumed.</p>'
    html += '<h2>Time to eligibility</h2>'+table(timing)
    html += '<p>Unobserved events are censored at the local cap. Conditional time and J means retain '
    html += 'their observed denominators and must not be ranked without the completion counts.</p>'
    html += '<h2>Per-family final results</h2>'+table(family)
    html += '<h2>Query modes</h2>'+table(modes)
    html += '<p>The query-mode costs are descriptive: fallback queries may occur at different histories '
    html += 'in different methods. They are not matched causal cost effects. BO-origin transfers record '
    html += 'incumbents first selected by a donor through BO and later verified by peers; they do not '
    html += 'claim global novelty. Full records are in bo_origin_transfers.csv.</p>'
    gap = learning[learning.method == METHODS[2]].groupby(['condition', 'seed'], sort=False).agg(
        median_gap=('optimization_gap', 'median'), max_gap=('optimization_gap', 'max')).reset_index()
    html += '<h2>Same-data federated optimization</h2>'+table(gap)
    html += '<p>The gap compares the federated weights with the exact ridge optimum on that arm\'s '
    html += 'current records. It is an offline diagnostic and never informs acquisition. Reference costs '
    html += 'and independent validation are opened only after all traces are complete. The full campaign '
    html += 'checks arm A against the completed one-test gain-sharing control. This finite-pool simulation '
    html += 'does not establish certified safety or unseen-trajectory generalization.</p></body></html>'
    (out/'report.html').write_text(html, encoding='utf-8')
