"""Matched quality/traffic reports, preserving per-seed and eligibility denominators."""
import base64
from html import escape
import io
import json

import numpy as np
import pandas as pd

from .tracking_dense_gain import A, B, D, E, METHODS
from .tracking_dense_gain_report import LABELS, paired_comparisons


def table(frame):
    frame = frame.copy()
    for column in ('method', 'baseline'):
        if column in frame:
            frame[column] = frame[column].map(LABELS)
    return '<div class="table">' + frame.to_html(index=False, float_format=lambda x: f'{x:.4g}', na_rep='unavailable') + '</div>'


def document(title, body):
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>' + escape(title) +
        '</title><style>body{font:16px/1.6 system-ui;max-width:1250px;margin:35px auto;padding:0 22px;color:#203747}'
        'table{border-collapse:collapse;font-size:12px}td,th{padding:8px;border:1px solid #d0dce2}'
        '.table{overflow-x:auto;margin:18px 0}img{max-width:100%}h2{margin-top:32px}</style></head><body><h1>' +
        escape(title) + '</h1>' + body + '</body></html>')


def write_audit_report(state, results):
    out = state['out']
    traffic = pd.DataFrame([dict(seed=r['seed'], **row) for r in results for row in r['traffic']])
    checks = pd.DataFrame([dict(seed=r['seed'], **row) for r in results for row in r['posterior_checks']])
    traffic.to_csv(out / 'traffic.csv', index=False)
    checks.to_csv(out / 'posterior_checks.csv', index=False)
    shared = traffic[traffic.method == D].copy()
    shared['model_reduction_factor'] = shared.legacy_dense_model_payload_bytes / shared.application_cost_bytes
    shared['legacy_total_over_compact_application'] = shared.legacy_total_reference_bytes / shared.application_total_bytes
    shared['same_catalogue_total_reduction_factor'] = shared.legacy_model_plus_same_peer_application_bytes / shared.application_total_bytes
    shared.to_csv(out / 'shared_cost_savings.csv', index=False)
    summary = dict(all_decisions_and_recommendations_identical=True, new_simulations=0, new_optimization=0,
        seeds=[r['seed'] for r in results], paid_decisions_replayed=sum(r['paid_decisions'] for r in results),
        posterior_queries_checked=len(checks), posterior_inputs_checked=int(checks.inputs.sum()),
        maximum_mean_error=float(checks.maximum_mean_error.max()), maximum_variance_error=float(checks.maximum_variance_error.max()),
        maximum_weight_error=float(checks.maximum_weight_error.max()),
        mean_legacy_D_model_bytes=float(shared.legacy_dense_model_payload_bytes.mean()),
        mean_compact_D_cost_application_bytes=float(shared.application_cost_bytes.mean()),
        mean_compact_D_total_application_bytes=float(shared.application_total_bytes.mean()),
        mean_compact_D_peer_application_bytes=float(shared.application_peer_bytes.mean()),
        signature=state['audit_signature'])
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    body = '<p>Read-only replay of the three archived dense-gain programmes. No simulator or GP hyperparameter optimizer is allowed. '
    body += 'Every A/B/D/E action, peer provenance, logged diagnostic and locally verified recommendation reproduces. '
    body += 'All allowed D/E predictions at each BO decision are compared, including mixture weights and shared coefficient covariances.</p>'
    body += table(pd.DataFrame([summary])) + '<h2>Before the last paid decision</h2>' + table(shared)
    body += '<p>Cost records contain client ID, trial ID, public gain/context input ID and float64 J (20 bytes). '
    body += 'Peer records also carry the source-mode code (24 bytes). Actual 16-byte packet headers and every recipient delivery are charged. '
    body += 'The historical dense-model and 12-byte peer numbers were analytical payload references, so their sum is not a measured network baseline. '
    body += 'The same-catalogue column combines the old model estimate with the new peer application traffic to isolate the cost-channel change. '
    body += 'IP/TLS, routing, acknowledgements, retries and public-design provisioning are excluded. The final unused flush is separate in traffic.csv.</p>'
    body += '<p>The coordinator relays records; every recipient reconstructs the same exact shared/local posterior. '
    body += 'A repeated public input from a different client remains a separate observation. Failed-test labels remain private. '
    body += 'This audit establishes representation equivalence for the frozen programmes; it makes no new control-performance claim.</p>'
    (out / 'report.html').write_text(document('Compact-exchange equivalence audit', body), encoding='utf-8')


def interval_comparisons(trials):
    rows, units = [], []
    for (method, budget), group in trials.groupby(['method', 'fleet_budget'], sort=False):
        reference = group[group.interval == 1].set_index(['seed', 'client']).sort_index()
        for interval in (2, 4):
            target = group[group.interval == interval].set_index(['seed', 'client']).sort_index()
            if not target.index.equals(reference.index) or not target.encountered_contexts.equals(reference.encountered_contexts):
                raise RuntimeError('Interval programmes or paired denominators differ.')
            effects = (target.context_success - reference.context_success).groupby(level='seed').mean()
            common = target.complete_eligible_on_encountered & reference.complete_eligible_on_encountered
            jt = float(target.loc[common, 'J'].mean()) if common.any() else None
            jr = float(reference.loc[common, 'J'].mean()) if common.any() else None
            rows.append(dict(method=method, interval=interval, baseline_interval=1, fleet_budget=budget,
                programme_seeds=len(effects), matched_client_runs=len(target), success_delta_pp=100 * float(effects.mean()),
                worst_seed_delta_pp=100 * float(effects.min()), positive_seeds=int((effects > 1e-12).sum()),
                negative_seeds=int((effects < -1e-12).sum()), tied_seeds=int((abs(effects) <= 1e-12).sum()),
                failed_requests_delta=int(target.ineligible_queries.sum() - reference.ineligible_queries.sum()),
                complete_policy_delta=int(target.complete_eligible_on_encountered.sum() - reference.complete_eligible_on_encountered.sum()),
                common_eligible_pairs=int(common.sum()), paired_mean_J_target=jt, paired_mean_J_k1=jr,
                paired_J_change_percent=100 * (jt / jr - 1) if jt is not None else None))
            units.extend(dict(method=method, interval=interval, baseline_interval=1, fleet_budget=budget,
                seed=int(seed), success_delta_pp=100 * float(effect)) for seed, effect in effects.items())
    return pd.DataFrame(rows), pd.DataFrame(units)


def communication_tables(state, results):
    rows, contacts, timing = [], [], []
    final = str(state['cfg']['fleet_budgets'][-1])
    for result in results:
        common = dict(interval=result['interval'], seed=result['seed'])
        for method, data in result['state']['methods'].items():
            network = data['network']
            before = data['snapshots'][final]
            used = [c for c in network['contacts'] if c['round'] < state['cfg']['queries_per_client']]
            rows.append(dict(**common, method=method, contacts_used=len(used), contacts_including_final=len(network['contacts']),
                used_application_bytes=before['payload_bytes'], used_peer_application_bytes=before['peer_payload_bytes'],
                used_cost_application_bytes=before['model_payload_bytes'], used_packet_deliveries=before['packet_deliveries'],
                used_coordinator_broadcasts=before['coordinator_broadcasts'],
                all_application_bytes=data['payload_bytes'], all_peer_application_bytes=data['peer_payload_bytes'],
                all_cost_application_bytes=data['model_payload_bytes'], all_packet_deliveries=network['packet_deliveries'],
                unused_final_flush_bytes=data['payload_bytes'] - before['payload_bytes'],
                cost_records_used=sum(c['cost']['records'] for c in used if c['cost']),
                peer_records_used=sum(c['peer']['records'] for c in used),
                all_unique_cost_records=len(network['summaries']),
                all_cost_body_bytes=network['model_body_bytes'], all_peer_body_bytes=network['peer_body_bytes']))
            for contact in network['contacts']:
                for channel in ('peer', 'cost'):
                    log = contact[channel]
                    if log is not None:
                        contacts.append(dict(**common, method=method, round=contact['round'], channel=channel,
                            used_before_final_decision=contact['round'] < state['cfg']['queries_per_client'],
                            **{k: v for k, v in log.items() if k != 'packets'}))
        timing.extend(dict(**common, **row) for row in result['timing'])
    communication = pd.DataFrame(rows)
    averages = communication.groupby(['interval', 'method'], sort=False).mean(numeric_only=True).drop(columns='seed').reset_index()
    baseline = averages[averages.interval == 1].set_index('method')
    averages['application_reduction_vs_k1_percent'] = [100 * (1 - r.used_application_bytes / baseline.loc[r.method, 'used_application_bytes'])
        for r in averages.itertuples()]
    averages['packet_reduction_vs_k1_percent'] = [100 * (1 - r.used_packet_deliveries / baseline.loc[r.method, 'used_packet_deliveries'])
        for r in averages.itertuples()]
    return communication, averages, pd.DataFrame(contacts), pd.DataFrame(timing)


def write_report(state, results, trials, cells, visits, learning, refs, audit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out, cfg = state['out'], state['cfg']
    final = cfg['fleet_budgets'][-1]
    pair_frames, seed_frames = [], []
    for interval in state['requested']['intervals']:
        pairs, seeds = paired_comparisons(trials[trials.interval == interval])
        pair_frames.append(pairs.assign(interval=interval))
        seed_frames.append(seeds.assign(interval=interval))
    pairs, seed_pairs = pd.concat(pair_frames, ignore_index=True), pd.concat(seed_frames, ignore_index=True)
    effects, seed_effects = interval_comparisons(trials)
    summary = trials.groupby(['interval', 'method', 'fleet_budget'], sort=False).agg(
        client_runs=('client', 'size'), programme_seeds=('seed', 'nunique'),
        context_success_percent=('context_success', lambda v: 100 * v.mean()),
        validation_eligible_coverage_percent=('validation_eligible_context_coverage', lambda v: 100 * v.mean()),
        complete_eligible_policies=('complete_eligible_on_encountered', 'sum'),
        failed_paid_requests=('ineligible_queries', 'sum'), conditional_mean_J=('J', 'mean'),
        mean_unvisited_contexts=('unvisited_contexts', 'mean'), reference_coverage=('reference_coverage', 'mean')).reset_index()
    per_seed = trials[trials.fleet_budget == final].groupby(['interval', 'method', 'seed'], sort=False).agg(
        context_success_percent=('context_success', lambda v: 100 * v.mean()),
        complete_eligible_policies=('complete_eligible_on_encountered', 'sum'), failed_paid_requests=('ineligible_queries', 'sum')).reset_index()
    metadata = pd.DataFrame(cfg['context_metadata'])[['id', 'speed', 'frequency']]
    context = cells.merge(metadata, left_on='context', right_on='id', validate='many_to_one')
    frequency_units = context.groupby(['interval', 'method', 'seed', 'client', 'fleet_budget', 'frequency'], sort=False).agg(
        context_success=('success', 'mean'), encountered_contexts=('success', 'size')).reset_index()
    frequency = frequency_units.groupby(['interval', 'method', 'fleet_budget', 'frequency'], sort=False).agg(
        context_success_percent=('context_success', lambda v: 100 * v.mean()), encountered_contexts=('encountered_contexts', 'sum')).reset_index()
    modes = visits.groupby(['interval', 'method', 'seed', 'mode'], sort=False).agg(requests=('eligible', 'size'),
        eligible=('eligible', 'sum'), new_incumbents=('became_incumbent', 'sum')).reset_index()
    modes['ineligible'] = modes.requests - modes.eligible
    traffic, average_traffic, contacts, timing = communication_tables(state, results)
    timing_summary = timing.groupby(['interval', 'method', 'phase'], sort=False).agg(
        calls=('seconds', 'size'), total_seconds=('seconds', 'sum'), mean_seconds=('seconds', 'mean'),
        maximum_seconds=('seconds', 'max')).reset_index()
    frames = dict(budget_trials=trials, budget_contexts=cells, paid_visits=visits, learning_diagnostics=learning,
        online_summary=summary, paired_comparisons=pairs, per_seed_comparisons=seed_pairs,
        interval_comparisons=effects, per_seed_interval_comparisons=seed_effects, per_seed_outcomes=per_seed,
        frequency_units=frequency_units, frequency_outcomes=frequency, query_modes=modes,
        communication=traffic, communication_summary=average_traffic, contacts=contacts,
        timings=timing, timing_summary=timing_summary,
        empirical_references=pd.DataFrame([dict(context=c, **r) for c, r in refs.items()]))
    for name, frame in frames.items():
        frame.to_csv(out / (name + '.csv'), index=False)
    readout = dict(confirmation_claim=False, mode=state['specification']['mode'],
        primary_metric='Encountered-context success within 5% of finite-pool reference; clients and seeds equally weighted.',
        equivalence_audit_signature=audit['audit_signature'],
        interpretation='Compare D-A first, D-B second and D-E for added cost sharing, separately at each interval. Compare each method at k=2/4 with its matched k=1 programme.',
        no_predeclared_noninferiority_margin=True,
        network_scope='Periodic synchronization only; no latency, loss, asynchronous updates, TLS or routing overhead.')
    (out / 'pilot_readout.json').write_text(json.dumps(readout, indent=2) + '\n', encoding='utf-8')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    colors = dict(zip(METHODS, ['#66748a', '#aa7935', '#007d75', '#7864ad']))
    for method in METHODS:
        rows = summary[(summary.method == method) & (summary.fleet_budget == final)].sort_values('interval')
        wire = average_traffic[average_traffic.method == method].sort_values('interval')
        axes[0].plot(rows.interval, rows.context_success_percent, marker='o', color=colors[method], label=LABELS[method])
        axes[1].plot(wire.used_application_bytes / 1000, rows.context_success_percent, marker='o', color=colors[method])
        axes[2].plot(wire.used_packet_deliveries, rows.context_success_percent, marker='o', color=colors[method])
        for index in range(len(wire)):
            axes[1].annotate(f'k={int(wire.iloc[index].interval)}', (wire.iloc[index].used_application_bytes / 1000, rows.iloc[index].context_success_percent), fontsize=7, xytext=(4, 4), textcoords='offset points')
    axes[0].set(xlabel='Synchronization interval [rounds]', xticks=[1, 2, 4])
    axes[1].set(xlabel='Mean application traffic used [decimal kB/fleet]')
    axes[2].set(xlabel='Packet deliveries used per fleet')
    for ax in axes:
        ax.set(ylabel='Final encountered-context success [%]', ylim=(-2, 102))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.savefig(out / 'quality_and_communication.png', dpi=160)
    image = io.BytesIO()
    fig.savefig(image, format='png', dpi=120)
    plt.close(fig)
    title = 'Compact communication: matched 1/2/4-round pilot'
    if state['specification']['mode'] == 'smoke':
        title = 'Communication SMOKE: software verification only'
    body = '<p>Question: how much synchronization can be removed while retaining the benefit of shared cost information over A, B and matched independent E?</p>'
    body += f'<p>{len(state["clients"])} homogeneous clients, {cfg["queries_per_client"]} paid requests each, {len(state["pool"]):,} public gains and 18 speed/frequency contexts. '
    body += 'Cold histories, one paid factory test, the same charged peer-first rule and the same local eligibility penalty. A/B beta=2; D/E beta=0.5. '
    body += 'The dynamics, trajectories, objective, prior and controller bounds remain those of the frozen dense-gain reference.</p>'
    body += '<p>All four arms refresh their own observations every round. Both the peer-gain catalogue and D cost records synchronize only after every k rounds. '
    body += 'A decision in round r can use remote information through k·floor((r−1)/k), plus its own current data. The coordinator relays summaries; '
    body += 'recipients reconstruct the exact shared/local GP. Failed labels remain private. Public input IDs and sender/trial IDs retain observation provenance.</p>'
    body += '<h2>Added cost-sharing value at each interval</h2>' + table(pairs[pairs.fleet_budget == final])
    body += '<h2>Matched change from every-round contact</h2>' + table(effects[effects.fleet_budget == final])
    body += '<p>Effects are paired by programme/noise seed and client. Success effects are percentage points; lower cost is better. '
    body += 'Cost comparisons include only common complete eligible policies and show their denominators. No noninferiority margin has been declared.</p>'
    body += '<img alt="Success versus interval, application traffic and packet deliveries" src="data:image/png;base64,' + base64.b64encode(image.getvalue()).decode() + '">'
    body += '<h2>Control outcomes</h2>' + table(summary[summary.fleet_budget == final])
    body += '<h2>Mean application traffic per fleet and arm</h2>' + table(average_traffic)
    body += '<p>Used traffic is delivered before the last paid decision; all traffic also includes the unused final flush. Each channel/contact uses '
    body += 'one uplink per client and one coordinator batch delivered to every client, including empty 16-byte headers. A 20-byte eligible-cost record is sent once per sender '
    body += 'then delivered to all clients. A full peer catalogue is repeated at each contact with 24-byte records. '
    body += 'Batching reduces contact/header overhead and repeated catalogues; it does not remove every new cost record. Different controller decisions can change eligibility counts. '
    body += 'Packet deliveries count each recipient; coordinator broadcasts count one generated batch. Public-design provisioning, IP/TLS, acknowledgements and retries are excluded.</p>'
    body += '<h2>Per-seed interval effects</h2>' + table(seed_effects[seed_effects.fleet_budget == final])
    body += '<h2>Frequency outcomes</h2>' + table(frequency[frequency.fleet_budget == final])
    body += '<h2>Failed tests and proposal allocation</h2>' + table(modes)
    body += '<h2>Local computation</h2>' + table(timing_summary)
    body += '<p>Proposal timings include local model fitting and acquisition; codec timings include serialization and cache updates, with no simulated network delay. '
    body += 'Wall times are descriptive and machine dependent. Recipient reconstruction moves computation to clients; it is not a compression-free compute claim.</p>'
    body += '<h2>Audit and scope</h2><p><a href="../' + escape(audit['out'].name) + '/report.html">Compact-exchange equivalence audit</a>: '
    body += 'three archived programmes reproduced with no new rollouts. This is separate from these fresh matched online runs. '
    body += 'Four fresh common validation repeats cover the union of paid controllers and references across all intervals and arms; archived calibration labels select references only. '
    body += 'Within encountered contexts, missing or ineligible recommendations remain failures. Unvisited contexts are reported separately. '
    body += 'Three new seeds on one plant remain exploratory; no heterogeneous-fleet or network-reliability claim follows.</p>'
    body += '<p>Signature: <code>' + escape(state['signature']) + '</code>.</p>'
    (out / 'report.html').write_text(document(title, body), encoding='utf-8')
    return frames
