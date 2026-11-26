"""Read-only audit of the compact communication pilot; no rollout or optimizer."""
import argparse
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

import tracking_communication as runner
import analyze_tracking_dense_gain as dense
import numpy as np
import pandas as pd

from src.tracking_budget import matern
from src.tracking_communication_report import write_report
from src.tracking_communication_wire import PEER, MODES, exchange
from src.tracking_dense_gain import A, B, D, E, METHODS, BONUSES
from src.tracking_kernel_features import query_posterior
from src.tracking_six_speed_report import evaluate

OUT = runner.REPO / 'docs/analysis/communication'
LETTERS = dict(zip(METHODS, ['A', 'B', 'D', 'E']))


def direct_worker(job):
    state = runner.base.WORKER
    result = runner.base.read(runner.run_path(state, job))
    domain, cfg, n = state['domain'], state['cfg'], len(state['clients'])
    rows = []
    for method in (D, E):
        data = result['state']['methods'][method]
        for r in range(1, cfg['queries_per_client']):
            contact = (r // job['interval']) * job['interval']
            groups = {}
            for event in data['events'][r * n:(r + 1) * n]:
                if event['mode'] != 'feasibility_penalized_LCB':
                    continue
                # Independently form the permitted observation set from trial
                # timestamps, without using the learner's cache/view helper.
                view = [[o for o in h[:r if j == event['client'] else contact] if o['eligible']]
                    if method == D or j == event['client'] else [] for j, h in enumerate(data['histories'])]
                key = tuple(tuple((o['action'], o['J']) for o in h) for h in view)
                groups.setdefault(key, (view, []))[1].append(event)
            for view, queries in groups.values():
                inputs = sorted({o['action'] for h in view for o in h} | {e['action'] for e in queries})
                positions = {a: j for j, a in enumerate(inputs)}
                kernel = matern(domain.X[inputs], domain.X[inputs], cfg['gp']['length_scales'], cfg['gp']['amplitude'], cfg['gp']['constant_variance'])
                local = [[dict(o, action=positions[o['action']]) for o in h] for h in view]
                control = cfg['hierarchical'] if method == D else dict(sharing_fractions=[0.], prior_weights=[1.])
                direct = query_posterior(local, kernel, control, cfg['gp'],
                    [e['client'] for e in queries], [positions[e['action']] for e in queries])
                for j, event in enumerate(queries):
                    mean, variance = direct['mean'][j], direct['variance'][j]
                    score = mean - BONUSES[method] * np.sqrt(variance) - cfg['acquisition']['feasibility_penalty'] * np.log(
                        max(event['eligibility_probability'], cfg['acquisition']['probability_floor']))
                    np.testing.assert_allclose([mean, variance, score],
                        [event['predicted_log_J'], event['predicted_variance'], event['acquisition_score']], rtol=2e-8, atol=2e-8)
                    rows.append(dict(interval=job['interval'], seed=job['seed'], method=method,
                        round=r + 1, client=event['client'], remote_information_round=contact,
                        eligible_observations=sum(map(len, local)), own_eligible=len(local[event['client']]),
                        absolute_mean_error=abs(mean - event['predicted_log_J']),
                        absolute_variance_error=abs(variance - event['predicted_variance']),
                        absolute_acquisition_error=abs(score - event['acquisition_score']),
                        expected_sharing_fraction=direct['expected_sharing_fraction']))
    pd.DataFrame(rows).to_csv(state['analysis_out'] / f'posterior_k{job["interval"]}_{job["seed"]}.csv', index=False)


def reconstruct(state, results, audit):
    # Reuse the independent dense-gain calibration/validation checks with the
    # new runner's task identities. Frozen runner files and labels stay intact.
    with patch.object(dense, 'runner', runner):
        refs, validation, raw, source_checks = dense.references_and_validation(state, results)
    evaluated = []
    for interval in state['requested']['intervals']:
        pieces = evaluate([r for r in results if r['interval'] == interval], state['domain'], state['cfg'], refs, validation)
        evaluated.append(tuple(frame.assign(interval=interval) for frame in pieces))
    data = [pd.concat([pieces[j] for pieces in evaluated], ignore_index=True) for j in range(4)]
    with tempfile.TemporaryDirectory(prefix='communication_report_audit_') as directory:
        temporary = dict(state, out=Path(directory))
        frames = write_report(temporary, results, *data, refs, audit)
        if runner.base.read(Path(directory) / 'pilot_readout.json') != runner.base.read(state['out'] / 'pilot_readout.json'):
            raise RuntimeError('Saved descriptive readout differs.')
    unit = ['interval', 'seed', 'method']
    keys = dict(budget_trials=unit + ['fleet_budget', 'client'], budget_contexts=unit + ['fleet_budget', 'client', 'context'],
        paid_visits=unit + ['round', 'client'], learning_diagnostics=unit + ['round'],
        online_summary=['interval', 'method', 'fleet_budget'], paired_comparisons=['interval', 'method', 'baseline', 'fleet_budget'],
        per_seed_comparisons=unit + ['baseline', 'fleet_budget'], interval_comparisons=['interval', 'method', 'fleet_budget'],
        per_seed_interval_comparisons=unit + ['fleet_budget'], per_seed_outcomes=unit,
        frequency_units=unit + ['client', 'fleet_budget', 'frequency'], frequency_outcomes=['interval', 'method', 'fleet_budget', 'frequency'],
        query_modes=unit + ['mode'], communication=unit, communication_summary=['interval', 'method'],
        contacts=unit + ['round', 'channel'], timings=unit + ['round', 'client', 'phase'],
        timing_summary=['interval', 'method', 'phase'], empirical_references=['context'])
    checks = [dense.verify_table(state, name, frame, keys[name]) for name, frame in frames.items()]
    visits = frames['paid_visits']
    expected = dict(signature=state['signature'], mode='pilot', paid_requests=len(visits),
        all_local_caps_respected=bool((visits.groupby(unit + ['client']).size() == state['cfg']['queries_per_client']).all()),
        all_paid_choices_and_packets_replayed=True, equivalence_signature=audit['audit_signature'],
        new_calibration_rollouts=0, reused_calibration_observations=len(state['pool']) * len(state['domain'].contexts) * 2,
        validation_rollouts=len(raw), historical_training_records=0, unavailable_references=sum(not r['eligible'] for r in refs.values()),
        unresolved_online_requests=int((~visits.integration_resolved).sum()), confirmation_claim=False)
    if expected != runner.base.read(state['out'] / 'verification.json'):
        raise RuntimeError('Saved verification metadata differs.')
    return frames, raw, checks, source_checks


def diagnostics(state, frames, out):
    budget = state['cfg']['fleet_budgets'][-1]
    for source, name in [('online_summary', 'final_outcomes'), ('paired_comparisons', 'final_method_comparisons'),
            ('per_seed_comparisons', 'final_seed_method_effects'), ('interval_comparisons', 'final_interval_effects'),
            ('per_seed_interval_comparisons', 'final_seed_interval_effects')]:
        frames[source].query('fleet_budget == @budget').to_csv(out / (name + '.csv'), index=False)
    frames['communication_summary'].to_csv(out / 'communication_summary.csv', index=False)
    final = frames['budget_contexts'].query('fleet_budget == @budget').merge(
        pd.DataFrame(state['cfg']['context_metadata'])[['id', 'frequency', 'speed']],
        left_on='context', right_on='id', validate='many_to_one')
    final.groupby(['interval', 'method', 'frequency'], sort=False).agg(
        encountered_contexts=('success', 'size'), successes=('success', 'sum'), eligible=('eligible', 'sum'),
        pooled_success_percent=('success', lambda x: 100 * x.mean())).reset_index().to_csv(out / 'frequency_counts.csv', index=False)
    missing = final.assign(no_local_incumbent=final.candidate.isna(),
        heldout_ineligible=final.candidate.notna() & ~final.eligible)
    missing.groupby(['interval', 'method'], sort=False)[['no_local_incumbent', 'heldout_ineligible']].sum().reset_index().to_csv(out / 'missing_recommendations.csv', index=False)
    visits = frames['paid_visits'].merge(pd.DataFrame(state['cfg']['context_metadata'])[['id', 'frequency']],
        left_on='context', right_on='id', suffixes=('', '_metadata'), validate='many_to_one')
    visits.groupby(['interval', 'method', 'mode'], sort=False).agg(requests=('eligible', 'size'),
        eligible=('eligible', 'sum'), new_incumbents=('became_incumbent', 'sum')).reset_index().to_csv(out / 'query_allocation.csv', index=False)
    visits[~visits.eligible].groupby(['interval', 'method', 'frequency'], sort=False).size().rename('failed_paid_requests').reset_index().to_csv(out / 'failed_contexts.csv', index=False)
    trials = frames['budget_trials'].query('fleet_budget == @budget')
    sensitivity = []
    for tolerance in (.025, .05, .10, .20):
        alt = final.assign(diagnostic_success=final.eligible & final.reference_available & (final.relative_gap <= tolerance))
        units = alt.groupby(['interval', 'method', 'seed', 'client']).diagnostic_success.mean()
        for interval in (1, 2, 4):
            for baseline in (A, B, E):
                effects = (units.loc[interval, D] - units.loc[interval, baseline]).groupby('seed').mean()
                sensitivity.extend(dict(interval=interval, tolerance=tolerance, baseline=baseline, seed=int(seed),
                    success_delta_pp=100 * float(effect), posthoc_diagnostic=True) for seed, effect in effects.items())
    pd.DataFrame(sensitivity).to_csv(out / 'posthoc_threshold_sensitivity.csv', index=False)
    # Cross-configuration comparisons are descriptive, not an isolation of
    # cost-sharing alone; both catalogue and cost contacts change together.
    cross = []
    for baseline in (A, B, E):
        target = trials[(trials.method == D) & (trials.interval == 2)].set_index(['seed', 'client']).sort_index()
        reference = trials[(trials.method == baseline) & (trials.interval == 1)].set_index(['seed', 'client']).sort_index()
        if not target.index.equals(reference.index) or not target.encountered_contexts.equals(reference.encountered_contexts):
            raise RuntimeError('Cross-configuration programmes differ.')
        traffic = frames['communication_summary'].set_index(['interval', 'method'])
        cross.append(dict(method=D, interval=2, baseline=baseline, baseline_interval=1,
            success_delta_pp=100 * float((target.context_success - reference.context_success).mean()),
            application_bytes_change_percent=100 * (traffic.loc[(2, D), 'used_application_bytes'] / traffic.loc[(1, baseline), 'used_application_bytes'] - 1),
            packet_deliveries_change_percent=100 * (traffic.loc[(2, D), 'used_packet_deliveries'] / traffic.loc[(1, baseline), 'used_packet_deliveries'] - 1)))
    pd.DataFrame(cross).to_csv(out / 'cross_configuration_comparisons.csv', index=False)


def figures(frames, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(zip(METHODS, ['#64748b', '#b58136', '#007d75', '#8265ad']))
    final = frames['online_summary'].query('fleet_budget == 216')
    traffic = frames['communication_summary'].query('method == @D').sort_values('interval')
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9), constrained_layout=True)
    for method in METHODS:
        rows = final[final.method == method].sort_values('interval')
        axes[0].plot(rows.interval, rows.context_success_percent, marker='o', color=colors[method], label=LETTERS[method])
    axes[0].set(xlabel='Contact interval [rounds]', ylabel='Encountered-context success [%]', xticks=[1, 2, 4], ylim=(0, 35))
    axes[0].legend(ncol=4, frameon=False)
    x = np.arange(3)
    peer = traffic.used_peer_application_bytes.to_numpy() / 1000
    cost = traffic.used_cost_application_bytes.to_numpy() / 1000
    axes[1].bar(x, peer, color='#b9c8cb', label='Peer catalogue')
    axes[1].bar(x, cost, bottom=peer, color=colors[D], label='Eligible costs')
    for j, value in enumerate(peer + cost):
        axes[1].text(j, value + 8, f'{value:.1f}', ha='center', fontsize=9)
    axes[1].set(xlabel='Contact interval [rounds]', ylabel='D application traffic [decimal kB/fleet]', xticks=x, xticklabels=['1', '2', '4'], ylim=(0, 530))
    axes[1].legend(frameon=False, fontsize=9)
    effects = frames['per_seed_comparisons'].query('fleet_budget == 216 and baseline == @E')
    for seed, group in effects.groupby('seed'):
        axes[2].plot(group.interval, group.success_delta_pp, marker='o', alpha=.7, label=str(seed))
    axes[2].axhline(0, color='#778899', lw=.8)
    axes[2].set(xlabel='Contact interval [rounds]', ylabel='D minus E success [percentage points]', xticks=[1, 2, 4])
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    for extension in ('svg', 'png'):
        path = out / ('quality_traffic_seed_effects.' + extension)
        fig.savefig(path, dpi=170)
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


def delta_catalogue_diagnostic(state, results, out):
    """Posthoc serialized-byte diagnostic; preserve every contact/cache entry.

    This does not change the live runner or claim a new online experiment.
    The fixed fleet has no removals, so changed best-eligible records suffice.
    Empty packet headers and all recipient deliveries are still charged.
    """
    rows, checked = [], 0
    domain, n = state['domain'], len(state['clients'])
    for result in results:
        for method, data in result['state']['methods'].items():
            cached, used, all_bytes, updates = {}, 0, 0, 0
            for contact in data['network']['contacts']:
                r, outgoing = contact['round'], []
                for client, history in enumerate(data['histories']):
                    records = []
                    for context in domain.contexts:
                        good = [(q + 1, o) for q, o in enumerate(history[:r]) if o['eligible'] and domain.actions[o['action']][0] == context]
                        if good:
                            trial, observation = min(good, key=lambda x: (x[1]['J'], domain.actions[x[1]['action']][1]))
                            event = data['events'][(trial - 1) * n + client]
                            records.append((client, trial, observation['action'], MODES.index(event['mode']), observation['J']))
                    outgoing.append(records)
                full, statistics = exchange(PEER, r, outgoing, n, len(domain.actions))
                if statistics != contact['peer']:
                    raise RuntimeError('Independent full catalogue reconstruction differs.')
                changes = [[rec for rec in records if cached.get((rec[0], domain.actions[rec[2]][0])) != rec] for records in outgoing]
                delta, statistics = exchange(PEER, r, changes, n, len(domain.actions))
                for rec in delta:
                    cached[rec[0], domain.actions[rec[2]][0]] = rec
                if cached != {(rec[0], domain.actions[rec[2]][0]): rec for rec in full}:
                    raise RuntimeError('Delta catalogue loses a cache entry.')
                checked += 1
                cost = contact['cost']['application_bytes'] if contact['cost'] else 0
                all_bytes += statistics['application_bytes'] + cost
                if r < state['cfg']['queries_per_client']:
                    used += statistics['application_bytes'] + cost
                    updates += len(delta)
            rows.append(dict(interval=result['interval'], seed=result['seed'], method=method,
                full_catalogue_used_bytes=data['snapshots'][str(state['cfg']['fleet_budgets'][-1])]['payload_bytes'],
                delta_catalogue_used_bytes=used, delta_catalogue_all_bytes=all_bytes, changed_peer_records_used=updates,
                posthoc_diagnostic=True, cache_equality_at_every_contact=True))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'delta_catalogue_diagnostic.csv', index=False)
    frame.groupby(['interval', 'method'], sort=False)[['full_catalogue_used_bytes', 'delta_catalogue_used_bytes',
        'delta_catalogue_all_bytes', 'changed_peer_records_used']].mean().reset_index().to_csv(out / 'delta_catalogue_summary.csv', index=False)
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    state = runner.setup()
    OUT.mkdir(parents=True, exist_ok=True)
    source_files = sorted(p for p in state['out'].rglob('*') if p.is_file())
    fingerprints = {p.relative_to(runner.REPO).as_posix(): dense.sha(p) for p in source_files}
    saved = runner.base.read(state['out'] / 'run_manifest.json')
    if runner.body_hash({k: v for k, v in saved.items() if k not in ('signature', 'git_head', 'runtime')}) != state['signature']:
        raise RuntimeError('Saved run manifest differs.')
    if runner.base.read(state['out'] / 'completion.json')['signature'] != state['signature']:
        raise RuntimeError('Pilot is incomplete.')
    with patch('src.tracking_budget.minimize', dense.prohibited), patch('src.tracking_six_speed.minimize', dense.prohibited), patch.object(runner.base, 'resolved_rollout', dense.prohibited):
        audit = runner.audit_setup(state)
        completion = runner.base.read(audit['out'] / 'completion.json')
        if completion['signature'] != audit['audit_signature'] or not completion['all_checks_passed']:
            raise RuntimeError('Frozen equivalence gate differs.')
        results = runner.completed_live(state, replay=True, workers=args.workers)
        frames, raw, checks, source_checks = reconstruct(state, results, audit)
        with tempfile.TemporaryDirectory(prefix='communication_direct_gp_') as directory:
            payload = dict(runner.payload(state), analysis_out=Path(directory))
            runner.base.parallel(direct_worker, state['jobs'], payload, args.workers, 'Independent causal GP checks')
            posterior = pd.concat([pd.read_csv(p) for p in sorted(Path(directory).glob('*.csv'))], ignore_index=True)
        posterior.to_csv(OUT / 'direct_gp_checks.csv', index=False)
        diagnostics(state, frames, OUT)
        figures(frames, OUT)
        delta_checks = delta_catalogue_diagnostic(state, results, OUT)
    for path, before in fingerprints.items():
        if dense.sha(runner.REPO / path) != before:
            raise RuntimeError('Analysis altered a source result: ' + path)
    source_commit = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--',
        state['out'].relative_to(runner.REPO).as_posix()], cwd=runner.REPO, text=True).strip()
    verification = dict(signature=state['signature'], source_commit=source_commit, paid_decisions_replayed=len(frames['paid_visits']),
        tables_verified=checks, independent_GP_predictions=len(posterior),
        maximum_mean_error=float(posterior.absolute_mean_error.max()), maximum_variance_error=float(posterior.absolute_variance_error.max()),
        maximum_acquisition_error=float(posterior.absolute_acquisition_error.max()),
        raw_validation_rollouts=len(raw), ineligible_validation_rollouts=int((~raw.eligible).sum()),
        unresolved_validation_rollouts=int((~raw.integration_resolved).sum()), source_checks=source_checks,
        new_rollouts=0, new_hyperparameter_optimization=0, source_results_unchanged=True, source_hashes=fingerprints,
        posthoc_delta_catalogue_cache_checks=delta_checks,
        scope='Exploratory three programme/noise seeds on one homogeneous plant; no noninferiority claim.')
    runner.base.save(OUT / 'verification.json', verification)
    print(f'Verified {verification["paid_decisions_replayed"]:,} choices, {len(checks)} tables, '
        f'{len(posterior):,} independent GP predictions and {len(raw):,} validation rollouts. Source results unchanged.', flush=True)


if __name__ == '__main__':
    main()
