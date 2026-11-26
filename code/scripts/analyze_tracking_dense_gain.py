"""Read-only paid replay, direct-GP checks and consolidation of dense-gain BO."""
import argparse
import hashlib
import io
from pathlib import Path
import tempfile
from unittest.mock import patch

import tracking_dense_gain as runner
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import analyze_tracking_gain_domain as source_audit
from src.tracking_budget import matern
from src.tracking_confirmation import calibration_reference
from src.tracking_dense_gain import A, B, D, E, METHODS, BONUSES
from src.tracking_dense_gain_report import write_report
from src.tracking_kernel_features import query_posterior
from src.tracking_six_speed_report import evaluate

OUT = runner.REPO / 'docs/analysis/dense_gain'
LABELS = {A: 'A', B: 'B', D: 'D', E: 'E'}


def prohibited(*args, **kwargs):
    raise RuntimeError('Read-only analysis attempted new simulation or optimization.')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_table(state, name, expected, keys):
    expected = pd.read_csv(io.StringIO(expected.to_csv(index=False)))
    actual = pd.read_csv(state['out'] / (name + '.csv'))
    if expected.duplicated(keys).any() or actual.duplicated(keys).any():
        raise RuntimeError('Duplicate saved table keys: ' + name)
    expected, actual = (f.sort_values(keys).reset_index(drop=True) for f in (expected, actual))
    assert_frame_equal(expected, actual, check_dtype=False, check_like=True, rtol=2e-10, atol=2e-10)
    return dict(table=name, rows=len(expected), verified=True)


def references_and_validation(state, results):
    for name, expected in state['specification']['reference_input_hashes'].items():
        if runner.base.digest(runner.REPO / name) != expected:
            raise RuntimeError('Archived evaluator input changed: ' + name)
    source = source_audit.load_state()
    if source['signature'] != state['specification']['source_audit_signature']:
        raise RuntimeError('Source audit signature differs.')
    source_checks = source_audit.runner.verify_source_checks(source)
    bank = source_audit.runner.load_calibration(source)
    allowed = set(state['domain'].candidates)
    chosen = {c: calibration_reference({k: row for k, row in rows.items() if k in allowed}) for c, rows in bank.items()}
    expected = dict(signature=state['signature'], choices=chosen,
        reused_calibration_observations=len(state['pool']) * len(state['domain'].contexts) * 2,
        new_calibration_observations=0, training_use=False)
    if runner.base.read(state['out'] / 'reference_selection.json') != expected:
        raise RuntimeError('Archived calibration-only reference selection differs.')
    tasks = runner.evaluation_tasks(state, runner.base.validation_selection(state, results, chosen))
    actual = set((state['out'] / 'evaluation/validation').glob('*.json'))
    if actual != {runner.evaluation_path(state, t) for t in tasks}:
        raise RuntimeError('Missing or unexpected validation batches.')
    validation = runner.load_evaluation(state, tasks)
    raw = pd.DataFrame([dict(context=t['context'], repeat=t['repeat'], **row)
        for t in tasks for row in runner.base.read(runner.evaluation_path(state, t))['rows']])
    if raw.duplicated(['context', 'repeat', 'candidate']).any() or not (
            raw.groupby(['context', 'candidate'])['repeat'].apply(lambda x: sorted(x) == list(range(4)))).all():
        raise RuntimeError('Held-out repeat identities differ.')
    complete = raw[raw.complete]
    score, feasibility = state['physics']['score'], state['physics']['feasibility']
    np.testing.assert_allclose(complete.J, complete.rmse_y / score['lateral_scale_m'] +
        complete.rms_rate_deg_s / score['steer_rate_scale_deg_s'], rtol=2e-12, atol=2e-12)
    expected_eligible = raw.complete & (raw.peak_y <= feasibility['peak_lateral_error_m']) & (
        raw.peak_beta_deg <= feasibility['peak_beta_deg']) & (
        raw.peak_speed_error <= feasibility['peak_speed_error_m_s'])
    np.testing.assert_array_equal(raw.feasible, expected_eligible)
    raw['eligible'] = expected_eligible & raw.integration_resolved
    grouped = raw.groupby(['context', 'candidate'], sort=False)
    for key, rows in grouped:
        context, candidate = key
        saved = validation[context][candidate]
        if saved['eligible'] != bool(rows.eligible.all()):
            raise RuntimeError('Independent held-out eligibility differs.')
        if saved['J'] is not None:
            np.testing.assert_allclose(saved['J'], rows.J.mean(), rtol=2e-12, atol=2e-12)
    refs = {c: dict(candidate=k, **validation[c][k]) if k is not None else dict(candidate=None, eligible=False, J=None)
        for c, k in chosen.items()}
    return refs, validation, raw, source_checks


def reconstruct(state, results, refs, validation):
    trials, cells, visits, learning = evaluate(results, state['domain'], state['cfg'], refs, validation)
    visits['confidence_multiplier'] = visits.method.map(BONUSES)
    with tempfile.TemporaryDirectory(prefix='dense_gain_analysis_') as directory:
        temporary = dict(state, out=Path(directory))
        frames = write_report(temporary, trials, cells, visits, learning, refs)
        expected_readout = runner.base.read(Path(directory) / 'pilot_readout.json')
    if expected_readout != runner.base.read(state['out'] / 'pilot_readout.json'):
        raise RuntimeError('Predeclared descriptive readout differs.')
    keys = dict(budget_trials=['seed', 'method', 'fleet_budget', 'client'],
        budget_contexts=['seed', 'method', 'fleet_budget', 'client', 'context'],
        paid_visits=['seed', 'method', 'round', 'client'], learning_diagnostics=['seed', 'method', 'round'],
        online_summary=['method', 'fleet_budget'], paired_comparisons=['method', 'baseline', 'fleet_budget'],
        per_seed_comparisons=['method', 'baseline', 'fleet_budget', 'seed'],
        frequency_units=['method', 'seed', 'client', 'fleet_budget', 'frequency'],
        frequency_outcomes=['method', 'fleet_budget', 'frequency'], query_modes=['method', 'seed', 'mode'],
        per_seed_outcomes=['method', 'seed'], visit_summary=['method', 'seed', 'group'], empirical_references=['context'])
    checks = [verify_table(state, name, frame, keys[name]) for name, frame in frames.items()]
    saved = runner.base.read(state['out'] / 'verification.json')
    expected = dict(signature=state['signature'], mode='pilot', paid_requests=len(visits),
        all_local_caps_respected=bool((visits.groupby(['seed', 'method', 'client']).size() == state['cfg']['queries_per_client']).all()),
        all_paid_choices_replayed=True, new_calibration_rollouts=0,
        reused_calibration_observations=len(state['pool']) * len(state['domain'].contexts) * 2,
        validation_rollouts=sum(len(t['candidates']) for t in runner.evaluation_tasks(state,
            runner.base.validation_selection(state, results, {c: r['candidate'] for c, r in refs.items()}))),
        historical_training_records=0, unavailable_references=sum(not r['eligible'] for r in refs.values()),
        unresolved_online_requests=int((~visits.integration_resolved).sum()), confirmation_claim=False)
    if expected != saved:
        raise RuntimeError('Saved verification metadata differs from the reconstructed run.')
    return frames, checks


def check_recorded_posteriors(state, results):
    """Independent observation-space GP checks of every D/E BO prediction.

    Only recorded earlier-round paid observations are used. Small subkernels
    cover observations and the actual queries, never the full public grid.
    """
    rows = []
    domain, cfg, n = state['domain'], state['cfg'], len(state['clients'])
    for result in results:
        for method in (D, E):
            data = result['state']['methods'][method]
            for previous_round in range(cfg['queries_per_client']):
                events = data['events'][previous_round * n:(previous_round + 1) * n]
                queries = [e for e in events if e['mode'] == 'feasibility_penalized_LCB']
                if not queries:
                    continue
                histories = [h[:previous_round] for h in data['histories']]
                inputs = sorted({o['action'] for h in histories for o in h if o['eligible']} | {e['action'] for e in queries})
                positions = {a: i for i, a in enumerate(inputs)}
                kernel = matern(domain.X[inputs], domain.X[inputs], cfg['gp']['length_scales'],
                    cfg['gp']['amplitude'], cfg['gp']['constant_variance'])
                local = [[dict(o, action=positions[o['action']]) for o in h if o['eligible']] for h in histories]
                control = cfg['hierarchical'] if method == D else dict(sharing_fractions=[0.], prior_weights=[1.])
                expected = query_posterior(local, kernel, control, cfg['gp'],
                    [e['client'] for e in queries], [positions[e['action']] for e in queries])
                learned_weights = data['learning'][previous_round - 1]['sharing_weights']
                np.testing.assert_allclose(expected['sharing_weights'], learned_weights, rtol=2e-8, atol=2e-8)
                for j, event in enumerate(queries):
                    mean, variance = expected['mean'][j], expected['variance'][j]
                    np.testing.assert_allclose([mean, variance], [event['predicted_log_J'], event['predicted_variance']], rtol=2e-8, atol=2e-8)
                    acquisition = cfg['acquisition']
                    score = mean - BONUSES[method] * np.sqrt(variance) - acquisition['feasibility_penalty'] * np.log(
                        max(event['eligibility_probability'], acquisition['probability_floor']))
                    np.testing.assert_allclose(score, event['acquisition_score'], rtol=2e-8, atol=2e-8)
                    rows.append(dict(seed=result['seed'], method=method, round=event['round'], client=event['client'],
                        context=event['context'], candidate=event['candidate'], observed_eligible=sum(map(len, local)),
                        absolute_mean_error=abs(mean - event['predicted_log_J']),
                        absolute_variance_error=abs(variance - event['predicted_variance']),
                        absolute_weight_error=float(np.max(abs(expected['sharing_weights'] - learned_weights))),
                        absolute_acquisition_error=abs(score - event['acquisition_score'])))
    return pd.DataFrame(rows)


def diagnostics(state, results, frames, raw_validation, out):
    final_budget = state['cfg']['fleet_budgets'][-1]
    trials, cells, visits = (frames[k] for k in ('budget_trials', 'budget_contexts', 'paid_visits'))
    final = cells[cells.fleet_budget == final_budget].copy()
    final_trials = trials[trials.fleet_budget == final_budget].copy()
    metadata = pd.DataFrame(state['cfg']['context_metadata'])[['id', 'speed', 'frequency']]
    final = final.merge(metadata, left_on='context', right_on='id', validate='many_to_one')
    cost_pairs = []
    for baseline in (A, B, E):
        for seed, group in final_trials.groupby('seed', sort=True):
            a, b = (group[group.method == m].set_index('client').sort_index() for m in (D, baseline))
            common = a.complete_eligible_on_encountered & b.complete_eligible_on_encountered
            ma, mb = (float(frame.loc[common, 'J'].mean()) if common.any() else None for frame in (a, b))
            cost_pairs.append(dict(seed=int(seed), method=D, baseline=baseline,
                common_complete_pairs=int(common.sum()), method_mean_J=ma, baseline_mean_J=mb,
                paired_J_change_percent=100 * (ma / mb - 1) if ma is not None else None))
    pd.DataFrame(cost_pairs).to_csv(out / 'per_seed_paired_costs.csv', index=False)
    sensitivity = []
    for tolerance in (.025, .05, .10, .20):
        candidate = final.assign(diagnostic_success=final.eligible & final.reference_available & (final.relative_gap <= tolerance))
        units = candidate.groupby(['method', 'seed', 'client']).diagnostic_success.mean()
        for baseline in (A, B, E):
            effects = (units.loc[D] - units.loc[baseline]).groupby('seed').mean()
            for seed, effect in effects.items():
                sensitivity.append(dict(diagnostic_tolerance=tolerance, baseline=baseline, seed=int(seed),
                    success_delta_pp=float(100 * effect), posthoc_diagnostic=True, declared_metric_unchanged=True))
    pd.DataFrame(sensitivity).to_csv(out / 'posthoc_threshold_sensitivity.csv', index=False)
    for source, target in [('online_summary', 'final_outcomes'), ('paired_comparisons', 'final_comparisons'),
                           ('per_seed_comparisons', 'final_seed_effects')]:
        frames[source].query('fleet_budget == @final_budget').to_csv(out / (target + '.csv'), index=False)
    final.groupby(['method', 'seed', 'frequency'], sort=False).agg(contexts=('success', 'size'),
        successes=('success', 'sum'), eligible=('eligible', 'sum'),
        pooled_success_percent=('success', lambda v: 100 * v.mean()),
        mean_eligible_relative_gap=('relative_gap', 'mean'), median_eligible_relative_gap=('relative_gap', 'median')).reset_index().to_csv(out / 'seed_frequency_outcomes.csv', index=False)
    final.groupby(['method', 'frequency'], sort=False).agg(contexts=('success', 'size'),
        successes=('success', 'sum'), eligible=('eligible', 'sum'),
        minimum_eligible_relative_gap=('relative_gap', 'min'), median_eligible_relative_gap=('relative_gap', 'median'),
        mean_eligible_relative_gap=('relative_gap', 'mean')).reset_index().to_csv(out / 'frequency_outcomes.csv', index=False)
    visits.groupby(['method', 'mode'], sort=False).agg(requests=('eligible', 'size'),
        eligible=('eligible', 'sum'), validation_eligible=('requested_eligible', 'sum'),
        requested_successes=('requested_success', 'sum'), new_incumbents=('became_incumbent', 'sum')).reset_index().to_csv(out / 'query_allocation.csv', index=False)
    pool = {c['id']: c for c in state['pool']}
    regions = visits.copy()
    regions['outside_old_box'] = regions.candidate.map(lambda c: pool[c]['gains'][0] > .05)
    regions['original_71'] = regions.candidate.isin(state['pool'][i]['id'] for i in range(71))
    regions.groupby(['method', 'frequency', 'mode', 'outside_old_box'], sort=False).agg(requests=('eligible', 'size'),
        eligible=('eligible', 'sum'), requested_successes=('requested_success', 'sum'),
        distinct_candidate_contexts=('action', 'nunique')).reset_index().to_csv(out / 'query_regions.csv', index=False)
    final['outside_old_box'] = final.candidate.map(lambda c: pool[c]['gains'][0] > .05 if pd.notna(c) else False)
    final.groupby(['method', 'frequency', 'outside_old_box'], sort=False).agg(contexts=('success', 'size'),
        successes=('success', 'sum'), eligible=('eligible', 'sum')).reset_index().to_csv(out / 'recommendation_regions.csv', index=False)
    discoveries, failures, repeated = [], [], []
    for result in results:
        for method, record in result['state']['methods'].items():
            for frequency in (.1, .3, .6):
                group = visits[(visits.seed == result['seed']) & (visits.method == method) & (visits.frequency == frequency)]
                good = group[group.requested_success].sort_values('fleet_evaluation')
                first = good.iloc[0] if len(good) else None
                discoveries.append(dict(seed=result['seed'], method=method, frequency=frequency,
                    requests=len(group), near_reference_requests=len(good),
                    first_round=int(first['round']) if first is not None else None,
                    first_context=first['context'] if first is not None else None,
                    first_candidate=first['candidate'] if first is not None else None,
                    first_mode=first['mode'] if first is not None else None))
            for event in record['events']:
                if event['eligible']:
                    continue
                raw = result['state']['oracle_cache'][runner.base.cache_key(event)]['result']
                failures.append(dict(seed=result['seed'], method=method, client=event['client'], round=event['round'],
                    frequency=event['frequency'], context=event['context'], candidate=event['candidate'], mode=event['mode'],
                    failure=raw['failure'], complete=raw['complete'], integration_resolved=raw['integration_resolved'],
                    peak_y=raw['peak_y'], peak_beta_deg=raw['peak_beta_deg']))
                previous = [e for e in record['events'] if e['round'] < event['round'] and e['client'] != event['client']
                    and e['action'] == event['action'] and not e['eligible']]
                if event['mode'] == 'feasibility_penalized_LCB' and previous:
                    repeated.append(dict(seed=result['seed'], method=method, client=event['client'], round=event['round'],
                        context=event['context'], candidate=event['candidate'], prior_failed_peer_count=len(previous)))
    pd.DataFrame(discoveries).to_csv(out / 'first_discoveries.csv', index=False)
    failure_frame = pd.DataFrame(failures)
    failure_frame.to_csv(out / 'ineligible_requests.csv', index=False)
    failure_frame.groupby(['method', 'mode', 'failure']).size().rename('requests').reset_index().to_csv(out / 'failure_reasons.csv', index=False)
    pd.DataFrame(repeated, columns=['seed', 'method', 'client', 'round', 'context', 'candidate', 'prior_failed_peer_count']).to_csv(out / 'prior_peer_failure_repeats.csv', index=False)
    final_trials[final_trials.method == A][['seed', 'client', 'encountered_contexts', 'encountered_speeds']].to_csv(out / 'programme_coverage.csv', index=False)
    sharing = frames['learning_diagnostics']
    sharing = sharing[(sharing.method == D) & (sharing['round'] == state['cfg']['queries_per_client'])]
    sharing[['seed', 'active_support_size', 'expected_sharing_fraction', 'sharing_weights', 'component_log_evidence']].to_csv(out / 'final_sharing.csv', index=False)
    allocation = visits.groupby('method', sort=False).agg(requests=('eligible', 'size'),
        eligible=('eligible', 'sum'), pooled_requested_successes=('requested_success', 'sum'))
    readout = dict(primary_scope='Exploratory matched homogeneous three-programme-seed comparison.',
        confirmation_claim=False, final_success_percent={m: float(final_trials[final_trials.method == m].context_success.mean() * 100) for m in METHODS},
        pooled_final_successes={m: int(final[final.method == m].success.sum()) for m in METHODS},
        encountered_context_pairs_per_method=int((final.method == A).sum()),
        failed_paid_requests={m: int(allocation.loc[m, 'requests'] - allocation.loc[m, 'eligible']) for m in METHODS},
        mean_encountered_contexts=float(final_trials.encountered_contexts.mean()),
        mean_encountered_speeds=float(final_trials.encountered_speeds.mean()),
        unresolved_validation_rollouts=int((~raw_validation.integration_resolved).sum()),
        ineligible_validation_rollouts=int((~raw_validation.eligible).sum()),
        repeated_BO_failures_after_prior_peer_failure=len(repeated),
        interpretation='Average sharing and benchmark gains coexist with seed dependence, weak 0.1 Hz success and a B coverage tradeoff.')
    runner.base.save(out / 'readout.json', readout)


def figures(frames, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt'] = 'tracking_dense_gain_analysis'
    summary = frames['online_summary'].query('fleet_budget == 216').set_index('method').loc[METHODS]
    effects = frames['per_seed_comparisons'].query('fleet_budget == 216')
    colors = ['#66748a', '#aa7935', '#007d75', '#7864ad']
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), constrained_layout=True)
    for ax, key, title, ylabel in zip(axes, ['context_success_percent', 'complete_eligible_policies', 'failed_paid_requests'],
            ['Near-reference success', 'Complete eligible policies', 'Failed paid tests'], ['Success [%]', 'Client programmes [of 27]', 'Requests [of 648]']):
        values = summary[key].to_numpy()
        ax.bar(range(4), values, color=colors)
        ax.set(xticks=range(4), xticklabels=['A', 'B', 'D', 'E'], ylabel=ylabel, title=title)
        ax.set_ylim(0, max(values) * 1.18)
        for x, value in enumerate(values):
            ax.text(x, value + max(values) * .025, f'{value:.2f}' if key == 'context_success_percent' else str(int(value)), ha='center', fontsize=10)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    save_svg(fig, out / 'success_coverage_failures.svg')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.8, 3.8), constrained_layout=True)
    for i, baseline in enumerate((A, B, E)):
        group = effects[effects.baseline == baseline].sort_values('seed')
        ax.plot(range(3), group.success_delta_pp, marker=['o', 's', '^'][i], label=f'D minus {LABELS[baseline]}')
    ax.axhline(0, color='#6b747c', linewidth=.8)
    ax.set(xticks=range(3), xticklabels=['8100', '8101', '8102'], xlabel='Programme/noise seed',
        ylabel='Success change [percentage points]', title='Paired effects differ across programmes')
    ax.legend(fontsize=9)
    ax.grid(alpha=.18)
    save_svg(fig, out / 'paired_seed_effects.svg')
    plt.close(fig)


def save_svg(fig, path):
    fig.savefig(path, metadata={'Date': None})
    path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    state = runner.setup()
    if args.out.resolve().is_relative_to(state['out'].resolve()):
        raise RuntimeError('Analysis output must be outside the frozen result directory.')
    manifest = runner.base.read(state['out'] / 'run_manifest.json')
    body = {k: v for k, v in manifest.items() if k not in ('signature', 'git_head', 'runtime')}
    if runner.body_hash(body) != manifest['signature'] or body != state['specification'] or manifest['signature'] != state['signature']:
        raise RuntimeError('Saved and generated source specifications differ.')
    for name, expected in state['specification']['source_hashes'].items():
        if runner.base.digest(runner.REPO / name) != expected:
            raise RuntimeError('Frozen source changed: ' + name)
    paths = {p for p in state['out'].rglob('*') if p.is_file()}
    paths.update(runner.REPO / name for name in state['specification']['reference_input_hashes'])
    hashes = {p.relative_to(runner.REPO).as_posix(): sha(p) for p in sorted(paths)}
    with patch.object(runner.base, 'resolved_rollout', prohibited), patch('src.tracking_budget.minimize', prohibited), patch('src.tracking_six_speed.minimize', prohibited):
        results = runner.completed_live(state, replay=True, workers=args.workers)
        refs, validation, raw, source_checks = references_and_validation(state, results)
        frames, table_checks = reconstruct(state, results, refs, validation)
        direct_checks = check_recorded_posteriors(state, results)
        args.out.mkdir(parents=True, exist_ok=True)
        direct_checks.to_csv(args.out / 'direct_GP_checks.csv', index=False)
        diagnostics(state, results, frames, raw, args.out)
        figures(frames, args.out)
    if any(sha(runner.REPO / name) != expected for name, expected in hashes.items()):
        raise RuntimeError('A source result changed during read-only analysis.')
    if runner.base.read(state['out'] / 'completion.json')['signature'] != state['signature']:
        raise RuntimeError('Incomplete source result.')
    runner.base.save(args.out / 'verification.json', dict(source_signature=state['signature'],
        analysis_source_sha256=runner.base.digest(Path(__file__)),
        source_results_commit='a7955c4d730fe0dc26642cf6e22ab0c58eea4ff4', paid_decisions=len(frames['paid_visits']),
        saved_tables=table_checks, archived_calibration_observations=46512,
        archived_source_check_observations=source_checks, fresh_validation_observations=len(raw),
        direct_GP_BO_checks=len(direct_checks),
        maximum_direct_GP_mean_error=float(direct_checks.absolute_mean_error.max()),
        maximum_direct_GP_variance_error=float(direct_checks.absolute_variance_error.max()),
        maximum_direct_GP_weight_error=float(direct_checks.absolute_weight_error.max()),
        original_inputs_unchanged=True, new_simulations=0, new_optimization=0,
        replay_scope='A/B restore logged pre-query fit states; D/E predictions/decisions rebuilt from earlier paid histories.',
        confirmation_claim=False, input_fingerprints=hashes))
    print(f'Verified {len(frames["paid_visits"]):,} paid decisions, {len(table_checks)} tables and {len(direct_checks)} direct GP predictions.', flush=True)
    print('Analysis: ' + str(args.out), flush=True)


if __name__ == '__main__':
    main()
