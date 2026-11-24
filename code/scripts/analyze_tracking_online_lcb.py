"""Read-only audit of the frozen one-test LCB online calibration pilot.

No new physical rollout, acquisition decision or optimization campaign is run.
Federated regression is replayed on the saved observations for a same-data
comparison with its exact ridge optimum. Peer alternatives are scored only
retrospectively on the existing independent validation bank.
"""
from pathlib import Path
import json

import tracking_online_lcb as runner
import numpy as np
import pandas as pd

from src.tracking_budget import dense_references, incumbents
from src.tracking_online import (LocalCostModel, centralized_fit, features,
                                 federated_fit, optimization_gap, peer_candidate)
from src.tracking_online_lcb_report import eligibility_times, evaluate, summarize, summarize_times


OUT = runner.REPO/'docs/analysis/online_calibration_lcb'


def verify():
    state = runner.setup()
    manifest = runner.read(state['out']/'run_manifest.json')
    completion = runner.read(state['out']/'completion.json')
    if manifest['signature'] != completion['signature'] or state['signature'] != manifest['signature']:
        raise RuntimeError('Completed online-run signature differs.')
    results = []
    for job in runner.jobs(state):
        result = runner.read(runner.path_for(state, job))
        runner.validate_job(state, job, result, complete=True)
        results.append(result)
    if len(results) != completion['jobs'] or len(results) != 6:
        raise RuntimeError('The prescribed six-job pilot is incomplete.')
    raw = pd.read_csv(state['archive']/'controlled_evaluations.csv', float_precision='round_trip')
    tables = dense_references(raw[raw.role == 'calibration'], raw[raw.role == 'validation'], state['domain'])
    if not all(v['eligible'] for v in tables[0].values()):
        raise RuntimeError('An empirical reference fails validation.')
    trials, contexts, queries, learning = evaluate(results, state['domain'], state['cfg'], tables)

    def compare(frame, name, keys):
        saved = pd.read_csv(state['out']/name)
        a = frame.sort_values(keys).reset_index(drop=True)
        b = saved[a.columns].sort_values(keys).reset_index(drop=True)
        if 'eligible_training_counts' in a.columns:
            a['eligible_training_counts'] = a.eligible_training_counts.map(json.dumps)
            b['eligible_training_counts'] = b.eligible_training_counts.map(lambda x: json.dumps(json.loads(x)))
        # Normalize absent objective/recommendation values before comparison.
        a = a.astype(object).where(pd.notna(a), None)
        b = b.astype(object).where(pd.notna(b), None)
        pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-12)

    compare(trials, 'budget_trials.csv', ['condition', 'seed', 'method', 'fleet_budget', 'client'])
    compare(contexts, 'budget_contexts.csv', ['condition', 'seed', 'method', 'fleet_budget', 'client', 'context'])
    summary = summarize(trials)
    compare(summary, 'online_summary.csv', ['condition', 'method', 'fleet_budget'])
    compare(queries, 'query_observations.csv', ['condition', 'seed', 'method', 'fleet_evaluation'])
    compare(learning, 'learning_diagnostics.csv', ['condition', 'seed', 'method', 'round'])
    times, context_times = eligibility_times(results, state['domain'], state['cfg'])
    compare(times, 'eligibility_times.csv', ['condition', 'seed', 'method', 'client'])
    compare(context_times, 'context_eligibility_times.csv',
            ['condition', 'seed', 'method', 'client', 'context'])
    compare(summarize_times(times), 'eligibility_timing_summary.csv', ['condition', 'method'])
    if (len(trials), len(contexts), len(times), len(context_times)) != (1944, 11664, 324, 1944):
        raise RuntimeError('Unexpected early-budget or eligibility denominator.')
    if manifest['historical_training_records'] != 0 or state['cfg']['initialization_tests'] != 1:
        raise RuntimeError('The prescribed empty-history, one-test protocol changed.')
    if len(queries) != 7776:
        raise RuntimeError('Unexpected query count.')
    if queries.loc[~queries.integration_resolved, 'eligible'].any():
        raise RuntimeError('A numerically unresolved rollout entered the eligible training set.')
    return state, results, tables, trials, contexts, queries, learning, summary


def mechanism_diagnostics(state, results, tables):
    phi = features(state['domain'], state['cfg']['model'])
    reference, validation = tables
    prediction_rows, alternatives = [], []
    for job in results:
        data = job['state']['methods']['federated_cost_bo']
        histories = [[] for _ in job['clients']]
        weights = np.zeros(phi.shape[1])
        for round_id in range(1, state['cfg']['queries_per_client']+1):
            events = [e for e in data['events'] if e['round'] == round_id]
            # Every alternative below uses only previous-round Fed-arm records.
            for event in events:
                if event['local_evaluation'] <= state['cfg']['initialization_tests']:
                    continue
                i, context = event['client'], event['context']
                candidate = peer_candidate(histories, state['domain'], i, context)
                current = incumbents(histories[i], state['domain'])[state['domain'].contexts.index(context)]
                plant = job['clients'][i]['plant_id']
                row = dict(condition=job['condition'], seed=job['seed'], round=round_id,
                    client=job['clients'][i]['id'], plant_id=plant, context=context, mode=event['mode'],
                    local_incumbent_missing=current is None, peer_available=candidate is not None,
                    chosen_candidate=event['candidate'], peer_candidate=None,
                    peer_validation_eligible=None, peer_validation_success=None,
                    peer_better_than_chosen_on_validation=None)
                if candidate is not None:
                    label = state['domain'].actions[candidate][1]
                    peer = validation[plant, context, label]
                    chosen = validation[plant, context, event['candidate']]
                    row.update(peer_candidate=label, peer_validation_eligible=peer['eligible'],
                        peer_validation_success=bool(peer['eligible'] and
                            peer['J']/reference[plant, context]['J']-1 <= state['cfg']['relative_tolerance']),
                        peer_better_than_chosen_on_validation=bool(peer['eligible'] and
                            (not chosen['eligible'] or peer['J'] < chosen['J']-1e-12)))
                alternatives.append(row)
            for event in events:
                histories[event['client']].append({k: event[k] for k in
                    ('action', 'eligible', 'J', 'complete', 'feasible', 'integration_resolved', 'substeps')})
            local = [LocalCostModel(h, phi, state['cfg']['model']['ridge']) for h in histories]
            weights, _ = federated_fit(local, weights, state['cfg']['model'])
            exact = centralized_fit(local)
            gap = optimization_gap(local, weights)
            old = next(r['optimization_gap'] for r in data['learning'] if r['round'] == round_id)
            if not np.isclose(gap, old, rtol=1e-7, atol=1e-11):
                raise RuntimeError('FedAvg same-data regression replay differs from recorded training.')
            difference = phi@(weights-exact)
            prediction_rows.append(dict(condition=job['condition'], seed=job['seed'], round=round_id,
                optimization_gap=gap, same_data_mean_log_J_prediction_RMS=float(np.sqrt(np.mean(difference**2))),
                same_data_max_log_J_prediction_difference=float(np.abs(difference).max())))
        np.testing.assert_allclose(weights, data['weights'], rtol=1e-7, atol=1e-9)
    return pd.DataFrame(prediction_rows), pd.DataFrame(alternatives)


def main():
    state, results, tables, trials, contexts, queries, learning, summary = verify()
    predictions, alternatives = mechanism_diagnostics(state, results, tables)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT/'budget_summary.csv', index=False)
    predictions.to_csv(OUT/'same_data_regression.csv', index=False)
    alternatives.to_csv(OUT/'retrospective_peer_alternatives.csv', index=False)
    queries[~queries.integration_resolved].to_csv(OUT/'numerically_unresolved_queries.csv', index=False)
    times, context_times = eligibility_times(results, state['domain'], state['cfg'])
    timing = summarize_times(times)
    restricted = times.assign(restricted_queries=times.first_all_contexts_query.fillna(
        state['cfg']['queries_per_client'])).groupby(['condition', 'method']).restricted_queries.mean()
    timing['mean_queries_to_full_coverage_truncated_at_cap'] = [restricted[c, m]
        for c, m in zip(timing.condition, timing.method)]
    timing.to_csv(OUT/'coverage_timing.csv', index=False)
    first_actions = queries[queries.local_evaluation <= 3].pivot(
        index=['condition', 'seed', 'client', 'local_evaluation'], columns='method', values='action').reset_index()
    first_actions['federated_differs_from_independent'] = (
        first_actions.federated_cost_bo != first_actions.independent_bo)
    first_actions['federated_differs_from_centralized'] = (
        first_actions.federated_cost_bo != first_actions.centralized_cost_bo)
    first_actions.to_csv(OUT/'first_three_query_actions.csv', index=False)
    queries[queries.local_evaluation > 1].groupby(['condition', 'method', 'mode']).agg(
        queries=('eligible', 'size'), eligible=('eligible', 'sum'),
        conditional_mean_observed_J=('J', 'mean')).reset_index().to_csv(OUT/'adaptive_query_modes.csv', index=False)
    final = trials[trials.fleet_budget == 216]
    final.groupby(['condition', 'seed', 'method']).agg(context_success=('context_success', 'mean'),
        eligible_vehicle_policies=('fully_feasible', 'sum'), all_contexts_success=('all_contexts_success', 'sum'),
        conditional_mean_J=('J', 'mean'), failed_tests=('failed_queries', 'sum')).reset_index().to_csv(
            OUT/'per_seed_final.csv', index=False)
    final[final.condition == 'heterogeneous'].groupby(['family', 'method']).agg(
        context_success=('context_success', 'mean'), eligible_policy_rate=('fully_feasible', 'mean'),
        conditional_mean_J=('J', 'mean')).reset_index().to_csv(OUT/'per_family_final.csv', index=False)
    missing = contexts[(contexts.fleet_budget == 216) & contexts.candidate.isna()]
    missing[['condition', 'seed', 'method', 'plant_id', 'context']].to_csv(OUT/'missing_final_contexts.csv', index=False)
    contexts[(contexts.fleet_budget == 216) & contexts.candidate.notna() & ~contexts.eligible].to_csv(
        OUT/'selected_gains_failing_validation.csv', index=False)
    paired = []
    for condition, group in final.groupby('condition'):
        a = group[group.method == 'federated_cost_bo'].set_index(['seed', 'client'])
        for baseline in state['cfg']['methods']:
            if baseline == 'federated_cost_bo':
                continue
            b = group[group.method == baseline].set_index(['seed', 'client']).loc[a.index]
            good = a.fully_feasible & b.fully_feasible
            paired.append(dict(condition=condition, baseline=baseline, matched_trials=len(a),
                jointly_feasible_pairs=int(good.sum()), context_success_delta=float((a.context_success-b.context_success).mean()),
                mean_J_delta_on_jointly_feasible_pairs=float((a.loc[good, 'J']-b.loc[good, 'J']).mean())
                    if good.any() else None,
                federated_mean_J_jointly_feasible=float(a.loc[good, 'J'].mean()) if good.any() else None,
                baseline_mean_J_jointly_feasible=float(b.loc[good, 'J'].mean()) if good.any() else None,
                relative_reduction_of_joint_mean_J=float(1-a.loc[good, 'J'].mean()/b.loc[good, 'J'].mean())
                    if good.any() else None,
                relative_J_improvement_on_jointly_feasible_pairs=float((1-a.loc[good, 'J']/b.loc[good, 'J']).mean())
                    if good.any() else None))
    pd.DataFrame(paired).to_csv(OUT/'paired_final.csv', index=False)
    old = pd.read_csv(runner.ROOT/'results/tracking_online_v1/online_summary.csv')
    common = ['context_success_rate', 'eligible_context_coverage', 'fully_feasible_policy_rate',
              'mean_fleet_failed_tests']
    # The old report has no explicit context-coverage column; keep only shared metrics.
    common = [c for c in common if c in old.columns]
    comparison = summary.merge(old[['condition', 'method', 'fleet_budget', *common]],
        on=['condition', 'method', 'fleet_budget'], suffixes=('_one_test_LCB', '_six_test_EI'))
    comparison.to_csv(OUT/'previous_protocol_comparison.csv', index=False)
    fingerprints = [state['out']/name for name in ('run_manifest.json', 'completion.json', 'budget_trials.csv',
        'budget_contexts.csv', 'query_observations.csv', 'learning_diagnostics.csv',
        'eligibility_times.csv', 'context_eligibility_times.csv', 'eligibility_timing_summary.csv')]
    fingerprints += [runner.ROOT/'results/tracking_online_v1/online_summary.csv']
    fingerprints += [runner.path_for(state, job) for job in runner.jobs(state)]
    runner.save(OUT/'verification.json', dict(signature=state['signature'], jobs=len(results),
        result_commit=runner.subprocess.check_output(['git', 'log', '-1', '--format=%H', '--',
            'code/results/tracking_online_lcb_v1/completion.json'], cwd=runner.REPO, text=True).strip(),
        analysis_runtime=dict(python=runner.platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
        charged_queries=len(queries), fresh_simulator_records=sum(len(j['state']['oracle_cache']) for j in results),
        numerically_unresolved_queries=int((~queries.integration_resolved).sum()),
        unresolved_queries_excluded_from_eligible_training=True,
        trial_rows=len(trials), context_rows=len(contexts), validation_rescoring_matches=True,
        first_eligibility_and_censoring_tables_reproduced=True,
        initialization_and_round_barriers_verified=True, noise_and_simulator_records_verified=True,
        same_data_fedavg_training_replayed=True, scout_budgets_verified=True, new_simulator_rollouts=0,
        input_hashes={p.relative_to(runner.REPO).as_posix(): runner.digest(p) for p in fingerprints},
        analysis_source_sha256=runner.digest(Path(__file__))))
    print(f'Verified {len(results)} jobs / {len(queries):,} charged queries. Analysis: {OUT}')
    print(predictions[predictions['round'].between(1, 23)].groupby('condition')[[
        'optimization_gap', 'same_data_mean_log_J_prediction_RMS']].agg(['median', 'max']).to_string())
    print(summary[summary.fleet_budget == 216][['condition', 'method', 'context_success_rate',
        'fully_feasible_policy_rate', 'conditional_mean_J', 'mean_fleet_failed_tests',
        'mean_payload_bytes']].to_string(index=False))
    usable = alternatives[alternatives.peer_available]
    print(usable.groupby(['condition', 'local_incumbent_missing']).agg(
        opportunities=('peer_available', 'size'), peer_eligible=('peer_validation_eligible', 'sum'),
        peer_success=('peer_validation_success', 'sum'),
        peer_better_than_chosen=('peer_better_than_chosen_on_validation', 'sum')).to_string())


if __name__ == '__main__':
    main()
