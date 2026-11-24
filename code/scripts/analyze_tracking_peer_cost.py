"""Audit completed peer-assisted calibration and diagnose saved online histories.

No physical trials or new full optimization campaign are run. The optional
candidate comparison replays BO proposals on arm B's fixed histories and scores
them retrospectively on the independent validation bank, without feeding back
the hypothetical outcomes.
"""
from pathlib import Path
import json

import tracking_peer_cost as runner
import numpy as np
import pandas as pd

from src.tracking_budget import Posterior, dense_references, feasibility_probability
from src.tracking_online import LocalCostModel, centralized_fit, features, federated_fit, optimization_gap, seed_for
from src.tracking_online_lcb import acquisition_values
from src.tracking_peer_cost import METHODS, OBSERVATION_FIELDS
from src.tracking_peer_cost_report import (eligibility_times, evaluate, mechanism_tables,
    paired_comparisons, summarize, summarize_times)

OUT = runner.REPO/'docs/analysis/peer_assisted_cost_model'


def verify():
    state = runner.setup()
    manifest, completion = [runner.read(state['out']/name) for name in ('run_manifest.json', 'completion.json')]
    if manifest['signature'] != state['signature'] or completion != dict(signature=state['signature'], jobs=6):
        raise RuntimeError('Completed peer-assisted signature or job count differs.')
    if manifest['historical_training_records'] != 0:
        raise RuntimeError('Nonempty historical initialization.')
    results, controls = [], []
    for job in runner.jobs(state):
        result = runner.read(runner.path_for(state, job))
        runner.validate_job(state, job, result, complete=True)
        old = runner.read(runner.path_for(dict(out=state['reference_out']), job))
        controls.append(runner.compare_reference(result, old))
        results.append(result)
    saved_control = runner.read(state['out']/'reference_control.json')
    if saved_control != dict(mode='full', checked=True, jobs=controls, smoke_note=None):
        raise RuntimeError('Saved reference-arm verification differs.')
    raw = pd.read_csv(state['archive']/'controlled_evaluations.csv', float_precision='round_trip')
    tables = dense_references(raw[raw.role == 'calibration'], raw[raw.role == 'validation'], state['domain'])
    if not all(v['eligible'] for v in tables[0].values()):
        raise RuntimeError('An empirical reference is ineligible on validation.')
    trials, cells, queries, learning = evaluate(results, state['domain'], state['cfg'], tables)
    times, context_times = eligibility_times(results, state['domain'], state['cfg'])
    summary, paired = summarize(trials), paired_comparisons(trials)
    per_seed = paired_comparisons(trials, per_seed=True)
    modes, peers, transfers = mechanism_tables(queries)

    def compare(frame, name, keys):
        saved = pd.read_csv(state['out']/(name+'.csv'))
        a = frame.sort_values(keys).reset_index(drop=True)
        b = saved[a.columns].sort_values(keys).reset_index(drop=True)
        if 'eligible_training_counts' in a:
            a['eligible_training_counts'] = a.eligible_training_counts.map(json.dumps)
            b['eligible_training_counts'] = b.eligible_training_counts.map(lambda x: json.dumps(json.loads(x)))
        a = a.astype(object).where(pd.notna(a), None)
        b = b.astype(object).where(pd.notna(b), None)
        pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-12)

    budget_keys = ['condition', 'seed', 'method', 'fleet_budget', 'client']
    query_keys = ['condition', 'seed', 'method', 'fleet_evaluation']
    compare(trials, 'budget_trials', budget_keys)
    compare(cells, 'budget_contexts', budget_keys+['context'])
    compare(queries, 'query_observations', query_keys)
    compare(learning, 'learning_diagnostics', ['condition', 'seed', 'method', 'round'])
    compare(summary, 'online_summary', ['condition', 'method', 'fleet_budget'])
    compare(paired, 'paired_comparisons', ['condition', 'fleet_budget', 'method', 'baseline'])
    compare(per_seed, 'per_seed_comparisons', ['condition', 'seed', 'fleet_budget', 'method', 'baseline'])
    primary = paired[(paired.method == METHODS[2]) & (paired.baseline == METHODS[1]) & (paired.fleet_budget == 216)]
    primary_seed = per_seed[(per_seed.method == METHODS[2]) & (per_seed.baseline == METHODS[1]) & (per_seed.fleet_budget == 216)]
    compare(primary, 'primary_comparison', ['condition'])
    compare(primary_seed, 'primary_per_seed', ['condition', 'seed'])
    compare(times, 'eligibility_times', ['condition', 'seed', 'method', 'client'])
    compare(context_times, 'context_eligibility_times', ['condition', 'seed', 'method', 'client', 'context'])
    compare(summarize_times(times), 'eligibility_timing_summary', ['condition', 'method'])
    compare(modes, 'query_modes', ['condition', 'seed', 'method', 'mode'])
    compare(peers, 'peer_verifications', query_keys)
    compare(transfers, 'bo_origin_transfers', ['condition', 'seed', 'method', 'peer_donor',
                                             'peer_donor_query', 'context', 'candidate'])
    if (len(trials), len(cells), len(queries), len(times), len(context_times)) != (972, 5832, 3888, 162, 972):
        raise RuntimeError('Unexpected budget/query/eligibility denominators.')
    if queries.loc[~queries.integration_resolved, 'eligible'].any():
        raise RuntimeError('Unresolved rollout used as eligible cost data.')
    return state, results, tables, trials, cells, queries, learning, times, summary, paired, per_seed


def same_data_regression(state, results):
    phi, rows = features(state['domain'], state['cfg']['model']), []
    for job in results:
        data = job['state']['methods'][METHODS[2]]
        histories = [[] for _ in job['clients']]
        weights = np.zeros(phi.shape[1])
        for r in range(1, state['cfg']['queries_per_client']+1):
            for e in data['events'][(r-1)*len(histories):r*len(histories)]:
                histories[e['client']].append({k: e[k] for k in OBSERVATION_FIELDS})
            local = [LocalCostModel(h, phi, state['cfg']['model']['ridge']) for h in histories]
            weights, _ = federated_fit(local, weights, state['cfg']['model'])
            gap = optimization_gap(local, weights)
            if not np.isclose(gap, data['learning'][r-1]['optimization_gap'], rtol=1e-7, atol=1e-11):
                raise RuntimeError('Federated regression replay differs from logged training.')
            diff = phi@(weights-centralized_fit(local))
            rows.append(dict(condition=job['condition'], seed=job['seed'], round=r,
                optimization_gap=gap, same_data_log_J_prediction_RMS=float(np.sqrt(np.mean(diff**2))),
                same_data_max_log_J_prediction_difference=float(np.abs(diff).max())))
        np.testing.assert_allclose(weights, data['weights'], rtol=1e-7, atol=1e-9)
    return pd.DataFrame(rows)


def common_history_proposals(state, results, tables):
    """Controlled fixed-kernel proposals on B histories, not a replay of actual BO.

    Fixed public GP hyperparameters hold covariance constant between mean
    variants. No counterfactual outcome enters any later history or model.
    """
    cfg, domain = state['cfg'], state['domain']
    phi, rows = features(domain, cfg['model']), []
    reference, validation = tables
    for job in results:
        data = job['state']['methods'][METHODS[1]]
        histories = [[] for _ in job['clients']]
        weights = np.zeros(phi.shape[1])
        initial = np.log([*cfg['gp']['length_scales'], cfg['gp']['amplitude']]).tolist()
        for r in range(1, cfg['queries_per_client']+1):
            events = data['events'][(r-1)*len(histories):r*len(histories)]
            for event in events:
                if event['mode'] != 'feasibility_penalized_LCB':
                    continue
                i, context = event['client'], event['context']
                row = dict(condition=job['condition'], seed=job['seed'], client=job['clients'][i]['id'],
                    plant_id=job['clients'][i]['plant_id'], family=job['clients'][i]['family'],
                    round=r, context=context, actual_candidate=event['candidate'])
                allowed = np.asarray([c == context for c, _ in domain.actions])
                allowed[[o['action'] for o in histories[i]]] = False
                probability = feasibility_probability(histories[i], domain,
                    np.full(len(domain.actions), .5), cfg['gp'])
                fixed_variance = None
                for name in ('zero', 'local', 'federated'):
                    prior = phi@(LocalCostModel(histories[i], phi, cfg['model']['ridge']).fit()
                                 if name == 'local' else weights)
                    if name == 'zero':
                        prior = np.zeros(len(domain.actions))
                    posterior = Posterior(domain, histories[i], prior, cfg['gp'], initial, optimize=False)
                    mean, variance = posterior.predict()
                    if fixed_variance is None:
                        fixed_variance = variance.copy()
                    else:
                        np.testing.assert_allclose(variance, fixed_variance, rtol=1e-12, atol=1e-12)
                    values = acquisition_values(mean, variance, probability, cfg['acquisition'])
                    values[~allowed] = np.inf
                    best = float(values.min())
                    ties = np.flatnonzero(values <= best+1e-12*max(1., abs(best)))
                    action = int(np.random.default_rng(seed_for('acquisition', job['seed'],
                        job['clients'][i]['id'], r-1)).choice(ties))
                    candidate = domain.actions[action][1]
                    val = validation[job['clients'][i]['plant_id'], context, candidate]
                    gap = val['J']/reference[job['clients'][i]['plant_id'], context]['J']-1 if val['eligible'] else None
                    row.update({name+'_candidate': candidate, name+'_eligible': val['eligible'],
                        name+'_J': val['J'] if val['eligible'] else None,
                        name+'_success': gap is not None and gap <= cfg['relative_tolerance']})
                row['different_candidate'] = row['local_candidate'] != row['federated_candidate']
                rows.append(row)
            # Only actual arm-B records update either model or GP history.
            for e in events:
                histories[e['client']].append({k: e[k] for k in OBSERVATION_FIELDS})
            local = [LocalCostModel(h, phi, cfg['model']['ridge']) for h in histories]
            weights, _ = federated_fit(local, weights, cfg['model'])
    return pd.DataFrame(rows)


def proposal_summary(proposals):
    rows = []
    for condition, group in proposals.groupby('condition', sort=False):
        both = group.local_eligible & group.federated_eligible
        a, b = group.loc[both, 'federated_J'], group.loc[both, 'local_J']
        rows.append(dict(condition=condition, common_history_fallbacks=len(group),
            changed_candidates=int(group.different_candidate.sum()), local_eligible=int(group.local_eligible.sum()),
            zero_eligible=int(group.zero_eligible.sum()), zero_near_reference=int(group.zero_success.sum()),
            federated_eligible=int(group.federated_eligible.sum()), local_near_reference=int(group.local_success.sum()),
            federated_near_reference=int(group.federated_success.sum()), jointly_eligible=int(both.sum()),
            federated_mean_J_joint=float(a.mean()) if both.any() else None,
            local_mean_J_joint=float(b.mean()) if both.any() else None,
            relative_reduction_of_joint_mean_J=float(1-a.mean()/b.mean()) if both.any() else None,
            federated_lower_J=int((a < b-1e-12).sum()), federated_higher_J=int((a > b+1e-12).sum())))
    return pd.DataFrame(rows)


def final_sources(state, results, tables):
    rows, domain = [], state['domain']
    for job in results:
        for method, data in job['state']['methods'].items():
            for i, client in enumerate(job['clients']):
                for context, candidate in zip(domain.contexts, data['snapshots']['216']['recommendations'][i]):
                    row = dict(condition=job['condition'], seed=job['seed'], method=method, client=client['id'],
                        family=client['family'], context=context, candidate=candidate, selected=candidate is not None,
                        local_mode=None, root_mode=None, root_client=None, root_query=None)
                    if candidate is not None:
                        action = domain.action(context, candidate)
                        e = next(e for e in data['events'] if e['client'] == i and e['action'] == action)
                        row['local_mode'] = e['mode']
                        while e['mode'] == 'peer_gain_verification':
                            source = data['events'][(e['peer_donor_query']-1)*len(job['clients'])+e['peer_donor']]
                            if source['round'] >= e['round'] or source['action'] != e['action']:
                                raise RuntimeError('Invalid controller-transfer lineage.')
                            e = source
                        row.update(root_mode=e['mode'], root_client=e['client'], root_query=e['local_evaluation'])
                    rows.append(row)
    return pd.DataFrame(rows)


def main():
    state, results, tables, trials, cells, queries, learning, times, summary, paired, per_seed = verify()
    regression = same_data_regression(state, results)
    proposals = common_history_proposals(state, results, tables)
    proposal_stats = proposal_summary(proposals)
    sources = final_sources(state, results, tables)
    modes, peers, transfers = mechanism_tables(queries)
    mode_summary = queries.groupby(['condition', 'method', 'mode'], sort=False).agg(
        queries=('eligible', 'size'), eligible=('eligible', 'sum'), became_incumbent=('became_incumbent', 'sum')).reset_index()
    transfer_summary = transfers.groupby(['condition', 'method'], sort=False).agg(
        BO_sources_transferred=('peer_donor_query', 'size'), target_tests=('target_tests', 'sum'),
        eligible_targets=('eligible_targets', 'sum'), target_incumbents=('target_incumbents', 'sum')).reset_index()
    final = trials[trials.fleet_budget == 216]
    seed_final = final.groupby(['condition', 'seed', 'method'], sort=False).agg(
        context_success=('context_success', 'mean'), eligible_policies=('fully_feasible', 'sum'),
        conditional_mean_J=('J', 'mean'), failed_tests=('failed_queries', 'sum')).reset_index()
    timing = summarize_times(times)
    restricted = times.assign(truncated_time=times.first_all_contexts_query.fillna(24)).groupby(
        ['condition', 'method']).truncated_time.mean()
    timing['mean_queries_to_coverage_truncated_at_24'] = [restricted[c, m] for c, m in zip(timing.condition, timing.method)]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, frame in [('budget_summary', summary), ('paired_final', paired[paired.fleet_budget == 216]),
        ('per_seed_final', seed_final), ('primary_per_seed', per_seed[(per_seed.method == METHODS[2]) &
            (per_seed.baseline == METHODS[1]) & (per_seed.fleet_budget == 216)]),
        ('coverage_timing', timing), ('query_modes_summary', mode_summary), ('BO_transfer_summary', transfer_summary),
        ('final_recommendation_sources', sources), ('same_data_regression', regression),
        ('common_history_proposals', proposals), ('common_history_summary', proposal_stats),
        ('missing_final_contexts', cells[(cells.fleet_budget == 216) & cells.candidate.isna()]),
        ('selected_gains_failing_validation', cells[(cells.fleet_budget == 216) & cells.candidate.notna() & ~cells.eligible])]:
        frame.to_csv(OUT/(name+'.csv'), index=False)
    inputs = [state['out']/name for name in ('run_manifest.json', 'completion.json', 'reference_control.json',
        'budget_trials.csv', 'budget_contexts.csv', 'query_observations.csv', 'learning_diagnostics.csv',
        'online_summary.csv', 'paired_comparisons.csv', 'primary_comparison.csv', 'eligibility_times.csv',
        'context_eligibility_times.csv', 'peer_verifications.csv', 'bo_origin_transfers.csv')]
    inputs += [runner.path_for(state, job) for job in runner.jobs(state)]
    inputs += [state['archive']/'controlled_evaluations.csv']
    runner.save(OUT/'verification.json', dict(signature=state['signature'],
        result_commit=runner.subprocess.check_output(['git', 'log', '-1', '--format=%H', '--',
            'code/results/tracking_peer_cost_v1/completion.json'], cwd=runner.REPO, text=True).strip(),
        jobs=len(results), charged_queries=len(queries),
        fresh_simulator_records=sum(len(j['state']['oracle_cache']) for j in results),
        trial_rows=len(trials), context_rows=len(cells), unresolved_queries=int((~queries.integration_resolved).sum()),
        reference_arm_and_peer_rule_verified=True, saved_evaluation_tables_reproduced=True,
        same_data_federated_training_replayed=True,
        common_history_diagnostic_fixed_public_GP_hyperparameters=True,
        common_history_diagnostic_equal_GP_variance_verified=True,
        common_history_diagnostic_replays_actual_adaptive_policy=False,
        common_history_proposals=len(proposals), hypothetical_outcomes_used_for_training=False,
        new_simulator_rollouts=0, analysis_runtime=dict(python=runner.base.platform.python_version(),
            numpy=np.__version__, pandas=pd.__version__),
        input_hashes={p.relative_to(runner.REPO).as_posix(): runner.digest(p) for p in inputs},
        analysis_source_sha256=runner.digest(Path(__file__))))
    print(f'Verified {len(results)} jobs / {len(queries):,} charged queries. Analysis: {OUT}')
    print(summary[summary.fleet_budget == 216][['condition', 'method', 'context_success_rate',
        'fully_feasible_policy_rate', 'conditional_mean_J', 'mean_fleet_ineligible_tests']].to_string(index=False))
    print(regression[regression['round'] < 24].groupby('condition')[[
        'optimization_gap', 'same_data_log_J_prediction_RMS']].agg(['median', 'max']).to_string())
    print('Retrospective common-history proposals, not full-policy effects:')
    print(proposal_stats.to_string(index=False))


if __name__ == '__main__':
    main()
