"""Audit completed joint cost BO and summarize its recorded on-policy forecasts.

No new physical rollouts, candidate proposals or model variants are evaluated.
Replay computes the same recorded choices from their paid previous histories.
"""
from pathlib import Path
import json

import tracking_hierarchical as runner
import numpy as np
import pandas as pd

from src.tracking_budget import dense_references
from src.tracking_hierarchical import LOCAL, METHODS, SHARED, run_fleet
from src.tracking_peer_cost import METHODS as BASELINES, OBSERVATION_FIELDS
from src.tracking_peer_cost_report import evaluate, eligibility_times, mechanism_tables, summarize
from src.tracking_hierarchical_report import (benchmark_readout, paired_comparisons, sharing_table)
from src.tracking_online_lcb_report import summarize_times

OUT = runner.REPO/'docs/analysis/joint_shared_local_cost_model'


def compare_saved(frame, path, keys):
    saved = pd.read_csv(path, float_precision='round_trip')
    a = frame.sort_values(keys).reset_index(drop=True)
    b = saved[a.columns].sort_values(keys).reset_index(drop=True)
    for column in ('eligible_training_counts', 'sharing_weights', 'component_log_evidence'):
        if column not in a:
            continue
        for actual, expected in zip(a[column], b[column]):
            if isinstance(actual, list):
                np.testing.assert_allclose(actual, json.loads(expected), rtol=1e-10, atol=1e-10)
            elif not pd.isna(actual) or not pd.isna(expected):
                raise RuntimeError('Missing list diagnostic differs: '+column)
        a[column] = b[column]
    a = a.astype(object).where(pd.notna(a), None)
    b = b.astype(object).where(pd.notna(b), None)
    pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-12)


def recorded_forecasts(state, job, result):
    """Score only actual, eligible next labels; do not feed diagnostics to BO."""
    clients, forecasts, previous = result['clients'], [], {}
    def oracle(plans, cache):
        observations = []
        for plan in plans:
            event = result['state']['methods'][plan['method']]['events'][
                (plan['round']-1)*len(clients)+plan['client']]
            if any(plan[k] != event[k] for k in plan):
                raise RuntimeError('Recorded query differs from previous-history replay.')
            if plan['round'] > 1:
                posterior = previous[plan['method']]
                if posterior['information_round'] != plan['information_round']:
                    raise RuntimeError('Forecast contains later-round information.')
                i, action = plan['client'], plan['action']
                mean, variance = posterior['mean'][i][action], posterior['variance'][i][action]
                row = dict(**job, method=plan['method'], client=clients[i]['id'],
                    family=clients[i]['family'], round=plan['round'], context=plan['context'],
                    candidate=plan['candidate'], mode=plan['mode'], eligible=event['eligible'],
                    mean_log_J=mean, latent_variance=variance,
                    expected_sharing_fraction=posterior['expected_sharing_fraction'])
                if event['eligible']:
                    error = float(np.log(event['J'])-mean)
                    std = float(np.sqrt(variance+state['cfg']['gp']['noise_std_log_J']**2+1e-9))
                    row.update(observed_log_J=float(np.log(event['J'])), log_error=error,
                        predictive_observation_std=std, standardized_error=error/std,
                        inside_2sd_moment_band=abs(error) <= 2*std)
                forecasts.append(row)
            observations.append({k: event[k] for k in OBSERVATION_FIELDS if k != 'action'})
        return observations
    def checkpoint(data):
        for method in METHODS:
            previous[method] = data['methods'][method]['cost_posterior']
    replay = run_fleet(state['domain'], dict(state['cfg'], methods=METHODS), clients,
                       job['seed'], oracle, checkpoint=checkpoint)
    for method in METHODS:
        if replay['methods'][method]['events'] != result['state']['methods'][method]['events']:
            raise RuntimeError('Recorded-event replay differs.')
        for a, b in zip(replay['methods'][method]['learning'], result['state']['methods'][method]['learning']):
            for key in ('sharing_weights', 'component_log_evidence', 'expected_sharing_fraction'):
                np.testing.assert_allclose(a[key], b[key], rtol=1e-10, atol=1e-10)
    return forecasts


def verify():
    state = runner.setup()
    manifest, completion = [runner.read(state['out']/name) for name in ('run_manifest.json', 'completion.json')]
    if manifest['signature'] != state['signature'] or completion != dict(signature=state['signature'], jobs=6):
        raise RuntimeError('Completed joint-model signature/job count differs.')
    if manifest['historical_training_records'] != 0 or manifest['worker_reference_access'] or manifest['worker_evaluation_access']:
        raise RuntimeError('Historical/evaluator access contract differs.')
    results, forecasts, unique_records = [], [], 0
    for job in runner.jobs(state):
        result = runner.read(runner.path_for(state, job))
        runner.validate_job(state, job, result, complete=True)
        forecasts.extend(recorded_forecasts(state, job, result))
        unique_records += len(result['state']['oracle_cache'])
        results.append(runner.comparison_result(state, job, result))
        print(f'Validated and replayed {job["condition"]}/{job["seed"]}', flush=True)
    raw = pd.read_csv(state['archive']/'controlled_evaluations.csv', float_precision='round_trip')
    tables = dense_references(raw[raw.role == 'calibration'], raw[raw.role == 'validation'], state['domain'])
    if not all(v['eligible'] for v in tables[0].values()):
        raise RuntimeError('An empirical reference is ineligible on validation.')
    trials, cells, queries, learning = evaluate(results, state['domain'], state['cfg'], tables)
    times, context_times = eligibility_times(results, state['domain'], state['cfg'])
    summary = summarize(trials)
    paired, seeds = paired_comparisons(trials), paired_comparisons(trials, per_seed=True)
    budget_keys = ['condition', 'seed', 'method', 'fleet_budget', 'client']
    query_keys = ['condition', 'seed', 'method', 'fleet_evaluation']
    final = state['cfg']['fleet_budgets'][-1]
    modes, peers, transfers = mechanism_tables(queries)
    family = trials[trials.fleet_budget == final].groupby(['condition', 'family', 'method'], sort=False).agg(
        client_trials=('client', 'size'), eligible_policies=('fully_feasible', 'sum'),
        context_success_rate=('context_success', 'mean'), conditional_mean_policy_J=('J', 'mean')).reset_index()
    by_context = cells[cells.fleet_budget == final].groupby(['condition', 'method', 'context'], sort=False).agg(
        client_context_trials=('client', 'size'), eligible_contexts=('eligible', 'sum'),
        successes=('success', 'sum'), context_success_rate=('success', 'mean'),
        conditional_mean_context_J=('J', 'mean')).reset_index()
    comparisons = [
        (trials, 'budget_trials', budget_keys), (cells, 'budget_contexts', budget_keys+['context']),
        (queries, 'query_observations', query_keys), (learning, 'learning_diagnostics', ['condition','seed','method','round']),
        (summary, 'online_summary', ['condition','method','fleet_budget']),
        (paired, 'paired_comparisons', ['condition','fleet_budget','baseline']),
        (seeds, 'per_seed_comparisons', ['condition','seed','fleet_budget','baseline']),
        (paired[(paired.fleet_budget == final) & (paired.baseline == BASELINES[0])], 'primary_comparison', ['condition']),
        (paired[(paired.fleet_budget == final) & (paired.baseline == BASELINES[1])], 'secondary_comparison', ['condition']),
        (seeds[(seeds.fleet_budget == final) & (seeds.baseline == BASELINES[0])], 'primary_per_seed', ['condition','seed']),
        (benchmark_readout(paired, seeds, final), 'benchmark_readout', ['condition','baseline']),
        (times, 'eligibility_times', ['condition','seed','method','client']),
        (context_times, 'context_eligibility_times', ['condition','seed','method','client','context']),
        (summarize_times(times), 'eligibility_timing_summary', ['condition','method']),
        (sharing_table(learning, state['cfg']), 'sharing_strength', ['condition','seed','round','sharing_fraction']),
        (modes, 'query_modes', ['condition','seed','method','mode']), (peers, 'peer_verifications', query_keys),
        (transfers, 'bo_origin_transfers', ['condition','seed','method','peer_donor','peer_donor_query','context','candidate']),
        (family, 'final_family_outcomes', ['condition','family','method']),
        (by_context, 'final_context_outcomes', ['condition','method','context'])]
    for frame, name, keys in comparisons:
        compare_saved(frame, state['out']/(name+'.csv'), keys)
    if (len(trials), len(cells), len(queries), len(times), len(context_times)) != (1620,9720,6480,270,1620):
        raise RuntimeError('Unexpected budget/query denominators.')
    counts = queries.groupby(['condition','seed','method','client']).size()
    if not (counts == 24).all() or len(queries[queries.method.isin(METHODS)]) != 2592:
        raise RuntimeError('Local caps or new-arm query counts differ.')
    if (~queries.integration_resolved).any():
        raise RuntimeError('Unresolved rollout in this campaign.')
    final_cells = cells[cells.fleet_budget == final]
    if (final_cells.has_local_eligible_gain & ~final_cells.eligible).any():
        raise RuntimeError('A selected final gain fails independent validation.')
    control = runner.read(state['out']/'reference_control.json')
    if control != dict(reused_for_evaluation_only=True, methods=BASELINES,
                       exact_fleet_and_factory_pairing=True, worker_reference_access=False):
        raise RuntimeError('Saved reference-control record differs.')
    return state, results, tables, trials, cells, queries, learning, times, summary, paired, seeds, pd.DataFrame(forecasts), unique_records


def contrast_tables(trials, cells, paired, seeds):
    final = trials.fleet_budget.max()
    outcomes = trials[trials.fleet_budget == final].groupby(['condition','method'], sort=False).agg(
        client_trials=('client','size'), eligible_policies=('fully_feasible','sum'),
        context_success=('context_success','mean'), conditional_mean_J=('J','mean'),
        mean_ineligible_tests_per_client=('ineligible_queries','mean')).reset_index()
    aggregate = paired[paired.fleet_budget == final].copy()
    signs = seeds[seeds.fleet_budget == final].groupby(['condition','baseline'], sort=False).agg(
        positive_seeds=('context_success_delta', lambda x: int((x > 1e-12).sum())),
        negative_seeds=('context_success_delta', lambda x: int((x < -1e-12).sum())),
        tied_seeds=('context_success_delta', lambda x: int((x.abs() <= 1e-12).sum()))).reset_index()
    aggregate = aggregate.merge(signs,on=['condition','baseline'],validate='one_to_one')
    rows = []
    final_cells = cells[cells.fleet_budget == final]
    for condition, group in final_cells.groupby('condition', sort=False):
        for baseline in BASELINES[:2]+[LOCAL]:
            a = group[group.method == SHARED].set_index(['seed','client','context']).sort_index()
            b = group[group.method == baseline].set_index(['seed','client','context']).sort_index()
            if not a.index.equals(b.index):
                raise RuntimeError('Context contrast is not paired.')
            for context in sorted(group.context.unique()):
                av, bv = a.xs(context,level='context'), b.xs(context,level='context')
                good = av.eligible & bv.eligible
                rows.append(dict(condition=condition, baseline=baseline, context=context,
                    successes_D=int(av.success.sum()), successes_baseline=int(bv.success.sum()),
                    success_count_delta=int(av.success.sum()-bv.success.sum()),
                    total_context_trials=len(av), jointly_eligible_context_pairs=int(good.sum()),
                    mean_J_D_joint=float(av.loc[good,'J'].mean()) if good.any() else None,
                    mean_J_baseline_joint=float(bv.loc[good,'J'].mean()) if good.any() else None))
    return outcomes, aggregate, pd.DataFrame(rows)


def forecast_summary(forecasts):
    rows = []
    for keys in [('condition','method','mode'), ('condition','method','context','mode')]:
        for values, group in forecasts.groupby(list(keys), sort=False):
            eligible = group[group.eligible]
            rows.append(dict(scope='context' if 'context' in keys else 'condition', **dict(zip(keys,values)),
                queries=len(group), eligible_cost_forecasts=len(eligible),
                mean_log_error=float(eligible.log_error.mean()) if len(eligible) else None,
                log_error_RMS=float(np.sqrt(np.mean(eligible.log_error**2))) if len(eligible) else None,
                fraction_inside_two_SD_moment_band=float(eligible.inside_2sd_moment_band.mean()) if len(eligible) else None,
                median_absolute_standardized_error=float(eligible.standardized_error.abs().median()) if len(eligible) else None,
                max_absolute_standardized_error=float(eligible.standardized_error.abs().max()) if len(eligible) else None))
    return pd.DataFrame(rows)


def main():
    state, results, tables, trials, cells, queries, learning, times, summary, paired, seeds, forecasts, unique = verify()
    outcomes, contrasts, contexts = contrast_tables(trials, cells, paired, seeds)
    OUT.mkdir(parents=True, exist_ok=True)
    forecast_stats = forecast_summary(forecasts)
    modes, _, _ = mechanism_tables(queries)
    modes = modes.groupby(['condition','method','mode'],sort=False).agg(
        queries=('queries','sum'), eligible_queries=('eligible_queries','sum'),
        became_incumbent=('became_incumbent','sum')).reset_index()
    for name, frame in [('final_outcomes',outcomes), ('final_contrasts',contrasts),
        ('per_seed_final_contrasts',seeds[seeds.fleet_budget == 216]), ('context_contrasts',contexts),
        ('budget_contrasts',paired), ('query_modes',modes),
        ('recorded_forecasts',forecasts), ('recorded_forecast_summary',forecast_stats),
        ('sharing_strength',sharing_table(learning,state['cfg']))]:
        frame.to_csv(OUT/(name+'.csv'),index=False)
    paths = [state['out']/name for name in ('run_manifest.json','completion.json','paired_comparisons.csv')]
    paths += [runner.path_for(state,j) for j in runner.jobs(state)]
    paths += [state['archive']/'controlled_evaluations.csv']
    runner.save(OUT/'verification.json',dict(result_signature=state['signature'],jobs=len(results),
        new_arm_charged_queries=2592, reused_baseline_queries=3888,total_compared_queries=len(queries),
        unique_new_arm_simulator_records=unique, all_new_decisions_replayed=True,
        saved_numeric_tables_recomputed=20,unresolved_queries=0,
        new_physical_rollouts=0,new_model_variants=0,new_counterfactual_proposals=0,
        forecast_scope='Actual queried next labels, scored after the frozen decision; descriptive on-policy moment bands.',
        analysis_source_sha256=runner.digest(Path(__file__)),
        input_hashes={p.relative_to(runner.REPO).as_posix():runner.digest(p) for p in paths}))
    print('Final outcomes:');print(outcomes.to_string(index=False))
    print('Final contrasts:');print(contrasts.to_string(index=False))
    print('Actual fallback forecast summaries:')
    print(forecast_stats[(forecast_stats.scope == 'condition') &
        (forecast_stats['mode'] == 'feasibility_penalized_LCB')].to_string(index=False))
    print('Analysis: '+str(OUT),flush=True)


if __name__ == '__main__':
    main()
