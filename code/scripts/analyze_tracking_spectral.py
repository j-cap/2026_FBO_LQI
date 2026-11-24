"""Audit the completed matched spectral campaign without changing its outputs.

Replays recorded decisions, recomputes saved numerical tables in a temporary
directory, and saves descriptive contrasts under docs/analysis. No simulator
or alternative controller policy is run.
"""
from pathlib import Path
import json
import subprocess
import tempfile

import tracking_spectral as runner
import numpy as np
import pandas as pd

from analyze_tracking_hierarchical import compare_saved
from src.tracking_budget import dense_references
from src.tracking_peer_cost_report import evaluate, eligibility_times
from src.tracking_spectral import LEGACY, shared, local
from src.tracking_spectral_report import paired_comparisons, write_report

OUT = runner.REPO/'docs/analysis/spectral_online_cost_model'


def verify():
    state = runner.setup()
    manifest = runner.read(state['out']/'run_manifest.json')
    completion = runner.read(state['out']/'completion.json')
    if (manifest['signature'] != state['signature'] or
        completion != dict(signature=state['signature'], jobs=6)):
        raise RuntimeError('Completed spectral campaign signature or job count differs.')
    if (manifest['historical_training_records'] != 0 or manifest['worker_reference_access'] or
        manifest['worker_evaluation_access'] or manifest['configuration']['primary_rank'] != 256 or
        manifest['evaluation_only_baselines'] != LEGACY):
        raise RuntimeError('Declared spectral information/comparison contract differs.')
    results, unique = [], 0
    for job in runner.jobs(state):
        result = runner.read(runner.path_for(state, job))
        runner.validate_job(state, job, result, complete=True, replay=True)
        unique += len(result['state']['oracle_cache'])
        results.append(runner.comparison_result(state, job, result))
        print(f'Validated and replayed {job["condition"]}/{job["seed"]}', flush=True)
    # Evaluator labels are opened only after the recorded new-arm decisions pass.
    raw = pd.read_csv(state['archive']/'controlled_evaluations.csv', float_precision='round_trip')
    tables = dense_references(raw[raw.role == 'calibration'], raw[raw.role == 'validation'], state['domain'])
    if not all(v['eligible'] for v in tables[0].values()):
        raise RuntimeError('An empirical reference fails validation.')
    trials, cells, queries, learning = evaluate(results, state['domain'], state['cfg'], tables)
    times, context_times = eligibility_times(results, state['domain'], state['cfg'])
    comparisons = [
        (trials, 'budget_trials', ['condition','seed','method','fleet_budget','client']),
        (cells, 'budget_contexts', ['condition','seed','method','fleet_budget','client','context']),
        (queries, 'query_observations', ['condition','seed','method','fleet_evaluation']),
        (learning, 'learning_diagnostics', ['condition','seed','method','round']),
        (times, 'eligibility_times', ['condition','seed','method','client']),
        (context_times, 'context_eligibility_times', ['condition','seed','method','client','context'])]
    for frame, name, keys in comparisons:
        compare_saved(frame, state['out']/(name+'.csv'), keys)
    # Exercise the frozen report's complete numeric path without overwriting it.
    with tempfile.TemporaryDirectory(prefix='spectral_audit_') as directory:
        temporary = Path(directory)
        write_report(temporary, trials, cells, queries, learning, times, state['cfg'], 'full', len(state['domain'].actions))
        regenerated = sorted(temporary.glob('*.csv'))
        for path in regenerated:
            actual = pd.read_csv(path, float_precision='round_trip')
            keys = [c for c in ('condition','seed','method','baseline','fleet_budget','client','round',
                'fleet_evaluation','mode','family','context','fraction','peer_donor','peer_donor_query','candidate')
                if c in actual.columns]
            compare_saved(actual, state['out']/path.name, keys)
    if (len(trials), len(cells), len(queries), len(times), len(context_times)) != (2592,15552,10368,432,2592):
        raise RuntimeError('Unexpected spectral comparison denominators.')
    counts = queries.groupby(['condition','seed','method','client']).size()
    if not (counts == 24).all() or len(queries[queries.method.isin(state['cfg']['methods'])]) != 5184:
        raise RuntimeError('Spectral local cap or charged request count differs.')
    if (~queries.integration_resolved).any():
        raise RuntimeError('An unresolved rollout remains in the compared campaign.')
    control = runner.read(state['out']/'reference_control.json')
    if control != dict(reused_for_evaluation_only=True, methods=LEGACY,
                       exact_fleet_and_factory_pairing=True, worker_reference_access=False):
        raise RuntimeError('Frozen comparator reuse contract differs.')
    provenance = runner.read(state['out']/'analysis_provenance.json')
    if provenance != dict(all_new_choices_replayed=True, public_bases=state['public_bases'],
                         report_sha256=runner.digest(runner.ROOT/'src/tracking_spectral_report.py')):
        raise RuntimeError('Saved spectral analysis provenance differs.')
    return state, results, trials, cells, queries, learning, unique, len(comparisons)+len(regenerated)


def outcome_tables(trials, cells, queries, learning, cfg):
    final = cfg['fleet_budgets'][-1]
    policies = trials[trials.fleet_budget == final]
    contexts = cells[cells.fleet_budget == final]
    outcomes = policies.groupby(['condition','method'], sort=False).agg(
        client_trials=('client','size'), eligible_policies=('fully_feasible','sum'),
        context_success_rate=('context_success','mean'), conditional_mean_J=('J','mean'),
        mean_ineligible_tests_per_client=('ineligible_queries','mean')).reset_index()
    successes = contexts.groupby(['condition','method'], sort=False).agg(
        context_trials=('success','size'), successes=('success','sum'),
        locally_available_contexts=('has_local_eligible_gain','sum'),
        validation_eligible_contexts=('eligible','sum')).reset_index()
    outcomes = outcomes.merge(successes, on=['condition','method'], validate='one_to_one')
    paired = paired_comparisons(trials, cfg)
    seeds = paired_comparisons(trials, cfg, per_seed=True)
    aggregate = paired[paired.fleet_budget == final].copy()
    signs = seeds[seeds.fleet_budget == final].groupby(['condition','method','baseline'], sort=False).agg(
        positive_success_seeds=('context_success_delta', lambda x: int((x > 1e-12).sum())),
        negative_success_seeds=('context_success_delta', lambda x: int((x < -1e-12).sum())),
        tied_success_seeds=('context_success_delta', lambda x: int((x.abs() <= 1e-12).sum())),
        lower_cost_seeds=('mean_J_delta_jointly_feasible', lambda x: int((x < 0).sum()))).reset_index()
    aggregate = aggregate.merge(signs, on=['condition','method','baseline'], validate='one_to_one')
    # Context localization is descriptive on the actual adaptive policies.
    rows = []
    for (condition, method, baseline, role), _ in aggregate.groupby(['condition','method','baseline','role'], sort=False):
        group = contexts[contexts.condition == condition]
        a = group[group.method == method].set_index(['seed','client','context']).sort_index()
        b = group[group.method == baseline].set_index(['seed','client','context']).sort_index()
        if not a.index.equals(b.index):
            raise RuntimeError('Unpaired context localization.')
        for seed in [None, *cfg['seeds']]:
            av = a if seed is None else a.xs(seed, level='seed')
            bv = b if seed is None else b.xs(seed, level='seed')
            for context in sorted(group.context.unique()):
                ac, bc = av.xs(context, level='context'), bv.xs(context, level='context')
                good = ac.eligible & bc.eligible
                rows.append(dict(condition=condition, method=method, baseline=baseline, role=role,
                    seed='all' if seed is None else str(seed), context=context, context_trials=len(ac),
                    method_successes=int(ac.success.sum()), baseline_successes=int(bc.success.sum()),
                    success_count_delta=int(ac.success.sum()-bc.success.sum()), jointly_eligible_pairs=int(good.sum()),
                    mean_J_method_joint=float(ac.loc[good,'J'].mean()) if good.any() else None,
                    mean_J_baseline_joint=float(bc.loc[good,'J'].mean()) if good.any() else None))
    mode_rows = queries.groupby(['condition','seed','method','mode'], sort=False).agg(
        requests=('eligible','size'), eligible_requests=('eligible','sum'),
        new_incumbents=('became_incumbent','sum')).reset_index()
    sharing = learning[(learning['round'] == cfg['queries_per_client']-1)&
        learning.method.isin([LEGACY[2], shared(256), shared(384)])][
        ['condition','seed','method','round','expected_sharing_fraction','sharing_weights']].copy()
    family = policies.groupby(['condition','family','method'], sort=False).agg(
        client_trials=('client','size'), eligible_policies=('fully_feasible','sum'),
        context_success_rate=('context_success','mean'), conditional_mean_J=('J','mean')).reset_index()
    return dict(final_outcomes=outcomes, final_contrasts=aggregate,
        per_seed_final_contrasts=seeds[seeds.fleet_budget == final], budget_contrasts=paired,
        context_contrasts=pd.DataFrame(rows), query_modes=mode_rows, final_family_outcomes=family,
        sharing_before_final_query=sharing)


def main():
    state, results, trials, cells, queries, learning, unique, tables_checked = verify()
    outputs = outcome_tables(trials, cells, queries, learning, state['cfg'])
    OUT.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(OUT/(name+'.csv'), index=False)
    final = cells[cells.fleet_budget == state['cfg']['fleet_budgets'][-1]]
    paths = [state['out']/name for name in ('run_manifest.json','completion.json','public_bases.npz',
        'analysis_provenance.json','reference_control.json')]
    paths += [runner.path_for(state, j) for j in runner.jobs(state)]
    paths += sorted(state['out'].glob('*.csv'))
    paths += [state['archive']/'controlled_evaluations.csv']
    runner.save(OUT/'verification.json', dict(result_signature=state['signature'], jobs=len(results),
        source_result_commit=subprocess.check_output(['git','log','-1','--format=%H','--',
            state['out'].relative_to(runner.REPO).as_posix()], cwd=runner.REPO, text=True).strip(),
        new_arm_charged_queries=5184, reused_baseline_queries=5184, total_compared_queries=len(queries),
        unique_new_arm_simulator_records=unique, all_new_decisions_replayed=True,
        saved_numeric_tables_recomputed=tables_checked, public_bases_verified=True,
        unresolved_queries=int((~queries.integration_resolved).sum()),
        selected_final_contexts_failing_validation=int((final.has_local_eligible_gain & ~final.eligible).sum()),
        new_physical_rollouts=0, new_model_variants=0, new_counterfactual_proposals=0,
        primary_method=shared(256), primary_baseline=LEGACY[0], secondary_baseline=LEGACY[1],
        inference_scope='Matched development campaign on fixed physical fleets; three programme/noise seeds.',
        analysis_source_sha256=runner.digest(Path(__file__)),
        input_hashes={p.relative_to(runner.REPO).as_posix():runner.digest(p) for p in paths}))
    print('Verified tables:', tables_checked)
    print(outputs['final_outcomes'].to_string(index=False))
    print('Analysis:', OUT, flush=True)


if __name__ == '__main__':
    main()
