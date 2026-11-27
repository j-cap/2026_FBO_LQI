"""Read-only verification and analysis of the frozen 100-test continuation."""
import argparse
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

import tracking_budget_extension as runner
import analyze_tracking_dense_gain as audit_helpers
import numpy as np
import pandas as pd

from src.tracking_budget import matern
from src.tracking_budget_extension_report import write_report
from src.tracking_delta_report import bootstrap_effects
from src.tracking_dense_gain import A, B, D, E, METHODS, BONUSES
from src.tracking_kernel_features import query_posterior
from src.tracking_six_speed_report import _outcome

OUT = runner.REPO / 'docs/analysis/budget_continuation'
LABELS = dict(zip(METHODS, ['A', 'B', 'D', 'E']))


def direct_worker(job):
    """Independent observation-space checks near each new budget checkpoint."""
    state = runner.base.WORKER
    result = runner.base.read(runner.run_path(state, job))
    domain, cfg, n = state['domain'], state['cfg'], len(state['clients'])
    checks = []
    for method in (D, E):
        data = result['state']['methods'][method]
        for checkpoint in (25, 50, 100):
            event = next(e for e in reversed(data['events'][:n * checkpoint])
                if e['mode'] == 'feasibility_penalized_LCB')
            r, client = event['round'] - 1, event['client']
            # Every-round delivery: only observations from earlier rounds.
            # This is formed directly from timestamps, without the cache helper.
            view = [[o for o in h[:r] if o['eligible']]
                if method == D or j == client else [] for j, h in enumerate(data['histories'])]
            inputs = sorted({o['action'] for h in view for o in h} | {event['action']})
            positions = {a: j for j, a in enumerate(inputs)}
            kernel = matern(domain.X[inputs], domain.X[inputs], cfg['gp']['length_scales'],
                cfg['gp']['amplitude'], cfg['gp']['constant_variance'])
            local = [[dict(o, action=positions[o['action']]) for o in h] for h in view]
            control = cfg['hierarchical'] if method == D else dict(sharing_fractions=[0.], prior_weights=[1.])
            direct = query_posterior(local, kernel, control, cfg['gp'], [client], [positions[event['action']]])
            mean, variance = direct['mean'][0], direct['variance'][0]
            score = mean - BONUSES[method] * np.sqrt(variance) - cfg['acquisition']['feasibility_penalty'] * np.log(
                max(event['eligibility_probability'], cfg['acquisition']['probability_floor']))
            np.testing.assert_allclose([mean, variance, score],
                [event['predicted_log_J'], event['predicted_variance'], event['acquisition_score']], rtol=2e-8, atol=2e-8)
            checks.append(dict(seed=job['seed'], method=method, checkpoint=checkpoint,
                checked_round=event['round'], client=client, eligible_observations=sum(map(len, local)),
                own_eligible_observations=len(local[client]),
                absolute_mean_error=abs(mean - event['predicted_log_J']),
                absolute_variance_error=abs(variance - event['predicted_variance']),
                absolute_acquisition_error=abs(score - event['acquisition_score'])))
    pd.DataFrame(checks).to_csv(state['analysis_out'] / f'{job["seed"]}.csv', index=False)


def reconstruct(state, results, audit):
    with patch.object(audit_helpers, 'runner', runner):
        refs, validation, raw, source_checks = audit_helpers.references_and_validation(state, results)
    reused = runner.old_validation_bank(state)
    counts = dict(reused=0, fresh=0)
    for path in sorted((state['out'] / 'evaluation/validation').glob('*.json')):
        batch = runner.base.read(path)
        cached = 0
        for row in batch['rows']:
            key = batch['context'], batch['repeat'], row['candidate']
            if key in reused:
                if row != reused[key]:
                    raise RuntimeError('Reused held-out observation changed.')
                cached += 1
        if (batch['reused_rollouts'], batch['new_rollouts']) != (cached, len(batch['rows']) - cached):
            raise RuntimeError('Saved held-out reuse accounting differs.')
        counts['reused'] += cached
        counts['fresh'] += len(batch['rows']) - cached
    with tempfile.TemporaryDirectory(prefix='budget_report_audit_') as directory:
        frames = write_report(dict(state, out=Path(directory)), results, validation, refs, audit)
        if runner.base.read(Path(directory) / 'budget_readout.json') != runner.base.read(state['out'] / 'budget_readout.json'):
            raise RuntimeError('Saved budget readout differs.')
    keys = dict(context_outcomes=['seed', 'method', 'queries', 'client', 'scope', 'context'],
        client_outcomes=['seed', 'method', 'queries', 'client', 'scope'],
        budget_summary=['scope', 'method', 'queries'], paired_comparisons=['queries', 'method', 'baseline'],
        per_programme_comparisons=['queries', 'method', 'baseline', 'seed'],
        programme_bootstrap=['queries', 'method', 'baseline'],
        changes_from_24=['method', 'queries', 'baseline_queries', 'seed'],
        change_programme_bootstrap=['method', 'queries', 'baseline_queries'],
        communication_and_query_allocation=['seed', 'method', 'queries'],
        frequency_units=['scope', 'method', 'queries', 'seed', 'client', 'frequency'],
        frequency_summary=['scope', 'method', 'queries', 'frequency'],
        gap_threshold_diagnostics=['method', 'queries', 'diagnostic_tolerance'], empirical_references=['context'])
    checks = [audit_helpers.verify_table(state, name, frame, keys[name]) for name, frame in frames.items()]
    expected = dict(signature=state['signature'], all_choices_and_packets_replayed=True, all_prefixes_unchanged=True,
        reused_paid_requests=8640, additional_paid_requests=27360, total_paid_requests=36000,
        heldout_rollouts=len(raw), reused_heldout_rollouts=counts['reused'], new_heldout_rollouts=counts['fresh'],
        new_calibration_rollouts=0, confirmation_claim=False)
    if expected != runner.base.read(state['out'] / 'verification.json'):
        raise RuntimeError('Saved continuation verification differs.')
    return frames, refs, validation, raw, checks, source_checks


def diagnostics(state, results, frames, refs, validation, raw):
    trials, cells = frames['client_outcomes'], frames['context_outcomes']
    fixed = trials[trials.scope == 'fixed_24_contexts']
    changes = []
    cost_changes = []
    for method, group in fixed.groupby('method', sort=False):
        for q0, q1 in [(24, 25), (25, 50), (50, 100), (24, 100)]:
            before, after = (group[group.queries == q].set_index(['seed', 'client']).sort_index() for q in (q0, q1))
            effects = (after.context_success - before.context_success).groupby('seed').mean()
            changes.extend(dict(method=method, from_queries=q0, to_queries=q1, seed=int(seed),
                success_delta_pp=100 * float(effect)) for seed, effect in effects.items())
            complete = before.complete_eligible & after.complete_eligible
            j0, j1 = (float(t.loc[complete, 'J'].mean()) for t in (before, after))
            cost_changes.append(dict(method=method, from_queries=q0, to_queries=q1,
                common_complete_policies=int(complete.sum()), mean_J_before=j0, mean_J_after=j1,
                change_percent=100 * (j1 / j0 - 1)))
    changes = pd.DataFrame(changes)
    changes.to_csv(OUT / 'programme_budget_changes.csv', index=False)
    bootstrap_effects(changes, state['requested'], ['method', 'from_queries', 'to_queries']).to_csv(OUT / 'budget_change_uncertainty.csv', index=False)
    pd.DataFrame(cost_changes).to_csv(OUT / 'paired_budget_costs.csv', index=False)

    policy_counts = cells.groupby(['scope', 'method', 'queries', 'seed', 'client']).agg(
        all_contexts_near_reference=('success', 'all'), complete_eligible=('eligible', 'all')).reset_index()
    policy_counts.groupby(['scope', 'method', 'queries']).agg(client_runs=('client', 'size'),
        all_contexts_near_reference=('all_contexts_near_reference', 'sum'),
        complete_eligible=('complete_eligible', 'sum')).reset_index().to_csv(OUT / 'complete_policies.csv', index=False)
    units = frames['frequency_units'].query('scope == "fixed_24_contexts"')
    frequency = []
    for (queries, frequency_value), group in units.groupby(['queries', 'frequency']):
        target = group[group.method == D].set_index(['seed', 'client']).success
        for method in (A, B, E):
            control = group[group.method == method].set_index(['seed', 'client']).success
            effect = (target - control).groupby('seed').mean()
            frequency.extend(dict(queries=queries, frequency=frequency_value, method=D, baseline=method,
                seed=int(seed), success_delta_pp=100 * float(value)) for seed, value in effect.items())
    bootstrap_effects(pd.DataFrame(frequency), state['requested'], ['queries', 'frequency', 'method', 'baseline']).to_csv(
        OUT / 'frequency_effect_uncertainty.csv', index=False)

    # Intermediate rounds are saved observations, not new optimization runs.
    # These dense curves and threshold crossings are posthoc diagnostics.
    curves, visits = [], []
    for result in results:
        seed, n = result['seed'], len(state['clients'])
        for method, data in result['state']['methods'].items():
            for q in range(1, 101):
                means = []
                for i, client in enumerate(state['clients']):
                    fixed_contexts = set(state['cfg']['programmes'][str(seed)][client['id']][:24])
                    recommendations = data['events'][(q - 1) * n + i]['recommendations']
                    means.append(np.mean([_outcome(c, recommendations[state['domain'].contexts.index(c)], refs,
                        validation, state['cfg']['relative_tolerance'])['success'] for c in fixed_contexts]))
                curves.append(dict(seed=seed, method=method, queries=q, context_success_percent=100 * float(np.mean(means))))
            for event in data['events']:
                observation = result['state']['oracle_cache'][runner.base.cache_key(event)]['result']
                visits.append(dict(seed=seed, method=method, round=event['round'], client=event['client'],
                    frequency=event['frequency'], mode=event['mode'], eligible=event['eligible'],
                    integration_resolved=event['integration_resolved'], failure=observation.get('failure'),
                    became_incumbent=event['became_incumbent']))
    curves = pd.DataFrame(curves)
    for _, row in fixed.groupby(['seed', 'method', 'queries']).context_success.mean().reset_index().iterrows():
        actual = curves[(curves.seed == row.seed) & (curves.method == row.method) & (curves.queries == row.queries)]
        np.testing.assert_allclose(actual.context_success_percent, 100 * row.context_success, rtol=2e-12, atol=2e-12)
    curves.to_csv(OUT / 'per_programme_learning_curves.csv', index=False)
    means = curves.groupby(['method', 'queries']).context_success_percent.mean().reset_index()
    means.to_csv(OUT / 'learning_curves.csv', index=False)
    d = means[means.method == D]
    crossings = []
    for method in (A, B, E):
        target = float(means[(means.method == method) & (means.queries == 100)].context_success_percent.iloc[0])
        crossing = int(d[d.context_success_percent >= target].queries.min())
        crossings.append(dict(method=D, baseline=method, baseline_queries=100, target_success_percent=target,
            first_D_crossing=crossing, descriptive_query_reduction_percent=100 - crossing,
            posthoc_diagnostic=True))
    pd.DataFrame(crossings).to_csv(OUT / 'posthoc_budget_equivalents.csv', index=False)
    visits = pd.DataFrame(visits)
    visits['period'] = pd.cut(visits['round'], [0, 24, 50, 100], labels=['1-24', '25-50', '51-100'])
    visits.groupby(['method', 'period', 'mode'], observed=True).agg(requests=('eligible', 'size'),
        eligible=('eligible', 'sum'), new_incumbents=('became_incumbent', 'sum')).reset_index().to_csv(OUT / 'query_allocation.csv', index=False)
    visits[~visits.eligible].groupby(['method', 'period', 'frequency', 'failure'], observed=True).size().rename(
        'failed_requests').reset_index().to_csv(OUT / 'failed_requests.csv', index=False)

    # Physical comparison only on the same initially and finally eligible cases.
    physical = raw.groupby(['context', 'candidate'])[['J', 'rmse_y', 'rms_rate_deg_s']].mean()
    selected = cells.query('scope == "fixed_24_contexts" and method == @D and queries == 100')
    paired = []
    for row in selected.to_dict('records'):
        initial = validation[row['context']].get(state['domain'].candidates[0])
        if row['eligible'] and initial and initial['eligible']:
            first = physical.loc[(row['context'], state['domain'].candidates[0])]
            last = physical.loc[(row['context'], row['candidate'])]
            paired.append(dict(seed=row['seed'], client=row['client'], context=row['context'],
                **{'initial_' + k: float(first[k]) for k in physical.columns},
                **{'final_' + k: float(last[k]) for k in physical.columns}))
    paired = pd.DataFrame(paired)
    paired.to_csv(OUT / 'initial_final_common_eligible_contexts.csv', index=False)
    return curves, visits


def figures(curves, frames):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(zip(METHODS, ['#64748b', '#b58136', '#007d75', '#8265ad']))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    summary = curves.groupby(['method', 'queries']).context_success_percent.mean().reset_index()
    for method in METHODS:
        rows = summary[summary.method == method]
        axes[0].plot(rows.queries, rows.context_success_percent, color=colors[method], label=LABELS[method])
    axes[0].set(xlabel='Total paid tests per client/arm', ylabel='Success on original contexts [%]', ylim=(0, 100))
    axes[0].legend(ncol=4, frameon=False, fontsize=9)
    bounds = frames['programme_bootstrap']
    for method in (A, B, E):
        rows = bounds[bounds.baseline == method].sort_values('queries')
        axes[1].plot(rows.queries, rows.success_delta_pp, marker='o', color=colors[method], label='D − ' + LABELS[method])
        axes[1].fill_between(rows.queries, rows.lower_percentile_pp, rows.upper_percentile_pp, color=colors[method], alpha=.12)
    axes[1].axhline(0, color='#778899', lw=.8)
    axes[1].set(xlabel='Total paid tests per client/arm', ylabel='D success advantage [pp]', ylim=(-2, 36))
    axes[1].legend(frameon=False, fontsize=9, ncol=3, loc='upper center')
    frequency = frames['frequency_summary'].query('scope == "fixed_24_contexts" and queries == 100')
    for j, method in enumerate(METHODS):
        rows = frequency[frequency.method == method].sort_values('frequency')
        axes[2].bar(np.arange(3) + (j - 1.5) * .19, rows.success_percent, width=.19, color=colors[method], label=LABELS[method])
    axes[2].set(xticks=[0, 1, 2], xticklabels=['0.1', '0.3', '0.6'], xlabel='Frequency [Hz]',
        ylabel='Success at 100 tests [%]', ylim=(0, 105))
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    for extension in ('png', 'svg'):
        path = OUT / ('budget_quality_and_sharing.' + extension)
        fig.savefig(path, dpi=160)
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--full-replay', action='store_true',
        help='Also repeat the expensive complete acquisition replay already recorded by collect.')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    state = runner.setup()
    OUT.mkdir(parents=True, exist_ok=True)
    source_files = sorted(p for p in state['out'].rglob('*') if p.is_file())
    fingerprints = {p.relative_to(runner.REPO).as_posix(): audit_helpers.sha(p) for p in source_files}
    saved = runner.base.read(state['out'] / 'run_manifest.json')
    if runner.body_hash({k: v for k, v in saved.items() if k not in ('signature', 'git_head', 'runtime')}) != state['signature']:
        raise RuntimeError('Saved run manifest differs.')
    if runner.base.read(state['out'] / 'completion.json')['signature'] != state['signature']:
        raise RuntimeError('Continuation is incomplete.')
    audit = runner.audit_setup(state)
    proof = runner.base.read(audit['out'] / 'completion.json')
    if proof['signature'] != audit['audit_signature'] or not proof['all_checks_passed'] or proof['paid_decisions_replayed'] != 8640:
        raise RuntimeError('Frozen prefix proof differs.')
    with patch('src.tracking_budget.minimize', runner.prohibited), patch('src.tracking_six_speed.minimize', runner.prohibited), patch.object(runner.base, 'resolved_rollout', runner.prohibited):
        results = runner.completed_live(state, replay=args.full_replay, workers=args.workers)
        frames, refs, validation, raw, checks, source_checks = reconstruct(state, results, audit)
        with tempfile.TemporaryDirectory(prefix='long_history_gp_check_') as directory:
            runner.base.parallel(direct_worker, state['jobs'], runner.payload(state) | dict(analysis_out=Path(directory)),
                args.workers, 'Independent long-history GP checks')
            direct = pd.concat([pd.read_csv(p) for p in sorted(Path(directory).glob('*.csv'))], ignore_index=True)
        direct.to_csv(OUT / 'independent_GP_checks.csv', index=False)
        curves, visits = diagnostics(state, results, frames, refs, validation, raw)
        figures(curves, frames)
    for path, before in fingerprints.items():
        if audit_helpers.sha(runner.REPO / path) != before:
            raise RuntimeError('Analysis changed a source result: ' + path)
    source_commit = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--',
        state['out'].relative_to(runner.REPO).as_posix()], cwd=runner.REPO, text=True).strip()
    runner.base.save(OUT / 'verification.json', dict(signature=state['signature'], source_commit=source_commit,
        total_paid_requests=len(visits), paid_request_records_validated=len(visits),
        continued_decisions_replayed_in_this_analysis=27360 if args.full_replay else 0,
        continued_decisions_reported_replayed_by_collection=27360, immutable_prefix_requests_checked=8640,
        original_24_scores_reproduced=True, tables_verified=checks, raw_heldout_rollouts=len(raw),
        independent_long_history_predictions=len(direct), maximum_GP_mean_error=float(direct.absolute_mean_error.max()),
        maximum_GP_variance_error=float(direct.absolute_variance_error.max()),
        maximum_acquisition_error=float(direct.absolute_acquisition_error.max()),
        unresolved_paid_requests=int((~visits.integration_resolved).sum()),
        unresolved_heldout_rollouts=int((~raw.integration_resolved).sum()),
        ineligible_heldout_rollouts=int((~raw.eligible).sum()), source_parity_checks=source_checks,
        new_simulations=0, new_hyperparameter_optimization=0, source_results_unchanged=True, source_hashes=fingerprints,
        scope='Matched continuation of an already examined ten-programme homogeneous cohort; descriptive programme-level uncertainty.'))
    print('Verified all 36,000 request records and 8,640 immutable prefix requests, '
        f'{len(checks)} tables, {len(direct)} independent GP predictions and {len(raw):,} held-out rows. Source results unchanged.', flush=True)


if __name__ == '__main__':
    main()
