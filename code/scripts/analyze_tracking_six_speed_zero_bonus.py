"""Read-only audit of the matched six-speed D/E zero-bonus experiment."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

# Sets single-thread BLAS before importing numerical libraries.
import tracking_six_speed_zero_bonus as runner
import numpy as np
import pandas as pd

from analyze_tracking_six_speed import sha, verify_table
from src.tracking_six_speed_report import evaluate
from src.tracking_six_speed_zero_bonus import A, B, D2, E2, D0, E0, METHODS, BONUSES
from src.tracking_six_speed_zero_bonus_report import CONTRASTS, LABELS, write_report

ROOT, REPO = runner.ROOT, runner.REPO
OUT = REPO / 'docs/analysis/six_speed_zero_bonus'


def strict_replay(job):
    def prohibited(*args, **kwargs):
        raise RuntimeError('Read-only audit attempted a new optimizer or simulator call.')
    with patch('src.tracking_budget.minimize', prohibited), patch('src.tracking_six_speed.minimize', prohibited), \
            patch.object(runner.base, 'resolved_rollout', prohibited):
        runner.replay_worker(job)


def reconstruct_tables(state, results, refs, validation):
    trials, cells, visits, learning = evaluate(results, state['domain'], state['cfg'], refs, validation)
    visits['confidence_multiplier'] = visits.method.map(BONUSES)
    with tempfile.TemporaryDirectory(prefix='zero_bonus_audit_') as directory:
        temporary = dict(state, out=Path(directory))
        frames = write_report(temporary, trials, cells, visits, learning, refs)
    unit = ['seed', 'method']
    keys = dict(budget_trials=unit + ['fleet_budget', 'client'],
        budget_contexts=unit + ['fleet_budget', 'client', 'context'],
        paid_visits=unit + ['round', 'client'], learning_diagnostics=unit + ['round'],
        online_summary=['method', 'fleet_budget'], paired_comparisons=['fleet_budget', 'method', 'baseline'],
        per_seed_comparisons=['fleet_budget', 'method', 'baseline', 'seed'], visit_summary=unit + ['group'],
        query_modes=unit + ['mode'], final_context_outcomes=['method', 'context'], empirical_references=['context'])
    checks = [verify_table(state, name, frame, keys[name]) for name, frame in frames.items()]
    return frames, checks


def diagnostics(state, results, frames, out):
    final_budget = state['cfg']['fleet_budgets'][-1]
    trials, cells, visits = (frames[k] for k in ('budget_trials', 'budget_contexts', 'paid_visits'))
    final = cells[cells.fleet_budget == final_budget].copy()
    final_trials = trials[trials.fleet_budget == final_budget].copy()
    for source, target in [('online_summary', 'final_outcomes'), ('paired_comparisons', 'final_comparisons'),
                           ('per_seed_comparisons', 'final_seed_effects')]:
        frames[source].query('fleet_budget == @final_budget').to_csv(out / (target + '.csv'), index=False)
    allocation = visits.groupby(['method', 'mode'], sort=False).agg(
        requests=('eligible', 'size'), eligible=('eligible', 'sum'), successes=('requested_success', 'sum'),
        new_incumbents=('became_incumbent', 'sum')).reset_index()
    allocation['ineligible'] = allocation.requests - allocation.eligible
    allocation['ineligible_percent'] = 100 * allocation.ineligible / allocation.requests
    allocation.to_csv(out / 'query_allocation.csv', index=False)
    summaries = []
    masks = dict(first_context_after_factory=visits.first_context_visit & ~visits.is_factory,
        first_speed_after_factory=visits.first_speed_visit & ~visits.is_factory,
        repeated_context=visits.context_visit >= 2)
    for group, mask in masks.items():
        for method, rows in visits[mask].groupby('method', sort=False):
            by_seed = rows.groupby('seed').agg(requested=('requested_success', 'mean'), incumbent=('incumbent_success', 'mean'))
            summaries.append(dict(group=group, method=method, requests=len(rows), eligible=int(rows.eligible.sum()),
                requested_successes=int(rows.requested_success.sum()), incumbent_successes=int(rows.incumbent_success.sum()),
                requested_success_percent_mean_seed=100 * by_seed.requested.mean(),
                incumbent_success_percent_mean_seed=100 * by_seed.incumbent.mean(),
                pooled_requested_success_percent=100 * rows.requested_success.mean(),
                peer_requests=int((rows['mode'] == 'peer_gain_verification').sum()),
                BO_requests=int((rows['mode'] == 'feasibility_penalized_LCB').sum())))
    pd.DataFrame(summaries).to_csv(out / 'visit_groups.csv', index=False)
    final_trials[final_trials.method == A][['seed', 'client', 'encountered_contexts', 'encountered_speeds',
        'unvisited_contexts']].to_csv(out / 'programme_coverage.csv', index=False)
    final[final.method == A].groupby('local_visits').size().rename('contexts').reset_index().to_csv(
        out / 'context_visit_counts.csv', index=False)
    final[~final.eligible].to_csv(out / 'final_ineligible_contexts.csv', index=False)
    per_seed = final_trials.groupby(['method', 'seed'], sort=False).agg(
        context_success_percent=('context_success', lambda x: 100 * x.mean()),
        complete_eligible_policies=('complete_eligible_on_encountered', 'sum'),
        ineligible_paid_requests=('ineligible_queries', 'sum')).reset_index()
    per_seed.to_csv(out / 'per_seed_outcomes.csv', index=False)
    eligibility, transitions, context_contrasts = [], [], []
    for method, baseline, role in CONTRASTS:
        aa, bb = (final_trials[final_trials.method == m].set_index(['seed', 'client']).complete_eligible_on_encountered
                  for m in (method, baseline))
        eligibility.append(dict(method=method, baseline=baseline, role=role, both=int((aa & bb).sum()),
            only_method=int((aa & ~bb).sum()), only_baseline=int((~aa & bb).sum()), neither=int((~aa & ~bb).sum())))
        for seed in sorted(final.seed.unique()):
            aa, bb = (final[(final.method == m) & (final.seed == seed)].set_index(['client', 'context']).success
                      for m in (method, baseline))
            if not aa.index.equals(bb.index):
                raise RuntimeError('Matched final context index differs.')
            transitions.append(dict(method=method, baseline=baseline, role=role, seed=int(seed),
                encountered_contexts=len(aa), both_success=int((aa & bb).sum()),
                only_method_success=int((aa & ~bb).sum()), only_baseline_success=int((~aa & bb).sum()),
                neither_success=int((~aa & ~bb).sum())))
            comparison = pd.DataFrame(dict(method_success=aa.astype(int), baseline_success=bb.astype(int)))
            for context_id, group in comparison.groupby(level='context'):
                context_contrasts.append(dict(method=method, baseline=baseline, role=role, seed=int(seed),
                    context=context_id, encountered_clients=len(group),
                    method_successes=int(group.method_success.sum()), baseline_successes=int(group.baseline_success.sum()),
                    success_count_change=int((group.method_success - group.baseline_success).sum())))
    pd.DataFrame(eligibility).to_csv(out / 'eligibility_transitions.csv', index=False)
    pd.DataFrame(transitions).to_csv(out / 'context_success_transitions.csv', index=False)
    pd.DataFrame(context_contrasts).to_csv(out / 'context_contrasts.csv', index=False)
    context = final.merge(pd.DataFrame(state['cfg']['context_metadata'])[['id', 'speed', 'frequency']],
        left_on='context', right_on='id', validate='many_to_one')
    context.groupby(['method', 'speed', 'frequency'], sort=False).agg(
        encounters=('success', 'size'), successes=('success', 'sum'), eligible=('eligible', 'sum'),
        missing_local_gains=('has_local_eligible_gain', lambda x: int((~x).sum())),
        pooled_success_percent=('success', lambda x: 100 * x.mean())).reset_index().to_csv(
        out / 'context_outcomes.csv', index=False)
    context.groupby(['method', 'frequency'], sort=False).agg(
        encounters=('success', 'size'), successes=('success', 'sum'), eligible=('eligible', 'sum'),
        missing_local_gains=('has_local_eligible_gain', lambda x: int((~x).sum())),
        pooled_success_percent=('success', lambda x: 100 * x.mean()),
        minimum_eligible_relative_gap=('relative_gap', 'min'),
        median_eligible_relative_gap=('relative_gap', 'median'),
        mean_eligible_relative_gap=('relative_gap', 'mean'),
        maximum_eligible_relative_gap=('relative_gap', 'max')).reset_index().to_csv(
        out / 'frequency_outcomes.csv', index=False)
    visits.groupby(['method', 'frequency', 'mode'], sort=False).agg(requests=('eligible', 'size'),
        eligible=('eligible', 'sum'), requested_successes=('requested_success', 'sum'),
        distinct_gains=('candidate', 'nunique')).reset_index().to_csv(out / 'frequency_query_modes.csv', index=False)
    discoveries = []
    for result in results:
        for method in METHODS:
            for context_id in state['domain'].contexts:
                events = visits[(visits.seed == result['seed']) & (visits.method == method) &
                    (visits.context == context_id)].sort_values('fleet_evaluation')
                successful = events[events.requested_success]
                first = successful.iloc[0] if len(successful) else None
                discoveries.append(dict(seed=result['seed'], method=method, context=context_id,
                    paid_requests=len(events), near_reference_requests=len(successful),
                    first_round=int(first['round']) if first is not None else None,
                    first_client=first['client'] if first is not None else None,
                    first_candidate=first['candidate'] if first is not None else None,
                    first_mode=first['mode'] if first is not None else None))
    pd.DataFrame(discoveries).to_csv(out / 'first_near_reference_discoveries.csv', index=False)
    coverage = final.assign(missing_local_gain=~final.has_local_eligible_gain,
        validation_ineligible_selected_gain=final.has_local_eligible_gain & ~final.eligible)
    coverage.groupby('method', sort=False).agg(encountered_contexts=('success', 'size'),
        missing_local_gains=('missing_local_gain', 'sum'),
        validation_ineligible_selected_gains=('validation_ineligible_selected_gain', 'sum')).reset_index().to_csv(
        out / 'local_coverage.csv', index=False)
    failures = []
    for result in results:
        for method, data in result['state']['methods'].items():
            for event in data['events']:
                if event['eligible']:
                    continue
                raw = result['state']['oracle_cache'][runner.base.cache_key(event)]['result']
                failures.append(dict(seed=result['seed'], method=method, client=event['client'], round=event['round'],
                    context=event['context'], candidate=event['candidate'], mode=event['mode'], failure=raw['failure'],
                    complete=raw['complete'], peak_y=raw['peak_y'], peak_beta_deg=raw['peak_beta_deg'],
                    peak_speed_error=raw['peak_speed_error']))
    failures = pd.DataFrame(failures)
    failures.to_csv(out / 'ineligible_requests.csv', index=False)
    failures.groupby(['method', 'mode', 'failure']).size().rename('requests').reset_index().to_csv(
        out / 'failure_reasons.csv', index=False)
    bo = visits[visits['mode'] == 'feasibility_penalized_LCB'].copy()
    acquisition = state['cfg']['acquisition']
    bo['std'] = np.sqrt(bo.predicted_variance)
    bo['uncertainty_bonus'] = bo.confidence_multiplier * bo['std']
    bo['eligibility_penalty'] = -acquisition['feasibility_penalty'] * np.log(
        np.maximum(bo.eligibility_probability, acquisition['probability_floor']))
    np.testing.assert_allclose(bo.acquisition_score,
        bo.predicted_log_J - bo.uncertainty_bonus + bo.eligibility_penalty, rtol=2e-10, atol=2e-10)
    bo.groupby(['method', 'eligible'], sort=False).agg(requests=('eligible', 'size'),
        mean_predicted_log_J=('predicted_log_J', 'mean'), mean_std=('std', 'mean'),
        mean_uncertainty_bonus=('uncertainty_bonus', 'mean'), mean_probability=('eligibility_probability', 'mean'),
        mean_eligibility_penalty=('eligibility_penalty', 'mean'), mean_score=('acquisition_score', 'mean')).reset_index().to_csv(
        out / 'BO_score_terms.csv', index=False)
    # These posterior records were produced after each round. The round-24 fit
    # does not influence any of the 24 paid decisions.
    frames['learning_diagnostics'].query('method in @sharing', local_dict=dict(sharing=[D2, D0])).to_csv(
        out / 'sharing_evolution.csv', index=False)
    final_success = {r.method: float(r.context_success_percent)
        for r in frames['online_summary'].query('fleet_budget == @final_budget').itertuples()}
    return dict(final_success=final_success,
        mean_encountered_contexts=float(final_trials.encountered_contexts.mean()),
        matched_context_outcomes_per_method=int(len(final) // len(METHODS)))


def plot(frames, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype': 'none', 'svg.hashsalt': 'six_speed_zero_bonus', 'font.size': 10})
    final = frames['online_summary'].fleet_budget.max()
    summary = frames['online_summary'].query('fleet_budget == @final').set_index('method').loc[METHODS]
    visits = frames['paid_visits']
    bo = visits[visits['mode'] == 'feasibility_penalized_LCB'].groupby('method').eligible.agg(['size', 'sum']).loc[METHODS]
    colors = ['#66748a', '#a77a39', '#84b9b0', '#b4acd1', '#007d75', '#7362ad']
    labels = ['A', 'B', 'D\nβ=2', 'E\nβ=2', 'D\nβ=0', 'E\nβ=0']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    panels = [
        ('Near-reference context success', summary.context_success_percent.to_numpy(),
         [f'{v:.2f}%' for v in summary.context_success_percent], 100),
        ('Complete eligible policies', 100 * summary.complete_eligible_on_encountered.to_numpy() / summary.client_runs.to_numpy(),
         [f'{n}/27' for n in summary.complete_eligible_on_encountered], 105),
        ('Ineligible BO requests', 100 * (bo['size'] - bo['sum']).to_numpy() / bo['size'].to_numpy(),
         [f'{n}/{d}' for n, d in zip(bo['size'] - bo['sum'], bo['size'])], 50)]
    for ax, (title, values, annotations, maximum) in zip(axes, panels):
        ax.bar(labels, values, color=colors, width=.65, zorder=3)
        for x, y, label in zip(range(len(METHODS)), values, annotations):
            ax.text(x, y + maximum * .025, label, ha='center', va='bottom', fontsize=9)
        ax.set(title=title, ylabel='Percent', ylim=(0, maximum))
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2, zorder=0)
        ax.tick_params(axis='x', length=0)
    axes[0].set_xlabel('Higher is better')
    axes[1].set_xlabel('Over encountered contexts; higher is better')
    axes[2].set_xlabel('Lower is better')
    fig.suptitle('Six-speed matched comparison: zero bonus improves D on average', fontsize=13)
    fig.text(.5, .025, 'Three fresh programme/noise seeds on one plant. Equal client/seed success weights. Exploratory evidence.',
        ha='center', fontsize=9, color='#526674')
    fig.subplots_adjust(left=.05, right=.985, top=.82, bottom=.24, wspace=.28)
    path = out / 'zero_bonus_tradeoff.svg'
    fig.savefig(path, metadata={'Date': None})
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)
    frequencies = [.1, .3, .6]
    frequency = pd.read_csv(out / 'frequency_outcomes.csv').pivot(
        index='method', columns='frequency', values='pooled_success_percent').loc[METHODS, frequencies]
    fig, ax = plt.subplots(figsize=(6.2, 4.7), constrained_layout=True)
    ax.imshow(frequency.to_numpy(), vmin=0, vmax=100, cmap='YlGnBu', aspect='auto')
    ax.set_xticks(range(3), [f'{f:.1f} Hz' for f in frequencies])
    ax.set_yticks(range(len(METHODS)), [LABELS[m] for m in METHODS])
    for i, row in enumerate(frequency.to_numpy()):
        for j, value in enumerate(row):
            ax.text(j, i, f'{value:.1f}%', ha='center', va='center',
                color='white' if value > 65 else '#203747')
    ax.set(title='Near-reference success by maneuver frequency', xlabel='Pooled encountered-context success; higher is better')
    path = out / 'frequency_success.svg'
    fig.savefig(path, metadata={'Date': None})
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    state = runner.setup()
    OUT.mkdir(parents=True, exist_ok=True)
    fingerprints = {p.relative_to(REPO).as_posix(): sha(p)
        for p in sorted(state['out'].rglob('*')) if p.is_file()}
    manifest = runner.base.read(state['out'] / 'run_manifest.json')
    if manifest['signature'] != state['signature']:
        raise RuntimeError('Source/configuration signature differs.')
    results = runner.completed_live(state)
    runner.base.parallel(strict_replay, state['jobs'], runner.base.payload(state), args.workers,
        'Strict six-arm no-refit/no-rollout replay')
    calibration_tasks = runner.base.evaluator_tasks(state, 'calibration')
    calibration = runner.base.load_evaluation(state, calibration_tasks)
    chosen = {c: runner.calibration_reference(rows) for c, rows in calibration.items()}
    validation_tasks = runner.base.evaluator_tasks(state, 'validation',
        runner.base.validation_selection(state, results, chosen))
    validation = runner.base.load_evaluation(state, validation_tasks)
    refs = {c: dict(candidate=k, **validation[c][k]) if k is not None else dict(candidate=None, eligible=False, J=None)
        for c, k in chosen.items()}
    frames, checks = reconstruct_tables(state, results, refs, validation)
    summary = diagnostics(state, results, frames, OUT)
    plot(frames, OUT)
    if fingerprints != {p.relative_to(REPO).as_posix(): sha(p)
            for p in sorted(state['out'].rglob('*')) if p.is_file()}:
        raise RuntimeError('Audit modified source results.')
    source_commit = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--',
        state['out'].relative_to(REPO).as_posix()], cwd=REPO, text=True).strip()
    verification = dict(source_commit=source_commit, signature=state['signature'], jobs=len(results),
        paid_requests=len(frames['paid_visits']), all_paid_decisions_replayed=True,
        A_B_refit_states_restored=True, fresh_GP_refitting_claimed=False,
        all_existing_result_files_unchanged=True, tables=checks,
        evaluator_batches=len(calibration_tasks) + len(validation_tasks),
        calibration_rollouts=sum(len(t['candidates']) for t in calibration_tasks),
        validation_rollouts=sum(len(t['candidates']) for t in validation_tasks),
        references_available=sum(r['eligible'] for r in refs.values()),
        unresolved_online_requests=int((~frames['paid_visits'].integration_resolved).sum()),
        GP_bound_recoveries=int(frames['paid_visits'].get('gp_optimizer_recovery', pd.Series(dtype=object)).notna().sum()),
        new_controller_rollouts=0, new_hyperparameter_optimizations=0,
        raw_input_sha256=fingerprints, basis_accuracy=state['specification']['basis_accuracy'], **summary)
    runner.base.save(OUT / 'verification.json', verification)
    print(json.dumps({k: v for k, v in verification.items() if k not in ('raw_input_sha256', 'tables')}, indent=2))


if __name__ == '__main__':
    main()
