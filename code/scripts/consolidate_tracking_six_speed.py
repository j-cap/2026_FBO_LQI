"""Read-only audit and consolidation of all three six-speed calibration studies."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

# Frozen runners set single-thread BLAS before numerical imports.
import tracking_six_speed as original
import tracking_six_speed_zero_bonus as zero
import tracking_six_speed_small_bonus as small
import analyze_tracking_six_speed as original_audit
import analyze_tracking_six_speed_zero_bonus as zero_audit
import pandas as pd

from src.tracking_six_speed_report import evaluate
from src.tracking_six_speed_small_bonus_report import write_report as small_report

ROOT, REPO = original.ROOT, original.REPO
OUT = REPO / 'docs/analysis/six_speed_consolidation'
LABELS = {original.METHODS[0]: 'A', original.METHODS[1]: 'B',
    original.METHODS[2]: 'D (beta=2)', original.METHODS[3]: 'E (beta=2)',
    zero.METHODS[4]: 'D (beta=0)', zero.METHODS[5]: 'E (beta=0)',
    small.METHODS[4]: 'D (beta=0.5)', small.METHODS[5]: 'E (beta=0.5)'}


def reconstruct_small(state, results, refs, validation):
    trials, cells, visits, learning = evaluate(results, state['domain'], state['cfg'], refs, validation)
    visits['confidence_multiplier'] = visits.method.map(small.BONUSES)
    with tempfile.TemporaryDirectory(prefix='small_bonus_score_audit_') as directory:
        frames = small_report(dict(state, out=Path(directory)), trials, cells, visits, learning, refs)
        expected = original.read(Path(directory) / 'pilot_readout.json')
        if expected != original.read(state['out'] / 'pilot_readout.json'):
            raise RuntimeError('Saved small-bonus descriptive criteria differ.')
        if ((Path(directory) / 'consolidation_inputs.md').read_text() !=
                (state['out'] / 'consolidation_inputs.md').read_text()):
            raise RuntimeError('Saved consolidation summary differs.')
    unit = ['seed', 'method']
    keys = dict(budget_trials=unit + ['fleet_budget', 'client'],
        budget_contexts=unit + ['fleet_budget', 'client', 'context'],
        paid_visits=unit + ['round', 'client'], learning_diagnostics=unit + ['round'],
        online_summary=['method', 'fleet_budget'], paired_comparisons=['fleet_budget', 'method', 'baseline'],
        per_seed_comparisons=['fleet_budget', 'method', 'baseline', 'seed'], visit_summary=unit + ['group'],
        query_modes=unit + ['mode'], empirical_references=['context'],
        frequency_trials=unit + ['fleet_budget', 'client', 'frequency'],
        frequency_outcomes=['method', 'fleet_budget', 'frequency'],
        frequency_discoveries=unit + ['frequency'], per_seed_outcomes=unit)
    checks = [original_audit.verify_table(state, name, frame, keys[name]) for name, frame in frames.items()]
    return frames, checks


def audit_study(name, runner, replay, reconstruct, workers):
    state = runner.setup()
    fingerprint = {p.relative_to(REPO).as_posix(): original_audit.sha(p)
        for p in sorted(state['out'].rglob('*')) if p.is_file()}
    if original.read(state['out'] / 'run_manifest.json')['signature'] != state['signature']:
        raise RuntimeError('Source/configuration signature differs: ' + name)
    results = runner.completed_live(state)
    original.parallel(replay, state['jobs'], original.payload(state), workers, name + ': strict decision replay')
    repair = None
    if name == 'original':
        _, transport = original_audit.verify_repair(state, results)
        repair = dict(paid_prefixes_preserved=True, backup_transport=transport)
    calibration_tasks = original.evaluator_tasks(state, 'calibration')
    calibration = original.load_evaluation(state, calibration_tasks)
    chosen = {c: original.calibration_reference(rows) for c, rows in calibration.items()}
    validation_tasks = original.evaluator_tasks(state, 'validation', original.validation_selection(state, results, chosen))
    validation = original.load_evaluation(state, validation_tasks)
    refs = {c: dict(candidate=k, **validation[c][k]) if k is not None else dict(candidate=None, eligible=False, J=None)
        for c, k in chosen.items()}
    frames, checks = reconstruct(state, results, refs, validation)
    after = {p.relative_to(REPO).as_posix(): original_audit.sha(p)
        for p in sorted(state['out'].rglob('*')) if p.is_file()}
    if fingerprint != after:
        raise RuntimeError('Source result files modified: ' + name)
    commit = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--',
        state['out'].relative_to(REPO).as_posix()], cwd=REPO, text=True).strip()
    verification = dict(study=name, source_commit=commit, signature=state['signature'], jobs=len(results),
        paid_requests=len(frames['paid_visits']), all_paid_decisions_replayed=True,
        A_B_logged_fits_restored=True, fresh_GP_refitting_claimed=False, tables=checks,
        source_results_unchanged=True, raw_input_sha256=fingerprint,
        calibration_rollouts=sum(len(t['candidates']) for t in calibration_tasks),
        validation_rollouts=sum(len(t['candidates']) for t in validation_tasks),
        evaluator_batches=len(calibration_tasks) + len(validation_tasks),
        available_references=sum(r['eligible'] for r in refs.values()),
        unresolved_paid_requests=int((~frames['paid_visits'].integration_resolved).sum()),
        repair_verification=repair)
    return state, results, frames, verification


def peer_failure_evidence(results):
    """Observed earlier-round same-arm peer information, not a policy replay."""
    rows = []
    for result in results:
        n = len(result['clients'])
        for method, data in result['state']['methods'].items():
            seen = {}
            for r in range(result['state']['round']):
                events = data['events'][r * n:(r + 1) * n]
                for event in events:
                    past = [e for e in seen.get(event['action'], []) if e['client'] != event['client']]
                    failures = [e for e in past if not e['eligible']]
                    rows.append(dict(seed=result['seed'], method=method, client=event['client'],
                        round=r + 1, context=event['context'], candidate=event['candidate'], mode=event['mode'],
                        eligible=event['eligible'], earlier_peer_failure=bool(failures),
                        earlier_peer_success=any(e['eligible'] for e in past),
                        earlier_peer_failure_count=len(failures),
                        earlier_peer_failure_clients=sorted({e['client'] for e in failures}),
                        first_peer_failure_round=min(e['round'] for e in failures) if failures else None))
                # Current-round results were not available at selection time.
                for event in events:
                    seen.setdefault(event['action'], []).append(event)
    return pd.DataFrame(rows)


def study_tables(name, state, results, frames):
    final = state['cfg']['fleet_budgets'][-1]
    trials = frames['budget_trials'].query('fleet_budget == @final')
    cells = frames['budget_contexts'].query('fleet_budget == @final')
    visits = frames['paid_visits']
    allocation = visits.groupby(['method', 'mode'], sort=False).agg(
        requests=('eligible', 'size'), eligible=('eligible', 'sum'),
        requested_successes=('requested_success', 'sum')).reset_index()
    allocation['ineligible'] = allocation.requests - allocation.eligible
    allocation['ineligible_percent'] = 100 * allocation.ineligible / allocation.requests
    outcome = trials.groupby('method', sort=False).agg(client_runs=('client', 'size'), seeds=('seed', 'nunique'),
        context_success_percent=('context_success', lambda x: 100 * x.mean()),
        eligible_context_coverage_percent=('validation_eligible_context_coverage', lambda x: 100 * x.mean()),
        complete_eligible_policies=('complete_eligible_on_encountered', 'sum'),
        failed_paid_requests=('ineligible_queries', 'sum'), conditional_mean_J=('J', 'mean'),
        mean_encountered_contexts=('encountered_contexts', 'mean'),
        mean_encountered_speeds=('encountered_speeds', 'mean')).reset_index()
    metadata = pd.DataFrame(state['cfg']['context_metadata'])[['id', 'frequency']]
    frequency_cells = cells.merge(metadata, left_on='context', right_on='id', validate='many_to_one')
    unit = frequency_cells.groupby(['method', 'seed', 'client', 'frequency'], sort=False).success.mean().reset_index()
    frequency = unit.groupby(['method', 'frequency'], sort=False).agg(
        context_success_percent=('success', lambda x: 100 * x.mean()), client_runs=('client', 'size')).reset_index()
    counts = frequency_cells.groupby(['method', 'frequency'], sort=False).agg(
        encountered_contexts=('success', 'size'), successes=('success', 'sum'),
        eligible=('eligible', 'sum'), median_eligible_relative_gap=('relative_gap', 'median')).reset_index()
    frequency = frequency.merge(counts, on=['method', 'frequency'], validate='one_to_one')
    coverage = trials[trials.method == original.METHODS[0]][['seed', 'client', 'encountered_contexts', 'encountered_speeds']]
    communication = trials.groupby(['method', 'seed'])[['payload_bytes', 'model_payload_bytes', 'peer_payload_bytes']].first()
    communication = communication.groupby('method').mean().reset_index()
    per_seed = trials.groupby(['method', 'seed'], sort=False).agg(
        context_success_percent=('context_success', lambda x: 100 * x.mean()),
        complete_eligible_policies=('complete_eligible_on_encountered', 'sum'),
        failed_paid_requests=('ineligible_queries', 'sum')).reset_index()
    discoveries = []
    for (method, seed, freq), group in visits.groupby(['method', 'seed', 'frequency'], sort=False):
        good = group[group.requested_success].sort_values('fleet_evaluation')
        first = good.iloc[0] if len(good) else None
        discoveries.append(dict(method=method, seed=int(seed), frequency=float(freq), paid_requests=len(group),
            near_reference_requests=len(good), first_round=int(first['round']) if first is not None else None,
            first_candidate=first['candidate'] if first is not None else None,
            first_mode=first['mode'] if first is not None else None))
    thresholds = []
    for threshold in (.025, .05, .10, .20):
        frame = cells.assign(success_at_threshold=cells.eligible & cells.reference_available & (cells.relative_gap <= threshold))
        rates = frame.groupby(['method', 'seed', 'client']).success_at_threshold.mean().groupby(['method', 'seed']).mean()
        for (method, seed), rate in rates.items():
            thresholds.append(dict(method=method, seed=int(seed), threshold=threshold, context_success_percent=100 * float(rate)))
    peer = peer_failure_evidence(results)
    bo = peer[peer['mode'] == 'feasibility_penalized_LCB'].copy()
    bo['failed'] = ~bo.eligible
    bo['failed_after_peer_failure'] = bo.failed & bo.earlier_peer_failure
    bo['eligible_after_peer_failure'] = bo.eligible & bo.earlier_peer_failure
    failure_summary = bo.groupby(['method', 'seed'], sort=False).agg(BO_requests=('eligible', 'size'),
        failed_BO_requests=('failed', 'sum'), requests_after_peer_failure=('earlier_peer_failure', 'sum'),
        failed_after_peer_failure=('failed_after_peer_failure', 'sum'),
        eligible_after_peer_failure=('eligible_after_peer_failure', 'sum')).reset_index()
    pairs = frames['paired_comparisons'].query('fleet_budget == @final').copy()
    units = frames['per_seed_comparisons'].query('fleet_budget == @final').copy()
    output = dict(final_outcomes=outcome, final_comparisons=pairs, final_seed_effects=units,
        query_allocation=allocation, frequency_outcomes=frequency, programme_coverage=coverage,
        communication=communication, per_seed_outcomes=per_seed,
        discoveries=pd.DataFrame(discoveries), threshold_sensitivity=pd.DataFrame(thresholds),
        earlier_peer_failure_events=peer[peer.earlier_peer_failure], earlier_peer_failure_summary=failure_summary)
    return {key: value.assign(study=name) for key, value in output.items()}


def plot(tables):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype': 'none', 'svg.hashsalt': 'six_speed_consolidation', 'font.size': 10})
    final = tables['final_outcomes'].query('study == "small_bonus"').set_index('method').loc[small.METHODS]
    allocation = tables['query_allocation'].query('study == "small_bonus" and mode == "feasibility_penalized_LCB"').set_index('method').loc[small.METHODS]
    colors = ['#66748a', '#a77a39', '#84b9b0', '#b4acd1', '#007d75', '#7362ad']
    names = ['A', 'B', 'D\nβ=0', 'E\nβ=0', 'D\nβ=0.5', 'E\nβ=0.5']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    panels = [
        ('Near-reference context success', final.context_success_percent.to_numpy(),
         [f'{v:.2f}%' for v in final.context_success_percent], 100),
        ('Complete eligible policies', 100 * final.complete_eligible_policies.to_numpy() / final.client_runs.to_numpy(),
         [f'{v}/27' for v in final.complete_eligible_policies], 100),
        ('Ineligible BO requests', allocation.ineligible_percent.to_numpy(),
         [f'{n}/{d}' for n, d in zip(allocation.ineligible, allocation.requests)], 40)]
    for ax, (title, values, labels, maximum) in zip(axes, panels):
        ax.bar(names, values, color=colors, width=.65, zorder=3)
        for x, y, label in zip(range(6), values, labels):
            ax.text(x, y + maximum * .025, label, ha='center', va='bottom', fontsize=9)
        ax.set(title=title, ylabel='Percent', ylim=(0, maximum))
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2, zorder=0)
    axes[0].set_xlabel('Higher is better')
    axes[1].set_xlabel('Over encountered contexts; higher is better')
    axes[2].set_xlabel('Lower is better')
    fig.suptitle('Small-bonus comparison: improved success with a failure tradeoff', fontsize=13)
    fig.text(.5, .025, 'Three fresh seeds on one plant. Primary success uses equal client/seed weights. Exploratory evidence.',
        ha='center', fontsize=9, color='#526674')
    fig.subplots_adjust(left=.05, right=.985, top=.82, bottom=.24, wspace=.28)
    fig.savefig(OUT / 'small_bonus_tradeoff.svg', metadata={'Date': None})
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.7), constrained_layout=True)
    studies = [('original', original.METHODS, 'Initial six-speed'), ('zero_bonus', zero.METHODS, 'Zero-bonus comparison'),
               ('small_bonus', small.METHODS, 'Small-bonus comparison')]
    for ax, (name, methods, title) in zip(axes, studies):
        data = tables['frequency_outcomes'].query('study == @name').pivot(
            index='method', columns='frequency', values='context_success_percent').loc[methods, [.1, .3, .6]]
        ax.imshow(data, vmin=0, vmax=100, cmap='YlGnBu', aspect='auto')
        ax.set(title=title, xticks=range(3), xticklabels=['0.1 Hz', '0.3 Hz', '0.6 Hz'],
            yticks=range(len(methods)), yticklabels=[LABELS[m] for m in methods])
        for i, row in enumerate(data.to_numpy()):
            for j, value in enumerate(row):
                ax.text(j, i, f'{value:.1f}%', ha='center', va='center', color='white' if value > 65 else '#203747')
    fig.suptitle('Frequency-specific final success: compare methods within each study', fontsize=12)
    fig.savefig(OUT / 'frequency_consolidation.svg', metadata={'Date': None})
    plt.close(fig)
    for path in OUT.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    OUT.mkdir(parents=True, exist_ok=True)
    studies = [('original', original, original_audit.strict_replay, original_audit.reconstruct_tables),
        ('zero_bonus', zero, zero_audit.strict_replay, zero_audit.reconstruct_tables),
        ('small_bonus', small, small.replay_worker, reconstruct_small)]
    combined, verifications = {}, []
    for name, runner, replay, reconstruct in studies:
        state, results, frames, verification = audit_study(name, runner, replay, reconstruct, args.workers)
        for key, frame in study_tables(name, state, results, frames).items():
            combined.setdefault(key, []).append(frame)
        verifications.append(verification)
    tables = {key: pd.concat(parts, ignore_index=True) for key, parts in combined.items()}
    for name, frame in tables.items():
        frame.to_csv(OUT / (name + '.csv'), index=False)
    plot(tables)
    for verification in verifications:
        if any(original_audit.sha(REPO / path) != digest
                for path, digest in verification['raw_input_sha256'].items()):
            raise RuntimeError('Source results modified during consolidation.')
    verification = dict(studies=verifications, total_paid_requests=sum(v['paid_requests'] for v in verifications),
        verified_numerical_tables=sum(len(v['tables']) for v in verifications),
        all_paid_decisions_replayed=True, all_source_results_unchanged=True,
        new_controller_rollouts=0, new_hyperparameter_optimizations=0,
        comparisons_rule='Within-study paired effects only; no pooled confirmation claim or cross-study score subtraction.',
        threshold_sensitivity_rule='Post-hoc diagnostic only; the declared primary threshold remains 5%.',
        peer_failure_rule='Earlier-round same-arm exact-context/gain peer observations only. No proposed policy or saved-query claim.')
    original.save(OUT / 'verification.json', verification)
    print(json.dumps({k: v for k, v in verification.items() if k != 'studies'}, indent=2))


if __name__ == '__main__':
    main()
