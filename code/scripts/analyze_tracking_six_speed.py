"""Read-only audit and diagnostics for the frozen six-speed pilot results."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import tracking_six_speed as runner
from src.tracking_six_speed import METHODS, PAIRS
from src.tracking_six_speed import CostSnapshot, _minimum
from src.tracking_budget import feasibility_probability
from src.tracking_online import seed_for
from src.tracking_six_speed_report import evaluate, paired_comparisons, visit_tables

LABELS = dict(zip(METHODS, ['A', 'B', 'D256', 'E256']))
OUT = REPO / 'docs/analysis/six_speed_contextual_pilot'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_table(state, name, frame, keys):
    # Reparse expected CSV to compare its persisted representation, including lists and nulls.
    expected = pd.read_csv(io.StringIO(frame.to_csv(index=False)))
    actual = pd.read_csv(state['out'] / (name + '.csv'))
    if expected.duplicated(keys).any() or actual.duplicated(keys).any():
        raise RuntimeError('Duplicate table keys: ' + name)
    expected, actual = (f.sort_values(keys).reset_index(drop=True) for f in (expected, actual))
    assert_frame_equal(expected, actual, check_dtype=False, check_like=True, rtol=2e-10, atol=2e-10)
    return dict(table=name, rows=len(frame), verified=True)


def verify_repair(state, results):
    directory = state['out'] / 'repairs/gp_bounds_v1'
    journal = runner.read(directory / 'repair.json')
    if (not journal['complete'] or journal['new_signature'] != state['signature'] or
            journal['paid_histories_modified'] or journal['resimulated_rollouts'] != 0):
        raise RuntimeError('Repair provenance differs.')
    transport = {}
    for name, expected in journal['backup_hashes'].items():
        raw = (directory / 'backup' / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() == expected:
            transport[name] = 'exact_bytes'
        elif hashlib.sha256(raw.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')).hexdigest() == expected:
            transport[name] = 'exact_after_restoring_Windows_CRLF_from_Git_LF'
        else:
            raise RuntimeError('Original repair backup differs beyond Git line endings: ' + name)
    original = runner.read(directory / 'backup/run_manifest.json')
    original_spec = {k: original[k] for k in state['specification']}
    signature = hashlib.sha256(json.dumps(original_spec, sort_keys=True).encode()).hexdigest()
    if signature != original['signature'] or signature != journal['old_signature']:
        raise RuntimeError('Original source manifest signature differs.')
    if {p for p in original['source_hashes'] if original['source_hashes'][p] != state['specification']['source_hashes'][p]} != set(runner.LEGACY_BOUND_SOURCES):
        raise RuntimeError('Unexpected repair source transition.')
    if any(original['source_hashes'][p] != expected for p, expected in runner.LEGACY_BOUND_SOURCES.items()):
        raise RuntimeError('Repair did not originate from the original six-speed release.')
    for key in state['specification']:
        if key != 'source_hashes' and original_spec[key] != state['specification'][key]:
            raise RuntimeError('Repair changed design: ' + key)
    rows = []
    for result in results:
        old = runner.read(directory / 'backup/runs' / f'{result["seed"]}.json')
        name = f'runs/{result["seed"]}.json'
        if runner._body_hash(old) != journal['data_bodies'][name]:
            raise RuntimeError('Backup body differs from repair journal.')
        before, after = old['state'], result['state']
        for method in METHODS:
            a, b = before['methods'][method], after['methods'][method]
            assert a['events'] == b['events'][:len(a['events'])]
            assert a['learning'] == b['learning'][:len(a['learning'])]
            assert all(h == bh[:len(h)] for h, bh in zip(a['histories'], b['histories']))
            assert all(b['snapshots'][k] == v for k, v in a['snapshots'].items())
        assert all(after['oracle_cache'][k] == v for k, v in before['oracle_cache'].items())
        complete_unchanged = runner._body_hash(result) == journal['data_bodies'][name]
        if before['round'] == state['cfg']['queries_per_client'] and not complete_unchanged:
            raise RuntimeError('Previously completed seed was modified.')
        rows.append(dict(seed=result['seed'], original_rounds=before['round'], final_rounds=after['round'],
            saved_paid_prefix_per_method=len(state['clients']) * before['round'],
            saved_histories_decisions_and_simulator_records_unchanged=True,
            original_complete_job_body_unchanged=complete_unchanged))
    return pd.DataFrame(rows), transport


def strict_replay(job):
    def prohibited(*args, **kwargs):
        raise RuntimeError('Read-only audit attempted a new optimization or rollout.')
    with patch('src.tracking_budget.minimize', prohibited), patch('src.tracking_six_speed.minimize', prohibited), patch.object(runner, 'resolved_rollout', prohibited):
        runner.replay_worker(job)


def reconstruct_tables(state, results, refs, validation):
    trials, cells, visits, learning = evaluate(results, state['domain'], state['cfg'], refs, validation)
    paired, units = paired_comparisons(trials)
    visit_summary = visit_tables(visits)
    summary = trials.groupby(['method', 'fleet_budget'], sort=False).agg(
        client_runs=('client', 'size'), seeds=('seed', 'nunique'),
        context_success_percent=('context_success', lambda x: 100*x.mean()),
        complete_eligible_on_encountered=('complete_eligible_on_encountered', 'sum'),
        complete_eligible_on_full_grid=('complete_eligible_on_full_grid', 'sum'),
        conditional_mean_J=('J', 'mean'), mean_encountered_contexts=('encountered_contexts', 'mean'),
        mean_unvisited_contexts=('unvisited_contexts', 'mean'), mean_ineligible_requests=('ineligible_queries', 'mean'),
        reference_coverage=('reference_coverage', 'mean')).reset_index()
    counts = cells.groupby(['method', 'fleet_budget'], sort=False).agg(encountered_context_outcomes=('success', 'size'),
        successes=('success', 'sum'), pooled_context_success_percent=('success', lambda x: 100*x.mean())).reset_index()
    summary = summary.merge(counts, on=['method', 'fleet_budget'], validate='one_to_one')
    modes = visits.groupby(['method', 'seed', 'mode'], sort=False).agg(requests=('eligible', 'size'),
        online_eligible=('eligible', 'sum'), new_incumbents=('became_incumbent', 'sum')).reset_index()
    context = cells[cells.fleet_budget == 216].groupby(['method', 'context'], sort=False).agg(
        encounters=('success', 'size'), success_percent=('success', lambda x: 100*x.mean()),
        eligible=('eligible', 'sum'), mean_local_visits=('local_visits', 'mean')).reset_index()
    frames = dict(budget_trials=trials, budget_contexts=cells, paid_visits=visits, learning_diagnostics=learning,
        online_summary=summary, paired_comparisons=paired, per_seed_comparisons=units, visit_summary=visit_summary,
        query_modes=modes, final_context_outcomes=context,
        empirical_references=pd.DataFrame([dict(context=c, **r) for c, r in refs.items()]))
    common = ['seed', 'method']
    keys = dict(budget_trials=common+['fleet_budget', 'client'], budget_contexts=common+['fleet_budget', 'client', 'context'],
        paid_visits=common+['round', 'client'], learning_diagnostics=common+['round'],
        online_summary=['method', 'fleet_budget'], paired_comparisons=['fleet_budget', 'baseline'],
        per_seed_comparisons=['fleet_budget', 'baseline', 'seed'], visit_summary=common+['group'],
        query_modes=common+['mode'], final_context_outcomes=['method', 'context'], empirical_references=['context'])
    checks = [verify_table(state, name, frame, keys[name]) for name, frame in frames.items()]
    return frames, checks


def diagnostics(state, results, frames, out):
    trials, cells, visits = (frames[k] for k in ('budget_trials', 'budget_contexts', 'paid_visits'))
    final = cells[cells.fleet_budget == 216].copy()
    final_trials = trials[trials.fleet_budget == 216].copy()
    frames['online_summary'].query('fleet_budget == 216').to_csv(out / 'final_outcomes.csv', index=False)
    frames['paired_comparisons'].query('fleet_budget == 216').to_csv(out / 'final_comparisons.csv', index=False)
    frames['per_seed_comparisons'].query('fleet_budget == 216').to_csv(out / 'final_seed_effects.csv', index=False)
    allocation = visits.groupby(['method', 'mode'], sort=False).agg(requests=('eligible', 'size'), eligible=('eligible', 'sum'),
        successes=('requested_success', 'sum'), new_incumbents=('became_incumbent', 'sum')).reset_index()
    allocation['ineligible'] = allocation.requests - allocation.eligible
    allocation['ineligible_percent'] = 100*allocation.ineligible/allocation.requests
    allocation.to_csv(out / 'query_allocation.csv', index=False)
    summaries = []
    masks = dict(first_context_after_factory=visits.first_context_visit & ~visits.is_factory,
        first_speed_after_factory=visits.first_speed_visit & ~visits.is_factory,
        repeated_context=visits.context_visit >= 2)
    for label, mask in masks.items():
        for method, g in visits[mask].groupby('method', sort=False):
            by_seed = g.groupby('seed').agg(requested=('requested_success', 'mean'), incumbent=('incumbent_success', 'mean'))
            summaries.append(dict(group=label, method=method, requests=len(g), eligible=int(g.eligible.sum()),
                requested_successes=int(g.requested_success.sum()), incumbent_successes=int(g.incumbent_success.sum()),
                requested_success_percent_mean_seed=100*by_seed.requested.mean(),
                incumbent_success_percent_mean_seed=100*by_seed.incumbent.mean(),
                pooled_requested_success_percent=100*g.requested_success.mean(),
                peer_requests=int((g['mode'] == 'peer_gain_verification').sum()),
                BO_requests=int((g['mode'] == 'feasibility_penalized_LCB').sum())))
    pd.DataFrame(summaries).to_csv(out / 'visit_groups.csv', index=False)
    final[~final.eligible].to_csv(out / 'final_ineligible_contexts.csv', index=False)
    final_trials[final_trials.method == METHODS[0]][['seed', 'client', 'encountered_contexts', 'encountered_speeds', 'unvisited_contexts']].to_csv(out / 'programme_coverage.csv', index=False)
    coverage = final[final.method == METHODS[0]].groupby('local_visits').size().rename('contexts').reset_index()
    coverage.to_csv(out / 'context_visit_counts.csv', index=False)
    transitions = []
    for method, baseline, role in PAIRS:
        a, b = (final_trials[final_trials.method == m].set_index(['seed', 'client']).complete_eligible_on_encountered
                for m in (method, baseline))
        transitions.append(dict(method=method, baseline=baseline, role=role, both=int((a&b).sum()),
            only_method=int((a&~b).sum()), only_baseline=int((~a&b).sum()), neither=int((~a&~b).sum())))
    pd.DataFrame(transitions).to_csv(out / 'eligibility_transitions.csv', index=False)
    failures = []
    for result in results:
        for method, data in result['state']['methods'].items():
            for event in data['events']:
                if event['eligible']:
                    continue
                raw = result['state']['oracle_cache'][runner.cache_key(event)]['result']
                failures.append(dict(seed=result['seed'], method=method, client=event['client'], round=event['round'],
                    context=event['context'], candidate=event['candidate'], mode=event['mode'],
                    failure=raw['failure'], complete=raw['complete'], peak_y=raw['peak_y'], peak_beta_deg=raw['peak_beta_deg'],
                    peak_speed_error=raw['peak_speed_error']))
    failures = pd.DataFrame(failures)
    failures.to_csv(out / 'ineligible_requests.csv', index=False)
    failures.groupby(['method', 'mode', 'failure']).size().rename('requests').reset_index().to_csv(out / 'failure_reasons.csv', index=False)
    bo = visits[visits['mode'] == 'feasibility_penalized_LCB'].copy()
    bo['std'] = np.sqrt(bo.predicted_variance)
    bo['uncertainty_bonus'] = 2*bo['std']
    bo['eligibility_penalty'] = -np.log(np.maximum(bo.eligibility_probability, 1e-6))
    np.testing.assert_allclose(bo.acquisition_score, bo.predicted_log_J-bo.uncertainty_bonus+bo.eligibility_penalty, rtol=1e-12, atol=1e-12)
    bo.groupby(['method', 'eligible'], sort=False).agg(requests=('eligible', 'size'),
        mean_predicted_log_J=('predicted_log_J', 'mean'), mean_std=('std', 'mean'),
        mean_uncertainty_bonus=('uncertainty_bonus', 'mean'), mean_probability=('eligibility_probability', 'mean'),
        mean_eligibility_penalty=('eligibility_penalty', 'mean'), mean_score=('acquisition_score', 'mean')).reset_index().to_csv(out / 'BO_score_terms.csv', index=False)
    visits[visits.get('gp_optimizer_recovery', pd.Series(index=visits.index, dtype=object)).notna()].to_csv(out / 'GP_recoveries.csv', index=False)
    return dict(final_success={LABELS[r.method]:r.context_success_percent for r in frames['online_summary'].query('fleet_budget == 216').itertuples()},
        coverage_mean_contexts=float(final_trials.encountered_contexts.mean()), context_outcomes_per_method=len(final)//4)


def uncertainty_diagnostic(state, results, calibration, out):
    """Fixed-history ranking check, not an alternative online policy campaign."""
    rows = []
    domain, cfg = state['domain'], state['cfg']
    for result in results:
        data = result['state']['methods'][METHODS[2]]
        for r in range(1, cfg['queries_per_client']):
            events = [e for e in data['events'] if e['round'] == r + 1 and e['mode'] == 'feasibility_penalized_LCB']
            if not events:
                continue
            histories = [h[:r] for h in data['histories']]
            model = CostSnapshot(histories, state['phi'], cfg['gp'], cfg['hierarchical'], shared=True)
            for event in events:
                i = event['client']
                tried = {o['action'] for o in histories[i]}
                allowed = [a for a,(c,_) in enumerate(domain.actions) if c == event['context'] and a not in tried]
                mean, variance = model.predict(i, allowed)
                probability = feasibility_probability(histories[i], domain, np.full(len(domain.actions), .5), cfg['gp'])[allowed]
                penalty = -np.log(np.maximum(probability, 1e-6))
                key = allowed.index(event['action'])
                np.testing.assert_allclose([mean[key], variance[key], probability[key]],
                    [event['predicted_log_J'], event['predicted_variance'], event['eligibility_probability']], atol=2e-8, rtol=2e-8)
                random = lambda: np.random.default_rng(seed_for('acquisition', result['seed'], result['clients'][i]['id'], r))
                assert _minimum(mean - 2*np.sqrt(variance) + penalty, allowed, random()) == event['action']
                alternative = _minimum(mean + penalty, allowed, random())
                candidate = domain.actions[alternative][1]
                actual = calibration[event['context']][event['candidate']]
                alternate = calibration[event['context']][candidate]
                rows.append(dict(seed=result['seed'], client=result['clients'][i]['id'], round=r + 1,
                    context=event['context'], actual_candidate=event['candidate'], zero_bonus_candidate=candidate,
                    changed=alternative != event['action'], actual_online_eligible=event['eligible'],
                    actual_calibration_eligible=actual['eligible'], zero_bonus_calibration_eligible=alternate['eligible'],
                    actual_calibration_J=actual['J'], zero_bonus_calibration_J=alternate['J']))
        print(f'Fixed-history uncertainty diagnostic: seed {result["seed"]}.', flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'fixed_history_uncertainty_diagnostic.csv', index=False)
    summary = dict(BO_decisions=len(frame), changed_rankings=int(frame.changed.sum()),
        actual_calibration_eligible=int(frame.actual_calibration_eligible.sum()),
        zero_bonus_calibration_eligible=int(frame.zero_bonus_calibration_eligible.sum()),
        ineligible_online_queries=int((~frame.actual_online_eligible).sum()),
        ineligible_online_queries_with_calibration_eligible_zero_bonus_alternative=int(
            ((~frame.actual_online_eligible) & frame.zero_bonus_calibration_eligible).sum()),
        interpretation='Post-hoc fixed D histories, same posterior/eligibility/candidate pool; zero uncertainty bonus only. Calibration labels score alternatives after ranking. No alternative histories, peer catalogues, held-out campaign or online superiority claim.')
    runner.save(out / 'fixed_history_uncertainty_summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    state = runner.setup()
    OUT.mkdir(parents=True, exist_ok=True)
    source_files = sorted(p for p in state['out'].rglob('*') if p.is_file())
    fingerprints = {p.relative_to(REPO).as_posix():sha(p) for p in source_files}
    manifest = runner.read(state['out'] / 'run_manifest.json')
    if manifest['signature'] != state['signature']:
        raise RuntimeError('Source/configuration signature differs.')
    results = runner.completed_live(state)
    runner.parallel(strict_replay, state['jobs'], runner.payload(state), args.workers, 'Strict no-refit/no-rollout replay')
    repair, backup_transport = verify_repair(state, results)
    repair.to_csv(OUT / 'repair_prefix_verification.csv', index=False)
    calibration_tasks = runner.evaluator_tasks(state, 'calibration')
    calibration = runner.load_evaluation(state, calibration_tasks)
    chosen = {c:runner.calibration_reference(rows) for c, rows in calibration.items()}
    validation_tasks = runner.evaluator_tasks(state, 'validation', runner.validation_selection(state, results, chosen))
    validation = runner.load_evaluation(state, validation_tasks)
    refs = {c:dict(candidate=k, **validation[c][k]) if k is not None else dict(candidate=None, eligible=False, J=None)
            for c, k in chosen.items()}
    frames, checks = reconstruct_tables(state, results, refs, validation)
    summary = diagnostics(state, results, frames, OUT)
    uncertainty = uncertainty_diagnostic(state, results, calibration, OUT)
    assert all(sha(REPO / name) == fingerprint for name, fingerprint in fingerprints.items())
    source_commit = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', state['out'].relative_to(REPO).as_posix()], cwd=REPO, text=True).strip()
    verification = dict(source_commit=source_commit, signature=state['signature'],
        paid_requests=len(frames['paid_visits']), jobs=len(results),
        all_paid_decisions_replayed=True, A_B_refit_states_restored=True, fresh_GP_refitting_claimed=False,
        all_existing_result_files_unchanged=True, tables=checks, evaluator_batches=len(calibration_tasks)+len(validation_tasks),
        calibration_rollouts=sum(len(t['candidates']) for t in calibration_tasks),
        validation_rollouts=sum(len(t['candidates']) for t in validation_tasks),
        references_available=sum(r['eligible'] for r in refs.values()), GP_bound_recoveries=int(frames['paid_visits'].gp_optimizer_recovery.notna().sum()),
        repair_backup_transport=backup_transport,
        new_controller_rollouts=0, new_hyperparameter_optimizations=0, raw_input_sha256=fingerprints,
        basis_accuracy=state['specification']['basis_accuracy'], fixed_history_uncertainty_diagnostic=uncertainty, **summary)
    runner.save(OUT / 'verification.json', verification)
    print(json.dumps({k:v for k,v in verification.items() if k not in ('raw_input_sha256', 'tables')}, indent=2))


if __name__ == '__main__':
    main()
