"""Lossless catalogue exchange and fresh matched 1/2-round confirmation.

Run from the repository root; see docs/tracking_delta_catalogue.md.
"""
import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import tracking_communication as previous
frozen = previous.frozen
import numpy as np

from src.tracking_delta_catalogue import run_fleet, validate_state
from src.tracking_delta_audit import replay as audit_replay
from src.tracking_delta_report import write_report, write_audit_report
from src.tracking_confirmation import pinned_design
from src.tracking_dense_gain import METHODS, BONUSES
from src.tracking_online import seed_for
from src.tracking_six_speed import program
from src.tracking_six_speed_report import evaluate
from src.tracking_spectral import load_bases, save_bases

base, ROOT, REPO = frozen.base, frozen.ROOT, frozen.REPO
CONFIG = ROOT / 'config/runs/tracking_delta_catalogue.json'
body_hash = frozen.body_hash


def setup(path=CONFIG, smoke=False):
    path = Path(path)
    requested = base.read(path)
    if (Path(requested['tag']).name != requested['tag'] or not requested['tag'].startswith('tracking_delta_catalogue_') or
            requested['intervals'] != [1, 2] or requested['primary_interval'] != 2 or
            requested['clients'] != 9 or requested['queries_per_client'] != 24 or
            requested['fleet_budgets'] != [9, 18, 27, 54, 108, 216] or requested['validation_repeats'] != 4 or
            len(requested['seeds']) != 10 or len(set(requested['seeds'])) != 10 or
            requested['bootstrap_repeats'] != 20000 or requested['bootstrap_confidence'] != .95):
        raise ValueError('Fixed confirmation: ten seeds, intervals 1/2, primary k=2, nine clients, 24 requests and four validation repeats.')
    source = previous.setup(ROOT / requested['source_config'], smoke=smoke)
    settings = requested['smoke'] if smoke else requested
    if (any(s < 10000 for s in requested['seeds']) or requested['smoke']['seed'] < 10000 or
            requested['smoke']['seed'] in requested['seeds'] or settings['clients'] != len(source['clients']) or
            settings['queries_per_client'] != source['cfg']['queries_per_client'] or settings['fleet_budgets'] != source['cfg']['fleet_budgets']):
        raise ValueError('Use fresh distinct programme/smoke seeds and the frozen client/request design.')
    cfg = deepcopy(source['cfg'])
    cfg['tag'] = requested['tag']
    physics, pool, clients, domain = (source[k] for k in ('physics', 'pool', 'clients', 'domain'))
    out = ROOT / 'results' / (cfg['tag'] + ('_smoke' if smoke else ''))
    if (out / 'run_manifest.json').exists():
        saved = base.read(out / 'run_manifest.json')
        design = pinned_design(dict(configuration=cfg, pool=pool, clients=clients, physics=physics),
            {k: saved[k] for k in ('configuration', 'pool', 'clients', 'physics')})
        cfg, pool, clients, physics = (design[k] for k in ('configuration', 'pool', 'clients', 'physics'))
    if (out / 'public_domain.npz').exists():
        load_bases(out / 'public_domain.npz', [], domain, {})
        with np.load(out / 'public_domain.npz', allow_pickle=False) as archive:
            domain.X = archive['domain_X'].copy()
    seeds = [settings['seed']] if smoke else requested['seeds']
    jobs = [dict(condition='homogeneous', fleet_id='homogeneous', replicate=i, seed=seed, interval=interval)
        for interval in requested['intervals'] for i, seed in enumerate(seeds)]
    paths = [Path(__file__), path, *[ROOT / 'src' / name for name in
        ('tracking_delta_catalogue.py', 'tracking_delta_cache.py', 'tracking_delta_wire.py', 'tracking_delta_audit.py', 'tracking_delta_report.py')]]
    hashes = dict(source['specification']['source_hashes'])
    hashes.update({p.relative_to(REPO).as_posix(): base.digest(p) for p in paths})
    mode = 'smoke' if smoke else 'confirmation'
    validation_seeds = [seed_for('delta_catalogue_validation', cfg['tag'], mode, r) for r in range(settings['validation_repeats'])]
    if len(set(validation_seeds)) != len(validation_seeds) or set(validation_seeds) & set(source['specification']['validation_seeds']):
        raise RuntimeError('Validation repeats must be fresh and common.')
    spec = dict(configuration=cfg, requested=requested, mode=mode, clients=clients, physics=physics, pool=pool, jobs=jobs,
        public_domain_sha256=hashlib.sha256(np.ascontiguousarray(domain.X, dtype='<f8').tobytes()).hexdigest(),
        programmes={str(s): {c['id']: program(domain, cfg, s, c['id']) for c in clients} for s in seeds},
        source_hashes=hashes, source_audit_signature=source['specification']['source_audit_signature'],
        reference_input_hashes=source['specification']['reference_input_hashes'], source_online_signature=source['signature'],
        validation_seeds=validation_seeds, historical_training_records=0, worker_evaluation_access=False, confirmation_claim=False,
        study_role='Fresh fixed-protocol programme/noise confirmation, one homogeneous plant.',
        context_rule=source['specification']['context_rule'], primary_metric=source['specification']['primary_metric'],
        communication_rule='Both channels contact every k rounds. Own data are always fresh; remote information is held between contacts.',
        catalogue_rule='Version-2 peer packets contain only changed best-eligible entries, with persistent sender/coordinator/all-recipient caches.',
        cost_rule='Unchanged version-1 incremental eligible-cost records for D. Failed labels remain private.',
        accounting_rule='Actual 16-byte headers, 24-byte changed peer entries and 20-byte cost records; every recipient and empty packet is counted.',
        tail_rule='Used-before-final-decision traffic and all traffic including the unused final contact are separate.',
        evaluator_rule='Archived calibration selects references only; four fresh common repeats cover the union over all programmes, intervals and arms.',
        comparison_order=['D2-A2 primary', 'D2-B2 secondary', 'D2-E2 added-cost-sharing control'],
        inference_rule='No model, prior, bonus, gain-pool, context, noise, eligibility or peer-first change.',
        uncertainty_rule='Paired programme-level 95% percentile bootstrap intervals, descriptive with ten programme units; no client pseudo-replication or noninferiority claim.')
    return dict(cfg=cfg, requested=requested, physics=physics, pool=pool, clients=clients, domain=domain,
        refs=source['refs'], out=out, jobs=jobs, specification=spec, signature=body_hash(spec))

def initialize(state):
    frozen.initialize(state)


def payload(state):
    # Fresh online workers have no source histories, evaluator labels or audit payload.
    return frozen.payload(state)


def audit_setup(state):
    source = previous.setup(ROOT / state['requested']['source_config'])
    saved = base.read(source['out'] / 'run_manifest.json')
    if saved['signature'] != source['signature'] or base.read(source['out'] / 'completion.json')['signature'] != source['signature']:
        raise RuntimeError('Completed compact-communication source is required.')
    for job in source['jobs']:
        previous.validate_job(source, job, base.read(previous.run_path(source, job)), complete=True)
    paths = [previous.run_path(source, j) for j in source['jobs']]
    spec = dict(source_signature=source['signature'],
        source_runs={p.relative_to(source['out']).as_posix(): base.digest(p) for p in paths},
        source_hashes=state['specification']['source_hashes'], new_simulations=0, new_optimization=0)
    audit = payload(source)
    audit.update(jobs=source['jobs'], source_out=source['out'],
        out=ROOT / 'results' / (state['cfg']['tag'] + '_equivalence'),
        audit_specification=spec, audit_signature=body_hash(spec))
    return audit

def prohibited(*args, **kwargs):
    raise RuntimeError('Read-only replay attempted fresh simulation or optimization.')


def audit_worker(job):
    state = base.WORKER
    path = run_path(state, job)
    if path.exists():
        result = base.read(path)
        if result['signature'] != state['audit_signature']:
            raise RuntimeError('Compact audit cache differs; use a fresh result tag.')
        return
    source = base.read(previous.run_path(dict(out=state['source_out']), job))
    with patch('src.tracking_budget.minimize', prohibited), patch('src.tracking_six_speed.minimize', prohibited), patch.object(base, 'resolved_rollout', prohibited):
        result = audit_replay(source, state['domain'], state['cfg'], state['clients'])
    base.save(path, dict(signature=state['audit_signature'], **result))


def ensure_audit(state, workers):
    audit = audit_setup(state)
    base.parallel(audit_worker, audit['jobs'], audit, workers, 'Lossless catalogue equivalence')
    results = [base.read(run_path(audit, j)) for j in audit['jobs']]
    if any(r['signature'] != audit['audit_signature'] or not r['all_decisions_identical'] or not r['all_recipient_caches_identical'] or
            r['new_simulations'] != 0 or r['new_optimization'] != 0 for r in results):
        raise RuntimeError('Compact exchange equivalence failed.')
    base.save(audit['out'] / 'run_manifest.json', dict(signature=audit['audit_signature'], **audit['audit_specification']))
    write_audit_report(audit, results)
    base.save(audit['out'] / 'completion.json', dict(signature=audit['audit_signature'], all_checks_passed=True,
        paid_decisions_replayed=sum(r['paid_decisions'] for r in results), new_simulations=0))
    print('Lossless catalogue gate: all archived choices, cost packets and recipient caches reproduced; zero new rollouts.', flush=True)
    return audit


def run_path(state, job):
    return state['out'] / 'runs' / f'k{job["interval"]}' / f'{job["seed"]}.json'


def validate_job(state, job, result, complete=False, replay=False):
    if result.get('signature') != state['signature'] or result.get('clients') != state['clients'] or any(result.get(k) != v for k, v in job.items()):
        raise RuntimeError('Online checkpoint identity differs.')
    validate_state(result['state'], state['domain'], state['cfg'], state['clients'], job['seed'], job['interval'], complete, replay)
    for data in result['state']['methods'].values():
        for event in data['events']:
            raw = result['state']['oracle_cache'].get(base.cache_key(event))
            if raw is None or any(raw[k] != event[k] for k in ('client', 'local_evaluation', 'context', 'noise_seed')):
                raise RuntimeError('Paid simulator provenance differs.')
            observation = raw['result']
            if observation['candidate'] != event['candidate'] or any(observation[k] != event[k] for k in ('complete', 'feasible', 'integration_resolved', 'substeps')):
                raise RuntimeError('Paid outcome differs.')
            if event['eligible'] and not np.isclose(observation['J'], event['J'], rtol=1e-12, atol=1e-12):
                raise RuntimeError('Paid objective differs.')


def live_worker(job):
    state = base.WORKER
    path = run_path(state, job)
    previous = base.read(path) if path.exists() else None
    if previous:
        validate_job(state, job, previous)
    elapsed = previous.get('elapsed_seconds', 0.) if previous else 0.
    timing = deepcopy(previous.get('timing', [])) if previous else []
    started = time.perf_counter()
    def checkpoint(data):
        base.save(path, dict(**job, signature=state['signature'], clients=state['clients'], state=data,
            timing=timing, elapsed_seconds=elapsed + time.perf_counter() - started))
        if data['round'] % 4 == 0:
            print(f'k={job["interval"]}, seed {job["seed"]}: {data["round"]}/{state["cfg"]["queries_per_client"]} rounds', flush=True)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        run_fleet(state['domain'], state['cfg'], state['clients'], job['seed'], job['interval'],
            lambda plans, cache: base.observe(plans, cache, state), checkpoint,
            previous['state'] if previous else None, timing_hook=timing.append)
    validate_job(state, job, base.read(path), complete=True)


def replay_worker(job):
    with patch('src.tracking_budget.minimize', prohibited), patch('src.tracking_six_speed.minimize', prohibited), patch.object(base, 'resolved_rollout', prohibited):
        validate_job(base.WORKER, job, base.read(run_path(base.WORKER, job)), True, True)


def completed_live(state, replay=False, workers=1):
    results = []
    for job in state['jobs']:
        result = base.read(run_path(state, job))
        validate_job(state, job, result, complete=True)
        results.append(result)
    if replay:
        base.parallel(replay_worker, state['jobs'], payload(state), workers, 'Paid decision and packet replay')
    return results


def evaluation_tasks(state, selection):
    tasks, width = [], state['requested']['evaluation_batch_size']
    for context, candidates in selection.items():
        ids = sorted(candidates)
        for repeat, noise_seed in enumerate(state['specification']['validation_seeds']):
            for start in range(0, len(ids), width):
                tasks.append(dict(role='validation', context=context, repeat=repeat, noise_seed=noise_seed,
                    chunk=start // width, candidates=ids[start:start + width]))
    return tasks


# These helpers operate solely on their explicit state, and write only to its output.
reference_selection = frozen.reference_selection
evaluation_path = frozen.evaluation_path
evaluation_worker = frozen.evaluation_worker
load_evaluation = frozen.load_evaluation


def preflight(state, workers=3):
    ensure_audit(state, workers)
    initialize(state)
    paid = len(state['jobs']) * len(state['clients']) * state['cfg']['queries_per_client'] * len(METHODS)
    base.save(state['out'] / 'preflight.json', dict(signature=state['signature'], mode=state['specification']['mode'],
        candidates=len(state['pool']), public_inputs=len(state['domain'].actions), paid_requests=paid,
        intervals=state['requested']['intervals'], methods=METHODS, bonus_by_method=BONUSES,
        historical_training_records=0, new_calibration_rollouts=0, equivalence_audit_passed=True,
        programme_statistics=base.programme_statistics(state)))
    print(f'{state["specification"]["mode"]}: {len(state["jobs"])} jobs; {paid:,} paid requests; '
        f'{len(state["pool"]):,} gains; intervals 1/2; archived evaluator references only.', flush=True)


def analyze(state, workers=3):
    audit = ensure_audit(state, workers)
    initialize(state)
    results = completed_live(state, replay=True, workers=workers)
    chosen = reference_selection(state)
    tasks = evaluation_tasks(state, base.validation_selection(state, results, chosen))
    validation = load_evaluation(state, tasks)
    refs = {c: dict(candidate=k, **validation[c][k]) if k is not None else dict(candidate=None, eligible=False, J=None)
        for c, k in chosen.items()}
    frames = []
    for interval in state['requested']['intervals']:
        evaluated = evaluate([r for r in results if r['interval'] == interval], state['domain'], state['cfg'], refs, validation)
        frames.append(tuple(f.assign(interval=interval) for f in evaluated))
    import pandas as pd
    trials, cells, visits, learning = (pd.concat([row[i] for row in frames], ignore_index=True) for i in range(4))
    write_report(state, results, trials, cells, visits, learning, refs, audit)
    base.save(state['out'] / 'verification.json', dict(signature=state['signature'], mode=state['specification']['mode'],
        paid_requests=len(visits), all_local_caps_respected=bool((visits.groupby(['interval', 'seed', 'method', 'client']).size() == state['cfg']['queries_per_client']).all()),
        all_paid_choices_and_packets_replayed=True, equivalence_signature=audit['audit_signature'],
        new_calibration_rollouts=0, reused_calibration_observations=len(state['pool']) * len(state['domain'].contexts) * 2,
        validation_rollouts=sum(len(t['candidates']) for t in tasks), historical_training_records=0,
        unavailable_references=sum(not r['eligible'] for r in refs.values()),
        unresolved_online_requests=int((~visits.integration_resolved).sum()), confirmation_claim=False))
    base.save(state['out'] / 'completion.json', dict(signature=state['signature']))
    print('Report: ' + str(state['out'] / 'report.html'), flush=True)


def run(state, workers=3):
    preflight(state, workers)
    pending = []
    for job in state['jobs']:
        path = run_path(state, job)
        if path.exists():
            previous = base.read(path)
            validate_job(state, job, previous)
            if previous['state']['round'] == state['cfg']['queries_per_client']:
                continue
        pending.append(job)
    base.parallel(live_worker, pending, payload(state), workers, 'Online jobs')
    results = completed_live(state)
    chosen = reference_selection(state)
    tasks = evaluation_tasks(state, base.validation_selection(state, results, chosen))
    base.parallel(evaluation_worker, tasks, payload(state), workers, 'Fresh common held-out validation')
    analyze(state, workers)


def collect(state, workers=3):
    if state['specification']['mode'] != 'confirmation':
        raise RuntimeError('Smoke outputs cannot be collected as the full confirmation.')
    analyze(state, workers)
    audit = audit_setup(state)
    subprocess.run(['git', 'add', '-f', '--', state['out'].relative_to(REPO).as_posix(),
        audit['out'].relative_to(REPO).as_posix()], cwd=REPO, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['audit', 'preflight', 'smoke', 'run', 'analyze', 'collect'])
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    state = setup(args.config, smoke=args.command == 'smoke')
    if args.command == 'audit':
        ensure_audit(state, args.workers)
    elif args.command == 'preflight':
        preflight(state, args.workers)
    elif args.command in ('run', 'smoke'):
        run(state, args.workers)
    elif args.command == 'analyze':
        analyze(state, args.workers)
    else:
        collect(state, args.workers)


if __name__ == '__main__':
    main()
