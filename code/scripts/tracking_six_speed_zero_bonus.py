"""Fresh matched six-speed bonus ablation. See docs/tracking_six_speed_zero_bonus.md."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import time

# Import before NumPy: the frozen runner limits BLAS to one thread per worker.
import tracking_six_speed as base
import numpy as np

from src.tracking_confirmation import calibration_reference, pinned_design
from src.tracking_six_speed import program
from src.tracking_six_speed_zero_bonus import BONUSES, METHODS, run_fleet, validate_state
from src.tracking_six_speed_report import evaluate
from src.tracking_six_speed_zero_bonus_report import CONTRASTS, write_report
from src.tracking_spectral import load_bases

ROOT, REPO = base.ROOT, base.REPO
CONFIG = ROOT / 'config/runs/tracking_six_speed_zero_bonus.json'


def setup(path=CONFIG, smoke=False):
    path = Path(path)
    requested = base.read(path)
    if (Path(requested['tag']).name != requested['tag'] or
            not requested['tag'].startswith('tracking_six_speed_zero_bonus_')):
        raise ValueError('Use a separate tracking_six_speed_zero_bonus_ result tag.')
    reference_path = ROOT / requested['reference_config']
    frozen = base.setup(reference_path)
    reference_manifest = frozen['out'] / 'run_manifest.json'
    if not reference_manifest.exists() or base.read(reference_manifest)['signature'] != frozen['signature']:
        raise RuntimeError('Frozen reference source/configuration differs; do not modify the previous experiment.')
    if (requested['clients'] != 9 or requested['queries_per_client'] != 24 or
            requested['fleet_budgets'] != [9, 18, 27, 54, 108, 216] or
            requested['calibration_repeats'] != 2 or requested['validation_repeats'] != 2 or
            len(requested['seeds']) != 3 or len(set(requested['seeds'])) != 3):
        raise ValueError('The matched pilot fixes three seeds, nine clients, 24 tests and two holdout repeats.')
    used = set(frozen['requested']['seeds']) | {frozen['requested']['smoke']['seed'], 1100, 1101, 1102}
    used |= set(range(2100, 2110)) | set(range(4100, 4130))
    if (set(requested['seeds']) & used or requested['smoke']['seed'] in used | set(requested['seeds'])):
        raise ValueError('Programme and smoke seeds must be fresh.')
    cfg = deepcopy(frozen['cfg'])
    cfg.update(tag=requested['tag'], methods=METHODS.copy(), bonus_by_method=BONUSES.copy(),
        queries_per_client=requested['queries_per_client'], fleet_budgets=requested['fleet_budgets'])
    seeds, n = requested['seeds'], requested['clients']
    if smoke:
        seeds, n = [requested['smoke']['seed']], requested['smoke']['clients']
        cfg.update(queries_per_client=requested['smoke']['queries_per_client'],
            fleet_budgets=requested['smoke']['fleet_budgets'])
    if (not 2 <= n <= 9 or cfg['fleet_budgets'] != sorted(set(cfg['fleet_budgets'])) or
            cfg['fleet_budgets'][0] < n or cfg['fleet_budgets'][-1] != n * cfg['queries_per_client'] or
            any(b % n for b in cfg['fleet_budgets']) or not 1 <= cfg['queries_per_client'] <= 71):
        raise ValueError('Invalid clients, local cap or full-round snapshots.')
    clients, pool = deepcopy(frozen['clients'][:n]), deepcopy(frozen['pool'])
    out = ROOT / 'results' / (cfg['tag'] + ('_smoke' if smoke else ''))
    manifest = base.read(out / 'run_manifest.json') if (out / 'run_manifest.json').exists() else None
    if manifest:
        pinned = pinned_design(dict(configuration=cfg, pool=pool, clients=clients),
            {k: manifest[k] for k in ('configuration', 'pool', 'clients')})
        cfg, pool, clients = (pinned[k] for k in ('configuration', 'pool', 'clients'))
    domain, phi = deepcopy(frozen['domain']), frozen['phi'].copy()
    if (out / 'public_bases.npz').exists():
        if manifest is None:
            raise RuntimeError('Pinned public basis lacks its manifest.')
        phi = load_bases(out / 'public_bases.npz', [256], domain, manifest['public_bases'])[256]
        with np.load(out / 'public_bases.npz') as pinned:
            domain.X = np.asarray(pinned['domain_X']).copy()
        if manifest['public_bases'] != frozen['specification']['public_bases']:
            raise RuntimeError('Matched experiment must use the exact frozen public basis.')
    jobs = [dict(condition='homogeneous', fleet_id='homogeneous', replicate=i, seed=s) for i, s in enumerate(seeds)]
    source_hashes = deepcopy(frozen['specification']['source_hashes'])
    paths = [Path(__file__), path, reference_path, ROOT / 'src/tracking_six_speed_zero_bonus.py',
             ROOT / 'src/tracking_six_speed_zero_bonus_report.py']
    source_hashes.update({p.relative_to(REPO).as_posix(): base.digest(p) for p in paths})
    specification = dict(configuration=cfg, requested=requested, mode='smoke' if smoke else 'pilot',
        clients=clients, pool=pool, jobs=jobs, public_bases=frozen['specification']['public_bases'],
        basis_accuracy=frozen['specification']['basis_accuracy'],
        programmes={str(s): {c['id']: program(domain, cfg, s, c['id']) for c in clients} for s in seeds},
        source_hashes=source_hashes, historical_training_records=0, worker_evaluation_access=False,
        reference=dict(tag=frozen['cfg']['tag'], signature=frozen['signature'],
            reused='Public physical design, candidate grid and pinned features only; no prior rollout labels or histories.'),
        comparisons=[dict(method=m, baseline=b, role=r) for m, b, r in CONTRASTS],
        context_rule=frozen['specification']['context_rule'],
        primary_metric=frozen['specification']['primary_metric'],
        inference_rule='Frozen shared/local Gaussian model and mixture uncertainty; beta=0 only in D0/E0 cost acquisition.',
        bonus_rule='A/B/D256/E256 retain beta=2; D256-zero/E256-zero use beta=0; eligibility penalty unchanged.',
        replay_rule='All six decisions reconstructed from own earlier histories; A/B restore logged GP fits without optimization.',
        evaluator_rule='Fresh common evaluator bank includes all six arms; no evaluator labels in worker payload.')
    signature = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    return dict(cfg=cfg, requested=requested, physics=frozen['physics'], clients=clients, pool=pool,
        refs=frozen['refs'], domain=domain, phi=phi, jobs=jobs, out=out, specification=specification, signature=signature)


def validate_job(state, job, result, complete=False, replay=False):
    if (result.get('signature') != state['signature'] or result.get('clients') != state['clients'] or
            any(result.get(k) != v for k, v in job.items())):
        raise RuntimeError('Checkpoint identity differs.')
    validate_state(result['state'], state['domain'], state['cfg'], state['clients'], job['seed'],
        state['phi'], complete=complete, replay=replay)
    for data in result['state']['methods'].values():
        for event in data['events']:
            entry = result['state']['oracle_cache'].get(base.cache_key(event))
            if entry is None or any(entry[k] != event[k] for k in ('noise_seed', 'client', 'local_evaluation', 'context')):
                raise RuntimeError('Requested simulation provenance differs.')
            raw = entry['result']
            if raw['candidate'] != event['candidate'] or any(raw[k] != event[k]
                    for k in ('complete', 'feasible', 'integration_resolved', 'substeps')):
                raise RuntimeError('Paid outcome differs from simulator record.')
            if event['eligible'] and not np.isclose(event['J'], raw['J'], rtol=1e-12, atol=1e-12):
                raise RuntimeError('Paid objective differs from simulator record.')


def live_worker(job):
    state = base.WORKER
    path = base.run_path(state, job)
    previous = base.read(path) if path.exists() else None
    if previous:
        validate_job(state, job, previous)
    elapsed = previous.get('elapsed_seconds', 0.) if previous else 0.
    started = time.perf_counter()
    def checkpoint(data):
        base.save(path, dict(**job, signature=state['signature'], clients=state['clients'], state=data,
            elapsed_seconds=elapsed + time.perf_counter() - started))
        if data['round'] % 4 == 0:
            print(f'Seed {job["seed"]}: {data["round"]}/{state["cfg"]["queries_per_client"]} rounds', flush=True)
    run_fleet(state['domain'], state['cfg'], state['clients'], job['seed'],
        lambda plans, cache: base.observe(plans, cache, state), state['phi'], checkpoint,
        previous['state'] if previous else None)
    validate_job(state, job, base.read(path), complete=True)


def replay_worker(job):
    state = base.WORKER
    validate_job(state, job, base.read(base.run_path(state, job)), complete=True, replay=True)


def completed_live(state, replay=False, workers=1):
    results = []
    for job in state['jobs']:
        path = base.run_path(state, job)
        if not path.exists():
            raise RuntimeError('Missing online job: ' + str(path))
        result = base.read(path)
        validate_job(state, job, result, complete=True)
        results.append(result)
    if replay:
        base.parallel(replay_worker, state['jobs'], base.payload(state), workers, 'Six-arm decision replay')
    return results


def preflight(state):
    base.initialize(state)
    calibration = base.evaluator_tasks(state, 'calibration')
    validation = base.evaluator_tasks(state, 'validation')
    record = dict(signature=state['signature'], mode=state['specification']['mode'], jobs=len(state['jobs']),
        online_requests=len(state['jobs']) * len(state['clients']) * state['cfg']['queries_per_client'] * len(METHODS),
        dense_calibration_rollouts=sum(len(t['candidates']) for t in calibration),
        validation_rollout_upper_bound=sum(len(t['candidates']) for t in validation),
        methods=METHODS, bonus_by_method=BONUSES, contexts=len(state['domain'].contexts),
        public_grid_inputs=len(state['domain'].actions), basis_accuracy=state['specification']['basis_accuracy'],
        historical_training_records=0, sampled_programme_statistics=base.programme_statistics(state))
    base.save(state['out'] / 'preflight.json', record)
    print(f'{record["mode"]}: {record["jobs"]} jobs, {record["online_requests"]:,} paid requests across six arms, '
        f'{record["dense_calibration_rollouts"]:,} common offline reference rollouts.', flush=True)
    print('A/B and D/E beta=2 controls; matched D/E beta=0. Fresh histories; unchanged eligibility penalty.', flush=True)


def analyze(state, workers):
    base.initialize(state)
    results = completed_live(state, replay=True, workers=workers)
    calibration = base.load_evaluation(state, base.evaluator_tasks(state, 'calibration'))
    chosen = {c: calibration_reference(rows) for c, rows in calibration.items()}
    tasks = base.evaluator_tasks(state, 'validation', base.validation_selection(state, results, chosen))
    validation = base.load_evaluation(state, tasks)
    refs = {c: dict(candidate=k, **validation[c][k]) if k is not None else dict(candidate=None, eligible=False, J=None)
        for c, k in chosen.items()}
    trials, cells, visits, learning = evaluate(results, state['domain'], state['cfg'], refs, validation)
    visits['confidence_multiplier'] = visits.method.map(BONUSES)
    write_report(state, trials, cells, visits, learning, refs)
    verification = dict(signature=state['signature'], jobs=len(results), online_requests=len(visits),
        methods=METHODS, bonus_by_method=BONUSES,
        all_local_caps_respected=bool((visits.groupby(['seed', 'method', 'client']).size() == state['cfg']['queries_per_client']).all()),
        all_paid_decisions_replayed_from_logged_fit_states=True, fresh_cross_platform_GP_refits_claimed=False,
        historical_training_records=0, calibration_rollouts=sum(len(t['candidates']) for t in base.evaluator_tasks(state, 'calibration')),
        validation_rollouts=sum(len(t['candidates']) for t in tasks),
        unavailable_reference_contexts=sum(not v['eligible'] for v in refs.values()),
        unresolved_online_requests=int((~visits.integration_resolved).sum()),
        gp_bound_recoveries=sum(bool(e.get('gp_optimizer_recovery')) for result in results
            for data in result['state']['methods'].values() for e in data['events']),
        public_bases=state['specification']['public_bases'], mode=state['specification']['mode'], confirmation_claim=False)
    base.save(state['out'] / 'verification.json', verification)
    base.save(state['out'] / 'completion.json', dict(signature=state['signature'], jobs=len(results)))
    print('Report: ' + str(state['out'] / 'report.html'), flush=True)


def run(state, workers):
    preflight(state)
    pending = []
    for job in state['jobs']:
        path = base.run_path(state, job)
        if path.exists():
            result = base.read(path)
            validate_job(state, job, result)
            if result['state']['round'] == state['cfg']['queries_per_client']:
                continue
        pending.append(job)
    base.parallel(live_worker, pending, base.payload(state), workers, 'Online jobs')
    results = completed_live(state)
    base.parallel(base.evaluation_worker, base.evaluator_tasks(state, 'calibration'),
        base.payload(state), workers, 'Dense calibration')
    calibration = base.load_evaluation(state, base.evaluator_tasks(state, 'calibration'))
    chosen = {c: calibration_reference(rows) for c, rows in calibration.items()}
    tasks = base.evaluator_tasks(state, 'validation', base.validation_selection(state, results, chosen))
    base.parallel(base.evaluation_worker, tasks, base.payload(state), workers, 'Held-out validation')
    analyze(state, workers)


def collect(state, workers):
    if state['specification']['mode'] != 'pilot':
        raise RuntimeError('Smoke results cannot be collected as research results.')
    analyze(state, workers)
    relative = state['out'].relative_to(REPO).as_posix()
    subprocess.run(['git', 'add', '-f', '--', relative], cwd=REPO, check=True)
    print('Staged only ' + relative, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight', 'smoke', 'run', 'analyze', 'collect'])
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    state = setup(args.config, smoke=args.command == 'smoke')
    if args.command == 'preflight':
        preflight(state)
    elif args.command in ('run', 'smoke'):
        run(state, args.workers)
    elif args.command == 'analyze':
        analyze(state, args.workers)
    else:
        collect(state, args.workers)


if __name__ == '__main__':
    main()
