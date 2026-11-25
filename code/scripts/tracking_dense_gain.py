"""Matched A/B/D/E comparison on the audited denser K_y-extended public pool."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import tracking_six_speed as base
import numpy as np

from src.tracking_budget import domain_from_manifest
from src.tracking_confirmation import aggregate_candidates, calibration_reference, pinned_design
from src.tracking_dense_gain import METHODS, BONUSES, PAIRS, run_fleet, validate_state
from src.tracking_dense_gain_check import representation_check
from src.tracking_dense_gain_report import write_report
from src.tracking_online import seed_for
from src.tracking_six_speed import program
from src.tracking_six_speed_report import evaluate
from src.tracking_spectral import save_bases, load_bases
from src.tracking_task import references

ROOT, REPO = base.ROOT, base.REPO
CONFIG = ROOT / 'config/runs/tracking_dense_gain.json'


def body_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def setup(path=CONFIG, smoke=False):
    path = Path(path)
    requested = base.read(path)
    if (Path(requested['tag']).name != requested['tag'] or not requested['tag'].startswith('tracking_dense_gain_') or
            requested['clients'] != 9 or requested['queries_per_client'] != 24 or requested['validation_repeats'] != 4 or
            requested['fleet_budgets'] != [9, 18, 27, 54, 108, 216] or
            requested['actual_upper_gains'] != [.15, 1., 1.] or len(requested['seeds']) != 3):
        raise ValueError('The matched design fixes nine clients, 24 requests, four validation repeats and a K_y-only extension.')
    used = {1100, 1101, 1102, 5100, 5101, 5102, 5199, 6100, 6101, 6102, 6199, 7100, 7101, 7102, 7199}
    used |= set(range(2100, 2110)) | set(range(4100, 4130))
    if len(set(requested['seeds'])) != 3 or set(requested['seeds']) & used or requested['smoke']['seed'] in used | set(requested['seeds']):
        raise ValueError('Use fresh distinct programme and smoke seeds.')
    source_out = ROOT / 'results' / requested['audit_tag']
    source_path = source_out / 'run_manifest.json'
    source = base.read(source_path)
    if source['mode'] != 'audit' or body_hash({k: v for k, v in source.items() if k not in ('signature', 'git_head', 'runtime')}) != source['signature']:
        raise RuntimeError('Completed gain-domain source signature differs.')
    if base.read(source_out / 'completion.json')['signature'] != source['signature']:
        raise RuntimeError('Gain-domain source is incomplete.')
    mode = 'smoke' if smoke else 'pilot'
    validation_seeds = [seed_for('dense_gain_validation', requested['tag'], mode, r)
        for r in range((requested['smoke'] if smoke else requested)['validation_repeats'])]
    old_seeds = set(source['validation_seeds']) | {
        seed_for('six_speed_evaluator', source['source_tag'], 'pilot', role, r)
        for role in ('calibration', 'validation') for r in range(2)}
    if len(set(validation_seeds)) != len(validation_seeds) or set(validation_seeds) & old_seeds:
        raise RuntimeError('Held-out evaluator noise must be fresh.')
    for name, expected in source['source_hashes'].items():
        if base.digest(REPO / name) != expected:
            raise RuntimeError('Frozen gain-domain source changed: ' + name)
    # Only coordinates/origins are inspected here. No rollout labels select this pool.
    pool = [deepcopy(c) for c in source['pool'] if np.all(np.asarray(c['gains']) <= requested['actual_upper_gains'])]
    if len(pool) != 1292 or pool[:71] != source['pool'][:71]:
        raise RuntimeError('Audited public pool geometry differs.')
    if smoke:
        keep = {c['id'] for c in pool[:71]}
        keep.update(c['id'] for c in pool if c['kind'] == 'old_upper_probe')
        keep.update(c['id'] for c in [c for c in pool if c['kind'] == 'dense_sobol'][:requested['smoke']['inside_added']])
        keep.update(c['id'] for c in [c for c in pool if source['labels'][c['id']]['outside_original']][:requested['smoke']['outside_added']])
        pool = [c for c in pool if c['id'] in keep]
    frozen = base.read(ROOT / 'results' / source['source_tag'] / 'run_manifest.json')
    cfg = deepcopy(frozen['configuration'])
    settings = requested['smoke'] if smoke else requested
    cfg.update(tag=requested['tag'], methods=METHODS.copy(), bonus_by_method=BONUSES.copy(),
        queries_per_client=settings['queries_per_client'], fleet_budgets=settings['fleet_budgets'],
        context_metadata=source['contexts'])
    clients = deepcopy(frozen['clients'][:settings['clients']])
    if any(c['params'] != source['plant'] for c in clients):
        raise RuntimeError('Expected the unchanged homogeneous plant.')
    physics = deepcopy(source['physics'])
    out = ROOT / 'results' / (cfg['tag'] + ('_smoke' if smoke else ''))
    old = base.read(out / 'run_manifest.json') if (out / 'run_manifest.json').exists() else None
    if old:
        pinned = pinned_design(dict(configuration=cfg, pool=pool, clients=clients, physics=physics),
            {k: old[k] for k in ('configuration', 'pool', 'clients', 'physics')})
        cfg, pool, clients, physics = (pinned[k] for k in ('configuration', 'pool', 'clients', 'physics'))
    domain = domain_from_manifest(dict(configuration=physics, candidates=pool,
        references=cfg['context_metadata'], vehicles=[dict(id='common', family='compact')]))
    if (out / 'public_domain.npz').exists():
        load_bases(out / 'public_domain.npz', [], domain, {})
        with np.load(out / 'public_domain.npz', allow_pickle=False) as data:
            domain.X = data['domain_X'].copy()
    seeds = [settings['seed']] if smoke else requested['seeds']
    jobs = [dict(condition='homogeneous', fleet_id='homogeneous', replicate=i, seed=s) for i, s in enumerate(seeds)]
    paths = [Path(__file__), path, ROOT / 'src/tracking_dense_gain.py', ROOT / 'src/tracking_dense_gain_check.py',
        ROOT / 'src/tracking_dense_gain_report.py', ROOT / 'scripts/analyze_tracking_gain_domain.py']
    hashes = deepcopy(frozen['source_hashes'])
    hashes.update({p.relative_to(REPO).as_posix(): base.digest(p) for p in paths})
    source_files = [source_path, source_out / 'completion.json']
    source_files += sorted((source_out / 'batches/calibration').glob('*.json'))
    source_files += sorted((source_out / 'batches/source_check').glob('*.json'))
    source_files += [REPO / name.replace('\\', '/') for name in source['source_inputs']]
    spec = dict(configuration=cfg, requested=requested, mode='smoke' if smoke else 'pilot',
        clients=clients, pool=pool, jobs=jobs, physics=physics,
        public_domain_sha256=hashlib.sha256(np.ascontiguousarray(domain.X, dtype='<f8').tobytes()).hexdigest(),
        programmes={str(s): {c['id']: program(domain, cfg, s, c['id']) for c in clients} for s in seeds},
        source_hashes=hashes, source_audit_signature=source['signature'],
        reference_input_hashes={p.relative_to(REPO).as_posix(): base.digest(p) for p in source_files},
        historical_training_records=0, worker_evaluation_access=False,
        context_rule=frozen['context_rule'], primary_metric=frozen['primary_metric'],
        comparisons=[dict(method=m, baseline=b, role=r) for m, b, r in PAIRS],
        representation='Exact target Matern kernel, active-input whitening and conditional remainder; no rank-256 truncation.',
        inference_rule='Same shared/local Gaussian prior, sharing-strength mixture, noise and information-message algebra.',
        bonus_rule='A/B beta=2, matched D/E beta=0.5; unchanged local eligibility penalty and paid peer-first rule.',
        evaluator_rule='Reuse two archived calibration repeats for reference selection only; four fresh common validation repeats.',
        domain_rule='Geometry-only filter of the audited pool; no eligibility filtering; original gain normalization retained.',
        confirmation_claim=False)
    signature = body_hash(spec)
    return dict(cfg=cfg, requested=requested, physics=physics, clients=clients, pool=pool,
        refs=references(physics), domain=domain, jobs=jobs, out=out, specification=spec, signature=signature)


def initialize(state):
    path = state['out'] / 'run_manifest.json'
    if path.exists():
        if base.read(path)['signature'] != state['signature']:
            raise RuntimeError('Code/configuration changed; use a fresh result tag.')
    else:
        base.save(path, dict(signature=state['signature'], **state['specification'],
            git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            runtime=dict(python=base.platform.python_version(), numpy=np.__version__,
                scipy=base.scipy.__version__, platform=base.platform.platform())))
    if not (state['out'] / 'public_domain.npz').exists():
        save_bases(state['out'] / 'public_domain.npz', {}, state['domain'])


def payload(state):
    return {k: state[k] for k in ('cfg', 'physics', 'clients', 'pool', 'refs', 'domain', 'out', 'signature')}


def validate_job(state, job, result, complete=False, replay=False):
    if result.get('signature') != state['signature'] or result.get('clients') != state['clients'] or any(result.get(k) != v for k, v in job.items()):
        raise RuntimeError('Online checkpoint identity differs.')
    validate_state(result['state'], state['domain'], state['cfg'], state['clients'], job['seed'], complete, replay)
    for record in result['state']['methods'].values():
        for event in record['events']:
            raw = result['state']['oracle_cache'][base.cache_key(event)]
            if any(raw[k] != event[k] for k in ('client', 'local_evaluation', 'context', 'noise_seed')):
                raise RuntimeError('Paid simulator provenance differs.')
            observation = raw['result']
            if observation['candidate'] != event['candidate'] or any(observation[k] != event[k]
                    for k in ('complete', 'feasible', 'integration_resolved', 'substeps')):
                raise RuntimeError('Paid outcome differs.')
            if event['eligible'] and not np.isclose(observation['J'], event['J'], rtol=1e-12, atol=1e-12):
                raise RuntimeError('Paid score differs.')


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
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        run_fleet(state['domain'], state['cfg'], state['clients'], job['seed'],
            lambda plans, cache: base.observe(plans, cache, state), checkpoint, previous['state'] if previous else None)
    validate_job(state, job, base.read(path), complete=True)


def replay_worker(job):
    def prohibited(*args, **kwargs):
        raise RuntimeError('Replay attempted fresh simulation or optimization.')
    with patch('src.tracking_budget.minimize', prohibited), patch('src.tracking_six_speed.minimize', prohibited), patch.object(base, 'resolved_rollout', prohibited):
        validate_job(base.WORKER, job, base.read(base.run_path(base.WORKER, job)), True, True)


def completed_live(state, replay=False, workers=1):
    results = []
    for job in state['jobs']:
        result = base.read(base.run_path(state, job))
        validate_job(state, job, result, complete=True)
        results.append(result)
    if replay:
        base.parallel(replay_worker, state['jobs'], payload(state), workers, 'Paid decision replay')
    return results


def reference_selection(state):
    # This entry point is called only after online decisions complete or replay.
    import analyze_tracking_gain_domain as audit
    for name, expected in state['specification']['reference_input_hashes'].items():
        if base.digest(REPO / name) != expected:
            raise RuntimeError('Frozen evaluator input changed: ' + name)
    source = audit.load_state()
    if source['signature'] != state['specification']['source_audit_signature']:
        raise RuntimeError('Evaluator source signature differs.')
    audit.runner.verify_source_checks(source)
    bank = audit.runner.load_calibration(source)
    chosen = {c: calibration_reference({k: row for k, row in rows.items() if k in state['domain'].candidates})
        for c, rows in bank.items()}
    base.save(state['out'] / 'reference_selection.json', dict(signature=state['signature'], choices=chosen,
        reused_calibration_observations=len(state['pool']) * len(state['domain'].contexts) * 2,
        new_calibration_observations=0, training_use=False))
    return chosen


def evaluation_tasks(state, selection):
    repeats = state['requested']['smoke']['validation_repeats'] if state['specification']['mode'] == 'smoke' else state['requested']['validation_repeats']
    tasks = []
    for context, candidates in selection.items():
        ids = sorted(candidates)
        for repeat in range(repeats):
            for start in range(0, len(ids), state['requested']['evaluation_batch_size']):
                tasks.append(dict(role='validation', context=context, repeat=repeat,
                    noise_seed=seed_for('dense_gain_validation', state['cfg']['tag'], state['specification']['mode'], repeat),
                    chunk=start // state['requested']['evaluation_batch_size'],
                    candidates=ids[start:start + state['requested']['evaluation_batch_size']]))
    return tasks


def evaluation_path(state, task):
    return state['out'] / 'evaluation/validation' / f'{task["context"]}__{task["repeat"]}__{task["chunk"]:04d}.json'


def evaluation_worker(task):
    state = base.WORKER
    path = evaluation_path(state, task)
    if path.exists():
        base.validate_batch(state, task, base.read(path))
        return
    lookup = {c['id']: c for c in state['pool']}
    ref = next(r for r in state['refs'] if r['id'] == task['context'])
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        rows, _ = base.resolved_rollout(state['clients'][0]['params'], ref, [lookup[c] for c in task['candidates']], state['physics'], task['noise_seed'])
    batch = dict(signature=state['signature'], **task, rows=rows)
    base.validate_batch(state, task, batch)
    base.save(path, batch)


def load_evaluation(state, tasks):
    grouped = {}
    for task in tasks:
        batch = base.read(evaluation_path(state, task))
        base.validate_batch(state, task, batch)
        grouped.setdefault(task['context'], []).append(batch)
    return {c: aggregate_candidates(batches) for c, batches in grouped.items()}


def preflight(state):
    initialize(state)
    path = state['out'] / 'representation_check.json'
    if path.exists():
        check = base.read(path)
        if check['signature'] != state['signature'] or not check['all_checks_passed']:
            raise RuntimeError('Representation check cache differs.')
    else:
        check = representation_check(state['domain'], state['cfg'], state['requested']['representation_seed'],
            state['requested']['representation_probe_count'])
        base.save(path, dict(signature=state['signature'], **check))
    paid = len(state['jobs']) * len(state['clients']) * state['cfg']['queries_per_client'] * len(METHODS)
    base.save(state['out'] / 'preflight.json', dict(signature=state['signature'], mode=state['specification']['mode'],
        candidates=len(state['pool']), public_inputs=len(state['domain'].actions), paid_requests=paid,
        methods=METHODS, bonus_by_method=BONUSES, historical_training_records=0,
        new_calibration_rollouts=0, representation_checks_passed=True,
        programme_statistics=base.programme_statistics(state)))
    print(f'{state["specification"]["mode"]}: {len(state["pool"]):,} gain vectors, {len(state["domain"].actions):,} public inputs, {paid:,} paid requests.', flush=True)
    print('Exact shared/local representation verified; archived reference labels are evaluator-only. No new calibration bank.', flush=True)


def analyze(state, workers):
    initialize(state)
    results = completed_live(state, replay=True, workers=workers)
    chosen = reference_selection(state)
    tasks = evaluation_tasks(state, base.validation_selection(state, results, chosen))
    validation = load_evaluation(state, tasks)
    refs = {c: dict(candidate=k, **validation[c][k]) if k is not None else dict(candidate=None, eligible=False, J=None) for c, k in chosen.items()}
    trials, cells, visits, learning = evaluate(results, state['domain'], state['cfg'], refs, validation)
    visits['confidence_multiplier'] = visits.method.map(BONUSES)
    write_report(state, trials, cells, visits, learning, refs)
    base.save(state['out'] / 'verification.json', dict(signature=state['signature'], mode=state['specification']['mode'],
        paid_requests=len(visits), all_local_caps_respected=bool((visits.groupby(['seed', 'method', 'client']).size() == state['cfg']['queries_per_client']).all()),
        all_paid_choices_replayed=True, new_calibration_rollouts=0,
        reused_calibration_observations=len(state['pool']) * len(state['domain'].contexts) * 2,
        validation_rollouts=sum(len(t['candidates']) for t in tasks), historical_training_records=0,
        unavailable_references=sum(not r['eligible'] for r in refs.values()),
        unresolved_online_requests=int((~visits.integration_resolved).sum()), confirmation_claim=False))
    base.save(state['out'] / 'completion.json', dict(signature=state['signature']))
    print('Report: ' + str(state['out'] / 'report.html'), flush=True)


def run(state, workers):
    preflight(state)
    pending = []
    for job in state['jobs']:
        path = base.run_path(state, job)
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
    base.parallel(evaluation_worker, tasks, payload(state), workers, 'Fresh held-out validation')
    analyze(state, workers)


def collect(state, workers):
    if state['specification']['mode'] != 'pilot':
        raise RuntimeError('Smoke outputs cannot be collected as the full pilot.')
    analyze(state, workers)
    subprocess.run(['git', 'add', '-f', '--', state['out'].relative_to(REPO).as_posix()], cwd=REPO, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight', 'smoke', 'run', 'analyze', 'collect'])
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    state = setup(args.config, args.command == 'smoke')
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
