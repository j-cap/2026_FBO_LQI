"""Six-speed homogeneous contextual BO pilot. See docs/tracking_six_speed.md."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import scipy
from src.tracking_budget import domain_from_manifest, matern
from src.tracking_confirmation import aggregate_candidates, calibration_reference, pinned_design
from src.tracking_kernel_features import basis_bank
from src.tracking_online import seed_for
from src.tracking_peer_cost import OBSERVATION_FIELDS
from src.tracking_six_speed import METHODS, program, run_fleet, validate_state
from src.tracking_six_speed_report import evaluate, write_report
from src.tracking_spectral import basis_descriptor, load_bases, save_bases
from src.tracking_task import candidates, fleet, reference_metadata, references, resolved_rollout

CONFIG = ROOT / 'config/runs/tracking_six_speed.json'
WORKER = None


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def setup(path=CONFIG, smoke=False):
    requested = read(path)
    tag = requested['tag']
    if Path(tag).name != tag or not tag.startswith('tracking_six_speed_'):
        raise ValueError('Use a new single-directory tracking_six_speed_ tag.')
    if (requested['speeds_m_s'] != [20., 22., 24., 26., 28., 30.] or requested['spectral_rank'] != 256 or
            requested['clients'] != 9 or requested['queries_per_client'] != 24 or
            len(requested['seeds']) != 3 or len(set(requested['seeds'])) != 3 or
            requested['calibration_repeats'] != 2 or requested['validation_repeats'] != 2):
        raise ValueError('This pilot fixes six speeds, rank 256, nine clients, three seeds and 24 requests.')
    if set(requested['seeds']) & ({1100, 1101, 1102} | set(range(2100, 2110)) | set(range(4100, 4130))):
        raise ValueError('Use fresh programme/noise seeds.')
    physics = deepcopy(read(ROOT / requested['physics_config']))
    physics['references']['speeds_m_s'] = requested['speeds_m_s']
    if physics['references']['frequencies_hz'] != [.1, .3, .6]:
        raise ValueError('Frozen maneuver frequencies differ.')
    baseline = read(ROOT / requested['baseline_config'])
    cfg = dict(tag=tag, methods=METHODS, conditions=['homogeneous'],
        gp=baseline['gp'], model=baseline['model'],
        hierarchical=read(ROOT / requested['hierarchical_config'])['hierarchical'],
        acquisition=baseline['acquisition'], relative_tolerance=baseline['relative_tolerance'],
        queries_per_client=requested['queries_per_client'], fleet_budgets=requested['fleet_budgets'])
    cfg['initial_gp_parameters'] = np.log([*cfg['gp']['length_scales'], cfg['gp']['amplitude']]).tolist()
    seeds, n = requested['seeds'], requested['clients']
    if smoke:
        seeds, n = [requested['smoke']['seed']], requested['smoke']['clients']
        cfg.update(queries_per_client=requested['smoke']['queries_per_client'],
                   fleet_budgets=requested['smoke']['fleet_budgets'])
    if (n < 2 or cfg['fleet_budgets'] != sorted(set(cfg['fleet_budgets'])) or
            cfg['fleet_budgets'][0] < n or cfg['fleet_budgets'][-1] != n * cfg['queries_per_client'] or
            any(b % n for b in cfg['fleet_budgets']) or not 1 <= cfg['queries_per_client'] <= 71):
        raise ValueError('Invalid clients, local cap or full-round snapshots.')
    refs, pool = references(physics), candidates(physics)
    cfg['context_metadata'] = [reference_metadata(r, physics['dt']) for r in refs]
    common = next(v for v in fleet(physics) if v['id'] == 'compact_nominal')
    clients = [dict(id=f'client_{i:02d}', plant_id='hom_compact', family='compact',
                    params=deepcopy(common['params'])) for i in range(n)]
    out = ROOT / 'results' / (tag + ('_smoke' if smoke else ''))
    manifest = read(out / 'run_manifest.json') if (out / 'run_manifest.json').exists() else None
    if manifest:
        pinned = pinned_design(dict(configuration=cfg, pool=pool, clients=clients),
            {k: manifest[k] for k in ('configuration', 'pool', 'clients')})
        cfg, pool, clients = (pinned[k] for k in ('configuration', 'pool', 'clients'))
    domain = domain_from_manifest(dict(configuration=physics, candidates=pool,
        references=cfg['context_metadata'], vehicles=[common]))
    basis_path = out / 'public_bases.npz'
    if basis_path.exists():
        if manifest is None:
            raise RuntimeError('Pinned basis lacks its manifest; use a fresh result tag.')
        phi = load_bases(basis_path, [256], domain, manifest['public_bases'])[256]
        with np.load(basis_path) as pinned:
            domain.X = np.asarray(pinned['domain_X']).copy()
    else:
        _, bank = basis_bank(domain, cfg['gp'], cfg['model'], [dict(family='spectral', rank=256)])
        phi = bank['spectral_256']['F'] / np.r_[np.sqrt(cfg['gp']['constant_variance']),
                                               np.full(256, cfg['gp']['amplitude'])]
        if manifest and {'256': basis_descriptor(phi)} != manifest['public_bases']:
            raise RuntimeError('Pinned basis is missing and reconstruction differs.')
    F = phi * np.r_[np.sqrt(cfg['gp']['constant_variance']), np.full(256, cfg['gp']['amplitude'])]
    exact = matern(domain.X, domain.X, cfg['gp']['length_scales'], cfg['gp']['amplitude'], cfg['gp']['constant_variance'])
    error = F @ F.T - exact
    accuracy = dict(relative_kernel_Frobenius_error=float(np.linalg.norm(error) / np.linalg.norm(exact)),
        max_relative_diagonal_error=float(np.max(np.abs(np.diag(error)) / np.diag(exact))))
    accuracy['targets'] = requested['basis_diagnostic_targets']
    accuracy['meets_earlier_kernel_targets'] = all(accuracy[k] <= v for k, v in accuracy['targets'].items())
    if manifest:
        # Derived numerical diagnostics are pinned; verify rather than rehashing roundoff.
        pinned_design({k: accuracy[k] for k in accuracy['targets']},
                      {k: manifest['basis_accuracy'][k] for k in accuracy['targets']})
        accuracy = manifest['basis_accuracy']
    jobs = [dict(condition='homogeneous', fleet_id='homogeneous', replicate=i, seed=s) for i, s in enumerate(seeds)]
    paths = [Path(__file__), ROOT / 'src/tracking_six_speed.py', ROOT / 'src/tracking_six_speed_report.py',
        ROOT / 'src/tracking_hierarchical.py', ROOT / 'src/tracking_peer_cost.py', ROOT / 'src/tracking_budget.py',
        ROOT / 'src/tracking_online.py', ROOT / 'src/tracking_online_lcb.py', ROOT / 'src/tracking_spectral.py',
        ROOT / 'src/tracking_kernel_features.py', ROOT / 'src/tracking_confirmation.py',
        ROOT / 'src/tracking_context.py', ROOT / 'src/tracking_task.py']
    paths += sorted((ROOT / 'src/tracking_reference').glob('*.py'))
    paths += [ROOT / requested[k] for k in ('physics_config', 'baseline_config', 'hierarchical_config')]
    specification = dict(configuration=cfg, requested=requested, mode='smoke' if smoke else 'pilot',
        clients=clients, pool=pool, jobs=jobs, public_bases={'256': basis_descriptor(phi)}, basis_accuracy=accuracy,
        programmes={str(s): {c['id']: program(domain, cfg, s, c['id']) for c in clients} for s in seeds},
        source_hashes={p.relative_to(REPO).as_posix(): digest(p) for p in paths},
        historical_training_records=0, worker_evaluation_access=False,
        context_rule='Uniform independent speed draws with replacement; balanced randomized frequency blocks; known before gain selection.',
        primary_metric='Final per-client success over encountered contexts, then mean over clients/seeds; no full-grid coverage claim.',
        inference_rule='Frozen Gaussian information messages; lazy same-model predictions only for the scheduled context.',
        replay_rule='D/E decisions recomputed; A/B replay from logged pre-query fit states, without fresh optimization.',
        evaluator_rule='Fresh dense calibration and independent validation union including every paid gain; never learner inputs.')
    signature = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    return dict(cfg=cfg, requested=requested, physics=physics, clients=clients, pool=pool, refs=refs,
        domain=domain, phi=phi, jobs=jobs, out=out, specification=specification, signature=signature)


def initialize(state):
    path = state['out'] / 'run_manifest.json'
    if path.exists():
        if read(path)['signature'] != state['signature']:
            raise RuntimeError('Code/configuration changed; use a new result tag.')
    else:
        save(path, dict(signature=state['signature'], **state['specification'],
            git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                         pandas=pd.__version__, platform=platform.platform())))
    if not (state['out'] / 'public_bases.npz').exists():
        save_bases(state['out'] / 'public_bases.npz', {256: state['phi']}, state['domain'])
    load_bases(state['out'] / 'public_bases.npz', [256], state['domain'], state['specification']['public_bases'])


def run_path(state, job):
    return state['out'] / 'runs' / f'{job["seed"]}.json'


def validate_job(state, job, result, complete=False, replay=False):
    if (result.get('signature') != state['signature'] or result.get('clients') != state['clients'] or
            any(result.get(k) != v for k, v in job.items())):
        raise RuntimeError('Checkpoint identity differs.')
    validate_state(result['state'], state['domain'], state['cfg'], state['clients'], job['seed'],
                   state['phi'], complete=complete, replay=replay)
    for data in result['state']['methods'].values():
        for e in data['events']:
            entry = result['state']['oracle_cache'].get(cache_key(e))
            if entry is None or any(entry[k] != e[k] for k in ('noise_seed', 'client', 'local_evaluation', 'context')):
                raise RuntimeError('Requested simulation provenance differs.')
            raw = entry['result']
            if raw['candidate'] != e['candidate'] or any(raw[k] != e[k] for k in ('complete', 'feasible', 'integration_resolved', 'substeps')):
                raise RuntimeError('Paid outcome differs from simulator record.')
            if e['eligible'] and not np.isclose(e['J'], raw['J'], rtol=1e-12, atol=1e-12):
                raise RuntimeError('Paid objective differs from simulator record.')


def cache_key(plan):
    return f'{plan["client"]}__{plan["local_evaluation"]}__{plan["context"]}__{plan["candidate"]}'


def observe(plans, cache, state):
    groups = {}
    for p in plans:
        groups.setdefault((p['client'], p['local_evaluation'], p['context'], p['noise_seed']), []).append(p)
    refs = {r['id']: r for r in state['refs']}
    gains = {c['id']: c for c in state['pool']}
    for (i, local, context, noise), requests in groups.items():
        missing = sorted({p['candidate'] for p in requests if cache_key(p) not in cache})
        if not missing:
            continue
        rows, _ = resolved_rollout(state['clients'][i]['params'], refs[context], [gains[c] for c in missing],
                                   state['physics'], noise)
        for row in rows:
            key = f'{i}__{local}__{context}__{row["candidate"]}'
            cache[key] = dict(client=i, local_evaluation=local, context=context, noise_seed=noise, result=row)
    result = []
    for p in plans:
        row = cache[cache_key(p)]['result']
        eligible = bool(row['complete'] and row['feasible'] and row['integration_resolved'])
        result.append(dict(eligible=eligible, J=row['J'] if eligible else None,
            **{k: row[k] for k in ('complete', 'feasible', 'integration_resolved', 'substeps')}))
    return result


def payload(state):
    # Explicit whitelist: no reports, reference winners, validation labels or old histories.
    return {k: state[k] for k in ('cfg', 'physics', 'clients', 'pool', 'refs', 'domain', 'phi', 'out', 'signature')}


def init_worker(state):
    global WORKER
    WORKER = state


def live_worker(job):
    state = WORKER
    path = run_path(state, job)
    previous = read(path) if path.exists() else None
    if previous:
        validate_job(state, job, previous)
    elapsed = previous.get('elapsed_seconds', 0.) if previous else 0.
    started = time.perf_counter()
    def checkpoint(data):
        save(path, dict(**job, signature=state['signature'], clients=state['clients'], state=data,
                        elapsed_seconds=elapsed + time.perf_counter() - started))
        if data['round'] % 4 == 0:
            print(f'Seed {job["seed"]}: {data["round"]}/{state["cfg"]["queries_per_client"]} rounds', flush=True)
    run_fleet(state['domain'], state['cfg'], state['clients'], job['seed'],
        lambda plans, cache: observe(plans, cache, state), state['phi'], checkpoint,
        previous['state'] if previous else None)
    validate_job(state, job, read(path), complete=True)


def replay_worker(job):
    validate_job(WORKER, job, read(run_path(WORKER, job)), complete=True, replay=True)


def parallel(function, tasks, state, workers, label):
    if workers == 1:
        init_worker(state)
        for i, task in enumerate(tasks, 1):
            function(task)
            if i == len(tasks) or i % 12 == 0:
                print(f'{label}: {i}/{len(tasks)}', flush=True)
    elif tasks:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=multiprocessing.get_context('spawn'),
                                 initializer=init_worker, initargs=(state,)) as pool:
            futures = [pool.submit(function, task) for task in tasks]
            for i, future in enumerate(as_completed(futures), 1):
                future.result()
                if i == len(tasks) or i % 12 == 0:
                    print(f'{label}: {i}/{len(tasks)}', flush=True)


def evaluator_tasks(state, role, selection=None):
    requested = state['specification']['requested']
    settings = requested['smoke'] if state['specification']['mode'] == 'smoke' else requested
    return [dict(role=role, context=c, repeat=rep,
                 noise_seed=seed_for('six_speed_evaluator', state['cfg']['tag'], state['specification']['mode'], role, rep),
                 candidates=state['domain'].candidates if selection is None else sorted(selection[c]))
            for c in state['domain'].contexts for rep in range(settings[role + '_repeats'])]


def evaluation_path(state, task):
    return state['out'] / 'evaluation' / task['role'] / f'{task["context"]}__{task["repeat"]}.json'


def validate_batch(state, task, batch):
    if batch.get('signature') != state['signature'] or any(batch.get(k) != v for k, v in task.items()):
        raise RuntimeError('Evaluator cache identity differs.')
    if [r['candidate'] for r in batch['rows']] != task['candidates']:
        raise RuntimeError('Evaluator gain set differs.')
    for row in batch['rows']:
        if row['complete'] and (row['J'] is None or not np.isfinite(row['J']) or row['J'] <= 0):
            raise RuntimeError('Invalid evaluator objective.')


def evaluation_worker(task):
    state = WORKER
    path = evaluation_path(state, task)
    if path.exists():
        validate_batch(state, task, read(path))
        return
    ref = next(r for r in state['refs'] if r['id'] == task['context'])
    lookup = {c['id']: c for c in state['pool']}
    rows, _ = resolved_rollout(state['clients'][0]['params'], ref, [lookup[c] for c in task['candidates']],
        state['physics'], task['noise_seed'])
    batch = dict(signature=state['signature'], **task, rows=rows)
    validate_batch(state, task, batch)
    save(path, batch)


def load_evaluation(state, tasks):
    buckets = {}
    for task in tasks:
        path = evaluation_path(state, task)
        if not path.exists():
            raise RuntimeError('Missing evaluator batch: ' + str(path))
        batch = read(path)
        validate_batch(state, task, batch)
        buckets.setdefault(task['context'], []).append(batch)
    return {c: aggregate_candidates(rows) for c, rows in buckets.items()}


def completed_live(state, replay=False, workers=1):
    result = []
    for job in state['jobs']:
        path = run_path(state, job)
        if not path.exists():
            raise RuntimeError('Missing online job: ' + str(path))
        data = read(path)
        validate_job(state, job, data, complete=True)
        result.append(data)
    if replay:
        parallel(replay_worker, state['jobs'], payload(state), workers, 'Logged-fit / Gaussian decision replay')
    return result


def validation_selection(state, results, chosen):
    selected = {c: {winner} if winner is not None else set() for c, winner in chosen.items()}
    for result in results:
        for record in result['state']['methods'].values():
            for event in record['events']:
                selected[event['context']].add(event['candidate'])
    # Every context has at least its reference when available; allow empty batches for unavailable references.
    return selected


def preflight(state):
    initialize(state)
    tasks = evaluator_tasks(state, 'calibration')
    record = dict(signature=state['signature'], mode=state['specification']['mode'], jobs=len(state['jobs']),
        online_requests=len(state['jobs']) * len(state['clients']) * state['cfg']['queries_per_client'] * len(METHODS),
        dense_calibration_rollouts=sum(len(t['candidates']) for t in tasks),
        validation_rollout_upper_bound=sum(len(t['candidates']) for t in tasks),
        contexts=len(state['domain'].contexts), public_grid_inputs=len(state['domain'].actions),
        basis_accuracy=state['specification']['basis_accuracy'], historical_training_records=0,
        sampled_programme_statistics=programme_statistics(state))
    save(state['out'] / 'preflight.json', record)
    print(f'{record["mode"]}: {record["jobs"]} jobs, {record["online_requests"]:,} paid requests, '
          f'{record["dense_calibration_rollouts"]:,} offline reference rollouts; {record["contexts"]} contexts.', flush=True)
    a = record['basis_accuracy']
    print(f'Basis errors: Frobenius {a["relative_kernel_Frobenius_error"]:.3%}; '
          f'max diagonal {a["max_relative_diagonal_error"]:.3%}; prior targets met: {a["meets_earlier_kernel_targets"]}.', flush=True)


def programme_statistics(state):
    rows = []
    metadata = {r['id']: r for r in state['cfg']['context_metadata']}
    for job in state['jobs']:
        for client in state['clients']:
            contexts = program(state['domain'], state['cfg'], job['seed'], client['id'])
            speeds = [metadata[c]['speed'] for c in contexts]
            rows.append(dict(seed=job['seed'], client=client['id'], encountered_contexts=len(set(contexts)),
                encountered_speeds=len(set(speeds)), consecutive_speed_repeats=sum(a == b for a, b in zip(speeds, speeds[1:]))))
    return rows


def analyze(state, workers):
    initialize(state)
    # Verify paid choices before opening any evaluator labels.
    results = completed_live(state, replay=True, workers=workers)
    calibration = load_evaluation(state, evaluator_tasks(state, 'calibration'))
    chosen = {c: calibration_reference(rows) for c, rows in calibration.items()}
    tasks = evaluator_tasks(state, 'validation', validation_selection(state, results, chosen))
    validation = load_evaluation(state, tasks)
    refs = {c: dict(candidate=k, **validation[c][k]) if k is not None else dict(candidate=None, eligible=False, J=None)
            for c, k in chosen.items()}
    trials, cells, visits, learning = evaluate(results, state['domain'], state['cfg'], refs, validation)
    write_report(state, trials, cells, visits, learning, refs)
    verification = dict(signature=state['signature'], jobs=len(results), online_requests=len(visits),
        all_local_caps_respected=bool((visits.groupby(['seed', 'method', 'client']).size() == state['cfg']['queries_per_client']).all()),
        all_paid_decisions_replayed_from_logged_fit_states=True, fresh_cross_platform_GP_refits_claimed=False,
        historical_training_records=0, calibration_rollouts=sum(len(t['candidates']) for t in evaluator_tasks(state, 'calibration')),
        validation_rollouts=sum(len(t['candidates']) for t in tasks), unavailable_reference_contexts=sum(not v['eligible'] for v in refs.values()),
        unresolved_online_requests=int((~visits.integration_resolved).sum()),
        public_bases=state['specification']['public_bases'], mode=state['specification']['mode'], confirmation_claim=False)
    save(state['out'] / 'verification.json', verification)
    save(state['out'] / 'completion.json', dict(signature=state['signature'], jobs=len(results)))
    print('Report: ' + str(state['out'] / 'report.html'), flush=True)


def run(state, workers):
    preflight(state)
    pending = []
    for job in state['jobs']:
        path = run_path(state, job)
        if path.exists():
            validate_job(state, job, read(path))
            if read(path)['state']['round'] == state['cfg']['queries_per_client']:
                continue
        pending.append(job)
    parallel(live_worker, pending, payload(state), workers, 'Online jobs')
    results = completed_live(state)
    parallel(evaluation_worker, evaluator_tasks(state, 'calibration'), payload(state), workers, 'Dense calibration')
    calibration = load_evaluation(state, evaluator_tasks(state, 'calibration'))
    chosen = {c: calibration_reference(rows) for c, rows in calibration.items()}
    tasks = evaluator_tasks(state, 'validation', validation_selection(state, results, chosen))
    parallel(evaluation_worker, tasks, payload(state), workers, 'Held-out validation')
    analyze(state, workers)


def collect(state, workers):
    if state['specification']['mode'] != 'pilot':
        raise RuntimeError('Smoke results cannot be collected as the pilot.')
    analyze(state, workers)
    relative = state['out'].relative_to(REPO).as_posix()
    subprocess.run(['git', 'add', '-f', '--', relative], cwd=REPO, check=True)
    print('Staged only ' + relative, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight', 'run', 'smoke', 'status', 'analyze', 'collect'])
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--workers', type=int, default=min(3, os.cpu_count() or 1))
    parser.add_argument('--smoke-results', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    state = setup(args.config, args.command == 'smoke' or args.smoke_results)
    if args.command in ('run', 'smoke'):
        run(state, args.workers)
    elif args.command == 'preflight':
        preflight(state)
    elif args.command == 'analyze':
        analyze(state, args.workers)
    elif args.command == 'collect':
        collect(state, args.workers)
    else:
        complete = sum(run_path(state, j).exists() and
                       read(run_path(state, j))['state']['round'] == state['cfg']['queries_per_client'] for j in state['jobs'])
        print(f'Online: {complete}/{len(state["jobs"])} complete. Report: {state["out"]}/report.html')


if __name__ == '__main__':
    main()
