"""Finite-pool tracking diagnostic. Run --help; protocol: docs/tracking_diagnostic.md."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

# Avoid oversubscribing BLAS when running independent simulation batches.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy

from src.tracking_task import (candidates, fleet, oracle_features, reference_metadata,
                               references, resolved_rollout)
from src.tracking_analysis import (aggregate, analyze_controlled, analyze_realism,
                                   realism_policies, select_policies, write_report)

CONFIG = ROOT/'config/runs/tracking_diagnostic.json'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save_arrays(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    tmp.replace(path)


def source_hashes():
    paths = [Path(__file__), ROOT/'src/tracking_task.py', ROOT/'src/tracking_analysis.py']
    paths += sorted((ROOT/'src/tracking_reference').glob('*.py'))
    paths += [ROOT/'src/tracking_reference/provenance.json']
    return {str(p.relative_to(REPO)).replace('\\', '/'):
            hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in paths}


def setup(config_path=CONFIG, mode='full'):
    cfg = read(config_path)
    if cfg['calibration_seeds'] == [] or cfg['validation_seeds'] == []:
        raise ValueError('Both calibration and validation seeds are required.')
    if set(cfg['calibration_seeds']) & set(cfg['validation_seeds']):
        raise ValueError('Calibration and validation noise seeds must be disjoint.')
    levels = cfg['integration']['substeps']
    if len(levels) < 2 or any(not isinstance(n, int) or n < 1 for n in levels) or levels != sorted(set(levels)):
        raise ValueError('Need at least two strictly increasing positive RK4 resolutions.')
    if cfg['gains']['scaling_anchors'][0] != 1.:
        raise ValueError('The first candidate must be the original nominal gains.')
    if abs(round(cfg['duration_s']/cfg['dt'])*cfg['dt']-cfg['duration_s']) > 1e-9:
        raise ValueError('Duration must be an integer number of controller periods.')
    if cfg['dt'] <= 0 or min(cfg['gains']['nominal']) <= 0:
        raise ValueError('Positive period and nominal gains are required.')
    if any(len(set(cfg[k])) != len(cfg[k]) for k in ('calibration_seeds', 'validation_seeds')):
        raise ValueError('Duplicate seeds are not allowed.')
    effective = deepcopy(cfg)
    if mode == 'smoke':
        effective['calibration_seeds'] = cfg['calibration_seeds'][:1]
        effective['validation_seeds'] = cfg['validation_seeds'][:1]
        effective['realism']['trajectory_seeds'] = cfg['smoke']['realism_trajectory_seeds']
        effective['realism']['speeds_m_s'] = cfg['smoke']['realism_speeds_m_s']
    vehicles, refs = fleet(effective), references(effective)
    if len(vehicles) != 9 or len(refs) != 6:
        raise ValueError('This frozen diagnostic requires nine plants and six controlled contexts.')
    if mode == 'smoke':
        vehicles = [v for v in vehicles if v['id'] in cfg['smoke']['vehicle_ids']]
        refs = [r for r in refs if r['id'] in cfg['smoke']['reference_ids']]
    realism = references(effective, realism=True)
    pool = candidates(effective, smoke=mode == 'smoke')
    for v in vehicles:
        p = v['params']
        if min(p['m'], p['I_z'], p['l_f'], p['l_r'], p['mu']) <= 0:
            raise ValueError('Nonphysical fleet parameters.')
    for r in refs+realism:
        if not all(np.isfinite(r[k]).all() for k in ('t', 'v', 'pose', 'kappa')):
            raise ValueError('Nonfinite reference.')
        if not np.allclose(r['r'], r['v']*r['kappa']):
            raise ValueError('Inconsistent yaw-rate reference.')
    signature_data = dict(configuration=effective, source_hashes=source_hashes(), mode=mode)
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
    tag = cfg['tag']+('_smoke' if mode == 'smoke' else '')
    if Path(tag).name != tag or tag in ('.', '..'):
        raise ValueError('Result tag must be a single directory name.')
    out = ROOT/'results'/tag
    return effective, vehicles, refs, realism, pool, out, signature_data, signature


def initialize(state):
    cfg, vehicles, refs, realism, pool, out, signature_data, signature = state
    path = out/'run_manifest.json'
    if path.exists():
        manifest = read(path)
        if manifest['signature'] != signature:
            raise RuntimeError('Results were created with different code/configuration. '
                               'Use a new config tag; existing results will not be overwritten.')
        if not (out/'reference_arrays.npz').exists():
            # Recover an interruption after the manifest's atomic write.
            arrays = {r['id']+'__'+k: r[k] for r in refs+realism
                      for k in ('t', 'v', 'kappa', 'pose', 'r', 'ay_ref')}
            save_arrays(out/'reference_arrays.npz', arrays)
        return manifest
    out.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                           capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=REPO,
                            capture_output=True, text=True, check=True).stdout.strip()
    expected = len(vehicles)*len(refs)*(len(cfg['calibration_seeds'])+len(cfg['validation_seeds']))
    manifest = dict(signature=signature, **signature_data, git_head=head,
                    git_worktree_dirty=bool(dirty), vehicles=vehicles, candidates=pool,
                    references=[reference_metadata(r, cfg['dt']) for r in refs],
                    realism_references=[reference_metadata(r, cfg['dt']) for r in realism],
                    expected_controlled_batches=expected,
                    expected_realism_batches=len(vehicles)*len(realism)*len(cfg['validation_seeds']),
                    cfl_provenance=read(ROOT/'src/tracking_reference/provenance.json'),
                    runtime=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                                 platform=platform.platform()))
    save(path, manifest)
    arrays = {r['id']+'__'+k: r[k] for r in refs+realism
              for k in ('t', 'v', 'kappa', 'pose', 'r', 'ay_ref')}
    save_arrays(out/'reference_arrays.npz', arrays)
    return manifest


def batch_path(out, bank, vehicle, ref, role, seed):
    return out/'batches'/bank/f'{vehicle}__{ref}__{role}__{seed}.json'


def validate_batch(batch, vehicle, ref, role, seed, pool, signature):
    expect = dict(vehicle=vehicle, reference=ref, role=role, seed=seed, signature=signature)
    if any(batch.get(k) != value for k, value in expect.items()):
        raise RuntimeError('Checkpoint identity or source signature mismatch.')
    ids = [r['candidate'] for r in batch['rows']]
    if sorted(ids) != sorted(c['id'] for c in pool):
        raise RuntimeError('Checkpoint does not contain exactly the expected candidate pool.')
    for row in batch['rows']:
        if any(k not in row for k in ('complete', 'feasible', 'integration_resolved', 'J')):
            raise RuntimeError('Incomplete checkpoint metrics.')
        if row['complete'] and (row['J'] is None or not np.isfinite(row['J'])):
            raise RuntimeError('A completed rollout must have a finite score.')


def worker(job):
    vehicle, ref, pool, cfg, seed, role, signature, trace_ids = job
    start = time.perf_counter()
    rows, traces = resolved_rollout(vehicle['params'], ref, pool, cfg, seed, trace_ids)
    return dict(vehicle=vehicle['id'], reference=ref['id'], role=role, seed=seed,
                bank=ref['bank'], signature=signature, rows=rows,
                runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                elapsed_seconds=time.perf_counter()-start), traces


def execute_jobs(jobs, out, workers):
    pending = []
    for job in jobs:
        v, r, pool, _, seed, role, signature, _ = job
        path = batch_path(out, r['bank'], v['id'], r['id'], role, seed)
        if path.exists():
            validate_batch(read(path), v['id'], r['id'], role, seed, pool, signature)
        else:
            pending.append(job)
    print(f'{len(jobs)-len(pending)}/{len(jobs)} batches already complete; '
          f'{len(pending)} pending ({workers} workers).', flush=True)

    def record(result):
        batch, traces = result
        path = batch_path(out, batch['bank'], batch['vehicle'], batch['reference'], batch['role'], batch['seed'])
        if traces:
            arrays = {gain+'__'+key: np.asarray(value) for gain, trace in traces.items()
                      for key, value in trace.items()}
            trace_path = path.with_suffix('.npz')
            save_arrays(trace_path, arrays)
            batch['trace_file'] = trace_path.name
        save(path, batch)
        eligible = sum(r['feasible'] and r['integration_resolved'] for r in batch['rows'])
        unresolved = sum(not r['integration_resolved'] for r in batch['rows'])
        print(f'{batch["vehicle"]} / {batch["reference"]} / {batch["role"]} / {batch["seed"]}: '
              f'{eligible}/{len(batch["rows"])} eligible, {unresolved} unresolved; '
              f'{batch["elapsed_seconds"]:.1f}s', flush=True)

    if workers == 1:
        for job in pending:
            record(worker(job))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for future in as_completed([executor.submit(worker, job) for job in pending]):
                record(future.result())


def controlled_jobs(state):
    cfg, vehicles, refs, _, pool, _, _, signature = state
    return [(v, r, pool, cfg, seed, role, signature, ())
            for v in vehicles for r in refs
            for role in ('calibration', 'validation') for seed in cfg[role+'_seeds']]


def load_jobs(jobs, out):
    batches = []
    for v, r, pool, _, seed, role, signature, _ in jobs:
        path = batch_path(out, r['bank'], v['id'], r['id'], role, seed)
        if not path.exists():
            raise RuntimeError(f'Missing batch: {path.name}. Run the experiment to complete it.')
        b = read(path)
        validate_batch(b, v['id'], r['id'], role, seed, pool, signature)
        batches.append(b)
    return batches


def freeze_choices(state, batches):
    cfg, vehicles, refs, realism, pool, out, _, _ = state
    _, agg = aggregate(batches)
    selected = select_policies(agg, vehicles, refs, pool)
    schedule = realism_policies(selected, vehicles, refs, realism, cfg)
    for name, value in [('selected_policies.json', selected), ('realism_schedule.json', schedule)]:
        path = out/name
        if path.exists() and read(path) != value:
            raise RuntimeError('Frozen calibration selections do not match the checkpoints.')
        save(path, value)
    return selected, schedule


def realism_jobs(state, schedule):
    cfg, vehicles, _, refs, pool, _, _, signature = state
    jobs = []
    for v in vehicles:
        for r in refs:
            ids = {row['candidate'] for row in schedule
                   if row['vehicle'] == v['id'] and row['context'] == r['id']}
            choices = [c for c in pool if c['id'] in ids]
            for seed in cfg['validation_seeds']:
                # Archive representative selected-policy traces on a predeclared
                # first reference/seed, not a case selected for a favorable result.
                trace_ids = tuple(c['id'] for c in choices) if (
                    r['id'] == refs[0]['id'] and seed == cfg['validation_seeds'][0]) else ()
                jobs.append((v, r, choices, cfg, seed, 'validation', signature, trace_ids))
    return jobs


def preflight(state):
    cfg, vehicles, refs, realism, pool, out, _, _ = state
    manifest = initialize(state)
    features = []
    for v in vehicles:
        for r in refs:
            f = oracle_features(v, r, cfg)
            features.append({k: value for k, value in f.items() if k not in ('hy', 'hd_rate')})
    save(out/'oracle_descriptors.json', features)
    save(out/'preflight.json', dict(vehicles=len(vehicles), contexts=len(refs),
                                  candidates=len(pool), realism_references=len(realism),
                                  baseline_closed_loop_stable=all(f['baseline_closed_loop_stable'] for f in features),
                                  expected_controlled_batches=manifest['expected_controlled_batches'],
                                  expected_realism_batches=manifest['expected_realism_batches']))
    print(f'Preflight: {len(vehicles)} plants x {len(refs)} contexts x {len(pool)} gains; '
          f'{len(realism)} held-out references. Output: {out}', flush=True)


def analyze(state):
    cfg, vehicles, refs, _, pool, out, _, _ = state
    manifest = initialize(state)
    batches = load_jobs(controlled_jobs(state), out)
    selected, schedule = freeze_choices(state, batches)
    realism_batches = load_jobs(realism_jobs(state, schedule), out)
    controlled = analyze_controlled(batches, selected, vehicles, refs, pool, cfg, out)
    _, realism_summary = analyze_realism(realism_batches, schedule, out)
    save(out/'oracle_descriptors.json', controlled['features'])
    summary = write_report(out, controlled, realism_summary, refs, cfg, manifest)
    save(out/'completion.json', dict(signature=manifest['signature'],
                                    controlled_batches=len(batches), realism_batches=len(realism_batches)))
    print(f'Report: {out/"report.html"}', flush=True)
    print(f'Controlled evaluations: {summary["failed_evaluations"]} failed; '
          f'{summary["unresolved_evaluations"]} numerically unresolved.', flush=True)
    return summary


def run(state, workers):
    preflight(state)
    jobs = controlled_jobs(state)
    calibration = [job for job in jobs if job[5] == 'calibration']
    validation = [job for job in jobs if job[5] == 'validation']
    execute_jobs(calibration, state[5], workers)
    _, schedule = freeze_choices(state, load_jobs(calibration, state[5]))
    execute_jobs(validation, state[5], workers)
    execute_jobs(realism_jobs(state, schedule), state[5], workers)
    analyze(state)


def collect(state):
    if state[6]['mode'] != 'full':
        raise RuntimeError('Collect accepts only the full run, never smoke outputs.')
    # Revalidate exact batch identities and completeness before staging anything.
    analyze(state)
    out = state[5]
    rel = out.relative_to(REPO).as_posix()
    subprocess.run(['git', 'add', '-f', '--', rel], cwd=REPO, check=True)
    print(f'Staged only {rel}. Review, commit, and push from this branch.', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight', 'smoke', 'run', 'status', 'analyze', 'collect'])
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--workers', type=int, default=1, help='Parallel vectorized batches; e.g. 4 on a desktop.')
    parser.add_argument('--smoke-results', action='store_true', help='Inspect/analyze smoke outputs instead of full results.')
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('--workers must be positive')
    state = setup(args.config, 'smoke' if args.command == 'smoke' or args.smoke_results else 'full')
    if args.command == 'preflight':
        preflight(state)
    elif args.command in ('smoke', 'run'):
        run(state, args.workers)
    elif args.command == 'analyze':
        analyze(state)
    elif args.command == 'collect':
        collect(state)
    else:
        initialize(state)
        jobs = controlled_jobs(state)
        complete = sum(batch_path(state[5], r['bank'], v['id'], r['id'], role, seed).exists()
                       for v, r, _, _, seed, role, _, _ in jobs)
        print(f'Controlled: {complete}/{len(jobs)} checkpoints. Output: {state[5]}')
        if complete == len(jobs):
            _, schedule = freeze_choices(state, load_jobs(jobs, state[5]))
            jobs = realism_jobs(state, schedule)
            count = sum(batch_path(state[5], r['bank'], v['id'], r['id'], role, seed).exists()
                        for v, r, _, _, seed, role, _, _ in jobs)
            print(f'Realism: {count}/{len(jobs)} checkpoints.')


if __name__ == '__main__':
    main()
