"""Retrospective contextual-BO comparison. See docs/tracking_budget.md."""
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

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import scipy
from src.tracking_budget import (TargetOracle, dense_references, domain_from_manifest,
    evaluate_job, incumbents, prepare_source, run_suite, source_ids)
from src.tracking_context import nearest_reference
from src.tracking_budget_report import write_report

CONFIG = ROOT/'config/runs/tracking_budget.json'
WORKER_DATA = None


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def setup(config_path=CONFIG, smoke=False):
    cfg = read(config_path)
    original = ROOT/'results'/cfg['input_tag']
    archive = read(original/'run_manifest.json')
    completion = read(original/'completion.json')
    if completion['signature'] != archive['signature']:
        raise RuntimeError('Original archive is incomplete or inconsistent.')
    for path, expected in archive['source_hashes'].items():
        if digest(REPO/path) != expected:
            raise RuntimeError('The archived v1 diagnostic source has changed: '+path)
    raw = pd.read_csv(original/'controlled_evaluations.csv', float_precision='round_trip')
    domain = domain_from_manifest(archive)
    acfg = archive['configuration']
    expected = len(domain.families)*len(domain.actions)*(len(acfg['calibration_seeds'])+len(acfg['validation_seeds']))
    keys = ['vehicle', 'context', 'role', 'seed', 'candidate']
    if len(raw) != expected or raw.duplicated(keys).any():
        raise RuntimeError('Original observation table has missing or duplicate cases.')
    if set(raw.vehicle) != set(domain.families) or set(raw.context) != set(domain.contexts):
        raise RuntimeError('Unexpected vehicles or contexts in original table.')
    for role in ('calibration', 'validation'):
        block = raw[raw.role == role]
        for vehicle in domain.families:
            for seed in acfg[role+'_seeds']:
                present = block[(block.vehicle == vehicle) & (block.seed == seed)]
                if set(zip(present.context, present.candidate)) != set(domain.actions):
                    raise RuntimeError('Missing target candidate observations.')
    calibration = raw[raw.role == 'calibration'].copy()
    validation = raw[raw.role == 'validation'].copy()
    if cfg['budgets'] != sorted(set(cfg['budgets'])) or min(cfg['budgets']) < len(domain.contexts):
        raise ValueError('Budgets must be increasing and cover one initial rollout per context.')
    if max(cfg['budgets']) > len(domain.actions):
        raise ValueError('Budget exceeds the finite action pool.')
    if len(set(cfg['optimizer_seeds'])) != len(cfg['optimizer_seeds']):
        raise ValueError('Duplicate optimizer seeds.')
    targets = sorted(domain.families)
    if smoke:
        cfg = deepcopy(cfg)
        cfg['optimizer_seeds'] = cfg['smoke']['optimizer_seeds']
        cfg['budgets'] = cfg['smoke']['budgets']
        targets = cfg['smoke']['target_ids']
    paths = [Path(__file__), ROOT/'src/tracking_budget.py', ROOT/'src/tracking_context.py']
    hashes = {p.relative_to(REPO).as_posix(): digest(p) for p in paths}
    inputs = {name: digest(original/name) for name in (
        'run_manifest.json', 'completion.json', 'controlled_evaluations.csv')}
    specification = dict(configuration=cfg, mode='smoke' if smoke else 'full', targets=targets,
                         source_hashes=hashes, input_hashes=inputs)
    signature = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    tag = cfg['tag']+('_smoke' if smoke else '')
    if Path(tag).name != tag or Path(cfg['input_tag']).name != cfg['input_tag']:
        raise ValueError('Tags must be single directory names.')
    return dict(cfg=cfg, archive=archive, domain=domain, calibration=calibration,
                validation=validation, targets=targets, original=original,
                out=ROOT/'results'/tag, specification=specification, signature=signature)


def fold_id(state, target, protocol):
    group = target if protocol == 'leave_vehicle_out' else state['domain'].families[target]
    return protocol+'__'+group


def initialize(state):
    out = state['out']; path = out/'run_manifest.json'
    if path.exists():
        manifest = read(path)
        if manifest['signature'] != state['signature']:
            raise RuntimeError('Different BO code/configuration/input archive: use a new result tag.')
        return manifest
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    manifest = dict(signature=state['signature'], **state['specification'], git_head=head,
                    runtime=dict(python=platform.python_version(), numpy=np.__version__,
                                 scipy=scipy.__version__, pandas=pd.__version__),
                    input_signature=state['archive']['signature'],
                    expected_jobs=len(jobs(state)), contexts=state['domain'].contexts,
                    candidates=state['domain'].candidates,
                    source_history='All archived calibration rollouts from permitted vehicles; data-rich benchmark.')
    save(path, manifest)
    # Use the archived descriptors directly; neither pose regeneration nor any
    # vehicle outcome influences the deterministic assignment fix.
    old = read(state['original']/'realism_schedule.json')
    previous = {r['context']: r['prototype_context'] for r in old}
    audit = [dict(reference=r['id'], previous=previous[r['id']],
                  stable=nearest_reference(r, state['archive']['references'],
                                           tolerance=state['cfg']['tie_tolerance']))
             for r in state['archive']['realism_references']]
    save(out/'reference_assignment_v2.json', audit)
    return manifest


def priors(state):
    initialize(state)
    prepared = {}
    for protocol in state['cfg']['protocols']:
        for target in state['targets']:
            key = fold_id(state, target, protocol)
            if key in prepared:
                continue
            path = state['out']/'sources'/(key+'.json')
            if path.exists():
                value = read(path)
                if value['signature'] != state['signature'] or value['source_ids'] != source_ids(
                        state['domain'], target, protocol):
                    raise RuntimeError('Source-prior identity mismatch.')
            else:
                value = prepare_source(state['calibration'], state['domain'], target,
                                       protocol, state['cfg']['gp'])
                value.update(signature=state['signature'], fold=key)
                save(path, value)
                print('Prepared '+key+f': {len(value["source_ids"])} sources, '
                      f'{value["source_observations"]} historical rollouts.', flush=True)
            prepared[key] = value
    return prepared


def jobs(state):
    noise = state['archive']['configuration']['calibration_seeds']
    return [dict(target=target, protocol=protocol, seed=seed,
                 noise_seed=noise[seed % len(noise)], fold=fold_id(state, target, protocol))
            for protocol in state['cfg']['protocols'] for target in state['targets']
            for seed in state['cfg']['optimizer_seeds']]


def checkpoint(state, job):
    return state['out']/'runs'/f'{job["protocol"]}__{job["target"]}__{job["seed"]}.json'


def init_worker(calibration, domain, source, cfg, signature):
    global WORKER_DATA
    # Deliberately exclude validation and dense target-reference tables.
    WORKER_DATA = (calibration, domain, source, cfg, signature)


def worker(job):
    calibration, domain, source, cfg, signature = WORKER_DATA
    start = time.perf_counter()
    methods = run_suite(calibration, domain, job['target'], source[job['fold']],
                        cfg, job['seed'], job['noise_seed'])
    return dict(**job, signature=signature, methods=methods,
                elapsed_seconds=time.perf_counter()-start,
                source_ids=source[job['fold']]['source_ids'])


def validate_job(state, expected, result):
    if result.get('signature') != state['signature'] or any(result.get(k) != v for k, v in expected.items()):
        raise RuntimeError('Replay checkpoint identity mismatch.')
    if result['source_ids'] != source_ids(state['domain'], expected['target'], expected['protocol']):
        raise RuntimeError('Target leaked into source history.')
    if set(result['methods']) != set(state['cfg']['methods']):
        raise RuntimeError('Missing method checkpoint.')
    for method, value in result['methods'].items():
        count = len(state['domain'].contexts) if method == 'catalogue' else max(state['cfg']['budgets'])
        if len(value['steps']) != count or value['actual_rollouts'] != count:
            raise RuntimeError('Incomplete rollout history.')
        oracle = TargetOracle(state['calibration'], expected['target'], expected['noise_seed'], state['domain'])
        observations = []
        for i, step in enumerate(value['steps']):
            if step['evaluation'] != i+1:
                raise RuntimeError('Invalid rollout accounting.')
            observation = oracle.observe(step['action'])
            for key, actual in observation.items():
                equal = (np.isclose(actual, step[key], rtol=1e-12, atol=1e-12)
                         if isinstance(actual, float) else actual == step[key])
                if not equal:
                    raise RuntimeError('Observation differs from the archived rollout.')
            if (step['context'], step['candidate']) != state['domain'].actions[step['action']]:
                raise RuntimeError('Action label mismatch.')
            observations.append(observation)
            if step['recommendations'] != incumbents(observations, state['domain']):
                raise RuntimeError('An incumbent was not selected from observed feasible trials.')


def load_jobs(state):
    results = []
    for job in jobs(state):
        path = checkpoint(state, job)
        if not path.exists():
            raise RuntimeError('Missing replay checkpoint: '+path.name)
        result = read(path); validate_job(state, job, result); results.append(result)
    return results


def preflight(state):
    manifest = initialize(state)
    source = priors(state)
    save(state['out']/'preflight.json', dict(expected_jobs=manifest['expected_jobs'], source_folds=len(source),
                                           methods=state['cfg']['methods'], budgets=state['cfg']['budgets'],
                                           archive_preserved=True))
    print(f'Preflight: {manifest["expected_jobs"]} replay jobs; budgets {state["cfg"]["budgets"]} '
          f'per vehicle across {len(state["domain"].contexts)} contexts. Output: {state["out"]}', flush=True)
    return source


def analyze(state):
    initialize(state)
    results = load_jobs(state)
    tables = dense_references(state['calibration'], state['validation'], state['domain'])
    if not all(r['eligible'] for r in tables[0].values()):
        raise RuntimeError('A dense calibration-selected reference fails validation; threshold undefined.')
    save(state['out']/'dense_reference.json', [dict(vehicle=v, context=c, **r)
                                              for (v, c), r in tables[0].items()])
    rows, cells = [], []
    for job in results:
        r, c = evaluate_job(job, state['calibration'], state['validation'], state['domain'],
                            state['cfg'], tables)
        rows.extend(r); cells.extend(c)
    frame, context = pd.DataFrame(rows), pd.DataFrame(cells)
    frame.to_csv(state['out']/'budget_trials.csv', index=False)
    context.to_csv(state['out']/'budget_contexts.csv', index=False)
    summary = write_report(state['out'], frame, context, state['cfg'], state['specification']['mode'])
    save(state['out']/'completion.json', dict(signature=state['signature'], jobs=len(results)))
    save(state['out']/'analysis_provenance.json', dict(
        report_source_sha256=digest(ROOT/'src/tracking_budget_report.py'), runtime_pandas=pd.__version__))
    print('Report: '+str(state['out']/'report.html'), flush=True)
    return summary


def run(state, workers):
    source = preflight(state)
    pending = []
    for job in jobs(state):
        path = checkpoint(state, job)
        if path.exists():
            validate_job(state, job, read(path))
        else:
            pending.append(job)
    print(f'{len(jobs(state))-len(pending)}/{len(jobs(state))} complete; {len(pending)} pending.', flush=True)
    args = (state['calibration'], state['domain'], source, state['cfg'], state['signature'])
    def record(result):
        validate_job(state, {k: result[k] for k in ('target', 'protocol', 'seed', 'noise_seed', 'fold')}, result)
        save(checkpoint(state, result), result)
        print(f'{result["protocol"]} / {result["target"]} / {result["seed"]}: '
              f'{result["elapsed_seconds"]:.1f}s', flush=True)
    if workers == 1:
        init_worker(*args)
        for job in pending:
            record(worker(job))
    elif pending:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=args) as executor:
            for future in as_completed([executor.submit(worker, job) for job in pending]):
                record(future.result())
    analyze(state)


def collect(state):
    if state['specification']['mode'] != 'full':
        raise RuntimeError('Smoke results cannot be collected.')
    analyze(state)
    relative = state['out'].relative_to(REPO).as_posix()
    subprocess.run(['git', 'add', '-f', '--', relative], cwd=REPO, check=True)
    print('Staged only '+relative)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight', 'smoke', 'run', 'status', 'analyze', 'collect'])
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--workers', type=int, default=1)
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
        analyze(state)
    elif args.command == 'collect':
        collect(state)
    else:
        initialize(state)
        found = sum(checkpoint(state, j).exists() for j in jobs(state))
        print(f'{found}/{len(jobs(state))} complete. Output: {state["out"]}')


if __name__ == '__main__':
    main()
