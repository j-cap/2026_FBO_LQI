"""Historical cost level versus gain dependence. Protocol: docs/tracking_cost_shape.md."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import tracking_budget as base
import tracking_information as previous_runner
from src.tracking_budget import dense_references, evaluate_job
from src.tracking_cost_shape import (ARMS, METHODS, PRIOR_DEFINITION,
                                     control_comparison, run_suite, source_for_arm)
from src.tracking_cost_shape_report import write_report

CONFIG = ROOT/'config/runs/tracking_cost_shape.json'
WORKER_DATA = None


def setup(config_path=CONFIG, smoke=False):
    raw = base.read(config_path)
    if Path(raw['reference_tag']).name != raw['reference_tag']:
        raise ValueError('Reference tag must be a single directory name.')
    reference = ROOT/'results'/raw['reference_tag']
    manifest = base.read(reference/'run_manifest.json')
    completion = base.read(reference/'completion.json')
    if completion['signature'] != manifest['signature'] or completion['jobs'] != manifest['expected_jobs']:
        raise RuntimeError('The reference information-ablation campaign is incomplete.')
    for field in ('gp', 'budgets', 'protocols', 'optimizer_seeds', 'relative_tolerance', 'tie_tolerance'):
        if raw[field] != manifest['configuration'][field]:
            raise ValueError('Only cost-prior form may change; mismatch in '+field)
    if raw['methods'] != METHODS:
        raise ValueError('The fixed catalogue and three prescribed cost-prior arms are required.')
    if raw.get('prior_definition') != PRIOR_DEFINITION:
        raise ValueError('The context-level prior definition is fixed.')
    for path, expected in manifest['source_hashes'].items():
        if base.digest(REPO/path) != expected:
            raise RuntimeError('Frozen optimizer source changed: '+path)
    state = base.setup(config_path, smoke)
    for name, expected in manifest['input_hashes'].items():
        if base.digest(state['original']/name) != expected:
            raise RuntimeError('Diagnostic archive differs from the reference information-ablation campaign.')
    if manifest['input_signature'] != state['archive']['signature']:
        raise RuntimeError('Different original experiment signature.')
    state['reference'] = reference
    state['reference_manifest'] = manifest
    state['reference_state'] = dict(state, cfg=manifest['configuration'],
                                   signature=manifest['signature'], out=reference)
    fingerprints = {name: base.digest(reference/name) for name in ('run_manifest.json', 'completion.json')}
    for job in base.jobs(state):
        path = base.checkpoint(state['reference_state'], job)
        if not path.exists():
            raise RuntimeError('Missing reference replay: '+path.name)
        fingerprints[path.relative_to(reference).as_posix()] = base.digest(path)
    for path in sorted((reference/'sources').glob('*.json')):
        fingerprints[path.relative_to(reference).as_posix()] = base.digest(path)
    for path in (Path(__file__), ROOT/'src/tracking_cost_shape.py',
                 ROOT/'scripts/tracking_information.py', ROOT/'src/tracking_information.py'):
        state['specification']['source_hashes'][path.relative_to(REPO).as_posix()] = base.digest(path)
    state['specification']['reference_input_hashes'] = fingerprints
    state['specification']['prior_modes'] = ARMS
    state['specification']['prior_definition'] = PRIOR_DEFINITION
    state['specification']['historical_feasibility_prior'] = False
    state['signature'] = hashlib.sha256(json.dumps(state['specification'], sort_keys=True).encode()).hexdigest()
    return state


frozen_sources = previous_runner.frozen_sources


def validate_job(state, expected, result):
    base.validate_job(state, expected, result)
    if result['methods']['catalogue'].get('prior_mode', 'missing') is not None:
        raise RuntimeError('The fixed catalogue has no BO prior mode.')
    initial = result['methods']['catalogue']['steps']
    for arm, mode in ARMS.items():
        value = result['methods'][arm]
        if value.get('prior_mode') != mode or value.get('historical_feasibility_prior') is not False:
            raise RuntimeError('Incorrect historical prior labels.')
        if value['steps'][:len(initial)] != initial:
            raise RuntimeError('The six initial target tests differ across arms.')


def load_jobs(state):
    results = []
    for job in base.jobs(state):
        path = base.checkpoint(state, job)
        if not path.exists():
            raise RuntimeError('Missing cost-prior checkpoint: '+path.name)
        result = base.read(path); validate_job(state, job, result); results.append(result)
    return results


def init_worker(calibration, domain, source, cfg, signature):
    global WORKER_DATA
    # No validation observations or dense target-reference costs in worker data.
    WORKER_DATA = calibration, domain, source, cfg, signature


def worker(job):
    calibration, domain, source, cfg, signature = WORKER_DATA
    started = time.perf_counter()
    methods = run_suite(calibration, domain, job['target'], source[job['fold']],
                        cfg, job['seed'], job['noise_seed'])
    return dict(**job, signature=signature, methods=methods,
                source_ids=source[job['fold']]['source_ids'],
                elapsed_seconds=time.perf_counter()-started)


def preflight(state):
    manifest = base.initialize(state)
    source = frozen_sources(state)
    audit = []
    for fold, value in source.items():
        full = np.asarray(value['prior_mean'])
        level = np.asarray(source_for_arm(value, state['domain'], 'context_level')['prior_mean'])
        for context in state['domain'].contexts:
            indices = [i for i, (c, _) in enumerate(state['domain'].actions) if c == context]
            if not np.all(level[indices] == level[indices[0]]) or not np.isclose(
                    level[indices[0]], full[indices].mean(), rtol=0., atol=1e-14):
                raise RuntimeError('Context-level prior did not preserve the uniform log-cost mean.')
            audit.append(dict(fold=fold, context=context, candidates=len(indices),
                              mean_log_cost=float(full[indices].mean()),
                              full_prior_std_log_cost=float(full[indices].std()),
                              full_prior_min_log_cost=float(full[indices].min()),
                              full_prior_max_log_cost=float(full[indices].max())))
    pd.DataFrame(audit).to_csv(state['out']/'source_level_audit.csv', index=False)
    for job in base.jobs(state):
        previous = base.read(base.checkpoint(state['reference_state'], job))
        previous_runner.validate_job(state['reference_state'], job, previous)
    base.save(state['out']/'preflight.json', dict(expected_jobs=manifest['expected_jobs'],
        source_folds=len(source), methods=state['cfg']['methods'], budgets=state['cfg']['budgets'],
        source_priors='Frozen arrays from the completed information ablation; no refitting.',
        identical_catalogue_initialization=True, previous_results_unchanged=True))
    print(f'Preflight: {manifest["expected_jobs"]} jobs; no / context-level / full cost prior; '
          f'budgets {state["cfg"]["budgets"]}. Output: {state["out"]}', flush=True)
    return source


def analyze(state):
    frozen_sources(state)
    results = load_jobs(state)
    tables = dense_references(state['calibration'], state['validation'], state['domain'])
    if not all(v['eligible'] for v in tables[0].values()):
        raise RuntimeError('A dense calibration-selected reference is ineligible on validation.')
    base.save(state['out']/'dense_reference.json', [dict(vehicle=v, context=c, **r)
                                                   for (v, c), r in tables[0].items()])
    rows, cells, controls, queries = [], [], [], []
    for result in results:
        r, c = evaluate_job(result, state['calibration'], state['validation'], state['domain'],
                            state['cfg'], tables)
        rows.extend(r); cells.extend(c)
        previous = base.read(base.checkpoint(state['reference_state'], result))
        controls.extend(control_comparison(result, previous))
        for method, trace in result['methods'].items():
            for step in trace['steps']:
                queries.append(dict(target=result['target'], protocol=result['protocol'], seed=result['seed'],
                                    method=method, evaluation=step['evaluation'], context=step['context'],
                                    candidate=step['candidate'], eligible=step['eligible'], J=step['J'],
                                    complete=step['complete'], feasible=step['feasible'],
                                    integration_resolved=step['integration_resolved']))
    frame, context = pd.DataFrame(rows), pd.DataFrame(cells)
    frame.to_csv(state['out']/'budget_trials.csv', index=False)
    context.to_csv(state['out']/'budget_contexts.csv', index=False)
    pd.DataFrame(controls).to_csv(state['out']/'previous_control_audit.csv', index=False)
    pd.DataFrame(queries).to_csv(state['out']/'query_observations.csv', index=False)
    summary = write_report(state['out'], frame, context, pd.DataFrame(controls),
                           state['cfg'], state['specification']['mode'])
    base.save(state['out']/'completion.json', dict(signature=state['signature'], jobs=len(results)))
    base.save(state['out']/'analysis_provenance.json', dict(
        report_source_sha256=base.digest(ROOT/'src/tracking_cost_shape_report.py'),
        summary_source_sha256=base.digest(ROOT/'src/tracking_budget_report.py')))
    print('Report: '+str(state['out']/'report.html'), flush=True)
    return summary


def run(state, workers):
    source = preflight(state)
    pending = []
    for job in base.jobs(state):
        path = base.checkpoint(state, job)
        if path.exists():
            validate_job(state, job, base.read(path))
        else:
            pending.append(job)
    print(f'{len(base.jobs(state))-len(pending)}/{len(base.jobs(state))} complete; {len(pending)} pending.', flush=True)
    args = state['calibration'], state['domain'], source, state['cfg'], state['signature']
    def record(expected, result):
        validate_job(state, expected, result)
        base.save(base.checkpoint(state, expected), result)
        print(f'{result["protocol"]} / {result["target"]} / {result["seed"]}: '
              f'{result["elapsed_seconds"]:.1f}s', flush=True)
    if workers == 1:
        init_worker(*args)
        for job in pending:
            record(job, worker(job))
    elif pending:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=args) as executor:
            futures = {executor.submit(worker, job): job for job in pending}
            for future in as_completed(futures):
                record(futures[future], future.result())
    analyze(state)


def collect(state):
    if state['specification']['mode'] != 'full':
        raise RuntimeError('Smoke results cannot be collected.')
    analyze(state)
    relative = state['out'].relative_to(REPO).as_posix()
    base.subprocess.run(['git', 'add', '-f', '--', relative], cwd=REPO, check=True)
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
    if args.command in ('smoke', 'run'):
        run(state, args.workers)
    elif args.command == 'preflight':
        preflight(state)
    elif args.command == 'analyze':
        analyze(state)
    elif args.command == 'collect':
        collect(state)
    else:
        base.initialize(state)
        found = sum(base.checkpoint(state, job).exists() for job in base.jobs(state))
        print(f'{found}/{len(base.jobs(state))} complete. Output: {state["out"]}')


if __name__ == '__main__':
    main()
