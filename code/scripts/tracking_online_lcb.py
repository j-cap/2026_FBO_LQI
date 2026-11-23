"""One-test online contextual calibration. Protocol: docs/tracking_online_lcb.md."""
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
from src.tracking_budget import dense_references, domain_from_manifest
from src.tracking_online_lcb import METHODS, run_fleet, seed_for, validate_state
from src.tracking_task import candidates, fleet, reference_metadata, references, resolved_rollout
from src.tracking_online_lcb_report import eligibility_times, evaluate, write_report

CONFIG = ROOT/'config/runs/tracking_online_lcb.json'
WORKER = None


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def setup(path=CONFIG, smoke=False):
    cfg = read(path)
    for field in ('tag', 'evaluation_tag'):
        if Path(cfg[field]).name != cfg[field] or cfg[field] in ('.', '..'):
            raise ValueError('Tags must be single directory names.')
    if cfg['methods'] != METHODS or cfg['conditions'] != ['homogeneous', 'heterogeneous']:
        raise ValueError('The prescribed six methods and two conditions are required.')
    if len(set(cfg['seeds'])) != len(cfg['seeds']):
        raise ValueError('Duplicate experiment seeds.')
    pc = ROOT/cfg['physics_config']
    physics = read(pc)
    archive = ROOT/'results'/cfg['evaluation_tag']
    old = read(archive/'run_manifest.json')
    if read(archive/'completion.json')['signature'] != old['signature'] or old['configuration'] != physics:
        raise RuntimeError('The frozen physics configuration or evaluation archive differs.')
    for name, expected in old['source_hashes'].items():
        if digest(REPO/name) != expected:
            raise RuntimeError('Frozen tracking source changed: '+name)
    vehicles, refs, pool = fleet(physics), references(physics), candidates(physics)
    all_vehicles = deepcopy(vehicles)
    if smoke:
        cfg = deepcopy(cfg)
        for field in ('seeds', 'queries_per_client', 'fleet_budgets', 'scout_search_queries'):
            cfg[field] = cfg['smoke'][field]
        vehicles = [v for v in vehicles if v['id'] in cfg['smoke']['client_plant_ids']]
    n, nc = len(vehicles), len(refs)
    if n < 2 or cfg['queries_per_client'] < nc or cfg['scout_search_queries'] < nc:
        raise ValueError('Need at least two clients and one scheduled context cycle per client/scout.')
    if cfg['scout_search_queries'] > cfg['queries_per_client']:
        raise ValueError('Scout search exceeds its local cap.')
    budgets = cfg['fleet_budgets']
    if budgets != sorted(set(budgets)) or budgets[-1] != n*cfg['queries_per_client'] or budgets[0] != n:
        raise ValueError('Fleet budgets must increase from one factory test per client to the full cap.')
    if cfg['queries_per_client'] > len(pool)*nc:
        raise ValueError('Local budget exceeds the finite action pool.')
    if cfg.get('initialization_tests') != 1 or cfg.get('acquisition') != dict(
            confidence_multiplier=2., feasibility_penalty=1., probability_floor=1e-6):
        raise ValueError('This protocol fixes one factory test and LCB settings (2, 1, 1e-6).')
    m = cfg['model']
    if min(m['features'], m['ridge'], m['local_steps'], m['server_rounds'], m['learning_rate']) <= 0:
        raise ValueError('Positive regression/FedAvg settings are required.')
    if m['learning_rate'] >= 1/(3+m['ridge']):
        raise ValueError('Learning rate violates the conservative fixed-feature stability bound.')
    metadata = dict(configuration=physics, references=[reference_metadata(r, physics['dt']) for r in refs],
                    candidates=pool, vehicles=all_vehicles)
    domain = domain_from_manifest(metadata)
    paths = [Path(__file__), ROOT/'src/tracking_online_lcb.py', ROOT/'src/tracking_online.py', ROOT/'src/tracking_budget.py',
             ROOT/'src/tracking_context.py', ROOT/'src/tracking_task.py', pc]
    paths += sorted((ROOT/'src/tracking_reference').glob('*.py'))
    paths += [ROOT/'src/tracking_reference/provenance.json']
    spec = dict(configuration=cfg, mode='smoke' if smoke else 'full',
        source_hashes={p.relative_to(REPO).as_posix(): digest(p) for p in paths},
        evaluation_hashes={p: digest(archive/p) for p in ('run_manifest.json', 'completion.json',
                                                        'controlled_evaluations.csv')},
        physical_vehicles=vehicles, historical_training_records=0,
        measurement_pairing='Independent client/query seeds; identical counterfactual noise across methods.')
    signature = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    return dict(cfg=cfg, physics=physics, vehicles=vehicles, all_vehicles=all_vehicles,
        refs=refs, pool=pool, domain=domain, archive=archive, specification=spec, signature=signature,
        out=ROOT/'results'/(cfg['tag']+('_smoke' if smoke else '')))


def jobs(state):
    return [dict(condition=c, seed=s) for c in state['cfg']['conditions'] for s in state['cfg']['seeds']]


def path_for(state, job):
    return state['out']/'runs'/f'{job["condition"]}__{job["seed"]}.json'


def clients_for(state, condition):
    common = next(v for v in state['all_vehicles'] if v['id'] == state['cfg']['homogeneous_plant_id'])
    return [dict(id=f'client_{i:02d}', plant_id=(v if condition == 'heterogeneous' else common)['id'],
                 family=(v if condition == 'heterogeneous' else common)['family'],
                 params=deepcopy((v if condition == 'heterogeneous' else common)['params']))
            for i, v in enumerate(state['vehicles'])]


def scout_for(state, seed):
    selected = state['cfg']['scout_plant_ids'][state['cfg']['seeds'].index(seed) % len(state['cfg']['scout_plant_ids'])]
    return [v['id'] for v in state['vehicles']].index(selected)


def initialize(state):
    path = state['out']/'run_manifest.json'
    if path.exists():
        if read(path)['signature'] != state['signature']:
            raise RuntimeError('Online code, configuration or archive changed; use a new result tag.')
        return
    manifest = dict(signature=state['signature'], **state['specification'], expected_jobs=len(jobs(state)),
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__),
        contexts=[reference_metadata(r, state['physics']['dt']) for r in state['refs']], candidates=state['pool'])
    save(path, manifest)


def observe(plans, cache, state, clients):
    """Batch only requested counterfactual gains with the same client/episode noise."""
    groups = {}
    for plan in plans:
        key = (plan['client'], plan['local_evaluation'], plan['context'], plan['noise_seed'])
        groups.setdefault(key, []).append(plan)
    lookup = {c['id']: c for c in state['pool']}
    refs = {r['id']: r for r in state['refs']}
    for (i, local, context, noise), requests in groups.items():
        missing = {}
        for plan in requests:
            key = f'{i}__{local}__{context}__{plan["candidate"]}'
            if key not in cache:
                missing[plan['candidate']] = key
        if missing:
            pool = [lookup[c] for c in sorted(missing)]
            results, _ = resolved_rollout(clients[i]['params'], refs[context], pool, state['physics'], noise)
            for result in results:
                cache[missing[result['candidate']]] = dict(noise_seed=noise, client=i,
                    local_evaluation=local, context=context, result=result)
    observations = []
    for plan in plans:
        key = f'{plan["client"]}__{plan["local_evaluation"]}__{plan["context"]}__{plan["candidate"]}'
        raw = cache[key]['result']
        eligible = bool(raw['complete'] and raw['feasible'] and raw['integration_resolved'])
        observations.append(dict(eligible=eligible, J=raw['J'] if eligible else None,
            complete=raw['complete'], feasible=raw['feasible'], integration_resolved=raw['integration_resolved'],
            substeps=raw['substeps']))
    return observations


def validate_job(state, job, result, complete=False):
    if result.get('signature') != state['signature'] or any(result.get(k) != v for k, v in job.items()):
        raise RuntimeError('Online checkpoint identity mismatch.')
    clients = clients_for(state, job['condition'])
    scout = scout_for(state, job['seed'])
    if result['clients'] != clients or result['scout'] != scout:
        raise RuntimeError('Physical fleet or scout assignment changed.')
    data = result['state']
    validate_state(data, state['domain'], state['cfg'], clients, job['seed'], scout, complete)
    for method in data['methods'].values():
        for event in method['events']:
            key = f'{event["client"]}__{event["local_evaluation"]}__{event["context"]}__{event["candidate"]}'
            entry = data['oracle_cache'].get(key)
            if entry is None or entry['noise_seed'] != event['noise_seed']:
                raise RuntimeError('Missing or mismatched fresh-rollout record.')
            raw = entry['result']
            for field in ('complete', 'feasible', 'integration_resolved', 'substeps'):
                if event[field] != raw[field]:
                    raise RuntimeError('Observation differs from the simulator record.')
            if event['eligible'] and not np.isclose(event['J'], raw['J'], rtol=1e-12, atol=1e-12):
                raise RuntimeError('Objective differs from the simulator record.')


def init_worker(payload):
    global WORKER
    # Deliberately exclude archive paths, validation observations and dense references.
    WORKER = payload


def worker(job):
    state = WORKER
    clients = clients_for(state, job['condition'])
    scout = scout_for(state, job['seed'])
    path = path_for(state, job)
    previous = read(path) if path.exists() else None
    if previous:
        validate_job(state, job, previous)
    elapsed = previous.get('elapsed_seconds', 0.) if previous else 0.
    started = time.perf_counter()
    def checkpoint(data):
        value = dict(**job, signature=state['signature'], clients=clients, scout=scout,
                     state=data, elapsed_seconds=elapsed+time.perf_counter()-started)
        save(path, value)
        if data['round'] % 3 == 0:
            count = len(data['methods']['federated_cost_bo']['events'])
            print(f'{job["condition"]}/{job["seed"]}: round {data["round"]}, '
                  f'federated {count}/{len(clients)*state["cfg"]["queries_per_client"]} tests', flush=True)
    data = run_fleet(state['domain'], state['cfg'], clients, job['seed'], scout,
        lambda plans, cache: observe(plans, cache, state, clients), checkpoint,
        previous['state'] if previous else None)
    result = read(path)
    validate_job(state, job, result, complete=True)
    return dict(**job, elapsed_seconds=result['elapsed_seconds'])


def preflight(state):
    initialize(state)
    planned = len(jobs(state))*len(state['cfg']['methods'])*len(state['vehicles'])*state['cfg']['queries_per_client']
    save(state['out']/'preflight.json', dict(jobs=len(jobs(state)), requested_online_rollouts=planned,
        historical_training_records=0, evaluation_archive_worker_access=False,
        factory_tests_per_concurrent_client=state['cfg']['initialization_tests'],
        first_adaptive_round=2, acquisition=state['cfg']['acquisition'],
        independent_client_noise=True, fleet_budgets=state['cfg']['fleet_budgets'],
        scout_assignments={str(s): scout_for(state, s) for s in state['cfg']['seeds']}))
    print(f'Preflight: {len(jobs(state))} jobs; {planned:,} charged online rollouts; '
          f'fleet budgets {state["cfg"]["fleet_budgets"]}. Output: {state["out"]}', flush=True)


def analyze(state):
    initialize(state)
    results = []
    for job in jobs(state):
        path = path_for(state, job)
        if not path.exists():
            raise RuntimeError('Missing online checkpoint: '+path.name)
        result = read(path); validate_job(state, job, result, complete=True); results.append(result)
    # Dense costs become available to the evaluator only after all query traces are complete.
    raw = pd.read_csv(state['archive']/'controlled_evaluations.csv', float_precision='round_trip')
    tables = dense_references(raw[raw.role == 'calibration'], raw[raw.role == 'validation'], state['domain'])
    if not all(v['eligible'] for v in tables[0].values()):
        raise RuntimeError('An archived empirical reference fails validation.')
    trials, contexts, queries, learning = evaluate(results, state['domain'], state['cfg'], tables)
    times, context_times = eligibility_times(results, state['domain'], state['cfg'])
    for name, frame in [('budget_trials', trials), ('budget_contexts', contexts),
                        ('query_observations', queries), ('learning_diagnostics', learning),
                        ('eligibility_times', times), ('context_eligibility_times', context_times)]:
        frame.to_csv(state['out']/(name+'.csv'), index=False)
    write_report(state['out'], trials, contexts, queries, learning, state['cfg'], state['specification']['mode'], times)
    save(state['out']/'analysis_provenance.json', dict(report_sha256=digest(ROOT/'src/tracking_online_lcb_report.py')))
    save(state['out']/'completion.json', dict(signature=state['signature'], jobs=len(results)))
    print('Report: '+str(state['out']/'report.html'), flush=True)


def run(state, workers):
    preflight(state)
    pending = []
    for job in jobs(state):
        path = path_for(state, job)
        if path.exists():
            value = read(path); validate_job(state, job, value)
            if all(len(m['events']) == len(state['vehicles'])*state['cfg']['queries_per_client']
                   for m in value['state']['methods'].values()):
                continue
        pending.append(job)
    print(f'{len(jobs(state))-len(pending)}/{len(jobs(state))} complete; {len(pending)} pending.', flush=True)
    payload = {k: state[k] for k in ('cfg', 'physics', 'vehicles', 'all_vehicles', 'refs', 'pool', 'domain', 'out', 'signature')}
    if workers == 1:
        init_worker(payload)
        for job in pending:
            worker(job)
    elif pending:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=init_worker, initargs=(payload,)) as executor:
            for future in as_completed([executor.submit(worker, job) for job in pending]):
                future.result()
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
    if args.command in ('smoke', 'run'):
        run(state, args.workers)
    elif args.command == 'preflight':
        preflight(state)
    elif args.command == 'analyze':
        analyze(state)
    elif args.command == 'collect':
        collect(state)
    else:
        initialize(state)
        counts = [read(path_for(state, job))['state']['round'] if path_for(state, job).exists() else 0
                  for job in jobs(state)]
        for job, rounds in zip(jobs(state), counts):
            print(f'{job["condition"]}/{job["seed"]}: {rounds} completed rounds')


if __name__ == '__main__':
    main()
