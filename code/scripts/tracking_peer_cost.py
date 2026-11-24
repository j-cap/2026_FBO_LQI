"""Offline peer-assisted cost-model comparison. See docs/tracking_peer_assisted_cost_model.md."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import time

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import tracking_online_lcb as base
from src.tracking_budget import dense_references
from src.tracking_peer_cost import METHODS, OBSERVATION_FIELDS, run_fleet, validate_state
from src.tracking_peer_cost_report import evaluate, eligibility_times, write_report

ROOT, REPO = base.ROOT, base.REPO
read, save, digest = base.read, base.save, base.digest
jobs, path_for, clients_for = base.jobs, base.path_for, base.clients_for
initialize, observe = base.initialize, base.observe
np, pd = base.np, base.pd
CONFIG = ROOT/'config/runs/tracking_peer_cost.json'
WORKER = None


def setup(path=CONFIG, smoke=False):
    cfg = read(path)
    if cfg['methods'] != METHODS:
        raise ValueError('The three prescribed peer-assisted methods are required.')
    if Path(cfg['tag']).name != cfg['tag'] or cfg['tag'] in ('.', '..', cfg['reference_tag']):
        raise ValueError('Use a new single-directory result tag.')
    reference = base.setup(ROOT/cfg['reference_config'])
    frozen = reference['cfg']
    fields = ('physics_config', 'evaluation_tag', 'conditions', 'seeds', 'queries_per_client',
        'fleet_budgets', 'homogeneous_plant_id', 'relative_tolerance', 'gp', 'model',
        'initialization_tests', 'acquisition')
    if any(cfg.get(k) != frozen[k] for k in fields) or cfg['reference_tag'] != frozen['tag']:
        raise ValueError('Physical, budget and BO/model settings must match the one-test pilot.')
    if cfg['smoke'] != {k: v for k, v in frozen['smoke'].items() if k != 'scout_search_queries'}:
        raise ValueError('Smoke must use the prescribed reduced client/query programme.')
    reference_manifest = read(reference['out']/'run_manifest.json')
    if (reference_manifest['signature'] != reference['signature'] or
        read(reference['out']/'completion.json') != dict(signature=reference['signature'], jobs=len(jobs(reference)))):
        raise RuntimeError('The completed gain-sharing reference has changed or is incomplete.')
    cfg = deepcopy(cfg)
    vehicles = reference['vehicles']
    if smoke:
        for key in ('seeds', 'queries_per_client', 'fleet_budgets'):
            cfg[key] = cfg['smoke'][key]
        vehicles = [v for v in vehicles if v['id'] in cfg['smoke']['client_plant_ids']]
    n = len(vehicles)
    if cfg['fleet_budgets'][0] != n or cfg['fleet_budgets'][-1] != n*cfg['queries_per_client']:
        raise ValueError('Snapshot budgets must span one factory test to the full local cap.')
    paths = [Path(__file__), ROOT/'src/tracking_peer_cost.py']
    source_hashes = dict(reference['specification']['source_hashes'])
    source_hashes.update({p.relative_to(REPO).as_posix(): digest(p) for p in paths})
    reference_paths = [reference['out']/name for name in ('run_manifest.json', 'completion.json')]
    reference_paths += [path_for(reference, job) for job in jobs(reference)]
    spec = dict(configuration=cfg, mode='smoke' if smoke else 'full',
        source_hashes=source_hashes, evaluation_hashes=reference['specification']['evaluation_hashes'],
        reference_hashes={p.relative_to(REPO).as_posix(): digest(p) for p in reference_paths},
        physical_vehicles=vehicles, historical_training_records=0,
        measurement_pairing=reference['specification']['measurement_pairing'],
        peer_rule='Prior-round eligible context incumbents; untried locally; donor-cost priority; mandatory local test.')
    signature = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    return dict(cfg=cfg, physics=reference['physics'], vehicles=vehicles,
        all_vehicles=reference['all_vehicles'], refs=reference['refs'], pool=reference['pool'],
        domain=reference['domain'], archive=reference['archive'], reference_out=reference['out'],
        specification=spec, signature=signature,
        out=ROOT/'results'/(cfg['tag']+('_smoke' if smoke else '')))


def validate_job(state, job, result, complete=False):
    if result.get('signature') != state['signature'] or any(result.get(k) != v for k, v in job.items()):
        raise RuntimeError('Peer-assisted checkpoint identity mismatch.')
    clients = clients_for(state, job['condition'])
    if result['clients'] != clients:
        raise RuntimeError('Physical fleet changed.')
    data = result['state']
    validate_state(data, state['domain'], state['cfg'], clients, job['seed'], complete)
    for method in data['methods'].values():
        for event in method['events']:
            key = f'{event["client"]}__{event["local_evaluation"]}__{event["context"]}__{event["candidate"]}'
            entry = data['oracle_cache'].get(key)
            if entry is None or any(entry[k] != event[k] for k in
                    ('noise_seed', 'client', 'local_evaluation', 'context')):
                raise RuntimeError('Missing or mismatched fresh simulator record.')
            raw = entry['result']
            if raw['candidate'] != event['candidate'] or any(event[k] != raw[k] for k in
                    ('complete', 'feasible', 'integration_resolved', 'substeps')):
                raise RuntimeError('Observation differs from its simulator record.')
            if event['eligible'] and not np.isclose(event['J'], raw['J'], rtol=1e-12, atol=1e-12):
                raise RuntimeError('Objective differs from its simulator record.')


def compare_reference(result, reference):
    """Evaluator-only check; the old reference never enters live worker payloads."""
    if result['clients'] != reference['clients']:
        raise RuntimeError('Reference control uses a different physical fleet.')
    current = result['state']['methods']['peer_gp_bo']
    previous = reference['state']['methods']['online_gain_sharing']
    if len(current['events']) != len(previous['events']):
        raise RuntimeError('Gain-sharing reference query count differs.')
    for a, b in zip(current['events'], previous['events']):
        for key in b:
            if key == 'method':
                continue
            equal = (np.isclose(a[key], b[key], rtol=1e-10, atol=1e-12)
                     if key == 'J' and b[key] is not None else a.get(key) == b[key])
            if not equal:
                raise RuntimeError(f'Gain-sharing reference differs at query {b["fleet_evaluation"]}: {key}')
    for budget, snapshot in previous['snapshots'].items():
        if any(current['snapshots'][budget].get(k) != v for k, v in snapshot.items()):
            raise RuntimeError('Gain-sharing reference budget snapshot differs.')
    return dict(condition=result['condition'], seed=result['seed'],
                queries=len(current['events']), matched=True)


def check_reference(state):
    """Replay A on its old queried outcomes, without any physical simulation."""
    if state['specification']['mode'] != 'full':
        raise RuntimeError('Reference replay requires the full nine-client configuration.')
    cfg = dict(state['cfg'], methods=['peer_gp_bo'])
    for job in jobs(state):
        previous = read(path_for(dict(out=state['reference_out']), job))
        old_events = previous['state']['methods']['online_gain_sharing']['events']
        def oracle(plans, cache):
            observations = []
            for plan in plans:
                old = old_events[(plan['local_evaluation']-1)*len(previous['clients'])+plan['client']]
                if any(plan[k] != old[k] for k in ('candidate', 'context', 'noise_seed', 'round')):
                    raise RuntimeError('Reference acquisition replay selected a different query.')
                observations.append({k: old[k] for k in OBSERVATION_FIELDS if k != 'action'})
            return observations
        data = run_fleet(state['domain'], cfg, previous['clients'], job['seed'], oracle)
        validate_state(data, state['domain'], cfg, previous['clients'], job['seed'], complete=True)
        compare_reference(dict(**job, clients=previous['clients'], state=data), previous)
        print(f'Reference replay matched {job["condition"]}/{job["seed"]}: 216 queries', flush=True)


def init_worker(payload):
    global WORKER
    WORKER = payload


def worker(job):
    state = WORKER
    clients = clients_for(state, job['condition'])
    path = path_for(state, job)
    previous = read(path) if path.exists() else None
    if previous:
        validate_job(state, job, previous)
    elapsed = previous.get('elapsed_seconds', 0.) if previous else 0.
    started = time.perf_counter()
    def checkpoint(data):
        save(path, dict(**job, signature=state['signature'], clients=clients, state=data,
            elapsed_seconds=elapsed+time.perf_counter()-started))
        if data['round'] % 3 == 0:
            print(f'{job["condition"]}/{job["seed"]}: round {data["round"]}/{state["cfg"]["queries_per_client"]}, '
                  f'{len(data["oracle_cache"])} unique simulations cached', flush=True)
    run_fleet(state['domain'], state['cfg'], clients, job['seed'],
        lambda plans, cache: observe(plans, cache, state, clients), checkpoint,
        previous['state'] if previous else None)
    result = read(path)
    validate_job(state, job, result, complete=True)


def preflight(state):
    initialize(state)
    planned = len(jobs(state))*len(METHODS)*len(state['vehicles'])*state['cfg']['queries_per_client']
    save(state['out']/'preflight.json', dict(jobs=len(jobs(state)), requested_online_rollouts=planned,
        historical_training_records=0, evaluation_archive_worker_access=False,
        reference_archive_worker_access=False, first_adaptive_round=2,
        common_peer_priority=True, fleet_budgets=state['cfg']['fleet_budgets']))
    print(f'Preflight: {len(jobs(state))} jobs, {planned:,} charged queries. Output: {state["out"]}', flush=True)


def analyze(state):
    initialize(state)
    results, control = [], []
    for job in jobs(state):
        path = path_for(state, job)
        if not path.exists():
            raise RuntimeError('Missing peer-assisted checkpoint: '+path.name)
        result = read(path)
        validate_job(state, job, result, complete=True)
        results.append(result)
        if state['specification']['mode'] == 'full':
            previous = read(path_for(dict(out=state['reference_out']), job))
            control.append(compare_reference(result, previous))
    raw = pd.read_csv(state['archive']/'controlled_evaluations.csv', float_precision='round_trip')
    tables = dense_references(raw[raw.role == 'calibration'], raw[raw.role == 'validation'], state['domain'])
    if not all(v['eligible'] for v in tables[0].values()):
        raise RuntimeError('An archived empirical reference fails independent validation.')
    trials, contexts, queries, learning = evaluate(results, state['domain'], state['cfg'], tables)
    times, context_times = eligibility_times(results, state['domain'], state['cfg'])
    for name, frame in [('budget_trials', trials), ('budget_contexts', contexts),
        ('query_observations', queries), ('learning_diagnostics', learning),
        ('eligibility_times', times), ('context_eligibility_times', context_times)]:
        frame.to_csv(state['out']/(name+'.csv'), index=False)
    write_report(state['out'], trials, contexts, queries, learning, times, state['cfg'], state['specification']['mode'])
    save(state['out']/'reference_control.json', dict(mode=state['specification']['mode'],
        checked=bool(control), jobs=control, smoke_note=None if control else 'Reduced fleet has no full-pilot counterpart.'))
    save(state['out']/'analysis_provenance.json', dict(source_hashes={p: digest(ROOT/p) for p in
        ('src/tracking_peer_cost_report.py', 'src/tracking_online_lcb_report.py')}))
    save(state['out']/'completion.json', dict(signature=state['signature'], jobs=len(results)))
    print('Report: '+str(state['out']/'report.html'), flush=True)


def run(state, workers):
    preflight(state)
    pending = []
    for job in jobs(state):
        path = path_for(state, job)
        if path.exists():
            result = read(path); validate_job(state, job, result)
            if result['state']['round'] == state['cfg']['queries_per_client']:
                continue
        pending.append(job)
    print(f'{len(jobs(state))-len(pending)}/{len(jobs(state))} complete; {len(pending)} pending.', flush=True)
    # Exclude dense tables, reference checkpoints, evaluation/archive paths and manifests.
    payload = {k: state[k] for k in ('cfg', 'physics', 'vehicles', 'all_vehicles', 'refs', 'pool', 'domain', 'out', 'signature')}
    if workers == 1:
        init_worker(payload)
        for job in pending:
            worker(job)
    elif pending:
        with ProcessPoolExecutor(max_workers=min(workers, len(pending)), mp_context=multiprocessing.get_context('spawn'),
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
    parser.add_argument('command', choices=['preflight', 'smoke', 'run', 'status', 'analyze', 'collect', 'check-reference'])
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--workers', type=int, default=min(6, os.cpu_count() or 1))
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
    elif args.command == 'check-reference':
        check_reference(state)
    else:
        for job in jobs(state):
            path = path_for(state, job)
            rounds = read(path)['state']['round'] if path.exists() else 0
            print(f'{job["condition"]}/{job["seed"]}: {rounds}/{state["cfg"]["queries_per_client"]} rounds')


if __name__ == '__main__':
    main()
