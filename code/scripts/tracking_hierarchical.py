"""Joint shared/local online BO. Protocol: docs/tracking_hierarchical.md."""
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

import tracking_peer_cost as reference_runner
from src.tracking_budget import dense_references
from src.tracking_hierarchical import METHODS, cost_posterior, replay_new_methods, run_fleet, validate_state
from src.tracking_online import features
from src.tracking_peer_cost import METHODS as BASELINES
from src.tracking_peer_cost_report import evaluate, eligibility_times
from src.tracking_hierarchical_report import write_report

ROOT, REPO = reference_runner.ROOT, reference_runner.REPO
read, save, digest = reference_runner.read, reference_runner.save, reference_runner.digest
jobs, path_for, clients_for = reference_runner.jobs, reference_runner.path_for, reference_runner.clients_for
initialize, observe = reference_runner.initialize, reference_runner.observe
np, pd = reference_runner.np, reference_runner.pd
CONFIG = ROOT/'config/runs/tracking_hierarchical.json'
WORKER = None


def setup(path=CONFIG, smoke=False):
    requested = read(path)
    if requested['methods'] != METHODS:
        raise ValueError('Both the shared/local Bayesian model and its local control are required.')
    tag = requested['tag']
    if Path(tag).name != tag or tag in ('.', '..', requested['reference_tag']):
        raise ValueError('Use a new single-directory result tag.')
    reference = reference_runner.setup(ROOT/requested['reference_config'])
    if requested['reference_tag'] != reference['cfg']['tag']:
        raise ValueError('The matched reference tag differs from its configuration.')
    manifest = read(reference['out']/'run_manifest.json')
    if (manifest['signature'] != reference['signature'] or
        read(reference['out']/'completion.json') != dict(signature=reference['signature'], jobs=len(jobs(reference)))):
        raise RuntimeError('The completed A/B/C reference changed or is incomplete.')
    cfg = deepcopy(reference['cfg'])
    cfg.update(tag=tag, reference_config=requested['reference_config'], reference_tag=requested['reference_tag'],
               methods=METHODS.copy(), hierarchical=deepcopy(requested['hierarchical']),
               comparison_methods=BASELINES+METHODS)
    fractions = cfg['hierarchical']['sharing_fractions']
    if not fractions or fractions != sorted(set(fractions)) or fractions[0] != 0 or fractions[-1] != 1:
        raise ValueError('The sharing prior must include ordered distinct endpoints 0 and 1.')
    cost_posterior([[]], features(reference['domain'], cfg['model']), cfg['gp'], cfg['hierarchical'])
    vehicles = reference['vehicles']
    if smoke:
        for key in ('seeds', 'queries_per_client', 'fleet_budgets'):
            cfg[key] = cfg['smoke'][key]
        vehicles = [v for v in vehicles if v['id'] in cfg['smoke']['client_plant_ids']]
        # Smoke exercises all report paths; its reduced fleet cannot reuse full traces.
        cfg['methods'] = BASELINES+METHODS
    paths = [Path(__file__), ROOT/'src/tracking_hierarchical.py', ROOT/'src/tracking_hierarchical_report.py',
             ROOT/'src/tracking_peer_cost_report.py', ROOT/'src/tracking_online_lcb_report.py']
    source_hashes = dict(reference['specification']['source_hashes'])
    source_hashes.update({p.relative_to(REPO).as_posix(): digest(p) for p in paths})
    reference_paths = [reference['out']/name for name in ('run_manifest.json', 'completion.json')]
    reference_paths += [path_for(reference, job) for job in jobs(reference)]
    spec = dict(configuration=cfg, mode='smoke' if smoke else 'full', source_hashes=source_hashes,
        evaluation_hashes=reference['specification']['evaluation_hashes'],
        reference_hashes={p.relative_to(REPO).as_posix(): digest(p) for p in reference_paths},
        physical_vehicles=vehicles, historical_training_records=0,
        measurement_pairing=reference['specification']['measurement_pairing'],
        peer_rule=reference['specification']['peer_rule'],
        primary_contrast=METHODS[0]+' minus '+BASELINES[0]+' at final budget',
        secondary_contrast=METHODS[0]+' minus '+BASELINES[1]+' at final budget',
        model_sharing_control=METHODS[0]+' minus '+METHODS[1],
        inference='Exact client information-message aggregation; Bayesian mixture over sharing strength; not FedAvg.',
        prior='Same public 65 features, GP initial amplitude/constant and noise; fixed lengths; uniform sharing prior.',
        evaluation_only_baselines=[] if smoke else BASELINES,
        worker_reference_access=False, worker_evaluation_access=False)
    signature = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    return dict(cfg=cfg, physics=reference['physics'], vehicles=vehicles, all_vehicles=reference['all_vehicles'],
        refs=reference['refs'], pool=reference['pool'], domain=reference['domain'], archive=reference['archive'],
        reference_out=reference['out'], reference_state=reference,
        specification=spec, signature=signature, out=ROOT/'results'/(tag+('_smoke' if smoke else '')))


def validate_job(state, job, result, complete=False, replay=False):
    if result.get('signature') != state['signature'] or any(result.get(k) != v for k, v in job.items()):
        raise RuntimeError('Joint-cost checkpoint identity mismatch.')
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
    if replay:
        replay_new_methods(data, state['domain'], state['cfg'], clients, job['seed'])


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
    validate_job(state, job, read(path), complete=True)


def preflight(state):
    initialize(state)
    per_method = len(jobs(state))*len(state['vehicles'])*state['cfg']['queries_per_client']
    planned = per_method*len(state['cfg']['methods'])
    save(state['out']/'preflight.json', dict(jobs=len(jobs(state)), requested_new_rollouts=planned,
        reused_baseline_rollout_records=0 if state['specification']['mode'] == 'smoke' else per_method*len(BASELINES),
        historical_training_records=0, worker_reference_access=False, worker_evaluation_access=False,
        first_adaptive_round=2, common_peer_priority=True, fleet_budgets=state['cfg']['fleet_budgets'],
        primary_baseline='A', secondary_baseline='B', statistical_sharing_control='E'))
    print(f'Preflight: {len(jobs(state))} jobs; {planned:,} newly charged requests. '
          f'Output: {state["out"]}', flush=True)


def comparison_result(state, job, result):
    if state['specification']['mode'] == 'smoke':
        return result
    old = read(path_for(dict(out=state['reference_out']), job))
    reference_runner.validate_job(state['reference_state'], job, old, complete=True)
    if result['clients'] != old['clients']:
        raise RuntimeError('Reused baselines have a different fleet.')
    # Confirm identical factory observations across historical and new arms.
    n = len(result['clients'])
    initial = old['state']['methods'][BASELINES[0]]['events'][:n]
    for data in result['state']['methods'].values():
        for a, b in zip(data['events'][:n], initial):
            for key in ('action', 'context', 'candidate', 'noise_seed', 'eligible', 'J'):
                equal = np.isclose(a[key], b[key], rtol=1e-10, atol=1e-12) if key == 'J' and a[key] is not None else a[key] == b[key]
                if not equal:
                    raise RuntimeError('Reused baseline factory test differs: '+key)
    combined = dict(result, state=dict(result['state'], methods={**old['state']['methods'], **result['state']['methods']}))
    return combined


def analyze(state):
    initialize(state)
    results = []
    for job in jobs(state):
        path = path_for(state, job)
        if not path.exists():
            raise RuntimeError('Missing joint-cost checkpoint: '+path.name)
        result = read(path)
        validate_job(state, job, result, complete=True, replay=True)
        results.append(comparison_result(state, job, result))
    # Open evaluator tables only after all live traces have passed replay.
    raw = pd.read_csv(state['archive']/'controlled_evaluations.csv', float_precision='round_trip')
    tables = dense_references(raw[raw.role == 'calibration'], raw[raw.role == 'validation'], state['domain'])
    if not all(v['eligible'] for v in tables[0].values()):
        raise RuntimeError('An empirical reference fails independent validation.')
    trials, contexts, queries, learning = evaluate(results, state['domain'], state['cfg'], tables)
    times, context_times = eligibility_times(results, state['domain'], state['cfg'])
    for name, frame in [('budget_trials', trials), ('budget_contexts', contexts),
        ('query_observations', queries), ('learning_diagnostics', learning),
        ('eligibility_times', times), ('context_eligibility_times', context_times)]:
        frame.to_csv(state['out']/(name+'.csv'), index=False)
    write_report(state['out'], trials, contexts, queries, learning, times, state['cfg'], state['specification']['mode'])
    save(state['out']/'reference_control.json', dict(reused_for_evaluation_only=state['specification']['mode'] == 'full',
        methods=BASELINES, exact_fleet_and_factory_pairing=True, worker_reference_access=False))
    save(state['out']/'analysis_provenance.json', dict(all_new_queries_replayed=True,
        source_hashes={p: digest(ROOT/p) for p in ('src/tracking_hierarchical_report.py',
            'src/tracking_peer_cost_report.py', 'src/tracking_online_lcb_report.py')}))
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
        raise RuntimeError('Smoke results cannot be collected as experimental evidence.')
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
        reference_runner.check_reference(state['reference_state'])
    else:
        for job in jobs(state):
            path = path_for(state, job)
            rounds = read(path)['state']['round'] if path.exists() else 0
            print(f'{job["condition"]}/{job["seed"]}: {rounds}/{state["cfg"]["queries_per_client"]} rounds')


if __name__ == '__main__':
    main()
