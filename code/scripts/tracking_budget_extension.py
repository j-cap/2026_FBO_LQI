"""Continue frozen every-round A/B/D/E runs to 25/50/100 total local tests."""
import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import tracking_delta_catalogue as source
import numpy as np

from src import tracking_budget_extension as policy
from src import tracking_peer_cost as peer
from src.tracking_confirmation import pinned_design, aggregate_candidates
from src.tracking_dense_gain import METHODS, BONUSES
from src.tracking_online import seed_for
from src.tracking_spectral import load_bases, save_bases
from src.tracking_budget_extension_report import write_report

base, ROOT, REPO = source.base, source.ROOT, source.REPO
CONFIG = ROOT / 'config/runs/tracking_budget_extension.json'
body_hash, prohibited = source.body_hash, source.prohibited


def setup(path=CONFIG, smoke=False):
    path = Path(path)
    requested = base.read(path)
    if (Path(requested['tag']).name != requested['tag'] or not requested['tag'].startswith('tracking_budget_extension_') or
            requested['source_interval'] != 1 or requested['prefix_queries'] != 24 or
            requested['queries_per_client'] != 100 or requested['checkpoints'] != [24, 25, 50, 100] or
            requested['smoke_queries'] != 28 or requested['bootstrap_confidence'] != .95):
        raise ValueError('Fixed continuation: every round, immutable 24-test prefix, 25/50/100 checkpoints, 28-test smoke.')
    previous = source.setup(ROOT / requested['source_config'])
    if base.read(previous['out'] / 'completion.json')['signature'] != previous['signature']:
        raise RuntimeError('Completed frozen confirmation is required.')
    all_jobs = [deepcopy(j) for j in previous['jobs'] if j['interval'] == 1]
    if len(all_jobs) != 10 or requested['smoke_seed'] not in {j['seed'] for j in all_jobs}:
        raise RuntimeError('Expected the ten completed every-round programmes.')
    prefixes = deepcopy(previous['specification']['programmes'])
    programmes = policy.make_programmes(prefixes, previous['cfg']['context_metadata'], maximum=100)
    jobs = [j for j in all_jobs if j['seed'] == requested['smoke_seed']] if smoke else all_jobs
    cap = requested['smoke_queries'] if smoke else requested['queries_per_client']
    checkpoints = [24, 25, cap] if smoke else requested['checkpoints']
    cfg = deepcopy(previous['cfg'])
    cfg.update(tag=requested['tag'], queries_per_client=cap, programmes=programmes,
        fleet_budgets=sorted(set(previous['cfg']['fleet_budgets'] + [len(previous['clients']) * b for b in checkpoints])))
    out = ROOT / 'results' / (requested['tag'] + ('_smoke' if smoke else ''))
    physics, pool, clients, domain = (previous[k] for k in ('physics', 'pool', 'clients', 'domain'))
    if (out / 'run_manifest.json').exists():
        saved = base.read(out / 'run_manifest.json')
        pinned = pinned_design(dict(configuration=cfg, physics=physics, pool=pool, clients=clients),
            {k: saved[k] for k in ('configuration', 'physics', 'pool', 'clients')})
        cfg, physics, pool, clients = (pinned[k] for k in ('configuration', 'physics', 'pool', 'clients'))
    if (out / 'public_domain.npz').exists():
        load_bases(out / 'public_domain.npz', [], domain, {})
        with np.load(out / 'public_domain.npz', allow_pickle=False) as archive:
            domain.X = archive['domain_X'].copy()
    prefix_paths = {str(j['seed']): source.run_path(previous, j) for j in all_jobs}
    prefix_hashes = {seed: base.digest(p) for seed, p in prefix_paths.items()}
    views = {seed: body_hash(policy.prefix_view(base.read(p)['state'], clients)) for seed, p in prefix_paths.items()}
    evaluation_files = sorted((previous['out'] / 'evaluation/validation').glob('*.json'))
    if not evaluation_files:
        raise RuntimeError('The original four-repeat held-out bank is required.')
    hashes = dict(previous['specification']['source_hashes'])
    files = [Path(__file__), path, ROOT / 'src/tracking_budget_extension.py', ROOT / 'src/tracking_budget_extension_report.py']
    hashes.update({p.relative_to(REPO).as_posix(): base.digest(p) for p in files})
    spec = dict(configuration=cfg, requested=requested, mode='smoke' if smoke else 'continuation',
        physics=physics, clients=clients, pool=pool, jobs=jobs, checkpoints=checkpoints, programmes=programmes,
        source_hashes=hashes, source_signature=previous['signature'], prefix_file_hashes=prefix_hashes,
        prefix_view_hashes=views, validation_seeds=previous['specification']['validation_seeds'],
        validation_input_hashes={p.relative_to(REPO).as_posix(): base.digest(p) for p in evaluation_files},
        source_audit_signature=previous['specification']['source_audit_signature'],
        reference_input_hashes=previous['specification']['reference_input_hashes'],
        public_domain_sha256=hashlib.sha256(np.ascontiguousarray(domain.X, dtype='<f8').tobytes()).hexdigest(),
        source_context_rule=previous['specification']['context_rule'],
        historical_training_records=0, reused_paid_prefix_queries=24, worker_evaluation_access=False,
        study_role='Matched budget continuation diagnostic, not a fresh independent confirmation.',
        primary_metric='Near-reference success on each client programme\'s fixed original 24-test context set.',
        context_rule='Immutable first 24 assignments; balanced-frequency suffix and independent speeds with replacement from fixed per-block/per-trial seeds.',
        method_rule='Unchanged A/B/D/E, public pool, priors, beta, eligibility, peer-first rule and every-round lossless exchange.',
        compute_rule='Identical D fits are memoized within each simulated round after all previous-round records are delivered; this is simulator compute reuse, not a client-CPU claim.',
        evaluator_rule='Reuse original four held-out seeds and archived rows; evaluate only missing gain/context/repeat rows after all live queries. No evaluator labels enter a proposal.',
        scoring_rule='Fixed original contexts, evolving encountered contexts, and all 18 contexts are separate. Missing gains remain failures.',
        tail_rule='Checkpoint traffic excludes the contact after its last decision; round-24 publication is preserved and becomes usable for query 25.')
    return dict(cfg=cfg, requested=requested, specification=spec, signature=body_hash(spec), jobs=jobs,
        domain=domain, physics=physics, pool=pool, clients=clients, refs=previous['refs'], out=out,
        source_state=previous, all_source_jobs=all_jobs, prefix_paths=prefix_paths)


def payload(state):
    return {k: state[k] for k in ('cfg', 'signature', 'domain', 'physics', 'pool', 'clients', 'refs', 'out')} | dict(
        prefix_file_hashes=state['specification']['prefix_file_hashes'], prefix_view_hashes=state['specification']['prefix_view_hashes'])


def initialize(state):
    source.initialize(state)


def run_path(state, job):
    return source.run_path(state, job)


def audit_setup(state):
    cfg = deepcopy(state['cfg'])
    cfg.update(queries_per_client=100, fleet_budgets=sorted(set(state['source_state']['cfg']['fleet_budgets'] + [225, 450, 900])))
    spec = dict(source_signature=state['specification']['source_signature'], prefix_file_hashes=state['specification']['prefix_file_hashes'],
        source_hashes=state['specification']['source_hashes'], programmes_hash=body_hash(state['specification']['programmes']))
    return payload(state) | dict(cfg=cfg, out=ROOT / 'results' / (state['requested']['tag'] + '_prefix_equivalence'),
        source_cfg=state['source_state']['cfg'], source_signature=state['specification']['source_signature'],
        prefix_paths=state['prefix_paths'], jobs=state['all_source_jobs'], audit_specification=spec, audit_signature=body_hash(spec))


def prefix_worker(job):
    state = base.WORKER
    path = run_path(state, job)
    if path.exists():
        if base.read(path)['signature'] != state['audit_signature']:
            raise RuntimeError('Prefix audit signature differs; use a fresh tag.')
        return
    original = base.read(state['prefix_paths'][str(job['seed'])])
    old_state = dict(state, cfg=state['source_cfg'], signature=state['source_signature'])
    source.validate_job(old_state, job, original, complete=True)
    with patch('src.tracking_budget.minimize', prohibited), patch('src.tracking_six_speed.minimize', prohibited), patch.object(base, 'resolved_rollout', prohibited):
        policy.validate_state(original['state'], state['domain'], state['cfg'], state['clients'], job['seed'], 1, replay=True)
    base.save(path, dict(signature=state['audit_signature'], seed=job['seed'], all_prefix_choices_identical=True,
        paid_decisions=len(state['clients']) * 24 * len(METHODS), new_rollouts=0, new_optimization=0))


def ensure_prefix_gate(state, workers):
    audit = audit_setup(state)
    base.parallel(prefix_worker, audit['jobs'], audit, workers, 'Frozen 24-test replay')
    records = [base.read(run_path(audit, j)) for j in audit['jobs']]
    if any(r['signature'] != audit['audit_signature'] or not r['all_prefix_choices_identical'] or r['new_rollouts'] or r['new_optimization'] for r in records):
        raise RuntimeError('Frozen-prefix gate failed.')
    base.save(audit['out'] / 'run_manifest.json', dict(signature=audit['audit_signature'], **audit['audit_specification']))
    base.save(audit['out'] / 'completion.json', dict(signature=audit['audit_signature'], all_checks_passed=True,
        paid_decisions_replayed=sum(r['paid_decisions'] for r in records), new_rollouts=0, new_optimization=0))
    return audit


def validate_job(state, job, result, complete=False):
    if result.get('signature') != state['signature'] or result.get('clients') != state['clients'] or any(result.get(k) != v for k, v in job.items()):
        raise RuntimeError('Continuation checkpoint identity differs.')
    if result['state']['round'] < 24 or result['prefix_file_hash'] != state['prefix_file_hashes'][str(job['seed'])]:
        raise RuntimeError('Missing or mismatched paid prefix.')
    if body_hash(policy.prefix_view(result['state'], state['clients'])) != state['prefix_view_hashes'][str(job['seed'])]:
        raise RuntimeError('A frozen prefix event, observation, packet or snapshot changed.')
    policy.validate_state(result['state'], state['domain'], state['cfg'], state['clients'], job['seed'], 1, complete=complete)
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


def import_prefixes(state):
    for job in state['jobs']:
        path = run_path(state, job)
        if path.exists():
            validate_job(payload(state), job, base.read(path))
            continue
        original = base.read(state['prefix_paths'][str(job['seed'])])
        result = dict(**job, signature=state['signature'], clients=state['clients'], state=deepcopy(original['state']),
            prefix_file_hash=state['specification']['prefix_file_hashes'][str(job['seed'])],
            timing=deepcopy(original.get('timing', [])), prefix_elapsed_seconds=original.get('elapsed_seconds', 0.),
            added_elapsed_seconds=0.)
        validate_job(payload(state), job, result)
        base.save(path, result)


def live_worker(job):
    state = base.WORKER
    path = run_path(state, job)
    previous = base.read(path)
    validate_job(state, job, previous)
    timing = deepcopy(previous['timing'])
    started = time.perf_counter()
    def checkpoint(data):
        base.save(path, dict(**job, signature=state['signature'], clients=state['clients'], state=data, timing=timing,
            prefix_file_hash=previous['prefix_file_hash'], prefix_elapsed_seconds=previous['prefix_elapsed_seconds'],
            added_elapsed_seconds=previous['added_elapsed_seconds'] + time.perf_counter() - started))
        if data['round'] % 5 == 0 or data['round'] == state['cfg']['queries_per_client']:
            print(f'Seed {job["seed"]}: {data["round"]}/{state["cfg"]["queries_per_client"]} total tests per client', flush=True)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        policy.run_fleet(state['domain'], state['cfg'], state['clients'], job['seed'], 1,
            lambda plans, cache: base.observe(plans, cache, state), checkpoint, state=previous['state'], timing_hook=timing.append)
    validate_job(state, job, base.read(path), complete=True)


def replay_worker(job):
    state = base.WORKER
    result = base.read(run_path(state, job))
    validate_job(state, job, result, complete=True)
    original = base.read(state['prefix_paths'][str(job['seed'])])
    def observe(plans, cache):
        observations = []
        for plan in plans:
            event = result['state']['methods'][plan['method']]['events'][(plan['round'] - 1) * len(state['clients']) + plan['client']]
            policy.old.assert_equivalent(plan, {k: event[k] for k in plan})
            observations.append({k: event[k] for k in peer.OBSERVATION_FIELDS if k != 'action'})
        return observations
    with patch('src.tracking_budget.minimize', prohibited), patch('src.tracking_six_speed.minimize', prohibited), patch.object(base, 'resolved_rollout', prohibited):
        rebuilt = policy.run_fleet(state['domain'], state['cfg'], state['clients'], job['seed'], 1, observe,
            state=original['state'], recorded=result['state'])
    policy.old.assert_equivalent(rebuilt['methods'], result['state']['methods'])


def completed_live(state, replay=False, workers=1):
    if replay:
        base.parallel(replay_worker, state['jobs'], payload(state) | dict(prefix_paths=state['prefix_paths']), workers, 'Continuation decision and packet replay')
    results = [base.read(run_path(state, j)) for j in state['jobs']]
    if not replay:
        for j, r in zip(state['jobs'], results):
            validate_job(payload(state), j, r, complete=True)
    return results


reference_selection = source.reference_selection
evaluation_path = source.evaluation_path
load_evaluation = source.load_evaluation


def old_validation_bank(state):
    bank = {}
    for name, expected in state['specification']['validation_input_hashes'].items():
        path = REPO / name
        if base.digest(path) != expected:
            raise RuntimeError('Frozen held-out input changed: ' + name)
        batch = base.read(path)
        if batch['signature'] != state['specification']['source_signature'] or batch['noise_seed'] != state['specification']['validation_seeds'][batch['repeat']]:
            raise RuntimeError('Held-out noise seed differs.')
        for row in batch['rows']:
            key = batch['context'], batch['repeat'], row['candidate']
            if key in bank:
                raise RuntimeError('Repeated held-out observation identity.')
            bank[key] = row
    return bank


def evaluation_tasks(state, selection):
    tasks, width = [], state['requested']['validation_batch_size']
    for context, candidates in selection.items():
        ids = sorted(candidates)
        for repeat, noise in enumerate(state['specification']['validation_seeds']):
            for start in range(0, len(ids), width):
                tasks.append(dict(role='validation', context=context, repeat=repeat, noise_seed=noise,
                    chunk=start // width, candidates=ids[start:start + width]))
    return tasks


def evaluation_worker(task):
    state = base.WORKER
    path = evaluation_path(state, task)
    if path.exists():
        base.validate_batch(state, task, base.read(path))
        return
    reused = state['validation_reuse']
    missing = [c for c in task['candidates'] if (task['context'], task['repeat'], c) not in reused]
    fresh = {}
    if missing:
        lookup = {c['id']: c for c in state['pool']}
        ref = next(r for r in state['refs'] if r['id'] == task['context'])
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            rows, _ = base.resolved_rollout(state['clients'][0]['params'], ref, [lookup[c] for c in missing], state['physics'], task['noise_seed'])
        fresh = {r['candidate']: r for r in rows}
    rows = [deepcopy(reused[(task['context'], task['repeat'], c)]) if c not in fresh else fresh[c] for c in task['candidates']]
    batch = dict(signature=state['signature'], **task, rows=rows, new_rollouts=len(missing), reused_rollouts=len(rows) - len(missing))
    base.validate_batch(state, task, batch)
    base.save(path, batch)


def preflight(state, workers=3):
    audit = ensure_prefix_gate(state, workers)
    initialize(state)
    import_prefixes(state)
    reused = len(state['jobs']) * len(state['clients']) * 24 * len(METHODS)
    total = len(state['jobs']) * len(state['clients']) * state['cfg']['queries_per_client'] * len(METHODS)
    base.save(state['out'] / 'preflight.json', dict(signature=state['signature'], prefix_gate_signature=audit['audit_signature'],
        reused_paid_requests=reused, additional_paid_requests=total - reused, total_paid_requests=total,
        checkpoints=state['specification']['checkpoints'], methods=METHODS, bonus_by_method=BONUSES))
    print(f'{state["specification"]["mode"]}: {len(state["jobs"])} jobs; {reused:,} reused paid tests; {total - reused:,} additional; {total:,} total.', flush=True)


def analyze(state, workers=3):
    audit = ensure_prefix_gate(state, workers)
    initialize(state)
    results = completed_live(state, replay=True, workers=workers)
    chosen = reference_selection(state)
    tasks = evaluation_tasks(state, base.validation_selection(state, results, chosen))
    validation = load_evaluation(state, tasks)
    refs = {c: dict(candidate=k, **validation[c][k]) if k is not None else dict(candidate=None, eligible=False, J=None) for c, k in chosen.items()}
    reused = old_validation_bank(state)
    fresh = sum((t['context'], t['repeat'], c) not in reused for t in tasks for c in t['candidates'])
    write_report(state, results, validation, refs, audit)
    total = len(state['jobs']) * len(state['clients']) * state['cfg']['queries_per_client'] * len(METHODS)
    old = len(state['jobs']) * len(state['clients']) * 24 * len(METHODS)
    base.save(state['out'] / 'verification.json', dict(signature=state['signature'], all_choices_and_packets_replayed=True,
        all_prefixes_unchanged=True, reused_paid_requests=old, additional_paid_requests=total - old, total_paid_requests=total,
        heldout_rollouts=sum(len(t['candidates']) for t in tasks), reused_heldout_rollouts=sum(len(t['candidates']) for t in tasks) - fresh,
        new_heldout_rollouts=fresh, new_calibration_rollouts=0, confirmation_claim=False))
    base.save(state['out'] / 'completion.json', dict(signature=state['signature']))
    print('Report: ' + str(state['out'] / 'report.html'), flush=True)


def run(state, workers=3):
    preflight(state, workers)
    pending = [j for j in state['jobs'] if base.read(run_path(state, j))['state']['round'] < state['cfg']['queries_per_client']]
    base.parallel(live_worker, pending, payload(state), workers, 'Budget continuations')
    results = completed_live(state)
    chosen = reference_selection(state)
    tasks = evaluation_tasks(state, base.validation_selection(state, results, chosen))
    base.parallel(evaluation_worker, tasks, payload(state) | dict(validation_reuse=old_validation_bank(state)), workers, 'Held-out evaluation (missing rows only)')
    analyze(state, workers)


def collect(state, workers=3):
    if state['specification']['mode'] != 'continuation':
        raise RuntimeError('Smoke output cannot be collected as the 100-test study.')
    analyze(state, workers)
    audit = audit_setup(state)
    subprocess.run(['git', 'add', '-f', '--', state['out'].relative_to(REPO).as_posix(), audit['out'].relative_to(REPO).as_posix()], cwd=REPO, check=True)


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
        ensure_prefix_gate(state, args.workers)
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
