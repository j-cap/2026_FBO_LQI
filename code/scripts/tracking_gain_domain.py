"""Offline gain-resolution and physical-boundary audit; see docs/tracking_gain_domain.md."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

# Reuse file/process infrastructure only; never call the BO runner's setup.
import tracking_six_speed as base
import numpy as np
import scipy

from src.tracking_confirmation import aggregate_candidates, pinned_design
from src.tracking_gain_domain import build_pool, stage_candidates, choose_stages, validation_selection
from src.tracking_gain_domain_report import write_report
from src.tracking_online import seed_for
from src.tracking_task import fleet, reference_metadata, references, resolved_rollout

ROOT, REPO = base.ROOT, base.REPO
CONFIG = ROOT / 'config/runs/tracking_gain_domain.json'


def setup(path=CONFIG, smoke=False):
    path = Path(path)
    requested = base.read(path)
    tag, source_tag = requested['tag'], requested['source_tag']
    if Path(tag).name != tag or not tag.startswith('tracking_gain_domain_') or Path(source_tag).name != source_tag:
        raise ValueError('Use separate single-directory audit/source tags.')
    if (requested['dense_sobol_power'] != 10 or requested['expanded_sobol_power'] != 11 or
            requested['expanded_multiplier_upper'] != 30. or requested['calibration_repeats'] != 2 or
            requested['validation_repeats'] != 4 or not 1 <= requested['batch_size'] <= 256):
        raise ValueError('This audit fixes 1024 inside draws, 2048 expanded draws, upper multiplier 30 and 2/4 repeats.')
    source_out = ROOT / 'results' / source_tag
    source = base.read(source_out / 'run_manifest.json')
    source_body = {k: v for k, v in source.items() if k not in ('signature', 'git_head', 'runtime')}
    if hashlib.sha256(json.dumps(source_body, sort_keys=True).encode()).hexdigest() != source['signature']:
        raise RuntimeError('Source manifest does not match its signature.')
    for relative, expected in source['source_hashes'].items():
        if base.digest(REPO / relative) != expected:
            raise RuntimeError('Frozen source/configuration changed: ' + relative)
    physics = deepcopy(base.read(ROOT / requested['physics_config']))
    source_metadata = source['configuration']['context_metadata']
    physics['references']['speeds_m_s'] = sorted({r['speed'] for r in source_metadata})
    refs = references(physics)
    metadata = pinned_design([reference_metadata(r, physics['dt']) for r in refs], source_metadata)
    common = next(v['params'] for v in fleet(physics) if v['id'] == 'compact_nominal')
    plant = pinned_design(common, source['clients'][0]['params'])
    if any(c['params'] != source['clients'][0]['params'] for c in source['clients']):
        raise ValueError('One common plant is required for this homogeneous audit.')
    contexts = metadata
    if smoke:
        selected = requested['smoke']['contexts']
        if len(selected) != len(set(selected)) or not set(selected) <= {r['id'] for r in metadata}:
            raise ValueError('Invalid smoke contexts.')
        contexts = [r for r in metadata if r['id'] in selected]
        refs = [r for r in refs if r['id'] in selected]
    out = ROOT / 'results' / (tag + ('_smoke' if smoke else ''))
    pool, labels, profiles = build_pool(source['pool'], physics['gains'], requested, smoke)
    old = base.read(out / 'run_manifest.json') if (out / 'run_manifest.json').exists() else None
    if old:
        pinned = pinned_design(dict(pool=pool, labels=labels, profiles=profiles, physics=physics, plant=plant),
            {k: old[k] for k in ('pool', 'labels', 'profiles', 'physics', 'plant')})
        pool, labels, profiles, physics, plant = (pinned[k] for k in ('pool', 'labels', 'profiles', 'physics', 'plant'))
    members = stage_candidates(pool, labels)
    source_inputs = {str((source_out / 'run_manifest.json').relative_to(REPO)): base.digest(source_out / 'run_manifest.json')}
    source_batches = {}
    for context in contexts:
        for repeat in range(requested['calibration_repeats']):
            file = source_out / 'evaluation/calibration' / f'{context["id"]}__{repeat}.json'
            batch = base.read(file)
            expected_noise = seed_for('six_speed_evaluator', source['configuration']['tag'], source['mode'], 'calibration', repeat)
            if (batch.get('signature') != source['signature'] or batch['context'] != context['id'] or
                    batch['role'] != 'calibration' or batch['repeat'] != repeat or batch['noise_seed'] != expected_noise or
                    batch['candidates'] != members['original'] or [r['candidate'] for r in batch['rows']] != members['original']):
                raise RuntimeError('Frozen calibration input identity differs.')
            source_batches[context['id'], repeat] = batch
            source_inputs[file.relative_to(REPO).as_posix()] = base.digest(file)
    settings = requested['smoke'] if smoke else requested
    validation_seeds = [seed_for('gain_domain_validation', tag, 'smoke' if smoke else 'audit', rep)
                        for rep in range(settings['validation_repeats'])]
    old_seeds = {b['noise_seed'] for b in source_batches.values()}
    old_seeds |= {seed_for('six_speed_evaluator', source['configuration']['tag'], source['mode'], 'validation', rep)
                  for rep in range(source['requested']['validation_repeats'])}
    if len(set(validation_seeds)) != len(validation_seeds) or set(validation_seeds) & old_seeds:
        raise RuntimeError('Fresh validation noise must be disjoint from source evaluator noise.')
    paths = [Path(__file__), path, ROOT / 'src/tracking_gain_domain.py', ROOT / 'src/tracking_gain_domain_report.py',
             ROOT / 'src/tracking_task.py', ROOT / 'src/tracking_confirmation.py', ROOT / 'src/tracking_online.py']
    paths += sorted((ROOT / 'src/tracking_reference').glob('*.py'))
    spec = dict(requested=requested, mode='smoke' if smoke else 'audit', pool=pool, labels=labels,
        profiles=profiles, physics=physics, plant=plant, contexts=contexts, members=members,
        source_tag=source_tag, source_signature=source['signature'], source_inputs=source_inputs,
        validation_seeds=validation_seeds,
        source_hashes={p.relative_to(REPO).as_posix(): base.digest(p) for p in paths},
        new_online_requests=0, online_query_cap_unchanged=24, spectral_basis_constructed=False,
        interpretation='Offline best-found feasible controllers; finite candidate sets, not continuous global optima.',
        normalization_rule='Original (log10 multiplier + 1)/2 coordinates retained, including values above 1; no GP or kernel is fitted.',
        selection_rule='Original/dense/expanded winners selected on two matched calibration repeats; no validation fallback or reselection.')
    signature = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    return dict(requested=requested, source=source, source_batches=source_batches, physics=physics,
        plant=plant, refs=refs, contexts=contexts, pool=pool, labels=labels, profiles=profiles,
        members=members, validation_seeds=validation_seeds, out=out, specification=spec, signature=signature)


def initialize(state):
    path = state['out'] / 'run_manifest.json'
    if path.exists():
        if base.read(path)['signature'] != state['signature']:
            raise RuntimeError('Code/configuration changed; use a new audit tag.')
    else:
        base.save(path, dict(signature=state['signature'], **state['specification'],
            git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)))


def tasks(state, role, selection=None):
    result = []
    for context in state['contexts']:
        cid = context['id']
        candidates = (state['requested']['source_check_candidates'] if role == 'source_check' else
            [c['id'] for c in state['pool'] if not state['labels'][c['id']]['original']] if role == 'calibration' else
            sorted(selection[cid]))
        repeats = range(len(state['validation_seeds'])) if role == 'validation' else range(state['requested']['calibration_repeats'])
        for repeat in repeats:
            noise = state['validation_seeds'][repeat] if role == 'validation' else state['source_batches'][cid, repeat]['noise_seed']
            for start in range(0, len(candidates), state['requested']['batch_size']):
                result.append(dict(role=role, context=cid, repeat=repeat, noise_seed=noise,
                    chunk=start // state['requested']['batch_size'],
                    candidates=candidates[start:start + state['requested']['batch_size']]))
    return result


def job_path(state, task):
    return state['out'] / 'batches' / task['role'] / f'{task["context"]}__{task["repeat"]}__{task["chunk"]:04d}.json'


def validate_batch(state, task, batch):
    if batch.get('signature') != state['signature'] or any(batch.get(k) != v for k, v in task.items()):
        raise RuntimeError('Audit batch identity differs.')
    if [r['candidate'] for r in batch['rows']] != task['candidates']:
        raise RuntimeError('Audit batch candidate set differs.')
    for row in batch['rows']:
        if (row['feasible'] and not row['complete']) or (row['complete'] and (
                row['J'] is None or not np.isfinite(row['J']) or row['J'] <= 0)):
            raise RuntimeError('Invalid simulator outcome.')


def worker(task):
    state = base.WORKER
    path = job_path(state, task)
    if path.exists():
        validate_batch(state, task, base.read(path))
        return
    started = time.perf_counter()
    ref = next(r for r in state['refs'] if r['id'] == task['context'])
    pool = [state['lookup'][c] for c in task['candidates']]
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        rows, _ = resolved_rollout(state['plant'], ref, pool, state['physics'], task['noise_seed'])
    batch = dict(signature=state['signature'], **task, rows=rows, elapsed_seconds=time.perf_counter() - started)
    validate_batch(state, task, batch)
    base.save(path, batch)


def payload(state):
    return {k: state[k] for k in ('out', 'signature', 'plant', 'refs', 'physics')} | {
        'lookup': {c['id']: c for c in state['pool']}}


def load_batch(state, task):
    path = job_path(state, task)
    if not path.exists():
        raise RuntimeError('Missing audit batch: ' + str(path))
    batch = base.read(path)
    validate_batch(state, task, batch)
    return batch


def verify_source_checks(state):
    checked = 0
    for task in tasks(state, 'source_check'):
        batch = load_batch(state, task)
        source = {r['candidate']: r for r in state['source_batches'][task['context'], task['repeat']]['rows']}
        for row in batch['rows']:
            old = source[row['candidate']]
            if any(row[k] != old[k] for k in ('complete', 'feasible', 'failure', 'integration_resolved', 'substeps')):
                raise RuntimeError('Source simulator classification differs; refusing calibration-label reuse.')
            for key in ('J', 'rmse_y', 'rms_rate_deg_s', 'peak_y', 'peak_beta_deg', 'peak_speed_error', 'peak_ay'):
                if row[key] is None or old[key] is None:
                    if row[key] != old[key]:
                        raise RuntimeError('Source null outcome differs.')
                elif not np.isclose(row[key], old[key], rtol=state['requested']['source_check_rtol'],
                                   atol=state['requested']['source_check_atol']):
                    raise RuntimeError('Source simulator numerical outcome differs; refusing label reuse.')
            checked += 1
    return checked


def load_calibration(state):
    grouped = {}
    for task in tasks(state, 'calibration'):
        grouped.setdefault((task['context'], task['repeat']), []).append(task)
    calibration = {}
    expected = [c['id'] for c in state['pool']]
    for context in state['contexts']:
        batches = []
        for repeat in range(state['requested']['calibration_repeats']):
            rows = list(state['source_batches'][context['id'], repeat]['rows'])
            for task in grouped[context['id'], repeat]:
                rows += load_batch(state, task)['rows']
            if [r['candidate'] for r in rows] != expected:
                raise RuntimeError('Missing or repeated calibration candidate.')
            batches.append(dict(rows=rows))
        calibration[context['id']] = aggregate_candidates(batches)
    return calibration


def load_validation(state, choices):
    selected = validation_selection(choices, state['profiles'], state['contexts'])
    validation_tasks = tasks(state, 'validation', selected)
    buckets = {}
    for task in validation_tasks:
        buckets.setdefault(task['context'], []).append(load_batch(state, task))
    return {context: aggregate_candidates(batches) for context, batches in buckets.items()}, validation_tasks, buckets


def preflight(state):
    initialize(state)
    counts = {k: len(v) for k, v in state['members'].items()}
    new_cal = sum(len(t['candidates']) for t in tasks(state, 'calibration'))
    parity = sum(len(t['candidates']) for t in tasks(state, 'source_check'))
    old_hi = np.asarray(state['physics']['gains']['nominal']) * 10
    expanded_hi = np.asarray(state['physics']['gains']['nominal']) * state['requested']['expanded_multiplier_upper']
    record = dict(signature=state['signature'], mode=state['specification']['mode'], candidate_counts=counts,
        new_outside_candidates=counts['expanded'] - counts['dense_original'],
        original_actual_upper_gains=old_hi.tolist(), expanded_actual_upper_gains=expanded_hi.tolist(),
        new_calibration_rollouts=new_cal, source_check_rollouts=parity,
        reused_calibration_rollouts=len(state['contexts']) * len(state['members']['original']) * state['requested']['calibration_repeats'],
        new_online_requests=0, spectral_basis_constructed=False)
    base.save(state['out'] / 'preflight.json', record)
    print(f'{record["mode"]}: candidates {counts}; {record["new_outside_candidates"]} genuine outside points.', flush=True)
    print(f'New calibration rollouts: {new_cal:,}; source checks: {parity}; reused calibration: {record["reused_calibration_rollouts"]:,}.', flush=True)
    print(f'Physical upper gains: {old_hi.tolist()} -> {expanded_hi.tolist()}. No BO or spectral kernel construction.', flush=True)


def analyze(state):
    initialize(state)
    checked = verify_source_checks(state)
    calibration = load_calibration(state)
    choices = choose_stages(calibration, state['members'])
    validation, val_tasks, buckets = load_validation(state, choices)
    write_report(state, calibration, choices, validation, buckets)
    if any(base.digest(REPO / name) != fingerprint for name, fingerprint in state['specification']['source_inputs'].items()):
        raise RuntimeError('Frozen source evaluator inputs changed.')
    base.save(state['out'] / 'verification.json', dict(signature=state['signature'],
        original_candidates_preserved=state['pool'][:len(state['members']['original'])] == state['source']['pool'],
        source_check_rollouts=checked, source_inputs_unchanged=True,
        new_calibration_rollouts=sum(len(t['candidates']) for t in tasks(state, 'calibration')),
        validation_rollouts=sum(len(t['candidates']) for t in val_tasks),
        new_online_requests=0, spectral_basis_constructed=False,
        validation_selection_from_calibration_only=True, mode=state['specification']['mode']))
    base.save(state['out'] / 'completion.json', dict(signature=state['signature']))
    print('Report: ' + str(state['out'] / 'report.html'), flush=True)


def collect(state):
    if state['specification']['mode'] != 'audit':
        raise RuntimeError('Smoke outputs cannot be collected as the full domain audit.')
    analyze(state)
    relative = state['out'].relative_to(REPO).as_posix()
    subprocess.run(['git', 'add', '-f', '--', relative], cwd=REPO, check=True)
    print('Staged only ' + relative, flush=True)


def run(state, workers):
    preflight(state)
    data = payload(state)
    base.parallel(worker, tasks(state, 'source_check'), data, workers, 'Source simulator checks')
    verify_source_checks(state)
    base.parallel(worker, tasks(state, 'calibration'), data, workers, 'New calibration batches')
    choices = choose_stages(load_calibration(state), state['members'])
    selected = validation_selection(choices, state['profiles'], state['contexts'])
    base.parallel(worker, tasks(state, 'validation', selected), data, workers, 'Fresh winner/profile validation')
    analyze(state)


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
        analyze(state)
    else:
        collect(state)


if __name__ == '__main__':
    main()
