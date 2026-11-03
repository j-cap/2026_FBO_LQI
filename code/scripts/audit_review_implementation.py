"""Read-only implementation audit for Markus's steering/near-best-set comments.

Run from the repository root or code/: python <path>/audit_review_implementation.py
No simulations, cache deserialization, manuscript edits, or figure regeneration.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import inspect
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
REPO = CODE.parent
sys.path.insert(0, str(CODE))
from src import candidate_eval
from src.episodes import EpisodeBank, EpisodeSpec
from src.io_utils import deep_update, load_yaml


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path):
    """User paths resolve from cwd; config references resolve from code/."""
    return Path(path).resolve()


def load_config(path):
    cfg = load_yaml(str(path))
    fc_path = CODE / cfg['fleet_config']
    cc_path = CODE / cfg['controller_config']
    fc = deep_update(load_yaml(str(fc_path)), cfg.get('fleet_overrides', {}))
    cc = deep_update(load_yaml(str(cc_path)), cfg.get('controller_overrides', {}))
    return cfg, fc, cc, [path, fc_path, cc_path]


def steering_checks(fc, cc):
    """Exercise the real evaluate() with deterministic injected rollout outputs.

    Only synthesis/plant simulation are stubbed. Constraint computation, aggregation,
    strictness, result flags and feasibility are the production implementation.
    """
    limit, ts = float(cc['u_rate_max']), float(fc['Ts'])
    if not (np.isfinite(limit) and limit > 0 and np.isfinite(ts) and ts > 0):
        raise ValueError('Expected positive finite steering increment limit and Ts')
    rows = []
    cases = [
        ('below', [np.array([0., limit * (1 - 1e-8), 0.])], False),
        ('equal', [np.array([0., limit, 0.])], False),
        ('above', [np.array([0., limit * (1 + 1e-8), 0.])], True),
        ('negative_jump', [np.array([0., -1.1 * limit, 0.])], True),
        ('constant_nonzero', [np.array([0.3, 0.3, 0.3])], False),
        ('second_episode', [np.zeros(3), np.array([0., 1.1 * limit, 0.])], True),
        ('single_sample', [np.array([0.3])], False),
    ]
    for time_scale in (1., 2.):
        for name, inputs, expected_violation in cases:
            client = SimpleNamespace(client_id='audit', cluster=SimpleNamespace(name='audit'),
                                     Ad_hat=np.eye(2), Bd_hat=np.ones((2, 1)))
            design = SimpleNamespace(nominal_feasible=True, Kx=np.zeros((1, 2)),
                                     Ki=np.zeros((1, 1)), k_r=0.)
            bank = EpisodeBank(t=np.arange(3) * ts * time_scale, r_ref=np.zeros(3),
                               episodes=[EpisodeSpec(seed=i, role='calibration')
                                         for i in range(len(inputs))])
            rollouts = [SimpleNamespace(u=u, y=np.zeros(len(u)), x=np.zeros((len(u), 2)))
                        for u in inputs]
            with patch.object(candidate_eval, 'synthesize', return_value=design), \
                 patch.object(candidate_eval.PlantClient, 'rollout', side_effect=rollouts) as mock:
                result = candidate_eval.evaluate(client, [0., 0., 0.], bank,
                                                 u_rate_max=limit, beta_max=1.)
            expected_increment = max(float(np.max(np.abs(np.diff(u)))) if len(u) > 1 else 0.
                                     for u in inputs)
            ok = (result.max_abs_input_rate == expected_increment
                  and bool(result.input_rate_limit_violated) == expected_violation
                  and bool(result.feasible) == (not expected_violation)
                  and result.failure_reason == ('input_rate_limit' if expected_violation else None)
                  and mock.call_count == len(inputs))
            rows.append(dict(case=name, time_step_s=ts*time_scale,
                             expected_increment_rad=expected_increment,
                             reported_max_abs_input_rate=result.max_abs_input_rate,
                             expected_violation=expected_violation,
                             observed_violation=result.input_rate_limit_violated, passed=bool(ok)))
    return rows, dict(
        configured_increment_rad=limit, time_step_s=ts,
        equivalent_rate_rad_s=limit/ts, equivalent_rate_deg_s=float(np.rad2deg(limit/ts)),
        difference_from_exact_300_deg_s=float(np.rad2deg(limit/ts)-300.),
        interpretation='max(abs(diff(applied steering))) in rad/step; strict > comparison',
        scope='All supplied rollouts complete before final feasibility decision. '
              'No initial jump from an assumed pre-rollout input is included by np.diff.',
        tested_time_step_invariance='Doubling bank time spacing does not rescale the checked increment.',
        limitation='Controlled rollout probes establish evaluator semantics, not physical safety '
                   'or historical rollout reproduction.')


def selection_source():
    """Compile ONLY the existing top_n/top_sets statements, without running main().

    This deliberately avoids copying the selection algorithm into the audit. A
    changed source structure fails explicitly instead of falling back to our own rule.
    """
    path = CODE / 'scripts/calibration_landscape_similarity.py'
    source = path.read_text(encoding='utf-8')
    tree = ast.parse(source)
    names = ('N_GRID_POINTS', 'N_STUDY_SEEDS', 'GRID_SEED', 'TOP_FRACTION')
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                constants[target.id] = ast.literal_eval(node.value)
    if set(constants) != set(names):
        raise ValueError('Landscape constants changed; audit needs review')
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    top_assigns = [n for n in ast.walk(main) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'top_n' for t in n.targets)]
    blocks = []
    for node in ast.walk(main):
        body = getattr(node, 'body', None)
        if not isinstance(body, list):
            continue
        for index, stmt in enumerate(body[:-1]):
            if (isinstance(stmt, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'top_sets' for t in stmt.targets)
                and isinstance(body[index+1], ast.For)):
                blocks.append([stmt, body[index+1]])
    if len(top_assigns) != 1 or len(blocks) != 1:
        raise ValueError('Landscape selection block changed; audit needs review')
    statements = [top_assigns[0], *blocks[0]]
    program = compile(ast.Module(body=statements, type_ignores=[]), str(path), 'exec')
    excerpt = '\n\n'.join(ast.get_source_segment(source, n) for n in statements)
    return constants, program, excerpt


def selection_checks(constants, program):
    size = int(constants['N_GRID_POINTS'])
    cap = max(1, math.ceil(constants['TOP_FRACTION'] * size))
    rows = []
    counts = sorted(set([0, 1, min(7, size), min(12, size), min(13, size), min(64, size), size]))
    for count in counts:
        # Noncontiguous feasible coordinates; strict costs so ties cannot hide errors.
        feasible_ids = np.random.default_rng(11).permutation(size)[:count]
        J = np.full((1, size), np.nan)
        mask = np.zeros((1, size), dtype=bool)
        mask[0, feasible_ids] = True
        J[0, feasible_ids] = np.arange(count, 0, -1)
        scope = dict(constants, np=np, math=math, J=J, feasible_mask=mask, fleet=[None])
        exec(program, scope)
        selected = scope['top_sets'][0]
        expected = set(feasible_ids[max(0, count-cap):].tolist())
        feasible_fraction_n = min(count, max(1, math.ceil(constants['TOP_FRACTION']*count)))
        rows.append(dict(n_grid=size, n_feasible=count, actual_count=len(selected),
                         fixed_grid_rule_count=min(cap, count),
                         feasible_fraction_rule_count=feasible_fraction_n,
                         selected_grid_indices=' '.join(map(str, sorted(selected))),
                         passed=selected == expected))
    return rows, cap


def archive_checks(path, cfg, constants, fc):
    if not path.is_file():
        return dict(status='NOT_AVAILABLE', path=str(path),
                    reason='Ignored historical pair archive is absent; no archive validation performed.'), []
    df = pd.read_csv(path)
    required = {'seed', 'client_i', 'client_j', 'overlap_10pct', 'rho_J'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing archive columns: {sorted(required-set(df.columns))}')
    expected_seeds = list(cfg['seeds'][:constants['N_STUDY_SEEDS']])
    n_clients = int(fc['clients_per_regime']) * 3
    rows = []
    all_ok = not df[list(required)].isna().any().any()
    for seed, group in df.groupby('seed'):
        pairs = [tuple(sorted((str(a), str(b)))) for a, b in zip(group.client_i, group.client_j)]
        clients = set(group.client_i) | set(group.client_j)
        valid = (len(group) == math.comb(n_clients, 2) and len(clients) == n_clients
                 and len(set(pairs)) == len(group) and all(a != b for a, b in pairs))
        all_ok = all_ok and valid
        rows.append(dict(seed=int(seed), n_clients=len(clients), n_pairs=len(group),
                         unique_pairs=len(set(pairs)), expected_pairs=math.comb(n_clients, 2),
                         passed=bool(valid)))
    all_ok = (all_ok and set(df.seed) == set(expected_seeds)
              and df.overlap_10pct.between(0, 1).all() and df.rho_J.between(-1, 1).all())
    finite_or_none = lambda x: float(x) if np.isfinite(x) else None
    return dict(status='PASS' if all_ok else 'FAIL', path=str(path), sha256=digest(path),
                expected_seeds=expected_seeds, n_pairs=len(df),
                rho_ge_095=int((df.rho_J >= .95).sum()),
                overlap_mean=finite_or_none(df.overlap_10pct.mean()),
                overlap_median=finite_or_none(df.overlap_10pct.median()),
                overlap_ge_075_fraction=finite_or_none((df.overlap_10pct >= .75).mean()),
                limitation='Aggregated pair rows cannot reconstruct selected grid indices or prove '
                           'which source revision generated them. This validates structure and summaries, '
                           'not historical membership or simulation replay.'), rows


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def git(*args):
    result = subprocess.run(['git', '-C', str(REPO), *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=resolve,
                        default=CODE/'config/runs/phase2_warmstart_confirmatory_c1cds02.yaml')
    parser.add_argument('--landscape-config', type=resolve,
                        default=CODE/'config/runs/phase2_warmstart_dev_c1cds02.yaml')
    parser.add_argument('--pairs', type=resolve,
                        default=CODE/'results/calibration_landscape_similarity/processed/landscape_similarity_pairs.csv')
    parser.add_argument('--output', type=resolve,
                        default=REPO/'reviews/implementation_audit')
    args = parser.parse_args()
    # Fail before overwriting any report. Different runs can use --output.
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  git_commit=git('rev-parse', 'HEAD'), git_status_before=git('status', '--short'),
                  python=platform.python_version(), platform=platform.platform(),
                  packages={name: importlib.metadata.version(name)
                            for name in ['numpy', 'scipy', 'pandas', 'PyYAML']})
    try:
        _, fc, cc, inputs = load_config(args.config)
        lc, lfc, _, linputs = load_config(args.landscape_config)
        constants, program, excerpt = selection_source()
        steering, steering_info = steering_checks(fc, cc)
        selection, cap = selection_checks(constants, program)
        archive, archive_rows = archive_checks(args.pairs, lc, constants, lfc)
        sources = [CODE/'src/candidate_eval.py', CODE/'src/interfaces/plant_client.py',
                   CODE/'src/nonlinear_plant.py', CODE/'scripts/calibration_landscape_similarity.py',
                   Path(__file__).resolve(), *inputs, *linputs]
        report.update(source_sha256={str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p): digest(p)
                                     for p in sources},
                      resolved_steering_config={'fleet': fc, 'controller': cc},
                      landscape_constants=constants, steering=steering_info,
                      selection_rule=f'Select up to {cap} lowest-cost feasible grid points per system; '
                                     'the cap is based on the full grid, not the feasible count.',
                      selection_limitation='Synthetic cases execute the actual selection statements. '
                                           'No historical per-system selected sets are reconstructed.',
                      archive=archive)
        passed = all(row['passed'] for row in steering + selection) and archive['status'] != 'FAIL'
        report['status'] = ('PASS' if archive['status'] == 'PASS' else 'PARTIAL') if passed else 'FAIL'
        write_csv(args.output/'steering_cases.csv', steering)
        write_csv(args.output/'selection_cases.csv', selection)
        write_csv(args.output/'archive_fleets.csv', archive_rows)
        (args.output/'selection_source.txt').write_text(excerpt+'\n', encoding='utf-8')
        (args.output/'evaluator_source.txt').write_text(inspect.getsource(candidate_eval.evaluate), encoding='utf-8')
        text = (f"Status: {report['status']}\nCommit: {report['git_commit']}\n\n"
                f"Steering probes: {sum(r['passed'] for r in steering)}/{len(steering)} passed.\n"
                f"Configured increment: {steering_info['configured_increment_rad']} rad/step.\n"
                f"At Ts={steering_info['time_step_s']} s: "
                f"{steering_info['equivalent_rate_deg_s']:.9f} deg/s.\n"
                "Keep the existing numeric threshold; 300 deg/s is rounded manuscript wording.\n\n"
                f"Selection probes: {sum(r['passed'] for r in selection)}/{len(selection)} passed.\n"
                f"{report['selection_rule']}\n\nArchive: {archive['status']}\n"
                f"{archive.get('reason', archive.get('limitation'))}\n\n"
                "No simulation, historical cache, manuscript, figure, or production algorithm was changed.\n"
                "A PASS verifies current implementation semantics and available pair-table checks; "
                "it does not certify historical selected-set membership or safe deployment.\n")
        (args.output/'summary.txt').write_text(text, encoding='utf-8')
        print(text)
        exit_code = 1 if not passed else 0
    except Exception as exc:
        report.update(status='ERROR', error=f'{type(exc).__name__}: {exc}')
        print(report['error'], file=sys.stderr)
        exit_code = 1
    (args.output/'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f'Report: {args.output}')
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
