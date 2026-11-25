"""Read-only reconstruction and diagnostics of the frozen gain-domain audit."""
import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import tracking_gain_domain as runner

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.tracking_gain_domain import STAGES, choose_stages
from src.tracking_gain_domain_report import tables

OUT = runner.REPO / 'docs/analysis/gain_domain'
LABELS = dict(original='Original 71', dense_original='Dense original box', expanded='Expanded box')


def prohibited(*args, **kwargs):
    raise AssertionError('This audit must not simulate, optimize or build a spectral basis.')


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable_inputs(spec):
    """Only canonicalize path separators; retain the signed Windows manifest."""
    spec = deepcopy(spec)
    inputs = spec['source_inputs']
    spec['source_inputs'] = {name.replace('\\', '/'): value for name, value in inputs.items()}
    if len(spec['source_inputs']) != len(inputs):
        raise RuntimeError('Manifest contains colliding source paths.')
    return spec


def load_state():
    state = runner.setup()
    saved = runner.base.read(state['out'] / 'run_manifest.json')
    body = {k: v for k, v in saved.items() if k not in ('signature', 'git_head', 'runtime')}
    signature = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    if signature != saved['signature']:
        raise RuntimeError('Saved audit manifest signature differs.')
    if portable_inputs(state['specification']) != portable_inputs(body):
        raise RuntimeError('Saved design differs beyond source-path separators.')
    # One source_inputs key uses pathlib str(), hence backslashes on Windows.
    # No physical values, source hashes, seeds, rows or saved files are changed.
    state['signature'], state['specification'] = signature, body
    for name, expected in portable_inputs(body)['source_inputs'].items():
        if runner.base.digest(runner.REPO / name) != expected:
            raise RuntimeError('Source evaluator fingerprint differs: ' + name)
    return state


def annotated(rows, state, context, repeat, role):
    context_info = next(c for c in state['contexts'] if c['id'] == context)
    for row in rows:
        label = state['labels'][row['candidate']]
        group = 'original' if label['original'] else 'outside' if label['outside_original'] else 'inside_added'
        yield dict(context=context, speed=context_info['speed'], frequency=context_info['frequency'],
            repeat=repeat, role=role, group=group, **row)


def check_outcomes(frame, state):
    if frame.duplicated(['context', 'repeat', 'candidate']).any():
        raise RuntimeError('Repeated candidate/repeat identity.')
    complete = frame[frame.complete]
    score = state['physics']['score']
    np.testing.assert_allclose(complete.J, complete.rmse_y / score['lateral_scale_m'] +
        complete.rms_rate_deg_s / score['steer_rate_scale_deg_s'], rtol=2e-12, atol=2e-12)
    bounds = state['physics']['feasibility']
    feasible = frame.complete & (frame.peak_y <= bounds['peak_lateral_error_m']) & (
        frame.peak_beta_deg <= bounds['peak_beta_deg']) & (
        frame.peak_speed_error <= bounds['peak_speed_error_m_s'])
    np.testing.assert_array_equal(frame.feasible, feasible)
    if frame.loc[~frame.complete, 'J'].notna().any():
        raise RuntimeError('Incomplete rollout has a usable score.')
    finite_agreement = complete[complete.integration_resolved].integration_error_ratio
    if finite_agreement.isna().any() or (finite_agreement > 1).any():
        raise RuntimeError('Resolved complete rollout lacks integration agreement.')
    if not set(frame.substeps) <= set(state['physics']['integration']['substeps'][1:]):
        raise RuntimeError('Unexpected integration resolution.')
    frame['eligible'] = feasible & frame.integration_resolved


def aggregate(frame, repeats):
    groups = frame.groupby(['context', 'candidate'], sort=False)
    if not groups['repeat'].apply(lambda s: sorted(s) == list(range(repeats))).all():
        raise RuntimeError('Missing or duplicated declared repeat.')
    result = groups.agg(eligible=('eligible', 'all'), complete=('complete', 'all'),
        feasible=('feasible', 'all'), resolved=('integration_resolved', 'all'), J=('J', 'mean'))
    result.loc[~result.complete, 'J'] = np.nan
    return result.reset_index()


def independent_choices(aggregated, state):
    choices = {}
    for context, group in aggregated.groupby('context', sort=False):
        choices[context] = {}
        for stage in STAGES:
            eligible = group[group.eligible & group.candidate.isin(state['members'][stage])]
            if eligible.empty:
                choices[context][stage] = None
                continue
            best = eligible.J.min()
            tied = eligible[eligible.J <= best + 1e-12 * max(1., abs(best))]
            choices[context][stage] = min(tied.candidate)
    return choices


def verify_table(state, name, frame, keys):
    expected = pd.read_csv(io.StringIO(frame.to_csv(index=False)))
    actual = pd.read_csv(state['out'] / (name + '.csv'))
    if expected.duplicated(keys).any() or actual.duplicated(keys).any():
        raise RuntimeError('Repeated table key: ' + name)
    expected, actual = (f.sort_values(keys).reset_index(drop=True) for f in (expected, actual))
    assert_frame_equal(expected, actual, check_dtype=False, check_like=True, rtol=2e-10, atol=2e-10)
    return dict(table=name, rows=len(frame), verified=True)


def reconstruct(state):
    parity = runner.verify_source_checks(state)
    calibration = runner.load_calibration(state)
    choices = choose_stages(calibration, state['members'])
    validation, validation_tasks, buckets = runner.load_validation(state, choices)
    selected, comparisons, summary, profiles = tables(state, calibration, choices, validation, buckets)
    rows = []
    for (context, repeat), batch in state['source_batches'].items():
        rows.extend(annotated(batch['rows'], state, context, repeat, 'calibration'))
    for task in runner.tasks(state, 'calibration'):
        rows.extend(annotated(runner.load_batch(state, task)['rows'], state, task['context'], task['repeat'], 'calibration'))
    raw_calibration = pd.DataFrame(rows)
    raw_validation = pd.DataFrame([row for context, batches in buckets.items() for batch in batches
        for row in annotated(batch['rows'], state, context, batch['repeat'], 'validation')])
    for frame in (raw_calibration, raw_validation):
        check_outcomes(frame, state)
    aggregated = aggregate(raw_calibration, state['requested']['calibration_repeats'])
    if independent_choices(aggregated, state) != choices:
        raise RuntimeError('Independent calibration-only choices differ.')
    if len(aggregated) != len(state['contexts']) * len(state['pool']):
        raise RuntimeError('Calibration bank is incomplete.')
    for row in aggregated.itertuples():
        original = calibration[row.context][row.candidate]
        if bool(row.eligible) != original['eligible']:
            raise RuntimeError('Independent calibration eligibility differs.')
        if original['J'] is None:
            if pd.notna(row.J):
                raise RuntimeError('Independent incomplete score differs.')
        else:
            np.testing.assert_allclose(row.J, original['J'], rtol=2e-12, atol=2e-12)
    heldout = aggregate(raw_validation, len(state['validation_seeds'])).set_index(['context', 'candidate'])
    for row in selected.itertuples():
        fresh = heldout.loc[(row.context, row.candidate)]
        if bool(fresh.eligible) != row.validation_eligible:
            raise RuntimeError('Independent validation eligibility differs.')
        if fresh.eligible:
            np.testing.assert_allclose(fresh.J, row.validation_J, rtol=2e-12, atol=2e-12)
    availability = pd.DataFrame([dict(context=cid, stage=stage, candidates=len(ids),
        calibration_eligible=sum(rows[k]['eligible'] for k in ids))
        for cid, rows in calibration.items() for stage, ids in state['members'].items()])
    pool = pd.DataFrame([dict(candidate=c['id'], origin=state['labels'][c['id']]['origin'],
        original=state['labels'][c['id']]['original'], dense_original=state['labels'][c['id']]['dense_original'],
        outside_original=state['labels'][c['id']]['outside_original'],
        K_y=c['gains'][0], K_psi=c['gains'][1], K_r=c['gains'][2],
        multiplier_y=c['multipliers'][0], multiplier_psi=c['multipliers'][1], multiplier_r=c['multipliers'][2]) for c in state['pool']])
    frames = dict(selected_controllers=selected, paired_improvements=comparisons, stage_summary=summary,
        boundary_profiles=profiles, calibration_availability=availability, candidate_pool=pool)
    keys = dict(selected_controllers=['context', 'stage'], paired_improvements=['context', 'effect'],
        stage_summary=['stage'], boundary_profiles=['context', 'coordinate', 'candidate'],
        calibration_availability=['context', 'stage'], candidate_pool=['candidate'])
    checks = [verify_table(state, name, frame, keys[name]) for name, frame in frames.items()]
    expected_tasks = runner.tasks(state, 'source_check') + runner.tasks(state, 'calibration') + validation_tasks
    expected_files = {runner.job_path(state, task) for task in expected_tasks}
    actual_files = set((state['out'] / 'batches').rglob('*.json'))
    if actual_files != expected_files:
        raise RuntimeError('Missing or unexpected raw batches.')
    if runner.base.read(state['out'] / 'completion.json')['signature'] != state['signature']:
        raise RuntimeError('Completed run signature differs.')
    verification = runner.base.read(state['out'] / 'verification.json')
    expected_verification = dict(signature=state['signature'], original_candidates_preserved=True,
        source_check_rollouts=parity, source_inputs_unchanged=True,
        new_calibration_rollouts=int((raw_calibration.group != 'original').sum()),
        validation_rollouts=len(raw_validation), new_online_requests=0,
        spectral_basis_constructed=False, validation_selection_from_calibration_only=True, mode='audit')
    if verification != expected_verification or state['pool'][:71] != state['source']['pool']:
        raise RuntimeError('Saved verification or original candidate preservation differs.')
    return frames, raw_calibration, raw_validation, aggregated, checks, parity


def diagnostics(state, frames, raw, validation, aggregated, out):
    selected, pairs = frames['selected_controllers'], frames['paired_improvements']
    frequencies = []
    for frequency, group in selected.groupby('frequency', sort=True):
        means = group.groupby('stage').validation_J.mean()
        frequencies.append(dict(frequency=frequency, contexts=group.context.nunique(),
            **{stage + '_mean_J': means[stage] for stage in STAGES},
            density_reduction_percent=100 * (1 - means['dense_original'] / means['original']),
            extension_reduction_percent=100 * (1 - means['expanded'] / means['dense_original']),
            total_reduction_percent=100 * (1 - means['expanded'] / means['original'])))
    frequency = pd.DataFrame(frequencies)
    frequency.to_csv(out / 'frequency_costs.csv', index=False)
    effects = pairs.groupby(['effect', 'frequency'], sort=False).agg(contexts=('context', 'size'),
        mean_context_reduction_percent=('validation_improvement_percent', 'mean'),
        improved=('validation_improvement_percent', lambda v: int((v > 1e-10).sum())),
        worsened=('validation_improvement_percent', lambda v: int((v < -1e-10).sum())),
        materially_improved=('material_improvement', 'sum'),
        positive_repeats=('positive_validation_repeats', 'sum'),
        negative_repeats=('negative_validation_repeats', 'sum')).reset_index()
    effects.to_csv(out / 'frequency_effects.csv', index=False)
    raw.groupby(['group', 'frequency'], sort=False).agg(rollouts=('candidate', 'size'),
        eligible=('eligible', 'sum'), incomplete=('complete', lambda v: int((~v).sum())),
        task_infeasible=('feasible', lambda v: int((~v).sum())),
        unresolved=('integration_resolved', lambda v: int((~v).sum()))).reset_index().to_csv(out / 'calibration_status.csv', index=False)
    raw[~raw.eligible].groupby(['group', 'failure', 'complete', 'feasible', 'integration_resolved'],
        dropna=False).size().rename('rollouts').reset_index().to_csv(out / 'calibration_failures.csv', index=False)
    profile_ids = {p['candidate_id'] for p in state['profiles']} | {p['candidate'] for p in state['profiles']}
    validation[validation.candidate.isin(profile_ids)].to_csv(out / 'profile_validation_repeats.csv', index=False)
    components = []
    for row in selected.itertuples():
        data = validation[(validation.context == row.context) & (validation.candidate == row.candidate)]
        components.append(dict(context=row.context, speed=row.speed, frequency=row.frequency, stage=row.stage,
            candidate=row.candidate, rmse_y=data.rmse_y.mean(), rms_rate_deg_s=data.rms_rate_deg_s.mean(),
            tracking_cost=data.rmse_y.mean() / state['physics']['score']['lateral_scale_m'],
            rate_cost=data.rms_rate_deg_s.mean() / state['physics']['score']['steer_rate_scale_deg_s'],
            maximum_feedback_clip_fraction=data.fb_clip_fraction.max(),
            maximum_total_clip_fraction=data.total_clip_fraction.max(),
            maximum_rate_clip_fraction=data.rate_clip_fraction.max()))
    components = pd.DataFrame(components)
    components.to_csv(out / 'selected_cost_components.csv', index=False)
    coordinates = {c['id']: ','.join(name for name, value in zip(['K_y', 'K_psi', 'K_r'], c['multipliers']) if value > 10)
        or 'inside' for c in state['pool']}
    geometry = aggregated.copy()
    geometry['outside_coordinates'] = geometry.candidate.map(coordinates)
    geometry.groupby('outside_coordinates', sort=True).agg(candidate_context_pairs=('candidate', 'size'),
        eligible_pairs=('eligible', 'sum')).reset_index().to_csv(out / 'outside_coordinate_eligibility.csv', index=False)
    winners = selected[selected.stage == 'expanded'].set_index('context')
    unresolved = aggregated[aggregated.complete & aggregated.feasible & ~aggregated.resolved].copy()
    unresolved['reference_J'] = unresolved.context.map(winners.calibration_J)
    unresolved['last_grid_gap_percent'] = 100 * (unresolved.J / unresolved.reference_J - 1)
    unresolved.groupby('context').agg(candidate_context_pairs=('candidate', 'size'),
        lower_last_grid_cost_than_resolved_reference=('last_grid_gap_percent', lambda v: int((v < 0).sum())),
        minimum_last_grid_gap_percent=('last_grid_gap_percent', 'min')).reset_index().to_csv(out / 'unresolved_cost_diagnostic.csv', index=False)
    # Exploratory calibration-set density curve. Held-out validation is available
    # only for the prespecified stage winners, not for these intermediate sets.
    dense_sobol = [c['id'] for c in state['pool'] if c['kind'] in ('sobol', 'dense_sobol')]
    anchors = [c['id'] for c in state['source']['pool'] if c['kind'] != 'sobol']
    resolutions = []
    for count in [64, 128, 256, 512, 1024]:
        members = set(dense_sobol[:count] + anchors)
        for context, group in aggregated.groupby('context', sort=False):
            good = group[group.eligible & group.candidate.isin(members)]
            resolutions.append(dict(context=context, sobol_points=count, candidates=len(members),
                calibration_J=good.J.min() if len(good) else None))
    pd.DataFrame(resolutions).to_csv(out / 'calibration_resolution_curve.csv', index=False)
    means = selected.groupby('stage').validation_J.mean()
    result = dict(mean_validation_J={stage: float(means[stage]) for stage in STAGES},
        density_reduction_in_mean_J_percent=100 * (1 - means['dense_original'] / means['original']),
        extension_reduction_in_mean_J_percent=100 * (1 - means['expanded'] / means['dense_original']),
        total_reduction_in_mean_J_percent=100 * (1 - means['expanded'] / means['original']),
        calibration_unresolved=int((~raw.integration_resolved).sum()),
        unresolved_task_feasible_pairs=len(unresolved),
        unresolved_task_feasible_pairs_with_lower_last_grid_J=int((unresolved.last_grid_gap_percent < 0).sum()),
        validation_ineligible_rollouts=int((~validation.eligible).sum()),
        selected_validation_ineligible=int((~selected.validation_eligible).sum()),
        selected_outside_original=int(winners.outside_original.sum()),
        selected_above_old_upper={name: int((winners[name] > limit).sum())
            for name, limit in zip(['K_y', 'K_psi', 'K_r'], [.05, 1., 1.])},
        selected_actual_gain_ranges={name: [float(winners[name].min()), float(winners[name].max())]
            for name in ['K_y', 'K_psi', 'K_r']})
    runner.base.save(out / 'readout.json', result)
    return frequency, effects, components


def figures(frames, frequency, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt'] = 'tracking_gain_domain_v1'
    colors = ['#65758b', '#3b83bd', '#007d75']
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4), constrained_layout=True)
    x = np.arange(len(frequency))
    for i, (stage, color) in enumerate(zip(STAGES, colors)):
        axes[0].bar(x + (i - 1) * .24, frequency[stage + '_mean_J'], width=.24,
            label=LABELS[stage], color=color)
    axes[0].set(xticks=x, xticklabels=[f'{f:g}' for f in frequency.frequency],
        xlabel='Maneuver frequency [Hz]', ylabel='Mean held-out J', title='Calibration-selected controllers')
    axes[0].legend(fontsize=8)
    pairs = frames['paired_improvements']
    for effect, color, marker in [('density', colors[1], 'o'), ('extension', colors[2], 's')]:
        for j, frequency_hz in enumerate(frequency.frequency):
            subset = pairs[(pairs.effect == effect) & (pairs.frequency == frequency_hz)].sort_values('speed')
            offset = -.06 if effect == 'density' else .06
            axes[1].scatter(np.full(len(subset), j) + np.linspace(-.035, .035, len(subset)) + offset,
                subset.validation_improvement_percent, color=color, marker=marker, s=24,
                label=('Denser original box' if effect == 'density' else 'Wider box') if j == 0 else None)
    axes[1].legend(fontsize=8)
    axes[1].axhline(5, color='#7a7a7a', linestyle='--', linewidth=.9)
    axes[1].axhline(0, color='#7a7a7a', linewidth=.7)
    axes[1].set(xticks=x, xticklabels=[f'{f:g}' for f in frequency.frequency],
        xlabel='Maneuver frequency [Hz]', ylabel='Held-out J reduction [%]',
        title='One point per speed; dashed line = 5%')
    for ax in axes:
        ax.grid(axis='y', alpha=.17)
        ax.set_axisbelow(True)
    path = out / 'density_extension.svg'
    fig.savefig(path, metadata={'Date': None})
    path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    with patch.object(runner, 'resolved_rollout', prohibited), patch.object(runner.base, 'basis_bank', prohibited):
        state = load_state()
        if args.out.resolve().is_relative_to(state['out'].resolve()):
            raise RuntimeError('Analysis output must be outside the frozen source result directory.')
        input_paths = sorted(p for p in state['out'].rglob('*') if p.is_file())
        input_paths += sorted(runner.REPO / name for name in portable_inputs(state['specification'])['source_inputs'])
        hashes = {p.relative_to(runner.REPO).as_posix(): fingerprint(p) for p in input_paths}
        frames, raw, validation, aggregated, checks, parity = reconstruct(state)
        args.out.mkdir(parents=True, exist_ok=True)
        frequency, effects, components = diagnostics(state, frames, raw, validation, aggregated, args.out)
        figures(frames, frequency, args.out)
        if any(fingerprint(runner.REPO / name) != expected for name, expected in hashes.items()):
            raise RuntimeError('Source result changed during read-only analysis.')
        runner.base.save(args.out / 'verification.json', dict(source_signature=state['signature'],
            source_results_commit='ff83d777c7697850cd7c1ec1a0cda72ec85e10d7', verified_tables=checks,
            calibration_observations=len(raw), reused_calibration_observations=len(state['source_batches']) * 71,
            new_calibration_observations=int((raw.group != 'original').sum()),
            source_check_observations=parity, validation_observations=len(validation),
            independent_calibration_aggregation_and_selection=True,
            independent_heldout_eligibility_and_J=True, source_files_unchanged=True,
            path_portability='Signed source_inputs keys canonicalized for comparison only; raw manifest retained.',
            new_simulations=0, new_optimization=0, spectral_basis_constructed=False, input_fingerprints=hashes))
    print(f'Verified {len(raw):,} calibration, {parity} source-check and {len(validation)} validation observations; all {len(checks)} saved tables match.')
    print(frequency.round(6).to_string(index=False))
    print('Analysis: ' + str(args.out))


if __name__ == '__main__':
    main()
