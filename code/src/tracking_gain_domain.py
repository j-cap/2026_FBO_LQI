"""Physical gain-domain audit without BO or a candidate-grid kernel matrix."""
from copy import deepcopy

import numpy as np
from scipy.stats import qmc

from .tracking_confirmation import calibration_reference

STAGES = ['original', 'dense_original', 'expanded']


def build_pool(original, gain_cfg, requested, smoke=False):
    """Keep the frozen 71 vectors, densify their box, then add exterior points.

    Profiles are retrospective diagnostic probes of the two original sampled
    boundary winners. They are not a data-independent online feature design.
    """
    nominal = np.asarray(gain_cfg['nominal'], float)
    lo, hi = gain_cfg['log10_multiplier_bounds']
    upper = float(requested['expanded_multiplier_upper'])
    if (lo, hi) != (-1., 1.) or upper <= 10**hi:
        raise ValueError('Expected the frozen original box and a genuine upper extension.')
    settings = requested['smoke'] if smoke else requested
    dense_power, expanded_power = settings['dense_sobol_power'], settings['expanded_sobol_power']
    if dense_power <= gain_cfg['sobol_power']:
        raise ValueError('The dense design must add points to the original Sobol prefix.')
    samples = qmc.Sobol(3, scramble=True, seed=gain_cfg['sobol_seed']).random_base2(dense_power)
    multipliers = 10**(lo + (hi - lo) * samples)
    original_sobol = [c for c in original if c['kind'] == 'sobol']
    if len(original_sobol) != 2**gain_cfg['sobol_power']:
        raise ValueError('Frozen Sobol prefix count differs.')
    np.testing.assert_allclose(multipliers[:len(original_sobol)],
        [c['multipliers'] for c in original_sobol], rtol=1e-13, atol=1e-14)
    pool = deepcopy(original)
    index = {tuple(np.round(c['multipliers'], 12)): c['id'] for c in pool}
    metadata = {c['id']: dict(origin='original', original=True, dense_original=True, expanded=True) for c in pool}
    def add(mult, origin, dense, gains=None):
        mult = np.asarray(mult, float)
        key = tuple(np.round(mult, 12))
        if key in index:
            return index[key]
        candidate = f'g{len(pool):03d}'
        pool.append(dict(id=candidate, multipliers=mult.tolist(),
            gains=(nominal * mult).tolist() if gains is None else list(gains), kind=origin))
        metadata[candidate] = dict(origin=origin, original=False, dense_original=bool(dense), expanded=True)
        index[key] = candidate
        return candidate
    for mult in multipliers[len(original_sobol):]:
        add(mult, 'dense_sobol', True)
    lookup = {c['id']: c for c in original}
    profiles = []
    # Include the old upper face in the dense-original stage, so its benefit
    # cannot be mislabeled as an improvement from expanding the domain.
    for source in requested['profile_sources']:
        candidate, j = lookup[source['candidate']], source['coordinate']
        if not 0 <= j < 3:
            raise ValueError('Invalid gain coordinate.')
        for value in requested['profile_values']:
            if value < 10**hi or value > upper:
                raise ValueError('Profile values must start at or beyond the original upper face.')
            mult, gains = candidate['multipliers'].copy(), candidate['gains'].copy()
            mult[j], gains[j] = float(value), float(nominal[j] * value)
            dense = value <= 10**hi
            cid = add(mult, 'old_upper_probe' if dense else 'expanded_upper_probe', dense, gains)
            profiles.append(dict(**source, candidate_id=cid, multiplier=float(value),
                actual_gain=float(gains[j]), outside_original=not dense))
    expanded_u = qmc.Sobol(3, scramble=True, seed=requested['expanded_sobol_seed']).random_base2(expanded_power)
    expanded = 10**(lo + (np.log10(upper) - lo) * expanded_u)
    outside = np.any(expanded > 10**hi, axis=1)
    for mult in expanded[outside]:
        add(mult, 'expanded_sobol', False)
    for value in requested['expanded_scaling_anchors']:
        if not 10**hi < value <= upper:
            raise ValueError('Expanded anchors must be outside the original domain.')
        add([value] * 3, 'expanded_scaling', False)
    for candidate in pool:
        row = metadata[candidate['id']]
        z = np.log10(candidate['multipliers'])
        actual_outside = bool(np.any((z < lo - 1e-12) | (z > hi + 1e-12)))
        if actual_outside == row['dense_original']:
            raise RuntimeError('Domain membership is inconsistent with physical gains.')
        row.update(outside_original=actual_outside,
            original_normalized_coordinates=((z - lo) / (hi - lo)).tolist())
    return pool, metadata, profiles


def stage_candidates(pool, metadata):
    return {stage: [c['id'] for c in pool if metadata[c['id']][stage]] for stage in STAGES}


def choose_stages(calibration, members):
    return {context: {stage: calibration_reference({c: rows[c] for c in candidates})
        for stage, candidates in members.items()} for context, rows in calibration.items()}


def validation_selection(choices, profiles, contexts):
    selection = {c['id']: {v for v in choices[c['id']].values() if v is not None} for c in contexts}
    for context in contexts:
        for profile in profiles:
            if context['frequency'] == profile['frequency_hz']:
                selection[context['id']].update([profile['candidate'], profile['candidate_id']])
    return selection


def boundary_flags(multipliers, lo, hi, fraction):
    z = np.log10(np.asarray(multipliers, float))
    return dict(near_lower_boundary=bool(np.any((z - lo) / (hi - lo) <= fraction)),
        near_upper_boundary=bool(np.any((hi - z) / (hi - lo) <= fraction)))
