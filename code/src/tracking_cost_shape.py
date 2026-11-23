"""Separate context-specific historical cost level from gain dependence.

The frozen optimizer runs unchanged. Only its log-cost prior mean is replaced;
all BO arms use neutral historical feasibility rates and learn target labels.
"""
from copy import deepcopy

import numpy as np
import pandas as pd

from .tracking_budget import TargetOracle, run_method


ARMS = {'no_extra_prior': 'zero', 'context_level': 'context_mean',
        'full_cost_prior': 'full'}
METHODS = ['catalogue', *ARMS]
PRIOR_DEFINITION = 'uniform_context_mean_of_frozen_log_J_prior'
CONTRASTS = {
    'level_minus_none': {'context_level': 1., 'no_extra_prior': -1.},
    'full_minus_level': {'full_cost_prior': 1., 'context_level': -1.},
    'full_minus_none': {'full_cost_prior': 1., 'no_extra_prior': -1.},
}
METRICS = {'success_fraction': 'higher', 'joint_success': 'higher',
           'fully_feasible': 'higher', 'failed_queries': 'lower', 'J': 'lower'}


def source_for_arm(source, domain, arm):
    if arm not in ARMS:
        raise ValueError('Unknown cost-prior arm.')
    n = len(domain.actions)
    mode = ARMS[arm]
    mu = np.zeros(n)
    if mode != 'zero':
        mu = np.asarray(source['prior_mean'], dtype=float).copy()
        if mu.shape != (n,) or not np.isfinite(mu).all():
            raise ValueError('Invalid frozen historical cost mean.')
        if mode == 'context_mean':
            for context in domain.contexts:
                indices = [i for i, (c, _) in enumerate(domain.actions) if c == context]
                if not indices:
                    raise ValueError('Context has no gain candidates.')
                # Uniform over the fixed candidate pool, in log(J) space.
                # No target observations, eligibility weights or minima enter.
                mu[indices] = float(np.mean(mu[indices]))
    return dict(catalogue=deepcopy(source['catalogue']), prior_mean=mu.tolist(),
                prior_feasibility=[.5]*n)


def run_suite(calibration, domain, target, source, cfg, seed, noise_seed):
    if cfg['methods'] != METHODS:
        raise ValueError('The catalogue and three prescribed cost-prior arms are required.')
    results = {'catalogue': run_method('catalogue', TargetOracle(calibration, target, noise_seed, domain),
                                      domain, source, cfg['gp'], seed, max(cfg['budgets']))}
    results['catalogue']['prior_mode'] = None
    for arm, mode in ARMS.items():
        result = run_method('pooled_bo', TargetOracle(calibration, target, noise_seed, domain),
                            domain, source_for_arm(source, domain, arm), cfg['gp'], seed,
                            max(cfg['budgets']))
        result.update(prior_mode=mode, historical_feasibility_prior=False)
        if result['steps'][:len(domain.contexts)] != results['catalogue']['steps']:
            raise RuntimeError('Cost-prior arms must share the initial target observations.')
        results[arm] = result
    return results


def paired_effects(frame):
    """Matched differences, with an explicit feasible denominator for each J pair."""
    keys = ['protocol', 'target', 'seed', 'budget']
    selected = frame[frame.method.isin(ARMS)].copy()
    if selected.duplicated(keys+['method']).any():
        raise ValueError('Duplicate cost-prior trial rows.')
    rows = []
    for identifiers, block in selected.groupby(keys, sort=True):
        if set(block.method) != set(ARMS):
            raise ValueError('Unmatched cost-prior arms.')
        values = block.set_index('method')
        for metric, direction in METRICS.items():
            for contrast, coefficients in CONTRASTS.items():
                included = list(coefficients)
                eligible = metric != 'J' or bool(values.loc[included, 'fully_feasible'].all())
                finite = values.loc[included, metric].notna().all()
                difference = (float(sum(weight*float(values.loc[arm, metric])
                                        for arm, weight in coefficients.items()))
                              if eligible and finite else None)
                rows.append(dict(zip(keys, identifiers), metric=metric, contrast=contrast,
                                 difference=difference, better_direction=direction))
    trials = pd.DataFrame(rows)
    summary = trials.groupby(['protocol', 'budget', 'metric', 'contrast', 'better_direction'], sort=False).agg(
        matched_trials=('difference', 'size'), eligible_trials=('difference', 'count'),
        mean_difference=('difference', 'mean')).reset_index()
    per_vehicle = trials.groupby(['protocol', 'target', 'budget', 'metric', 'contrast'], sort=False).agg(
        matched_trials=('difference', 'size'), eligible_trials=('difference', 'count'),
        mean_difference=('difference', 'mean')).reset_index()
    return trials, summary, per_vehicle


def control_comparison(job, previous):
    rows = []
    for current, old in [('catalogue', 'catalogue'), ('no_extra_prior', 'no_extra_prior'),
                         ('full_cost_prior', 'cost_only')]:
        a = job['methods'][current]['steps']
        b = previous['methods'][old]['steps'][:len(a)]
        rows.append(dict(target=job['target'], protocol=job['protocol'], seed=job['seed'],
                         method=current, compared_rollouts=len(a),
                         action_trace_match=[s['action'] for s in a] == [s['action'] for s in b],
                         recommendation_trace_match=[s['recommendations'] for s in a] ==
                         [s['recommendations'] for s in b]))
    return rows
