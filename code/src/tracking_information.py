"""Factorial attribution of historical information after catalogue initialization.

The frozen optimizer is reused verbatim. Only its two historical prior arrays
are masked; target learning and the common source-derived catalogue stay active.
"""
from copy import deepcopy

import numpy as np
import pandas as pd

from .tracking_budget import TargetOracle, run_method


ARMS = {
    'no_extra_prior': (False, False),
    'cost_only': (True, False),
    'feasibility_only': (False, True),
    'both_priors': (True, True),
}
METHODS = ['catalogue', *ARMS]


def masked_source(source, domain, arm):
    if arm not in ARMS:
        raise ValueError('Unknown information arm.')
    cost, feasibility = ARMS[arm]
    n = len(domain.actions)
    mu = list(source['prior_mean']) if cost else [0.]*n
    probability = list(source['prior_feasibility']) if feasibility else [.5]*n
    if len(mu) != n or len(probability) != n or not np.isfinite(mu).all() or not np.isfinite(probability).all():
        raise ValueError('Enabled historical channel is invalid.')
    if np.any(np.asarray(probability) < 0) or np.any(np.asarray(probability) > 1):
        raise ValueError('Feasibility prior is outside [0,1].')
    return dict(catalogue=deepcopy(source['catalogue']), prior_mean=mu,
                prior_feasibility=probability)


def run_suite(calibration, domain, target, source, cfg, seed, noise_seed):
    if cfg['methods'] != METHODS:
        raise ValueError('The catalogue and all four factorial arms are required in fixed order.')
    results = {'catalogue': run_method('catalogue', TargetOracle(calibration, target, noise_seed, domain),
                                      domain, source, cfg['gp'], seed, max(cfg['budgets']))}
    results['catalogue']['channels'] = None
    for arm, (cost, feasibility) in ARMS.items():
        result = run_method('pooled_bo', TargetOracle(calibration, target, noise_seed, domain),
                            domain, masked_source(source, domain, arm), cfg['gp'], seed,
                            max(cfg['budgets']))
        result['channels'] = dict(cost_prior=cost, feasibility_prior=feasibility)
        if result['steps'][:len(domain.contexts)] != results['catalogue']['steps']:
            raise RuntimeError('Factorial arms do not share the same initial target observations.')
        results[arm] = result
    return results


CONTRASTS = {
    'cost_with_feasibility_off': {'cost_only': 1., 'no_extra_prior': -1.},
    'cost_with_feasibility_on': {'both_priors': 1., 'feasibility_only': -1.},
    'feasibility_with_cost_off': {'feasibility_only': 1., 'no_extra_prior': -1.},
    'feasibility_with_cost_on': {'both_priors': 1., 'cost_only': -1.},
    'cost_main_effect': {'cost_only': .5, 'no_extra_prior': -.5,
                         'both_priors': .5, 'feasibility_only': -.5},
    'feasibility_main_effect': {'feasibility_only': .5, 'no_extra_prior': -.5,
                                'both_priors': .5, 'cost_only': -.5},
    'interaction': {'both_priors': 1., 'cost_only': -1.,
                     'feasibility_only': -1., 'no_extra_prior': 1.},
}
METRICS = {'success_fraction': 'higher', 'joint_success': 'higher',
           'fully_feasible': 'higher', 'failed_queries': 'lower', 'J': 'lower'}


def factorial_effects(frame):
    """Matched contrasts, keeping every trial for success and failure metrics.

    A cost contrast is available only when all policies entering that contrast
    have fully feasible recommendations. Its eligible denominator is retained.
    """
    keys = ['protocol', 'target', 'seed', 'budget']
    selected = frame[frame.method.isin(ARMS)].copy()
    if selected.duplicated(keys+['method']).any():
        raise ValueError('Duplicate factorial trial rows.')
    rows = []
    for identifiers, block in selected.groupby(keys, sort=True):
        if set(block.method) != set(ARMS):
            raise ValueError('Unmatched factorial arms.')
        values = block.set_index('method')
        common = dict(zip(keys, identifiers))
        for metric, direction in METRICS.items():
            for contrast, coefficients in CONTRASTS.items():
                included = list(coefficients)
                eligible = metric != 'J' or bool(values.loc[included, 'fully_feasible'].all())
                finite = values.loc[included, metric].notna().all()
                difference = (float(sum(weight*float(values.loc[arm, metric])
                                        for arm, weight in coefficients.items()))
                              if eligible and finite else None)
                rows.append(dict(**common, metric=metric, contrast=contrast,
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
    """Diagnostic against the old campaign; all new arms are rerun together."""
    rows = []
    for current, old in [('catalogue', 'catalogue'),
                         ('no_extra_prior', 'catalogue_warmstart_bo'),
                         ('both_priors', 'pooled_bo')]:
        a = job['methods'][current]['steps']
        b = previous['methods'][old]['steps'][:len(a)]
        rows.append(dict(target=job['target'], protocol=job['protocol'], seed=job['seed'],
                         method=current, compared_rollouts=len(a),
                         action_trace_match=[s['action'] for s in a] == [s['action'] for s in b],
                         recommendation_trace_match=[s['recommendations'] for s in a] ==
                         [s['recommendations'] for s in b]))
    return rows
