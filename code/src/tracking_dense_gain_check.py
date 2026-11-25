"""Synthetic, outcome-independent checks of the exact dictionary posterior."""
from copy import deepcopy

import numpy as np

from .tracking_budget import matern
from .tracking_dense_gain import ExactCostSnapshot
from .tracking_kernel_features import query_posterior


def representation_check(domain, cfg, seed=1732, probe_count=512):
    rng = np.random.default_rng(seed)
    probes = np.unique(np.r_[rng.choice(len(domain.actions), min(probe_count, len(domain.actions)), replace=False),
        [domain.action(c, domain.candidates[0]) for c in domain.contexts],
        [domain.action(c, domain.candidates[-1]) for c in domain.contexts]])
    checks = []
    for count in (0, 1, 4, 24):
        histories = []
        for client in range(9):
            ids = rng.choice(len(domain.actions), count, replace=False).tolist()
            if count:
                ids[0] = 0  # repeated input across clients exercises shared/local covariance.
                ids = list(dict.fromkeys(ids))
            histories.append([dict(action=a, eligible=True, J=float(np.exp(.2*np.sin(a + client)))) for a in ids])
        indices = sorted(set(probes) | {o['action'] for h in histories for o in h})
        positions = {int(a): j for j, a in enumerate(indices)}
        kernel = matern(domain.X[indices], domain.X[indices], cfg['gp']['length_scales'],
            cfg['gp']['amplitude'], cfg['gp']['constant_variance'])
        local_histories = [[dict(o, action=positions[o['action']]) for o in h] for h in histories]
        for shared in (True, False):
            model = ExactCostSnapshot(histories, domain, cfg['gp'], cfg['hierarchical'], shared)
            maximum_mean = maximum_variance = maximum_weight = 0.
            for client in range(9):
                control = cfg['hierarchical'] if shared else dict(sharing_fractions=[0.], prior_weights=[1.])
                expected = query_posterior(local_histories, kernel, control, cfg['gp'],
                    np.full(len(probes), client), [positions[int(a)] for a in probes])
                mean, variance = model.predict(client, probes)
                np.testing.assert_allclose(mean, expected['mean'], rtol=2e-8, atol=2e-8)
                np.testing.assert_allclose(variance, expected['variance'], rtol=2e-8, atol=2e-8)
                np.testing.assert_allclose(model.weights, expected['sharing_weights'], rtol=2e-8, atol=2e-8)
                maximum_mean = max(maximum_mean, float(np.max(abs(mean - expected['mean']))))
                maximum_variance = max(maximum_variance, float(np.max(abs(variance - expected['variance']))))
                maximum_weight = max(maximum_weight, float(np.max(abs(model.weights - expected['sharing_weights']))))
            checks.append(dict(eligible_per_client=count, shared=shared, queried_clients=9,
                probe_inputs=len(probes), maximum_absolute_mean_error=maximum_mean,
                maximum_absolute_variance_error=maximum_variance, maximum_absolute_weight_error=maximum_weight))
    return dict(representation='exact_active_input_dictionary', synthetic_labels_only=True,
        probes_from_exact_public_grid=True, full_public_square_kernel_constructed=False,
        exact_conditional_variance_retained=True, all_checks_passed=True, checks=checks,
        kernel_approximation='None; finite observed-support whitening plus the exact conditional GP remainder.')
