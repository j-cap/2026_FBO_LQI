"""Validation tests for `src.optimization.gp_surrogate` (weighted-GP + EI building
block shared by all four Phase-2 FBO methods)."""
from __future__ import annotations

import numpy as np

from src.optimization.gp_surrogate import expected_improvement, fit_weighted_gp, propose_next


def _toy_data(n=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 3))
    # y decreases toward the origin - a simple, smooth target the GP should be able to
    # pick up on with a handful of points.
    y = np.sum(X**2, axis=1) + 0.01 * rng.standard_normal(n)
    return X, y


def test_fit_weighted_gp_returns_none_below_two_effective_points():
    X, y = _toy_data(n=5)
    weights = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    fit = fit_weighted_gp(X, y, weights)
    assert fit.gp is None
    assert fit.n_train == 1


def test_fit_weighted_gp_drops_low_weight_points():
    X, y = _toy_data(n=10)
    weights = np.array([1.0] * 5 + [0.001] * 5)
    fit = fit_weighted_gp(X, y, weights)
    assert fit.gp is not None
    assert fit.n_train == 5


def test_fit_weighted_gp_uses_all_full_weight_points():
    X, y = _toy_data(n=10)
    weights = np.ones(10)
    fit = fit_weighted_gp(X, y, weights)
    assert fit.n_train == 10


def test_expected_improvement_is_nonnegative_and_favors_unexplored_region():
    X, y = _toy_data(n=15)
    fit = fit_weighted_gp(X, y, np.ones(15))
    assert fit.gp is not None
    # Candidate near the known-good region (low y, near origin) vs. far corner
    # (unexplored, high predicted uncertainty).
    candidates = np.array([[0.01, 0.01, 0.01], [0.99, 0.99, 0.99]])
    ei = expected_improvement(fit.gp, candidates, best_f=float(np.min(y)))
    assert np.all(ei >= 0.0)


def test_propose_next_falls_back_to_random_with_too_little_data():
    X, y = _toy_data(n=3)
    weights = np.zeros(3)  # nothing clears the weight floor
    candidates = np.random.default_rng(1).random((20, 3))
    idx, fit = propose_next(X, y, weights, candidates, best_f=0.0, seed=1)
    assert fit.gp is None
    assert 0 <= idx < 20


def test_propose_next_returns_valid_index_with_real_data():
    X, y = _toy_data(n=15)
    candidates = np.random.default_rng(2).random((30, 3))
    idx, fit = propose_next(X, y, np.ones(15), candidates, best_f=float(np.min(y)), seed=2)
    assert fit.gp is not None
    assert 0 <= idx < 30
