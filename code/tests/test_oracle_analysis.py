"""
Regression test for `bootstrap_spearman_ci`'s NaN handling: the full-scale
phase1a_matched_model_full run's transfer-loss matrix (many missing entries from
infeasible transplant evaluations) silently produced rho=nan for every bootstrap draw
before this fix, because scipy's spearmanr returns nan for a whole sample if even one
pair is nan.
"""
from __future__ import annotations

import numpy as np

from src.oracle_analysis import bootstrap_spearman_ci


def test_bootstrap_spearman_ci_ignores_nan_entries():
    rng = np.random.default_rng(0)
    N = 12
    D = rng.uniform(0, 10, size=(N, N))
    D = (D + D.T) / 2
    S = D + rng.normal(scale=0.5, size=(N, N))  # strongly correlated with D
    S = (S + S.T) / 2
    np.fill_diagonal(D, 0.0)
    np.fill_diagonal(S, 0.0)

    # Sprinkle NaNs into S, similar to a sparse transfer-loss matrix - most pairs missing.
    S_sparse = S.copy()
    mask = rng.uniform(size=(N, N)) < 0.85
    mask = np.triu(mask, k=1)
    mask = mask | mask.T
    S_sparse[mask] = np.nan

    median_rho, ci = bootstrap_spearman_ci(D, S_sparse, n_boot=200, seed=1)

    assert np.isfinite(median_rho), "bootstrap should recover a finite rho despite sparse NaN entries"
    assert median_rho > 0, "S was constructed to positively correlate with D"
    assert np.isfinite(ci[0]) and np.isfinite(ci[1])


def test_bootstrap_spearman_ci_returns_nan_when_too_sparse():
    N = 6
    D = np.ones((N, N))
    S = np.full((N, N), np.nan)  # nothing usable at all
    median_rho, ci = bootstrap_spearman_ci(D, S, n_boot=50, seed=0)
    assert np.isnan(median_rho)
    assert np.isnan(ci[0]) and np.isnan(ci[1])
