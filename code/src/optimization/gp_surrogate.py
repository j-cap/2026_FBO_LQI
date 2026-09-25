"""
Minimal GP + Expected-Improvement surrogate for the Phase-2 FBO comparison. No BoTorch
/GPyTorch in this environment (env_paper_FBO_LQI.yaml) - `sklearn.gaussian_process` is
sufficient at this scale (a handful to a few hundred pooled training points per round,
3-D input).

Per-point importance weighting (the mechanism that turns one GP implementation into all
four comparison methods - independent/global/similarity/recipient-aware differ only in
what weight vector they hand this module, see `src.optimization.fleet_bo`) is done via
sklearn's heteroscedastic `alpha` (per-sample noise variance): a low-weight point is
fit with high noise, so the posterior mean barely moves toward it and its uncertainty
contribution is suppressed - a standard, simple stand-in for a weighted-likelihood GP
that avoids hand-rolling one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

# Points with weight below this are dropped from the training set entirely rather than
# fit with an (numerically unstable) enormous alpha - they contribute ~nothing to the
# posterior either way.
WEIGHT_FLOOR = 0.02
# Noise variance for a full-weight (w=1) point; scaled by 1/w for w in (WEIGHT_FLOOR, 1].
BASE_ALPHA = 1.0e-4


@dataclass
class GPFitResult:
    gp: Optional[GaussianProcessRegressor]
    n_train: int


def fit_weighted_gp(
    X_norm: np.ndarray, y: np.ndarray, weights: np.ndarray, *, seed: int = 0
) -> GPFitResult:
    """X_norm in [0,1]^d. Returns gp=None if fewer than 2 points clear WEIGHT_FLOOR
    (too little to fit a meaningful surrogate; caller should fall back to random
    exploration)."""
    keep = weights >= WEIGHT_FLOOR
    if int(np.sum(keep)) < 2:
        return GPFitResult(gp=None, n_train=int(np.sum(keep)))

    Xk, yk, wk = X_norm[keep], y[keep], weights[keep]
    alpha = BASE_ALPHA / np.clip(wk, WEIGHT_FLOOR, 1.0)

    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
        length_scale=np.full(X_norm.shape[1], 0.3), length_scale_bounds=(0.05, 3.0), nu=2.5
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=2, random_state=seed
    )
    gp.fit(Xk, yk)
    return GPFitResult(gp=gp, n_train=int(np.sum(keep)))


def expected_improvement(
    gp: GaussianProcessRegressor, X_candidates: np.ndarray, best_f: float, *, xi: float = 0.01
) -> np.ndarray:
    """Minimization-EI over candidate points, given the best (lowest) objective value
    observed so far `best_f`. `xi` is a small exploration margin (standard BO
    convention), not related to the controller-weight vector also called xi elsewhere
    in this codebase - a candidate CALIBRATION VECTOR is what X_candidates rows are."""
    mu, sigma = gp.predict(X_candidates, return_std=True)
    sigma = np.clip(sigma, 1e-12, None)
    imp = best_f - mu - xi
    Z = imp / sigma
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    return np.clip(ei, 0.0, None)


def propose_next(
    X_train_norm: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    X_candidates_norm: np.ndarray,
    best_f: float,
    *,
    seed: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[int, GPFitResult]:
    """Returns the index (into X_candidates_norm) of the proposed next point, and the
    GPFitResult used to propose it (gp=None means the fallback path - a uniformly
    random candidate - was taken, e.g. before enough weighted data exists yet)."""
    fit = fit_weighted_gp(X_train_norm, y_train, weights, seed=seed)
    if fit.gp is None:
        rng = rng if rng is not None else np.random.default_rng(seed)
        return int(rng.integers(0, X_candidates_norm.shape[0])), fit
    ei = expected_improvement(fit.gp, X_candidates_norm, best_f)
    return int(np.argmax(ei)), fit
