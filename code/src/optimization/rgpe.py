"""Ranking-Weighted Gaussian Process Ensemble (RGPE) transfer surrogate.

Implements the RGPE construction of Feurer, Letham, and Bakshy (2018) using
the sklearn GP stack already used by this project. It is kept separate from the
existing pooled warm-start methods so their frozen behavior remains unchanged.

Frozen comparison choices:
- one GP per completed source task plus one target GP;
- task-wise standardized outputs;
- posterior-sampled ranking-loss weights;
- target ranking loss from leave-one-out predictions using the fitted target
  hyperparameters;
- original RGPE weight-dilution guard: discard a source whose median loss is
  above the 95th percentile of target-model loss;
- target wins a minimum-loss tie; source-only ties use seeded random choice;
- ensemble moments sum w_i mu_i and w_i^2 sigma_i^2;
- same closed-form minimization EI and fixed Sobol candidate set as the
  existing pipeline.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from src.optimization.gp_surrogate import BASE_ALPHA


@dataclass
class RGPETaskModel:
    gp: GaussianProcessRegressor
    y_mean: float
    y_std: float
    task_id: str = ""


@dataclass
class RGPEWeightResult:
    weights: np.ndarray  # source models first, target model last
    active_source_mask: np.ndarray
    target_weight: float
    n_active_sources: int
    weight_entropy: float
    source_ranking_losses: np.ndarray
    target_ranking_losses: np.ndarray


@dataclass
class RGPEProposalResult:
    candidate_index: int
    weights: RGPEWeightResult
    target_model: RGPETaskModel


def _standardization_stats(y: np.ndarray) -> Tuple[float, float]:
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.size == 0:
        raise ValueError("cannot standardize an empty array")
    mean = float(np.mean(y))
    std = float(np.std(y))
    if not np.isfinite(std) or std < 1e-12:
        std = 1.0
    return mean, std


def _default_kernel(d: int):
    return ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
        length_scale=np.full(d, 0.3),
        length_scale_bounds=(0.05, 3.0),
        nu=2.5,
    )


def fit_rgpe_task_model(
    X_norm: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 0,
    task_id: str = "",
    y_mean: float | None = None,
    y_std: float | None = None,
    fixed_kernel=None,
) -> RGPETaskModel:
    """Fit one standardized task GP.

    If fixed_kernel is provided, its fitted hyperparameters are reused without
    re-optimization. This is used for target LOOCV models.
    """
    X = np.asarray(X_norm, dtype=float)
    yy = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or len(X) != len(yy) or len(yy) < 1:
        raise ValueError("RGPE task model requires X shape (n,d), y shape (n,), n>=1")

    if y_mean is None or y_std is None:
        y_mean, y_std = _standardization_stats(yy)
    y_std = float(y_std)
    if y_std < 1e-12:
        y_std = 1.0
    yz = (yy - float(y_mean)) / y_std

    if fixed_kernel is None:
        kernel = _default_kernel(X.shape[1])
        optimizer = "fmin_l_bfgs_b"
        n_restarts = 2
    else:
        kernel = copy.deepcopy(fixed_kernel)
        optimizer = None
        n_restarts = 0

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=BASE_ALPHA,
        normalize_y=False,
        optimizer=optimizer,
        n_restarts_optimizer=n_restarts,
        random_state=seed,
    )
    gp.fit(X, yz)
    return RGPETaskModel(
        gp=gp,
        y_mean=float(y_mean),
        y_std=y_std,
        task_id=task_id,
    )


def _safe_joint_samples(
    model: RGPETaskModel,
    X: np.ndarray,
    *,
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    mean, cov = model.gp.predict(np.asarray(X, float), return_cov=True)
    cov = np.asarray(cov, float)
    cov = 0.5 * (cov + cov.T)
    if cov.size:
        min_eig = float(np.min(np.linalg.eigvalsh(cov)))
        cov = cov + max(1e-10, -min_eig + 1e-10) * np.eye(cov.shape[0])
    return rng.multivariate_normal(
        np.asarray(mean, float),
        cov,
        size=int(num_samples),
        check_valid="ignore",
    )


def compute_ranking_loss(samples: np.ndarray, target_y: np.ndarray) -> np.ndarray:
    """Posterior-sampled pairwise ranking loss.

    Source samples have shape (S,n). Target LOOCV samples have shape (S,n,n),
    where axis 1 indexes the left-out model and axis 2 the target points.
    """
    s = np.asarray(samples)
    y = np.asarray(target_y, dtype=float).reshape(-1)
    n = len(y)
    truth = y[None, :, None] < y[None, None, :]

    if s.ndim == 2:
        if s.shape[1] != n:
            raise ValueError("source samples must have shape (S,n)")
        pred = s[:, :, None] < s[:, None, :]
    elif s.ndim == 3:
        if s.shape[1:] != (n, n):
            raise ValueError("target LOO samples must have shape (S,n,n)")
        left_out = np.diagonal(s, axis1=1, axis2=2)
        pred = left_out[:, :, None] < s
    else:
        raise ValueError("samples must be rank 2 or rank 3")

    return np.logical_xor(pred, truth).sum(axis=(1, 2)).astype(int)


def _target_loo_samples(
    X_target: np.ndarray,
    y_target: np.ndarray,
    target_model: RGPETaskModel,
    *,
    num_samples: int,
    rng: np.random.Generator,
    seed: int,
) -> np.ndarray:
    X = np.asarray(X_target, float)
    y = np.asarray(y_target, float).reshape(-1)
    n = len(y)
    if n < 2:
        raise ValueError("RGPE target ranking requires at least two observations")

    out = np.empty((num_samples, n, n), dtype=float)
    for j in range(n):
        keep = np.ones(n, dtype=bool)
        keep[j] = False
        loo = fit_rgpe_task_model(
            X[keep],
            y[keep],
            seed=seed + j + 1,
            task_id=f"target_loo_{j}",
            y_mean=target_model.y_mean,
            y_std=target_model.y_std,
            fixed_kernel=target_model.gp.kernel_,
        )
        out[:, j, :] = _safe_joint_samples(
            loo, X, num_samples=num_samples, rng=rng
        )
    return out


def rank_weights_from_losses(
    source_losses: np.ndarray,
    target_losses: np.ndarray,
    *,
    rng: np.random.Generator,
    prune_quantile: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert sampled losses to RGPE weights with original dilution control."""
    sl = np.asarray(source_losses, dtype=float)
    tl = np.asarray(target_losses, dtype=float).reshape(-1)
    if sl.ndim != 2:
        raise ValueError("source_losses must have shape (n_sources,S)")
    if sl.shape[1] != len(tl):
        raise ValueError("source and target losses need equal sample counts")
    if not (0.0 < prune_quantile <= 1.0):
        raise ValueError("prune_quantile must be in (0,1]")

    threshold = float(np.quantile(tl, prune_quantile))
    active = (
        np.median(sl, axis=1) <= threshold
        if sl.shape[0]
        else np.zeros(0, dtype=bool)
    )

    all_losses = np.vstack([sl, tl[None, :]])
    if len(active):
        inactive_idx = np.flatnonzero(~active)
        all_losses[inactive_idx, :] = np.inf
    target_idx = all_losses.shape[0] - 1

    wins = np.zeros(all_losses.shape[0], dtype=int)
    for sample_idx in range(all_losses.shape[1]):
        col = all_losses[:, sample_idx]
        best = np.min(col)
        ties = np.flatnonzero(col == best)
        if target_idx in ties:
            winner = target_idx
        else:
            winner = int(rng.choice(ties))
        wins[winner] += 1

    return wins.astype(float) / float(len(tl)), active


def compute_rgpe_weights(
    source_models: Sequence[RGPETaskModel],
    X_target: np.ndarray,
    y_target: np.ndarray,
    *,
    num_samples: int = 256,
    prune_quantile: float = 0.95,
    seed: int = 0,
) -> Tuple[RGPEWeightResult, RGPETaskModel]:
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    X = np.asarray(X_target, float)
    y = np.asarray(y_target, float).reshape(-1)
    if len(y) < 2:
        raise ValueError("RGPE requires at least two target observations")

    rng = np.random.default_rng(seed)
    target_model = fit_rgpe_task_model(X, y, seed=seed, task_id="target")

    source_loss_rows: List[np.ndarray] = []
    for model in source_models:
        samps = _safe_joint_samples(
            model, X, num_samples=num_samples, rng=rng
        )
        source_loss_rows.append(compute_ranking_loss(samps, y))
    source_losses = (
        np.vstack(source_loss_rows)
        if source_loss_rows
        else np.empty((0, num_samples), dtype=int)
    )

    target_samples = _target_loo_samples(
        X,
        y,
        target_model,
        num_samples=num_samples,
        rng=rng,
        seed=seed + 10_000,
    )
    target_losses = compute_ranking_loss(target_samples, y)

    weights, active = rank_weights_from_losses(
        source_losses,
        target_losses,
        rng=rng,
        prune_quantile=prune_quantile,
    )
    nz = weights[weights > 0]
    entropy = float(-np.sum(nz * np.log(nz))) if len(nz) else 0.0
    result = RGPEWeightResult(
        weights=weights,
        active_source_mask=active,
        target_weight=float(weights[-1]),
        n_active_sources=int(active.sum()),
        weight_entropy=entropy,
        source_ranking_losses=source_losses,
        target_ranking_losses=target_losses,
    )
    return result, target_model


def combine_rgpe_marginals(
    models: Sequence[RGPETaskModel],
    weights: np.ndarray,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    ww = np.asarray(weights, float).reshape(-1)
    if len(models) != len(ww):
        raise ValueError("number of models and weights must match")
    if not np.isclose(np.sum(ww), 1.0):
        raise ValueError("RGPE weights must sum to one")

    Xq = np.asarray(X, float)
    mean = np.zeros(len(Xq), dtype=float)
    var = np.zeros(len(Xq), dtype=float)
    for model, w in zip(models, ww):
        if w <= 0:
            continue
        mu, std = model.gp.predict(Xq, return_std=True)
        mean += w * np.asarray(mu, float)
        var += (w * np.asarray(std, float)) ** 2
    return mean, np.sqrt(np.clip(var, 1e-24, None))


def expected_improvement_from_moments(
    mean: np.ndarray,
    std: np.ndarray,
    best_f: float,
    *,
    xi: float = 0.01,
) -> np.ndarray:
    mu = np.asarray(mean, float)
    sigma = np.clip(np.asarray(std, float), 1e-12, None)
    imp = float(best_f) - mu - xi
    z = imp / sigma
    return np.clip(imp * norm.cdf(z) + sigma * norm.pdf(z), 0.0, None)


def propose_next_rgpe(
    source_models: Sequence[RGPETaskModel],
    X_target: np.ndarray,
    y_target: np.ndarray,
    X_candidates: np.ndarray,
    *,
    num_samples: int = 256,
    prune_quantile: float = 0.95,
    seed: int = 0,
) -> RGPEProposalResult:
    """Select next candidate from the existing fixed Sobol candidate set."""
    weight_result, target_model = compute_rgpe_weights(
        source_models,
        X_target,
        y_target,
        num_samples=num_samples,
        prune_quantile=prune_quantile,
        seed=seed,
    )
    models = list(source_models) + [target_model]
    mean, std = combine_rgpe_marginals(
        models, weight_result.weights, X_candidates
    )
    best_f = float(
        (np.min(y_target) - target_model.y_mean) / target_model.y_std
    )
    ei = expected_improvement_from_moments(mean, std, best_f)
    return RGPEProposalResult(
        candidate_index=int(np.argmax(ei)),
        weights=weight_result,
        target_model=target_model,
    )
