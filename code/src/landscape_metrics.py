"""
Landscape similarity metrics between two clients' J(xi) evaluations over the SAME
candidate set Xi (doc §10.4) - feeds Figure B (doc §10.6): does uncertainty-aware model
distance predict landscape similarity?
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr, spearmanr


@dataclass
class LandscapePairMetrics:
    pearson_r: float
    spearman_r: float
    normalized_rmse: float
    optimum_distance: float
    n_common: int


def _normalize(J: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return (J - np.min(J)) / (np.std(J) + eps)


def compare_landscapes(
    J_i: np.ndarray, J_j: np.ndarray, xi_i_star: np.ndarray, xi_j_star: np.ndarray
) -> LandscapePairMetrics:
    """J_i, J_j must be aligned to the same candidate ordering; NaN entries (infeasible
    candidates for that client) are dropped pairwise before computing correlations."""
    J_i, J_j = np.asarray(J_i, float), np.asarray(J_j, float)
    mask = np.isfinite(J_i) & np.isfinite(J_j)
    n_common = int(np.sum(mask))
    if n_common < 8:
        return LandscapePairMetrics(np.nan, np.nan, np.nan, float(np.linalg.norm(xi_i_star - xi_j_star)), n_common)

    Ji, Jj = J_i[mask], J_j[mask]
    pearson_r = float(pearsonr(Ji, Jj)[0])
    spearman_r = float(spearmanr(Ji, Jj)[0])
    normalized_rmse = float(np.sqrt(np.mean((_normalize(Ji) - _normalize(Jj)) ** 2)))
    optimum_distance = float(np.linalg.norm(np.asarray(xi_i_star) - np.asarray(xi_j_star)))
    return LandscapePairMetrics(pearson_r, spearman_r, normalized_rmse, optimum_distance, n_common)
