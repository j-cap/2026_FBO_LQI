"""
Uncertainty-aware dynamics distance between identified client models (doc §1.4, §10.5).

Ports `pairwise_mahalanobis_D2` out of
`paper_UncertaintyAwareClustering/exp_02.ipynb` (cell 69) into a real, importable
function - same precision-averaged Mahalanobis form used there, not a
reimplementation - plus a plain Euclidean baseline (doc §10.5's "weaker baseline",
reused again in ablation A1).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def pairwise_uncertainty_aware_distance(
    thetas: Sequence[np.ndarray], precisions: Sequence[np.ndarray]
) -> np.ndarray:
    """
    D2[i,j] = (theta_i - theta_j)^T ((W_i + W_j)/2) (theta_i - theta_j), where W is the
    RLS precision (inverse covariance) - the exact form used inline in
    paper_UncertaintyAwareClustering/exp_02.ipynb's `pairwise_mahalanobis_D2`.
    """
    n = len(thetas)
    D2 = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.asarray(thetas[i]) - np.asarray(thetas[j])
            W_avg = 0.5 * (np.asarray(precisions[i]) + np.asarray(precisions[j]))
            d2 = float(diff @ W_avg @ diff)
            D2[i, j] = D2[j, i] = d2
    return D2


def pairwise_euclidean_distance(thetas: Sequence[np.ndarray]) -> np.ndarray:
    """Plain ||theta_i - theta_j||_2 baseline, ignoring identification uncertainty."""
    T = np.stack([np.asarray(t) for t in thetas])
    diffs = T[:, None, :] - T[None, :, :]
    return np.sqrt(np.sum(diffs**2, axis=-1))
