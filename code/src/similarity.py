"""
Symmetric dynamics-similarity prior s_ij for the Phase-2 FBO comparison (Next-steps
item 4 / Phase 1D "Layer 1"): s_ij = exp(-(d_ij^dyn)^2 / (2*l_d^2)), where d_ij^dyn is
the uncertainty-aware Mahalanobis distance (`src.dynamics_distance
.pairwise_uncertainty_aware_distance`, which already returns SQUARED distance D2) and
l_d is set via the standard median-distance heuristic - a fixed, data-driven bandwidth,
not a free hyperparameter tuned per method.
"""
from __future__ import annotations

import numpy as np


def median_distance_lengthscale(D2: np.ndarray) -> float:
    """Median heuristic: median of the off-diagonal pairwise distances (sqrt of the
    squared Mahalanobis distance matrix). Falls back to 1.0 if fewer than 2 finite
    off-diagonal entries exist (degenerate single-client "fleet")."""
    n = D2.shape[0]
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(np.clip(D2[iu], 0.0, None))
    d = d[np.isfinite(d)]
    return float(np.median(d)) if len(d) else 1.0


def similarity_matrix(D2: np.ndarray, lengthscale: float) -> np.ndarray:
    """s_ij = exp(-D2_ij / (2*l^2)); s_ii = 1 by construction (a client is maximally
    similar to itself, regardless of numerical noise in the diagonal of D2)."""
    l2 = max(float(lengthscale), 1e-12) ** 2
    S = np.exp(-np.clip(D2, 0.0, None) / (2.0 * l2))
    np.fill_diagonal(S, 1.0)
    return S
