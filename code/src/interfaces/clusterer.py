"""
Clusterer adapter (doc §9.1): wraps the IFAC uncertainty-aware Mahalanobis k-means +
robust-GLS cluster-prototype construction into the ClusterResult shape the plan doc
expects, and adds the pairwise-distance matrix needed for Phase-1 landscape analysis
(the IFAC pipeline only ever computed that inline in a notebook cell).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from src.dynamics_distance import pairwise_uncertainty_aware_distance
from src.ifac_bridge import gls_mean_robust, mahalanobis_kmeans, spd_condition


@dataclass
class ClusterResult:
    client_cluster_labels: np.ndarray
    cluster_models: List[Dict[str, Any]]  # [{"theta": ..., "stats": ...}, ...]
    cluster_covariances: List[np.ndarray]  # SPD precision sums S_c (doc's "cluster_covariances")
    pairwise_distances: np.ndarray  # uncertainty-aware D2, see dynamics_distance.py


def fit(
    thetas: Sequence[np.ndarray],
    precisions: Sequence[np.ndarray],
    K: int,
    *,
    seed: int = 1337,
    max_iters: int = 30,
    tol: float = 1e-6,
) -> ClusterResult:
    thetas = list(thetas)
    weights = [spd_condition(np.asarray(W)) for W in precisions]
    rng = np.random.default_rng(seed)
    assignments, mus = mahalanobis_kmeans(
        thetas, weights, K, max_iters=max_iters, tol=tol, rng=rng, print_debug=False
    )

    d = thetas[0].shape[0]
    cluster_models: List[Dict[str, Any]] = []
    cluster_covariances: List[np.ndarray] = []
    for c_id in range(K):
        members = [i for i, a in enumerate(assignments) if a == c_id]
        if not members:
            cluster_models.append({"theta": mus[c_id], "stats": {"n": 0}})
            cluster_covariances.append(1e-6 * np.eye(d))
            continue
        theta_c, stats_c = gls_mean_robust(
            thetas=[thetas[i] for i in members],
            precisions=[weights[i] for i in members],
        )
        cluster_models.append({"theta": theta_c, "stats": stats_c})
        S_c = sum(weights[i] for i in members)
        cluster_covariances.append(spd_condition(S_c))

    pairwise = pairwise_uncertainty_aware_distance(thetas, weights)
    return ClusterResult(
        client_cluster_labels=np.asarray(assignments),
        cluster_models=cluster_models,
        cluster_covariances=cluster_covariances,
        pairwise_distances=pairwise,
    )
