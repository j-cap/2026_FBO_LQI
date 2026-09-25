import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from scipy.linalg import cho_factor, cho_solve


# def kkt_residual(cluster_idx, assignments, thetas, precisions, mus):
#     members = [i for i,a in enumerate(assignments) if a == cluster_idx]
#     S = sum(precisions[i] for i in members)
#     b = sum(precisions[i] @ thetas[i] for i in members)
#     mu = mus[cluster_idx]
#     return float(np.linalg.norm(S @ mu - b)), len(members)

# def mahalanobis_kmeans_old(thetas, precisions, K,  max_iters=30, tol=1e-6, rng=None, print_debug=True):

#     """K-means clustering with Mahalanobis distance defined by per-point precision matrices. """
#     N = len(thetas)
#     d = thetas[0].shape[0]
#     rng = np.random.default_rng() if rng is None else rng

#     # --- kmeans++-like init in the client-specific metric
#     mus = [thetas[rng.integers(0, N)].copy()]
#     for _ in range(1, K):
#         d2 = np.array([min((xi-m) @ Wi @ (xi-m) for m in mus) for xi,Wi in zip(thetas,precisions)])
#         p  = d2 / (d2.sum() + 1e-12)
#         mus.append(thetas[rng.choice(N, p=p)].copy())

#     assignments = np.full(N, -1, dtype=int)        
#     for it in range(max_iters):
#         # --- assignment
#         changed = 0
#         for i,(xi,Wi) in enumerate(zip(thetas,precisions)):
#             ds = np.array([(xi-m) @ Wi @ (xi-m) for m in mus])
#             c_new = int(np.argmin(ds))
#             changed += (c_new != assignments[i])
#             assignments[i] = c_new

#         # --- update (GLS centroids) with empty-cluster handling
#         counts = np.zeros(K, dtype=int)
#         S_list = [np.zeros((d,d)) for _ in range(K)]
#         b_list = [np.zeros(d)     for _ in range(K)]

#         for i,(xi,Wi) in enumerate(zip(thetas,precisions)):
#             c = assignments[i]
#             counts[c] += 1
#             S_list[c] += Wi
#             b_list[c] += Wi @ xi

#         mus_new = [None]*K
#         for c in range(K):
#             if counts[c] == 0:
#                 # re-seed to the point farthest from *any* current centroid
#                 d2 = np.array([min((xi-m) @ Wi @ (xi-m) for m in mus) for xi,Wi in zip(thetas,precisions)])
#                 j  = int(np.argmax(d2))
#                 mus_new[c] = thetas[j].copy()
#                 counts[c] = 1
#                 S_list[c] = precisions[j].copy()
#                 b_list[c] = precisions[j] @ thetas[j]

#             S = 0.5*(S_list[c] + S_list[c].T)
#             # ridge scaled to cluster mass
#             mass = max(1.0, np.trace(S)/d)
#             S += (1e-6 * mass) * np.eye(d)
#             mus_new[c] = np.linalg.solve(S, b_list[c])

#         # convergence
#         delta = sum(np.linalg.norm(mus_new[c] - mus[c]) for c in range(K))
#         mus = mus_new
#         if changed == 0 or delta < tol:
#             break
    
#     # ---------- SAFETY CHECK / POLISH ----------
#     # recompute per-cluster S_c, b_c, μ_gls and compare with mus
#     S_bar = [np.zeros((d,d)) for _ in range(K)]
#     b_bar = [np.zeros(d)     for _ in range(K)]
#     n_mem = [0]*K
#     for i,(xi,Wi) in enumerate(zip(thetas,precisions)):
#         c = assignments[i]
#         S_bar[c] += Wi
#         b_bar[c] += Wi @ xi
#         n_mem[c] += 1

#     mus_polished = mus[:]  # copy
#     for c in range(K):
#         if n_mem[c] == 0:
#             if print_debug:
#                 print(f"[check] Cluster {c}: empty.")
#             continue
#         S = 0.5*(S_bar[c] + S_bar[c].T)
#         mass = max(1.0, np.trace(S)/d)
#         S += (1e-6 * mass) * np.eye(d)
#         mu_gls = np.linalg.solve(S, b_bar[c])

#         # metrics
#         r = S @ mu_gls - b_bar[c]
#         denom = (np.linalg.norm(S,2)*np.linalg.norm(mu_gls,2) + np.linalg.norm(b_bar[c],2) + 1e-18)
#         kkt = float(np.linalg.norm(r,2) / denom)
#         rel_delta = float(np.linalg.norm(mu_gls - mus[c]) / (np.linalg.norm(mu_gls) + 1e-18))
#         condS = float(np.linalg.cond(S))

#     if print_debug:
#         for c in range(K):
#             res, n_mem = kkt_residual_relative(c, assignments, thetas, precisions, mus)
#             res_abs, _ = kkt_residual(c, assignments, thetas, precisions, mus)
#             print(f"Cluster {c}: members={n_mem}, KKT residual={res:.4e}, KKT residual (abs)={res_abs:.4e}")

#     return assignments, mus

def mahalanobis_kmeans(thetas, precisions, K, max_iters=30, tol=1e-6, rng=None, print_debug=True):
    """
    K-means with Mahalanobis distance defined by per-point precision matrices.

    Inputs
    ------
    thetas : list or array-like of shape (N, d)
        Parameter vectors θ_i for each client.
    precisions : list or array-like of shape (N, d, d)
        Precision matrices W_i (ideally SPD, e.g. from RLS).
    K : int
        Number of clusters.
    max_iters : int
        Maximum number of K-means iterations.
    tol : float
        Convergence tolerance on centroid movement (sum of 2-norms).
    rng : np.random.Generator or None
        Random generator for reproducibility. If None, a new default RNG is used.
    print_debug : bool
        If True, prints a small log per iteration.

    Returns
    -------
    assignments : np.ndarray of shape (N,)
        Cluster index (0..K-1) for each client.
    mus : list of np.ndarray, length K
        Cluster centroids μ_c (each of shape (d,)).
    """
    # Ensure list form
    thetas = [np.asarray(theta).ravel() for theta in thetas]
    precisions = [np.asarray(W) for W in precisions]

    N = len(thetas)
    if N == 0:
        raise ValueError("Empty 'thetas' list.")
    d = thetas[0].shape[0]
    rng = np.random.default_rng() if rng is None else rng

    # ---------- k-means++-like initialization ----------
    mus = [thetas[rng.integers(0, N)].copy()]
    for _ in range(1, K):
        # squared distance to nearest existing centroid, in client metric W_i
        d2 = np.array([
            min((xi - m) @ Wi @ (xi - m) for m in mus)
            for xi, Wi in zip(thetas, precisions)
        ])
        d2 = np.clip(d2, 0.0, None)
        s = float(d2.sum())
        if not np.isfinite(s) or s <= 0.0:
            # all distances zero or NaN: fall back to uniform choice
            j = rng.integers(0, N)
        else:
            p = d2 / s
            j = rng.choice(N, p=p)
        mus.append(thetas[j].copy())

    assignments = np.full(N, -1, dtype=int)

    # ---------- main K-means loop ----------
    for it in range(max_iters):
        # --- assignment step ---
        changed = 0
        for i, (xi, Wi) in enumerate(zip(thetas, precisions)):
            ds = np.array([(xi - m) @ Wi @ (xi - m) for m in mus])
            c_new = int(np.argmin(ds))
            changed += (c_new != assignments[i])
            assignments[i] = c_new

        # --- accumulate per-cluster S_c and b_c ---
        counts = np.zeros(K, dtype=int)
        S_list = [np.zeros((d, d)) for _ in range(K)]
        b_list = [np.zeros(d) for _ in range(K)]
        for i, (xi, Wi) in enumerate(zip(thetas, precisions)):
            c = assignments[i]
            counts[c] += 1
            S_list[c] += Wi
            b_list[c] += Wi @ xi

        # --- centroid update (GLS) ---
        mus_new = [None] * K

        for c in range(K):
            # Handle empty clusters: reseed to farthest point from any centroid
            if counts[c] == 0:
                d2 = np.array([
                    min((xi - m) @ Wi @ (xi - m) for m in mus)
                    for xi, Wi in zip(thetas, precisions)
                ])
                j = int(np.argmax(d2))
                mus_new[c] = thetas[j].copy()
                counts[c] = 1
                S_list[c] = precisions[j].copy()
                b_list[c] = precisions[j] @ thetas[j]
                continue

            # GLS system: S_c μ_c = b_c, with mild regularization
            S = 0.5 * (S_list[c] + S_list[c].T)  # symmetrize
            mass = max(1.0, float(np.trace(S) / d))  # scale of S
            S_reg = S + (1e-6 * mass) * np.eye(d)    # small ridge

            try:
                # Cholesky is natural for SPD
                cc, lower = cho_factor(S_reg, overwrite_a=False, check_finite=False)
                mu_c = cho_solve((cc, lower), b_list[c], check_finite=False)
            except np.linalg.LinAlgError:
                # Very rare fallback: generic solve with a tiny extra jitter
                mu_c = np.linalg.solve(S_reg + 1e-9 * np.eye(d), b_list[c])

            mus_new[c] = mu_c

        # --- convergence check ---
        delta = sum(np.linalg.norm(mus_new[c] - mus[c]) for c in range(K))
        mus = mus_new

        if print_debug:
            print(f"[iter {it:02d}] changed={changed}, delta={delta:.3e}")

        if changed == 0 or delta < tol:
            break

    return assignments, mus

# ----------------------


def _rect_confusion(y_true, y_pred, T, P):
    """Build a rectangular confusion matrix with rows=true classes (0..T-1),
    cols=predicted clusters (0..P-1)."""
    C = np.zeros((T, P), dtype=int)
    for i in range(T):
        mask_i = (y_true == i)
        if mask_i.any():
            counts = np.bincount(y_pred[mask_i], minlength=P)
            C[i, :P] = counts[:P]
    return C

def clustering_agreement(true, assignments):
    """
    Works when #predicted clusters != #true clusters.
    Returns accuracy (Hungarian on a padded rectangular confusion), purity, NMI, ARI,
    rectangular confusion C (T x P), a pred->true mapping for matched pairs,
    and composition fractions for compact plotting.
    """
    true = np.asarray(true)
    assignments = np.asarray(assignments)

    # --- 0) Basic sanity checks ---
    if true.shape[0] != assignments.shape[0]:
        raise ValueError(f"Length mismatch: len(true)={true.shape[0]} vs len(assignments)={assignments.shape[0]}")
    if true.ndim != 1 or assignments.ndim != 1:
        raise ValueError("Inputs must be 1D arrays.")

    # Drop any NaNs (if present) consistently
    mask = ~np.isnan(true) & ~np.isnan(assignments)
    if mask.sum() < true.shape[0]:
        true = true[mask]
        assignments = assignments[mask]

    # --- 1) Relabel to contiguous 0..T-1 and 0..P-1 (handles arbitrary label ids) ---
    true_vals, y_true = np.unique(true, return_inverse=True)     # T distinct true classes
    pred_vals, y_pred = np.unique(assignments, return_inverse=True)  # P distinct predicted clusters
    T, P = len(true_vals), len(pred_vals)

    total = y_true.shape[0]
    if total == 0:
        # Degenerate case
        return dict(
            accuracy=0.0, purity=0.0, nmi=0.0, ari=0.0,
            confusion=np.zeros((T, P), dtype=int),
            pred_to_true_map={}, composition_fractions=np.zeros((T, P)),
            diag_fraction=0.0, offdiag_fraction=1.0,
            true_labels=true_vals, pred_labels=pred_vals,
        )

    # --- 2) Rectangular confusion (T x P) ---
    C = _rect_confusion(y_true, y_pred, T, P)

    # --- 3) Accuracy via optimal 1-1 matching (Hungarian) on a padded square matrix ---
    if T == P:
        C_pad = C.copy()
    elif T > P:
        C_pad = np.hstack([C, np.zeros((T, T - P), dtype=int)])
    else:  # P > T
        C_pad = np.vstack([C, np.zeros((P - T, P), dtype=int)])

    cost = C_pad.max() - C_pad  # maximize trace
    row_ind, col_ind = linear_sum_assignment(cost)

    matched = 0
    pred_to_true = {}
    for r, c in zip(row_ind, col_ind):
        if r < T and c < P:  # ignore padded rows/cols
            matched += C[r, c]
            pred_to_true[pred_vals[c]] = true_vals[r]
    accuracy = matched / total

    # --- 4) Purity: sum of per-predicted-cluster majorities / N ---
    purity = (C.max(axis=0).sum()) / total

    # --- 5) Information-theoretic & pairwise scores ---
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)

    # --- 6) Fractions for compact stacked-bar plots ---
    col_sums = C.sum(axis=0, keepdims=True).clip(min=1)   # avoid divide-by-zero
    composition = C / col_sums

    return {
        "accuracy": float(accuracy),
        "purity": float(purity),
        "nmi": float(nmi),
        "ari": float(ari),
        "confusion": C,                    # rectangular (T x P)
        "pred_to_true_map": pred_to_true,  # real matches only
        "composition_fractions": composition,
        "diag_fraction": float(accuracy),
        "offdiag_fraction": float(1.0 - accuracy),
        "true_labels": true_vals,
        "pred_labels": pred_vals,
    }

# ------- Quick test (3 true classes, 2 predicted clusters) -------
