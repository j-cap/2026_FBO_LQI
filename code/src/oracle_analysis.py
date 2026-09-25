"""
Oracle-scan analysis (doc §10.3, §10.6, §10.7): per-client / per-regime /
per-estimated-cluster / global oracle optima, and the automatic go/no-go criteria that
decide whether the rest of the paper is worth pursuing. Operates on the tidy
oracle_landscapes DataFrame `scripts/phase1_oracle_landscapes.py` builds - one row per
(client_id, xi_idx) evaluation with columns: client_id, true_regime, estimated_cluster,
xi_idx, xi_1, xi_2, xi_3, feasible, tracking_rmse, ...
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

XI_COLS = ["xi_1", "xi_2", "xi_3"]


def individual_optima(df: pd.DataFrame) -> pd.DataFrame:
    """Per-client argmin tracking_rmse among feasible candidates."""
    feasible = df[df["feasible"] & df["tracking_rmse"].notna()]
    idx = feasible.groupby("client_id")["tracking_rmse"].idxmin()
    return feasible.loc[idx].reset_index(drop=True)


def grouped_optima(df: pd.DataFrame, group_col: str, *, min_coverage: float = 0.5) -> pd.DataFrame:
    """
    For each group (true_regime / estimated_cluster / a constant "_global" column) and
    each shared candidate xi_idx, average tracking_rmse over feasible clients in that
    group; candidates covered by fewer than `min_coverage` of the group's clients are
    dropped before taking the argmin (doc §10.3).
    """
    group_sizes = df.groupby(group_col)["client_id"].nunique()
    feasible = df[df["feasible"] & df["tracking_rmse"].notna()].copy()
    agg = (
        feasible.groupby([group_col, "xi_idx"])
        .agg(
            mean_rmse=("tracking_rmse", "mean"),
            n_covered=("client_id", "nunique"),
            **{c: (c, "first") for c in XI_COLS},
        )
        .reset_index()
    )
    agg["coverage"] = agg.apply(lambda r: r["n_covered"] / group_sizes[r[group_col]], axis=1)
    agg = agg[agg["coverage"] >= min_coverage]
    if agg.empty:
        return agg
    idx = agg.groupby(group_col)["mean_rmse"].idxmin()
    return agg.loc[idx].reset_index(drop=True)


def landscape_matrix(df: pd.DataFrame, xi_cols=XI_COLS) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Pivot to (client_ids, Xi (n_xi x 3), J (n_clients x n_xi), NaN where infeasible)."""
    client_ids = sorted(df["client_id"].unique())
    xi_idx_sorted = sorted(df["xi_idx"].unique())
    xi_lookup = df.drop_duplicates("xi_idx").set_index("xi_idx")[xi_cols]
    Xi = xi_lookup.loc[xi_idx_sorted].to_numpy()
    pivot = df.pivot_table(index="client_id", columns="xi_idx", values="tracking_rmse")
    pivot = pivot.reindex(index=client_ids, columns=xi_idx_sorted)
    return client_ids, Xi, pivot.to_numpy()


def _perf_under_group_optimum(df: pd.DataFrame, group_opt: pd.DataFrame, group_col: str) -> float:
    """Mean feasible tracking_rmse of every client, each evaluated at ITS group's
    oracle xi_idx (a cluster/regime's own optimum, not one shared xi_idx)."""
    if group_opt.empty:
        return float("nan")
    xi_idx_map = group_opt.set_index(group_col)["xi_idx"].rename("target_xi_idx")
    merged = df.merge(xi_idx_map, left_on=group_col, right_index=True, how="inner")
    sel = merged[(merged["xi_idx"] == merged["target_xi_idx"]) & merged["feasible"]]
    return float(sel["tracking_rmse"].mean()) if len(sel) else float("nan")


def bootstrap_spearman_ci(
    D: np.ndarray, S: np.ndarray, *, n_boot: int = 1000, seed: int = 0, ci: float = 0.95
) -> Tuple[float, Tuple[float, float]]:
    """
    Client-level (node) bootstrap of the Spearman correlation between two (N,N)
    pairwise matrices: resample client INDICES with replacement and recompute over the
    induced pairs, skipping self-pairs from a duplicate draw. Avoids treating
    non-independent client PAIRS as independent samples (doc §19).
    """
    rng = np.random.default_rng(seed)
    N = D.shape[0]
    corrs = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        d_vals, s_vals = [], []
        for a in range(N):
            for b in range(a + 1, N):
                i, j = idx[a], idx[b]
                if i == j:
                    continue
                d_ij, s_ij = D[i, j], S[i, j]
                # Drop NaN pairs (e.g. an infeasible transfer evaluation) - unlike the
                # dense landscape-distance matrix this was first written for, S here can
                # be a transfer-loss matrix with many missing entries; scipy's spearmanr
                # returns NaN for the WHOLE sample if even one pair is NaN, which without
                # this filter silently produced nan for every bootstrap draw.
                if np.isfinite(d_ij) and np.isfinite(s_ij):
                    d_vals.append(d_ij)
                    s_vals.append(s_ij)
        if len(d_vals) < 8:
            continue
        rho = spearmanr(d_vals, s_vals)[0]
        if np.isfinite(rho):
            corrs.append(rho)
    if not corrs:
        return float("nan"), (float("nan"), float("nan"))
    corrs_arr = np.asarray(corrs)
    lo = float(np.percentile(corrs_arr, (1 - ci) / 2 * 100))
    hi = float(np.percentile(corrs_arr, (1 + ci) / 2 * 100))
    return float(np.median(corrs_arr)), (lo, hi)


@dataclass
class GoNoGoReport:
    global_perf: float
    cluster_perf: float
    individual_perf: float
    criterion_1_specialization_gain: float
    criterion_2_personalization_gap_closed: float
    criterion_3_dyn_landscape_spearman: float
    criterion_3_ci: Tuple[float, float]
    criterion_4_boundary_fraction: float
    kill_condition_triggered: bool
    kill_condition_detail: str
    notes: List[str]


def evaluate_go_no_go(
    df: pd.DataFrame,
    dyn_distance: np.ndarray,
    pairwise_landscape_1_minus_spearman: np.ndarray,
    xi_lower,
    xi_upper,
    *,
    boundary_tol_frac: float = 0.05,
    n_boot: int = 1000,
    seed: int = 0,
) -> GoNoGoReport:
    notes: List[str] = []

    ind = individual_optima(df)
    df_g = df.copy()
    df_g["_global"] = "global"
    global_opt = grouped_optima(df_g, "_global")
    cluster_opt = grouped_optima(df, "estimated_cluster")

    global_perf = _perf_under_group_optimum(df_g, global_opt, "_global")
    cluster_perf = _perf_under_group_optimum(df, cluster_opt, "estimated_cluster")
    individual_perf = float(ind["tracking_rmse"].mean()) if len(ind) else float("nan")

    specialization_gain = (
        (global_perf - cluster_perf) / global_perf
        if np.isfinite(global_perf) and global_perf != 0
        else float("nan")
    )
    denom = global_perf - individual_perf
    gap_closed = (
        (global_perf - cluster_perf) / denom if np.isfinite(denom) and abs(denom) > 1e-12 else float("nan")
    )

    median_rho, ci_rho = bootstrap_spearman_ci(
        dyn_distance, pairwise_landscape_1_minus_spearman, n_boot=n_boot, seed=seed
    )

    xi_lower_a, xi_upper_a = np.asarray(xi_lower, float), np.asarray(xi_upper, float)
    span = xi_upper_a - xi_lower_a
    if len(ind):
        xi_vals = ind[XI_COLS].to_numpy(dtype=float)
        near_lo = np.abs(xi_vals - xi_lower_a) <= boundary_tol_frac * span
        near_hi = np.abs(xi_vals - xi_upper_a) <= boundary_tol_frac * span
        boundary_fraction = float(np.mean(np.any(near_lo | near_hi, axis=1)))
    else:
        boundary_fraction = float("nan")

    optimum_spread = (
        float(np.mean(np.std(ind[XI_COLS].to_numpy(dtype=float), axis=0))) if len(ind) else float("nan")
    )
    near_optimal = (
        np.isfinite(global_perf)
        and np.isfinite(individual_perf)
        and individual_perf > 0
        and (global_perf - individual_perf) / individual_perf <= 0.02
    )
    kill_triggered = bool(near_optimal and np.isfinite(optimum_spread) and optimum_spread < 0.15)
    if np.isfinite(optimum_spread) and np.isfinite(individual_perf) and individual_perf > 0:
        rel_gap_pct = (global_perf - individual_perf) / individual_perf * 100
        kill_detail = (
            f"mean per-dim std of individual optima={optimum_spread:.3f}, "
            f"global-vs-individual relative gap={rel_gap_pct:.2f}%"
        )
    else:
        kill_detail = "insufficient feasible data to evaluate the kill condition"

    if np.isfinite(boundary_fraction) and boundary_fraction > 0.3:
        notes.append(
            f"{boundary_fraction:.0%} of individual optima sit on the search-box boundary - "
            "consider widening xi_lower/xi_upper before Phase 2 (doc §10.7 Criterion 4)."
        )
    # Config-dependent notes (state_scale / u_rate_max / beta_max_deg) are appended by
    # the caller (scripts/phase1_oracle_landscapes.py::_write_summary), which has the
    # actual resolved controller_cfg - NOT here, where they'd be a static claim that
    # goes stale the moment those values change (as happened in an earlier version of
    # this function).

    return GoNoGoReport(
        global_perf=global_perf,
        cluster_perf=cluster_perf,
        individual_perf=individual_perf,
        criterion_1_specialization_gain=specialization_gain,
        criterion_2_personalization_gap_closed=gap_closed,
        criterion_3_dyn_landscape_spearman=median_rho,
        criterion_3_ci=ci_rho,
        criterion_4_boundary_fraction=boundary_fraction,
        kill_condition_triggered=kill_triggered,
        kill_condition_detail=kill_detail,
        notes=notes,
    )
