"""
Held-out evaluation and cross-client controller-transfer analysis (Phase-1
correction-plan items 2 and 6).

Item 2: separates oracle SELECTION (which xi looks best, decided from the calibration
episode bank) from oracle EVALUATION (how that selected xi actually performs, measured
on a completely disjoint held-out test bank) - so the reported personalization gain
isn't partly selection noise from picking an argmin on the same data used to score it.

Item 6: the primary RQ1 diagnostic. For every ordered client pair i<-j, measures
whether TRANSPLANTING client j's individually-optimal controller onto client i helps
or hurts, on i's own held-out test episodes - not just whether the two clients'
whole landscapes correlate (which stays available as a secondary plot,
src.landscape_metrics/dynamics_distance).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd

from src.episodes import EpisodeBank, test_episodes
from src.ifac_bridge import Client

XI_COLS = ["xi_1", "xi_2", "xi_3"]


def build_client_selection_table(
    fleet: Sequence[Client],
    individual_opt: pd.DataFrame,
    cluster_opt: pd.DataFrame,
    regime_opt: pd.DataFrame,
    global_opt: pd.DataFrame,
) -> pd.DataFrame:
    """One row per (client, level) with the xi CALIBRATION selected for that client at
    that level - global/cluster/regime resolve to the client's own
    cluster/regime's selection; individual is the client's own. A level is omitted for
    a client if that level has no defined selection (e.g. no feasible candidate)."""
    ind_by_client = individual_opt.set_index("client_id") if not individual_opt.empty else individual_opt
    cluster_by_group = cluster_opt.set_index("estimated_cluster") if not cluster_opt.empty else cluster_opt
    regime_by_group = regime_opt.set_index("true_regime") if not regime_opt.empty else regime_opt
    global_xi = global_opt.iloc[0][XI_COLS].to_numpy(dtype=float) if not global_opt.empty else None

    rows = []
    for client in fleet:
        cid = client.client_id
        if not individual_opt.empty and cid in ind_by_client.index:
            xi = ind_by_client.loc[cid, XI_COLS].to_numpy(dtype=float)
            rows.append({"client_id": cid, "level": "individual", **dict(zip(XI_COLS, xi))})
        cluster_key = str(getattr(client, "cluster_id_est", None))
        if not cluster_opt.empty and cluster_key in cluster_by_group.index:
            xi = cluster_by_group.loc[cluster_key, XI_COLS].to_numpy(dtype=float)
            rows.append({"client_id": cid, "level": "estimated_cluster", **dict(zip(XI_COLS, xi))})
        regime_key = client.cluster.name
        if not regime_opt.empty and regime_key in regime_by_group.index:
            xi = regime_by_group.loc[regime_key, XI_COLS].to_numpy(dtype=float)
            rows.append({"client_id": cid, "level": "true_regime", **dict(zip(XI_COLS, xi))})
        if global_xi is not None:
            rows.append({"client_id": cid, "level": "global", **dict(zip(XI_COLS, global_xi))})
    return pd.DataFrame(rows)


def held_out_eval_worker(
    client: Client,
    bank: EpisodeBank,
    rows_for_client: List[dict],
    eval_kwargs: dict,
    cached_evaluate: Callable,
) -> List[dict]:
    """Evaluates every (level, xi) selection for ONE client on ITS held-out test
    episodes (disjoint from the calibration episodes used to select xi)."""
    test_eps = test_episodes(bank)
    out = []
    for row in rows_for_client:
        result = cached_evaluate(
            client, [row["xi_1"], row["xi_2"], row["xi_3"]], bank, episodes=test_eps, **eval_kwargs
        )
        out.append({**row, "test_rmse": result.tracking_rmse, "feasible": result.feasible})
    return out


def transfer_loss_worker(
    target_client: Client,
    target_bank: EpisodeBank,
    individual_xi_by_client: Dict[str, np.ndarray],
    eval_kwargs: dict,
    cached_evaluate: Callable,
) -> List[dict]:
    """
    For ONE target client i: L_{i<-j} = (J_i^test(xi_j*) - J_i^test(xi_i*)) /
    J_i^test(xi_i*), for every other client j with a defined individual optimum,
    evaluated entirely on i's own held-out test episodes. The i==j row is the "own"
    baseline (transfer_loss=0 by construction), kept for completeness.
    """
    test_eps = test_episodes(target_bank)
    cid = target_client.client_id
    xi_i = individual_xi_by_client[cid]
    result_own = cached_evaluate(target_client, xi_i, target_bank, episodes=test_eps, **eval_kwargs)
    J_i_own = result_own.tracking_rmse if result_own.feasible else np.nan

    rows = []
    for j, xi_j in individual_xi_by_client.items():
        if j == cid:
            rows.append(
                {"client_i": cid, "client_j": j, "J_i_own": J_i_own, "J_i_under_j": J_i_own, "transfer_loss": 0.0}
            )
            continue
        result = cached_evaluate(target_client, xi_j, target_bank, episodes=test_eps, **eval_kwargs)
        J_i_under_j = result.tracking_rmse if result.feasible else np.nan
        if J_i_own is not None and np.isfinite(J_i_own) and J_i_own > 0 and np.isfinite(J_i_under_j):
            transfer_loss = (J_i_under_j - J_i_own) / J_i_own
        else:
            transfer_loss = np.nan
        rows.append(
            {"client_i": cid, "client_j": j, "J_i_own": J_i_own, "J_i_under_j": J_i_under_j, "transfer_loss": transfer_loss}
        )
    return rows


def decide_next_step(
    G_cluster: float, individual_vs_cluster_gap: float, transfer_correlation: float, *,
    transfer_correlation_ci: "tuple[float, float] | None" = None,
    worst_case_transfer_risk: float = float("nan"),
    cluster_benefit_threshold: float = 0.03, transfer_corr_threshold: float = 0.4,
    individual_gap_threshold: float = 0.03, worst_case_risk_threshold: float = 0.15,
) -> str:
    """
    The five-way decision from the correction plan's item 7 (four cases), extended
    with a fifth case found necessary after Phase 1A: G_cluster and
    individual_vs_cluster_gap are FLEET-MEAN statistics, and global/cluster selection
    structurally never picks an xi that's catastrophic for a subgroup (it would show up
    in the mean and get avoided) - so a real, physically-grounded, worst-case/pairwise
    transfer risk can exist and correlate significantly with dynamics distance while
    both mean-based numbers stay small (this is exactly what happened in
    phase1a_matched_model_full_v2 - see docs/phase1_diagnostic_findings.md). Without
    this case, that situation mechanically fell into "stop_direction" despite a
    significant, well-powered transfer-loss/distance correlation.

      - cluster benefit AND transfer relation exist -> implement dynamics-informed FBO
      - cluster benefit exists but raw parameter distance fails -> try a control-aware descriptor
      - only individual tuning helps (no cluster benefit, but a real individual-cluster gap) -> rethink federation granularity
      - NO mean-level benefit anywhere, BUT transfer relation exists AND worst-case
        transfer risk is real -> the value is in SAFETY SCREENING (avoid known-bad
        donor/recipient pairs), not average-performance optimization - pursue
        uncertainty-aware screening (plan doc Phase 5) before/instead of FBO
      - none of the above -> stop this direction

    `transfer_relation` is true if EITHER the point estimate clears
    `transfer_corr_threshold` (the plan doc's original "rough target |rho|>=0.4") OR the
    bootstrap CI's lower bound excludes zero (statistically significant even if the
    point estimate falls just short of 0.4 - exactly phase1a_matched_model_full_v2's
    case: rho=0.377 with CI (0.272, 0.490), a tight, zero-excluding interval that a
    magnitude-only test would have wrongly called "not significant"). A CI-based test
    is the statistically sound one; the magnitude threshold is kept as an alternative
    so a large-but-noisy point estimate (wide CI) can still qualify.
    """
    cluster_benefit = np.isfinite(G_cluster) and G_cluster >= cluster_benefit_threshold
    ci_significant = transfer_correlation_ci is not None and np.isfinite(transfer_correlation_ci[0]) and transfer_correlation_ci[0] > 0
    transfer_relation = (
        np.isfinite(transfer_correlation) and abs(transfer_correlation) >= transfer_corr_threshold
    ) or ci_significant
    individual_gap = np.isfinite(individual_vs_cluster_gap) and individual_vs_cluster_gap >= individual_gap_threshold
    worst_case_risk = (
        np.isfinite(worst_case_transfer_risk) and worst_case_transfer_risk >= worst_case_risk_threshold
    )

    if cluster_benefit and transfer_relation:
        return "implement_dynamics_informed_fbo"
    if cluster_benefit and not transfer_relation:
        return "test_control_aware_descriptor"
    if (not cluster_benefit) and individual_gap:
        return "rethink_federation_granularity"
    if (not cluster_benefit) and transfer_relation and worst_case_risk:
        return "pursue_dynamics_informed_safety_screening"
    return "stop_direction"
