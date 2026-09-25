"""
Fixed-speed, plant-property fleet generation (Next-steps fleet redesign).

Replaces LOW_SPEED/HIGH_SPEED (speed as the heterogeneity axis) with three plant
PROPERTY families, all at U=15 m/s: nominal (unperturbed baseline), payload
(src2's own PAYLOAD prototype: m+20%, Iz+25%), and tire-degraded (src2's own
AGING_TIRES prototype: Cf-25%). Rationale (see docs/phase1_diagnostic_findings.md):
speed is an operating condition, not an intrinsic client property, and Phase 1B's
feasibility audit found HIGH_SPEED (U=20 m/s) combined with the nonlinear tire law
produces genuine large-angle sideslip divergence (mean |beta|=30 deg, max 219 deg) that
swamped the controller-tuning-heterogeneity signal with a qualitatively different
"did the car spin out" failure mode.

Mirrors `src2.server.generate_fleet`'s per-client construction loop exactly (same
RLS/gains/discretization setup, same T_lc split, same nominal-model Ad_hat/Bd_hat
initialization) but takes explicit prototypes instead of Cluster-enum-driven ones,
since no existing `Cluster` member represents an unperturbed "nominal" family - adding
one would mean modifying the sibling IFAC package, which is out of scope.

Family labels are diagnostic-only, exactly like the LOW_SPEED/PAYLOAD/HIGH_SPEED labels
they replace: clustering and identification only ever see (theta_hat, Sigma_i); nothing
in the pipeline is told which family a client belongs to.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.candidate_eval import evaluate
from src.episodes import (
    build_episode_bank,
    build_multi_maneuver_episode_bank,
    calibration_episodes,
    deterministic_client_seed,
)
from src.ifac_bridge import (
    BikeParams,
    Client,
    Cluster,
    LQIConfig,
    RLSAB_BlockCov,
    build_bicycle_beta_r,
    cluster_prototype,
    discretize,
    nominal_params,
    perturb_params,
    reset_gains,
)


@dataclass(frozen=True)
class FamilyLabel:
    """Stand-in for src2's `Cluster` enum for the "nominal" family, which has no
    corresponding Cluster member (there is no "apply no perturbation" prototype).
    Only `.name` is read downstream (labeling/grouping/figures) - this is never passed
    back into src2's own logic, which doesn't use `Client.cluster` for computation."""

    name: str


NOMINAL = FamilyLabel("NOMINAL")
PAYLOAD = FamilyLabel("PAYLOAD")
TIRE_DEGRADED = FamilyLabel("TIRE_DEGRADED")


def family_prototypes() -> Dict[FamilyLabel, BikeParams]:
    """The three generating families, all at U=15 m/s: nominal (unperturbed),
    payload and tire-degraded reuse src2's own PAYLOAD / AGING_TIRES prototypes
    (their m/Iz/Cf/Cr arithmetic is not re-derived here)."""
    p0 = nominal_params()
    return {
        NOMINAL: p0,
        PAYLOAD: cluster_prototype(Cluster.PAYLOAD, p0),
        TIRE_DEGRADED: cluster_prototype(Cluster.AGING_TIRES, p0),
    }


def generate_fixed_speed_fleet(
    Ts: float,
    n_per_family: int,
    *,
    variability: float = 0.025,
    seed: int = 1337,
    delta: float = 1.0,
    lambda_f: float = 0.99,
    lane_change_time: Sequence[float] = (1.0, 2.0),
) -> List[Client]:
    """Mirrors `generate_fleet`'s per-client construction loop exactly, iterating the
    three `family_prototypes()` entries instead of `Cluster`-enum-driven prototypes."""
    rng = np.random.default_rng(seed)
    p0 = nominal_params()
    A_nom, B_nom, C_nom, D_nom = build_bicycle_beta_r(p0)
    Ad_nom, Bd_nom, Cd_nom, Dd_nom = discretize(A_nom, B_nom, C_nom, D_nom, Ts)
    theta_init = np.hstack([Ad_nom, Bd_nom])

    clients: List[Client] = []
    for label, proto in family_prototypes().items():
        for i in range(n_per_family):
            p_i = perturb_params(proto, rng, variability)
            A, B, C, D = build_bicycle_beta_r(p_i)
            Ad, Bd, Cd, Dd = discretize(A, B, C, D, Ts)
            T_lc = lane_change_time[0] if i < n_per_family / 2 else lane_change_time[1]
            client = Client(
                client_id=f"{label.name.lower()}_{i:02d}",
                cluster=label,
                params=p_i,
                A=A, B=B, C=C, D=D,
                Ad_hat=Ad_nom, Bd_hat=Bd_nom, Cd_hat=Cd_nom, Dd_hat=Dd_nom,
                Ad_true=Ad, Bd_true=Bd, Cd_true=Cd, Dd_true=Dd,
                lqi_cfg=LQIConfig(),
                rls=RLSAB_BlockCov(n=2, lambda_f=lambda_f, theta_init=theta_init, delta=delta),
                T_lc=T_lc, W_raw=np.eye(6),
            )
            reset_gains(client, p0)
            clients.append(client)
    return clients


GENERATION_VERSION_V2 = "fixed_speed_v2_baseline_feasible"


def generate_fixed_speed_fleet_v2_baseline_feasible(
    Ts: float,
    n_per_family: int,
    *,
    variability: float = 0.025,
    seed: int = 1337,
    delta: float = 1.0,
    lambda_f: float = 0.99,
    lane_change_time: Sequence[float] = (1.0, 2.0),
    baseline_xi: Sequence[float],
    T_total: float,
    T0: float,
    r_max: float,
    n_calibration_episodes: int,
    n_test_episodes: int,
    episode_base_seed: int,
    tfilter: Optional[float],
    state_scale: Sequence[float],
    fixed_R: float,
    eval_kwargs: dict,
    max_attempts: int = 50,
) -> Tuple[List[Client], pd.DataFrame]:
    """Versioned fleet generator (Step C, docs/phase2_diagnostic_findings.md,
    2026-09-02) - does NOT modify `generate_fixed_speed_fleet` (v1), which stays
    exactly as-is for full reproducibility of every prior experiment. Admission
    criterion: `baseline_xi` (the SAME `hand_tuned_reference_xi` and evaluation
    machinery `compute_baseline_reference` uses to build every client's normalization
    denominator) must be feasible for a candidate's true plant, or that slot is
    resampled with an independent draw. Defensible because this benchmark is about
    calibrating an already-operable controller, not recovering plants for which the
    starting controller violates the benchmark's own constraints - and it eliminates a
    real confound: a baseline-infeasible client's training_y is `raw_rmse/1.0`
    (~0.01-0.02) while a normal client's is `raw_rmse/J_base` (~0.5-5) - wildly
    different scales feeding the SAME pooled GP for `similarity`/`global`/
    `recipient_aware`, sometimes at high s_ij weight between same-family clients (Part
    5/6 outlier audits, docs/phase2_diagnostic_findings.md).

    RNG design (the resampling-independence requirement): each (family, slot,
    attempt) draws its physical parameters from `deterministic_client_seed(seed,
    f"{family}_{slot:02d}_attempt{attempt}")` - an INDEPENDENT stream per attempt,
    not a shared sequentially-advancing rng. This guarantees: rejecting one slot's
    early attempts never changes any OTHER slot's accepted client; the SAME fleet
    seed always reproduces the SAME accepted fleet; and every accepted client's
    provenance (which attempt succeeded, and every attempt's outcome) is recoverable
    from the returned DataFrame - never only from a printed warning.

    Raises RuntimeError (loudly, not a silent fallback) if a slot exhausts
    `max_attempts` without a feasible draw - if this fires routinely for one family,
    that family's prototype/variability may be too close to the admissibility
    boundary (worth investigating, not raising max_attempts blindly).

    Returns `(clients, provenance_df)` - `provenance_df` has one row per ATTEMPT
    (accepted or not): columns `client_id, family, slot, attempt, accepted,
    baseline_rmse`.
    """
    p0 = nominal_params()
    A_nom, B_nom, C_nom, D_nom = build_bicycle_beta_r(p0)
    Ad_nom, Bd_nom, Cd_nom, Dd_nom = discretize(A_nom, B_nom, C_nom, D_nom, Ts)
    theta_init = np.hstack([Ad_nom, Bd_nom])
    baseline_xi_arr = np.asarray(baseline_xi, float)

    clients: List[Client] = []
    provenance_rows: List[dict] = []
    for label, proto in family_prototypes().items():
        for i in range(n_per_family):
            client_id = f"{label.name.lower()}_{i:02d}"
            T_lc = lane_change_time[0] if i < n_per_family / 2 else lane_change_time[1]
            accepted: Optional[Client] = None
            for attempt in range(max_attempts):
                slot_seed = deterministic_client_seed(seed, f"{label.name}_{i:02d}_attempt{attempt}")
                rng = np.random.default_rng(slot_seed)
                p_i = perturb_params(proto, rng, variability)
                A, B, C, D = build_bicycle_beta_r(p_i)
                Ad, Bd, Cd, Dd = discretize(A, B, C, D, Ts)
                candidate = Client(
                    client_id=client_id, cluster=label, params=p_i,
                    A=A, B=B, C=C, D=D,
                    Ad_hat=Ad_nom, Bd_hat=Bd_nom, Cd_hat=Cd_nom, Dd_hat=Dd_nom,
                    Ad_true=Ad, Bd_true=Bd, Cd_true=Cd, Dd_true=Dd,
                    lqi_cfg=LQIConfig(),
                    rls=RLSAB_BlockCov(n=2, lambda_f=lambda_f, theta_init=theta_init, delta=delta),
                    T_lc=T_lc, W_raw=np.eye(6),
                )
                reset_gains(candidate, p0)

                bank = build_episode_bank(
                    candidate, Ts=Ts, T_total=T_total, T0=T0, r_max=r_max,
                    n_calibration=n_calibration_episodes, n_test=n_test_episodes,
                    base_seed=episode_base_seed, tfilter=tfilter,
                )
                ev = evaluate(
                    candidate, baseline_xi_arr, bank, state_scale=state_scale, fixed_R=fixed_R,
                    episodes=calibration_episodes(bank), **eval_kwargs,
                )
                feasible = bool(ev.feasible and ev.tracking_rmse is not None and ev.tracking_rmse > 0)
                provenance_rows.append(
                    {
                        "client_id": client_id, "family": label.name.lower(), "slot": i, "attempt": attempt,
                        "accepted": feasible,
                        "baseline_rmse": float(ev.tracking_rmse) if ev.tracking_rmse is not None else None,
                    }
                )
                if feasible:
                    accepted = candidate
                    break
            if accepted is None:
                raise RuntimeError(
                    f"generate_fixed_speed_fleet_v2_baseline_feasible: slot {client_id!r} found no "
                    f"baseline-feasible draw in {max_attempts} attempts (fleet seed={seed}) - raising loudly "
                    "rather than silently accepting a baseline-infeasible client."
                )
            clients.append(accepted)
    return clients, pd.DataFrame(provenance_rows)


GENERATION_VERSION_V3 = "fixed_speed_v3_common_task"


def generate_fixed_speed_fleet_v3_common_task(
    Ts: float,
    n_per_family: int,
    *,
    variability: float = 0.025,
    seed: int = 1337,
    delta: float = 1.0,
    lambda_f: float = 0.99,
    lane_change_time: Sequence[float] = (1.0, 2.0),
    baseline_xi: Sequence[float],
    T_total: float,
    T0: float,
    r_max: float,
    n_calibration_episodes_per_maneuver: int,
    n_test_episodes_per_maneuver: int,
    episode_base_seed: int,
    tfilter: Optional[float],
    state_scale: Sequence[float],
    fixed_R: float,
    eval_kwargs: dict,
    max_attempts: int = 50,
) -> Tuple[List[Client], pd.DataFrame]:
    """Common-task-distribution fleet generator (Part 8, docs/phase2_diagnostic_findings.md,
    2026-09-03) - does NOT modify v1 or v2, both of which stay exactly as-is for full
    reproducibility of every prior experiment. Builds on v2's baseline-feasibility
    admission criterion (same rationale, same independent-per-attempt RNG design - see
    `generate_fixed_speed_fleet_v2_baseline_feasible`'s docstring), but ALSO removes the
    T_lc confound Part 8 found: v1/v2 assign `T_lc` by slot index
    (`i < n_per_family/2` -> `lane_change_time[0]`, else `lane_change_time[1]`), so
    `s_ij` (dynamics-only similarity) can't see a real, uniform, ~10% task-difficulty
    shift between the two halves of every family (confirmed materially and uniformly
    across 30/30 tested clients, `results/tlc_effect_audit/`). Here every client is
    evaluated - both for the admission check below and for the actual BO run via
    `build_multi_maneuver_episode_bank` - against the SAME calibration-trajectory
    distribution spanning ALL `lane_change_time` values, so `f_i(xi) = (1/N_omega) *
    sum_omega J_i(xi; omega)` for every client alike. `client.T_lc` is set to
    `lane_change_time[0]` for every client (used only as this codebase's fixed
    identification maneuver, see `build_fleet_and_identify`'s v3 branch in
    `scripts/phase2_fbo_comparison.py` - not read anywhere in the calibration/BO path
    any more, since `build_multi_maneuver_episode_bank` ignores it entirely).

    Returns `(clients, provenance_df)`, same shape as v2's.
    """
    p0 = nominal_params()
    A_nom, B_nom, C_nom, D_nom = build_bicycle_beta_r(p0)
    Ad_nom, Bd_nom, Cd_nom, Dd_nom = discretize(A_nom, B_nom, C_nom, D_nom, Ts)
    theta_init = np.hstack([Ad_nom, Bd_nom])
    baseline_xi_arr = np.asarray(baseline_xi, float)

    clients: List[Client] = []
    provenance_rows: List[dict] = []
    for label, proto in family_prototypes().items():
        for i in range(n_per_family):
            client_id = f"{label.name.lower()}_{i:02d}"
            accepted: Optional[Client] = None
            for attempt in range(max_attempts):
                slot_seed = deterministic_client_seed(seed, f"{label.name}_{i:02d}_attempt{attempt}")
                rng = np.random.default_rng(slot_seed)
                p_i = perturb_params(proto, rng, variability)
                A, B, C, D = build_bicycle_beta_r(p_i)
                Ad, Bd, Cd, Dd = discretize(A, B, C, D, Ts)
                candidate = Client(
                    client_id=client_id, cluster=label, params=p_i,
                    A=A, B=B, C=C, D=D,
                    Ad_hat=Ad_nom, Bd_hat=Bd_nom, Cd_hat=Cd_nom, Dd_hat=Dd_nom,
                    Ad_true=Ad, Bd_true=Bd, Cd_true=Cd, Dd_true=Dd,
                    lqi_cfg=LQIConfig(),
                    rls=RLSAB_BlockCov(n=2, lambda_f=lambda_f, theta_init=theta_init, delta=delta),
                    T_lc=lane_change_time[0], W_raw=np.eye(6),
                )
                reset_gains(candidate, p0)

                bank = build_multi_maneuver_episode_bank(
                    candidate, Ts=Ts, T_total=T_total, T0=T0, r_max=r_max, lane_change_time=lane_change_time,
                    n_calibration_per_maneuver=n_calibration_episodes_per_maneuver,
                    n_test_per_maneuver=n_test_episodes_per_maneuver, base_seed=episode_base_seed, tfilter=tfilter,
                )
                ev = evaluate(
                    candidate, baseline_xi_arr, bank, state_scale=state_scale, fixed_R=fixed_R,
                    episodes=calibration_episodes(bank), **eval_kwargs,
                )
                feasible = bool(ev.feasible and ev.tracking_rmse is not None and ev.tracking_rmse > 0)
                provenance_rows.append(
                    {
                        "client_id": client_id, "family": label.name.lower(), "slot": i, "attempt": attempt,
                        "accepted": feasible,
                        "baseline_rmse": float(ev.tracking_rmse) if ev.tracking_rmse is not None else None,
                    }
                )
                if feasible:
                    accepted = candidate
                    break
            if accepted is None:
                raise RuntimeError(
                    f"generate_fixed_speed_fleet_v3_common_task: slot {client_id!r} found no "
                    f"baseline-feasible draw in {max_attempts} attempts (fleet seed={seed}) - raising loudly "
                    "rather than silently accepting a baseline-infeasible client."
                )
            clients.append(accepted)
    return clients, pd.DataFrame(provenance_rows)
