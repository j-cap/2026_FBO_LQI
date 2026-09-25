"""
The ONE standardized candidate-evaluation function (doc §7.4, Task 4). Every optimizer
built in later phases must call this - not `design_lqi` / `run_CL_client_model`
directly - so BO, safety screening, and this Phase-1 oracle scan all produce identical
structured records for identical (client, xi, episode).

Controller synthesis uses the client's IDENTIFIED model (Ad_hat, Bd_hat); the rollout
always simulates the client's TRUE plant (Ad_true, Bd_true, via `run_CL_client_model`,
or the nonlinear tanh-tire plant via `plant_mode="nonlinear_tanh"` - correction-plan
item 5 / Phase 1B) - i.e. "model-based structure from identified dynamics, evaluated on
the real system" (doc §1.5), not a train/eval leak.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from src.episodes import EpisodeBank, EpisodeSpec, calibration_episodes
from src.ifac_bridge import Client
from src.interfaces.lqi_synthesizer import ControllerDesign, synthesize
from src.interfaces.plant_client import PlantClient


@dataclass
class EvaluationResult:
    client_id: str
    cluster_id: str
    xi: np.ndarray
    controller_design: ControllerDesign
    feasible: bool
    failure_reason: Optional[str]
    tracking_rmse: Optional[float]  # mean over episodes
    tracking_rmse_per_episode: List[float] = field(default_factory=list)
    max_abs_input: Optional[float] = None
    input_saturation_fraction: Optional[float] = None
    max_abs_input_rate: Optional[float] = None
    max_abs_beta: Optional[float] = None
    input_rate_limit_violated: bool = False
    beta_limit_violated: bool = False
    episode_seeds: List[int] = field(default_factory=list)
    wall_time_s: float = 0.0


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def evaluate(
    client: Client,
    xi: Sequence[float],
    bank: EpisodeBank,
    *,
    state_scale: Sequence[float] = (1.0, 1.0, 1.0),
    fixed_R: float = 1.0,
    u_min: float = -0.5,
    u_max: float = 0.5,
    u_rate_max: Optional[float] = None,
    beta_max: Optional[float] = None,
    noise_std: float = 0.0,
    process_noise: Sequence[float] = (0.0, 0.0),
    episodes: Optional[List[EpisodeSpec]] = None,
    plant_mode: str = "linear",
    mu: float = 1.0,
) -> EvaluationResult:
    t0 = time.perf_counter()
    episodes = episodes if episodes is not None else calibration_episodes(bank)
    cluster_id = str(getattr(client, "cluster_id_est", client.cluster.name))
    xi_arr = np.asarray(xi, dtype=float)

    design = synthesize(client.Ad_hat, client.Bd_hat, xi_arr, state_scale=state_scale, fixed_R=fixed_R)
    if not design.nominal_feasible:
        return EvaluationResult(
            client_id=client.client_id,
            cluster_id=cluster_id,
            xi=xi_arr,
            controller_design=design,
            feasible=False,
            failure_reason=design.synthesis_failure_reason,
            tracking_rmse=None,
            episode_seeds=[e.seed for e in episodes],
            wall_time_s=time.perf_counter() - t0,
        )

    client.Kx, client.Ki, client.k_r = design.Kx, design.Ki, design.k_r
    plant = PlantClient(client)

    rmses: List[float] = []
    max_u, max_u_rate, max_beta = 0.0, 0.0, 0.0
    n_saturated, n_samples = 0, 0
    for ep in episodes:
        # Multi-maneuver episode banks (Part 8, docs/phase2_diagnostic_findings.md,
        # 2026-09-03) tag each episode with its own T_lc and look it up in
        # `bank.refs`; a v1/v2 bank's episodes all have `T_lc=None` and fall back to
        # the bank-level `t`/`r_ref` exactly as before this change.
        t_ep, r_ref_ep = bank.refs[ep.T_lc] if (bank.refs is not None and ep.T_lc is not None) else (bank.t, bank.r_ref)
        rollout = plant.rollout(
            t_ep, r_ref_ep, seed=ep.seed, noise_std=noise_std,
            process_noise=process_noise, u_min=u_min, u_max=u_max,
            plant_mode=plant_mode, mu=mu,
        )
        rmses.append(_rmse(r_ref_ep[: len(rollout.y)], rollout.y))
        max_u = max(max_u, float(np.max(np.abs(rollout.u))))
        if rollout.u.size > 1:
            max_u_rate = max(max_u_rate, float(np.max(np.abs(np.diff(rollout.u)))))
        max_beta = max(max_beta, float(np.max(np.abs(rollout.x[:, 0]))))
        sat_mask = np.isclose(rollout.u, u_min, atol=1e-6) | np.isclose(rollout.u, u_max, atol=1e-6)
        n_saturated += int(np.sum(sat_mask))
        n_samples += rollout.u.size

    rate_violated = (u_rate_max is not None) and (max_u_rate > u_rate_max)
    beta_violated = (beta_max is not None) and (max_beta > beta_max)

    return EvaluationResult(
        client_id=client.client_id,
        cluster_id=cluster_id,
        xi=xi_arr,
        controller_design=design,
        feasible=not (rate_violated or beta_violated),
        failure_reason=("input_rate_limit" if rate_violated else "beta_limit" if beta_violated else None),
        tracking_rmse=float(np.mean(rmses)),
        tracking_rmse_per_episode=rmses,
        max_abs_input=max_u,
        input_saturation_fraction=(n_saturated / n_samples) if n_samples else None,
        max_abs_input_rate=max_u_rate,
        max_abs_beta=max_beta,
        input_rate_limit_violated=rate_violated,
        beta_limit_violated=beta_violated,
        episode_seeds=[e.seed for e in episodes],
        wall_time_s=time.perf_counter() - t0,
    )
