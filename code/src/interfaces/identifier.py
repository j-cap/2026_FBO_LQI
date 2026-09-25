"""
Identifier adapter (doc §9.1): runs one or more identification-data episodes on a
client and returns (theta_hat, precision, A_hat, B_hat, diagnostics), instead of
reaching into `run_episode_update_rls`'s client-mutation side effects directly.

Running several episodes (one `run_episode_update_rls` call per seed, on the SAME RLS
instance) lets identification converge past a single noisy draw - this mirrors how
paper_UncertaintyAwareClustering's own exp_02.ipynb uses this function (E=12 rounds per
client in the original federated-training loop), not a new convention introduced here.

`plant_mode` (correction-plan item 5 / Phase 1B): "linear" (default) rolls out the
client's TRUE linear plant, matching Phase 1A. "nonlinear_tanh" rolls out the
nonlinear tanh-tire-force plant instead (`src.nonlinear_plant`) while RLS still fits a
LINEAR model to the resulting data - identification remains "unchanged" in the sense
the correction plan means (same estimator, same assumed model structure), but now sees
plant/model-mismatched data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

from src.ifac_bridge import Client, pack_theta_AB, run_episode_update_rls
from src.nonlinear_plant import run_episode_update_rls_nonlinear


@dataclass
class IdentificationResult:
    theta_hat: np.ndarray
    precision: np.ndarray  # W_raw: inverse-covariance: doc's Sigma_i is precision^{-1}
    A_hat: np.ndarray
    B_hat: np.ndarray
    diagnostics: Dict[str, Any]


def fit(
    client: Client,
    r_ref: np.ndarray,
    *,
    seeds: Sequence[int],
    noise_std: float = 0.0,
    process_noise: Sequence[float] = (0.0, 0.0),
    u_min: float = -0.5,
    u_max: float = 0.5,
    plant_mode: str = "linear",
    mu: float = 1.0,
    Ts: Optional[float] = None,
) -> IdentificationResult:
    """
    Run len(seeds) identification episodes in sequence: rolls out `client`'s TRUE plant
    under its CURRENT controller while updating its online RLS estimator, once per
    seed. This mutates `client.Ad_hat`, `Bd_hat`, `W_raw`, ... in place, and each call
    re-syncs the RLS parameter estimate to the client's just-updated Ad_hat/Bd_hat
    while continuing to refine the shared regressor covariance - that is
    `run_episode_update_rls`'s existing contract, not something this wrapper changes.
    Only the FINAL episode's result is returned. `Ts` is required when
    `plant_mode="nonlinear_tanh"` (needed for RK4 integration; the linear path doesn't
    need it, it's baked into Ad/Bd already).
    """
    if not seeds:
        raise ValueError("fit() requires at least one seed.")
    if plant_mode == "nonlinear_tanh" and Ts is None:
        raise ValueError("Ts is required when plant_mode='nonlinear_tanh'.")
    for seed in seeds:
        np.random.seed(seed)
        if plant_mode == "linear":
            run_episode_update_rls(
                client,
                r_ref,
                noise_std=noise_std,
                process_noise=list(process_noise),
                U_min=u_min,
                U_max=u_max,
            )
        elif plant_mode == "nonlinear_tanh":
            run_episode_update_rls_nonlinear(
                client,
                r_ref,
                Ts=Ts,
                noise_std=noise_std,
                process_noise=list(process_noise),
                U_min=u_min,
                U_max=u_max,
                mu=mu,
            )
        else:
            raise ValueError(f"Unknown plant_mode {plant_mode!r} (expected 'linear' or 'nonlinear_tanh').")
    return IdentificationResult(
        theta_hat=pack_theta_AB(client.Ad_hat, client.Bd_hat),
        precision=client.W_raw,
        A_hat=client.Ad_hat,
        B_hat=client.Bd_hat,
        diagnostics={
            "std_rows": client.std_rows,
            "P_hat": client.P_hat,
            "round_counter": client.round_counter,
        },
    )
