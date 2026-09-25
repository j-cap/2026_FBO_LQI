"""
PlantClient adapter (doc §9.1): wraps an IFAC `Client` + its true plant so Phase 0/1
code has one place to call for closed-loop rollouts and true-fleet metadata, instead of
reaching into `src2.run` / `Client` fields directly.

Only `rollout()` and `get_true_metadata()` are implemented - Phase 0/1 only ever needs
full-episode evaluation, never manual step-by-step control, so a stateful step()/reset()
API is left until a later phase actually needs it.

`plant_mode` (correction-plan item 5 / Phase 1B): "linear" (default) rolls out
`client.Ad_true/Bd_true` via `src2.run.run_CL_client_model`, matching Phase 1A exactly.
"nonlinear_tanh" instead rolls out `client.params` through
`src.nonlinear_plant`'s tanh-saturated tire-force ODE (RK4) - the SAME controller
(Kx, Ki, k_r, still designed from the linear identified model), a deliberately
structural plant-vs-model mismatch rather than a new controller-design path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np

from src.ifac_bridge import Client, run_CL_client_model
from src.nonlinear_plant import run_cl_client_model_nonlinear


@dataclass
class RolloutResult:
    t: np.ndarray
    r_ref: np.ndarray
    y: np.ndarray
    u: np.ndarray
    x: np.ndarray


class PlantClient:
    """Thin rollout wrapper around one IFAC `Client`."""

    def __init__(self, client: Client):
        self.client = client

    def rollout(
        self,
        t: np.ndarray,
        r_ref: np.ndarray,
        *,
        seed: int,
        noise_std: float = 0.0,
        process_noise: Sequence[float] = (0.0, 0.0),
        u_min: float = -0.5,
        u_max: float = 0.5,
        plant_mode: str = "linear",
        mu: float = 1.0,
    ) -> RolloutResult:
        """Roll out the client's TRUE plant under its CURRENTLY assigned (Kx, Ki, k_r)."""
        np.random.seed(seed)
        if plant_mode == "linear":
            out = run_CL_client_model(
                self.client,
                r_ref,
                noise_std=noise_std,
                process_noise=list(process_noise),
                u_min=u_min,
                u_max=u_max,
            )
        elif plant_mode == "nonlinear_tanh":
            Ts = float(np.asarray(t)[1] - np.asarray(t)[0])
            out = run_cl_client_model_nonlinear(
                self.client,
                r_ref,
                Ts=Ts,
                noise_std=noise_std,
                process_noise=list(process_noise),
                u_min=u_min,
                u_max=u_max,
                mu=mu,
            )
        else:
            raise ValueError(f"Unknown plant_mode {plant_mode!r} (expected 'linear' or 'nonlinear_tanh').")
        # `run_CL_client_model` pre-allocates length-N arrays but its loop only fills
        # indices 0..N-2 (`for k in range(N-1)`), leaving y_hist[-1]/u_hist[-1] at their
        # np.zeros() initial value rather than a simulated one. Trim that fictitious
        # last sample here, once, so no downstream RMSE/constraint computation is
        # biased by comparing a real r_ref[-1] against a fake zero residual.
        return RolloutResult(
            t=np.asarray(t)[:-1],
            r_ref=np.asarray(r_ref)[:-1],
            y=out["y_hist"][:-1],
            u=out["u_hist"][:-1],
            x=out["x_hist"][:-1],
        )

    def get_true_metadata(self) -> Dict[str, Any]:
        """Simulation-only ground truth (never available to a real deployed method)."""
        return {
            "client_id": self.client.client_id,
            "true_regime": self.client.cluster.name,
            "true_params": self.client.params,
            "Ad_true": self.client.Ad_true,
            "Bd_true": self.client.Bd_true,
        }
