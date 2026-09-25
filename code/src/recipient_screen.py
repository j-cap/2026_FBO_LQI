"""
Recipient-side transfer gate g_i(xi) (Phase 1D methodological response to Phase 1C's
asymmetry finding: L_{i<-j} != L_{j<-i}, but d_ij^dyn is symmetric by construction -
correlating one against the other has a structural ceiling independent of sample size
or fleet design). Rather than inventing an ad hoc asymmetric distance metric (which
would couple the method to this specific fleet), this evaluates a candidate controller
directly on the RECIPIENT's own identified model before trusting it - naturally
directional, since it's evaluated at i using (Ad_hat_i, Bd_hat_i), not at j.

For a candidate xi (however it was proposed - by client j, by a shared/global search,
etc.), client i can synthesize K_i(xi) = LQI(Ad_hat_i, Bd_hat_i, Q(xi), R) and cheaply
ask, using ONLY its own identified model - no true-plant simulation, no other client's
data: is it nominally stable, what stability margin does it have, what steering demand
and sideslip/yaw response does the model predict? This module computes those margins;
turning them into a [0,1] gate g_i(xi) (thresholding logic) is deferred to the BO
implementation once Phase 1D confirms the margins actually predict transfer loss.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from src.ifac_bridge import Client
from src.interfaces.lqi_synthesizer import synthesize
from src.interfaces.plant_client import PlantClient

# Effectively unclipped - this predicts RAW steering DEMAND before actuator
# saturation would hide how aggressive the recipient's model wants to be, not the
# clipped input an actual deployment would apply.
_UNCLIPPED = 100.0


@dataclass
class RecipientMargins:
    nominal_feasible: bool
    spectral_radius: Optional[float]
    max_abs_beta_hat: Optional[float]
    max_abs_delta_hat: Optional[float]
    max_abs_delta_rate_hat: Optional[float]


def predict_recipient_margins(
    client_i: Client,
    xi: Sequence[float],
    t: np.ndarray,
    r_ref: np.ndarray,
    *,
    state_scale: Sequence[float] = (1.0, 1.0, 1.0),
    fixed_R: float = 1.0,
) -> RecipientMargins:
    """
    Synthesizes K_i(xi) from client_i's OWN identified model (Ad_hat, Bd_hat) and
    rolls xi out on that SAME identified (linear) model - zero noise, deterministic,
    unclipped input - to predict how the candidate WOULD behave for this recipient,
    without touching the recipient's true plant, any other client's data, or an
    already-known "good" answer. Cheap: one Riccati solve (already returns the
    closed-loop spectral radius) plus one linear rollout, reusing `PlantClient.rollout`
    via a shallow-copied "shadow client" whose Ad_true/Bd_true are set to
    client_i.Ad_hat/Bd_hat (client_i itself is never mutated).
    """
    design = synthesize(client_i.Ad_hat, client_i.Bd_hat, xi, state_scale=state_scale, fixed_R=fixed_R)
    if not design.nominal_feasible:
        return RecipientMargins(False, None, None, None, None)

    shadow = copy.copy(client_i)
    shadow.Ad_true, shadow.Bd_true = client_i.Ad_hat, client_i.Bd_hat
    shadow.Kx, shadow.Ki, shadow.k_r = design.Kx, design.Ki, design.k_r

    rollout = PlantClient(shadow).rollout(
        t, r_ref, seed=0, noise_std=0.0, process_noise=(0.0, 0.0),
        u_min=-_UNCLIPPED, u_max=_UNCLIPPED, plant_mode="linear",
    )
    spectral_radius = float(np.max(np.abs(design.closed_loop_eigs)))
    max_abs_beta = float(np.max(np.abs(rollout.x[:, 0])))
    max_abs_delta = float(np.max(np.abs(rollout.u)))
    max_abs_delta_rate = float(np.max(np.abs(np.diff(rollout.u)))) if rollout.u.size > 1 else 0.0

    return RecipientMargins(True, spectral_radius, max_abs_beta, max_abs_delta, max_abs_delta_rate)
