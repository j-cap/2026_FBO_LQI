"""
xi -> Q(xi) -> LQI (Kx, Ki, k_r) synthesis (doc §6.1-6.2, §9.1's LQISynthesizer).

Q(xi) = diag(10**xi) in NORMALIZED coordinates, R fixed - the log-diagonal
parameterization the plan doc recommends so BO later searches over a well-scaled,
always-positive weight space. This is a thin transform in front of the IFAC pipeline's
own `design_lqi`, which already builds the integral-augmented system and a static DC
prefilter - nothing about the LQI math itself is reimplemented here.

State normalization (correction-plan item 1, doc §6.3): `state_scale` = (beta_s, r_s,
z_s) are DIVISORS, x_tilde = [beta/beta_s, r/r_s, z/z_s] (see
scripts/compute_state_scales.py). Cost in normalized coordinates is
x_tilde^T Q(xi) x_tilde = x^T diag(1/s_i) Q(xi) diag(1/s_i) x, i.e. the RAW diagonal
weight is q_i(xi) = 10**xi_i / s_i**2 (DIVIDE by s_i**2, not multiply) - so it folds
directly into `design_lqi`'s existing scalar q_beta/q_r/q_int arguments without needing
a coordinate transform of the returned gains, and without touching `design_lqi`'s
internals (which only accept a diagonal Q by construction anyway).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from src.ifac_bridge import design_lqi

_C_R = np.array([[0.0, 1.0]])  # yaw-rate output row, matches design_lqi's own C_r


@dataclass
class ControllerDesign:
    Kx: Optional[np.ndarray]
    Ki: Optional[np.ndarray]
    k_r: Optional[float]
    Q_diag_raw: Optional[np.ndarray]
    R: float
    closed_loop_eigs: Optional[np.ndarray]
    nominal_feasible: bool
    synthesis_failure_reason: Optional[str]


def synthesize(
    A: np.ndarray,
    B: np.ndarray,
    xi: Sequence[float],
    *,
    state_scale: Sequence[float] = (1.0, 1.0, 1.0),
    fixed_R: float = 1.0,
    spectral_radius_margin: float = 0.01,
) -> ControllerDesign:
    """xi in R^3 -> Q(xi)=diag(10**xi) (normalized) -> design_lqi(A, B, ..., rho=fixed_R)."""
    xi = np.asarray(xi, dtype=float).reshape(3)
    s = np.asarray(state_scale, dtype=float).reshape(3)
    q_raw = np.power(10.0, xi) / (s**2)

    try:
        Kx, Ki, k_r = design_lqi(A, B, q_beta=q_raw[0], q_r=q_raw[1], q_int=q_raw[2], rho=fixed_R)
    except (np.linalg.LinAlgError, ValueError) as exc:
        # design_lqi's Riccati solve (scipy.linalg.solve_discrete_are) fails two
        # different ways for a numerically bad (A, B): a plain LinAlgError for a
        # singular/degenerate case, or - found running Phase 1B, where an under-fit
        # RLS estimate from mismatched (nonlinear-plant) identification data is more
        # likely to produce an ill-conditioned (A, B) than Phase 1A ever did - a bare
        # ValueError from scipy's internal `ordqz` generalized-Schur reordering when
        # the problem is "too ill-conditioned" to reorder reliably. Both mean the same
        # thing here: this candidate's Riccati equation has no reliable solution, so
        # it's a synthesis failure like any other, not a crash.
        return ControllerDesign(None, None, None, q_raw, fixed_R, None, False, f"riccati_failure: {exc}")

    if not (np.all(np.isfinite(Kx)) and np.all(np.isfinite(Ki)) and np.isfinite(k_r)):
        return ControllerDesign(Kx, Ki, k_r, q_raw, fixed_R, None, False, "non_finite_gain")

    # Verify against the SAME augmented (beta, r, integrator) closed loop design_lqi
    # actually solved the Riccati equation for - not just the 2-state feedback loop.
    n = A.shape[0]
    A_aug = np.block([[A, np.zeros((n, 1))], [_C_R, np.eye(1)]])
    B_aug = np.vstack([B, np.zeros((1, 1))])
    K_aug = np.hstack([Kx, Ki])
    Acl_aug = A_aug - B_aug @ K_aug
    eigs = np.linalg.eigvals(Acl_aug)
    spectral_radius = float(np.max(np.abs(eigs)))

    if not np.isfinite(spectral_radius) or spectral_radius >= 1.0 - spectral_radius_margin:
        return ControllerDesign(
            Kx, Ki, k_r, q_raw, fixed_R, eigs, False,
            f"unstable_closed_loop: spectral_radius={spectral_radius:.4f}",
        )

    return ControllerDesign(Kx, Ki, float(k_r), q_raw, fixed_R, eigs, True, None)
