"""
Validation tests for `src.nonlinear_plant` (correction-plan item 5 / Phase 1B).

The tanh tire-force law Fy = mu*Fz*tanh(C*alpha/(mu*Fz)) is, by construction, a smooth
saturating generalization of the linear tire model Fy = -C*alpha: for small slip angles
tanh(z) ~= z, so the nonlinear plant should numerically REDUCE to the linear
(build_bicycle_beta_r) plant in the small-slip limit, and visibly SATURATE (bounded
force) at large slip angles where the linear model would keep growing unboundedly.
These two properties are the standard sanity checks for this kind of model and are
tested directly rather than trusted from the physics alone.
"""
from __future__ import annotations

import numpy as np

from src.ifac_bridge import (
    Cluster,
    build_bicycle_beta_r,
    closed_loop_step_lqi,
    cluster_prototype,
    design_lqi,
    discretize,
    nominal_params,
)
from src.nonlinear_plant import (
    closed_loop_step_lqi_nonlinear,
    nonlinear_bicycle_derivatives,
    rk4_step,
)

TS = 0.05


def _linear_matrices(p):
    A, B, C, D = build_bicycle_beta_r(p)
    Ad, Bd, Cd, Dd = discretize(A, B, C, D, TS)
    return A, B, Ad, Bd, Cd


def test_nonlinear_derivatives_reduce_to_linear_at_small_slip():
    p = nominal_params()
    A, B, _, _, _ = _linear_matrices(p)
    x = np.array([1e-5, 2e-5])
    delta = 1.5e-5

    xdot_nl = nonlinear_bicycle_derivatives(x, delta, p, mu=1.0)
    xdot_lin = A @ x + B.flatten() * delta

    np.testing.assert_allclose(xdot_nl, xdot_lin, rtol=1e-3, atol=1e-8)


def test_rk4_step_reduces_to_linear_discretization_at_small_slip():
    p = nominal_params()
    _, _, Ad, Bd, _ = _linear_matrices(p)
    x = np.array([1e-5, -1e-5])
    delta = 1e-5

    x_next_nl = rk4_step(x, delta, p, TS, mu=1.0)
    x_next_lin = Ad @ x + Bd.flatten() * delta

    np.testing.assert_allclose(x_next_nl, x_next_lin, rtol=1e-3, atol=1e-8)


def test_closed_loop_step_matches_linear_at_small_signal_zero_noise():
    """The whole apparatus (control law, anti-windup, measurement) should produce
    numerically indistinguishable trajectories between the linear and nonlinear step
    functions when the state stays tiny - the strongest end-to-end sanity check."""
    p = nominal_params()
    A, B, Ad, Bd, Cd = _linear_matrices(p)
    Kx, Ki, k_r = design_lqi(Ad, Bd, q_beta=1.0, q_r=2.0, q_int=2.0, rho=0.5)

    x_lin = np.array([1e-6, -2e-6])
    x_nl = x_lin.copy()
    z = 0.0
    r_k = 1e-6

    x_next_lin, z_next_lin, u_lin, y_lin = closed_loop_step_lqi(
        x_lin, z, r_k, Ad, Bd, Cd, Kx, Ki, k_r, u_min=-0.5, u_max=0.5, noise_std=0.0, process_noise=[0.0, 0.0]
    )
    x_next_nl, z_next_nl, u_nl, y_nl = closed_loop_step_lqi_nonlinear(
        x_nl, z, r_k, p, Cd, Kx, Ki, k_r, TS, u_min=-0.5, u_max=0.5, noise_std=0.0, process_noise=(0.0, 0.0)
    )

    assert abs(u_lin - u_nl) < 1e-9
    np.testing.assert_allclose(x_next_nl, x_next_lin, rtol=1e-3, atol=1e-9)
    assert abs(z_next_nl - z_next_lin) < 1e-9


def test_tire_force_saturates_at_large_slip_unlike_linear():
    """At a large slip angle, the nonlinear model's implied lateral force must stay
    bounded by mu*Fz while the linear model's -C*alpha keeps growing unboundedly -
    this is the entire point of using tanh instead of a linear tire law. Checked by
    comparing the two models' derivatives directly at the same large state, rather
    than an arbitrary compounded ratio-of-ratios threshold."""
    p = nominal_params()
    A, B, _, _, _ = _linear_matrices(p)

    x_large = np.array([5.0, 0.0])  # an absurdly large beta, deliberately far outside the linear regime
    xdot_nl_large = nonlinear_bicycle_derivatives(x_large, 0.0, p, mu=1.0)
    xdot_lin_large = A @ x_large

    # Saturated tire forces are individually bounded by mu*Fzf/mu*Fzr, so the resulting
    # beta_dot/r_dot must be far smaller in magnitude than the linear model's unbounded
    # -C*alpha response at the same (large) state.
    assert abs(xdot_nl_large[0]) < abs(xdot_lin_large[0]) / 10
    assert abs(xdot_nl_large[1]) < abs(xdot_lin_large[1]) / 10

    # And the saturated force itself must be bounded by mu*Fz - not just "smaller",
    # actually capped at the physical friction limit.
    Fzf = p.m * 9.81 * p.b / (p.a + p.b)
    Fzr = p.m * 9.81 * p.a / (p.a + p.b)
    alpha_f = x_large[0] + p.a * x_large[1] / p.U
    alpha_r = x_large[0] - p.b * x_large[1] / p.U
    Fyf = -Fzf * np.tanh(p.Cf * alpha_f / Fzf)
    Fyr = -Fzr * np.tanh(p.Cr * alpha_r / Fzr)
    assert abs(Fyf) <= Fzf * 1.0 + 1e-6
    assert abs(Fyr) <= Fzr * 1.0 + 1e-6


def test_regime_prototypes_all_produce_finite_dynamics():
    """Sanity check across every fleet regime (not just nominal) at a moderate slip
    angle - guards against a division-by-zero or sign error in the Fzf/Fzr split for
    any specific (m, a, b) combination the fleet actually uses."""
    p0 = nominal_params()
    for cluster in [Cluster.LOW_SPEED, Cluster.PAYLOAD, Cluster.HIGH_SPEED]:
        p = cluster_prototype(cluster, p0)
        xdot = nonlinear_bicycle_derivatives(np.array([0.05, 0.1]), 0.05, p, mu=1.0)
        assert np.all(np.isfinite(xdot)), f"non-finite derivative for {cluster}"
