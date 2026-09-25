"""
Phase 1B minimal structural mismatch (correction-plan item 5): a smooth nonlinear
lateral-tire-force law replacing the TRUE plant's linear tire model, while RLS,
clustering, (theta_hat, Sigma_i), and LQI synthesis stay exactly as they are - they
still identify/design against a LINEAR (beta, r) model. This creates a controlled
model-to-plant gap without rebuilding the pipeline around a new state representation.

    Fy = mu * Fz * tanh(C_alpha * alpha / (mu * Fz))    (saturating tire force)

Uses the SAME 2-state (beta, r) representation and the SAME control law
(`src2.utils.closed_loop_step_lqi`'s u = -Kx@x - Ki*z + k_r*r_k, anti-windup, noise) as
the linear plant - only the state UPDATE (beta_dot, r_dot) is computed from nonlinear
tire forces instead of the linear Ad@x + Bd@u map, integrated via RK4 over one Ts.

Static front/rear normal load split (Fzf, Fzr) from CG position (a, b) is used since
BikeParams carries no separate load/height parameters - a deliberately minimal
addition, not a full nonlinear vehicle model (no load transfer, no combined-slip
coupling with longitudinal force).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from src.ifac_bridge import BikeParams, Client, pack_theta_AB

_C_R = np.array([[0.0, 1.0]])  # yaw-rate output row, matches src2.utils.design_lqi's C_r
_GRAVITY = 9.81


def nonlinear_bicycle_derivatives(x: np.ndarray, delta: float, p: BikeParams, *, mu: float = 1.0) -> np.ndarray:
    """Continuous-time dx/dt for x=[beta, r], nonlinear (tanh-saturated) tire forces."""
    beta, r = float(x[0]), float(x[1])
    Fzf = p.m * _GRAVITY * p.b / (p.a + p.b)
    Fzr = p.m * _GRAVITY * p.a / (p.a + p.b)
    alpha_f = beta + p.a * r / p.U - delta
    alpha_r = beta - p.b * r / p.U
    Fyf = -mu * Fzf * np.tanh(p.Cf * alpha_f / (mu * Fzf))
    Fyr = -mu * Fzr * np.tanh(p.Cr * alpha_r / (mu * Fzr))
    beta_dot = (Fyf + Fyr) / (p.m * p.U) - r
    r_dot = (p.a * Fyf - p.b * Fyr) / p.Iz
    return np.array([beta_dot, r_dot])


def rk4_step(x: np.ndarray, delta: float, p: BikeParams, Ts: float, *, mu: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    k1 = nonlinear_bicycle_derivatives(x, delta, p, mu=mu)
    k2 = nonlinear_bicycle_derivatives(x + 0.5 * Ts * k1, delta, p, mu=mu)
    k3 = nonlinear_bicycle_derivatives(x + 0.5 * Ts * k2, delta, p, mu=mu)
    k4 = nonlinear_bicycle_derivatives(x + Ts * k3, delta, p, mu=mu)
    return x + (Ts / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def closed_loop_step_lqi_nonlinear(
    x, z, r_k, p: BikeParams, Cd, Kx, Ki, k_r, Ts, *,
    u_min=-0.5, u_max=0.5, k_aw=0.2, leak=0.0,
    noise_std=0.0, process_noise=(0.0, 0.0), mu=1.0,
):
    """Mirrors `src2.utils.closed_loop_step_lqi`'s control law EXACTLY (same
    anti-windup, same measurement model) - only the plant update is different (RK4 over
    the nonlinear ODE instead of Ad@x+Bd@u)."""
    x = np.asarray(x, dtype=float)
    u_unsat = -float((Kx @ x + Ki * z).item()) + float(k_r) * float(r_k)
    u = float(np.clip(u_unsat, u_min, u_max))
    y = Cd @ x + noise_std * np.random.randn(Cd.shape[0])
    if Cd.shape[0] == 1:
        e = y[0] - float(r_k)
    elif Cd.shape[0] == 2:
        e = y[1] - float(r_k)
    else:
        raise ValueError("Cd has unsupported number of outputs")
    z_next = ((1.0 - leak) * z + e + k_aw * (u - u_unsat)).item()
    x_next = rk4_step(x, u, p, Ts, mu=mu) + np.random.multivariate_normal(np.zeros(2), np.diag(process_noise))
    return x_next, z_next, u, y


def run_cl_client_model_nonlinear(
    client: Client, r_ref: np.ndarray, *, Ts: float, init_x=None, use_preview: bool = False,
    noise_std: float = 0.0, process_noise: Sequence[float] = (0.0, 0.0),
    u_min: float = -0.5, u_max: float = 0.5, mu: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Mirrors `src2.run.run_CL_client_model` exactly (same array layout - including
    the fictitious un-set last sample `PlantClient.rollout` trims), but rolls out the
    NONLINEAR true plant (`client.params`, tanh tire law) under the client's CURRENT
    (Kx, Ki, k_r) instead of the linear Ad_true/Bd_true update. No RLS/model update."""
    p = client.params
    Kx, Ki, k_r = client.Kx, client.Ki, client.k_r
    Ki_s = float(Ki)

    N = len(r_ref)
    x = np.zeros(2) if init_x is None else np.asarray(init_x, dtype=float).copy()
    rr0 = float(r_ref[1] if (use_preview and N > 1) else r_ref[0])
    z = (-(Kx @ x).item() + k_r * rr0) / Ki_s

    y_hist = np.zeros(N)
    u_hist = np.zeros(N)
    x_hist = np.zeros((N, 2))
    x_hist[0, :] = x

    for k in range(N - 1):
        rr = float(r_ref[k + 1] if use_preview else r_ref[k])
        x_next, z_next, u, y = closed_loop_step_lqi_nonlinear(
            x, z, rr, p, _C_R, Kx, Ki, k_r, Ts,
            u_min=u_min, u_max=u_max, k_aw=0.2, leak=0.002,
            noise_std=noise_std, process_noise=process_noise, mu=mu,
        )
        y_hist[k] = float(y[0])
        u_hist[k] = float(u)
        x, z = x_next, z_next
        x_hist[k + 1] = x

    return {"x_hist": x_hist, "y_hist": y_hist, "u_hist": u_hist}


def run_episode_update_rls_nonlinear(
    client: Client, r_ref: np.ndarray, *, Ts: float, init_x=None, use_preview: bool = False,
    noise_std: float = 0.0, process_noise: Sequence[float] = (0.0, 0.0),
    U_min: float = -0.5, U_max: float = 0.5, mu: float = 1.0,
) -> None:
    """Mirrors `src2.run.run_episode_update_rls` exactly (same client-mutation
    contract), but rolls out the NONLINEAR true plant (`client.params`) instead of the
    linear Ad_true/Bd_true update, while RLS still fits a LINEAR x_{k+1}=Ad@x_k+Bd@u_k
    model to the resulting (x_k, u_k, x_{k+1}) data - this mismatch between what's
    simulated and what's identified IS Phase 1B's structural gap."""
    p = client.params
    Ad_hat, Bd_hat = client.Ad_hat, client.Bd_hat
    Kx, Ki, k_r = client.Kx, client.Ki, client.k_r
    Ki_s = float(Ki.squeeze())

    N = len(r_ref)
    x = np.zeros(2) if init_x is None else np.asarray(init_x, dtype=float).copy()
    r0 = float(r_ref[1] if (use_preview and N > 1) else r_ref[0])
    z = (-(Kx @ x).item() + k_r * r0) / Ki_s

    rls_new = client.rls
    rls_new.reset_theta(pack_theta_AB(Ad_hat, Bd_hat))

    for k in range(N - 1):
        rr = float(r_ref[k + 1] if use_preview else r_ref[k])
        x_next, z_next, u, y = closed_loop_step_lqi_nonlinear(
            x, z, rr, p, _C_R, Kx, Ki, k_r, Ts,
            u_min=U_min, u_max=U_max, k_aw=0.2, leak=0.002,
            noise_std=noise_std, process_noise=process_noise, mu=mu,
        )
        rls_new.update(u_k=u, x_k=x, x_kp1=x_next)
        x, z = x_next, z_next

    Ad_hat, Bd_hat = rls_new.get_AB()
    client.Ad_hat = Ad_hat
    client.Bd_hat = Bd_hat
    client.P_hat = rls_new.P
    client.std_rows = np.array([rls_new.var_rows[0].std, rls_new.var_rows[1].std], float)
    W_raw = rls_new.get_W()
    client.W_A = W_raw[:2, :2]
    client.W_B = W_raw[2:, 2:]
    client.W_raw = W_raw
    client.theta_hist_RLS.append(rls_new.get_theta().copy())
    client.Ad_hist_RLS.append(Ad_hat.copy())
    client.Bd_hist_RLS.append(Bd_hat.copy())
    client.round_counter += 1
