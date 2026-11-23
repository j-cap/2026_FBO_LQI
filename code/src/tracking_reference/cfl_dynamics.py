from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np

# Unused relative utils import removed for this standalone reference.

# ---------- helpers ----------

def magic_formula_norm(x: float, B: float, C: float, E: float) -> float:
    """
    Pacejka 'magic formula' normalized output in [-1,1], using the common structure:
      y = sin( C * atan( B*x*(1-E) + E*atan(B*x) ) )
    """
    x = float(x)
    return float(np.sin(C * np.arctan(B * x * (1.0 - E) + E * np.arctan(B * x))))

def lateral_tire_force_pacejka(alpha: float, Fz: float, mu: float, lat: Dict[str, float]) -> float:
    """
    Lateral tire force via normalized Pacejka Magic Formula: Fy = -Fmax * MF(alpha),
    Fmax = mu*Fz.

    NOTE (bugfix 2026-08): this replaces the former tire_forces_pacejka(alpha, kappa,
    ..., lon), which also routed a longitudinal slip ratio `kappa` through a *second*
    Pacejka MF to get Fx. That was wrong: Fx_req (the axle-load-weighted share of
    m*ax_cmd, computed in dbm_ct below) was already the target longitudinal force --
    re-running it through a nonlinear MF (kappa = Fx_req/(mu*Fz) fed in as if it were a
    slip ratio) silently attenuated/amplified Fx even when (Fx_req, Fy) was well inside
    the friction circle (confirmed empirically: ~10-15x over-realization in this
    fleet's normal operating regime, since MF's near-origin slope is B*C =~ 10-11, not
    1). dbm_ct now uses Fx_req directly as the pre-friction-circle Fx and lets
    friction_circle_clip (unchanged, was never the bug) do the only nonlinear
    saturation step.
    """
    Fmax = float(max(0.0, mu * Fz))
    return -Fmax * magic_formula_norm(alpha, lat["B"], lat["C"], lat["E"])

def friction_circle_clip(Fx: float, Fy: float, Fmax: float) -> Tuple[float, float]:
    """
    Project (Fx, Fy) onto a friction circle with radius Fmax.
    (This is a simple friction ellipse with equal axes.)
    """
    Fmax = float(max(0.0, Fmax))
    if Fmax <= 0.0:
        return 0.0, 0.0
    s2 = (Fx * Fx + Fy * Fy) / (Fmax * Fmax)
    if (not np.isfinite(s2)) or s2 <= 1.0:
        return float(Fx), float(Fy)
    s = 1.0 / np.sqrt(s2)
    return float(Fx * s), float(Fy * s)
# ---------- main DBM CT evaluation ----------


def dbm_ct(params: Dict[str, Any], state: np.ndarray, control: np.ndarray, *, return_aux: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Continuous-time 6-state dynamic bicycle model with Pacejka + friction circle.

    State x = [X, Y, psi, vx, vy, r]
    Control u = [delta_f, ax_cmd]

    Required params keys (no fallback if missing; filled in by
    vehicle.py::normalize_dbm6_params_schema before this is called):
      m, I_z, l_f, l_r, mu, g, delta_r_offset, eps_vx, alpha_max,
      ax_clip_min, ax_clip_max
      pacejka_f_lat, pacejka_r_lat  (dicts with B,C,E)

    NOT read by dbm_ct (bugfix 2026-08 -- kept in the params schema/config for other
    reasons, don't remove them there):
      kappa_max          -- dead here since the direct-force fix; no slip ratio is
                             computed by this function anymore (see
                             lateral_tire_force_pacejka's docstring).
      pacejka_f_lon,
      pacejka_r_lon       -- still required by vehicle.py::normalize_dbm6_params_schema
                             and consumed directly by plotting.py's fleet-heterogeneity
                             PCA feature vector; just no longer physics-relevant here.
    Optional params (true optional, default applied here via .get()):
      oob_beta_max_deg (default 40.0)  # instability: sideslip angle bound
      oob_vx_min (default -2.0)        # instability: runaway reverse speed
      oob_r_max (default 5.0)          # instability: unrealistic yaw rate
    """
    x = np.asarray(state, dtype=float).reshape(6)
    u = np.asarray(control, dtype=float).reshape(2)

    X, Y, psi, vx, vy, r = map(float, x)
    delta_f, ax_cmd = map(float, u)

    m  = params["m"]
    Iz = params["I_z"]
    lf = params["l_f"]
    lr = params["l_r"]
    mu = params["mu"]
    g  = params["g"]

    delta_r = params["delta_r_offset"]

    eps_vx    = float(params["eps_vx"])
    alpha_max = float(params["alpha_max"])
    # kappa_max intentionally not read: no slip ratio is computed post-fix (see docstring).

    ax_min = float(params["ax_clip_min"])
    ax_max = float(params["ax_clip_max"])
    ax_cmd = float(np.clip(ax_cmd, ax_min, ax_max))

    # --- static normal loads (no load transfer)
    L = lf + lr
    if L <= 0:
        raise ValueError("Invalid geometry: l_f + l_r must be positive")
    Fz_f = m * g * lr / L
    Fz_r = m * g * lf / L

    # --- slip angles (bicycle conventions)
    # Use vx_safe with sign to keep continuity for reverse (if ever)
    vx_safe = float(max(abs(vx), eps_vx)) * (1.0 if vx >= 0 else -1.0)

    alpha_f = np.arctan2(vy + lf * r, vx_safe) - delta_f
    alpha_r = np.arctan2(vy - lr * r, vx_safe) - delta_r

    alpha_f = np.clip(alpha_f, -alpha_max, alpha_max)
    alpha_r = np.clip(alpha_r, -alpha_max, alpha_max)

    # --- longitudinal force request (direct-force formulation, bugfix 2026-08)
    # In this minimal model we do *not* have wheel angular speed, so there is no
    # physical slip ratio available. We request a total longitudinal force
    # Fx_total = m*ax_cmd and distribute it front/rear proportional to static
    # normal load. That per-axle Fx_req IS the pre-friction-circle longitudinal
    # force -- it is fed directly into friction_circle_clip below (together with
    # the Pacejka-derived lateral force), NOT routed through a second Pacejka
    # nonlinearity via an invented slip ratio (see lateral_tire_force_pacejka's
    # docstring for why the old kappa-based construction was wrong).
    w_f = Fz_f / (Fz_f + Fz_r)
    w_r = 1.0 - w_f
    Fx_req_f = m * ax_cmd * w_f
    Fx_req_r = m * ax_cmd * w_r

    # --- tire parameter blocks (lateral only; pacejka_f_lon/pacejka_r_lon are no
    # longer consumed here -- see docstring)
    f_lat = params["pacejka_f_lat"]
    r_lat = params["pacejka_r_lat"]

    # --- uncoupled forces: Fy via Pacejka(alpha), Fx == Fx_req directly (no MF applied to Fx)
    Fy_f_u = lateral_tire_force_pacejka(alpha=alpha_f, Fz=Fz_f, mu=mu, lat=f_lat)
    Fy_r_u = lateral_tire_force_pacejka(alpha=alpha_r, Fz=Fz_r, mu=mu, lat=r_lat)
    Fx_f_u, Fx_r_u = Fx_req_f, Fx_req_r

    # --- friction circle coupling per axle (unchanged -- was never the bug)
    Fx_f, Fy_f = friction_circle_clip(Fx_f_u, Fy_f_u, mu * Fz_f)
    Fx_r, Fy_r = friction_circle_clip(Fx_r_u, Fy_r_u, mu * Fz_r)

    # --- transform front/rear forces to body frame (steered front, fixed rear)
    cdf, sdf = float(np.cos(delta_f)), float(np.sin(delta_f))
    cdr, sdr = float(np.cos(delta_r)), float(np.sin(delta_r))

    Fx_body = Fx_f * cdf - Fy_f * sdf + Fx_r * cdr - Fy_r * sdr
    Fy_body = Fx_f * sdf + Fy_f * cdf + Fx_r * sdr + Fy_r * cdr

    # --- kinematics
    dot_X   = vx * float(np.cos(psi)) - vy * float(np.sin(psi))
    dot_Y   = vx * float(np.sin(psi)) + vy * float(np.cos(psi))
    dot_psi = r

    # --- dynamics in body frame
    # Standard rigid-body planar:
    #   vx_dot = Fx/m + r*vy
    #   vy_dot = Fy/m - r*vx
    dot_vx = Fx_body / m + r * vy
    dot_vy = Fy_body / m - r * vx

    # yaw moment from lateral body components at axles
    Fy_f_body = Fx_f * sdf + Fy_f * cdf
    Fy_r_body = Fx_r * sdr + Fy_r * cdr
    dot_r = (lf * Fy_f_body - lr * Fy_r_body) / Iz

    xdot = np.array([dot_X, dot_Y, dot_psi, dot_vx, dot_vy, dot_r], dtype=float)

    if not return_aux:
        return xdot, {}

    # body-frame accelerations (what an IMU would see, ignoring gravity projection)
    ax_body = dot_vx - r * vy
    ay_body = dot_vy + r * vx

    util_f = float(np.sqrt((Fx_f / max(1e-12, mu * Fz_f))**2 + (Fy_f / max(1e-12, mu * Fz_f))**2))
    util_r = float(np.sqrt((Fx_r / max(1e-12, mu * Fz_r))**2 + (Fy_r / max(1e-12, mu * Fz_r))**2))

    beta = float(np.arctan2(vy, max(abs(vx), 1e-6)))

    # --- out-of-bounds / instability detection
    # Thresholds are optional params (default to generous but finite bounds) so a
    # divergent/unstable rollout (NaN state, spin-out, runaway reverse speed, or
    # unrealistic yaw rate) can be flagged and truncated instead of silently
    # propagating garbage into the logged training data.
    oob_beta_max = float(np.deg2rad(params.get("oob_beta_max_deg", 40.0)))
    oob_vx_min = float(params.get("oob_vx_min", -2.0))
    oob_r_max = float(params.get("oob_r_max", 5.0))

    oob_flag = False
    oob_reason = ""
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(xdot))):
        oob_flag, oob_reason = True, "non-finite state or derivative"
    elif abs(beta) > oob_beta_max:
        oob_flag, oob_reason = True, f"|beta|={np.degrees(beta):.1f}deg exceeds oob_beta_max_deg={np.degrees(oob_beta_max):.1f}"
    elif vx < oob_vx_min:
        oob_flag, oob_reason = True, f"vx={vx:.2f} m/s below oob_vx_min={oob_vx_min:.2f} (runaway reverse/spin-out)"
    elif abs(r) > oob_r_max:
        oob_flag, oob_reason = True, f"|r|={r:.2f} rad/s exceeds oob_r_max={oob_r_max:.2f}"

    aux = {
        "X": X, "Y": Y, "psi": psi,
        "vx": vx, "vy": vy, "r": r,
        "delta_f": delta_f, "delta_r": delta_r,

        "alpha_f": alpha_f, "alpha_r": alpha_r,

        "Fz_f": float(Fz_f), "Fz_r": float(Fz_r),

        "Fx_f_unc": Fx_f_u, "Fy_f_unc": Fy_f_u,
        "Fx_r_unc": Fx_r_u, "Fy_r_unc": Fy_r_u,

        "Fx_f": float(Fx_f), "Fy_f": float(Fy_f),
        "Fx_r": float(Fx_r), "Fy_r": float(Fy_r),

        "Fx_body": float(Fx_body), "Fy_body": float(Fy_body),
        "ax": float(ax_body), "ay": float(ay_body),
        "beta": float(beta),

        "util_f": util_f, "util_r": util_r, "util": float(max(util_f, util_r)),
        "oob": (oob_flag, oob_reason),
    }
    return xdot, aux
