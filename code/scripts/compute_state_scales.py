"""
Derives the (beta_s, r_s, z_s) state-normalization scales used by
`src.interfaces.lqi_synthesizer.synthesize` (plan doc §6.3; correction-plan item 1).

Q(xi) = diag(10**xi) is defined in NORMALIZED coordinates x_tilde = [beta/beta_s,
r/r_s, z/z_s] - so a scale should reflect the PHYSICAL magnitude that state channel
actually reaches, not an arbitrary constant. This script gets that from the
"fleet-wide trajectory envelope": roll out EVERY regime's cluster-prototype plant
(not just the single nominal-params plant) under the IFAC pipeline's own hand-tuned
reference controller (Q_beta=1, Q_r=2, Q_int=2, R=1 -> the paper's
hand_tuned_reference_xi) over both lane-change half-durations in the fleet's
`lane_change_time` list, and takes the max |beta|, |r|, |z| observed across ALL of
that - not just the nominal (U=15 m/s) plant.

BUG HISTORY: the first version of this script only used `nominal_params()` (U=15
m/s). That under-estimated max|beta| for the HIGH_SPEED regime (U=20 m/s, which has
genuinely different beta dynamics at the same yaw-rate maneuver), making the derived
beta_max_deg systematically too tight for that regime - 7/20 HIGH_SPEED clients ended
up with ZERO feasible Sobol candidates out of 256 in `phase1a_matched_model_full`
(246/256 rejected specifically on `beta_limit`), and a client with only 1 feasible
candidate became a degenerate "individual optimum" that corrupted the transfer-loss
analysis when transplanted onto other clients. See
docs/phase1_diagnostic_findings.md's "outlier donor" investigation.

This is a one-time, config-time precomputation (not run inside the Phase-1 driver) -
its output is meant to be transcribed into config/controller.yaml's `state_scale`
field, with this script's invocation recorded there for reproducibility.

Usage:
    python scripts/compute_state_scales.py --config config/controller.yaml --fleet-config config/fleet.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from src.ifac_bridge import (  # noqa: E402
    Cluster,
    build_bicycle_beta_r,
    closed_loop_step_lqi,
    cluster_prototype,
    design_lqi,
    discretize,
    init_integrator,
    lc_yaw_rate_ref,
    nominal_params,
)
from src.io_utils import load_yaml  # noqa: E402


def envelope_for_T_lc(Ad, Bd, Cd, Kx, Ki, k_r, t, r_ref, *, u_min, u_max, seed=0):
    """Rolls out the closed loop (nominal plant, nominal controller, ZERO noise - this
    is meant to capture the deterministic/expected envelope, not a noisy one) and
    returns (max|beta|, max|r_tracking_output|, max|z|)."""
    np.random.seed(seed)
    N = len(r_ref)
    x = np.zeros(2)
    z = init_integrator(Kx, Ki, k_r, x, r_ref, preview=False)
    beta_hist = np.zeros(N)
    z_hist = np.zeros(N)
    beta_hist[0] = x[0]
    z_hist[0] = z
    for k in range(N - 1):
        x_next, z_next, u, y = closed_loop_step_lqi(
            x, z, float(r_ref[k]), Ad, Bd, Cd, Kx, Ki, k_r,
            u_min=u_min, u_max=u_max, k_aw=0.2, leak=0.002, noise_std=0.0, process_noise=[0.0, 0.0],
        )
        x, z = x_next, z_next
        beta_hist[k + 1] = x[0]
        z_hist[k + 1] = z
    return float(np.max(np.abs(beta_hist))), float(np.max(np.abs(r_ref))), float(np.max(np.abs(z_hist)))


def main(controller_config_path: str, fleet_config_path: str) -> None:
    controller_cfg = load_yaml(controller_config_path)
    fleet_cfg = load_yaml(fleet_config_path)

    p0 = nominal_params()
    regime_params = {"nominal": p0}
    for name in fleet_cfg["regimes"]:
        regime_params[name] = cluster_prototype(Cluster[name], p0)

    hand_tuned_xi = np.asarray(controller_cfg["hand_tuned_reference_xi"], float)
    q = 10.0**hand_tuned_xi

    beta_max, r_max_seen, z_max = 0.0, 0.0, 0.0
    worst = {"beta": None, "r": None, "z": None}
    for regime_name, p in regime_params.items():
        A, B, C, D = build_bicycle_beta_r(p)
        Ad, Bd, Cd, Dd = discretize(A, B, C, D, fleet_cfg["Ts"])
        # LQI gains from EACH regime's own plant, matching how design_lqi is actually
        # used downstream (controller design tracks the identified/regime model, not a
        # single shared nominal one) - not strictly necessary for r_s (bounded by
        # r_max regardless) but matters for the beta/z envelope each regime's own
        # closed loop actually produces.
        Kx, Ki, k_r = design_lqi(Ad, Bd, q_beta=q[0], q_r=q[1], q_int=q[2], rho=controller_cfg["fixed_R"])
        for T_lc in fleet_cfg["lane_change_time"]:
            t, r_ref = lc_yaw_rate_ref(
                fleet_cfg["Ts"], controller_cfg["T_total"], controller_cfg["T0"], T_lc,
                controller_cfg["r_max"], tfilter=controller_cfg.get("tfilter"),
            )
            beta_env, r_env, z_env = envelope_for_T_lc(
                Ad, Bd, Cd, Kx, Ki, k_r, t, r_ref,
                u_min=controller_cfg["u_min"], u_max=controller_cfg["u_max"],
            )
            print(f"{regime_name:10s} T_lc={T_lc}: max|beta|={beta_env:.6f} rad, max|r|={r_env:.6f} rad/s, max|z|={z_env:.6f}")
            if beta_env > beta_max:
                beta_max, worst["beta"] = beta_env, f"{regime_name}, T_lc={T_lc}"
            if r_env > r_max_seen:
                r_max_seen, worst["r"] = r_env, f"{regime_name}, T_lc={T_lc}"
            if z_env > z_max:
                z_max, worst["z"] = z_env, f"{regime_name}, T_lc={T_lc}"

    print()
    print("Derived state_scale (beta_s, r_s, z_s), max over ALL regimes + both T_lc:")
    print(f"  beta_s = {beta_max:.6f}  (worst case: {worst['beta']})")
    print(f"  r_s    = {r_max_seen:.6f}  (worst case: {worst['r']})")
    print(f"  z_s    = {z_max:.6f}  (worst case: {worst['z']})")
    print()
    print("Transcribe into config/controller.yaml's state_scale field as:")
    print(f"  state_scale: [{beta_max:.6f}, {r_max_seen:.6f}, {z_max:.6f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/controller.yaml")
    parser.add_argument("--fleet-config", default="config/fleet.yaml")
    args = parser.parse_args()
    main(args.config, args.fleet_config)
