"""
Cheap T_lc effect audit (docs/phase2_diagnostic_findings.md, "Open items", user
request 2026-09-03): holding each client's TRUE physical plant (theta_i) fixed,
evaluate it under BOTH lane-change half-durations in `lane_change_time` and compare

  - baseline (hand-tuned xi) feasibility and RMSE
  - coarse feasible-support phi_i (same Sobol-scan estimate as Step D)
  - oracle RMSE (best feasible point over the scan)
  - Spearman rank correlation between the two T_lc settings' objective landscapes,
    evaluated on the SAME fixed Sobol xi set, restricted to xi points feasible under
    both settings

Answers: holding theta_i fixed, how much does T_lc change f_i(xi)? If baseline
feasibility/phi/oracle_rmse move materially between T1 and T2 for the SAME physical
client, T_lc is a hidden task confound entangled with the family/slot assignment
(every v1 baseline failure traced to the T_lc=2.0 half, docs/phase2_diagnostic_findings.md
Part 7) - not just a fleet-generation admission detail.

Uses v1 fleet generation (`generate_fixed_speed_fleet`) to get a representative set of
physical clients - v1, not v2, because this question is about T_lc's own effect on
f_i(xi), independent of the v2 admission criterion. `client.T_lc` is mutated in place
between the two evaluations (Client is an unfrozen dataclass) and the episode bank is
rebuilt each time (`build_episode_bank` reads `client.T_lc`).

IMPORTANT (same class of bug as Part 2's cross-seed cache collision): the joblib cache
key is (client_id, rounded_xi, episode_seeds, plant_version, kwargs) - it does NOT
encode T_lc, and `episode_seeds` doesn't change with T_lc either (they're derived from
client_id + base_seed only). Reusing one cache dir across T1/T2 for the same client_id
would silently reuse T1's cached RMSE for T2's query. Each (client_id, T_lc label) gets
its own cache subdirectory to avoid this.

Usage:
    python scripts/tlc_effect_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import qmc, spearmanr  # noqa: E402

from src.caching import make_cached_evaluate  # noqa: E402
from src.episodes import build_episode_bank, calibration_episodes  # noqa: E402
from src.fleet_families import generate_fixed_speed_fleet  # noqa: E402
from src.io_utils import load_yaml, write_run_manifest  # noqa: E402

RUN_TAG = "tlc_effect_audit"
FLEET_SEED = 42  # fleet_fixed_speed.yaml's own default seed - a representative draw
N_SOBOL = 32  # same scale as the campaign's n_oracle_sobol (Step D)
SOBOL_SEED = 0  # ONE fixed Xi set shared by every client and both T_lc settings
N_CALIBRATION_EPISODES = 3
EPISODE_BASE_SEED = 5000


def main() -> None:
    fleet_cfg = load_yaml(str(_ROOT / "config" / "fleet_fixed_speed.yaml"))
    controller_cfg = load_yaml(str(_ROOT / "config" / "controller.yaml"))
    # Match the actual phase2 campaign/validation controller_overrides.
    controller_cfg = dict(controller_cfg, plant_mode="nonlinear_tanh", tire_mu=1.0)

    T1, T2 = fleet_cfg["lane_change_time"]
    xi_lower = np.asarray(controller_cfg["xi_lower"], float)
    xi_upper = np.asarray(controller_cfg["xi_upper"], float)
    state_scale = controller_cfg["state_scale"]
    fixed_R = controller_cfg["fixed_R"]
    beta_max = np.deg2rad(controller_cfg["beta_max_deg"])
    baseline_xi = np.asarray(controller_cfg["hand_tuned_reference_xi"], float)
    eval_kwargs = dict(
        u_min=controller_cfg["u_min"], u_max=controller_cfg["u_max"],
        u_rate_max=controller_cfg.get("u_rate_max"), beta_max=beta_max,
        noise_std=controller_cfg["noise_std"], process_noise=controller_cfg["process_noise"],
        plant_mode=controller_cfg["plant_mode"], mu=controller_cfg["tire_mu"],
    )

    out_dir = _ROOT / "results" / RUN_TAG
    processed_dir, cache_dir = out_dir / "processed", out_dir / "cache"
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    fleet = generate_fixed_speed_fleet(
        Ts=fleet_cfg["Ts"], n_per_family=fleet_cfg["clients_per_regime"], variability=fleet_cfg["variability"],
        seed=FLEET_SEED, delta=fleet_cfg["delta"], lambda_f=fleet_cfg["lambda_f"],
        lane_change_time=fleet_cfg["lane_change_time"],
    )
    print(f"[tlc_audit] {len(fleet)} physical clients (fleet seed={FLEET_SEED}), "
          f"comparing T_lc={T1} vs T_lc={T2} for each, holding theta_i fixed")

    sampler = qmc.Sobol(d=3, scramble=True, seed=SOBOL_SEED)
    Xi = qmc.scale(sampler.random(N_SOBOL), xi_lower, xi_upper)

    landscape_rows: list[dict] = []
    summary_rows: list[dict] = []
    for client in fleet:
        family = client.cluster.name.lower()
        per_setting: dict[str, dict] = {}
        for label, T_lc_val in (("T1", T1), ("T2", T2)):
            client.T_lc = T_lc_val
            bank = build_episode_bank(
                client, Ts=fleet_cfg["Ts"], T_total=controller_cfg["T_total"], T0=controller_cfg["T0"],
                r_max=controller_cfg["r_max"], n_calibration=N_CALIBRATION_EPISODES, n_test=0,
                base_seed=EPISODE_BASE_SEED, tfilter=controller_cfg.get("tfilter"),
            )
            cached_evaluate = make_cached_evaluate(str(cache_dir / f"{client.client_id}__{label}"))
            episodes = calibration_episodes(bank)

            base_ev = cached_evaluate(
                client, baseline_xi, bank, state_scale=state_scale, fixed_R=fixed_R, episodes=episodes,
                **eval_kwargs,
            )
            baseline_feasible = bool(base_ev.feasible and base_ev.tracking_rmse is not None and base_ev.tracking_rmse > 0)
            baseline_rmse = float(base_ev.tracking_rmse) if baseline_feasible else float("nan")

            n_feasible = 0
            best = float("inf")
            landscape: dict[int, float] = {}
            for xi_idx, xi in enumerate(Xi):
                ev = cached_evaluate(
                    client, xi, bank, state_scale=state_scale, fixed_R=fixed_R, episodes=episodes, **eval_kwargs,
                )
                feasible = bool(ev.feasible and ev.tracking_rmse is not None and ev.tracking_rmse > 0)
                rmse = float(ev.tracking_rmse) if feasible else float("nan")
                landscape[xi_idx] = rmse
                landscape_rows.append(
                    {
                        "client_id": client.client_id, "family": family, "T_lc_label": label, "T_lc": T_lc_val,
                        "xi_idx": xi_idx, "xi_1": float(xi[0]), "xi_2": float(xi[1]), "xi_3": float(xi[2]),
                        "feasible": feasible, "rmse": rmse,
                    }
                )
                if feasible:
                    n_feasible += 1
                    best = min(best, rmse)
            phi = n_feasible / N_SOBOL
            oracle_rmse = best if np.isfinite(best) else float("nan")
            per_setting[label] = dict(
                baseline_feasible=baseline_feasible, baseline_rmse=baseline_rmse, phi=phi,
                oracle_rmse=oracle_rmse, landscape=landscape,
            )
            print(
                f"[tlc_audit]   {client.client_id} T_lc={T_lc_val} ({label}): "
                f"baseline_feasible={baseline_feasible} baseline_rmse={baseline_rmse:.4f} "
                f"phi={phi:.3f} oracle_rmse={oracle_rmse:.4f}"
            )

        common_idx = [
            idx for idx in range(N_SOBOL)
            if np.isfinite(per_setting["T1"]["landscape"][idx]) and np.isfinite(per_setting["T2"]["landscape"][idx])
        ]
        if len(common_idx) >= 4:
            v1 = [per_setting["T1"]["landscape"][idx] for idx in common_idx]
            v2 = [per_setting["T2"]["landscape"][idx] for idx in common_idx]
            rho, p = spearmanr(v1, v2)
        else:
            rho, p = float("nan"), float("nan")

        summary_rows.append(
            {
                "client_id": client.client_id, "family": family,
                "baseline_feasible_T1": per_setting["T1"]["baseline_feasible"],
                "baseline_feasible_T2": per_setting["T2"]["baseline_feasible"],
                "baseline_rmse_T1": per_setting["T1"]["baseline_rmse"],
                "baseline_rmse_T2": per_setting["T2"]["baseline_rmse"],
                "phi_T1": per_setting["T1"]["phi"], "phi_T2": per_setting["T2"]["phi"],
                "oracle_rmse_T1": per_setting["T1"]["oracle_rmse"], "oracle_rmse_T2": per_setting["T2"]["oracle_rmse"],
                "n_common_feasible": len(common_idx), "landscape_spearman_rho": rho, "landscape_spearman_p": p,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    landscape_df = pd.DataFrame(landscape_rows)
    summary_df.to_csv(processed_dir / "tlc_client_summary.csv", index=False)
    landscape_df.to_csv(processed_dir / "tlc_landscape_points.csv", index=False)
    print(f"\n[tlc_audit] wrote {processed_dir / 'tlc_client_summary.csv'} ({len(summary_df)} clients)")

    # ---- console report -----------------------------------------------------------
    summary_df["baseline_flip"] = summary_df["baseline_feasible_T1"] != summary_df["baseline_feasible_T2"]
    summary_df["phi_delta"] = summary_df["phi_T2"] - summary_df["phi_T1"]
    both_baseline_feasible = summary_df["baseline_feasible_T1"] & summary_df["baseline_feasible_T2"]
    summary_df["baseline_rmse_ratio"] = np.where(
        both_baseline_feasible, summary_df["baseline_rmse_T2"] / summary_df["baseline_rmse_T1"], np.nan
    )
    both_oracle_feasible = summary_df["oracle_rmse_T1"].notna() & summary_df["oracle_rmse_T2"].notna()
    summary_df["oracle_rmse_ratio"] = np.where(
        both_oracle_feasible, summary_df["oracle_rmse_T2"] / summary_df["oracle_rmse_T1"], np.nan
    )

    print("\n=== T_lc effect audit ===")
    print(f"n_clients={len(summary_df)}  T_lc: T1={T1}, T2={T2}\n")
    print("-- baseline feasibility flips (same theta_i, T1 vs T2) --")
    print(summary_df.groupby("family")["baseline_flip"].agg(["sum", "count"]))
    print(f"\ntotal flips: {int(summary_df['baseline_flip'].sum())}/{len(summary_df)}")

    print("\n-- phi (feasible-support fraction) by family --")
    print(summary_df.groupby("family")[["phi_T1", "phi_T2", "phi_delta"]].mean().round(3))

    print("\n-- baseline_rmse ratio (T2/T1), only where BOTH feasible --")
    print(summary_df.groupby("family")["baseline_rmse_ratio"].describe()[["count", "mean", "50%", "min", "max"]].round(4))

    print("\n-- oracle_rmse ratio (T2/T1), only where BOTH have >=1 feasible point --")
    print(summary_df.groupby("family")["oracle_rmse_ratio"].describe()[["count", "mean", "50%", "min", "max"]].round(4))

    print("\n-- landscape rank correlation (Spearman rho, T1 vs T2 on common feasible xi) --")
    print(summary_df.groupby("family")["landscape_spearman_rho"].describe()[["count", "mean", "50%", "min", "max"]].round(4))

    write_run_manifest(
        str(out_dir),
        {
            "fleet_seed": FLEET_SEED, "n_sobol": N_SOBOL, "sobol_seed": SOBOL_SEED,
            "lane_change_time": list(fleet_cfg["lane_change_time"]),
            "n_calibration_episodes": N_CALIBRATION_EPISODES, "episode_base_seed": EPISODE_BASE_SEED,
            "controller_config": controller_cfg,
        },
    )
    print(f"\n[tlc_audit] wrote manifest to {out_dir / 'run_manifest.yaml'}")


if __name__ == "__main__":
    main()
