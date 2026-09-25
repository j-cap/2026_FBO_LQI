"""
One-off backfill: adds the actual dynamics distance `d_ij` (not just similarity `s_ij`)
to `results/calibration_landscape_similarity/processed/landscape_similarity_pairs.csv`
(Part 19) - needed for Fig. 2b's x-axis (docs/phase2_diagnostic_findings.md, paper
figure planning, 2026-09-08). Cheap: identification only (`_prepare_seed_context`,
no BO), same 5 seeds already used by `calibration_landscape_similarity.py`.

Usage:
    python scripts/backfill_landscape_distances.py --config config/runs/phase2_warmstart_dev_c1cds02.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.dynamics_distance import pairwise_uncertainty_aware_distance  # noqa: E402
from src.ifac_bridge import pack_theta_AB  # noqa: E402
from src.io_utils import deep_update, load_run_config, load_yaml  # noqa: E402

from calibration_landscape_similarity import N_STUDY_SEEDS  # noqa: E402
from phase2_fbo_comparison import _prepare_seed_context, _resolve  # noqa: E402


def main(config_path: str) -> None:
    cfg = load_run_config(config_path)
    fleet_cfg = deep_update(load_yaml(_resolve(cfg["fleet_config"])), cfg.get("fleet_overrides", {}))
    controller_cfg = deep_update(load_yaml(_resolve(cfg["controller_config"])), cfg.get("controller_overrides", {}))
    beta_max = np.deg2rad(controller_cfg["beta_max_deg"])
    xi_lower, xi_upper = np.asarray(controller_cfg["xi_lower"], float), np.asarray(controller_cfg["xi_upper"], float)
    eval_kwargs = dict(
        u_min=controller_cfg["u_min"], u_max=controller_cfg["u_max"], u_rate_max=controller_cfg.get("u_rate_max"),
        beta_max=beta_max, noise_std=controller_cfg["noise_std"], process_noise=controller_cfg["process_noise"],
        plant_mode=controller_cfg.get("plant_mode", "linear"), mu=controller_cfg.get("tire_mu", 1.0),
    )
    state_scale, fixed_R = controller_cfg["state_scale"], controller_cfg["fixed_R"]

    out_dir = _ROOT / "results" / "calibration_landscape_similarity"
    cache_dir = out_dir / "cache"
    pairs_path = out_dir / "processed" / "landscape_similarity_pairs.csv"
    pairs_df = pd.read_csv(pairs_path)

    study_seeds = cfg["seeds"][:N_STUDY_SEEDS]
    d_rows = []
    for seed in study_seeds:
        print(f"[backfill] seed={seed}: preparing fleet context (identification, no BO) ...")
        ctx = _prepare_seed_context(
            seed, cfg, fleet_cfg, controller_cfg, cache_dir=cache_dir, xi_lower=xi_lower, xi_upper=xi_upper,
            state_scale=state_scale, fixed_R=fixed_R, eval_kwargs=eval_kwargs,
        )
        fleet = ctx["fleet"]
        client_ids = [c.client_id for c in fleet]
        thetas = [pack_theta_AB(c.Ad_hat, c.Bd_hat) for c in fleet]
        precisions = [c.W_raw for c in fleet]
        D2 = pairwise_uncertainty_aware_distance(thetas, precisions)
        for i in range(len(fleet)):
            for j in range(i + 1, len(fleet)):
                d_rows.append({"seed": seed, "client_i": client_ids[i], "client_j": client_ids[j], "d_ij": float(D2[i, j])})
        print(f"[backfill] seed={seed}: done")

    d_df = pd.DataFrame(d_rows)
    merged = pairs_df.merge(d_df, on=["seed", "client_i", "client_j"], how="left")
    assert merged["d_ij"].notna().all(), "some pairs failed to merge - client_i/client_j ordering mismatch?"
    merged.to_csv(pairs_path, index=False)
    print(f"\n[backfill] updated {pairs_path} with d_ij ({len(merged)} pairs)")
    print(merged[["s_ij", "d_ij"]].describe().round(4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/runs/phase2_warmstart_dev_c1cds02.yaml")
    args = parser.parse_args()
    main(args.config)
