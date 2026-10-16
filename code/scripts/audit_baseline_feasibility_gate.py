"""Audit the baseline-feasibility admission gate on the frozen confirmatory fleets.

This is a deterministic provenance audit, not a new BO experiment. It replays only the
fleet generator's admission check for the exact seeds/configuration of the confirmatory
campaign. No identification, BO, dense reference, or landscape computation is run.

Scientific question:
    Does conditioning each generated system on feasibility of the fixed baseline
    controller materially narrow the fleet distribution?

Primary diagnostics:
- total candidate draws / accepted systems;
- rejection fraction;
- fraction of system slots requiring any resampling;
- distribution of accepted attempt indices;
- family-specific rejection/resampling rates.

Usage (from code/):
    python scripts/audit_baseline_feasibility_gate.py

Optional:
    python scripts/audit_baseline_feasibility_gate.py \
        --config config/runs/phase2_warmstart_confirmatory_c1cds02.yaml
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

from src.fleet_families import (  # noqa: E402
    GENERATION_VERSION_V3,
    generate_fixed_speed_fleet_v3_common_task,
)
from src.io_utils import deep_update, load_run_config, load_yaml  # noqa: E402

from phase2_fbo_comparison import _resolve  # noqa: E402


DEFAULT_CONFIG = "config/runs/phase2_warmstart_confirmatory_c1cds02.yaml"
DEFAULT_OUTPUT_TAG = "baseline_feasibility_gate_audit"


def _eval_kwargs(controller_cfg: dict) -> dict:
    return dict(
        u_min=controller_cfg["u_min"],
        u_max=controller_cfg["u_max"],
        u_rate_max=controller_cfg.get("u_rate_max"),
        beta_max=np.deg2rad(controller_cfg["beta_max_deg"]),
        noise_std=controller_cfg["noise_std"],
        process_noise=controller_cfg["process_noise"],
        plant_mode=controller_cfg.get("plant_mode", "linear"),
        mu=controller_cfg.get("tire_mu", 1.0),
    )


def _replay_seed(seed: int, cfg: dict, fleet_cfg: dict, controller_cfg: dict) -> pd.DataFrame:
    fleet_cfg_seed = deep_update(fleet_cfg, {"seed": fleet_cfg["seed"] + seed})
    if fleet_cfg_seed.get("generation_version") != GENERATION_VERSION_V3:
        raise ValueError(
            "This audit is frozen to the v3 common-task fleet generator; "
            f"got generation_version={fleet_cfg_seed.get('generation_version')!r}"
        )

    clients, provenance = generate_fixed_speed_fleet_v3_common_task(
        Ts=fleet_cfg_seed["Ts"],
        n_per_family=fleet_cfg_seed["clients_per_regime"],
        variability=fleet_cfg_seed["variability"],
        seed=fleet_cfg_seed["seed"],
        delta=fleet_cfg_seed["delta"],
        lambda_f=fleet_cfg_seed["lambda_f"],
        lane_change_time=fleet_cfg_seed["lane_change_time"],
        baseline_xi=controller_cfg["hand_tuned_reference_xi"],
        T_total=controller_cfg["T_total"],
        T0=controller_cfg["T0"],
        r_max=controller_cfg["r_max"],
        n_calibration_episodes_per_maneuver=cfg["n_calibration_episodes"],
        n_test_episodes_per_maneuver=cfg["n_test_episodes"],
        episode_base_seed=cfg.get("episode_base_seed", 5000),
        tfilter=controller_cfg.get("tfilter"),
        state_scale=controller_cfg["state_scale"],
        fixed_R=controller_cfg["fixed_R"],
        eval_kwargs=_eval_kwargs(controller_cfg),
        max_attempts=fleet_cfg_seed.get("max_generation_attempts", 50),
    )
    if len(clients) != 30:
        raise RuntimeError(f"Expected 30 accepted systems for seed={seed}, got {len(clients)}")
    provenance.insert(0, "seed", int(seed))
    provenance.insert(1, "fleet_generator_seed", int(fleet_cfg_seed["seed"]))
    return provenance


def _summary(prov: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    accepted = prov[prov["accepted"]].copy()
    if accepted.empty:
        raise RuntimeError("No accepted systems in provenance audit")

    slot_cols = ["seed", "client_id", "family", "slot"]
    # Exactly one accepted attempt must terminate every slot.
    accepted_counts = accepted.groupby(slot_cols).size()
    if not (accepted_counts == 1).all():
        raise RuntimeError("Expected exactly one accepted attempt per system slot")

    slot = (
        accepted[slot_cols + ["attempt", "baseline_rmse"]]
        .rename(columns={"attempt": "accepted_attempt"})
        .sort_values(["seed", "family", "slot"])
        .reset_index(drop=True)
    )
    slot["n_rejected_before_accept"] = slot["accepted_attempt"]
    slot["required_resampling"] = slot["accepted_attempt"] > 0

    rows = []
    for scope, g in [("all", prov)] + [
        (family, fg) for family, fg in prov.groupby("family", sort=True)
    ]:
        g_acc = g[g["accepted"]]
        slot_scope = slot if scope == "all" else slot[slot["family"] == scope]
        n_draws = len(g)
        n_accept = len(g_acc)
        n_reject = n_draws - n_accept
        rows.append(
            {
                "scope": scope,
                "n_system_slots": int(len(slot_scope)),
                "n_candidate_draws": int(n_draws),
                "n_accepted": int(n_accept),
                "n_rejected": int(n_reject),
                "rejection_fraction": float(n_reject / n_draws),
                "fraction_slots_resampled": float(slot_scope["required_resampling"].mean()),
                "mean_attempts_per_slot": float((slot_scope["accepted_attempt"] + 1).mean()),
                "median_accepted_attempt": float(slot_scope["accepted_attempt"].median()),
                "max_accepted_attempt": int(slot_scope["accepted_attempt"].max()),
                "mean_baseline_rmse_accepted": float(slot_scope["baseline_rmse"].mean()),
            }
        )
    summary = pd.DataFrame(rows)

    attempt_dist = (
        slot.groupby(["family", "accepted_attempt"])
        .size()
        .rename("n_slots")
        .reset_index()
    )
    attempt_dist["fraction_within_family"] = attempt_dist.groupby("family")["n_slots"].transform(
        lambda s: s / s.sum()
    )

    return summary, slot, attempt_dist


def main(config_path: str, output_tag: str) -> None:
    cfg = load_run_config(config_path)
    fleet_cfg = deep_update(
        load_yaml(_resolve(cfg["fleet_config"])),
        cfg.get("fleet_overrides", {}),
    )
    controller_cfg = deep_update(
        load_yaml(_resolve(cfg["controller_config"])),
        cfg.get("controller_overrides", {}),
    )

    frames = []
    for k, seed in enumerate(cfg["seeds"], start=1):
        print(f"[gate-audit] {k}/{len(cfg['seeds'])} seed={seed}")
        frames.append(_replay_seed(int(seed), cfg, fleet_cfg, controller_cfg))

    prov = pd.concat(frames, ignore_index=True)
    summary, slot, attempt_dist = _summary(prov)

    out = _ROOT / "results" / output_tag / "processed"
    out.mkdir(parents=True, exist_ok=True)
    prov.to_csv(out / "fleet_generation_provenance.csv", index=False)
    slot.to_csv(out / "slot_resampling_summary.csv", index=False)
    summary.to_csv(out / "gate_summary.csv", index=False)
    attempt_dist.to_csv(out / "accepted_attempt_distribution.csv", index=False)

    print("\nBaseline-feasibility admission-gate audit")
    print("=" * 100)
    print(summary.round(5).to_string(index=False))

    print("\nAccepted-attempt distribution")
    print("=" * 100)
    print(attempt_dist.round(5).to_string(index=False))

    all_row = summary[summary["scope"] == "all"].iloc[0]
    print("\nInterpretation-ready totals")
    print("=" * 100)
    print(f"accepted system slots: {int(all_row['n_system_slots'])}")
    print(f"candidate draws: {int(all_row['n_candidate_draws'])}")
    print(f"rejected draws: {int(all_row['n_rejected'])}")
    print(f"rejection fraction: {all_row['rejection_fraction']:.4%}")
    print(f"slots requiring resampling: {all_row['fraction_slots_resampled']:.4%}")
    print(f"maximum accepted attempt index: {int(all_row['max_accepted_attempt'])}")

    print("\nWrote:")
    for name in [
        "fleet_generation_provenance.csv",
        "slot_resampling_summary.csv",
        "gate_summary.csv",
        "accepted_attempt_distribution.csv",
    ]:
        print(f"  {out / name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-tag", default=DEFAULT_OUTPUT_TAG)
    args = parser.parse_args()
    main(args.config, args.output_tag)
