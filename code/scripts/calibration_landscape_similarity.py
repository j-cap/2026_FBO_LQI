"""
Calibration-landscape universality study (Part 19, docs/phase2_diagnostic_findings.md,
2026-09-08) - the user's Step 2 after Part 18's decisive negative result for
`s_ij`-weighting: WHY does fleet calibration knowledge transfer so well even across
families with substantially different identified dynamics? Tests the explanation Part
18 pointed at - "plant heterogeneity does not necessarily imply calibration-objective
heterogeneity" - directly, by comparing dynamics similarity (`s_ij`, the actual
quantity the FBO methods use) against calibration-LANDSCAPE similarity computed
independently of it.

Method: a FIXED Sobol grid of M ξ-points (shared across every client and every seed -
sampled once, not per-seed, so pairwise comparisons pool cleanly across seeds), every
client evaluated at every grid point (no BO, no cross-client pooling - each client's
own true-plant rollout only). For every client pair (i, j) within a seed's 30-client
fleet:
  - rho_J_ij = Spearman(J_i(xi_1:M), J_j(xi_1:M)) over the jointly-feasible grid points -
    do client i and client j agree on WHICH controller gains are good/bad, independent
    of dynamics distance? Restricted to jointly-feasible points BY DESIGN (`joint =
    feasible_mask[i] & feasible_mask[j]` below) specifically so a shared infeasibility
    penalty can never inflate this correlation - infeasible points never enter it at all.
  - S_ij_feas = Jaccard overlap between each client's own FEASIBLE-point set
    (`feasible_mask[i]` vs `feasible_mask[j]`) - a separate question from rho_J: do
    client i and j even agree on WHICH xi are admissible, independent of how they rank
    the admissible ones?
  - overlap_10pct_ij = Jaccard overlap between each client's own top-10%-by-performance
    grid points (among ITS OWN feasible points) - do their "good regions" of xi-space
    coincide?
Both rho_J and overlap_10pct compared directly against `s_ij` (already computed via the
same `pairwise_uncertainty_aware_distance`/similarity_matrix machinery every other
Phase 2 script uses) - the key test: does `s_ij` predict `rho_J_ij`/`overlap_10pct_ij`
at all, or does calibration-landscape similarity stay uniformly high regardless of
dynamics distance? The pooled-pairs Spearman(s_ij, rho_J) printed at the bottom treats
~30*29/2 pairs per seed as independent, which they are not (each client appears in many
pairs) - `_seed_level_summary` below additionally reports this at the fleet-REALIZATION
level (per-seed median rho_J, bootstrapped over seeds), the correct unit of replication
for a confidence claim (doc §19 / Part 4's frozen convention).

Cheap by design - no BO, only `M` x 30 x (a handful of) seeds independent evaluations
per client, no cross-client simulation.

Usage:
    python scripts/calibration_landscape_similarity.py --config config/runs/phase2_warmstart_dev_c1cds02.yaml
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import qmc, spearmanr  # noqa: E402

from src.caching import make_cached_evaluate  # noqa: E402
from src.episodes import calibration_episodes  # noqa: E402
from src.io_utils import deep_update, load_run_config, load_yaml  # noqa: E402

from phase2_fbo_comparison import _prepare_seed_context, _resolve  # noqa: E402
from phase2_medium_analysis import _family_of  # noqa: E402
from phase2_triggered_dev import _seed_bootstrap_median_ci  # noqa: E402

N_GRID_POINTS = 128  # bumped from Part 19's 64 - denser grid firms up the jointly-feasible
# sample for lower-phi (payload/tire_degraded) pairs without materially changing cost
N_STUDY_SEEDS = 5  # first 5 of the 20 development seeds, not cherry-picked
GRID_SEED = 0  # fixed, independent of any BO/fleet seed - the SAME grid for every seed
TOP_FRACTION = 0.10
MIN_JOINT_FEASIBLE = 10  # below this, rho_J/overlap are too noisy to report - NaN instead


def main(config_path: str) -> None:
    cfg = load_run_config(config_path)
    fleet_cfg = deep_update(load_yaml(_resolve(cfg["fleet_config"])), cfg.get("fleet_overrides", {}))
    controller_cfg = deep_update(load_yaml(_resolve(cfg["controller_config"])), cfg.get("controller_overrides", {}))
    beta_max = np.deg2rad(controller_cfg["beta_max_deg"])
    xi_lower = np.asarray(controller_cfg["xi_lower"], float)
    xi_upper = np.asarray(controller_cfg["xi_upper"], float)
    eval_kwargs = dict(
        u_min=controller_cfg["u_min"], u_max=controller_cfg["u_max"], u_rate_max=controller_cfg.get("u_rate_max"),
        beta_max=beta_max, noise_std=controller_cfg["noise_std"], process_noise=controller_cfg["process_noise"],
        plant_mode=controller_cfg.get("plant_mode", "linear"), mu=controller_cfg.get("tire_mu", 1.0),
    )
    state_scale, fixed_R = controller_cfg["state_scale"], controller_cfg["fixed_R"]

    out_dir = _ROOT / "results" / "calibration_landscape_similarity"
    (out_dir / "processed").mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    sampler = qmc.Sobol(d=3, scramble=True, seed=GRID_SEED)
    Xi_grid = qmc.scale(sampler.random(N_GRID_POINTS), xi_lower, xi_upper)
    top_n = max(1, math.ceil(TOP_FRACTION * N_GRID_POINTS))
    print(f"[landscape] fixed shared grid: {N_GRID_POINTS} points, top-{int(TOP_FRACTION*100)}% = {top_n} points")

    study_seeds = cfg["seeds"][:N_STUDY_SEEDS]
    pair_rows = []

    for seed in study_seeds:
        print(f"[landscape] seed={seed}: preparing fleet context (identification, no BO) ...")
        ctx = _prepare_seed_context(
            seed, cfg, fleet_cfg, controller_cfg, cache_dir=cache_dir, xi_lower=xi_lower, xi_upper=xi_upper,
            state_scale=state_scale, fixed_R=fixed_R, eval_kwargs=eval_kwargs,
        )
        fleet, banks, S = ctx["fleet"], ctx["banks"], ctx["similarity_S"]
        client_ids = [c.client_id for c in fleet]
        client_idx = {cid: i for i, cid in enumerate(client_ids)}
        cached_evaluate = make_cached_evaluate(str(ctx["seed_cache_dir"]))

        print(f"[landscape] seed={seed}: evaluating {len(fleet)} clients x {N_GRID_POINTS} shared grid points ...")
        J = np.full((len(fleet), N_GRID_POINTS), np.nan)
        feasible_mask = np.zeros((len(fleet), N_GRID_POINTS), dtype=bool)
        for ci, client in enumerate(fleet):
            bank = banks[client.client_id]
            for gi, xi in enumerate(Xi_grid):
                ev = cached_evaluate(
                    client, xi, bank, state_scale=state_scale, fixed_R=fixed_R,
                    episodes=calibration_episodes(bank), **eval_kwargs,
                )
                if ev.feasible and ev.tracking_rmse is not None and ev.tracking_rmse > 0:
                    J[ci, gi] = float(ev.tracking_rmse)
                    feasible_mask[ci, gi] = True

        top_sets = []
        for ci in range(len(fleet)):
            feas_idx = np.where(feasible_mask[ci])[0]
            if len(feas_idx) == 0:
                top_sets.append(set())
                continue
            order = feas_idx[np.argsort(J[ci, feas_idx])]
            top_sets.append(set(order[:top_n].tolist()))

        for i in range(len(fleet)):
            for j in range(i + 1, len(fleet)):
                joint = feasible_mask[i] & feasible_mask[j]
                n_joint = int(joint.sum())
                rho_J = float("nan")
                if n_joint >= MIN_JOINT_FEASIBLE:
                    rho_J, _ = spearmanr(J[i, joint], J[j, joint])
                union, inter = top_sets[i] | top_sets[j], top_sets[i] & top_sets[j]
                overlap = (len(inter) / len(union)) if union else float("nan")
                # Feasibility-set Jaccard - a SEPARATE question from rho_J/overlap_10pct
                # above (which both already condition on/restrict to feasible points):
                # do i and j even agree on WHICH xi are admissible at all?
                feas_union = int((feasible_mask[i] | feasible_mask[j]).sum())
                feas_inter = int((feasible_mask[i] & feasible_mask[j]).sum())
                s_ij_feas = (feas_inter / feas_union) if feas_union else float("nan")
                cid_i, cid_j = client_ids[i], client_ids[j]
                fam_i, fam_j = _family_of(cid_i), _family_of(cid_j)
                pair_rows.append(
                    {
                        "seed": seed, "client_i": cid_i, "client_j": cid_j, "family_i": fam_i, "family_j": fam_j,
                        "same_family": fam_i == fam_j, "s_ij": float(S[i, j]), "rho_J": rho_J,
                        "overlap_10pct": overlap, "n_joint_feasible": n_joint, "s_ij_feas": s_ij_feas,
                    }
                )
        print(f"[landscape] seed={seed}: done ({len(fleet)*(len(fleet)-1)//2} pairs)")

    pairs_df = pd.DataFrame(pair_rows)
    pairs_df.to_csv(out_dir / "processed" / "landscape_similarity_pairs.csv", index=False)
    print(f"\n[landscape] wrote {out_dir / 'processed' / 'landscape_similarity_pairs.csv'} ({len(pairs_df)} pairs)")

    print("\n" + "=" * 90)
    print("rho_J (calibration-landscape rank correlation) and top-10% overlap, same-family vs cross-family")
    print("=" * 90)
    print(pairs_df.groupby("same_family")[["s_ij", "rho_J", "overlap_10pct"]].describe().round(4))

    print("\n" + "=" * 90)
    print("KEY COMPARISON: does dynamics similarity (s_ij) predict calibration-landscape similarity?")
    print("(pooled across all pairs/seeds - descriptive; see the realization-level summary below for a")
    print(" comparison that does not treat non-independent pairs as independent samples)")
    print("=" * 90)
    valid_rho = pairs_df.dropna(subset=["rho_J"])
    rho_sij_rhoJ, p1 = spearmanr(valid_rho["s_ij"], valid_rho["rho_J"])
    print(f"Spearman(s_ij, rho_J) = {rho_sij_rhoJ:.4f}, p={p1:.4g}, n={len(valid_rho)}")
    valid_ov = pairs_df.dropna(subset=["overlap_10pct"])
    rho_sij_ov, p2 = spearmanr(valid_ov["s_ij"], valid_ov["overlap_10pct"])
    print(f"Spearman(s_ij, overlap_10pct) = {rho_sij_ov:.4f}, p={p2:.4g}, n={len(valid_ov)}")
    valid_feas = pairs_df.dropna(subset=["s_ij_feas"])
    rho_sij_feas, p3 = spearmanr(valid_feas["s_ij"], valid_feas["s_ij_feas"])
    print(f"Spearman(s_ij, s_ij_feas) = {rho_sij_feas:.4f}, p={p3:.4g}, n={len(valid_feas)} "
          f"(does dynamics similarity predict FEASIBILITY-set agreement, separately from rank agreement?)")

    print("\n(if these are weak/near-zero while rho_J/overlap are uniformly high across the s_ij range, "
          "that supports 'dynamics heterogeneity does not imply calibration-landscape heterogeneity')")

    # ---- Realization-level summary (Part 4's frozen convention: the SEED, not the ----
    # pair, is the unit of replication for any confidence claim) --------------------
    print("\n" + "=" * 90)
    print("REALIZATION-LEVEL summary (per-seed median, bootstrapped OVER SEEDS - n=%d seeds)" % len(study_seeds))
    print("=" * 90)
    per_seed_median_rhoJ = valid_rho.groupby("seed")["rho_J"].median().to_dict()
    res_rhoJ = _seed_bootstrap_median_ci(per_seed_median_rhoJ)
    print(f"median(rho_J) per seed -> across-seed median={res_rhoJ['median']:.4f}, "
          f"95% CI={res_rhoJ['bootstrap_ci_95']}, n_seeds={res_rhoJ['n_seeds']}")

    per_seed_median_feas = valid_feas.groupby("seed")["s_ij_feas"].median().to_dict()
    res_feas = _seed_bootstrap_median_ci(per_seed_median_feas)
    print(f"median(s_ij_feas) per seed -> across-seed median={res_feas['median']:.4f}, "
          f"95% CI={res_feas['bootstrap_ci_95']}, n_seeds={res_feas['n_seeds']}")

    per_seed_corr = {}
    for seed, g in valid_rho.groupby("seed"):
        if len(g) >= MIN_JOINT_FEASIBLE:
            rho, _ = spearmanr(g["s_ij"], g["rho_J"])
            per_seed_corr[seed] = rho
    if per_seed_corr:
        res_corr = _seed_bootstrap_median_ci(per_seed_corr)
        print(f"Spearman(s_ij, rho_J) computed WITHIN each seed, then bootstrapped across seeds -> "
              f"median={res_corr['median']:.4f}, 95% CI={res_corr['bootstrap_ci_95']}, "
              f"n_seeds={res_corr['n_seeds']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/runs/phase2_warmstart_dev_c1cds02.yaml")
    args = parser.parse_args()
    main(args.config)
