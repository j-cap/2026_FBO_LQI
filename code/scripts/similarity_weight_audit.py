"""
Effective-weight audit (Part 16, docs/phase2_diagnostic_findings.md, 2026-09-07) - the
user's Step 1 after Part 15 found `global` and `similarity` warm start statistically
indistinguishable (57/60 target problems exactly tied on N_5pct): is dynamics-aware
weighting genuinely unnecessary for this fleet, or does the current weighting mechanism
just not differ enough from unweighted pooling to matter?

Pure analysis, no BO campaign - reuses the ALREADY-COMPLETED `phase2_warmstart_dev_c1cds02`
run's `mature_fleet_records.parquet` (peer histories, already on local disk) and only
requires cheap, LOCAL, no-remote-machine-needed work: per-seed fleet
identification/similarity (`_prepare_seed_context`, no BO - the same cheap "stage 1"
cost as Part 9's provenance check) plus 2 fresh evaluations per target (its own n_init=2
Sobol points, deterministically reproduced from the SAME seed the original run used) -
120 evaluations total across all 60 targets.

For every target, this:
  1. Records the s_ij distribution over its 29 peers, split same-family vs cross-family,
     and the induced GP noise inflation `alpha_ij = BASE_ALPHA / clip(s_ij, WEIGHT_FLOOR, 1)`
     - the ACTUAL mechanism `src.optimization.gp_surrogate.fit_weighted_gp` uses, not a
     guess - including how many peers fall below WEIGHT_FLOOR and get dropped entirely
     under `similarity` (never just "very noisy").
  2. Rebuilds the round-1 pool (29 peers' full mature histories + the target's own 2
     Sobol points) exactly as `run_target_warmstart` would have, fits the weighted GP
     for BOTH `global` and `similarity` on the SAME candidate grid the original run
     used, and compares mu/sigma/EI directly - plus whether the two methods would
     propose the SAME first BO point.
  3. Reports fleet-redundancy context (nearest-peer similarity, k-th-largest similarity)
     - cheap, from the same similarity matrices, and directly relevant to interpreting
     (1)/(2): a highly redundant fleet (many near-duplicate peers) would explain why
     dropping/discounting a minority of dissimilar peers barely changes anything.

Usage:
    python scripts/similarity_weight_audit.py --config config/runs/phase2_warmstart_dev_c1cds02.yaml
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
from scipy.stats import qmc, spearmanr  # noqa: E402

from src.caching import make_cached_evaluate  # noqa: E402
from src.episodes import calibration_episodes  # noqa: E402
from src.io_utils import deep_update, load_run_config, load_yaml  # noqa: E402
from src.optimization.fleet_bo import INFEASIBLE_PENALTY_NORM, _denormalize, _normalize  # noqa: E402
from src.optimization.gp_surrogate import BASE_ALPHA, WEIGHT_FLOOR, expected_improvement, fit_weighted_gp, propose_next  # noqa: E402

from phase2_fbo_comparison import _prepare_seed_context, _resolve  # noqa: E402
from phase2_medium_analysis import _family_of  # noqa: E402
from phase2_warmstart_dev import FAMILIES, TARGET_N_INIT, TARGET_SLOT  # noqa: E402


def main(config_path: str, mature_records_path: str) -> None:
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

    mature = pd.read_parquet(mature_records_path)
    out_dir = _ROOT / "results" / "similarity_weight_audit"
    (out_dir / "processed").mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    s_ij_rows, gp_rows, redundancy_rows = [], [], []

    for seed in cfg["seeds"]:
        print(f"[audit] seed={seed}: preparing fleet context (identification + similarity, no BO) ...")
        ctx = _prepare_seed_context(
            seed, cfg, fleet_cfg, controller_cfg, cache_dir=cache_dir, xi_lower=xi_lower, xi_upper=xi_upper,
            state_scale=state_scale, fixed_R=fixed_R, eval_kwargs=eval_kwargs,
        )
        fleet, banks, S = ctx["fleet"], ctx["banks"], ctx["similarity_S"]
        client_ids = [c.client_id for c in fleet]
        client_idx = {cid: i for i, cid in enumerate(client_ids)}
        cached_evaluate = make_cached_evaluate(str(ctx["seed_cache_dir"]))
        seed_mature = mature[mature["seed"] == seed]

        for family in FAMILIES:
            target_id = f"{family}_{TARGET_SLOT}"
            ti = client_idx[target_id]
            target = fleet[ti]
            target_bank = banks[target_id]
            peer_ids = [cid for cid in client_ids if cid != target_id]

            # ---- (1) s_ij distribution + induced GP noise inflation --------------------
            for peer_id in peer_ids:
                pj = client_idx[peer_id]
                s = float(S[ti, pj])
                same_family = _family_of(peer_id) == family
                alpha = BASE_ALPHA / np.clip(s, WEIGHT_FLOOR, 1.0)
                s_ij_rows.append(
                    {
                        "seed": seed, "target_id": target_id, "peer_id": peer_id, "same_family": same_family,
                        "s_ij": s, "alpha_ij": alpha, "alpha_inflation_x": alpha / BASE_ALPHA,
                        "dropped_under_similarity": s < WEIGHT_FLOOR,
                    }
                )

            # ---- (3) fleet-redundancy context ------------------------------------------
            s_vals = np.array([float(S[ti, client_idx[p]]) for p in peer_ids])
            s_sorted = np.sort(s_vals)[::-1]

            def _kth(k: int) -> float:
                return float(s_sorted[k - 1]) if len(s_sorted) >= k else float("nan")

            redundancy_rows.append(
                {
                    "seed": seed, "target_id": target_id, "family": family, "s_max": _kth(1),
                    "s_3rd": _kth(3), "s_5th": _kth(5), "s_10th": _kth(10),
                    "n_below_weight_floor": int((s_vals < WEIGHT_FLOOR).sum()),
                }
            )

            # ---- (2) round-1 GP comparison: global vs similarity -----------------------
            init_sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
            Xi_init_norm = init_sampler.random(TARGET_N_INIT)
            cand_sampler = qmc.Sobol(d=3, scramble=True, seed=seed + 1)
            Xi_cand_norm = cand_sampler.random(cfg["n_candidates"])

            own_xi, own_y, own_feasible = [], [], []
            for k in range(TARGET_N_INIT):
                xi = _denormalize(Xi_init_norm[k], xi_lower, xi_upper)
                ev = cached_evaluate(
                    target, xi, target_bank, state_scale=state_scale, fixed_R=fixed_R,
                    episodes=calibration_episodes(target_bank), **eval_kwargs,
                )
                feasible = bool(ev.feasible and ev.tracking_rmse is not None)
                training_y = (float(ev.tracking_rmse) / ctx["baseline_rmse"][target_id]) if feasible else INFEASIBLE_PENALTY_NORM
                own_xi.append(xi)
                own_y.append(training_y)
                own_feasible.append(feasible)

            own_X_norm = np.array([_normalize(xi, xi_lower, xi_upper) for xi in own_xi])
            own_y_arr = np.array(own_y)
            own_w = np.ones(len(own_xi))
            best_f = min([y for y, f in zip(own_y, own_feasible) if f], default=float(np.min(own_y_arr)))

            peer_records = seed_mature[seed_mature["client_id"].isin(peer_ids)]
            peer_X = peer_records[["xi_1", "xi_2", "xi_3"]].to_numpy()
            peer_X_norm = np.array([_normalize(xi, xi_lower, xi_upper) for xi in peer_X])
            peer_y = peer_records["training_y"].to_numpy()
            peer_client_ids = peer_records["client_id"].to_numpy()
            s_lookup = {p: float(S[ti, client_idx[p]]) for p in peer_ids}
            peer_w_global = np.ones(len(peer_y))
            peer_w_sim = np.array([s_lookup[cid] for cid in peer_client_ids])

            fits = {}
            for method, peer_w in (("global", peer_w_global), ("similarity", peer_w_sim)):
                X_train = np.vstack([peer_X_norm, own_X_norm])
                y_train = np.concatenate([peer_y, own_y_arr])
                w_train = np.concatenate([peer_w, own_w])
                cand_idx, fit = propose_next(X_train, y_train, w_train, Xi_cand_norm, best_f, seed=seed + 1)
                mu, sigma, ei = (None, None, None)
                if fit.gp is not None:
                    mu, sigma = fit.gp.predict(Xi_cand_norm, return_std=True)
                    ei = expected_improvement(fit.gp, Xi_cand_norm, best_f)
                fits[method] = dict(cand_idx=cand_idx, n_train=fit.n_train, mu=mu, sigma=sigma, ei=ei)

            g, s_fit = fits["global"], fits["similarity"]
            same_proposal = g["cand_idx"] == s_fit["cand_idx"]
            xi_dist = float(np.linalg.norm(Xi_cand_norm[g["cand_idx"]] - Xi_cand_norm[s_fit["cand_idx"]]))
            row = {
                "seed": seed, "target_id": target_id, "family": family,
                "n_train_global": g["n_train"], "n_train_similarity": s_fit["n_train"],
                "same_first_proposal": same_proposal, "proposal_xi_distance": xi_dist,
            }
            if g["mu"] is not None and s_fit["mu"] is not None:
                mu_rho, _ = spearmanr(g["mu"], s_fit["mu"])
                sig_rho, _ = spearmanr(g["sigma"], s_fit["sigma"])
                ei_rho, _ = spearmanr(g["ei"], s_fit["ei"])
                row["mu_spearman"] = float(mu_rho)
                row["sigma_spearman"] = float(sig_rho)
                row["ei_spearman"] = float(ei_rho)
                row["mu_max_abs_diff"] = float(np.max(np.abs(g["mu"] - s_fit["mu"])))
                row["mu_mean_abs_diff"] = float(np.mean(np.abs(g["mu"] - s_fit["mu"])))
            gp_rows.append(row)
            print(f"[audit] seed={seed} target={target_id}: same_first_proposal={same_proposal}, "
                  f"n_train(global/similarity)={g['n_train']}/{s_fit['n_train']}")

    s_ij_df = pd.DataFrame(s_ij_rows)
    gp_df = pd.DataFrame(gp_rows)
    redundancy_df = pd.DataFrame(redundancy_rows)
    s_ij_df.to_csv(out_dir / "processed" / "s_ij_distribution.csv", index=False)
    gp_df.to_csv(out_dir / "processed" / "round1_gp_comparison.csv", index=False)
    redundancy_df.to_csv(out_dir / "processed" / "fleet_redundancy.csv", index=False)

    print("\n" + "=" * 90)
    print("s_ij distribution: same-family vs cross-family peers")
    print("=" * 90)
    print(s_ij_df.groupby("same_family")[["s_ij", "alpha_inflation_x"]].describe().round(4))
    print(f"\nfraction of peers dropped entirely under similarity (s_ij < WEIGHT_FLOOR={WEIGHT_FLOOR}): "
          f"{s_ij_df['dropped_under_similarity'].mean():.4f}")
    print(s_ij_df.groupby("same_family")["dropped_under_similarity"].mean().round(4))

    print("\n" + "=" * 90)
    print("Round-1 GP comparison: global vs similarity")
    print("=" * 90)
    print(f"n_train (global vs similarity): mean {gp_df['n_train_global'].mean():.2f} vs {gp_df['n_train_similarity'].mean():.2f}")
    print(f"same first proposal: {gp_df['same_first_proposal'].mean():.3f} ({gp_df['same_first_proposal'].sum()}/{len(gp_df)})")
    print(f"proposal xi distance (normalized [0,1]^3 space) when DIFFERENT: "
          f"{gp_df.loc[~gp_df['same_first_proposal'], 'proposal_xi_distance'].describe()}")
    if "mu_spearman" in gp_df:
        print(gp_df[["mu_spearman", "sigma_spearman", "ei_spearman", "mu_max_abs_diff", "mu_mean_abs_diff"]].describe().round(4))

    print("\n" + "=" * 90)
    print("Fleet redundancy (peer similarity to target, excluding self)")
    print("=" * 90)
    print(redundancy_df[["s_max", "s_3rd", "s_5th", "s_10th", "n_below_weight_floor"]].describe().round(4))

    print(f"\n[audit] wrote {out_dir / 'processed'}/{{s_ij_distribution.csv, round1_gp_comparison.csv, fleet_redundancy.csv}}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/runs/phase2_warmstart_dev_c1cds02.yaml")
    parser.add_argument(
        "--mature-records",
        default="results/phase2_warmstart_dev_c1cds02/processed/mature_fleet_records.parquet",
    )
    args = parser.parse_args()
    main(args.config, args.mature_records)
