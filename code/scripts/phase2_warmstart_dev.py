"""
Frozen mature-fleet leave-one-out warm-start diagnostic (Part 14,
docs/phase2_diagnostic_findings.md, 2026-09-07) - the cleanest test yet of the actual
application claim: "how many new local experiments does a NEW client need, given what
the fleet already knows?" Deliberately does NOT mix in stagnation adaptation (Part 12) -
this establishes whether a mature fleet prior buys local sample efficiency at all,
before combining it with anything else.

Two phases, run once per development seed:

  Phase A (establish): identify the standard 30-client v3 fleet, run `independent` BO
  for ALL 30 clients at the STANDARD mature-fleet budget (n_init=8, n_iters=17 - same
  protocol as every prior campaign), and persist the FULL raw per-evaluation
  (xi, training_y) history for every client - not just the checkpoint summaries the
  other drivers save. This is the one-time cost of having genuine "mature fleet
  knowledge" to warm-start from; reusable for any future expansion (more targets, more
  n_init values) without rerunning.

  Phase B (leave-one-out): for one fixed, deterministically-selected target client per
  family (client slot "_00" - `nominal_00`/`payload_00`/`tire_degraded_00`, chosen
  BEFORE looking at any result, not cherry-picked), remove that client's own history
  entirely, and run THREE fresh warm-start BO loops for it alone
  (`src.optimization.fleet_bo.run_target_warmstart`) starting from `n_init=2`:
  `independent` (no peer access - the target's own reduced-budget baseline),
  `global` (unweighted mature-fleet prior from round 1), `similarity` (`s_ij`-weighted
  mature-fleet prior from round 1). The frozen peer histories are `independent`-method
  histories specifically (never `global`/`similarity`), so the "fleet knowledge" being
  reused isn't itself already federated.

20 seeds x 3 families = 60 target problems, 3 methods each = 180 fresh target BO runs
(25 evaluations each = 4,500 target evaluations) PLUS the one-time Phase A cost (20
seeds x 30 clients x 25 evaluations = 15,000) - roughly 20-30% of a full multi-method
campaign's simulation volume, not the ~10% first estimated (that estimate assumed the
mature histories could be reused for free; they can't, since only checkpoint summaries
were ever persisted by the other drivers - Phase A's raw-record capture is genuinely a
new cost, but a ONE-TIME, reusable one).

Primary metric: N_{i,5%} = evaluations to within 5% of the target's own oracle
(`ClientTrace.evals_to_5pct_of_oracle`, already tracked). Secondary: AURC_star over the
first 10 target-local evaluations, and the full oracle-relative regret curve.

`cfg["target_slots"]` (default `["00"]`, one target/family) and `cfg["target_n_iters"]`
(default 23, i.e. 25 total with `n_init=2`) are configurable, added for the
confirmatory campaign (Part 20, docs/phase2_diagnostic_findings.md, 2026-09-08), which
uses `["00", "01"]` (2 targets/family = 6/seed) and a shorter `n_iters=10` (12 total) -
existing dev configs are unaffected since they don't set these keys.

Usage:
    python scripts/phase2_warmstart_dev.py --config config/runs/phase2_warmstart_dev_c1cds05.yaml
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
from joblib import Parallel, delayed  # noqa: E402

from src.caching import make_cached_evaluate  # noqa: E402
from src.io_utils import deep_update, load_run_config, load_yaml, write_run_manifest  # noqa: E402
from src.optimization.fleet_bo import (  # noqa: E402
    WARMSTART_METHODS,
    FrozenPeerRecord,
    run_fbo_method,
    run_target_warmstart,
)
from src.optimization.regret import normalized_simple_regret, oracle_relative_regret  # noqa: E402

from phase2_fbo_comparison import _prepare_seed_context, _resolve  # noqa: E402
from phase2_medium_analysis import _family_of  # noqa: E402
from phase2_triggered_dev import _seed_bootstrap_mean_ci, _seed_bootstrap_median_ci  # noqa: E402

FAMILIES = ("nominal", "payload", "tire_degraded")
TARGET_SLOT = "00"  # fixed, deterministic, chosen before any result was seen - DEFAULT
# for Part 15's dev sweep (1 target/family); the confirmatory campaign (Part 20)
# overrides via cfg["target_slots"] (e.g. ["00", "01"] for 2 targets/family).
MATURE_N_INIT = 8
MATURE_N_ITERS = 17  # 25 total - matches every prior campaign's "mature fleet" protocol
TARGET_N_INIT = 2
TARGET_N_ITERS = 23  # 25 total, DEFAULT - Part 15's dev sweep; the confirmatory
# campaign (Part 20) overrides via cfg["target_n_iters"] (12 total, per the user's
# "the application claim is sample efficiency, we don't need another 25-eval run")


def _run_one_seed(seed: int, cfg: dict, fleet_cfg: dict, controller_cfg: dict, *, method_kwargs: dict, cache_dir: Path):
    ctx = _prepare_seed_context(
        seed, cfg, fleet_cfg, controller_cfg, cache_dir=cache_dir,
        xi_lower=method_kwargs["xi_lower"], xi_upper=method_kwargs["xi_upper"],
        state_scale=method_kwargs["state_scale"], fixed_R=method_kwargs["fixed_R"],
        eval_kwargs=method_kwargs["eval_kwargs"],
    )
    fleet, banks = ctx["fleet"], ctx["banks"]
    similarity_S = ctx["similarity_S"]
    client_ids = [c.client_id for c in fleet]
    client_idx = {cid: i for i, cid in enumerate(client_ids)}
    cached_evaluate = make_cached_evaluate(str(ctx["seed_cache_dir"]))

    # ---- Phase A: establish the mature fleet's `independent` history, ALL 30 clients ---
    print(f"[warmstart_dev] seed={seed}: Phase A - establishing mature fleet (independent, "
          f"n_init={MATURE_N_INIT}, n_iters={MATURE_N_ITERS}) for {len(fleet)} clients ...")
    mature_result = run_fbo_method(
        fleet, banks, "independent", cache_evaluate=cached_evaluate, similarity_S=similarity_S,
        baseline_rmse=ctx["baseline_rmse"], oracle_rmse=ctx["oracle_rmse"], n_init=MATURE_N_INIT,
        n_iters=MATURE_N_ITERS, n_candidates=cfg["n_candidates"], seed=seed,
        xi_lower=method_kwargs["xi_lower"], xi_upper=method_kwargs["xi_upper"],
        state_scale=method_kwargs["state_scale"], fixed_R=method_kwargs["fixed_R"],
        beta_max=method_kwargs["beta_max"], eval_kwargs=method_kwargs["eval_kwargs"],
    )
    mature_records_by_client: dict[str, list[FrozenPeerRecord]] = {cid: [] for cid in client_ids}
    frozen_rows = []
    for rec in mature_result.all_records:
        fpr = FrozenPeerRecord(client_id=rec.client_id, xi=rec.xi, training_y=rec.training_y)
        mature_records_by_client[rec.client_id].append(fpr)
        frozen_rows.append(
            {"seed": seed, "client_id": rec.client_id, "xi_1": float(rec.xi[0]), "xi_2": float(rec.xi[1]),
             "xi_3": float(rec.xi[2]), "training_y": rec.training_y, "feasible": rec.feasible}
        )
    print(f"[warmstart_dev] seed={seed}: Phase A done ({len(mature_result.all_records)} mature evaluations captured)")

    # ---- Phase B: leave-one-out warm start for the configured target slot(s)/family -----
    target_slots = cfg.get("target_slots", [TARGET_SLOT])
    target_n_iters = cfg.get("target_n_iters", TARGET_N_ITERS)
    trace_rows: list[dict] = []
    summary_rows: list[dict] = []
    for family in FAMILIES:
        for slot in target_slots:
            target_id = f"{family}_{slot}"
            if target_id not in client_idx:
                raise ValueError(f"seed={seed}: expected target client {target_id!r} not found in fleet")
            target = fleet[client_idx[target_id]]
            target_bank = banks[target_id]
            peer_ids = [cid for cid in client_ids if cid != target_id]
            frozen_peer_records = [rec for cid in peer_ids for rec in mature_records_by_client[cid]]
            peer_similarity = {cid: float(similarity_S[client_idx[target_id], client_idx[cid]]) for cid in peer_ids}

            for method in WARMSTART_METHODS:
                print(f"[warmstart_dev] seed={seed}: Phase B - target={target_id!r} method={method!r} "
                      f"(n_init={TARGET_N_INIT}, n_iters={target_n_iters}) ...")
                trace = run_target_warmstart(
                    target, target_bank, method, cache_evaluate=cached_evaluate, xi_lower=method_kwargs["xi_lower"],
                    xi_upper=method_kwargs["xi_upper"], state_scale=method_kwargs["state_scale"],
                    fixed_R=method_kwargs["fixed_R"], eval_kwargs=method_kwargs["eval_kwargs"], n_init=TARGET_N_INIT,
                    n_iters=target_n_iters, n_candidates=cfg["n_candidates"], frozen_peer_records=frozen_peer_records,
                    peer_similarity=peer_similarity, baseline_rmse=ctx["baseline_rmse"][target_id],
                    oracle_rmse=ctx["oracle_rmse"][target_id], seed=seed,
                )
                for k in range(len(trace.n_local_evals)):
                    trace_rows.append(
                        {
                            "seed": seed, "method": method, "client_id": target_id, "family": family,
                            "n_local_evals": trace.n_local_evals[k],
                            "best_calibration_rmse": trace.best_calibration_rmse[k],
                            "best_held_out_rmse": trace.best_held_out_rmse[k],
                            "xi_1": float(trace.best_xi[k][0]), "xi_2": float(trace.best_xi[k][1]),
                            "xi_3": float(trace.best_xi[k][2]),
                            "oracle_rmse": trace.oracle_rmse, "baseline_rmse": trace.baseline_rmse,
                        }
                    )
                summary_rows.append(
                    {
                        "seed": seed, "method": method, "client_id": target_id, "family": family,
                        "oracle_rmse": trace.oracle_rmse, "baseline_rmse": trace.baseline_rmse,
                        "final_held_out_rmse": trace.best_held_out_rmse[-1] if trace.best_held_out_rmse else np.nan,
                        "n_infeasible": trace.n_infeasible, "n_degrading_proposals": trace.n_degrading_proposals,
                        "evals_to_5pct_of_oracle": trace.evals_to_5pct_of_oracle,
                    }
                )

    print(f"[warmstart_dev] seed={seed}: done")
    return trace_rows, summary_rows, frozen_rows


def main(config_path: str) -> None:
    cfg = load_run_config(config_path)
    fleet_cfg = deep_update(load_yaml(_resolve(cfg["fleet_config"])), cfg.get("fleet_overrides", {}))
    controller_cfg = deep_update(load_yaml(_resolve(cfg["controller_config"])), cfg.get("controller_overrides", {}))

    run_tag = cfg["run_tag"]
    out_dir = _ROOT / "results" / run_tag
    processed_dir, cache_dir = out_dir / "processed", out_dir / "cache"
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    beta_max = np.deg2rad(controller_cfg["beta_max_deg"])
    method_kwargs = dict(
        xi_lower=controller_cfg["xi_lower"], xi_upper=controller_cfg["xi_upper"],
        state_scale=controller_cfg["state_scale"], fixed_R=controller_cfg["fixed_R"], beta_max=beta_max,
        eval_kwargs=dict(
            u_min=controller_cfg["u_min"], u_max=controller_cfg["u_max"],
            u_rate_max=controller_cfg.get("u_rate_max"), beta_max=beta_max,
            noise_std=controller_cfg["noise_std"], process_noise=controller_cfg["process_noise"],
            plant_mode=controller_cfg.get("plant_mode", "linear"), mu=controller_cfg.get("tire_mu", 1.0),
        ),
    )

    n_jobs = cfg.get("n_jobs", 1)
    if n_jobs == 1:
        results = [
            _run_one_seed(seed, cfg, fleet_cfg, controller_cfg, method_kwargs=method_kwargs, cache_dir=cache_dir)
            for seed in cfg["seeds"]
        ]
    else:
        print(f"[warmstart_dev] dispatching {len(cfg['seeds'])} seeds across n_jobs={n_jobs} worker processes")
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
            delayed(_run_one_seed)(seed, cfg, fleet_cfg, controller_cfg, method_kwargs=method_kwargs, cache_dir=cache_dir)
            for seed in cfg["seeds"]
        )

    trace_rows = [row for tr, _, _ in results for row in tr]
    summary_rows = [row for _, sr, _ in results for row in sr]
    frozen_rows = [row for _, _, fr in results for row in fr]

    trace_df = pd.DataFrame(trace_rows)
    trace_df["regret"] = normalized_simple_regret(
        trace_df["best_calibration_rmse"], trace_df["oracle_rmse"], trace_df["baseline_rmse"]
    )
    trace_df["regret_star"] = oracle_relative_regret(trace_df["best_calibration_rmse"], trace_df["oracle_rmse"])
    trace_df.to_parquet(processed_dir / "target_convergence_traces.parquet", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_parquet(processed_dir / "target_summary.parquet", index=False)
    pd.DataFrame(frozen_rows).to_parquet(processed_dir / "mature_fleet_records.parquet", index=False)
    print(f"\n[warmstart_dev] wrote {processed_dir / 'target_convergence_traces.parquet'} ({len(trace_df)} rows)")
    print(f"[warmstart_dev] wrote {processed_dir / 'mature_fleet_records.parquet'} ({len(frozen_rows)} rows)")

    write_run_manifest(str(out_dir), {"config": cfg, "fleet_config": fleet_cfg, "controller_config": controller_cfg})

    # ---- PRIMARY: N_{i,5%} - evaluations to within 5% of oracle ------------------------
    print("\n" + "=" * 90)
    print("PRIMARY: N_{i,5%} - target-local evaluations to reach within 5% of the target's own oracle")
    print("=" * 90)
    target_n_iters = cfg.get("target_n_iters", TARGET_N_ITERS)
    budget = TARGET_N_INIT + target_n_iters
    n5 = summary_df.copy()
    n5["N_5pct"] = n5["evals_to_5pct_of_oracle"].fillna(budget)  # impute full budget for non-convergent, same convention as evals_to_threshold_mean
    print(n5.groupby("method")["N_5pct"].agg(["mean", "median", "std"]).round(2))
    print(f"\nfraction reaching 5% within budget ({budget}):")
    print(n5.assign(reached=n5["evals_to_5pct_of_oracle"].notna()).groupby("method")["reached"].mean().round(3))

    print("\n-- fixed-budget success probability P(N_5pct<=3), P(N_5pct<=5) --")
    n5["le3"], n5["le5"] = n5["N_5pct"] <= 3, n5["N_5pct"] <= 5
    print(n5.groupby("method")[["le3", "le5"]].mean().round(3))

    # Part 18's lesson: the per-seed MEDIAN (appropriate for large, discretization-
    # swamping effects) collapses to a coarse, uninformative statistic when there are
    # few targets/seed and small point differences. Report the per-seed MEAN as
    # PRIMARY (more sensitive - the right tool when the design has multiple
    # targets/seed, as the confirmatory campaign, Part 20, does), median kept for
    # continuity with Parts 10-15's convention.
    print("\n-- paired seed-level N_5pct deltas (negative = method_a needs FEWER evaluations) --")
    for a, b in [("similarity", "independent"), ("global", "independent"), ("similarity", "global")]:
        piv = n5[n5["method"].isin([a, b])].pivot_table(index=["seed", "client_id"], columns="method", values="N_5pct")
        piv["delta"] = piv[a] - piv[b]
        per_seed_mean = piv.groupby("seed")["delta"].mean().to_dict()
        res_mean = _seed_bootstrap_mean_ci(per_seed_mean)
        per_seed_median = piv.groupby("seed")["delta"].median().to_dict()
        res_median = _seed_bootstrap_median_ci(per_seed_median)
        print(
            f"  {a} - {b}: PRIMARY mean(Delta_s)={res_mean['mean']:.3f}, 95% CI={res_mean['bootstrap_ci_95']}, "
            f"{res_mean['n_negative']}/{res_mean['n_seeds']} seeds favorable, Wilcoxon p={res_mean['wilcoxon_p']:.4f}"
        )
        print(
            f"    (secondary) median(Delta_s)={res_median['median']:.3f}, 95% CI={res_median['bootstrap_ci_95']}, "
            f"{res_median['n_negative']}/{res_median['n_seeds']} seeds favorable, Wilcoxon p={res_median['wilcoxon_p']:.4f}"
        )

    # ---- SECONDARY: AURC_star over the first 10 target-local evaluations ---------------
    print("\n" + "=" * 90)
    print("SECONDARY: AURC_star over the first 10 target-local evaluations")
    print("=" * 90)
    early = trace_df[trace_df["n_local_evals"] <= 10].sort_values("n_local_evals")

    def _trapz(y, x):
        return float(np.sum((y[1:] + y[:-1]) / 2.0 * np.diff(x)))

    aurc_rows = []
    for (seed, method, cid), g in early.groupby(["seed", "method", "client_id"]):
        if len(g) < 2:
            continue
        aurc_rows.append({"seed": seed, "method": method, "client_id": cid, "aurc_star_early": _trapz(g["regret_star"].to_numpy(), g["n_local_evals"].to_numpy())})
    aurc_early_df = pd.DataFrame(aurc_rows)
    print(aurc_early_df.groupby("method")["aurc_star_early"].agg(["mean", "median", "std"]).round(4))
    for a, b in [("similarity", "independent"), ("global", "independent"), ("similarity", "global")]:
        piv = aurc_early_df[aurc_early_df["method"].isin([a, b])].pivot_table(
            index=["seed", "client_id"], columns="method", values="aurc_star_early"
        )
        piv["delta"] = piv[a] - piv[b]
        per_seed = piv.groupby("seed")["delta"].median().to_dict()
        res = _seed_bootstrap_median_ci(per_seed)
        print(
            f"  {a} - {b}: median(Delta_s)={res['median']:.4f}, 95% CI={res['bootstrap_ci_95']}, "
            f"{res['n_negative']}/{res['n_seeds']} seeds favorable"
        )

    aurc_early_df.to_csv(processed_dir / "aurc_star_early_window.csv", index=False)
    print(f"\n[warmstart_dev] wrote {processed_dir / 'aurc_star_early_window.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/runs/phase2_warmstart_dev_c1cds05.yaml")
    args = parser.parse_args()
    main(args.config)
