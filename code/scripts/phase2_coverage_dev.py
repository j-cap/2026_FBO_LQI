"""
Controlled fleet-coverage warm-start experiment (Part 17, docs/phase2_diagnostic_findings.md,
2026-09-07) - the user's follow-up to Part 16: dynamics-aware weighting is confirmed
numerically real (up to 26.5x noise inflation, 17% different first BO proposals), but
outcome-indistinguishable from unweighted pooling because every target in the full
30-client fleet has >=5 near-duplicate same-family peers. This asks the sharper
question: does dynamics-aware selection start to matter once relevant peer coverage is
scarce, rather than abundant?

Reuses the ALREADY-PERSISTED mature `independent`-method peer histories
(`results/phase2_warmstart_dev_c1cds02/processed/mature_fleet_records.parquet`) - no
Phase A re-run needed, only fresh target BO. For each of the 60 (seed, target) problems
from Part 14/15/16, and each same-family COVERAGE level `q_same` in {0, 1, 3, 5} (fixed
total peer-pool size M=9, so `9 - q_same` peers are drawn from the OTHER two families
combined, not evenly split by family - "cross-family" is the relevant experimental
category here, not which specific other family), 3 deterministic pool-subset
replicates (seeded independently of the BO seed, so replicate choice never depends on
anything the BO loop does), compare:

  - independent:          no peers at all (run ONCE per target - doesn't depend on
                           q_same/replicate since it never sees the pool)
  - global:                unweighted pool of all 9 peers
  - similarity:             s_ij-weighted pool of all 9 peers
  - same_family_reference: (only when q_same>0) unweighted pool of ONLY the q_same
                           same-family peers already drawn into this condition's pool
                           (the cross-family peers in the SAME draw are excluded) - an
                           experimental reference that uses the latent family label,
                           NOT a deployable method: answers "if we somehow knew which
                           historical clients were relevant, would selecting them
                           matter here?"

Shorter budget than Part 14/15 (n_init=2, n_iters=10, 12 total) - the application
claim is sample efficiency, and Part 15 already showed warm-started targets are
usually near-optimal by evaluation 3, so a 25-evaluation budget wastes compute here.

Usage:
    python scripts/phase2_coverage_dev.py --config config/runs/phase2_coverage_dev_c1cds05.yaml
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
from src.episodes import deterministic_client_seed  # noqa: E402
from src.io_utils import deep_update, load_run_config, load_yaml, write_run_manifest  # noqa: E402
from src.optimization.fleet_bo import FrozenPeerRecord, run_target_warmstart  # noqa: E402
from src.optimization.regret import normalized_simple_regret, oracle_relative_regret  # noqa: E402

from phase2_fbo_comparison import _prepare_seed_context, _resolve  # noqa: E402
from phase2_medium_analysis import _family_of  # noqa: E402
from phase2_triggered_dev import _seed_bootstrap_median_ci  # noqa: E402
from phase2_warmstart_dev import FAMILIES, TARGET_SLOT  # noqa: E402

COVERAGE_N_INIT = 2
COVERAGE_N_ITERS = 10  # 12 total - shorter than Part 14/15's 25, per the user's request
POOL_SIZE = 9
Q_SAME_VALUES = (0, 1, 3, 5)
N_REPLICATES = 3


def _draw_pool(seed: int, target_id: str, family: str, q_same: int, replicate: int, all_client_ids: list[str]) -> list[str]:
    """Deterministic, BO-seed-INDEPENDENT peer-pool draw - `deterministic_client_seed`
    keyed on a string that has nothing to do with the BO seed used inside
    `run_target_warmstart`, so which peers land in the pool never depends on anything
    the optimizer does."""
    same_family_pool = [cid for cid in all_client_ids if cid != target_id and _family_of(cid) == family]
    cross_family_pool = [cid for cid in all_client_ids if _family_of(cid) != family]
    rng = np.random.default_rng(deterministic_client_seed(seed, f"{target_id}_qsame{q_same}_rep{replicate}"))
    same_pick = list(rng.choice(same_family_pool, size=q_same, replace=False)) if q_same > 0 else []
    cross_pick = list(rng.choice(cross_family_pool, size=POOL_SIZE - q_same, replace=False))
    return same_pick, cross_pick


def _run_one_seed(seed: int, cfg: dict, fleet_cfg: dict, controller_cfg: dict, *, method_kwargs: dict, cache_dir: Path, mature: pd.DataFrame):
    ctx = _prepare_seed_context(
        seed, cfg, fleet_cfg, controller_cfg, cache_dir=cache_dir,
        xi_lower=method_kwargs["xi_lower"], xi_upper=method_kwargs["xi_upper"],
        state_scale=method_kwargs["state_scale"], fixed_R=method_kwargs["fixed_R"],
        eval_kwargs=method_kwargs["eval_kwargs"],
    )
    fleet, banks, S = ctx["fleet"], ctx["banks"], ctx["similarity_S"]
    client_ids = [c.client_id for c in fleet]
    client_idx = {cid: i for i, cid in enumerate(client_ids)}
    cached_evaluate = make_cached_evaluate(str(ctx["seed_cache_dir"]))
    seed_mature = mature[mature["seed"] == seed]

    def _records_for(cids: list[str]) -> list[FrozenPeerRecord]:
        sub = seed_mature[seed_mature["client_id"].isin(cids)]
        return [
            FrozenPeerRecord(client_id=row.client_id, xi=np.array([row.xi_1, row.xi_2, row.xi_3]), training_y=row.training_y)
            for row in sub.itertuples()
        ]

    def _run(target, target_bank, method, frozen_records, peer_similarity, label_extra, method_kwargs=method_kwargs, cfg=cfg, ctx=ctx):
        trace = run_target_warmstart(
            target, target_bank, method, cache_evaluate=cached_evaluate, xi_lower=method_kwargs["xi_lower"],
            xi_upper=method_kwargs["xi_upper"], state_scale=method_kwargs["state_scale"], fixed_R=method_kwargs["fixed_R"],
            eval_kwargs=method_kwargs["eval_kwargs"], n_init=COVERAGE_N_INIT, n_iters=COVERAGE_N_ITERS,
            n_candidates=cfg["n_candidates"], frozen_peer_records=frozen_records, peer_similarity=peer_similarity,
            baseline_rmse=ctx["baseline_rmse"][target.client_id], oracle_rmse=ctx["oracle_rmse"][target.client_id], seed=seed,
        )
        rows = []
        for k in range(len(trace.n_local_evals)):
            rows.append(
                {
                    "seed": seed, "target_id": target.client_id, "family": _family_of(target.client_id), **label_extra,
                    "n_local_evals": trace.n_local_evals[k], "best_calibration_rmse": trace.best_calibration_rmse[k],
                    "best_held_out_rmse": trace.best_held_out_rmse[k], "oracle_rmse": trace.oracle_rmse,
                    "baseline_rmse": trace.baseline_rmse,
                }
            )
        summary = {
            "seed": seed, "target_id": target.client_id, "family": _family_of(target.client_id), **label_extra,
            "oracle_rmse": trace.oracle_rmse, "baseline_rmse": trace.baseline_rmse,
            "n_infeasible": trace.n_infeasible, "evals_to_5pct_of_oracle": trace.evals_to_5pct_of_oracle,
        }
        return rows, summary

    trace_rows, summary_rows = [], []
    for family in FAMILIES:
        target_id = f"{family}_{TARGET_SLOT}"
        target = fleet[client_idx[target_id]]
        target_bank = banks[target_id]

        # independent: run ONCE, doesn't depend on q_same/replicate
        r, s = _run(target, target_bank, "independent", [], {}, {"q_same": -1, "replicate": -1, "method": "independent"})
        trace_rows.extend(r)
        summary_rows.append(s)

        for q_same in Q_SAME_VALUES:
            for rep in range(N_REPLICATES):
                same_pick, cross_pick = _draw_pool(seed, target_id, family, q_same, rep, client_ids)
                pool_ids = same_pick + cross_pick
                pool_records = _records_for(pool_ids)
                peer_similarity = {cid: float(S[client_idx[target_id], client_idx[cid]]) for cid in pool_ids}

                for method, records, sim in (
                    ("global", pool_records, {}),
                    ("similarity", pool_records, peer_similarity),
                ):
                    r, s = _run(
                        target, target_bank, method, records, sim,
                        {"q_same": q_same, "replicate": rep, "method": method},
                    )
                    trace_rows.extend(r)
                    summary_rows.append(s)

                if q_same > 0:
                    same_records = _records_for(same_pick)
                    r, s = _run(
                        target, target_bank, "global", same_records, {},
                        {"q_same": q_same, "replicate": rep, "method": "same_family_reference"},
                    )
                    trace_rows.extend(r)
                    summary_rows.append(s)

        print(f"[coverage_dev] seed={seed}: target={target_id!r} done")

    return trace_rows, summary_rows


def main(config_path: str) -> None:
    cfg = load_run_config(config_path)
    fleet_cfg = deep_update(load_yaml(_resolve(cfg["fleet_config"])), cfg.get("fleet_overrides", {}))
    controller_cfg = deep_update(load_yaml(_resolve(cfg["controller_config"])), cfg.get("controller_overrides", {}))
    mature = pd.read_parquet(_resolve(cfg["mature_records_path"]))

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
            _run_one_seed(seed, cfg, fleet_cfg, controller_cfg, method_kwargs=method_kwargs, cache_dir=cache_dir, mature=mature)
            for seed in cfg["seeds"]
        ]
    else:
        print(f"[coverage_dev] dispatching {len(cfg['seeds'])} seeds across n_jobs={n_jobs} worker processes")
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
            delayed(_run_one_seed)(seed, cfg, fleet_cfg, controller_cfg, method_kwargs=method_kwargs, cache_dir=cache_dir, mature=mature)
            for seed in cfg["seeds"]
        )

    trace_rows = [row for tr, _ in results for row in tr]
    summary_rows = [row for _, sr in results for row in sr]

    trace_df = pd.DataFrame(trace_rows)
    trace_df["regret"] = normalized_simple_regret(trace_df["best_calibration_rmse"], trace_df["oracle_rmse"], trace_df["baseline_rmse"])
    trace_df["regret_star"] = oracle_relative_regret(trace_df["best_calibration_rmse"], trace_df["oracle_rmse"])
    trace_df.to_parquet(processed_dir / "coverage_traces.parquet", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_parquet(processed_dir / "coverage_summary.parquet", index=False)
    print(f"\n[coverage_dev] wrote {processed_dir / 'coverage_traces.parquet'} ({len(trace_df)} rows)")
    write_run_manifest(str(out_dir), {"config": cfg, "fleet_config": fleet_cfg, "controller_config": controller_cfg})

    # ---- console report -----------------------------------------------------------
    budget = COVERAGE_N_INIT + COVERAGE_N_ITERS
    n5 = summary_df.copy()
    n5["N_5pct"] = n5["evals_to_5pct_of_oracle"].fillna(budget)

    print("\n" + "=" * 90)
    print("N_5pct by q_same x method (independent is q_same=-1, doesn't vary with coverage)")
    print("=" * 90)
    print(n5.groupby(["q_same", "method"])["N_5pct"].agg(["mean", "median", "std", "count"]).round(3))

    print("\n" + "=" * 90)
    print("Success probability P(N_5pct<=3) and P(N_5pct<=5), by q_same x method")
    print("=" * 90)
    n5["le3"] = n5["N_5pct"] <= 3
    n5["le5"] = n5["N_5pct"] <= 5
    print(n5.groupby(["q_same", "method"])[["le3", "le5"]].mean().round(3))

    print("\n" + "=" * 90)
    print("PRIMARY: similarity - global paired seed-level N_5pct delta, by q_same")
    print("=" * 90)
    for q_same in Q_SAME_VALUES:
        sub = n5[n5["q_same"] == q_same]
        piv = sub[sub["method"].isin(["global", "similarity"])].pivot_table(
            index=["seed", "target_id", "replicate"], columns="method", values="N_5pct"
        )
        piv["delta"] = piv["similarity"] - piv["global"]
        per_seed = piv.groupby("seed")["delta"].median().to_dict()
        res = _seed_bootstrap_median_ci(per_seed)
        print(
            f"  q_same={q_same}: median(Delta_s)={res['median']:.3f}, 95% CI={res['bootstrap_ci_95']}, "
            f"{res['n_negative']}/{res['n_seeds']} seeds favorable, Wilcoxon p={res['wilcoxon_p']:.4f}"
        )

    print("\n" + "=" * 90)
    print("Diagnostic: same_family_reference vs global/similarity, by q_same (q_same>0 only)")
    print("=" * 90)
    for q_same in [q for q in Q_SAME_VALUES if q > 0]:
        sub = n5[n5["q_same"] == q_same]
        for comp_a, comp_b in [("same_family_reference", "global"), ("same_family_reference", "similarity")]:
            piv = sub[sub["method"].isin([comp_a, comp_b])].pivot_table(
                index=["seed", "target_id", "replicate"], columns="method", values="N_5pct"
            )
            piv["delta"] = piv[comp_a] - piv[comp_b]
            per_seed = piv.groupby("seed")["delta"].median().to_dict()
            res = _seed_bootstrap_median_ci(per_seed)
            print(
                f"  q_same={q_same}: {comp_a} - {comp_b}: median={res['median']:.3f}, CI={res['bootstrap_ci_95']}, "
                f"{res['n_negative']}/{res['n_seeds']} favorable"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/runs/phase2_coverage_dev_c1cds05.yaml")
    args = parser.parse_args()
    main(args.config)
