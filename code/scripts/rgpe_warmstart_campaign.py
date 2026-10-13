"""Frozen RGPE literature-baseline campaign for the FedBO/LQI paper.

This driver reuses the EXACT mature independent-BO source histories from the frozen
confirmatory campaign and executes RGPE only on the same held-out target systems.
It does not rerun or modify the independent/global/dynamics-weighted methods.

Workflow:
  1) load mature_fleet_records.parquet from the frozen confirmatory campaign;
  2) reconstruct the same fleet, episode banks, baseline and identification context;
  3) prefit one RGPE source GP per historical system once per fleet realization;
  4) run RGPE on the same targets, with the same 2-point Sobol initialization,
     12-evaluation local budget, 256-point candidate set and target episode banks;
  5) save a target_convergence_traces.parquet compatible with the existing
     rescore_bo_trajectories.py robustness pipeline.

Use --check-inputs before a smoke/full run.
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
    RGPE_WARMSTART,
    FrozenPeerRecord,
    fit_rgpe_source_models,
    run_target_warmstart,
)
from src.optimization.regret import normalized_simple_regret, oracle_relative_regret  # noqa: E402

from phase2_fbo_comparison import _prepare_seed_context, _resolve  # noqa: E402
from phase2_triggered_dev import _seed_bootstrap_mean_ci  # noqa: E402

FAMILIES = ("nominal", "payload", "tire_degraded")


def _source_records_path(cfg: dict) -> Path:
    tag = cfg["source_campaign_tag"]
    return _ROOT / "results" / tag / "processed" / "mature_fleet_records.parquet"


def check_inputs(config_path: str) -> None:
    cfg = load_run_config(config_path)
    path = _source_records_path(cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing frozen mature source histories: {path}\n"
            "Copy the frozen confirmatory campaign result folder locally first."
        )
    df = pd.read_parquet(path)
    required = {"seed", "client_id", "xi_1", "xi_2", "xi_3", "training_y"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    expected = int(cfg.get("expected_source_history_per_system", 25))
    wanted_seeds = [int(s) for s in cfg["seeds"]]
    available = set(int(s) for s in df["seed"].unique())
    absent = [s for s in wanted_seeds if s not in available]
    if absent:
        raise ValueError(f"Source-history file is missing configured seeds: {absent}")

    bad = []
    for seed in wanted_seeds:
        counts = df[df["seed"] == seed].groupby("client_id").size()
        if len(counts) != 30 or not np.all(counts.to_numpy() == expected):
            bad.append(
                (seed, len(counts), sorted(set(int(v) for v in counts.to_numpy())))
            )
    if bad:
        raise ValueError(
            "Frozen source histories do not match the expected 30 systems x "
            f"{expected} records: {bad[:5]}"
        )

    print(f"[rgpe] OK: {path}")
    print(
        f"[rgpe] {len(wanted_seeds)} configured seeds; "
        f"30 systems x {expected} mature records per seed"
    )


def _records_for_seed(df: pd.DataFrame, seed: int) -> list[FrozenPeerRecord]:
    sub = df[df["seed"] == seed]
    return [
        FrozenPeerRecord(
            client_id=str(r.client_id),
            xi=np.array([r.xi_1, r.xi_2, r.xi_3], dtype=float),
            training_y=float(r.training_y),
        )
        for r in sub.itertuples()
    ]


def _run_one_seed(
    seed: int,
    cfg: dict,
    fleet_cfg: dict,
    controller_cfg: dict,
    *,
    method_kwargs: dict,
    cache_dir: Path,
    source_records_path: Path,
):
    ctx = _prepare_seed_context(
        seed,
        cfg,
        fleet_cfg,
        controller_cfg,
        cache_dir=cache_dir,
        xi_lower=method_kwargs["xi_lower"],
        xi_upper=method_kwargs["xi_upper"],
        state_scale=method_kwargs["state_scale"],
        fixed_R=method_kwargs["fixed_R"],
        eval_kwargs=method_kwargs["eval_kwargs"],
    )
    fleet, banks = ctx["fleet"], ctx["banks"]
    client_ids = [c.client_id for c in fleet]
    client_idx = {cid: i for i, cid in enumerate(client_ids)}
    similarity_S = ctx["similarity_S"]
    cached_evaluate = make_cached_evaluate(str(ctx["seed_cache_dir"]))

    source_df = pd.read_parquet(source_records_path)
    frozen_all = _records_for_seed(source_df, seed)
    source_models_all = fit_rgpe_source_models(
        frozen_all,
        xi_lower=method_kwargs["xi_lower"],
        xi_upper=method_kwargs["xi_upper"],
        seed=seed + 300_000,
    )
    by_client: dict[str, list[FrozenPeerRecord]] = {}
    for rec in frozen_all:
        by_client.setdefault(rec.client_id, []).append(rec)

    target_slots = cfg.get("target_slots", ["00", "01"])
    n_init = int(cfg.get("target_n_init", 2))
    n_iters = int(cfg.get("target_n_iters", 10))
    num_samples = int(cfg.get("rgpe_num_posterior_samples", 256))
    prune_quantile = float(cfg.get("rgpe_prune_quantile", 0.95))

    trace_rows: list[dict] = []
    summary_rows: list[dict] = []
    weight_rows: list[dict] = []

    for family in FAMILIES:
        for slot in target_slots:
            target_id = f"{family}_{slot}"
            if target_id not in client_idx:
                raise ValueError(
                    f"seed={seed}: expected target {target_id!r} not found"
                )
            target = fleet[client_idx[target_id]]
            peer_ids = [cid for cid in client_ids if cid != target_id]
            frozen_peer_records = [
                rec for peer in peer_ids for rec in by_client[peer]
            ]
            peer_similarity = {
                cid: float(similarity_S[client_idx[target_id], client_idx[cid]])
                for cid in peer_ids
            }

            print(
                f"[rgpe] seed={seed} target={target_id}: "
                f"n_init={n_init}, n_iters={n_iters}, "
                f"sources={len(peer_ids)}, posterior_samples={num_samples}"
            )
            trace = run_target_warmstart(
                target,
                banks[target_id],
                RGPE_WARMSTART,
                cache_evaluate=cached_evaluate,
                xi_lower=method_kwargs["xi_lower"],
                xi_upper=method_kwargs["xi_upper"],
                state_scale=method_kwargs["state_scale"],
                fixed_R=method_kwargs["fixed_R"],
                eval_kwargs=method_kwargs["eval_kwargs"],
                n_init=n_init,
                n_iters=n_iters,
                n_candidates=cfg["n_candidates"],
                frozen_peer_records=frozen_peer_records,
                peer_similarity=peer_similarity,
                baseline_rmse=ctx["baseline_rmse"][target_id],
                oracle_rmse=ctx["oracle_rmse"][target_id],
                seed=seed,
                rgpe_source_models=source_models_all,
                rgpe_num_posterior_samples=num_samples,
                rgpe_prune_quantile=prune_quantile,
            )

            for k in range(len(trace.n_local_evals)):
                trace_rows.append(
                    {
                        "seed": seed,
                        "method": RGPE_WARMSTART,
                        "client_id": target_id,
                        "family": family,
                        "n_local_evals": trace.n_local_evals[k],
                        "best_calibration_rmse": trace.best_calibration_rmse[k],
                        "best_held_out_rmse": trace.best_held_out_rmse[k],
                        "xi_1": float(trace.best_xi[k][0]),
                        "xi_2": float(trace.best_xi[k][1]),
                        "xi_3": float(trace.best_xi[k][2]),
                        "oracle_rmse": trace.oracle_rmse,
                        "baseline_rmse": trace.baseline_rmse,
                    }
                )

            for r, (tw, ns, ent) in enumerate(
                zip(
                    trace.rgpe_target_weight,
                    trace.rgpe_active_sources,
                    trace.rgpe_weight_entropy,
                ),
                start=1,
            ):
                weight_rows.append(
                    {
                        "seed": seed,
                        "client_id": target_id,
                        "family": family,
                        "round": r,
                        "n_local_evals": n_init + r,
                        "target_weight": tw,
                        "n_active_sources": ns,
                        "weight_entropy": ent,
                    }
                )

            summary_rows.append(
                {
                    "seed": seed,
                    "method": RGPE_WARMSTART,
                    "client_id": target_id,
                    "family": family,
                    "oracle_rmse": trace.oracle_rmse,
                    "baseline_rmse": trace.baseline_rmse,
                    "final_held_out_rmse": (
                        trace.best_held_out_rmse[-1]
                        if trace.best_held_out_rmse
                        else np.nan
                    ),
                    "n_infeasible": trace.n_infeasible,
                    "n_degrading_proposals": trace.n_degrading_proposals,
                    "evals_to_5pct_of_oracle": trace.evals_to_5pct_of_oracle,
                }
            )

    return trace_rows, summary_rows, weight_rows


def main(config_path: str, *, n_jobs_override: int | None = None) -> None:
    cfg = load_run_config(config_path)
    check_inputs(config_path)

    fleet_cfg = deep_update(
        load_yaml(_resolve(cfg["fleet_config"])),
        cfg.get("fleet_overrides", {}),
    )
    controller_cfg = deep_update(
        load_yaml(_resolve(cfg["controller_config"])),
        cfg.get("controller_overrides", {}),
    )
    run_tag = cfg["run_tag"]
    out_dir = _ROOT / "results" / run_tag
    processed_dir, cache_dir = out_dir / "processed", out_dir / "cache"
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    beta_max = np.deg2rad(controller_cfg["beta_max_deg"])
    method_kwargs = dict(
        xi_lower=controller_cfg["xi_lower"],
        xi_upper=controller_cfg["xi_upper"],
        state_scale=controller_cfg["state_scale"],
        fixed_R=controller_cfg["fixed_R"],
        eval_kwargs=dict(
            u_min=controller_cfg["u_min"],
            u_max=controller_cfg["u_max"],
            u_rate_max=controller_cfg.get("u_rate_max"),
            beta_max=beta_max,
            noise_std=controller_cfg["noise_std"],
            process_noise=controller_cfg["process_noise"],
            plant_mode=controller_cfg.get("plant_mode", "linear"),
            mu=controller_cfg.get("tire_mu", 1.0),
        ),
    )
    source_path = _source_records_path(cfg)

    n_jobs = (
        int(n_jobs_override)
        if n_jobs_override is not None
        else int(cfg.get("n_jobs", 1))
    )
    if n_jobs == 1:
        results = [
            _run_one_seed(
                seed,
                cfg,
                fleet_cfg,
                controller_cfg,
                method_kwargs=method_kwargs,
                cache_dir=cache_dir,
                source_records_path=source_path,
            )
            for seed in cfg["seeds"]
        ]
    else:
        print(
            f"[rgpe] dispatching {len(cfg['seeds'])} seeds across "
            f"{n_jobs} workers"
        )
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
            delayed(_run_one_seed)(
                seed,
                cfg,
                fleet_cfg,
                controller_cfg,
                method_kwargs=method_kwargs,
                cache_dir=cache_dir,
                source_records_path=source_path,
            )
            for seed in cfg["seeds"]
        )

    trace_rows = [row for tr, _, _ in results for row in tr]
    summary_rows = [row for _, sr, _ in results for row in sr]
    weight_rows = [row for _, _, wr in results for row in wr]

    trace_df = pd.DataFrame(trace_rows)
    trace_df["regret"] = normalized_simple_regret(
        trace_df["best_calibration_rmse"],
        trace_df["oracle_rmse"],
        trace_df["baseline_rmse"],
    )
    trace_df["regret_star"] = oracle_relative_regret(
        trace_df["best_calibration_rmse"],
        trace_df["oracle_rmse"],
    )
    summary_df = pd.DataFrame(summary_rows)
    weights_df = pd.DataFrame(weight_rows)

    trace_path = processed_dir / "target_convergence_traces.parquet"
    summary_path = processed_dir / "target_summary.parquet"
    weights_path = processed_dir / "rgpe_weight_diagnostics.csv"
    trace_df.to_parquet(trace_path, index=False)
    summary_df.to_parquet(summary_path, index=False)
    weights_df.to_csv(weights_path, index=False)
    print(f"[rgpe] wrote {trace_path} ({len(trace_df)} rows)")
    print(f"[rgpe] wrote {summary_path} ({len(summary_df)} rows)")
    print(f"[rgpe] wrote {weights_path} ({len(weights_df)} rows)")

    write_run_manifest(
        str(out_dir),
        {
            "config": cfg,
            "fleet_config": fleet_cfg,
            "controller_config": controller_cfg,
            "source_mature_records": str(source_path),
            "method": RGPE_WARMSTART,
        },
    )

    budget = int(cfg.get("target_n_init", 2)) + int(
        cfg.get("target_n_iters", 10)
    )
    n5 = summary_df.copy()
    n5["N_5pct"] = n5["evals_to_5pct_of_oracle"].fillna(budget)
    print("\nPRELIMINARY (legacy oracle, before dense-reference rescoring):")
    print(n5["N_5pct"].describe().round(3))
    print(
        "fraction reaching within budget:",
        round(float(n5["evals_to_5pct_of_oracle"].notna().mean()), 3),
    )
    if not weights_df.empty:
        print("\nRGPE weight diagnostics:")
        print(
            weights_df[
                ["target_weight", "n_active_sources", "weight_entropy"]
            ].describe().round(3)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/runs/phase2_rgpe_smoke_c1cds02.yaml",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Optional local override of config n_jobs.",
    )
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help="Validate the frozen mature source-history package and exit.",
    )
    args = parser.parse_args()
    if args.check_inputs:
        check_inputs(args.config)
    else:
        main(args.config, n_jobs_override=args.n_jobs)
