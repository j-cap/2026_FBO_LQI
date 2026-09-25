"""
Fleet-maturity sensitivity experiment for fleet-informed BO.

Scientific question
-------------------
How much historical calibration experience must each previously commissioned source
system contribute before fleet warm start materially reduces the number of new-system
experiments?

Design
------
For every fleet realization, generate ONE 25-evaluation independent-BO history for
all source systems, starting from n_init=2. The source-history maturity conditions are
nested prefixes of these exact same histories:

    M_hist in {2, 5, 10, 25}

For each held-out target, remove its own source history entirely. Run:
  * independent BO once, with no peer data
  * global warm-start BO once per maturity level, using the first M_hist records from
    every other source system

The target-side optimizer, initialization, candidate set, evaluation randomness, and
12-evaluation local budget are identical across maturity conditions. Only the amount
of available historical source information changes.

This driver stores the raw target incumbent trajectories and source histories. Its
built-in N_5% statistics use the legacy 32-point monitoring reference only as a quick
run diagnostic. Paper-level results must be produced with
scripts/rescore_maturity_sweep.py against the dense independently scored reference.

Usage:
    python scripts/fleet_maturity_sweep.py --config config/runs/fleet_maturity_sweep_smoke.yaml
    python scripts/run_remote.py fleet_maturity_sweep_c1cds02.yaml --script fleet_maturity_sweep.py
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
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
from src.optimization.fleet_bo import FrozenPeerRecord, run_fbo_method, run_target_warmstart  # noqa: E402

from phase2_fbo_comparison import _prepare_seed_context, _resolve  # noqa: E402
from phase2_triggered_dev import _seed_bootstrap_mean_ci  # noqa: E402

FAMILIES = ("nominal", "payload", "tire_degraded")
DEFAULT_MATURITIES = (2, 5, 10, 25)
DEFAULT_SOURCE_N_INIT = 2
DEFAULT_SOURCE_N_TOTAL = 25
DEFAULT_TARGET_N_INIT = 2
DEFAULT_TARGET_N_ITERS = 10


def _records_by_client(all_records):
    """Preserve each client's actual evaluation order from run_fbo_method."""
    grouped = defaultdict(list)
    for rec in all_records:
        grouped[rec.client_id].append(rec)
    return dict(grouped)


def _frozen_prefix(records_by_client, peer_ids, maturity: int):
    """Return exactly the first maturity records from every requested peer."""
    frozen = []
    for cid in peer_ids:
        records = records_by_client[cid]
        if len(records) < maturity:
            raise RuntimeError(
                f"source client {cid!r} has only {len(records)} records, cannot build maturity={maturity}"
            )
        frozen.extend(
            FrozenPeerRecord(client_id=rec.client_id, xi=np.asarray(rec.xi, float), training_y=float(rec.training_y))
            for rec in records[:maturity]
        )
    return frozen


def _append_trace_rows(rows, trace, *, seed, client_id, family, method, maturity):
    for k in range(len(trace.n_local_evals)):
        rows.append(
            {
                "seed": int(seed),
                "method": method,
                "maturity": int(maturity),
                "client_id": client_id,
                "family": family,
                "n_local_evals": int(trace.n_local_evals[k]),
                "best_calibration_rmse": float(trace.best_calibration_rmse[k]),
                "best_held_out_rmse": float(trace.best_held_out_rmse[k]),
                "xi_1": float(trace.best_xi[k][0]),
                "xi_2": float(trace.best_xi[k][1]),
                "xi_3": float(trace.best_xi[k][2]),
                "oracle_rmse_weak_32pt": float(trace.oracle_rmse),
                "baseline_rmse": float(trace.baseline_rmse),
            }
        )


def _append_summary_row(rows, trace, *, seed, client_id, family, method, maturity):
    rows.append(
        {
            "seed": int(seed),
            "method": method,
            "maturity": int(maturity),
            "client_id": client_id,
            "family": family,
            "oracle_rmse_weak_32pt": float(trace.oracle_rmse),
            "baseline_rmse": float(trace.baseline_rmse),
            "final_held_out_rmse": float(trace.best_held_out_rmse[-1]) if trace.best_held_out_rmse else np.nan,
            "n_infeasible": int(trace.n_infeasible),
            "evals_to_5pct_of_weak_oracle": trace.evals_to_5pct_of_oracle,
        }
    )


def _run_one_seed(seed: int, cfg: dict, fleet_cfg: dict, controller_cfg: dict, *, method_kwargs: dict, cache_dir: Path):
    maturities = tuple(int(m) for m in cfg.get("maturity_levels", DEFAULT_MATURITIES))
    source_n_init = int(cfg.get("source_n_init", DEFAULT_SOURCE_N_INIT))
    source_n_total = int(cfg.get("source_n_total", DEFAULT_SOURCE_N_TOTAL))
    target_n_init = int(cfg.get("target_n_init", DEFAULT_TARGET_N_INIT))
    target_n_iters = int(cfg.get("target_n_iters", DEFAULT_TARGET_N_ITERS))
    target_slots = [str(x).zfill(2) for x in cfg.get("target_slots", ["00", "01"])]

    if sorted(set(maturities)) != list(maturities):
        raise ValueError(f"maturity_levels must be unique and sorted ascending, got {maturities}")
    if min(maturities) < 1 or max(maturities) > source_n_total:
        raise ValueError(f"maturity levels {maturities} must lie within [1, source_n_total={source_n_total}]")
    if source_n_init > min(maturities):
        raise ValueError(
            f"source_n_init={source_n_init} exceeds smallest maturity={min(maturities)}; "
            "the low-maturity condition would not represent a complete source trajectory prefix"
        )

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
    similarity_S = ctx["similarity_S"]
    client_ids = [c.client_id for c in fleet]
    client_idx = {cid: i for i, cid in enumerate(client_ids)}
    cached_evaluate = make_cached_evaluate(str(ctx["seed_cache_dir"]))

    # Phase A: ONE nested 25-evaluation independent history per fleet system.
    source_n_iters = source_n_total - source_n_init
    print(
        f"[maturity] seed={seed}: generating nested source histories "
        f"(n_init={source_n_init}, total={source_n_total}) for {len(fleet)} systems"
    )
    source_result = run_fbo_method(
        fleet,
        banks,
        "independent",
        cache_evaluate=cached_evaluate,
        similarity_S=similarity_S,
        baseline_rmse=ctx["baseline_rmse"],
        oracle_rmse=ctx["oracle_rmse"],
        n_init=source_n_init,
        n_iters=source_n_iters,
        n_candidates=cfg["n_candidates"],
        seed=seed,
        xi_lower=method_kwargs["xi_lower"],
        xi_upper=method_kwargs["xi_upper"],
        state_scale=method_kwargs["state_scale"],
        fixed_R=method_kwargs["fixed_R"],
        beta_max=method_kwargs["beta_max"],
        eval_kwargs=method_kwargs["eval_kwargs"],
    )
    source_by_client = _records_by_client(source_result.all_records)

    missing = {cid: len(source_by_client.get(cid, [])) for cid in client_ids if len(source_by_client.get(cid, [])) != source_n_total}
    if missing:
        raise RuntimeError(f"source history length mismatch, expected {source_n_total} per client: {missing}")

    source_rows = []
    for cid in client_ids:
        for eval_index, rec in enumerate(source_by_client[cid], start=1):
            source_rows.append(
                {
                    "seed": int(seed),
                    "client_id": cid,
                    "family": cid.rsplit("_", 1)[0],
                    "eval_index": int(eval_index),
                    "xi_1": float(rec.xi[0]),
                    "xi_2": float(rec.xi[1]),
                    "xi_3": float(rec.xi[2]),
                    "training_y": float(rec.training_y),
                    "feasible": bool(rec.feasible),
                    "source": rec.source,
                    "round_idx": int(rec.round_idx),
                }
            )

    # Phase B: target commissioning. Independent is run exactly once; global is run for
    # every nested source maturity. Same target seed -> same target init/candidate grid.
    trace_rows = []
    summary_rows = []
    for family in FAMILIES:
        for slot in target_slots:
            target_id = f"{family}_{slot}"
            if target_id not in client_idx:
                raise ValueError(f"seed={seed}: target {target_id!r} not found")
            target = fleet[client_idx[target_id]]
            target_bank = banks[target_id]
            peer_ids = [cid for cid in client_ids if cid != target_id]
            peer_similarity = {
                cid: float(similarity_S[client_idx[target_id], client_idx[cid]]) for cid in peer_ids
            }

            independent = run_target_warmstart(
                target,
                target_bank,
                "independent",
                cache_evaluate=cached_evaluate,
                xi_lower=method_kwargs["xi_lower"],
                xi_upper=method_kwargs["xi_upper"],
                state_scale=method_kwargs["state_scale"],
                fixed_R=method_kwargs["fixed_R"],
                eval_kwargs=method_kwargs["eval_kwargs"],
                n_init=target_n_init,
                n_iters=target_n_iters,
                n_candidates=cfg["n_candidates"],
                frozen_peer_records=[],
                peer_similarity={},
                baseline_rmse=ctx["baseline_rmse"][target_id],
                oracle_rmse=ctx["oracle_rmse"][target_id],
                seed=seed,
            )
            _append_trace_rows(
                trace_rows, independent, seed=seed, client_id=target_id, family=family,
                method="independent", maturity=0,
            )
            _append_summary_row(
                summary_rows, independent, seed=seed, client_id=target_id, family=family,
                method="independent", maturity=0,
            )

            for maturity in maturities:
                frozen = _frozen_prefix(source_by_client, peer_ids, maturity)
                expected = len(peer_ids) * maturity
                if len(frozen) != expected:
                    raise AssertionError(f"expected {expected} frozen records, got {len(frozen)}")
                trace = run_target_warmstart(
                    target,
                    target_bank,
                    "global",
                    cache_evaluate=cached_evaluate,
                    xi_lower=method_kwargs["xi_lower"],
                    xi_upper=method_kwargs["xi_upper"],
                    state_scale=method_kwargs["state_scale"],
                    fixed_R=method_kwargs["fixed_R"],
                    eval_kwargs=method_kwargs["eval_kwargs"],
                    n_init=target_n_init,
                    n_iters=target_n_iters,
                    n_candidates=cfg["n_candidates"],
                    frozen_peer_records=frozen,
                    peer_similarity=peer_similarity,
                    baseline_rmse=ctx["baseline_rmse"][target_id],
                    oracle_rmse=ctx["oracle_rmse"][target_id],
                    seed=seed,
                )
                _append_trace_rows(
                    trace_rows, trace, seed=seed, client_id=target_id, family=family,
                    method="global", maturity=maturity,
                )
                _append_summary_row(
                    summary_rows, trace, seed=seed, client_id=target_id, family=family,
                    method="global", maturity=maturity,
                )

    print(f"[maturity] seed={seed}: done")
    return trace_rows, summary_rows, source_rows


def _diagnostic_summary(summary_df: pd.DataFrame, *, budget: int, maturities):
    """Legacy-reference run diagnostic only; not paper-level inference."""
    diag = summary_df.copy()
    diag["N_5pct_weak"] = diag["evals_to_5pct_of_weak_oracle"].fillna(budget)
    diag["reached_weak"] = diag["evals_to_5pct_of_weak_oracle"].notna()

    print("\n" + "=" * 92)
    print("RUN DIAGNOSTIC ONLY: N_5% against the legacy 32-point monitoring reference")
    print("Paper-level maturity results must come from rescore_maturity_sweep.py.")
    print("=" * 92)
    ind = diag[diag["method"] == "independent"]
    print(
        f"independent: restricted mean={ind['N_5pct_weak'].mean():.3f}, "
        f"reach={ind['reached_weak'].mean():.3f}"
    )
    for maturity in maturities:
        g = diag[(diag["method"] == "global") & (diag["maturity"] == maturity)]
        print(
            f"M={maturity:>2}: global restricted mean={g['N_5pct_weak'].mean():.3f}, "
            f"reach={g['reached_weak'].mean():.3f}"
        )


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
        xi_lower=controller_cfg["xi_lower"],
        xi_upper=controller_cfg["xi_upper"],
        state_scale=controller_cfg["state_scale"],
        fixed_R=controller_cfg["fixed_R"],
        beta_max=beta_max,
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

    n_jobs = int(cfg.get("n_jobs", 1))
    if n_jobs == 1:
        results = [
            _run_one_seed(seed, cfg, fleet_cfg, controller_cfg, method_kwargs=method_kwargs, cache_dir=cache_dir)
            for seed in cfg["seeds"]
        ]
    else:
        print(f"[maturity] dispatching {len(cfg['seeds'])} fleet realizations across n_jobs={n_jobs}")
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
            delayed(_run_one_seed)(
                seed, cfg, fleet_cfg, controller_cfg, method_kwargs=method_kwargs, cache_dir=cache_dir
            )
            for seed in cfg["seeds"]
        )

    trace_df = pd.DataFrame([row for tr, _, _ in results for row in tr])
    summary_df = pd.DataFrame([row for _, sr, _ in results for row in sr])
    source_df = pd.DataFrame([row for _, _, fr in results for row in fr])

    trace_path = processed_dir / "maturity_target_traces.parquet"
    summary_path = processed_dir / "maturity_target_summary.parquet"
    source_path = processed_dir / "nested_source_histories.parquet"
    trace_df.to_parquet(trace_path, index=False)
    summary_df.to_parquet(summary_path, index=False)
    source_df.to_parquet(source_path, index=False)

    maturities = tuple(int(m) for m in cfg.get("maturity_levels", DEFAULT_MATURITIES))
    target_n_init = int(cfg.get("target_n_init", DEFAULT_TARGET_N_INIT))
    target_n_iters = int(cfg.get("target_n_iters", DEFAULT_TARGET_N_ITERS))
    _diagnostic_summary(summary_df, budget=target_n_init + target_n_iters, maturities=maturities)

    write_run_manifest(
        str(out_dir),
        {
            "config": cfg,
            "fleet_config": fleet_cfg,
            "controller_config": controller_cfg,
            "design": {
                "nested_source_histories": True,
                "maturity_levels": list(maturities),
                "source_n_init": int(cfg.get("source_n_init", DEFAULT_SOURCE_N_INIT)),
                "source_n_total": int(cfg.get("source_n_total", DEFAULT_SOURCE_N_TOTAL)),
                "target_n_init": target_n_init,
                "target_n_iters": target_n_iters,
                "primary_analysis": "dense independently scored reference via rescore_maturity_sweep.py",
            },
        },
    )
    print(f"\n[maturity] wrote {trace_path}")
    print(f"[maturity] wrote {summary_path}")
    print(f"[maturity] wrote {source_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/runs/fleet_maturity_sweep_smoke.yaml")
    args = parser.parse_args()
    main(args.config)
