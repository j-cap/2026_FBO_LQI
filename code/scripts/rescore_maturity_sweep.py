"""
Paper-level rescoring and statistical analysis for the fleet-maturity sweep.

The maturity driver deliberately stores BO trajectories without changing the frozen
optimizer. This script evaluates every stored incumbent on the SAME independent
reference-test bank used by scripts/generate_offline_reference.py and computes
N_5% against J_ref_test.

Expected inputs:
  results/<run_tag>/processed/maturity_target_traces.parquet
  results/<offline_reference_tag>/processed/dense_reference.csv

For the full sensitivity analysis, the maturity sweep reuses the same held-out fleets
as the final confirmatory campaign, so the established dense references can be reused.

Usage:
  python scripts/rescore_maturity_sweep.py --config config/runs/fleet_maturity_sweep_c1cds02.yaml
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
from src.episodes import build_multi_maneuver_episode_bank, test_episodes  # noqa: E402
from src.io_utils import deep_update, load_run_config, load_yaml, write_run_manifest  # noqa: E402

from generate_offline_reference import REFERENCE_TEST_BASE_SEED  # noqa: E402
from phase2_fbo_comparison import _prepare_seed_context, _resolve  # noqa: E402
from phase2_triggered_dev import _seed_bootstrap_mean_ci  # noqa: E402


def _rescore_one_seed(
    seed,
    seed_group: pd.DataFrame,
    cfg: dict,
    fleet_cfg: dict,
    controller_cfg: dict,
    *,
    state_scale,
    fixed_R,
    eval_kwargs: dict,
    cache_dir: Path,
    ref_lookup: dict,
    n_ref_test_per_maneuver: int,
):
    ctx = _prepare_seed_context(
        int(seed),
        cfg,
        fleet_cfg,
        controller_cfg,
        cache_dir=cache_dir,
        xi_lower=controller_cfg["xi_lower"],
        xi_upper=controller_cfg["xi_upper"],
        state_scale=state_scale,
        fixed_R=fixed_R,
        eval_kwargs=eval_kwargs,
    )
    fleet = ctx["fleet"]
    client_idx = {c.client_id: i for i, c in enumerate(fleet)}
    cached_evaluate = make_cached_evaluate(str(ctx["seed_cache_dir"]))

    rows = []
    for client_id, client_group in seed_group.groupby("client_id"):
        target = fleet[client_idx[client_id]]
        J_ref = float(ref_lookup[(int(seed), client_id)])
        ref_bank = build_multi_maneuver_episode_bank(
            target,
            Ts=fleet_cfg["Ts"],
            T_total=controller_cfg["T_total"],
            T0=controller_cfg["T0"],
            r_max=controller_cfg["r_max"],
            lane_change_time=fleet_cfg["lane_change_time"],
            n_calibration_per_maneuver=0,
            n_test_per_maneuver=n_ref_test_per_maneuver,
            base_seed=REFERENCE_TEST_BASE_SEED,
            tfilter=controller_cfg.get("tfilter"),
        )
        ref_eps = test_episodes(ref_bank)

        for (method, maturity), group in client_group.groupby(["method", "maturity"]):
            group = group.sort_values("n_local_evals")
            best_test = np.inf
            n5 = None
            for row in group.itertuples():
                xi = np.array([row.xi_1, row.xi_2, row.xi_3], float)
                ev = cached_evaluate(
                    target,
                    xi,
                    ref_bank,
                    state_scale=state_scale,
                    fixed_R=fixed_R,
                    episodes=ref_eps,
                    **eval_kwargs,
                )
                J = (
                    float(ev.tracking_rmse)
                    if ev.feasible and ev.tracking_rmse is not None and ev.tracking_rmse > 0
                    else np.nan
                )
                if np.isfinite(J):
                    best_test = min(best_test, J)
                if n5 is None and np.isfinite(best_test) and best_test <= 1.05 * J_ref:
                    n5 = int(row.n_local_evals)
                rows.append(
                    {
                        "seed": int(seed),
                        "client_id": client_id,
                        "family": row.family,
                        "method": method,
                        "maturity": int(maturity),
                        "n_local_evals": int(row.n_local_evals),
                        "J_incumbent_test": J,
                        "best_test_so_far": best_test if np.isfinite(best_test) else np.nan,
                        "J_ref_test": J_ref,
                        "regret_ref": (best_test - J_ref) / J_ref if np.isfinite(best_test) else np.nan,
                        "n5pct_test": n5,
                    }
                )
    print(f"[maturity-rescore] seed={seed}: {len(rows)} rescored checkpoints")
    return rows


def _paired_delta(n5_df: pd.DataFrame, maturity: int):
    g = n5_df[(n5_df["method"] == "global") & (n5_df["maturity"] == maturity)][
        ["seed", "client_id", "N_5pct_test"]
    ].rename(columns={"N_5pct_test": "global"})
    ind = n5_df[n5_df["method"] == "independent"][
        ["seed", "client_id", "N_5pct_test"]
    ].rename(columns={"N_5pct_test": "independent"})
    p = g.merge(ind, on=["seed", "client_id"], validate="one_to_one")
    p["delta"] = p["global"] - p["independent"]
    per_seed = p.groupby("seed")["delta"].mean().to_dict()
    return p, _seed_bootstrap_mean_ci(per_seed)


def main(config_path: str, n_jobs: int | None = None) -> None:
    cfg = load_run_config(config_path)
    fleet_cfg = deep_update(load_yaml(_resolve(cfg["fleet_config"])), cfg.get("fleet_overrides", {}))
    controller_cfg = deep_update(load_yaml(_resolve(cfg["controller_config"])), cfg.get("controller_overrides", {}))

    run_tag = cfg["run_tag"]
    maturity_dir = _ROOT / "results" / run_tag
    trace_path = maturity_dir / "processed" / "maturity_target_traces.parquet"
    if not trace_path.exists():
        raise FileNotFoundError(
            f"{trace_path} does not exist. Run scripts/fleet_maturity_sweep.py first."
        )
    trace_df = pd.read_parquet(trace_path)

    reference_tag = cfg.get("offline_reference_tag", "offline_reference")
    ref_dir = _ROOT / "results" / reference_tag
    ref_path = ref_dir / "processed" / "dense_reference.csv"
    if not ref_path.exists():
        raise FileNotFoundError(
            f"{ref_path} does not exist. Generate the dense reference with:\n"
            f"  python scripts/generate_offline_reference.py --config {config_path} "
            f"--mode full --n-sobol 1024"
        )

    ref_df = pd.read_csv(ref_path)
    ref_df = ref_df[ref_df["reference_feasible"]].copy()
    ref_lookup = {(int(r.seed), r.client_id): float(r.J_ref_test) for r in ref_df.itertuples()}
    required = set(zip(trace_df["seed"].astype(int), trace_df["client_id"]))
    missing = required - set(ref_lookup)
    if missing:
        preview = sorted(missing)[:10]
        raise RuntimeError(
            f"dense reference is missing {len(missing)} maturity targets, e.g. {preview}. "
            "Use the same fleets/target slots or generate a matching reference."
        )

    manifest_path = ref_dir / "run_manifest.yaml"
    n_ref_test_per_maneuver = int(cfg.get("n_ref_test_per_maneuver", 20))
    if manifest_path.exists():
        manifest = load_yaml(str(manifest_path)).get("resolved_config", {})
        n_ref_test_per_maneuver = int(manifest.get("n_ref_test_per_maneuver", n_ref_test_per_maneuver))

    beta_max = np.deg2rad(controller_cfg["beta_max_deg"])
    eval_kwargs = dict(
        u_min=controller_cfg["u_min"],
        u_max=controller_cfg["u_max"],
        u_rate_max=controller_cfg.get("u_rate_max"),
        beta_max=beta_max,
        noise_std=controller_cfg["noise_std"],
        process_noise=controller_cfg["process_noise"],
        plant_mode=controller_cfg.get("plant_mode", "linear"),
        mu=controller_cfg.get("tire_mu", 1.0),
    )

    out_tag = cfg.get("maturity_analysis_tag", f"{run_tag}_reference_robustness")
    out_dir = _ROOT / "results" / out_tag
    processed_dir = out_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = ref_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    jobs = int(cfg.get("n_jobs", 1) if n_jobs is None else n_jobs)
    seed_groups = list(trace_df.groupby("seed"))
    kwargs = dict(
        cfg=cfg,
        fleet_cfg=fleet_cfg,
        controller_cfg=controller_cfg,
        state_scale=controller_cfg["state_scale"],
        fixed_R=controller_cfg["fixed_R"],
        eval_kwargs=eval_kwargs,
        cache_dir=cache_dir,
        ref_lookup=ref_lookup,
        n_ref_test_per_maneuver=n_ref_test_per_maneuver,
    )
    if jobs == 1:
        results = [_rescore_one_seed(seed, group, **kwargs) for seed, group in seed_groups]
    else:
        results = Parallel(n_jobs=jobs, backend="loky", verbose=5)(
            delayed(_rescore_one_seed)(seed, group, **kwargs) for seed, group in seed_groups
        )

    scores = pd.DataFrame([row for seed_rows in results for row in seed_rows])
    scores_path = processed_dir / "maturity_candidate_test_scores.csv"
    scores.to_csv(scores_path, index=False)

    keys = ["seed", "client_id", "family", "method", "maturity"]
    n5 = scores.sort_values("n_local_evals").groupby(keys, as_index=False).agg(
        n5pct_test=("n5pct_test", "max"),
        J_ref_test=("J_ref_test", "first"),
    )
    budget = int(trace_df["n_local_evals"].max())
    n5["N_5pct_test"] = n5["n5pct_test"].fillna(budget)
    n5["reached"] = n5["n5pct_test"].notna()
    n5["le3"] = n5["n5pct_test"].notna() & (n5["n5pct_test"] <= 3)
    n5["le5"] = n5["n5pct_test"].notna() & (n5["n5pct_test"] <= 5)
    n5_path = processed_dir / "maturity_n5pct_test.csv"
    n5.to_csv(n5_path, index=False)

    maturities = [int(m) for m in cfg["maturity_levels"]]
    summary_rows = []
    ind = n5[n5["method"] == "independent"]
    ind_mean = float(ind["N_5pct_test"].mean())
    for maturity in maturities:
        g = n5[(n5["method"] == "global") & (n5["maturity"] == maturity)]
        paired, stat = _paired_delta(n5, maturity)
        summary_rows.append(
            {
                "maturity": maturity,
                "n_hist_total": maturity * (int(fleet_cfg["clients_per_regime"]) * 3 - 1),
                "mean_N5_global": float(g["N_5pct_test"].mean()),
                "mean_N5_independent": ind_mean,
                "mean_experiments_saved": float(-paired["delta"].mean()),
                "delta_global_minus_independent": float(stat["mean"]),
                "ci95_low": float(stat["bootstrap_ci_95"][0]),
                "ci95_high": float(stat["bootstrap_ci_95"][1]),
                "n_favorable_seeds": int(stat["n_negative"]),
                "n_seeds": int(stat["n_seeds"]),
                "wilcoxon_p": float(stat["wilcoxon_p"]),
                "p_reach_3_global": float(g["le3"].mean()),
                "p_reach_5_global": float(g["le5"].mean()),
                "p_reach_budget_global": float(g["reached"].mean()),
                "p_reach_3_independent": float(ind["le3"].mean()),
                "p_reach_5_independent": float(ind["le5"].mean()),
                "p_reach_budget_independent": float(ind["reached"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary_path = processed_dir / "maturity_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 96)
    print("FLEET MATURITY SENSITIVITY - DENSE/INDEPENDENT REFERENCE")
    print("=" * 96)
    print(
        summary[
            [
                "maturity",
                "n_hist_total",
                "mean_N5_global",
                "mean_N5_independent",
                "mean_experiments_saved",
                "ci95_low",
                "ci95_high",
                "p_reach_3_global",
                "p_reach_budget_global",
            ]
        ].round(3).to_string(index=False)
    )

    write_run_manifest(
        str(out_dir),
        {
            "config": cfg,
            "source_run_tag": run_tag,
            "offline_reference_tag": reference_tag,
            "n_ref_test_per_maneuver": n_ref_test_per_maneuver,
            "budget": budget,
        },
    )
    print(f"\n[maturity-rescore] wrote {scores_path}")
    print(f"[maturity-rescore] wrote {n5_path}")
    print(f"[maturity-rescore] wrote {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/runs/fleet_maturity_sweep_c1cds02.yaml")
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()
    main(args.config, args.n_jobs)
