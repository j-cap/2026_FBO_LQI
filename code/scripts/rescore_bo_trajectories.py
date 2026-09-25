"""
Rescores the confirmatory campaign's (`phase2_warmstart_confirmatory_c1cds02`, Part 21)
already-completed BO incumbent trajectories against the dense/independent offline
reference (`scripts/generate_offline_reference.py`) - no BO rerun, since every
incumbent's `xi` was already persisted in `target_convergence_traces.parquet`. This is
the Stage-3-consistent extension the offline-reference design calls for: BO itself still
only ever sees the SAME 6-episode calibration bank it always used (nothing about the
optimization changes), but the primary metric (`N_{i,5%}`) is now computed by scoring
each round's incumbent on the SAME independent reference-test bank used to build
J_i^ref, rather than comparing a calibration-noise-observed value to it - closing the
"BO's own optimization signal vs. the offline reference use different noise" gap a
reviewer could otherwise point at. Also naturally gives every incumbent xi a properly
independent (never-BO-queried) test score, i.e. exactly the codebase's existing
`best_held_out_rmse` pattern (`run_target_warmstart`, `src/optimization/fleet_bo.py`),
just against the larger/more-independent bank and the better reference.

Requires `results/offline_reference/processed/dense_reference.csv` to exist (run
`generate_offline_reference.py --mode full` first) - reads the reference-test-bank
parameters (`n_ref_test_per_maneuver`, base seed) from THAT run's `run_manifest.yaml`
rather than hardcoding them again, so the two scripts cannot silently drift apart.

Usage:
    python scripts/rescore_bo_trajectories.py --config config/runs/phase2_warmstart_confirmatory_c1cds02.yaml
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
    seed, seed_group: pd.DataFrame, cfg: dict, fleet_cfg: dict, controller_cfg: dict, *, xi_lower, xi_upper,
    state_scale, fixed_R, eval_kwargs: dict, cache_dir: Path, ref_lookup: dict, n_ref_test_per_maneuver: int,
) -> list[dict]:
    print(f"[rescore] seed={seed}: preparing context ...")
    ctx = _prepare_seed_context(
        int(seed), cfg, fleet_cfg, controller_cfg, cache_dir=cache_dir, xi_lower=xi_lower, xi_upper=xi_upper,
        state_scale=state_scale, fixed_R=fixed_R, eval_kwargs=eval_kwargs,
    )
    fleet, banks = ctx["fleet"], ctx["banks"]
    client_idx = {c.client_id: i for i, c in enumerate(fleet)}
    cached_evaluate = make_cached_evaluate(str(ctx["seed_cache_dir"]))

    rows: list[dict] = []
    for client_id, client_group in seed_group.groupby("client_id"):
        target = fleet[client_idx[client_id]]
        ref_test_bank = build_multi_maneuver_episode_bank(
            target, Ts=fleet_cfg["Ts"], T_total=controller_cfg["T_total"], T0=controller_cfg["T0"],
            r_max=controller_cfg["r_max"], lane_change_time=fleet_cfg["lane_change_time"],
            n_calibration_per_maneuver=0, n_test_per_maneuver=n_ref_test_per_maneuver,
            base_seed=REFERENCE_TEST_BASE_SEED, tfilter=controller_cfg.get("tfilter"),
        )
        ref_test_episodes = test_episodes(ref_test_bank)
        J_ref = ref_lookup[(seed, client_id)]

        for method, method_group in client_group.groupby("method"):
            method_group = method_group.sort_values("n_local_evals")
            best_test_so_far = np.inf
            n5pct_test = None
            group_rows: list[dict] = []
            for row in method_group.itertuples():
                xi = np.array([row.xi_1, row.xi_2, row.xi_3], float)
                ev = cached_evaluate(target, xi, ref_test_bank, state_scale=state_scale, fixed_R=fixed_R,
                                      episodes=ref_test_episodes, **eval_kwargs)
                j_incumbent_test = (
                    float(ev.tracking_rmse)
                    if (ev.feasible and ev.tracking_rmse is not None and ev.tracking_rmse > 0)
                    else np.nan
                )
                if np.isfinite(j_incumbent_test):
                    best_test_so_far = min(best_test_so_far, j_incumbent_test)
                if n5pct_test is None and np.isfinite(best_test_so_far) and best_test_so_far <= 1.05 * J_ref:
                    n5pct_test = int(row.n_local_evals)
                group_rows.append({
                    "seed": seed, "client_id": client_id, "family": row.family, "method": method,
                    "n_local_evals": row.n_local_evals, "J_incumbent_test": j_incumbent_test,
                    "best_test_so_far": best_test_so_far if np.isfinite(best_test_so_far) else np.nan,
                    "J_ref_test": J_ref,
                })
            for r in group_rows:
                r["n5pct_test"] = n5pct_test
            rows.extend(group_rows)
    print(f"[rescore] seed={seed}: done ({len(rows)} rows)")
    return rows


def main(config_path: str, campaign_tag: str, n_jobs: int) -> None:
    cfg = load_run_config(config_path)
    fleet_cfg = deep_update(load_yaml(_resolve(cfg["fleet_config"])), cfg.get("fleet_overrides", {}))
    controller_cfg = deep_update(load_yaml(_resolve(cfg["controller_config"])), cfg.get("controller_overrides", {}))

    beta_max = np.deg2rad(controller_cfg["beta_max_deg"])
    xi_lower = np.asarray(controller_cfg["xi_lower"], float)
    xi_upper = np.asarray(controller_cfg["xi_upper"], float)
    state_scale, fixed_R = controller_cfg["state_scale"], controller_cfg["fixed_R"]
    eval_kwargs = dict(
        u_min=controller_cfg["u_min"], u_max=controller_cfg["u_max"], u_rate_max=controller_cfg.get("u_rate_max"),
        beta_max=beta_max, noise_std=controller_cfg["noise_std"], process_noise=controller_cfg["process_noise"],
        plant_mode=controller_cfg.get("plant_mode", "linear"), mu=controller_cfg.get("tire_mu", 1.0),
    )

    ref_dir = _ROOT / "results" / "offline_reference"
    ref_manifest = load_yaml(str(ref_dir / "run_manifest.yaml"))["resolved_config"]
    n_ref_test_per_maneuver = ref_manifest["n_ref_test_per_maneuver"]
    print(f"[rescore] reference-test bank: n_test_per_maneuver={n_ref_test_per_maneuver}, "
          f"base_seed={REFERENCE_TEST_BASE_SEED} (from {ref_dir / 'run_manifest.yaml'})")

    ref_df = pd.read_csv(ref_dir / "processed" / "dense_reference.csv")
    ref_df = ref_df[ref_df["reference_feasible"]].copy()
    ref_lookup = {(r.seed, r.client_id): r.J_ref_test for r in ref_df.itertuples()}
    print(f"[rescore] loaded {len(ref_lookup)} feasible (seed, client_id) references")

    campaign_dir = _ROOT / "results" / campaign_tag
    trace_df = pd.read_parquet(campaign_dir / "processed" / "target_convergence_traces.parquet")
    trace_df = trace_df[trace_df.apply(lambda r: (r["seed"], r["client_id"]) in ref_lookup, axis=1)]
    print(f"[rescore] {len(trace_df)} incumbent-trajectory rows to rescore "
          f"(seeds/targets without a feasible dense reference are dropped)")

    out_dir = _ROOT / "results" / "reference_robustness"
    (out_dir / "processed").mkdir(parents=True, exist_ok=True)
    cache_dir = ref_dir / "cache"  # reuse the SAME cache as generate_offline_reference.py's Stage 3

    seed_groups = list(trace_df.groupby("seed"))
    common_kwargs = dict(
        xi_lower=xi_lower, xi_upper=xi_upper, state_scale=state_scale, fixed_R=fixed_R, eval_kwargs=eval_kwargs,
        cache_dir=cache_dir, ref_lookup=ref_lookup, n_ref_test_per_maneuver=n_ref_test_per_maneuver,
    )
    if n_jobs == 1:
        results = [_rescore_one_seed(seed, g, cfg, fleet_cfg, controller_cfg, **common_kwargs) for seed, g in seed_groups]
    else:
        print(f"[rescore] dispatching {len(seed_groups)} seeds across n_jobs={n_jobs} worker processes")
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
            delayed(_rescore_one_seed)(seed, g, cfg, fleet_cfg, controller_cfg, **common_kwargs) for seed, g in seed_groups
        )
    rows = [row for seed_rows in results for row in seed_rows]

    scores_df = pd.DataFrame(rows)
    scores_path = out_dir / "processed" / "candidate_test_scores.csv"
    scores_df.to_csv(scores_path, index=False)
    print(f"\n[rescore] wrote {scores_path} ({len(scores_df)} rows)")

    n5_df = scores_df.drop_duplicates(subset=["seed", "client_id", "method"])[
        ["seed", "client_id", "family", "method", "n5pct_test", "J_ref_test"]
    ].copy()
    budget = int(trace_df["n_local_evals"].max())
    n5_df["N_5pct_test"] = n5_df["n5pct_test"].fillna(budget)
    n5_df["reached"] = n5_df["n5pct_test"].notna()
    n5_path = out_dir / "processed" / "n5pct_test.csv"
    n5_df.to_csv(n5_path, index=False)
    print(f"[rescore] wrote {n5_path} ({len(n5_df)} rows)")

    write_run_manifest(str(out_dir), {"config": cfg, "campaign_tag": campaign_tag, "n_ref_test_per_maneuver": n_ref_test_per_maneuver})

    print("\n" + "=" * 90)
    print("N_5pct_test (vs. dense/independent reference) - compare against Part 21's original N_5pct")
    print("=" * 90)
    print(n5_df.groupby("method")["N_5pct_test"].agg(["mean", "median", "std"]).round(3))
    print("\nfraction reaching 5% (of the NEW reference) within budget:")
    print(n5_df.groupby("method")["reached"].mean().round(3))

    print("\n-- paired seed-level N_5pct_test deltas (negative = method_a needs FEWER evaluations) --")
    for a, b in [("global", "independent"), ("similarity", "independent"), ("similarity", "global")]:
        piv = n5_df[n5_df["method"].isin([a, b])].pivot_table(index=["seed", "client_id"], columns="method",
                                                                values="N_5pct_test")
        piv = piv.dropna()
        piv["delta"] = piv[a] - piv[b]
        per_seed_mean = piv.groupby("seed")["delta"].mean().to_dict()
        res = _seed_bootstrap_mean_ci(per_seed_mean)
        print(f"  {a} - {b}: mean(Delta_s)={res['mean']:.3f}, 95% CI={res['bootstrap_ci_95']}, "
              f"{res['n_negative']}/{res['n_seeds']} seeds favorable, Wilcoxon p={res['wilcoxon_p']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/runs/phase2_warmstart_confirmatory_c1cds02.yaml")
    parser.add_argument("--campaign-tag", default="phase2_warmstart_confirmatory_c1cds02",
                         help="results/<tag> directory holding target_convergence_traces.parquet to rescore.")
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    main(args.config, args.campaign_tag, args.n_jobs)
