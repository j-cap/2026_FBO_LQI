"""
Development-seed sweep for diagnostic-triggered adaptive sharing (Part 12,
docs/phase2_diagnostic_findings.md, 2026-09-04) - the method the user specified after
Part 11's prospective stagnation result cleared the bar:

  For n <= c (trigger_cutoff): every client runs ordinary independent BO.
  At n = c, compute D_i(c) (the already-validated stagnation fraction, same formula as
  `phase2_medium_analysis.compute_stagnation`) from the client's OWN trajectory only.
  For n > c: w_{i<-j}(n) = s_ij if D_i(c) >= tau (and j!=i), else 0 (stays independent).
  The decision is made ONCE and stays fixed for the rest of the run.

Runs on the SAME 20 v3 development seeds as the confirmatory campaign
(`phase2_fbo_comparison_campaign_v3_c1cds05`) - explicitly development data, not the
confirmatory set, per the user's development/confirmatory separation. For each seed:
`independent`, `global`, `similarity` (static) run once each; `triggered_similarity`
(the SAME algorithm, `src.optimization.fleet_bo.TRIGGERED_SIMILARITY`) runs once per
candidate threshold in TAU_CANDIDATES, plus once more as a random-count-matched
control (same number of clients activated as under the REFERENCE_TAU stagnation rule,
chosen uniformly at random instead) - a cheap way to check the gain comes from
identifying the RIGHT clients, not just from reducing how much pooling happens.

`recipient_aware` is deliberately NOT run here - repeatedly shown (Parts 2/3/5/10) to
never beat plain `similarity`; retired from the main experiment per the user's request,
kept in the codebase for the historical ablation only.

Every triggered-variant run shares the SAME per-seed joblib cache as `independent`
(built once per seed by `_prepare_seed_context`): since the trigger's PRE-cutoff
weighting is byte-identical to `independent`'s (verified by
`tests/test_fleet_bo.py::test_triggered_similarity_pre_cutoff_matches_independent_even_when_activated`),
and a NEVER-activated client's weighting stays identical to `independent`'s for the
WHOLE run, most of each triggered variant's proposals are exact cache hits against
`independent`'s already-computed evaluations - only genuinely activated clients' post-
cutoff rounds require fresh simulation.

Usage:
    python scripts/phase2_triggered_dev.py --config config/runs/phase2_triggered_dev_c1cds05.yaml
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
from scipy import stats  # noqa: E402

from src.caching import make_cached_evaluate  # noqa: E402
from src.episodes import deterministic_client_seed  # noqa: E402
from src.io_utils import deep_update, load_run_config, load_yaml, write_run_manifest  # noqa: E402
from src.optimization.fleet_bo import TRIGGERED_SIMILARITY, run_fbo_method  # noqa: E402
from src.optimization.regret import normalized_simple_regret, oracle_relative_regret  # noqa: E402

from phase2_fbo_comparison import _prepare_seed_context, _resolve  # noqa: E402
from phase2_medium_analysis import _family_of  # noqa: E402

# tau values, expressed as fractions of (trigger_cutoff - n_init) transitions - see
# module docstring; TAU_CANDIDATES maps a filesystem/column-safe label to the exact
# float threshold. D_i(15) with n_init=8 takes values in {0, 1/7, ..., 1} (7
# transitions) - these three are the "at most two / at most one / zero improvements"
# readings the user specified, not an arbitrary sweep.
TAU_CANDIDATES = {"5_7": 5 / 7, "6_7": 6 / 7, "1": 1.0}
REFERENCE_TAU_LABEL = "6_7"  # the user's prior preference - used to size the random-trigger control
RANDOM_CONTROL_LABEL = "triggered_random_control"
TRIGGER_CUTOFF = 15


def _D_i_from_trace(n_local_evals: list, best_calibration_rmse: list, *, cutoff: int) -> float:
    """Same formula as phase2_medium_analysis.compute_stagnation, applied directly to
    an in-memory ClientTrace instead of round-tripping through a saved parquet -
    needed here because the trigger decision must be made INSIDE the same run that
    produces it, from the independent method's own live result."""
    pairs = sorted(zip(n_local_evals, best_calibration_rmse))
    vals = [v for n, v in pairs if n <= cutoff]
    if len(vals) < 2:
        return float("nan")
    diffs = np.diff(vals)
    return float(np.mean(diffs >= 0))


def _rows_from_result(seed: int, method_label: str, result) -> tuple[list[dict], list[dict]]:
    trace_rows: list[dict] = []
    summary_rows: list[dict] = []
    for cid, trace in result.client_traces.items():
        for k in range(len(trace.n_local_evals)):
            trace_rows.append(
                {
                    "seed": seed, "method": method_label, "client_id": cid,
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
                "seed": seed, "method": method_label, "client_id": cid, "oracle_rmse": trace.oracle_rmse,
                "baseline_rmse": trace.baseline_rmse,
                "final_held_out_rmse": trace.best_held_out_rmse[-1] if trace.best_held_out_rmse else np.nan,
                "n_infeasible": trace.n_infeasible, "n_degrading_proposals": trace.n_degrading_proposals,
            }
        )
    return trace_rows, summary_rows


def _run_one_seed(seed: int, cfg: dict, fleet_cfg: dict, controller_cfg: dict, *, method_kwargs: dict, cache_dir: Path):
    ctx = _prepare_seed_context(
        seed, cfg, fleet_cfg, controller_cfg, cache_dir=cache_dir,
        xi_lower=method_kwargs["xi_lower"], xi_upper=method_kwargs["xi_upper"],
        state_scale=method_kwargs["state_scale"], fixed_R=method_kwargs["fixed_R"],
        eval_kwargs=method_kwargs["eval_kwargs"],
    )
    cached_evaluate = make_cached_evaluate(str(ctx["seed_cache_dir"]))
    run_kwargs = dict(
        cache_evaluate=cached_evaluate, similarity_S=ctx["similarity_S"], baseline_rmse=ctx["baseline_rmse"],
        oracle_rmse=ctx["oracle_rmse"], n_init=cfg["n_init"], n_iters=cfg["n_iters"], n_candidates=cfg["n_candidates"],
        seed=seed, **method_kwargs,
    )

    trace_rows: list[dict] = []
    summary_rows: list[dict] = []
    activation_rows: list[dict] = []

    print(f"[triggered_dev] seed={seed}: running independent/global/similarity ...")
    result_ind = run_fbo_method(ctx["fleet"], ctx["banks"], "independent", **run_kwargs)
    result_glob = run_fbo_method(ctx["fleet"], ctx["banks"], "global", **run_kwargs)
    result_sim = run_fbo_method(ctx["fleet"], ctx["banks"], "similarity", **run_kwargs)
    for label, result in (("independent", result_ind), ("global", result_glob), ("similarity", result_sim)):
        tr, sr = _rows_from_result(seed, label, result)
        trace_rows.extend(tr)
        summary_rows.extend(sr)

    D_i = {
        cid: _D_i_from_trace(trace.n_local_evals, trace.best_calibration_rmse, cutoff=TRIGGER_CUTOFF)
        for cid, trace in result_ind.client_traces.items()
    }
    client_ids = list(D_i)

    reference_activated = None
    for tau_label, tau_value in TAU_CANDIDATES.items():
        activated = {cid: bool(D_i[cid] >= tau_value) for cid in client_ids}
        if tau_label == REFERENCE_TAU_LABEL:
            reference_activated = activated
        method_label = f"triggered_tau_{tau_label}"
        print(f"[triggered_dev] seed={seed}: running {method_label!r} (tau={tau_value:.4f}, "
              f"{sum(activated.values())}/{len(activated)} activated) ...")
        result_trig = run_fbo_method(
            ctx["fleet"], ctx["banks"], TRIGGERED_SIMILARITY, trigger_cutoff=TRIGGER_CUTOFF, activated=activated,
            **run_kwargs,
        )
        tr, sr = _rows_from_result(seed, method_label, result_trig)
        trace_rows.extend(tr)
        summary_rows.extend(sr)
        for cid in client_ids:
            activation_rows.append(
                {"seed": seed, "client_id": cid, "family": _family_of(cid), "tau_label": tau_label,
                 "D_i": D_i[cid], "activated": activated[cid]}
            )

    assert reference_activated is not None
    k = sum(reference_activated.values())
    rng = np.random.default_rng(deterministic_client_seed(seed, "random_trigger_control"))
    random_activated_ids = set(rng.choice(client_ids, size=k, replace=False)) if k > 0 else set()
    random_activated = {cid: (cid in random_activated_ids) for cid in client_ids}
    print(f"[triggered_dev] seed={seed}: running {RANDOM_CONTROL_LABEL!r} ({k}/{len(client_ids)} activated, "
          f"count-matched to tau={REFERENCE_TAU_LABEL}) ...")
    result_rand = run_fbo_method(
        ctx["fleet"], ctx["banks"], TRIGGERED_SIMILARITY, trigger_cutoff=TRIGGER_CUTOFF, activated=random_activated,
        **run_kwargs,
    )
    tr, sr = _rows_from_result(seed, RANDOM_CONTROL_LABEL, result_rand)
    trace_rows.extend(tr)
    summary_rows.extend(sr)
    for cid in client_ids:
        activation_rows.append(
            {"seed": seed, "client_id": cid, "family": _family_of(cid), "tau_label": "random_control",
             "D_i": D_i[cid], "activated": random_activated[cid]}
        )

    print(f"[triggered_dev] seed={seed}: done")
    return trace_rows, summary_rows, activation_rows


def _seed_bootstrap_median_ci(per_seed_values: dict, *, n_boot: int = 10000, seed: int = 0) -> dict:
    vals = np.asarray(list(per_seed_values.values()), dtype=float)
    vals = vals[np.isfinite(vals)]
    rng = np.random.default_rng(seed)
    boot_median = [np.median(rng.choice(vals, size=len(vals), replace=True)) for _ in range(n_boot)]
    ci_lo, ci_hi = np.percentile(boot_median, [2.5, 97.5])
    try:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(vals)
    except ValueError:
        wilcoxon_stat, wilcoxon_p = float("nan"), float("nan")
    return {
        "n_seeds": len(vals), "median": float(np.median(vals)), "bootstrap_ci_95": (float(ci_lo), float(ci_hi)),
        "n_negative": int((vals < 0).sum()), "wilcoxon_p": float(wilcoxon_p),
    }


def _seed_bootstrap_mean_ci(per_seed_values: dict, *, n_boot: int = 10000, seed: int = 0) -> dict:
    """Same bootstrap-over-seeds methodology as `_seed_bootstrap_median_ci`, but on the
    MEAN of the per-seed values rather than the median - Part 18's lesson (Parts 10-15
    used the median; with only a handful of observations/seed and small, discrete-ish
    point differences, the seed-level median collapses to exactly 0 and hides real
    effects the mean picks up). Use this when the design has multiple
    targets/replicates per seed and the effect size may be small - not a universal
    replacement for the median version, which stays the right tool for large,
    discretization-swamping effects."""
    vals = np.asarray(list(per_seed_values.values()), dtype=float)
    vals = vals[np.isfinite(vals)]
    rng = np.random.default_rng(seed)
    boot_mean = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)]
    ci_lo, ci_hi = np.percentile(boot_mean, [2.5, 97.5])
    try:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(vals)
    except ValueError:
        wilcoxon_stat, wilcoxon_p = float("nan"), float("nan")
    return {
        "n_seeds": len(vals), "mean": float(np.mean(vals)), "bootstrap_ci_95": (float(ci_lo), float(ci_hi)),
        "n_negative": int((vals < 0).sum()), "wilcoxon_p": float(wilcoxon_p),
    }


def _post_trigger_aurc_star(trace_df: pd.DataFrame, *, cutoff: int = TRIGGER_CUTOFF) -> pd.DataFrame:
    """A_i^post = sum_{n=cutoff+1}^{final} regret_star(n) - the PRIMARY post-trigger
    statistic the user specified, per (seed, method, client_id)."""
    sub = trace_df[trace_df["n_local_evals"] > cutoff]
    return sub.groupby(["seed", "method", "client_id"])["regret_star"].sum().rename("A_post").reset_index()


def _report_paired(post_df: pd.DataFrame, method_a: str, method_b: str, *, label: str) -> dict:
    piv = post_df[post_df["method"].isin([method_a, method_b])].pivot_table(
        index=["seed", "client_id"], columns="method", values="A_post"
    )
    piv["delta"] = piv[method_a] - piv[method_b]
    per_seed = piv.groupby("seed")["delta"].median().to_dict()
    res = _seed_bootstrap_median_ci(per_seed)
    print(
        f"  {label}: median(Delta_s)={res['median']:.4f}, 95% CI={res['bootstrap_ci_95']}, "
        f"{res['n_negative']}/{res['n_seeds']} seeds favorable (negative), Wilcoxon p={res['wilcoxon_p']:.4f}"
    )
    return res


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
        print(f"[triggered_dev] dispatching {len(cfg['seeds'])} seeds across n_jobs={n_jobs} worker processes")
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
            delayed(_run_one_seed)(seed, cfg, fleet_cfg, controller_cfg, method_kwargs=method_kwargs, cache_dir=cache_dir)
            for seed in cfg["seeds"]
        )

    trace_rows = [row for tr, _, _ in results for row in tr]
    summary_rows = [row for _, sr, _ in results for row in sr]
    activation_rows = [row for _, _, ar in results for row in ar]

    trace_df = pd.DataFrame(trace_rows)
    trace_df["regret"] = normalized_simple_regret(
        trace_df["best_calibration_rmse"], trace_df["oracle_rmse"], trace_df["baseline_rmse"]
    )
    trace_df["regret_star"] = oracle_relative_regret(trace_df["best_calibration_rmse"], trace_df["oracle_rmse"])
    trace_df.to_parquet(processed_dir / "fbo_convergence_traces.parquet", index=False)
    pd.DataFrame(summary_rows).to_parquet(processed_dir / "fbo_client_summary.parquet", index=False)
    activation_df = pd.DataFrame(activation_rows)
    activation_df.to_csv(processed_dir / "trigger_activation.csv", index=False)
    print(f"\n[triggered_dev] wrote {processed_dir / 'fbo_convergence_traces.parquet'} ({len(trace_df)} rows)")
    print(f"[triggered_dev] wrote {processed_dir / 'trigger_activation.csv'} ({len(activation_df)} rows)")

    write_run_manifest(str(out_dir), {"config": cfg, "fleet_config": fleet_cfg, "controller_config": controller_cfg})

    # ---- primary analysis: post-trigger AURC_star, paired seed-level ------------------
    print("\n" + "=" * 90)
    print(f"PRIMARY: post-trigger (n>{TRIGGER_CUTOFF}) paired seed-level AURC_star deltas")
    print("=" * 90)
    post_df = _post_trigger_aurc_star(trace_df)

    all_results = {}
    for tau_label in TAU_CANDIDATES:
        method_label = f"triggered_tau_{tau_label}"
        print(f"\n-- tau={tau_label} ({TAU_CANDIDATES[tau_label]:.4f}) --")
        all_results[f"{method_label}_vs_independent"] = _report_paired(
            post_df, method_label, "independent", label=f"{method_label} - independent"
        )
        all_results[f"{method_label}_vs_similarity"] = _report_paired(
            post_df, method_label, "similarity", label=f"{method_label} - similarity (static)"
        )

    print(f"\n-- random-count-matched control (k matched to tau={REFERENCE_TAU_LABEL}) --")
    all_results["random_vs_independent"] = _report_paired(
        post_df, RANDOM_CONTROL_LABEL, "independent", label=f"{RANDOM_CONTROL_LABEL} - independent"
    )
    all_results[f"triggered_tau_{REFERENCE_TAU_LABEL}_vs_random"] = _report_paired(
        post_df, f"triggered_tau_{REFERENCE_TAU_LABEL}", RANDOM_CONTROL_LABEL,
        label=f"triggered_tau_{REFERENCE_TAU_LABEL} - {RANDOM_CONTROL_LABEL}",
    )

    # ---- subgroup breakdown: triggered vs non-triggered clients, per tau --------------
    print("\n" + "=" * 90)
    print("SUBGROUP: post-trigger Delta (triggered_tau_X - independent), split by activation status")
    print("=" * 90)
    for tau_label in TAU_CANDIDATES:
        method_label = f"triggered_tau_{tau_label}"
        act = activation_df[activation_df["tau_label"] == tau_label][["seed", "client_id", "activated"]]
        piv = post_df[post_df["method"].isin([method_label, "independent"])].pivot_table(
            index=["seed", "client_id"], columns="method", values="A_post"
        )
        piv["delta"] = piv[method_label] - piv["independent"]
        piv = piv.reset_index().merge(act, on=["seed", "client_id"])
        for activated_flag, name in ((True, "triggered clients"), (False, "non-triggered clients")):
            sub = piv[piv["activated"] == activated_flag]
            if sub.empty:
                print(f"  tau={tau_label}, {name}: (no clients in this subgroup)")
                continue
            per_seed = sub.groupby("seed")["delta"].median().to_dict()
            res = _seed_bootstrap_median_ci(per_seed)
            print(
                f"  tau={tau_label}, {name} (n={len(sub)}): median(Delta_s)={res['median']:.4f}, "
                f"95% CI={res['bootstrap_ci_95']}, {res['n_negative']}/{res['n_seeds']} seeds favorable"
            )
        # sanity check: non-triggered clients must be EXACTLY 0 delta (byte-identical to
        # independent - see tests/test_fleet_bo.py::test_triggered_similarity_never_activated_matches_pure_independent).
        nontrig = piv[piv["activated"] == False]  # noqa: E712
        if not nontrig.empty:
            max_abs_delta = float(nontrig["delta"].abs().max())
            print(f"  tau={tau_label}: max |delta| among non-triggered clients = {max_abs_delta:.2e} (should be ~0)")

    processed_dir_results = processed_dir / "triggered_dev_stats.csv"
    pd.DataFrame(
        [{"comparison": k, **v} for k, v in all_results.items()]
    ).to_csv(processed_dir_results, index=False)
    print(f"\n[triggered_dev] wrote {processed_dir_results}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/runs/phase2_triggered_dev_c1cds05.yaml")
    args = parser.parse_args()
    main(args.config)
