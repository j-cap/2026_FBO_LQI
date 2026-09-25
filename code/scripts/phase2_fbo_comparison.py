"""
Phase 2: the actual paper experiment (Next-steps item 4, refined by Phase 1D). Runs
four calibration methods on the SAME fixed-speed/plant-property fleet, all built on one
weighted-GP-EI loop (`src.optimization.fleet_bo`), differing only in what weight a
client's surrogate gives another client's pooled evaluation:

  - independent:      no sharing - each client's own Sobol init + BO loop only
  - global:            fully pooled, unweighted (share everything, ignore heterogeneity)
  - similarity:        pooled, weighted by symmetric dynamics similarity s_ij only
  - recipient_aware:   pooled, weighted by s_ij * g_i(xi) (Phase 1D's directional gate)

Primary decision metrics (user correction, 2026-08-29 - see
docs/phase1_diagnostic_findings.md Part 6): (1) normalized simple regret
r_i(n) = (J_i_best(n) - J_i^star) / (J_i^base - J_i^star) vs. local evaluations n,
oracle/baseline used ONLY here, post-hoc, never inside the BO loop
(`src.optimization.fleet_bo`/`regret`); (2) evaluations to within 5% of the oracle;
(3) infeasible-evaluation count. Held-out (test-episode) RMSE vs. local evaluations is
reported alongside regret as the physical-units version of the same curve. Separation
at n=10/15/20 matters more than separation at final convergence - if every method
eventually converges, the paper's contribution is sample efficiency, not final quality.
"Degrading proposal count" (a pooled-data method's feasible proposal that landed worse
than the client's own pre-round incumbent) is reported as a DIAGNOSTIC only, not a
decision metric - BO intentionally explores worse-than-incumbent points, so this alone
isn't proof of harmful transfer.

Usage:
    python scripts/phase2_fbo_comparison.py --config config/runs/phase2_fbo_comparison_smoke.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from src.caching import make_cached_evaluate  # noqa: E402
from src.dynamics_distance import pairwise_uncertainty_aware_distance  # noqa: E402
from src.episodes import (  # noqa: E402
    EpisodeBank,
    build_episode_bank,
    build_multi_maneuver_episode_bank,
    identification_episode_seeds,
)
from src.fleet_families import (  # noqa: E402
    GENERATION_VERSION_V2,
    GENERATION_VERSION_V3,
    generate_fixed_speed_fleet,
    generate_fixed_speed_fleet_v2_baseline_feasible,
    generate_fixed_speed_fleet_v3_common_task,
)
from src.ifac_bridge import pack_theta_AB  # noqa: E402
from src.interfaces.identifier import fit as identifier_fit  # noqa: E402
from src.io_utils import deep_update, load_run_config, load_yaml, require_keys, set_global_seed, write_run_manifest  # noqa: E402
from src.optimization.decision import decide_fbo_verdict, evals_to_threshold_mean  # noqa: E402
from src.optimization.fleet_bo import METHODS, compute_baseline_reference, compute_oracle_reference, run_fbo_method  # noqa: E402
from src.optimization.regret import normalized_simple_regret, oracle_relative_regret  # noqa: E402
from src.similarity import median_distance_lengthscale, similarity_matrix  # noqa: E402

REGRET_CHECKPOINTS = (10, 15, 20)

plt.rcParams.update(
    {"font.family": "serif", "font.size": 9, "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3}
)
_METHOD_COLORS = {
    "independent": "#999999",
    "global": "#D55E00",
    "similarity": "#0072B2",
    "recipient_aware": "#009E73",
}


def _resolve(rel_path: str) -> str:
    p = Path(rel_path)
    return str(p if p.is_absolute() else _ROOT / p)


def build_fleet_and_identify(
    fleet_cfg, controller_cfg, cfg, *, eval_kwargs: dict = None, state_scale=None, fixed_R: float = None,
    provenance_out: list = None,
):
    """`fleet_cfg["generation_version"]` (default "v1") selects the fleet generator -
    "v1" is `generate_fixed_speed_fleet`, unchanged since Phase 1; `GENERATION_VERSION_V2`
    ("fixed_speed_v2_baseline_feasible", Step C, docs/phase2_diagnostic_findings.md,
    2026-09-02) additionally requires the hand-tuned reference `baseline_xi` to be
    feasible for every accepted client, resampling otherwise; `GENERATION_VERSION_V3`
    ("fixed_speed_v3_common_task", Part 8, 2026-09-03) builds on v2's admission
    criterion but ALSO evaluates every client against ALL `lane_change_time` maneuvers
    (`build_multi_maneuver_episode_bank`) instead of one slot-assigned T_lc, removing
    the T_lc-as-hidden-task-confound Part 8 found. `cfg["n_calibration_episodes"]`/
    `cfg["n_test_episodes"]` mean PER MANEUVER under v3 (so the existing value of 3
    yields 6 total calibration episodes/client, per the frozen "double the episodes"
    choice) - the same total under v1/v2, where there's only one maneuver per client.
    Needs `eval_kwargs`/`state_scale`/`fixed_R` for v2/v3 (unused by v1). `provenance_out`,
    if given, is extended with v2/v3's per-attempt provenance records (a no-op for v1)."""
    generation_version = fleet_cfg.get("generation_version", "v1")
    if generation_version == "v1":
        fleet = generate_fixed_speed_fleet(
            Ts=fleet_cfg["Ts"], n_per_family=fleet_cfg["clients_per_regime"], variability=fleet_cfg["variability"],
            seed=fleet_cfg["seed"], delta=fleet_cfg["delta"], lambda_f=fleet_cfg["lambda_f"],
            lane_change_time=fleet_cfg["lane_change_time"],
        )
    elif generation_version == GENERATION_VERSION_V2:
        if eval_kwargs is None or state_scale is None or fixed_R is None:
            raise ValueError(f"generation_version={generation_version!r} requires eval_kwargs/state_scale/fixed_R")
        fleet, provenance_df = generate_fixed_speed_fleet_v2_baseline_feasible(
            Ts=fleet_cfg["Ts"], n_per_family=fleet_cfg["clients_per_regime"], variability=fleet_cfg["variability"],
            seed=fleet_cfg["seed"], delta=fleet_cfg["delta"], lambda_f=fleet_cfg["lambda_f"],
            lane_change_time=fleet_cfg["lane_change_time"], baseline_xi=controller_cfg["hand_tuned_reference_xi"],
            T_total=controller_cfg["T_total"], T0=controller_cfg["T0"], r_max=controller_cfg["r_max"],
            n_calibration_episodes=cfg["n_calibration_episodes"], n_test_episodes=cfg["n_test_episodes"],
            episode_base_seed=cfg.get("episode_base_seed", 5000), tfilter=controller_cfg.get("tfilter"),
            state_scale=state_scale, fixed_R=fixed_R, eval_kwargs=eval_kwargs,
            max_attempts=fleet_cfg.get("max_generation_attempts", 50),
        )
        if provenance_out is not None:
            provenance_out.extend(provenance_df.assign(fleet_seed=fleet_cfg["seed"]).to_dict("records"))
    elif generation_version == GENERATION_VERSION_V3:
        if eval_kwargs is None or state_scale is None or fixed_R is None:
            raise ValueError(f"generation_version={generation_version!r} requires eval_kwargs/state_scale/fixed_R")
        fleet, provenance_df = generate_fixed_speed_fleet_v3_common_task(
            Ts=fleet_cfg["Ts"], n_per_family=fleet_cfg["clients_per_regime"], variability=fleet_cfg["variability"],
            seed=fleet_cfg["seed"], delta=fleet_cfg["delta"], lambda_f=fleet_cfg["lambda_f"],
            lane_change_time=fleet_cfg["lane_change_time"], baseline_xi=controller_cfg["hand_tuned_reference_xi"],
            T_total=controller_cfg["T_total"], T0=controller_cfg["T0"], r_max=controller_cfg["r_max"],
            n_calibration_episodes_per_maneuver=cfg["n_calibration_episodes"],
            n_test_episodes_per_maneuver=cfg["n_test_episodes"],
            episode_base_seed=cfg.get("episode_base_seed", 5000), tfilter=controller_cfg.get("tfilter"),
            state_scale=state_scale, fixed_R=fixed_R, eval_kwargs=eval_kwargs,
            max_attempts=fleet_cfg.get("max_generation_attempts", 50),
        )
        if provenance_out is not None:
            provenance_out.extend(provenance_df.assign(fleet_seed=fleet_cfg["seed"]).to_dict("records"))
    else:
        raise ValueError(f"Unknown fleet_cfg['generation_version']: {generation_version!r}")

    if generation_version == GENERATION_VERSION_V3:
        banks: dict[str, EpisodeBank] = {
            c.client_id: build_multi_maneuver_episode_bank(
                c, Ts=fleet_cfg["Ts"], T_total=controller_cfg["T_total"], T0=controller_cfg["T0"],
                r_max=controller_cfg["r_max"], lane_change_time=fleet_cfg["lane_change_time"],
                n_calibration_per_maneuver=cfg["n_calibration_episodes"],
                n_test_per_maneuver=cfg["n_test_episodes"], base_seed=cfg.get("episode_base_seed", 5000),
                tfilter=controller_cfg.get("tfilter"),
            )
            for c in fleet
        }
    else:
        banks = {
            c.client_id: build_episode_bank(
                c, Ts=fleet_cfg["Ts"], T_total=controller_cfg["T_total"], T0=controller_cfg["T0"],
                r_max=controller_cfg["r_max"], n_calibration=cfg["n_calibration_episodes"],
                n_test=cfg["n_test_episodes"], base_seed=cfg.get("episode_base_seed", 5000),
                tfilter=controller_cfg.get("tfilter"),
            )
            for c in fleet
        }
    plant_mode = controller_cfg.get("plant_mode", "linear")
    tire_mu = controller_cfg.get("tire_mu", 1.0)
    id_seed_offset = cfg.get("identification_seed_offset", 1000)
    n_id_episodes = cfg.get("n_identification_episodes", 1)
    for client in fleet:
        bank = banks[client.client_id]
        id_seeds = identification_episode_seeds(client.client_id, n_id_episodes, id_seed_offset)
        identifier_fit(
            client, bank.r_ref, seeds=id_seeds, noise_std=controller_cfg["noise_std"],
            process_noise=controller_cfg["process_noise"], u_min=controller_cfg["u_min"],
            u_max=controller_cfg["u_max"], plant_mode=plant_mode, mu=tire_mu, Ts=fleet_cfg["Ts"],
        )
    return fleet, banks


def _prepare_seed_context(
    seed: int, cfg: dict, fleet_cfg: dict, controller_cfg: dict, *, xi_lower, xi_upper, state_scale, fixed_R,
    eval_kwargs: dict, cache_dir: Path,
) -> dict:
    """Builds everything one BO seed needs ONCE - fleet identification, similarity
    matrix, baseline/oracle references - independent of `method`, so it's shared across
    all 4 methods rather than recomputed per (seed, method) job. Returns a plain dict
    (fleet/banks are plain dataclasses of numpy arrays, already proven to survive a
    loky worker boundary the same way `phase1_oracle_landscapes.py` passes `client`
    objects INTO workers) rather than bundling a `cached_evaluate` closure - each
    consumer builds its own from `seed_cache_dir` inside whichever process runs it.

    Each seed gets its OWN cache subdirectory (`cache_dir/seed_<seed>/`), not a shared
    one: a prior version of this script pointed every seed at the same `cache_dir`, and
    `src/caching.py`'s cache key is (client_id, rounded_xi, episode_seeds,
    plant_version, kwargs) - none of which encode the BO seed, even though
    `fleet_cfg["seed"]` (perturbed per-seed just below) drives the client's TRUE plant
    parameters (`generate_fixed_speed_fleet`). Sharing one cache dir across seeds
    therefore let a later seed silently reuse an earlier seed's cached RMSE for the
    same client_id/xi - confirmed to have actually happened for
    `compute_baseline_reference`'s fixed `hand_tuned_reference_xi` query, which is
    identical across all seeds and was hitting the seed=42 cache entry for every other
    seed in the completed medium-scale run (`baseline_rmse` was byte-identical across
    all 5 seeds per client_id, which it must NOT be). Per-seed cache directories fix
    this and, as a side effect, make seeds safe to run in separate processes; methods
    within one seed still share a cache subdirectory, matching the concurrent-write
    pattern `phase1_oracle_landscapes.py` already uses across clients.
    """
    set_global_seed(seed)
    print(f"\n[phase2] === seed={seed}: preparing context ===")
    fleet_cfg_seed = deep_update(fleet_cfg, {"seed": fleet_cfg["seed"] + seed})
    fleet_generation_provenance: list = []
    fleet, banks = build_fleet_and_identify(
        fleet_cfg_seed, controller_cfg, cfg, eval_kwargs=eval_kwargs, state_scale=state_scale, fixed_R=fixed_R,
        provenance_out=fleet_generation_provenance,
    )
    print(f"[phase2] seed={seed}: fleet {len(fleet)} clients, identification complete")

    thetas = [pack_theta_AB(c.Ad_hat, c.Bd_hat) for c in fleet]
    precisions = [c.W_raw for c in fleet]
    D2 = pairwise_uncertainty_aware_distance(thetas, precisions)
    l_d = median_distance_lengthscale(D2)
    S = similarity_matrix(D2, l_d)
    print(f"[phase2] seed={seed}: median-heuristic lengthscale l_d={l_d:.4f}")

    seed_cache_dir = cache_dir / f"seed_{seed}"
    seed_cache_dir.mkdir(parents=True, exist_ok=True)
    cached_evaluate = make_cached_evaluate(str(seed_cache_dir))
    baseline_feasible: dict = {}
    baseline_rmse = compute_baseline_reference(
        fleet, banks, cached_evaluate, baseline_xi=controller_cfg["hand_tuned_reference_xi"],
        state_scale=state_scale, fixed_R=fixed_R, eval_kwargs=eval_kwargs, feasibility_out=baseline_feasible,
    )
    print(f"[phase2] seed={seed}: baseline (hand-tuned reference) computed")
    oracle_feasibility: dict = {}
    oracle_rmse = compute_oracle_reference(
        fleet, banks, cached_evaluate, xi_lower=xi_lower, xi_upper=xi_upper, state_scale=state_scale,
        fixed_R=fixed_R, eval_kwargs=eval_kwargs, n_oracle_sobol=cfg["n_oracle_sobol"], seed=seed,
        feasibility_out=oracle_feasibility,
    )
    print(f"[phase2] seed={seed}: oracle reference computed ({cfg['n_oracle_sobol']} Sobol points/client)")

    # Step D feasible-support instrumentation (docs/phase2_diagnostic_findings.md,
    # 2026-09-02): reuses the oracle Sobol scan and baseline check above - no extra
    # simulation. One row per client, persisted by main() to
    # results/<run_tag>/processed/feasible_support.csv.
    feasible_support_rows = [
        {
            "seed": seed, "client_id": c.client_id, "family": c.cluster.name.lower(),
            "n_feasible": oracle_feasibility[c.client_id]["n_feasible"],
            "n_sobol": oracle_feasibility[c.client_id]["n_sobol"],
            "phi": oracle_feasibility[c.client_id]["phi"],
            "baseline_feasible": baseline_feasible[c.client_id],
            "baseline_rmse": baseline_rmse[c.client_id], "oracle_rmse": oracle_rmse[c.client_id],
            **{f"param_{k}": v for k, v in vars(c.params).items()},
        }
        for c in fleet
    ]

    return dict(
        seed=seed, fleet=fleet, banks=banks, similarity_S=S, baseline_rmse=baseline_rmse, oracle_rmse=oracle_rmse,
        seed_cache_dir=seed_cache_dir, feasible_support_rows=feasible_support_rows,
        fleet_generation_provenance=fleet_generation_provenance,
    )


def _run_seed_method(
    ctx: dict, method: str, cfg: dict, *, xi_lower, xi_upper, state_scale, fixed_R, beta_max: float,
    eval_kwargs: dict,
) -> tuple[list[dict], list[dict]]:
    """Runs ONE method against an already-prepared seed context. The unit of work
    dispatched when `parallel_granularity: seed_method` flattens parallelism fully
    across (seed, method) pairs; also the inner step `_run_seed_all_methods` loops over
    sequentially under the default `parallel_granularity: seed`."""
    seed = ctx["seed"]
    cached_evaluate = make_cached_evaluate(str(ctx["seed_cache_dir"]))
    print(f"[phase2] seed={seed}: running method={method!r} ...")
    result = run_fbo_method(
        ctx["fleet"], ctx["banks"], method, cache_evaluate=cached_evaluate, xi_lower=xi_lower, xi_upper=xi_upper,
        state_scale=state_scale, fixed_R=fixed_R, beta_max=beta_max, eval_kwargs=eval_kwargs,
        n_init=cfg["n_init"], n_iters=cfg["n_iters"], n_candidates=cfg["n_candidates"],
        similarity_S=ctx["similarity_S"], baseline_rmse=ctx["baseline_rmse"], oracle_rmse=ctx["oracle_rmse"],
        negative_transfer_threshold=cfg.get("negative_transfer_threshold", 0.10), seed=seed,
    )
    trace_rows: list[dict] = []
    summary_rows: list[dict] = []
    for cid, trace in result.client_traces.items():
        for k in range(len(trace.n_local_evals)):
            trace_rows.append(
                {
                    "seed": seed, "method": method, "client_id": cid,
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
                "seed": seed, "method": method, "client_id": cid, "oracle_rmse": trace.oracle_rmse,
                "baseline_rmse": trace.baseline_rmse,
                "final_held_out_rmse": trace.best_held_out_rmse[-1] if trace.best_held_out_rmse else np.nan,
                "n_infeasible": trace.n_infeasible, "n_degrading_proposals": trace.n_degrading_proposals,
                "evals_to_5pct_of_oracle": trace.evals_to_5pct_of_oracle,
            }
        )
    print(f"[phase2] seed={seed}: {method!r} done: {len(result.all_records)} true-plant evaluations")
    return trace_rows, summary_rows


def _run_seed_all_methods(ctx: dict, cfg: dict, **method_kwargs) -> tuple[list[dict], list[dict]]:
    """All 4 methods for one seed context, sequentially - the unit of work dispatched
    when `parallel_granularity: seed` (the default) parallelizes across seeds only."""
    trace_rows: list[dict] = []
    summary_rows: list[dict] = []
    for method in METHODS:
        tr, sr = _run_seed_method(ctx, method, cfg, **method_kwargs)
        trace_rows.extend(tr)
        summary_rows.extend(sr)
    return trace_rows, summary_rows


def main(config_path: str, *, stage1_only: bool = False) -> None:
    cfg = load_run_config(config_path)
    require_keys(
        cfg,
        ["run_tag", "fleet_config", "controller_config", "n_init", "n_iters", "n_candidates",
         "n_calibration_episodes", "n_test_episodes", "n_oracle_sobol", "seeds"],
    )
    fleet_cfg = deep_update(load_yaml(_resolve(cfg["fleet_config"])), cfg.get("fleet_overrides", {}))
    controller_cfg = deep_update(load_yaml(_resolve(cfg["controller_config"])), cfg.get("controller_overrides", {}))

    run_tag = cfg["run_tag"]
    out_dir = _ROOT / "results" / run_tag
    processed_dir, figures_dir, cache_dir = out_dir / "processed", out_dir / "figures", out_dir / "cache"
    for d in (processed_dir, figures_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    xi_lower = controller_cfg["xi_lower"]
    xi_upper = controller_cfg["xi_upper"]
    state_scale = controller_cfg["state_scale"]
    fixed_R = controller_cfg["fixed_R"]
    beta_max = np.deg2rad(controller_cfg["beta_max_deg"])
    eval_kwargs = dict(
        u_min=controller_cfg["u_min"], u_max=controller_cfg["u_max"],
        u_rate_max=controller_cfg.get("u_rate_max"), beta_max=beta_max,
        noise_std=controller_cfg["noise_std"], process_noise=controller_cfg["process_noise"],
        plant_mode=controller_cfg.get("plant_mode", "linear"), mu=controller_cfg.get("tire_mu", 1.0),
    )

    n_jobs = cfg.get("n_jobs", 1)
    parallel_granularity = cfg.get("parallel_granularity", "seed")
    if parallel_granularity not in ("seed", "seed_method"):
        raise ValueError(f"parallel_granularity must be 'seed' or 'seed_method', got {parallel_granularity!r}")

    context_kwargs = dict(
        xi_lower=xi_lower, xi_upper=xi_upper, state_scale=state_scale, fixed_R=fixed_R,
        eval_kwargs=eval_kwargs, cache_dir=cache_dir,
    )
    method_kwargs = dict(
        xi_lower=xi_lower, xi_upper=xi_upper, state_scale=state_scale, fixed_R=fixed_R, beta_max=beta_max,
        eval_kwargs=eval_kwargs,
    )

    # ---- stage 1: per-seed context (fleet identification + baseline/oracle refs) -----
    if n_jobs == 1:
        contexts = [
            _prepare_seed_context(seed, cfg, fleet_cfg, controller_cfg, **context_kwargs) for seed in cfg["seeds"]
        ]
    else:
        print(f"[phase2] preparing {len(cfg['seeds'])} seed contexts across n_jobs={n_jobs} worker processes")
        contexts = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
            delayed(_prepare_seed_context)(seed, cfg, fleet_cfg, controller_cfg, **context_kwargs)
            for seed in cfg["seeds"]
        )

    # Step D feasible-support instrumentation - written as soon as stage 1 finishes,
    # independent of the (possibly much longer) BO stage below.
    feasible_support_df = pd.DataFrame(
        [row for ctx in contexts for row in ctx["feasible_support_rows"]]
    )
    feasible_support_df.to_csv(processed_dir / "feasible_support.csv", index=False)
    print(f"[phase2] wrote {processed_dir / 'feasible_support.csv'} ({len(feasible_support_df)} rows)")

    # Step C provenance (only non-empty for fleet_cfg["generation_version"] ==
    # GENERATION_VERSION_V2 - a no-op file for v1 runs).
    generation_provenance_rows = [row for ctx in contexts for row in ctx["fleet_generation_provenance"]]
    if generation_provenance_rows:
        pd.DataFrame(generation_provenance_rows).to_csv(
            processed_dir / "fleet_generation_provenance.csv", index=False
        )
        print(f"[phase2] wrote {processed_dir / 'fleet_generation_provenance.csv'} ({len(generation_provenance_rows)} rows)")

    if stage1_only:
        print("[phase2] --stage1-only: skipping the BO stage, stopping here.")
        return

    # ---- stage 2: the 4 methods/seed - one job per seed, or fully flattened ----------
    if parallel_granularity == "seed_method":
        jobs = [(ctx, method) for ctx in contexts for method in METHODS]
        if n_jobs == 1:
            seed_results = [_run_seed_method(ctx, method, cfg, **method_kwargs) for ctx, method in jobs]
        else:
            print(f"[phase2] dispatching {len(jobs)} (seed, method) jobs across n_jobs={n_jobs} worker processes")
            seed_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
                delayed(_run_seed_method)(ctx, method, cfg, **method_kwargs) for ctx, method in jobs
            )
    else:
        if n_jobs == 1:
            seed_results = [_run_seed_all_methods(ctx, cfg, **method_kwargs) for ctx in contexts]
        else:
            print(f"[phase2] dispatching {len(contexts)} seeds across n_jobs={n_jobs} worker processes")
            seed_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
                delayed(_run_seed_all_methods)(ctx, cfg, **method_kwargs) for ctx in contexts
            )

    trace_rows = [row for seed_trace_rows, _ in seed_results for row in seed_trace_rows]
    summary_rows = [row for _, seed_summary_rows in seed_results for row in seed_summary_rows]

    trace_df = pd.DataFrame(trace_rows)
    summary_df = pd.DataFrame(summary_rows)
    # Normalized simple regret (post-hoc analysis ONLY - oracle/baseline never touched
    # the BO loop itself, see src/optimization/{fleet_bo,regret}.py). `regret` (baseline-
    # normalized, Parts 2-7's metric) stays the secondary continuity metric; `regret_star`
    # (oracle-relative, Part 8's recommendation, 2026-09-03) is the new PRIMARY metric for
    # any run using GENERATION_VERSION_V3 - doesn't collapse toward a near-zero
    # denominator when a client's baseline happens to sit close to its oracle (Part 7's
    # validation-rerun outlier audit confirmed that failure mode survives the Step C fix).
    trace_df["regret"] = normalized_simple_regret(
        trace_df["best_calibration_rmse"], trace_df["oracle_rmse"], trace_df["baseline_rmse"]
    )
    trace_df["regret_star"] = oracle_relative_regret(trace_df["best_calibration_rmse"], trace_df["oracle_rmse"])
    trace_df.to_parquet(processed_dir / "fbo_convergence_traces.parquet", index=False)
    summary_df.to_parquet(processed_dir / "fbo_client_summary.parquet", index=False)

    # ---- Figure 1: median + IQR held-out RMSE vs. local evaluations, per method ------
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for method in METHODS:
        sub = trace_df[trace_df["method"] == method]
        if sub.empty:
            continue
        agg = sub.groupby("n_local_evals")["best_held_out_rmse"].agg(
            median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75)
        )
        ax.plot(agg.index, agg["median"], label=method, color=_METHOD_COLORS.get(method), linewidth=1.8)
        ax.fill_between(agg.index, agg["q25"], agg["q75"], color=_METHOD_COLORS.get(method), alpha=0.15)
    ax.set_xlabel("local (true-plant) evaluations")
    ax.set_ylabel("best held-out tracking RMSE (rad/s)")
    ax.set_title(f"Phase 2 - FBO comparison ({run_tag})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "fig2_convergence_curves.png", dpi=150)
    plt.close(fig)

    # ---- Figure 2: median + IQR normalized simple regret vs. local evaluations -------
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for method in METHODS:
        sub = trace_df[trace_df["method"] == method]
        if sub.empty:
            continue
        agg = sub.groupby("n_local_evals")["regret"].agg(
            median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75)
        )
        ax.plot(agg.index, agg["median"], label=method, color=_METHOD_COLORS.get(method), linewidth=1.8)
        ax.fill_between(agg.index, agg["q25"], agg["q75"], color=_METHOD_COLORS.get(method), alpha=0.15)
    ax.axhline(0.0, color="black", linewidth=0.7, linestyle=":")
    ax.set_xlabel("local (true-plant) evaluations")
    ax.set_ylabel(r"normalized simple regret $r_i(n)$")
    ax.set_title(f"Phase 2 - normalized regret ({run_tag})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "fig2b_regret_curves.png", dpi=150)
    plt.close(fig)

    # ---- per-method regret at the decision checkpoints + aggregate secondary metrics --
    # `budget` is the run's TRUE final local-eval count - not necessarily 20 (the
    # original smoke/medium budget REGRET_CHECKPOINTS was tuned around). Always add it
    # to the displayed/decision checkpoint set so decide_fbo_verdict's final-checkpoint
    # tolerance check (`final_n`) compares at the actual final budget, not a stale
    # mid-budget snapshot - a no-op for any run whose budget IS 20 (the common case so
    # far), but required for the >20-eval campaign configs (2026-09-01).
    budget = cfg["n_init"] + cfg["n_iters"]
    regret_checkpoints = tuple(sorted(set(REGRET_CHECKPOINTS) | {budget}))
    regret_at_checkpoints = (
        trace_df[trace_df["n_local_evals"].isin(regret_checkpoints)]
        .groupby(["method", "n_local_evals"])["regret"]
        .median()
        .unstack("n_local_evals")
        .reindex(METHODS)
    )
    agg_summary = summary_df.groupby("method").agg(
        final_held_out_rmse_mean=("final_held_out_rmse", "mean"),
        final_held_out_rmse_median=("final_held_out_rmse", "median"),
        n_infeasible_mean=("n_infeasible", "mean"),
        n_degrading_proposals_mean=("n_degrading_proposals", "mean"),
        frac_reached_5pct=("evals_to_5pct_of_oracle", lambda s: float(s.notna().mean())),
        evals_to_5pct_mean=("evals_to_5pct_of_oracle", lambda s: evals_to_threshold_mean(s, budget)),
    ).reindex(METHODS)

    metrics_for_decision = {}
    for method in METHODS:
        m = {f"regret_n{n}": regret_at_checkpoints.loc[method].get(n, float("nan")) for n in regret_checkpoints}
        m["evals_to_5pct_mean"] = agg_summary.loc[method, "evals_to_5pct_mean"]
        m["frac_reached_5pct"] = agg_summary.loc[method, "frac_reached_5pct"]
        m["n_infeasible_mean"] = agg_summary.loc[method, "n_infeasible_mean"]
        metrics_for_decision[method] = m
    # checkpoints stays the original early/mid set (majority-vote "beats" semantics,
    # per this module's docstring: separation at n=10/15/20 matters more than final
    # convergence) - only final_n changes to track the run's true budget.
    verdict, verdict_detail = decide_fbo_verdict(metrics_for_decision, checkpoints=REGRET_CHECKPOINTS, final_n=budget)
    print(f"\n[phase2] verdict: {verdict} ({verdict_detail})")

    # ---- summary markdown --------------------------------------------------------------
    lines = [f"# Phase 2 FBO comparison - {run_tag}", ""]
    lines.append(
        f"Fleet: {fleet_cfg['clients_per_regime']} clients/family x 3 families; seeds={cfg['seeds']}; "
        f"n_init={cfg['n_init']}, n_iters={cfg['n_iters']}, n_candidates={cfg['n_candidates']}; "
        f"plant_mode={controller_cfg.get('plant_mode', 'linear')!r}."
    )
    lines.append("")
    lines.append(
        "## Primary decision metrics: median normalized simple regret at n=10/15/20, "
        "evaluations-to-5%-of-oracle, infeasible count"
    )
    lines.append("")
    lines.append("```")
    lines.append(regret_at_checkpoints.round(4).to_string())
    lines.append("")
    lines.append(
        agg_summary[
            ["frac_reached_5pct", "evals_to_5pct_mean", "n_infeasible_mean", "final_held_out_rmse_mean"]
        ].round(4).to_string()
    )
    lines.append("```")
    lines.append("")
    lines.append(f"## Verdict: `{verdict}`")
    lines.append("")
    lines.append(verdict_detail)
    lines.append("")
    lines.append(
        "## Diagnostic (NOT a decision metric): mean degrading-proposal count per client "
        "(a pooled-data method's feasible proposal that landed worse than the client's own "
        "pre-round incumbent - BO intentionally explores such points, so this alone isn't "
        "proof of harmful transfer)"
    )
    lines.append("")
    lines.append("```")
    lines.append(agg_summary[["n_degrading_proposals_mean"]].round(4).to_string())
    lines.append("```")
    (out_dir / "phase2_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[phase2] wrote summary to {out_dir / 'phase2_summary.md'}")

    write_run_manifest(str(out_dir), {"config": cfg, "fleet_config": fleet_cfg, "controller_config": controller_cfg})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/runs/phase2_fbo_comparison_smoke.yaml")
    parser.add_argument(
        "--stage1-only", action="store_true",
        help="Write feasible_support.csv (Step D, docs/phase2_diagnostic_findings.md) and stop - skips the "
        "expensive BO stage entirely. Re-running an ALREADY-COMPLETED config's stage 1 this way, on the same "
        "machine/local disk as the original run, hits that run's joblib cache (identification/baseline/oracle "
        "evals were already computed) instead of re-simulating - minutes, not hours, to backfill feasible_support.csv "
        "onto a completed campaign's local results folder.",
    )
    args = parser.parse_args()
    main(args.config, stage1_only=args.stage1_only)
