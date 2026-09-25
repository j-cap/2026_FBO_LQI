"""
Phase-1 oracle feasibility study (plan doc §10, §38's "Immediate Next Action"):
Sobol-sample the LQI weight space, evaluate every (client, xi) on the identified-model
controller / true-plant rollout, and report whether identified dynamics predict
controller-performance-landscape similarity - the go/no-go gate for the rest of the
paper. Every optimizer in later phases must reuse `src.candidate_eval.evaluate`, not
reimplement this loop.

Usage:
    python scripts/phase1_oracle_landscapes.py --config config/runs/phase1_oracle_landscape_smoke.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import qmc  # noqa: E402

from src import oracle_analysis  # noqa: E402
from src.caching import make_cached_evaluate  # noqa: E402
from src.dynamics_distance import pairwise_euclidean_distance, pairwise_uncertainty_aware_distance  # noqa: E402
from src.episodes import (  # noqa: E402
    EpisodeBank,
    build_episode_bank,
    identification_episode_seeds,
)
from src.fleet_families import generate_fixed_speed_fleet  # noqa: E402
from src.ifac_bridge import Cluster, generate_fleet, pack_theta_AB  # noqa: E402
from src.interfaces import clusterer as clusterer_iface  # noqa: E402
from src.interfaces.identifier import fit as identifier_fit  # noqa: E402
from src.io_utils import deep_update, load_run_config, load_yaml, require_keys, set_global_seed, write_run_manifest  # noqa: E402
from src.landscape_metrics import compare_landscapes  # noqa: E402
from src.transfer_analysis import (  # noqa: E402
    build_client_selection_table,
    decide_next_step,
    held_out_eval_worker,
    transfer_loss_worker,
)

plt.rcParams.update(
    {"font.family": "serif", "font.size": 9, "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3}
)
# Figure B's "same regime" scatter color - a fixed constant, not indexed by any
# particular fleet's actual regime names (works for cluster_enum and
# fixed_speed_families fleets alike).
_SAME_REGIME_COLOR = "#0072B2"


def _run_client_candidates(client, Xi, xi_indices, bank, cached_evaluate, eval_kwargs):
    rows = []
    for xi_idx, xi in zip(xi_indices, Xi):
        result = cached_evaluate(client, xi, bank, **eval_kwargs)
        rows.append(
            {
                "client_id": client.client_id,
                "true_regime": client.cluster.name,
                "estimated_cluster": str(client.cluster_id_est),
                "xi_idx": int(xi_idx),
                "xi_1": float(xi[0]),
                "xi_2": float(xi[1]),
                "xi_3": float(xi[2]),
                "feasible": result.feasible,
                "failure_reason": result.failure_reason,
                "tracking_rmse": result.tracking_rmse,
                "max_abs_input": result.max_abs_input,
                "input_saturation_fraction": result.input_saturation_fraction,
                "max_abs_input_rate": result.max_abs_input_rate,
                "max_abs_beta": result.max_abs_beta,
                "wall_time_s": result.wall_time_s,
            }
        )
    return rows


def main(config_path: str) -> None:
    cfg = load_run_config(config_path)
    require_keys(cfg, ["run_tag", "fleet_config", "controller_config", "n_sobol", "n_calibration_episodes", "seed"])

    def _resolve(rel_path: str) -> str:
        """Config paths in run configs are written relative to the project root
        (paper_FBO_LQI/), not the invocation cwd - resolve them explicitly so this
        script works regardless of where it's launched from."""
        p = Path(rel_path)
        return str(p if p.is_absolute() else _ROOT / p)

    fleet_cfg = deep_update(load_yaml(_resolve(cfg["fleet_config"])), cfg.get("fleet_overrides", {}))
    controller_cfg = deep_update(
        load_yaml(_resolve(cfg["controller_config"])), cfg.get("controller_overrides", {})
    )

    run_tag = cfg["run_tag"]
    out_dir = _ROOT / "results" / run_tag
    processed_dir = out_dir / "processed"
    figures_dir = out_dir / "figures"
    cache_dir = out_dir / "cache"
    for d in (processed_dir, figures_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    set_global_seed(cfg["seed"])

    # ---- 1) fleet ------------------------------------------------------------------
    # fleet_kind (Next-steps fleet redesign): "cluster_enum" (default) is the original
    # LOW_SPEED/PAYLOAD/HIGH_SPEED (speed-as-heterogeneity) fleet via src2's own
    # generate_fleet; "fixed_speed_families" is the redesigned fleet
    # (src/fleet_families.py) - nominal/payload/tire-degraded, all ~U=15 m/s, no
    # deliberate speed shift between families.
    fleet_kind = fleet_cfg.get("fleet_kind", "cluster_enum")
    if fleet_kind == "cluster_enum":
        avail_clusters = [Cluster[name] for name in fleet_cfg["regimes"]]
        fleet = generate_fleet(
            Ts=fleet_cfg["Ts"],
            avail_clusters=avail_clusters,
            n_per_cluster=fleet_cfg["clients_per_regime"],
            variability=fleet_cfg["variability"],
            seed=fleet_cfg["seed"],
            set_gains=True,
            compute_gains=False,
            delta=fleet_cfg["delta"],
            lambda_f=fleet_cfg["lambda_f"],
            lane_change_time=fleet_cfg["lane_change_time"],
        )
    elif fleet_kind == "fixed_speed_families":
        fleet = generate_fixed_speed_fleet(
            Ts=fleet_cfg["Ts"],
            n_per_family=fleet_cfg["clients_per_regime"],
            variability=fleet_cfg["variability"],
            seed=fleet_cfg["seed"],
            delta=fleet_cfg["delta"],
            lambda_f=fleet_cfg["lambda_f"],
            lane_change_time=fleet_cfg["lane_change_time"],
        )
    else:
        raise ValueError(f"Unknown fleet_kind {fleet_kind!r} (expected 'cluster_enum' or 'fixed_speed_families').")

    regime_names = sorted({c.cluster.name for c in fleet})
    print(f"[phase1] generated {len(fleet)} clients across regimes {regime_names} (fleet_kind={fleet_kind!r})")

    # Correction-plan item 5 / Phase 1B: "linear" (default, Phase 1A) rolls out
    # client.Ad_true/Bd_true; "nonlinear_tanh" instead rolls out client.params through
    # the tanh-saturated tire-force plant (src/nonlinear_plant.py) for BOTH
    # identification and calibration/test evaluation - RLS, clustering, and LQI
    # synthesis are untouched, still identifying/designing against the linear model.
    plant_mode = controller_cfg.get("plant_mode", "linear")
    tire_mu = controller_cfg.get("tire_mu", 1.0)

    # ---- 2) per-client episode banks + identification -------------------------------
    banks: dict[str, EpisodeBank] = {}
    for client in fleet:
        banks[client.client_id] = build_episode_bank(
            client,
            Ts=fleet_cfg["Ts"],
            T_total=controller_cfg["T_total"],
            T0=controller_cfg["T0"],
            r_max=controller_cfg["r_max"],
            n_calibration=cfg["n_calibration_episodes"],
            n_test=cfg.get("n_test_episodes", 0),
            base_seed=cfg.get("episode_base_seed", 5000),
            tfilter=controller_cfg.get("tfilter"),
        )

    id_seed_offset = cfg.get("identification_seed_offset", 1000)
    n_id_episodes = cfg.get("n_identification_episodes", 1)
    for client in fleet:
        bank = banks[client.client_id]
        id_seeds = identification_episode_seeds(client.client_id, n_id_episodes, id_seed_offset)
        identifier_fit(
            client,
            bank.r_ref,
            seeds=id_seeds,
            noise_std=controller_cfg["noise_std"],
            process_noise=controller_cfg["process_noise"],
            u_min=controller_cfg["u_min"],
            u_max=controller_cfg["u_max"],
            plant_mode=plant_mode,
            mu=tire_mu,
            Ts=fleet_cfg["Ts"],
        )
    print(f"[phase1] identification complete ({n_id_episodes} episode(s)/client, plant_mode={plant_mode!r})")

    # ---- 3) uncertainty-aware clustering on the IDENTIFIED models -------------------
    thetas = [pack_theta_AB(c.Ad_hat, c.Bd_hat) for c in fleet]
    precisions = [c.W_raw for c in fleet]
    cluster_result = clusterer_iface.fit(
        thetas, precisions, K=cfg.get("n_clusters", 3), seed=cfg.get("cluster_seed", 1337)
    )
    for client, label in zip(fleet, cluster_result.client_cluster_labels):
        client.cluster_id_est = int(label)
    print(f"[phase1] estimated clusters: {np.bincount(cluster_result.client_cluster_labels)}")

    # ---- 4) common Sobol candidate set Xi -------------------------------------------
    xi_lower = np.asarray(controller_cfg["xi_lower"], float)
    xi_upper = np.asarray(controller_cfg["xi_upper"], float)
    sampler = qmc.Sobol(d=3, scramble=True, seed=cfg["seed"])
    Xi = qmc.scale(sampler.random(cfg["n_sobol"]), xi_lower, xi_upper)
    xi_indices = list(range(cfg["n_sobol"]))
    print(f"[phase1] Sobol candidate set: {len(Xi)} points in xi in [{xi_lower}, {xi_upper}]")

    # ---- 5) evaluate every (client, xi) ---------------------------------------------
    cached_evaluate = make_cached_evaluate(str(cache_dir))
    eval_kwargs = dict(
        state_scale=tuple(controller_cfg["state_scale"]),
        fixed_R=controller_cfg["fixed_R"],
        u_min=controller_cfg["u_min"],
        u_max=controller_cfg["u_max"],
        u_rate_max=controller_cfg.get("u_rate_max"),
        beta_max=(
            np.deg2rad(controller_cfg["beta_max_deg"]) if controller_cfg.get("beta_max_deg") is not None else None
        ),
        noise_std=controller_cfg["noise_std"],
        process_noise=tuple(controller_cfg["process_noise"]),
        plant_mode=plant_mode,
        mu=tire_mu,
    )

    job_results = Parallel(n_jobs=cfg.get("n_jobs", -1), backend="loky", verbose=5)(
        delayed(_run_client_candidates)(client, Xi, xi_indices, banks[client.client_id], cached_evaluate, eval_kwargs)
        for client in fleet
    )
    rows = [row for client_rows in job_results for row in client_rows]
    df = pd.DataFrame(rows)
    n_feasible = int(df["feasible"].sum())
    print(f"[phase1] evaluated {len(df)} (client, xi) pairs, {n_feasible} feasible ({n_feasible / len(df):.1%})")

    # ---- 6) oracle optima -------------------------------------------------------------
    ind = oracle_analysis.individual_optima(df)
    df_g = df.copy()
    df_g["_global"] = "global"
    global_opt = oracle_analysis.grouped_optima(df_g, "_global")
    cluster_opt = oracle_analysis.grouped_optima(df, "estimated_cluster")
    regime_opt = oracle_analysis.grouped_optima(df, "true_regime")

    optima_rows = []
    for _, r in ind.iterrows():
        optima_rows.append({"level": "individual", "group": r["client_id"], **{c: r[c] for c in ("xi_1", "xi_2", "xi_3")}, "objective": r["tracking_rmse"]})
    for _, r in regime_opt.iterrows():
        optima_rows.append({"level": "true_regime", "group": r["true_regime"], **{c: r[c] for c in ("xi_1", "xi_2", "xi_3")}, "objective": r["mean_rmse"]})
    for _, r in cluster_opt.iterrows():
        optima_rows.append({"level": "estimated_cluster", "group": r["estimated_cluster"], **{c: r[c] for c in ("xi_1", "xi_2", "xi_3")}, "objective": r["mean_rmse"]})
    for _, r in global_opt.iterrows():
        optima_rows.append({"level": "global", "group": r["_global"], **{c: r[c] for c in ("xi_1", "xi_2", "xi_3")}, "objective": r["mean_rmse"]})
    oracle_optima_df = pd.DataFrame(optima_rows)

    # ---- 7) pairwise dynamics distance + landscape similarity ------------------------
    client_ids, _xi_pivot, J = oracle_analysis.landscape_matrix(df)
    theta_by_id = {c.client_id: pack_theta_AB(c.Ad_hat, c.Bd_hat) for c in fleet}
    precision_by_id = {c.client_id: c.W_raw for c in fleet}
    regime_by_id = {c.client_id: c.cluster.name for c in fleet}
    thetas_ordered = [theta_by_id[cid] for cid in client_ids]
    precisions_ordered = [precision_by_id[cid] for cid in client_ids]

    D2_unc = pairwise_uncertainty_aware_distance(thetas_ordered, precisions_ordered)
    D_unc = np.sqrt(np.clip(D2_unc, 0.0, None))
    D_euc = pairwise_euclidean_distance(thetas_ordered)

    n = len(client_ids)
    S_1m_spearman = np.zeros((n, n))
    landscape_rows = []
    ind_xi_by_client = ind.set_index("client_id")[["xi_1", "xi_2", "xi_3"]]
    for i in range(n):
        for j in range(i + 1, n):
            xi_i_star = ind_xi_by_client.loc[client_ids[i]].to_numpy() if client_ids[i] in ind_xi_by_client.index else np.full(3, np.nan)
            xi_j_star = ind_xi_by_client.loc[client_ids[j]].to_numpy() if client_ids[j] in ind_xi_by_client.index else np.full(3, np.nan)
            m = compare_landscapes(J[i], J[j], xi_i_star, xi_j_star)
            S_1m_spearman[i, j] = S_1m_spearman[j, i] = 1.0 - m.spearman_r if np.isfinite(m.spearman_r) else np.nan
            landscape_rows.append(
                {
                    "client_i": client_ids[i],
                    "client_j": client_ids[j],
                    "pearson_r": m.pearson_r,
                    "spearman_r": m.spearman_r,
                    "normalized_rmse": m.normalized_rmse,
                    "optimum_distance": m.optimum_distance,
                    "n_common": m.n_common,
                    "same_true_regime": regime_by_id[client_ids[i]] == regime_by_id[client_ids[j]],
                    "d_uncertainty_aware": D_unc[i, j],
                    "d_euclidean": D_euc[i, j],
                }
            )
    pairwise_landscape_df = pd.DataFrame(landscape_rows)
    pairwise_dynamics_df = pairwise_landscape_df[
        ["client_i", "client_j", "d_uncertainty_aware", "d_euclidean", "same_true_regime"]
    ].copy()

    # ---- 8) calibration-based go/no-go report (secondary - see item 8b/8c for the
    #         held-out-based criteria correction-plan items 2/6/7 actually ask for) ------
    report = oracle_analysis.evaluate_go_no_go(
        df, D_unc, S_1m_spearman, xi_lower, xi_upper, n_boot=1000, seed=cfg["seed"]
    )

    # ---- 8b) held-out evaluation (correction-plan item 2): every level's CALIBRATION
    #          selection, re-scored on each client's DISJOINT test episodes ---------------
    selection_table = build_client_selection_table(fleet, ind, cluster_opt, regime_opt, global_opt)
    client_by_id = {c.client_id: c for c in fleet}
    selection_rows_by_client: dict[str, list] = {}
    for _, row in selection_table.iterrows():
        selection_rows_by_client.setdefault(row["client_id"], []).append(row.to_dict())

    held_out_job_results = Parallel(n_jobs=cfg.get("n_jobs", -1), backend="loky", verbose=5)(
        delayed(held_out_eval_worker)(
            client_by_id[cid], banks[cid], rows, eval_kwargs, cached_evaluate
        )
        for cid, rows in selection_rows_by_client.items()
    )
    held_out_df = pd.DataFrame([row for client_rows in held_out_job_results for row in client_rows])
    print(f"[phase1] held-out evaluation complete ({len(held_out_df)} (client, level) pairs)")

    # ---- 8c) transfer-loss matrix (correction-plan item 6, primary RQ1 diagnostic) -----
    individual_xi_by_client = {
        row["client_id"]: row[["xi_1", "xi_2", "xi_3"]].to_numpy(dtype=float) for _, row in ind.iterrows()
    }
    transfer_job_results = Parallel(n_jobs=cfg.get("n_jobs", -1), backend="loky", verbose=5)(
        delayed(transfer_loss_worker)(client, banks[client.client_id], individual_xi_by_client, eval_kwargs, cached_evaluate)
        for client in fleet
        if client.client_id in individual_xi_by_client
    )
    transfer_df = pd.DataFrame([row for client_rows in transfer_job_results for row in client_rows])
    print(f"[phase1] transfer-loss matrix complete ({len(transfer_df)} ordered client pairs)")

    # correlate d_ij^dyn against transfer loss L_{i<-j} (both directions of each pair,
    # since transfer is NOT symmetric: L_{i<-j} != L_{j<-i} in general)
    client_id_to_idx = {cid: i for i, cid in enumerate(client_ids)}
    transfer_df = transfer_df[transfer_df["client_i"] != transfer_df["client_j"]].copy()
    transfer_df["d_dyn"] = transfer_df.apply(
        lambda r: D_unc[client_id_to_idx[r["client_i"]], client_id_to_idx[r["client_j"]]]
        if r["client_i"] in client_id_to_idx and r["client_j"] in client_id_to_idx else np.nan,
        axis=1,
    )
    valid_transfer = transfer_df.dropna(subset=["d_dyn", "transfer_loss"])
    if len(valid_transfer) >= 8:
        transfer_median_rho, transfer_ci = oracle_analysis.bootstrap_spearman_ci(
            D_unc, _transfer_loss_matrix_from_long(transfer_df, client_ids), n_boot=1000, seed=cfg["seed"]
        )
    else:
        transfer_median_rho, transfer_ci = float("nan"), (float("nan"), float("nan"))
    print(f"[phase1] transfer-loss vs dynamics-distance: median rho={transfer_median_rho:.3f}, 95% CI={transfer_ci}")

    # ---- 8d) held-out criteria + final decision (correction-plan item 7) ---------------
    def _mean_test_rmse(level: str) -> float:
        sel = held_out_df[(held_out_df["level"] == level) & held_out_df["feasible"]]
        return float(sel["test_rmse"].mean()) if len(sel) else float("nan")

    global_test_perf = _mean_test_rmse("global")
    cluster_test_perf = _mean_test_rmse("estimated_cluster")
    individual_test_perf = _mean_test_rmse("individual")
    G_cluster = (
        (global_test_perf - cluster_test_perf) / global_test_perf
        if np.isfinite(global_test_perf) and global_test_perf != 0
        else float("nan")
    )
    individual_vs_cluster_gap = (
        (cluster_test_perf - individual_test_perf) / cluster_test_perf
        if np.isfinite(cluster_test_perf) and cluster_test_perf != 0
        else float("nan")
    )
    # 90th percentile (not max) of pairwise transfer loss - a fleet-mean statistic like
    # G_cluster structurally can't see worst-case/pairwise risk (see decide_next_step's
    # docstring); this is the deliberately-not-mean-based counterpart. 90th rather than
    # max to stay robust to a single residual outlier pair, the same lesson learned
    # from the outlier-donor investigation in docs/phase1_diagnostic_findings.md.
    finite_transfer_loss = transfer_df["transfer_loss"].dropna()
    worst_case_transfer_risk = (
        float(np.percentile(finite_transfer_loss, 90)) if len(finite_transfer_loss) >= 10 else float("nan")
    )
    decision = decide_next_step(
        G_cluster, individual_vs_cluster_gap, transfer_median_rho,
        transfer_correlation_ci=transfer_ci, worst_case_transfer_risk=worst_case_transfer_risk,
    )
    print(
        f"[phase1] held-out G_cluster={G_cluster:.1%}, individual-vs-cluster gap={individual_vs_cluster_gap:.1%}, "
        f"worst_case_transfer_risk(p90)={worst_case_transfer_risk:.1%}, decision={decision}"
    )

    # ---- 9) write processed outputs -----------------------------------------------------
    df.to_parquet(processed_dir / "oracle_landscapes.parquet", index=False)
    pairwise_dynamics_df.to_parquet(processed_dir / "pairwise_dynamics_distance.parquet", index=False)
    pairwise_landscape_df.to_parquet(processed_dir / "pairwise_landscape_similarity.parquet", index=False)
    oracle_optima_df.to_parquet(processed_dir / "oracle_optima.parquet", index=False)
    held_out_df.to_parquet(processed_dir / "held_out_performance.parquet", index=False)
    transfer_df.to_parquet(processed_dir / "transfer_loss_matrix.parquet", index=False)
    print(f"[phase1] wrote processed tables to {processed_dir}")

    # ---- 10) figures ---------------------------------------------------------------------
    _figure_a_landscape_slices(fleet, ind, banks, cached_evaluate, eval_kwargs, controller_cfg, cfg, figures_dir)
    _figure_b_distance_vs_similarity(pairwise_landscape_df, report, figures_dir)
    _figure_b2_transfer_loss_vs_distance(transfer_df, transfer_median_rho, transfer_ci, figures_dir)
    _figure_c_specialization_hierarchy(held_out_df, figures_dir)

    # ---- 11) summary + manifest -----------------------------------------------------------
    _write_summary(
        report, df, ind, cluster_result, cfg, fleet_cfg, controller_cfg, out_dir, regime_names,
        global_test_perf=global_test_perf, cluster_test_perf=cluster_test_perf,
        individual_test_perf=individual_test_perf, G_cluster=G_cluster,
        individual_vs_cluster_gap=individual_vs_cluster_gap,
        transfer_median_rho=transfer_median_rho, transfer_ci=transfer_ci,
        worst_case_transfer_risk=worst_case_transfer_risk, decision=decision,
    )
    write_run_manifest(str(out_dir), {"run_config": cfg, "fleet_config": fleet_cfg, "controller_config": controller_cfg})
    print(f"[phase1] done - see {out_dir}")


def _transfer_loss_matrix_from_long(transfer_df: pd.DataFrame, client_ids: list) -> np.ndarray:
    """Long-form (client_i, client_j, transfer_loss) -> dense (N,N) matrix, symmetrized
    by averaging L_{i<-j} and L_{j<-i} (transfer is directional; the dyn-distance
    correlation check needs one symmetric summary per pair, same convention as the
    landscape-distance matrix it's compared against)."""
    idx = {cid: i for i, cid in enumerate(client_ids)}
    n = len(client_ids)
    M = np.full((n, n), np.nan)
    for _, row in transfer_df.iterrows():
        i, j = idx.get(row["client_i"]), idx.get(row["client_j"])
        if i is None or j is None or not np.isfinite(row["transfer_loss"]):
            continue
        M[i, j] = row["transfer_loss"]
    sym = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            vals = [v for v in (M[i, j], M[j, i]) if np.isfinite(v)]
            if vals:
                sym[i, j] = float(np.mean(vals))
    return sym


_XI_LABELS = {0: r"$\xi_1$ (v_y / beta)", 1: r"$\xi_2$ (r)", 2: r"$\xi_3$ (integral)"}


def _grid_eval_worker(client, bank, cached_evaluate, eval_kwargs, dim_x, dim_y, fixed, xi_lower, xi_upper, grid_res):
    """Evaluate a 2D grid over (dim_x, dim_y), holding the third xi component at
    `fixed`. Runs as ONE joblib job per (client, slice) - at full scale (grid_res=25)
    this is 625 simulations; doing all 6 (regime x slice) combos sequentially in the
    caller's process (as an earlier version of this function did) took over an hour
    with 8 idle cores, so this is submitted as a job like every other evaluation stage
    in this script."""
    x_vals = np.linspace(xi_lower[dim_x], xi_upper[dim_x], grid_res)
    y_vals = np.linspace(xi_lower[dim_y], xi_upper[dim_y], grid_res)
    Z = np.full((grid_res, grid_res), np.nan)
    for a, xv in enumerate(x_vals):
        for b, yv in enumerate(y_vals):
            xi = [fixed, fixed, fixed]
            xi[dim_x], xi[dim_y] = xv, yv
            result = cached_evaluate(client, xi, bank, **eval_kwargs)
            if result.feasible:
                Z[b, a] = result.tracking_rmse
    return x_vals, y_vals, Z


def _figure_a_landscape_slices(fleet, ind, banks, cached_evaluate, eval_kwargs, controller_cfg, cfg, figures_dir):
    xi_lower = np.asarray(controller_cfg["xi_lower"], float)
    xi_upper = np.asarray(controller_cfg["xi_upper"], float)
    grid_res = cfg.get("fig_a_grid_resolution", 15)
    regimes = list(dict.fromkeys(c.cluster.name for c in fleet))
    ind_xi = ind.set_index("client_id")[["xi_1", "xi_2", "xi_3"]]

    # Row 0: (xi_1, xi_2) slice at each client's own optimal xi_3. Row 1: (xi_2, xi_3)
    # slice at each client's own optimal xi_1 - checks whether the integral weight has
    # any effect the first slice (fixed xi_3) couldn't show.
    slices = [(0, 1, "xi_3"), (1, 2, "xi_1")]
    representative = {regime: next(c for c in fleet if c.cluster.name == regime) for regime in regimes}
    jobs = []  # (row, col, regime, client, fixed_val) in submission order
    for row, (dim_x, dim_y, fixed_name) in enumerate(slices):
        for col, regime in enumerate(regimes):
            client = representative[regime]
            has_opt = client.client_id in ind_xi.index
            fixed_val = float(ind_xi.loc[client.client_id, fixed_name]) if has_opt else 0.0
            jobs.append((row, col, regime, client, fixed_val))

    job_results = Parallel(n_jobs=cfg.get("n_jobs", -1), backend="loky", verbose=5)(
        delayed(_grid_eval_worker)(
            client, banks[client.client_id], cached_evaluate, eval_kwargs,
            slices[row][0], slices[row][1], fixed_val, xi_lower, xi_upper, grid_res,
        )
        for row, col, regime, client, fixed_val in jobs
    )

    fig, axes = plt.subplots(2, len(regimes), figsize=(4 * len(regimes), 7), squeeze=False)
    for (row, col, regime, client, fixed_val), (x_vals, y_vals, Z) in zip(jobs, job_results):
        dim_x, dim_y, fixed_name = slices[row]
        ax = axes[row][col]
        has_opt = client.client_id in ind_xi.index
        im = ax.pcolormesh(x_vals, y_vals, Z, shading="auto", cmap="viridis")
        if has_opt:
            ax.scatter(
                *ind_xi.loc[client.client_id, [f"xi_{dim_x + 1}", f"xi_{dim_y + 1}"]],
                marker="*", s=120, c="red", edgecolors="white", label="optimum",
            )
        ax.set_title(f"{regime} ({client.client_id})\n{fixed_name}={fixed_val:.2f} fixed")
        ax.set_xlabel(_XI_LABELS[dim_x])
        ax.set_ylabel(_XI_LABELS[dim_y])
        fig.colorbar(im, ax=ax, label="tracking RMSE")
    fig.suptitle("Figure A - oracle performance landscape slices (third xi fixed at each client's own optimum)")
    fig.tight_layout()
    fig.savefig(figures_dir / "figA_oracle_landscape_slice.png")
    plt.close(fig)


def _figure_b_distance_vs_similarity(pairwise_landscape_df, report, figures_dir):
    df = pairwise_landscape_df.dropna(subset=["spearman_r"])
    fig, ax = plt.subplots(figsize=(5, 4))
    for same, color, label in [(True, _SAME_REGIME_COLOR, "same true regime"), (False, "#999999", "different true regime")]:
        sub = df[df["same_true_regime"] == same]
        ax.scatter(sub["d_uncertainty_aware"], 1 - sub["spearman_r"], s=14, alpha=0.6, color=color, label=label)
    ax.set_xlabel(r"uncertainty-aware dynamics distance $d^{dyn}_{ij}$")
    ax.set_ylabel(r"$1 - \rho^{Spearman}_{ij}$ (landscape distance)")
    ax.set_title(
        f"Figure B - dynamics distance vs landscape distance\n"
        f"bootstrap median Spearman(rho)={report.criterion_3_dyn_landscape_spearman:.2f}, "
        f"95% CI=({report.criterion_3_ci[0]:.2f}, {report.criterion_3_ci[1]:.2f})"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "figB_dynamics_vs_landscape_distance.png")
    plt.close(fig)


def _figure_c_specialization_hierarchy(held_out_df, figures_dir):
    """Correction-plan item 2: uses HELD-OUT test_rmse (not calibration tracking_rmse) -
    each level's calibration-selected xi, re-scored on disjoint test episodes, so this
    reflects real performance rather than partly selection noise."""
    pivot = held_out_df[held_out_df["feasible"]].pivot_table(
        index="client_id", columns="level", values="test_rmse"
    )
    order = [lvl for lvl in ["global", "estimated_cluster", "true_regime", "individual"] if lvl in pivot.columns]
    if "individual" not in pivot.columns:
        return
    normalized = pivot[order].div(pivot["individual"], axis=0)

    fig, ax = plt.subplots(figsize=(5, 4))
    data = [normalized[c].dropna().to_numpy() for c in order]
    ax.boxplot(data, tick_labels=order, showmeans=True)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("held-out test RMSE / individual-optimum held-out test RMSE")
    ax.set_title("Figure C - global vs cluster vs true-regime vs individual, held-out test performance")
    fig.tight_layout()
    fig.savefig(figures_dir / "figC_specialization_hierarchy.png")
    plt.close(fig)


def _figure_b2_transfer_loss_vs_distance(transfer_df, median_rho, ci, figures_dir):
    """Correction-plan item 6, the PRIMARY RQ1 diagnostic: does identified-dynamics
    distance predict controller-TRANSFER loss (not just whole-landscape correlation,
    which stays available as the secondary Figure B)."""
    sub = transfer_df.dropna(subset=["d_dyn", "transfer_loss"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(sub["d_dyn"], sub["transfer_loss"], s=10, alpha=0.4, color="#0072B2")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(r"uncertainty-aware dynamics distance $d^{dyn}_{ij}$")
    ax.set_ylabel(r"transfer loss $L_{i \leftarrow j}$ (relative RMSE increase, held-out)")
    ax.set_title(
        "Figure B2 (PRIMARY) - dynamics distance vs controller transfer loss\n"
        f"bootstrap median Spearman(rho)={median_rho:.2f}, 95% CI=({ci[0]:.2f}, {ci[1]:.2f})"
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "figB2_transfer_loss_vs_distance.png")
    plt.close(fig)


_DECISION_TEXT = {
    "implement_dynamics_informed_fbo": (
        "Cluster benefit AND a dynamics-distance/transfer-loss relation both exist on "
        "held-out data - proceed to implement dynamics-informed FBO (correction-plan "
        "item 7, case 1)."
    ),
    "test_control_aware_descriptor": (
        "Cluster benefit exists on held-out data, but raw identified-parameter distance "
        "does not predict transfer loss - test one control-aware descriptor (closed-loop "
        "poles/sensitivity, frequency response) before implementing FBO on raw theta "
        "distance (correction-plan item 7, case 2)."
    ),
    "rethink_federation_granularity": (
        "Only fully individual tuning helps on held-out data (no real cluster-level "
        "benefit) - rethink the intended federation granularity before implementing "
        "cluster-level FBO (correction-plan item 7, case 3)."
    ),
    "pursue_dynamics_informed_safety_screening": (
        "No mean-level (fleet-average) benefit from cluster/global tuning, BUT dynamics "
        "distance significantly predicts transfer loss AND a real worst-case/pairwise "
        "transfer risk exists - global/cluster selection structurally can't see this "
        "(it never picks an xi that's bad in the mean), so it's invisible to G_cluster "
        "even though it's real. The value here looks like SAFETY SCREENING (reject "
        "candidate controllers likely to transfer badly to a specific client, using "
        "identification uncertainty - plan doc Phase 5) rather than average-performance "
        "FBO. Pursue uncertainty-aware screening before/instead of standard FBO."
    ),
    "stop_direction": (
        "Global ~= cluster ~= individual on held-out data, AND no significant worst-case "
        "transfer risk either - stop this direction rather than forcing FBO "
        "(correction-plan item 7, case 4)."
    ),
}


def _config_dependent_notes(controller_cfg: dict) -> list:
    """Notes that depend on the ACTUAL resolved controller_cfg for this run - kept
    dynamic (not hardcoded in oracle_analysis.py) so they can't go stale the way an
    earlier version of this summary did (claiming state_scale=identity /
    u_rate_max=unconstrained on runs where both were already set)."""
    notes = []
    scale = controller_cfg.get("state_scale")
    if scale is not None and tuple(scale) == (1.0, 1.0, 1.0):
        notes.append(
            "state_scale is identity - raw xi magnitudes are not guaranteed physically "
            "comparable across clients; see plan doc §6.3."
        )
    else:
        notes.append(f"state_scale = {scale} (correction-plan item 1; see scripts/compute_state_scales.py).")
    u_rate_max = controller_cfg.get("u_rate_max")
    beta_max_deg = controller_cfg.get("beta_max_deg")
    if u_rate_max is None and beta_max_deg is None:
        notes.append("u_rate_max / beta_max_deg are both unconstrained (null).")
    else:
        notes.append(f"u_rate_max={u_rate_max}, beta_max_deg={beta_max_deg} (correction-plan item 3).")
    return notes


def _write_summary(
    report, df, ind, cluster_result, cfg, fleet_cfg, controller_cfg, out_dir, regime_names, *,
    global_test_perf, cluster_test_perf, individual_test_perf, G_cluster,
    individual_vs_cluster_gap, transfer_median_rho, transfer_ci, worst_case_transfer_risk, decision,
):
    n_feasible = int(df["feasible"].sum())
    lines = [
        f"# Phase 1 oracle feasibility study - {cfg['run_tag']}",
        "",
        f"Fleet: {fleet_cfg['clients_per_regime']} clients/regime x {len(regime_names)} regimes "
        f"({', '.join(regime_names)}); Sobol candidates M={cfg['n_sobol']}; "
        f"{cfg['n_calibration_episodes']} calibration episodes/candidate; "
        f"{cfg.get('n_test_episodes', 0)} held-out test episodes/selected controller; "
        f"plant_mode={controller_cfg.get('plant_mode', 'linear')!r}.",
        f"Evaluated {len(df)} (client, xi) calibration pairs, {n_feasible} feasible ({n_feasible / len(df):.1%}).",
        f"Estimated cluster sizes: {np.bincount(cluster_result.client_cluster_labels).tolist()}.",
        "",
        "## Held-out criteria (correction-plan items 2, 6, 7 - PRIMARY)",
        "",
        f"- **Global performance** (mean HELD-OUT test RMSE under the global-selected xi): {global_test_perf:.4g}",
        f"- **Cluster performance** (mean HELD-OUT test RMSE under each client's estimated-cluster xi): {cluster_test_perf:.4g}",
        f"- **Individual performance** (mean HELD-OUT test RMSE under each client's own xi): {individual_test_perf:.4g}",
        f"- **G_cluster** = (J_global - J_cluster) / J_global, on held-out data: {G_cluster:.1%}",
        f"- **Individual-vs-cluster gap** (held-out): {individual_vs_cluster_gap:.1%}",
        f"- **Transfer-loss vs dynamics-distance** (primary RQ1 diagnostic, Figure B2): "
        f"bootstrap median Spearman(rho)={transfer_median_rho:.3f}, 95% CI=({transfer_ci[0]:.3f}, {transfer_ci[1]:.3f})",
        f"- **Worst-case transfer risk** (90th percentile of pairwise transfer loss - "
        f"a non-mean-based counterpart to G_cluster): {worst_case_transfer_risk:.1%}",
        "",
        f"## Decision (correction-plan item 7): `{decision}`",
        "",
        _DECISION_TEXT[decision],
        "",
        "## Calibration-based go/no-go criteria (doc §10.7 - SECONDARY, selection-biased; "
        "see held-out criteria above for the corrected read)",
        "",
        f"- Global performance (calibration): {report.global_perf:.4g}",
        f"- Cluster performance (calibration): {report.cluster_perf:.4g}",
        f"- Individual performance (calibration): {report.individual_perf:.4g}",
        f"- Criterion 1 (calibration specialization gain): {report.criterion_1_specialization_gain:.1%}",
        f"- Criterion 2 (calibration personalization gap closed): {report.criterion_2_personalization_gap_closed:.1%}",
        f"- Criterion 3 (whole-landscape dyn-distance correlation, secondary Figure B): "
        f"rho={report.criterion_3_dyn_landscape_spearman:.3f}, 95% CI=({report.criterion_3_ci[0]:.3f}, {report.criterion_3_ci[1]:.3f})",
        f"- Criterion 4 (search space boundary fraction): {report.criterion_4_boundary_fraction:.1%}",
        "",
        f"Kill condition (calibration-based): {'**TRIGGERED**' if report.kill_condition_triggered else 'not triggered'} - {report.kill_condition_detail}",
        "",
        "## Notes / open items",
        "",
    ]
    lines += [f"- {note}" for note in report.notes]
    lines += _config_dependent_notes(controller_cfg)
    (out_dir / "phase1_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[phase1] wrote summary to {out_dir / 'phase1_summary.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
