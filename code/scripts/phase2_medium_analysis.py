"""
Post-hoc analysis of a completed phase2_fbo_comparison run: AURC (area under the
normalized-regret curve), paired seed-level statistics, family-wise breakdown, and a
peer-similarity-vs-benefit check (user request, 2026-09-01, following the corrected
medium-scale c1cds02 verdict in docs/phase2_diagnostic_findings.md). Reads
already-computed results/<run_tag>/processed/*.parquet - does NOT re-run any BO
evaluations.

The one thing this DOES recompute is the per-seed similarity matrix s_ij (fleet
identification + distance/similarity only, no true-plant simulation - the same cheap
step `_prepare_seed_context` does before the expensive BO loop), since s_ij isn't saved
to disk by phase2_fbo_comparison.py.

Usage:
    python scripts/phase2_medium_analysis.py --config config/runs/phase2_fbo_comparison_medium_c1cds02.yaml
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from src.dynamics_distance import pairwise_uncertainty_aware_distance  # noqa: E402
from src.ifac_bridge import pack_theta_AB  # noqa: E402
from src.io_utils import deep_update, load_run_config, load_yaml, set_global_seed  # noqa: E402
from src.similarity import median_distance_lengthscale, similarity_matrix  # noqa: E402

from phase2_fbo_comparison import _resolve, build_fleet_and_identify  # noqa: E402

FAMILIES = ("nominal", "payload", "tire_degraded")


def _family_of(client_id: str) -> str:
    for fam in FAMILIES:
        if client_id.startswith(fam):
            return fam
    raise ValueError(f"Unrecognized client_id family: {client_id!r}")


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Plain trapezoidal rule - avoids the np.trapz/np.trapezoid version churn."""
    return float(np.sum((y[1:] + y[:-1]) / 2.0 * np.diff(x)))


def compute_aurc(trace_df: pd.DataFrame, *, metric_col: str = "regret") -> pd.DataFrame:
    """AURC = trapezoidal area under the given regret metric vs n, per (seed, method,
    client_id) - lower is better, same direction as the metric it integrates.
    `metric_col="regret"` (default) is the baseline-normalized metric (Parts 2-7,
    secondary continuity metric as of Part 8). `metric_col="regret_star"` is the
    oracle-relative metric (Part 8's new PRIMARY metric for any GENERATION_VERSION_V3
    run, docs/phase2_diagnostic_findings.md, 2026-09-03) - doesn't collapse toward a
    near-zero denominator the way `regret`'s `J_base - J_star` can."""
    rows = []
    for (seed, method, cid), g in trace_df.groupby(["seed", "method", "client_id"]):
        g = g.sort_values("n_local_evals")
        rows.append(
            {
                "seed": seed, "method": method, "client_id": cid,
                "aurc": _trapz(g[metric_col].to_numpy(), g["n_local_evals"].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def compute_stagnation(trace_df: pd.DataFrame, *, method: str = "independent", cutoff: int = 20) -> pd.DataFrame:
    """D_i: fraction of consecutive-checkpoint round-pairs (from n_init through
    `cutoff`) in which `method`'s own best_calibration_rmse did NOT improve - a local
    "difficulty"/stagnation descriptor computed BEFORE federation acts (uses only
    `method`'s own trajectory - self-only for `independent`) and built from PHYSICAL
    calibration RMSE rather than normalized regret, to avoid the same small
    baseline-oracle-gap sensitivity that inflated Part 5's seed-37050 AURC deltas.
    Stops at `cutoff` (< the run's true final budget) so it's a leading indicator, not
    a lookahead into the window `G_i` (AURC-based eventual benefit) is measured over.
    User request 2026-09-02, replacing the peer-similarity-sum check (S_i vs G_i,
    ruled out at rho=-0.011/p=0.79 with 20-seed data - not revisited)."""
    rows = []
    for (seed, cid), g in trace_df[trace_df["method"] == method].groupby(["seed", "client_id"]):
        g = g[g["n_local_evals"] <= cutoff].sort_values("n_local_evals")
        vals = g["best_calibration_rmse"].to_numpy()
        if len(vals) < 2:
            continue
        # best-so-far is monotone non-increasing by construction; a >=0 step is a
        # round that produced no improvement over the incumbent.
        stagnant_rounds = np.diff(vals) >= 0
        rows.append({"seed": seed, "client_id": cid, "D_i": float(stagnant_rounds.mean())})
    return pd.DataFrame(rows)


def per_seed_similarity_matrices(config_path: str) -> dict:
    """Recomputes s_ij per seed (identification + distance/similarity only - no
    true-plant simulation, the same cheap step `_prepare_seed_context` does before the
    expensive BO loop) since s_ij isn't saved to disk by phase2_fbo_comparison.py.
    Returns {seed: (client_ids, S)} with S[i,i]=0 (self-similarity excluded).

    Builds the same eval_kwargs/state_scale/fixed_R phase2_fbo_comparison.py's main()
    does and passes them through to build_fleet_and_identify - required (not just to
    avoid a crash) whenever fleet_cfg["generation_version"] is the Step C v2 generator,
    since it needs them to run the SAME baseline-feasibility admission check the actual
    run used - otherwise this recompute could accept a DIFFERENT physical draw for a
    given client_id than the run it's meant to describe."""
    cfg = load_run_config(config_path)
    fleet_cfg = deep_update(load_yaml(_resolve(cfg["fleet_config"])), cfg.get("fleet_overrides", {}))
    controller_cfg = deep_update(load_yaml(_resolve(cfg["controller_config"])), cfg.get("controller_overrides", {}))
    beta_max = np.deg2rad(controller_cfg["beta_max_deg"])
    eval_kwargs = dict(
        u_min=controller_cfg["u_min"], u_max=controller_cfg["u_max"],
        u_rate_max=controller_cfg.get("u_rate_max"), beta_max=beta_max,
        noise_std=controller_cfg["noise_std"], process_noise=controller_cfg["process_noise"],
        plant_mode=controller_cfg.get("plant_mode", "linear"), mu=controller_cfg.get("tire_mu", 1.0),
    )

    out = {}
    for seed in cfg["seeds"]:
        set_global_seed(seed)
        fleet_cfg_seed = deep_update(fleet_cfg, {"seed": fleet_cfg["seed"] + seed})
        fleet, _ = build_fleet_and_identify(
            fleet_cfg_seed, controller_cfg, cfg,
            eval_kwargs=eval_kwargs, state_scale=controller_cfg["state_scale"], fixed_R=controller_cfg["fixed_R"],
        )
        thetas = [pack_theta_AB(c.Ad_hat, c.Bd_hat) for c in fleet]
        precisions = [c.W_raw for c in fleet]
        D2 = pairwise_uncertainty_aware_distance(thetas, precisions)
        l_d = median_distance_lengthscale(D2)
        S = similarity_matrix(D2, l_d)
        np.fill_diagonal(S, 0.0)
        out[seed] = ([c.client_id for c in fleet], S)
    return out


def similarity_sums(S_matrices: dict) -> pd.DataFrame:
    """S_i = sum_{j!=i} s_ij per (seed, client_id) - a client's total available peer
    relevance under the SAME similarity weighting the FBO methods actually use."""
    rows = []
    for seed, (client_ids, S) in S_matrices.items():
        for i, cid in enumerate(client_ids):
            rows.append({"seed": seed, "client_id": cid, "S_i": float(S[i].sum())})
    return pd.DataFrame(rows)


def paired_seed_stats(aurc_df: pd.DataFrame, method_a: str, method_b: str, n_boot: int = 10000, seed: int = 0) -> dict:
    """Per-seed median AURC delta (method_a - method_b, across clients), a percentile
    bootstrap 95% CI over seeds, and a Wilcoxon signed-rank test across seeds - power
    depends entirely on how many seeds `aurc_df` covers (the caller reports n and
    flags low power below n=10)."""
    piv = aurc_df[aurc_df["method"].isin([method_a, method_b])].pivot_table(
        index=["seed", "client_id"], columns="method", values="aurc"
    )
    piv["delta"] = piv[method_a] - piv[method_b]
    per_seed_delta = piv.groupby("seed")["delta"].median()

    rng = np.random.default_rng(seed)
    vals = per_seed_delta.to_numpy()
    # Two different summaries of the SAME per-seed deltas, bootstrapped separately:
    # the mean-of-seed-medians is sensitive to a single extreme-outlier seed (a mean
    # can be dragged arbitrarily far by one value); the median-of-seed-medians -
    # what the Part 4 frozen go/no-go criterion actually specifies - is robust to
    # exactly that. Report both; treat the median as primary.
    boot_mean = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)]
    boot_median = [np.median(rng.choice(vals, size=len(vals), replace=True)) for _ in range(n_boot)]
    mean_ci_lo, mean_ci_hi = np.percentile(boot_mean, [2.5, 97.5])
    median_ci_lo, median_ci_hi = np.percentile(boot_median, [2.5, 97.5])

    wilcoxon_stat, wilcoxon_p = float("nan"), float("nan")
    try:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(vals)
    except ValueError:
        pass  # e.g. all-zero differences - too few seeds for the test to run

    return {
        "method_a": method_a, "method_b": method_b,
        "per_seed_median_delta": per_seed_delta.to_dict(),
        "mean_of_seed_medians": float(per_seed_delta.mean()),
        "median_of_seed_medians": float(np.median(vals)),
        "bootstrap_ci_95_of_mean": (float(mean_ci_lo), float(mean_ci_hi)),
        "bootstrap_ci_95_of_median": (float(median_ci_lo), float(median_ci_hi)),
        "wilcoxon_stat": float(wilcoxon_stat), "wilcoxon_p": float(wilcoxon_p),
    }


def family_breakdown(aurc_df: pd.DataFrame) -> pd.DataFrame:
    df = aurc_df.copy()
    df["family"] = df["client_id"].apply(_family_of)
    return df.groupby(["family", "method"])["aurc"].agg(
        median="median", mean="mean", std="std", p90=lambda s: s.quantile(0.9), count="count"
    )


def paired_seed_stats_by_family(aurc_df: pd.DataFrame, method_a: str, method_b: str) -> pd.DataFrame:
    """Delta_{s,f} = median_{i in family f, seed s}(AURC_a - AURC_b) - the seed-level
    family breakdown the user asked for (2026-09-01), so a family's apparent
    benefit/variance can be checked for persistence across seeds rather than pooled
    across clients within one seed."""
    df = aurc_df.copy()
    df["family"] = df["client_id"].apply(_family_of)
    piv = df[df["method"].isin([method_a, method_b])].pivot_table(
        index=["seed", "family", "client_id"], columns="method", values="aurc"
    )
    piv["delta"] = piv[method_a] - piv[method_b]
    return piv.groupby(["family", "seed"])["delta"].median().unstack("seed")


def outlier_audit(
    aurc_df: pd.DataFrame, summ_df: pd.DataFrame, S_matrices: dict, *,
    family: str, method: str, baseline_method: str = "independent", top_n: int = 5,
) -> pd.DataFrame:
    """Identifies the top_n worst-AURC (seed, client_id) pairs within `family` under
    `method`, and for each reports what's needed to distinguish REAL negative transfer
    from a normalized-regret metric artifact (user request, 2026-09-01): the
    baseline-oracle gap (small gap -> normalized regret can blow up from a tiny
    absolute RMSE difference), the held-out RMSE under `method` vs `baseline_method`
    (does the controller ACTUALLY perform worse in physical units, or just look bad
    normalized?), infeasible-evaluation counts, and the client's largest peer
    similarity weights (is a specific risky donor implicated?). Does not classify
    automatically - `main()` prints the raw numbers for eyeball classification, per
    "do not tune anything based on this audit, just classify the failure"."""
    fam_client_ids = sorted(cid for cid in summ_df["client_id"].unique() if _family_of(cid) == family)
    sub = aurc_df[(aurc_df["method"] == method) & (aurc_df["client_id"].isin(fam_client_ids))]
    worst = sub.sort_values("aurc", ascending=False).head(top_n)

    rows = []
    for _, r in worst.iterrows():
        seed, cid = r["seed"], r["client_id"]
        s_row = summ_df[(summ_df.seed == seed) & (summ_df.client_id == cid) & (summ_df.method == method)].iloc[0]
        b_row = summ_df[
            (summ_df.seed == seed) & (summ_df.client_id == cid) & (summ_df.method == baseline_method)
        ].iloc[0]
        ind_aurc = aurc_df[
            (aurc_df.seed == seed) & (aurc_df.client_id == cid) & (aurc_df.method == baseline_method)
        ]["aurc"].iloc[0]

        client_ids, S = S_matrices[seed]
        i = client_ids.index(cid)
        top_peer_idx = np.argsort(-S[i])[:3]
        top_peers = ", ".join(f"{client_ids[j]}={S[i, j]:.2f}" for j in top_peer_idx)

        rows.append(
            {
                "seed": seed, "client_id": cid,
                f"aurc_{method}": round(float(r["aurc"]), 4), f"aurc_{baseline_method}": round(float(ind_aurc), 4),
                "baseline_rmse": round(float(s_row["baseline_rmse"]), 5),
                "oracle_rmse": round(float(s_row["oracle_rmse"]), 5),
                "base_minus_oracle_gap": round(float(s_row["baseline_rmse"] - s_row["oracle_rmse"]), 5),
                f"final_held_out_rmse_{method}": round(float(s_row["final_held_out_rmse"]), 5),
                f"final_held_out_rmse_{baseline_method}": round(float(b_row["final_held_out_rmse"]), 5),
                f"n_infeasible_{method}": int(s_row["n_infeasible"]),
                f"n_infeasible_{baseline_method}": int(b_row["n_infeasible"]),
                "top_peers": top_peers,
            }
        )
    return pd.DataFrame(rows)


def plot_feasible_support_by_family(feasible_support_df: pd.DataFrame, out_path: Path) -> None:
    """Step D's first requested plot (docs/phase2_diagnostic_findings.md, 2026-09-02):
    distribution of phi_i (coarse Sobol-scan feasible fraction) by family, jittered
    strip plot so individual clients are visible, baseline-infeasible clients marked
    with an open red ring so it's visually obvious whether they're the same clients
    dragging phi down or a separate population."""
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "figure.dpi": 150})
    families = FAMILIES
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, fam in enumerate(families):
        sub = feasible_support_df[feasible_support_df["family"] == fam]
        jitter = rng.uniform(-0.15, 0.15, size=len(sub))
        feasible_mask = sub["baseline_feasible"].astype(bool)
        ax.scatter(
            i + jitter[feasible_mask.values], sub.loc[feasible_mask, "phi"],
            s=14, alpha=0.5, color="#0072B2", label="baseline feasible" if i == 0 else None,
        )
        ax.scatter(
            i + jitter[(~feasible_mask).values], sub.loc[~feasible_mask, "phi"],
            s=36, alpha=0.9, facecolors="none", edgecolors="#D55E00", linewidths=1.3,
            label="baseline INfeasible" if i == 0 else None,
        )
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families)
    ax.set_ylabel(r"$\hat\phi_i$ (feasible fraction of $N_{\rm Sobol}$)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Feasible-support estimate by family (Step D)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(config_path: str) -> None:
    cfg = load_run_config(config_path)
    run_tag = cfg["run_tag"]
    out_dir = _ROOT / "results" / run_tag
    trace_df = pd.read_parquet(out_dir / "processed" / "fbo_convergence_traces.parquet")
    summ_df = pd.read_parquet(out_dir / "processed" / "fbo_client_summary.parquet")

    feasible_support_path = out_dir / "processed" / "feasible_support.csv"
    if feasible_support_path.exists():
        feasible_support_df = pd.read_csv(feasible_support_path)
        print("=== feasible-support (phi_i) by family - Step D ===")
        print(feasible_support_df.groupby("family")["phi"].describe().round(3))
        print(feasible_support_df.groupby("family")["baseline_feasible"].mean().rename("baseline_feasible_rate").round(3))
        fig_path = out_dir / "figures" / "fig_phi_by_family.png"
        plot_feasible_support_by_family(feasible_support_df, fig_path)
        print(f"Wrote {fig_path}")
        print()
    else:
        print(f"(no feasible_support.csv at {feasible_support_path} - skipping Step D plot; pre-2026-09-02 run)")
        print()

    has_regret_star = "regret_star" in trace_df.columns

    print("=== AURC (trapezoidal area under normalized-regret curve, lower=better) ===")
    aurc_df = compute_aurc(trace_df)  # baseline-normalized `regret` - secondary continuity metric as of Part 8
    print(aurc_df.groupby("method")["aurc"].agg(median="median", mean="mean", std="std", p90=lambda s: s.quantile(0.9)).round(4))
    print()

    print("=== family-wise AURC breakdown (incl. p90 tail) ===")
    print(family_breakdown(aurc_df).round(4))
    print()

    if has_regret_star:
        # Part 8 (docs/phase2_diagnostic_findings.md, 2026-09-03): regret_star = (J_best -
        # J_star) / J_star doesn't have `regret`'s near-zero-denominator failure mode, so
        # its AURC is the PRIMARY pre-registered statistic for any GENERATION_VERSION_V3
        # run - printed first and labeled accordingly; the regret-based block below stays
        # for continuity with Parts 2-7's numbers, now explicitly SECONDARY.
        aurc_star_df = compute_aurc(trace_df, metric_col="regret_star")
        print("=== AURC_star (oracle-relative regret_star, PRIMARY metric, lower=better) ===")
        print(
            aurc_star_df.groupby("method")["aurc"]
            .agg(median="median", mean="mean", std="std", p90=lambda s: s.quantile(0.9)).round(4)
        )
        print()
        print("=== PRIMARY: paired seed-level AURC_star deltas (negative = method_a better) ===")
        for a, b in [("similarity", "independent"), ("similarity", "global"), ("recipient_aware", "similarity")]:
            res = paired_seed_stats(aurc_star_df, a, b)
            n_seeds = len(res["per_seed_median_delta"])
            n_favorable = sum(1 for v in res["per_seed_median_delta"].values() if v < 0)
            power_note = " - low power, preview only" if n_seeds < 10 else ""
            print(
                f"{a} - {b}: median-of-seed-medians={res['median_of_seed_medians']:.4f} "
                f"(bootstrap 95% CI={res['bootstrap_ci_95_of_median']}) [PRIMARY, regret_star], "
                f"mean-of-seed-medians={res['mean_of_seed_medians']:.4f} "
                f"(bootstrap 95% CI={res['bootstrap_ci_95_of_mean']}), "
                f"Wilcoxon p={res['wilcoxon_p']:.4f} (n={n_seeds} seeds{power_note}), "
                f"{n_favorable}/{n_seeds} seeds favorable"
            )
            print(f"  per-seed medians: { {k: round(v, 4) for k, v in res['per_seed_median_delta'].items()} }")
        print()
    else:
        aurc_star_df = None

    print(
        "=== paired seed-level AURC deltas, baseline-normalized regret "
        f"({'SECONDARY - see regret_star above' if has_regret_star else 'PRIMARY'}) (negative = method_a better) ==="
    )
    for a, b in [("similarity", "independent"), ("similarity", "global"), ("recipient_aware", "similarity")]:
        res = paired_seed_stats(aurc_df, a, b)
        n_seeds = len(res["per_seed_median_delta"])
        n_favorable = sum(1 for v in res["per_seed_median_delta"].values() if v < 0)
        power_note = " - low power, preview only" if n_seeds < 10 else ""
        primary_tag = "[SECONDARY - see regret_star above]" if has_regret_star else "[PRIMARY, per Part 4's frozen criterion]"
        print(
            f"{a} - {b}: median-of-seed-medians={res['median_of_seed_medians']:.4f} "
            f"(bootstrap 95% CI={res['bootstrap_ci_95_of_median']}) {primary_tag}, "
            f"mean-of-seed-medians={res['mean_of_seed_medians']:.4f} "
            f"(bootstrap 95% CI={res['bootstrap_ci_95_of_mean']}), "
            f"Wilcoxon p={res['wilcoxon_p']:.4f} (n={n_seeds} seeds{power_note}), "
            f"{n_favorable}/{n_seeds} seeds favorable"
        )
        print(f"  per-seed medians: { {k: round(v, 4) for k, v in res['per_seed_median_delta'].items()} }")
    print()

    print("=== seed-level family deltas, similarity - independent (negative = similarity better) ===")
    family_deltas_df = paired_seed_stats_by_family(aurc_df, "similarity", "independent")
    print(family_deltas_df.round(4))
    print()

    # G_i = AURC_independent - AURC_similarity, the "eventual benefit" half of every
    # mechanism check below. The peer-similarity-sum check (S_i vs G_i) is RULED OUT
    # (rho=-0.011, p=0.79 at 20 seeds, see docs/phase2_diagnostic_findings.md Part 5) -
    # not recomputed/reprinted by default; use similarity_sums()/per_seed_similarity_matrices()
    # directly if it ever needs revisiting.
    aurc_ind = aurc_df[aurc_df["method"] == "independent"].set_index(["seed", "client_id"])["aurc"]
    aurc_sim = aurc_df[aurc_df["method"] == "similarity"].set_index(["seed", "client_id"])["aurc"]
    gain = (aurc_ind - aurc_sim).rename("G_i")

    print("=== stagnation vs benefit (does independent BO's OWN stagnation predict FBO gain?) ===")
    stag_df = compute_stagnation(trace_df, method="independent", cutoff=20)
    stag_merged = stag_df.set_index(["seed", "client_id"]).join(gain).dropna()
    rho, p = stats.spearmanr(stag_merged["D_i"], stag_merged["G_i"])
    print(f"Spearman rho(D_i, G_i) = {rho:.3f}, p = {p:.4f}, n = {len(stag_merged)}")
    print()

    S_matrices = per_seed_similarity_matrices(config_path)  # still needed for the audit's peer-weight column
    print("=== outlier audit: worst-AURC payload clients under similarity (top 5) ===")
    audit_df = outlier_audit(aurc_df, summ_df, S_matrices, family="payload", method="similarity", top_n=5)
    pd.set_option("display.width", 200)
    print(audit_df.to_string(index=False))

    processed_dir = out_dir / "processed"
    aurc_df.to_csv(processed_dir / "aurc_per_client_seed_method.csv", index=False)
    stag_merged.to_csv(processed_dir / "stagnation_vs_gain.csv")
    audit_df.to_csv(processed_dir / "payload_outlier_audit.csv", index=False)
    family_deltas_df.to_csv(processed_dir / "seed_level_family_deltas_sim_vs_ind.csv")
    print(f"\nWrote {processed_dir / 'aurc_per_client_seed_method.csv'}")
    print(f"Wrote {processed_dir / 'stagnation_vs_gain.csv'}")
    print(f"Wrote {processed_dir / 'payload_outlier_audit.csv'}")
    print(f"Wrote {processed_dir / 'seed_level_family_deltas_sim_vs_ind.csv'}")
    if aurc_star_df is not None:
        aurc_star_df.to_csv(processed_dir / "aurc_star_per_client_seed_method.csv", index=False)
        print(f"Wrote {processed_dir / 'aurc_star_per_client_seed_method.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/runs/phase2_fbo_comparison_medium_c1cds02.yaml")
    args = parser.parse_args()
    main(args.config)
