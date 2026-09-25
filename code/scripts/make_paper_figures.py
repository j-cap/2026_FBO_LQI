"""
Final-paper figures, publication-styled per the user's regeneration spec (2026-09-10,
docs/phase2_diagnostic_findings.md). Every multi-panel figure from the first pass is
now SPLIT into separate standalone single-panel files (no in-plot titles, no (a)/(b)
panel labels - the user recombines them in LaTeX) - one figure = one file. Reads
ALREADY-COMPLETED results only, no new simulation, except Fig. 6 (lane-change
reference trajectories), which is a pure function of the project's own fixed maneuver
parameters, not simulated data.

Data sources: see each fig_*() function's docstring.

Terminology updated throughout (user's Section 7): "oracle" -> "reference",
"target-local evaluations" -> "new-client experiments", "similarity" -> "dynamics-weighted"
in display labels only (the underlying method identifier `"similarity"` in the data is
untouched - this is a display-label change, not a data/code rename).

Output: figures/fig{2a,2b,3,4a,4b,5a,5b,6}_*.{pdf,png} - PDF (vector) primary, PNG at
600 dpi fallback, tight bbox, opaque background.

Usage:
    python scripts/make_paper_figures.py
"""
from __future__ import annotations

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

FIG_DIR = _ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---- global style (user's Section 1) -----------------------------------------------
SINGLE_COL = (3.45, 2.55)
DOUBLE_COL = (7.1, 3.1)

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 9,
    }
)

METHOD_COLORS = {"independent": "#999999", "global": "#D55E00", "similarity": "#0072B2"}
METHOD_LABELS = {"independent": "Independent", "global": "Global warm start", "similarity": "Dynamics-weighted warm start"}
FAMILY_COLORS = {"nominal": "#0072B2", "payload": "#D55E00", "tire_degraded": "#009E73"}
FAMILY_LABELS = {"nominal": "Nominal", "payload": "Payload", "tire_degraded": "Tire-degraded"}

LANDSCAPE_STUDY_SEEDS = [71063, 45424, 30434, 94266, 40624]
MAIN_LW = 2.2
REF_LW = 1.3
BAND_ALPHA = 0.15


def _save(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        path = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.03, transparent=False, dpi=600 if ext == "png" else None)
        print(f"wrote {path}")


def _seed_bootstrap_ci(vals: np.ndarray, *, n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap 95% CI of the mean - same methodology used throughout the
    analysis (scripts/phase2_triggered_dev.py::_seed_bootstrap_mean_ci), reused here
    directly on an array so it can drive matplotlib error bars."""
    rng = np.random.default_rng(seed)
    boot = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(np.mean(vals)), float(lo), float(hi)


# ============================================================================
# Figure 2a - vehicle-parameter heterogeneity by family
# ============================================================================
def fig2a():
    """Source: phase2_fbo_comparison_campaign_v3_c1cds05/processed/feasible_support.csv
    (Part 9 Step D per-client physical params), filtered to the 5 landscape-study seeds."""
    params_df = pd.read_csv(
        _ROOT / "results/phase2_fbo_comparison_campaign_v3_c1cds05/processed/feasible_support.csv"
    )
    params_df = params_df[params_df["seed"].isin(LANDSCAPE_STUDY_SEEDS)]

    fig, ax = plt.subplots(figsize=SINGLE_COL)
    param_cols = ["param_m", "param_Iz", "param_Cf", "param_Cr"]
    param_labels = [r"$m$", r"$I_z$", r"$C_f$", r"$C_r$"]
    nominal_mean = params_df[params_df["family"] == "nominal"][param_cols].mean()
    rng = np.random.default_rng(0)
    for pi, col in enumerate(param_cols):
        for fam in ("nominal", "payload", "tire_degraded"):
            vals = params_df.loc[params_df["family"] == fam, col] / nominal_mean[col] * 100
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            offset = {"nominal": -0.22, "payload": 0.0, "tire_degraded": 0.22}[fam]
            ax.scatter(
                np.full(len(vals), pi) + offset + jitter, vals, s=13, color=FAMILY_COLORS[fam], alpha=0.55,
                linewidths=0, label=FAMILY_LABELS[fam] if pi == 0 else None,
            )
    ax.axhline(100, color="black", linewidth=REF_LW * 0.7, linestyle=":")
    ax.set_xticks(range(len(param_cols)))
    ax.set_xticklabels(param_labels)
    ax.set_ylim(72, 133)
    ax.set_ylabel("Relative to nominal-family mean [%]")
    leg = ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, framealpha=1.0, handletextpad=0.25,
        columnspacing=0.8, borderpad=0.35,
    )
    for handle in leg.legend_handles:
        handle.set_alpha(1.0)

    fig.tight_layout()
    _save(fig, "fig2a_vehicle_parameter_heterogeneity")
    plt.close(fig)


# ============================================================================
# Figure 2b - dynamics distance vs. calibration-landscape rank correlation
# ============================================================================
def fig2b():
    """Source: results/calibration_landscape_similarity/processed/landscape_similarity_pairs.csv
    (Part 19, d_ij backfilled by scripts/backfill_landscape_distances.py)."""
    pairs_df = pd.read_csv(_ROOT / "results/calibration_landscape_similarity/processed/landscape_similarity_pairs.csv")
    if "d_ij" not in pairs_df.columns:
        raise RuntimeError("landscape_similarity_pairs.csv has no d_ij column - run scripts/backfill_landscape_distances.py first")

    fig, ax = plt.subplots(figsize=SINGLE_COL)
    for same, marker, color, label in (
        (True, "o", "#0072B2", "Same family"), (False, "^", "#D55E00", "Cross family"),
    ):
        sub = pairs_df[pairs_df["same_family"] == same]
        ax.scatter(sub["d_ij"], sub["rho_J"], s=10, alpha=0.4, marker=marker, color=color, linewidths=0, label=label)
    ax.axhline(0.95, color="#333333", linewidth=REF_LW, linestyle="--")
    ax.annotate(
        r"$\rho_J=0.95$", xy=(0.98, 0.95), xycoords=("axes fraction", "data"), ha="right", va="bottom", fontsize=8,
    )
    ax.set_ylim(0.94, 1.002)
    ax.set_xlabel(r"Dynamics distance $d_{ij}^{\mathrm{dyn}}$")
    ax.set_ylabel(r"Landscape rank correlation $\rho_{ij}^{J}$")
    leg = ax.legend(loc="lower left", framealpha=1.0, handletextpad=0.3, borderpad=0.4)
    for handle in leg.legend_handles:
        handle.set_alpha(1.0)

    fig.tight_layout()
    _save(fig, "fig2b_dynamics_vs_landscape_similarity")
    plt.close(fig)


# ============================================================================
# Figure 3 - main confirmatory convergence result
# ============================================================================
def fig3():
    """Source: results/reference_robustness/processed/candidate_test_scores.csv - the
    confirmatory campaign's (Part 21, 20 seeds, 120 targets) incumbent trajectories
    RESCORED against the dense/independent offline reference (`generate_offline_reference.py`
    + `rescore_bo_trajectories.py`; robustness follow-up, 2026-09-16), not the original
    32-point search/score-leaking oracle. `regret_star_test = (best_test_so_far -
    J_ref_test) / J_ref_test` - same `oracle_relative_regret` formula as before
    (`src/optimization/regret.py`), just with both terms coming from the stronger
    reference/independent test bank. The rescored numbers reproduce the original
    (weak-reference) result to within noise (see docs/phase2_diagnostic_findings.md) -
    this swap is purely a robustness/defensibility improvement, not a changed finding."""
    scores_df = pd.read_csv(_ROOT / "results/reference_robustness/processed/candidate_test_scores.csv")
    scores_df = scores_df.dropna(subset=["best_test_so_far", "J_ref_test"])
    scores_df["regret_star_test"] = (scores_df["best_test_so_far"] - scores_df["J_ref_test"]) / scores_df["J_ref_test"]

    fig, ax = plt.subplots(figsize=DOUBLE_COL)
    for method in ("independent", "global", "similarity"):
        sub = scores_df[scores_df["method"] == method]
        agg = sub.groupby("n_local_evals")["regret_star_test"].agg(
            median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75)
        )
        ax.plot(
            agg.index, agg["median"], label=METHOD_LABELS[method], color=METHOD_COLORS[method], linewidth=MAIN_LW,
            marker="o", markersize=3,
        )
        ax.fill_between(agg.index, agg["q25"], agg["q75"], color=METHOD_COLORS[method], alpha=BAND_ALPHA, linewidth=0)
    ax.axhline(0.0, color="black", linewidth=0.6, linestyle=":", alpha=0.6)
    ax.set_xlim(2, 12)
    ax.set_xticks([2, 4, 6, 8, 10, 12])
    ax.set_xlabel(r"New-client experiments $n$")
    ax.set_ylabel(r"Reference-relative regret $r_i^{\mathrm{ref}}(n)$")
    ax.text(
        0.97, 0.60, "54% fewer experiments\non average (mean $N_{5\\%}$)", transform=ax.transAxes, fontsize=7,
        ha="right", va="top", color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", linewidth=0.6),
    )
    leg = ax.legend(loc="upper right", framealpha=1.0, edgecolor="#999999", handletextpad=0.4, borderpad=0.4)
    leg.get_frame().set_linewidth(0.6)

    fig.tight_layout()
    _save(fig, "fig3_confirmatory_convergence")
    plt.close(fig)


# ============================================================================
# Figure 4a - ECDF of N_5%
# ============================================================================
def fig4a():
    """Source: results/reference_robustness/processed/n5pct_test.csv - `N_5pct`
    recomputed against the dense/independent offline reference (see fig3's docstring;
    reproduces the original target_summary.parquet-based numbers to within noise).

    Drawn as a proper right-censored curve, not a plain ECDF of `fillna(budget)` values.
    With `n5pct_test` == NaN for a run that never reached 5% within the budget,
    imputing `budget` and taking the ECDF of THAT (the original version of this figure)
    forces the curve to y=1.0 at x=budget for every method BY CONSTRUCTION, regardless
    of the true reach rate - a censored ("never observed to succeed") run is then
    visually indistinguishable from a genuine success at exactly n=budget. Since every
    run here IS followed to the same fixed budget (no early stopping), the correct curve
    only steps up on OBSERVED successes; a method with `frac_reached < 1` (e.g.
    `independent`) stays visibly below y=1.0 at x=budget, with an open marker flagging
    the unresolved fraction rather than the line silently implying 100% eventual
    success."""
    summary_df = pd.read_csv(_ROOT / "results/reference_robustness/processed/n5pct_test.csv")
    budget = 12

    fig, ax = plt.subplots(figsize=SINGLE_COL)
    for method in ("independent", "global", "similarity"):
        sub = summary_df[summary_df["method"] == method]
        n_total = len(sub)
        observed = np.sort(sub["n5pct_test"].dropna().to_numpy())
        frac_reached = len(observed) / n_total if n_total else float("nan")
        y = np.arange(1, len(observed) + 1) / n_total
        if len(observed):
            ax.step(observed, y, where="post", label=METHOD_LABELS[method], color=METHOD_COLORS[method],
                     linewidth=MAIN_LW)
            # Dotted closure out to the budget at the last OBSERVED height - never
            # implies a success occurred between the last observed success and budget.
            ax.plot([observed[-1], budget], [frac_reached, frac_reached], color=METHOD_COLORS[method],
                     linewidth=MAIN_LW, linestyle=":")
        if frac_reached < 1.0:
            ax.plot(budget, frac_reached, marker="o", markerfacecolor="white",
                     markeredgecolor=METHOD_COLORS[method], markersize=4, zorder=5, clip_on=False)
    for x_ref in (3, 5):
        ax.axvline(x_ref, color="#999999", linewidth=1.0, linestyle="--", alpha=0.8, zorder=0)
    ax.set_xlabel(r"$N_{5\%}$ (new-client experiments" + "\nto reach 5% of reference)", fontsize=7.5)
    ax.set_ylabel("Fraction of new clients")
    ax.set_xlim(2, budget)
    ax.set_xticks([2, 4, 6, 8, 10, 12])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, 1.18)
    leg = ax.legend(loc="upper left", framealpha=1.0, handletextpad=0.4, borderpad=0.4, fontsize=7)

    fig.tight_layout()
    _save(fig, "fig4a_n5pct_ecdf")
    plt.close(fig)


# ============================================================================
# Figure 4b - fixed-budget success probability
# ============================================================================
def fig4b():
    """Source: results/reference_robustness/processed/n5pct_test.csv (see fig3's
    docstring - rescored against the dense/independent offline reference)."""
    n5 = pd.read_csv(_ROOT / "results/reference_robustness/processed/n5pct_test.csv")
    n5["le3"], n5["le5"] = n5["N_5pct_test"] <= 3, n5["N_5pct_test"] <= 5
    probs = n5.groupby("method")[["le3", "le5"]].mean().reindex(["independent", "global", "similarity"])

    fig, ax = plt.subplots(figsize=SINGLE_COL)
    x = np.arange(2)
    width = 0.24
    for mi, method in enumerate(["independent", "global", "similarity"]):
        bars = ax.bar(
            x + (mi - 1) * width, probs.loc[method, ["le3", "le5"]].to_numpy(), width, color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([r"$P(N_{5\%}\leq 3)$", r"$P(N_{5\%}\leq 5)$"])
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.12)
    leg = ax.legend(loc="upper left", framealpha=1.0, handletextpad=0.4, borderpad=0.4, fontsize=7)

    fig.tight_layout()
    _save(fig, "fig4b_n5pct_success_probability")
    plt.close(fig)


# ============================================================================
# Figure 5a - held-out comparison: global vs. dynamics-weighted (seed-level means)
# ============================================================================
def fig5a():
    """Source: results/reference_robustness/processed/n5pct_test.csv (see fig3's
    docstring - rescored against the dense/independent offline reference). Points are
    PER-SEED MEANS (mean N_5% over the 6 targets in that fleet realization), not
    per-target values - one point per confirmatory seed."""
    n5 = pd.read_csv(_ROOT / "results/reference_robustness/processed/n5pct_test.csv")
    seed_means = n5[n5["method"].isin(["global", "similarity"])].groupby(["seed", "method"])["N_5pct_test"].mean().unstack("method")

    lo = min(seed_means["global"].min(), seed_means["similarity"].min()) - 0.3
    hi = max(seed_means["global"].max(), seed_means["similarity"].max()) + 0.3

    fig, ax = plt.subplots(figsize=(SINGLE_COL[0], SINGLE_COL[0]))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--", zorder=0)
    ax.scatter(seed_means["global"], seed_means["similarity"], s=24, alpha=0.6, color="#0072B2", linewidths=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Mean $N_{5\%}$ per fleet realization, global")
    ax.set_ylabel(r"Mean $N_{5\%}$ per fleet realization," + "\ndynamics-weighted")

    fig.tight_layout()
    _save(fig, "fig5a_global_vs_weighted_parity")
    plt.close(fig)


# ============================================================================
# Figure 5b - source-coverage sensitivity
# ============================================================================
def fig5b():
    """Source: phase2_coverage_dev_c1cds02/processed/coverage_summary.parquet (Part 18).
    Error bars: 95% bootstrap CI of the mean, over the 20 development seeds (per-seed
    mean N_5% across its targets/replicates for that q_same/method cell)."""
    coverage_df = pd.read_parquet(_ROOT / "results/phase2_coverage_dev_c1cds02/processed/coverage_summary.parquet")
    cov_budget = 12
    cov = coverage_df.copy()
    cov["N_5pct"] = cov["evals_to_5pct_of_oracle"].fillna(cov_budget)

    q_values = sorted(q for q in cov["q_same"].unique() if q >= 0)

    fig, ax = plt.subplots(figsize=SINGLE_COL)
    series = [
        ("global", METHOD_COLORS["global"], "-", "o", "Global warm start"),
        ("similarity", METHOD_COLORS["similarity"], "-", "o", "Dynamics-weighted warm start"),
        ("same_family_reference", "#999999", ":", "s", "Same-family-only reference"),
    ]
    for method, color, ls, marker, label in series:
        means, los, his = [], [], []
        for q in q_values:
            if method == "same_family_reference" and q == 0:
                means.append(np.nan); los.append(np.nan); his.append(np.nan)
                continue
            per_seed = cov.loc[(cov["q_same"] == q) & (cov["method"] == method)].groupby("seed")["N_5pct"].mean().to_numpy()
            m, lo, hi = _seed_bootstrap_ci(per_seed)
            means.append(m); los.append(lo); his.append(hi)
        means, los, his = np.array(means), np.array(los), np.array(his)
        yerr = np.vstack([means - los, his - means])
        ax.errorbar(
            q_values, means, yerr=yerr, color=color, linestyle=ls, marker=marker, markersize=4.5, linewidth=2.0,
            capsize=2.5, elinewidth=1.0, label=label,
        )
    ax.set_xticks(q_values)
    ax.set_xlabel(r"$q_{\mathrm{same}}$ (same-family peers in 9-peer pool)")
    ax.set_ylabel(r"Mean $N_{5\%}$")
    leg = ax.legend(loc="upper right", framealpha=1.0, fontsize=6.5, handletextpad=0.4, borderpad=0.4)

    fig.tight_layout()
    _save(fig, "fig5b_coverage_sensitivity")
    plt.close(fig)


# ============================================================================
# Figure 6 - lane-change yaw-rate reference trajectories
# ============================================================================
def fig6():
    """Pure function of the project's fixed maneuver parameters
    (config/fleet_fixed_speed.yaml's lane_change_time, config/controller.yaml's
    Ts/T_total/T0/r_max/tfilter) - not simulated data, just the reference generator
    every episode bank in this project calls (src.ifac_bridge.lc_yaw_rate_ref)."""
    from src.ifac_bridge import lc_yaw_rate_ref

    Ts, T_total, T0, r_max, tfilter = 0.05, 6.0, 1.0, 0.6108652381980153, 0.1

    fig, ax = plt.subplots(figsize=SINGLE_COL)
    for T_lc, color, ls, label in (
        (1.0, "#0072B2", "-", r"$T_{\mathrm{lc}}=1.0\,\mathrm{s}$"),
        (2.0, "#D55E00", "--", r"$T_{\mathrm{lc}}=2.0\,\mathrm{s}$"),
    ):
        t, r = lc_yaw_rate_ref(Ts, T_total, T0, T_lc, r_max, tfilter=tfilter)
        ax.plot(t, r, color=color, linestyle=ls, linewidth=MAIN_LW, label=label)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Yaw-rate reference $r^{\mathrm{ref}}$ [rad/s]")
    ax.legend(loc="upper right", framealpha=1.0, handletextpad=0.4, borderpad=0.4)

    fig.tight_layout()
    _save(fig, "fig6_lane_change_reference")
    plt.close(fig)


if __name__ == "__main__":
    fig2a()
    fig2b()
    fig3()
    fig4a()
    fig4b()
    fig5a()
    fig5b()
    fig6()
