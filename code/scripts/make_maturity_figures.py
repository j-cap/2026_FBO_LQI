"""
Publication figures for the fleet-maturity sensitivity analysis.

Reads the dense-reference rescoring output produced by rescore_maturity_sweep.py and
writes vector PDFs plus 600-dpi PNG fallbacks directly to ../latex/figures/.

Main paper figure:
  source maturity -> budget-restricted mean N_5% with realization-level 95% CIs.

The paired-savings and early-success views are retained as supplementary diagnostics.

All uncertainty intervals use the fleet realization as the resampling unit. Target
systems within the same realization are first averaged, avoiding visual
pseudo-replication of the six targets from one generated fleet.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = _ROOT.parent
FIG_DIR = REPO_ROOT / "latex" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SINGLE_COL = (3.45, 2.55)
MAIN_LW = 2.0
N_BOOT = 10_000
BOOTSTRAP_SEED = 0

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _save(fig, stem: str):
    for ext in ("pdf", "png"):
        path = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(
            path,
            bbox_inches="tight",
            pad_inches=0.03,
            transparent=False,
            dpi=600 if ext == "png" else None,
        )
        print(f"wrote {path}")


def _load(analysis_tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed = _ROOT / "results" / analysis_tag / "processed"
    summary_path = processed / "maturity_summary.csv"
    n5_path = processed / "maturity_n5pct_test.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"{summary_path} does not exist. Run scripts/rescore_maturity_sweep.py first."
        )
    if not n5_path.exists():
        raise FileNotFoundError(
            f"{n5_path} does not exist. Run scripts/rescore_maturity_sweep.py first."
        )
    summary = pd.read_csv(summary_path).sort_values("maturity")
    n5 = pd.read_csv(n5_path)
    return summary, n5


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Percentile-bootstrap 95% CI of a mean over independent fleet realizations."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("bootstrap input must be a non-empty one-dimensional array")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def _absolute_mean_ci(n5: pd.DataFrame, *, method: str, maturity: int) -> tuple[float, float, float]:
    """Fleet-realization bootstrap CI after averaging the six targets within each seed."""
    sub = n5[(n5["method"] == method) & (n5["maturity"] == maturity)]
    per_seed = sub.groupby("seed")["N_5pct_test"].mean().to_numpy()
    return _bootstrap_mean_ci(per_seed)


def fig_maturity_absolute(summary: pd.DataFrame, n5: pd.DataFrame):
    fig, ax = plt.subplots(figsize=SINGLE_COL)
    x = summary["maturity"].to_numpy(dtype=float)
    total_hist = summary["n_hist_total"].to_numpy(dtype=int)

    global_mean, global_lo, global_hi = [], [], []
    for maturity in summary["maturity"].astype(int):
        m, lo, hi = _absolute_mean_ci(n5, method="global", maturity=maturity)
        global_mean.append(m)
        global_lo.append(lo)
        global_hi.append(hi)
    global_mean = np.asarray(global_mean)
    global_lo = np.asarray(global_lo)
    global_hi = np.asarray(global_hi)
    global_yerr = np.vstack([global_mean - global_lo, global_hi - global_mean])

    independent, independent_lo, independent_hi = _absolute_mean_ci(
        n5, method="independent", maturity=0
    )

    ax.errorbar(
        x,
        global_mean,
        yerr=global_yerr,
        marker="o",
        linewidth=MAIN_LW,
        capsize=3,
        label="Fleet warm start",
    )
    ax.axhline(
        independent,
        color="0.45",
        linestyle="--",
        linewidth=1.4,
        label="Independent BO",
    )
    ax.axhspan(independent_lo, independent_hi, color="0.45", alpha=0.10, linewidth=0)

    # Highlight the practically relevant knee using the stored paired-savings analysis.
    savings = -summary["delta_global_minus_independent"].to_numpy(dtype=float)
    idx_10 = np.flatnonzero(summary["maturity"].to_numpy(dtype=int) == 10)
    idx_25 = np.flatnonzero(summary["maturity"].to_numpy(dtype=int) == 25)
    if len(idx_10) == 1 and len(idx_25) == 1 and savings[idx_25[0]] > 0:
        share = 100.0 * savings[idx_10[0]] / savings[idx_25[0]]
        x10 = float(x[idx_10[0]])
        y10 = float(global_mean[idx_10[0]])
        ax.annotate(
            f"{share:.0f}% of mature-fleet saving",
            xy=(x10, y10),
            xytext=(12.0, y10 + 1.0),
            fontsize=7,
            ha="left",
            va="center",
            arrowprops=dict(arrowstyle="->", linewidth=0.7, color="0.35"),
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.8", linewidth=0.6),
        )

    ax.set_xticks(x)
    ax.set_xlabel(r"Historical evaluations per source system $M_{\mathrm{hist}}$")
    ax.set_ylabel(r"Restricted mean $N_{5\%}$")
    ax.legend(loc="best", framealpha=1.0)

    # Secondary axis reports the actual number of historical scalar records available
    # to each target. The primary axis remains M_hist because repeated source
    # initialization points are not equivalent to distinct calibration-space locations.
    top = ax.secondary_xaxis("top")
    top.set_xticks(x)
    top.set_xticklabels([str(v) for v in total_hist])
    top.set_xlabel("Total source observations", fontsize=8)
    top.tick_params(labelsize=7)

    fig.tight_layout()
    _save(fig, "fig7a_fleet_maturity_absolute")
    plt.close(fig)


def fig_maturity_savings(summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=SINGLE_COL)
    x = summary["maturity"].to_numpy(dtype=float)

    # Store paired deltas in the analysis as global - independent. Plot the sign-flipped
    # quantity here so positive values directly mean local experiments saved.
    savings = -summary["delta_global_minus_independent"].to_numpy(dtype=float)
    savings_lo = -summary["ci95_high"].to_numpy(dtype=float)
    savings_hi = -summary["ci95_low"].to_numpy(dtype=float)
    yerr = np.vstack([savings - savings_lo, savings_hi - savings])

    ax.axhline(0.0, color="0.25", linestyle="--", linewidth=1.0)
    ax.errorbar(x, savings, yerr=yerr, marker="o", linewidth=MAIN_LW, capsize=3)
    ax.set_xticks(x)
    ax.set_xlabel(r"Historical evaluations per source system $M_{\mathrm{hist}}$")
    ax.set_ylabel(r"Mean target evaluations saved")
    fig.tight_layout()
    _save(fig, "fig7b_fleet_maturity_savings")
    plt.close(fig)


def fig_maturity_early_success(summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=SINGLE_COL)
    x = summary["maturity"].to_numpy(dtype=float)
    fleet = summary["p_reach_5_global"].to_numpy(dtype=float)
    independent = float(summary["p_reach_5_independent"].iloc[0])

    ax.plot(x, fleet, marker="o", linewidth=MAIN_LW, label="Fleet warm start")
    ax.axhline(
        independent,
        color="0.45",
        linestyle="--",
        linewidth=1.4,
        label="Independent BO",
    )
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"Historical evaluations per source system $M_{\mathrm{hist}}$")
    ax.set_ylabel(r"$P(T_{5\%}\leq 5)$")
    ax.legend(loc="best", framealpha=1.0)
    fig.tight_layout()
    _save(fig, "fig7c_fleet_maturity_early_success")
    plt.close(fig)


def main(analysis_tag: str):
    summary, n5 = _load(analysis_tag)
    fig_maturity_absolute(summary, n5)
    fig_maturity_savings(summary)
    fig_maturity_early_success(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-tag",
        default="fleet_maturity_sweep_c1cds02_reference_robustness",
        help="results/<tag> containing the maturity postprocessing CSV files",
    )
    args = parser.parse_args()
    main(args.analysis_tag)
