"""
Publication figures for the fleet-maturity sensitivity analysis.

Reads the dense-reference rescoring output produced by rescore_maturity_sweep.py and
writes vector PDFs plus 600-dpi PNG fallbacks directly to ../latex/figures/.

Main paper figure:
  (a) source maturity -> budget-restricted mean N_5%
  (b) source maturity -> paired experiments saved vs. independent BO

A third secondary figure shows P(T_5% <= 3) versus source maturity.
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
        fig.savefig(path, bbox_inches="tight", pad_inches=0.03, transparent=False, dpi=600 if ext == "png" else None)
        print(f"wrote {path}")


def _load(analysis_tag: str):
    path = _ROOT / "results" / analysis_tag / "processed" / "maturity_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run scripts/rescore_maturity_sweep.py first."
        )
    return pd.read_csv(path).sort_values("maturity")


def fig_maturity_absolute(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=SINGLE_COL)
    x = df["maturity"].to_numpy()
    global_mean = df["mean_N5_global"].to_numpy()
    independent = float(df["mean_N5_independent"].iloc[0])

    ax.plot(x, global_mean, marker="o", linewidth=MAIN_LW, label="Fleet warm start")
    ax.axhline(independent, color="0.45", linestyle="--", linewidth=1.4, label="Independent BO")
    ax.set_xticks(x)
    ax.set_xlabel(r"Historical evaluations per source system $M_{\mathrm{hist}}$")
    ax.set_ylabel(r"Restricted mean $N_{5\%}$")
    ax.legend(loc="best", framealpha=1.0)
    fig.tight_layout()
    _save(fig, "fig7a_fleet_maturity_absolute")
    plt.close(fig)


def fig_maturity_savings(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=SINGLE_COL)
    x = df["maturity"].to_numpy()
    delta = df["delta_global_minus_independent"].to_numpy()
    lo = df["ci95_low"].to_numpy()
    hi = df["ci95_high"].to_numpy()
    yerr = np.vstack([delta - lo, hi - delta])

    ax.axhline(0.0, color="0.25", linestyle="--", linewidth=1.0)
    ax.errorbar(x, delta, yerr=yerr, marker="o", linewidth=MAIN_LW, capsize=3)
    ax.set_xticks(x)
    ax.set_xlabel(r"Historical evaluations per source system $M_{\mathrm{hist}}$")
    ax.set_ylabel(r"$\Delta N_{5\%}$: fleet $-$ independent")
    fig.tight_layout()
    _save(fig, "fig7b_fleet_maturity_savings")
    plt.close(fig)


def fig_maturity_early_success(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=SINGLE_COL)
    x = df["maturity"].to_numpy()
    fleet = df["p_reach_3_global"].to_numpy()
    independent = float(df["p_reach_3_independent"].iloc[0])

    ax.plot(x, fleet, marker="o", linewidth=MAIN_LW, label="Fleet warm start")
    ax.axhline(independent, color="0.45", linestyle="--", linewidth=1.4, label="Independent BO")
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"Historical evaluations per source system $M_{\mathrm{hist}}$")
    ax.set_ylabel(r"$P(T_{5\%}\leq 3)$")
    ax.legend(loc="best", framealpha=1.0)
    fig.tight_layout()
    _save(fig, "fig7c_fleet_maturity_early_success")
    plt.close(fig)


def main(analysis_tag: str):
    df = _load(analysis_tag)
    fig_maturity_absolute(df)
    fig_maturity_savings(df)
    fig_maturity_early_success(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-tag",
        default="fleet_maturity_sweep_c1cds02_reference_robustness",
        help="results/<tag> containing processed/maturity_summary.csv",
    )
    args = parser.parse_args()
    main(args.analysis_tag)
