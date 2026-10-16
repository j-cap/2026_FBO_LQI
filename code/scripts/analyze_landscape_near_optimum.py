"""Analyze BO-relevant local landscape overlap from the frozen post-hoc study.

This script performs NO simulation. It reads the already-computed pair table from
calibration_landscape_similarity.py and summarizes two diagnostics that complement the
global Spearman rank correlation:

1) overlap_10pct:
   Jaccard overlap of each system's best 13 feasible Sobol points (13 = ceil(10% of
   the common 128-point grid)). This asks whether the near-best regions in calibration
   space coincide, not merely whether the full feasible landscapes are monotonically
   related.

2) s_ij_feas:
   Jaccard overlap of the full feasible-point sets. This asks whether two systems agree
   on which calibration points are admissible.

The landscape study is post-hoc and contains only five development-fleet realizations.
Accordingly, bootstrap intervals below are explicitly descriptive uncertainty summaries
over the five fleet-level medians, not confirmatory population inference.

Usage (from code/):
    python scripts/analyze_landscape_near_optimum.py

Optional:
    python scripts/analyze_landscape_near_optimum.py --input <pairs.csv> --output-tag <tag>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT = (
    _ROOT
    / "results"
    / "calibration_landscape_similarity"
    / "processed"
    / "landscape_similarity_pairs.csv"
)
DEFAULT_OUTPUT_TAG = "calibration_landscape_similarity"

N_BOOT = 10_000
BOOT_SEED = 0


def _bootstrap_median_ci(values: np.ndarray) -> tuple[float, float, float]:
    """Median and descriptive percentile-bootstrap 95% interval."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(BOOT_SEED)
    boot = np.empty(N_BOOT, dtype=float)
    for b in range(N_BOOT):
        boot[b] = np.median(rng.choice(vals, size=len(vals), replace=True))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(np.median(vals)), float(lo), float(hi)


def _pooled_summary(df: pd.DataFrame, metric: str, label: str) -> dict:
    x = df[metric].dropna().to_numpy(dtype=float)
    return {
        "metric": metric,
        "label": label,
        "scope": "all_pairs",
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "q05": float(np.quantile(x, 0.05)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "q95": float(np.quantile(x, 0.95)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _family_summary(df: pd.DataFrame, metric: str, label: str, same: bool) -> dict:
    x = df.loc[df["same_family"] == same, metric].dropna().to_numpy(dtype=float)
    return {
        "metric": metric,
        "label": label,
        "scope": "same_family" if same else "cross_family",
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "q05": float(np.quantile(x, 0.05)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "q95": float(np.quantile(x, 0.95)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _fleet_summary(df: pd.DataFrame, metric: str, label: str) -> tuple[dict, pd.DataFrame]:
    per_seed = (
        df.dropna(subset=[metric])
        .groupby("seed")[metric]
        .median()
        .rename("fleet_median")
        .reset_index()
    )
    med, lo, hi = _bootstrap_median_ci(per_seed["fleet_median"].to_numpy())
    row = {
        "metric": metric,
        "label": label,
        "scope": "fleet_medians",
        "n": int(len(per_seed)),
        "mean": float(per_seed["fleet_median"].mean()),
        "median": med,
        "q05": float(per_seed["fleet_median"].quantile(0.05)),
        "q25": float(per_seed["fleet_median"].quantile(0.25)),
        "q75": float(per_seed["fleet_median"].quantile(0.75)),
        "q95": float(per_seed["fleet_median"].quantile(0.95)),
        "min": float(per_seed["fleet_median"].min()),
        "max": float(per_seed["fleet_median"].max()),
        "bootstrap95_low_descriptive": lo,
        "bootstrap95_high_descriptive": hi,
    }
    per_seed.insert(1, "metric", metric)
    per_seed.insert(2, "label", label)
    return row, per_seed


def _threshold_rows(df: pd.DataFrame) -> list[dict]:
    specs = {
        "overlap_10pct": [0.25, 0.50, 0.75],
        "s_ij_feas": [0.80, 0.90, 0.95],
    }
    rows: list[dict] = []
    for metric, thresholds in specs.items():
        valid = df.dropna(subset=[metric])
        for threshold in thresholds:
            for scope, sub in [
                ("all_pairs", valid),
                ("same_family", valid[valid["same_family"]]),
                ("cross_family", valid[~valid["same_family"]]),
            ]:
                rows.append(
                    {
                        "metric": metric,
                        "scope": scope,
                        "threshold": threshold,
                        "fraction_at_or_above": float((sub[metric] >= threshold).mean()),
                        "n": int(len(sub)),
                    }
                )
    return rows


def main(input_path: Path, output_tag: str) -> None:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing frozen landscape pair file: {input_path}\n"
            "Restore/copy the archived calibration_landscape_similarity result folder locally; "
            "this analysis intentionally does not rerun the simulation."
        )

    df = pd.read_csv(input_path)
    required = {
        "seed",
        "same_family",
        "rho_J",
        "overlap_10pct",
        "s_ij_feas",
        "n_joint_feasible",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {sorted(missing)}")

    n_seeds = df["seed"].nunique()
    if n_seeds != 5:
        raise ValueError(
            f"Expected the frozen five-fleet post-hoc study, found {n_seeds} seeds: "
            f"{sorted(df['seed'].unique().tolist())}"
        )

    metrics = [
        (
            "overlap_10pct",
            "Jaccard overlap of best 13 feasible points (10% of 128-point grid)",
        ),
        ("s_ij_feas", "Jaccard overlap of feasible-point sets"),
    ]

    rows: list[dict] = []
    fleet_frames: list[pd.DataFrame] = []
    for metric, label in metrics:
        rows.append(_pooled_summary(df, metric, label))
        rows.append(_family_summary(df, metric, label, True))
        rows.append(_family_summary(df, metric, label, False))
        fleet_row, fleet_df = _fleet_summary(df, metric, label)
        rows.append(fleet_row)
        fleet_frames.append(fleet_df)

    summary = pd.DataFrame(rows)
    thresholds = pd.DataFrame(_threshold_rows(df))
    fleet_medians = pd.concat(fleet_frames, ignore_index=True)

    out_dir = _ROOT / "results" / output_tag / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "near_optimum_overlap_summary.csv"
    threshold_path = out_dir / "near_optimum_overlap_thresholds.csv"
    fleet_path = out_dir / "near_optimum_overlap_fleet_medians.csv"
    summary.to_csv(summary_path, index=False)
    thresholds.to_csv(threshold_path, index=False)
    fleet_medians.to_csv(fleet_path, index=False)

    print("\nBO-relevant landscape-overlap diagnostics")
    print("=" * 96)
    for metric, label in metrics:
        print(f"\n{label}")
        print("-" * 96)
        show = summary[summary["metric"] == metric][
            [
                "scope",
                "n",
                "mean",
                "median",
                "q05",
                "q25",
                "q75",
                "q95",
                "min",
                "max",
            ]
        ]
        print(show.round(4).to_string(index=False))
        fleet = summary[
            (summary["metric"] == metric) & (summary["scope"] == "fleet_medians")
        ].iloc[0]
        print(
            "descriptive 95% bootstrap interval of the median fleet-level median: "
            f"({fleet['bootstrap95_low_descriptive']:.4f}, "
            f"{fleet['bootstrap95_high_descriptive']:.4f}); n_fleets=5"
        )

    print("\nThreshold fractions")
    print("=" * 96)
    print(thresholds.round(4).to_string(index=False))

    print("\nWrote:")
    print(f"  {summary_path}")
    print(f"  {threshold_path}")
    print(f"  {fleet_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Frozen landscape_similarity_pairs.csv produced by calibration_landscape_similarity.py",
    )
    parser.add_argument(
        "--output-tag",
        default=DEFAULT_OUTPUT_TAG,
        help="results/<tag>/processed output directory; default writes beside the frozen landscape package.",
    )
    args = parser.parse_args()
    main(args.input, args.output_tag)
