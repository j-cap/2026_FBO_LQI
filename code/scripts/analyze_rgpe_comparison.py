"""Combine the frozen three-method confirmatory results with the RGPE baseline.

Inputs are the dense-reference rescored CSVs produced by
scripts/rescore_bo_trajectories.py.  Fleet realization is the replication unit,
matching the manuscript's frozen statistical convention.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

import pandas as pd  # noqa: E402

from phase2_triggered_dev import _seed_bootstrap_mean_ci  # noqa: E402


def _load(tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = _ROOT / "results" / tag / "processed"
    n5 = pd.read_csv(base / "n5pct_test.csv")
    scores = pd.read_csv(base / "candidate_test_scores.csv")
    return n5, scores


def main(
    baseline_tag: str,
    rgpe_tag: str,
    output_tag: str,
) -> None:
    base_n5, base_scores = _load(baseline_tag)
    rgpe_n5, rgpe_scores = _load(rgpe_tag)

    if set(rgpe_n5["method"].unique()) != {"rgpe"}:
        raise ValueError(
            f"Expected only method='rgpe' in {rgpe_tag}, got "
            f"{sorted(rgpe_n5['method'].unique())}"
        )

    combined_n5 = pd.concat([base_n5, rgpe_n5], ignore_index=True)
    combined_scores = pd.concat([base_scores, rgpe_scores], ignore_index=True)

    out = _ROOT / "results" / output_tag / "processed"
    out.mkdir(parents=True, exist_ok=True)
    combined_n5.to_csv(out / "combined_n5pct_test.csv", index=False)
    combined_scores.to_csv(out / "combined_candidate_test_scores.csv", index=False)

    summary = (
        combined_n5.groupby("method")
        .agg(
            restricted_mean_N5=("N_5pct_test", "mean"),
            median_N5=("N_5pct_test", "median"),
            std_N5=("N_5pct_test", "std"),
            success_within_budget=("reached", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out / "method_summary.csv", index=False)

    print("\nDense-reference comparison")
    print("=" * 90)
    print(summary.round(4).to_string(index=False))

    work = combined_n5.copy()
    work["le3"] = work["N_5pct_test"] <= 3
    work["le5"] = work["N_5pct_test"] <= 5
    fixed = work.groupby("method")[["le3", "le5"]].mean()
    print("\nFixed-budget success:")
    print(fixed.round(4))

    comparisons = [
        ("rgpe", "independent"),
        ("rgpe", "global"),
        ("rgpe", "similarity"),
    ]
    delta_rows = []
    print("\nPaired fleet-realization deltas (negative = RGPE needs fewer evaluations)")
    print("=" * 90)
    for a, b in comparisons:
        piv = work[work["method"].isin([a, b])].pivot_table(
            index=["seed", "client_id"],
            columns="method",
            values="N_5pct_test",
        )
        if a not in piv.columns or b not in piv.columns:
            raise ValueError(f"Missing method columns for {a} vs {b}")
        piv = piv.dropna(subset=[a, b])
        piv["delta"] = piv[a] - piv[b]
        per_seed = piv.groupby("seed")["delta"].mean().to_dict()
        res = _seed_bootstrap_mean_ci(per_seed)
        row = {
            "method_a": a,
            "method_b": b,
            "mean_delta": res["mean"],
            "ci95_low": res["bootstrap_ci_95"][0],
            "ci95_high": res["bootstrap_ci_95"][1],
            "n_negative": res["n_negative"],
            "n_seeds": res["n_seeds"],
            "wilcoxon_p": res["wilcoxon_p"],
        }
        delta_rows.append(row)
        print(
            f"{a} - {b}: mean delta={row['mean_delta']:.3f}, "
            f"95% CI=({row['ci95_low']:.3f},{row['ci95_high']:.3f}), "
            f"{row['n_negative']}/{row['n_seeds']} seeds favorable, "
            f"Wilcoxon p={row['wilcoxon_p']:.4f}"
        )

    pd.DataFrame(delta_rows).to_csv(out / "paired_deltas.csv", index=False)
    print(f"\nWrote combined analysis to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-tag", default="reference_robustness")
    parser.add_argument("--rgpe-tag", default="reference_robustness_rgpe")
    parser.add_argument("--output-tag", default="rgpe_comparison")
    args = parser.parse_args()
    main(args.baseline_tag, args.rgpe_tag, args.output_tag)
