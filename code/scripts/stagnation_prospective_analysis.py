"""
Seed-aware and prospective reanalysis of the stagnation-predicts-benefit mechanism
(user request, 2026-09-04, following Part 10's confirmatory campaign result -
docs/phase2_diagnostic_findings.md). Pure post-hoc analysis of the ALREADY-COMPLETED
`phase2_fbo_comparison_campaign_v3_c1cds05` traces - no new simulation.

Two concerns raised about Part 6/10's pooled rho(D_i, G_i)=0.218 (n=600, p=0.0000):
  1. The 600 clients are nested inside 20 fleet seeds (the pre-registered replication
     unit) - the pooled p-value is too optimistic unless clustering is accounted for.
  2. D_i and G_i are both computed over/from the SAME window (through n=20) - a client
     whose independent search stagnates mechanically has more room for a large AURC gap,
     which isn't the same claim as "stagnation PREDICTS later benefit."

This script:
  (1) Reanalyses the ORIGINAL (retrospective, cutoff=20) stagnation check at seed level -
      per-seed rho_s, median/bootstrap-CI-over-seeds/count-positive, plus a
      family-controlled version (residualizing D_i/G_i within each seed x family cell).
      Also switches G_i to the PRIMARY regret_star-based AURC_star (Part 9/10), not the
      old regret-based AURC Part 6 originally used.
  (2) Runs the PROSPECTIVE test the user specified: for cutoffs c in {11, 13, 15},
      D_i(c) (stagnation using ONLY evaluations up to c) vs. G_i^future(c) = sum over
      n in (c, 25] of [regret_star_independent(n) - regret_star_similarity(n)] - a
      genuine leading-indicator test, not same-window post-hoc association.
  (3) Compares D_i(c) against two more discriminative online stagnation variables at
      the same cutoffs: L_i(c) (current no-improvement streak) and Delta_i^(h=3)(c)
      (relative improvement over a fixed 3-evaluation window) - retrospectively, no
      threshold sweep.

Usage:
    python scripts/stagnation_prospective_analysis.py --config config/runs/phase2_fbo_comparison_campaign_v3_c1cds05.yaml
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
from scipy import stats  # noqa: E402

from src.io_utils import load_run_config  # noqa: E402

from phase2_medium_analysis import _family_of, compute_aurc, compute_stagnation  # noqa: E402

PROSPECTIVE_CUTOFFS = (11, 13, 15)
FINAL_N = 25
WINDOW_H = 3  # for Delta_i^(h): c - h must stay >= n_init (8), holds exactly at c=11


def _seed_bootstrap_median_ci(per_seed_values: dict, *, n_boot: int = 10000, seed: int = 0) -> dict:
    """Percentile bootstrap over SEEDS (the pre-registered replication unit, Part 4) -
    same methodology as phase2_medium_analysis.paired_seed_stats, factored out here
    since this script needs it for several different per-seed statistics (rho_s at
    several cutoffs, not just one AURC delta)."""
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
        "n_seeds": len(vals),
        "median": float(np.median(vals)),
        "bootstrap_ci_95": (float(ci_lo), float(ci_hi)),
        "n_positive": int((vals > 0).sum()),
        "wilcoxon_p": float(wilcoxon_p),
    }


def per_seed_spearman(merged: pd.DataFrame, x_col: str, y_col: str) -> dict:
    """rho_s per seed - the seed-level analogue of a pooled Spearman correlation,
    computed separately within each seed's ~30 clients rather than pooling across
    seeds (concern 1 in the module docstring)."""
    out = {}
    for seed, g in merged.groupby("seed"):
        if g[x_col].nunique() < 2 or g[y_col].nunique() < 2:
            out[seed] = float("nan")
            continue
        rho, _ = stats.spearmanr(g[x_col], g[y_col])
        out[seed] = float(rho)
    return out


def compute_L_and_delta(trace_df: pd.DataFrame, *, method: str, cutoff: int, n_init: int, h: int) -> pd.DataFrame:
    """L_i(c): current no-improvement streak (in evaluation-count units) ending at
    `cutoff` - counts back from `cutoff` to the most recent checkpoint where a STRICT
    decrease in best_calibration_rmse occurred (the same "improvement" definition
    D_i/compute_stagnation already uses); if no improvement occurred anywhere in
    [n_init, cutoff], L_i = cutoff - n_init (stagnant for the client's entire window
    so far). Delta_i^(h): relative improvement over the fixed h-evaluation window
    ending at `cutoff`, (J(cutoff-h) - J(cutoff)) / J(cutoff-h) - requires
    cutoff - h >= n_init."""
    rows = []
    for (seed, cid), g in trace_df[trace_df["method"] == method].groupby(["seed", "client_id"]):
        g = g[(g["n_local_evals"] >= n_init) & (g["n_local_evals"] <= cutoff)].sort_values("n_local_evals")
        ns = g["n_local_evals"].to_numpy()
        vals = g["best_calibration_rmse"].to_numpy()
        if len(vals) < 2:
            continue
        improved = np.diff(vals) < 0  # improved[k] True means an improvement happened AT ns[k+1]
        improve_positions = np.nonzero(improved)[0] + 1  # indices into ns/vals where an improvement landed
        last_improve_idx = improve_positions[-1] if len(improve_positions) else 0
        L_i = float(ns[-1] - ns[last_improve_idx])

        delta_i = float("nan")
        target_n = cutoff - h
        if target_n >= n_init and target_n in ns:
            j_before = vals[ns == target_n][0]
            j_now = vals[-1]
            if j_before > 0:
                delta_i = float((j_before - j_now) / j_before)

        rows.append({"seed": seed, "client_id": cid, "L_i": L_i, "delta_i": delta_i})
    return pd.DataFrame(rows)


def compute_future_gain(trace_df: pd.DataFrame, *, cutoff: int, final_n: int = FINAL_N) -> pd.DataFrame:
    """G_i^future(c) = sum_{n=c+1}^{final_n} [regret_star_independent(n) - regret_star_similarity(n)]
    - a literal discrete sum over the recorded checkpoints strictly after `cutoff`
    (not a lookahead into D_i(c)'s own window, since D_i(c) only ever sees n<=c)."""
    ind = trace_df[(trace_df["method"] == "independent") & (trace_df["n_local_evals"] > cutoff) & (trace_df["n_local_evals"] <= final_n)]
    sim = trace_df[(trace_df["method"] == "similarity") & (trace_df["n_local_evals"] > cutoff) & (trace_df["n_local_evals"] <= final_n)]
    ind_sum = ind.groupby(["seed", "client_id"])["regret_star"].sum()
    sim_sum = sim.groupby(["seed", "client_id"])["regret_star"].sum()
    gain = (ind_sum - sim_sum).rename("G_future").reset_index()
    return gain


def report_association(merged: pd.DataFrame, x_col: str, y_col: str, *, label: str) -> None:
    merged = merged.dropna(subset=[x_col, y_col])
    pooled_rho, pooled_p = stats.spearmanr(merged[x_col], merged[y_col])
    print(f"  {label}: pooled Spearman rho={pooled_rho:.3f}, p={pooled_p:.4g}, n={len(merged)} (clients, NOT independent - see seed-level below)")
    rho_s = per_seed_spearman(merged, x_col, y_col)
    stats_s = _seed_bootstrap_median_ci(rho_s)
    print(
        f"    seed-level: median(rho_s)={stats_s['median']:.3f}, bootstrap 95% CI={stats_s['bootstrap_ci_95']}, "
        f"{stats_s['n_positive']}/{stats_s['n_seeds']} seeds positive, Wilcoxon p={stats_s['wilcoxon_p']:.4f}"
    )
    print(f"    per-seed rho_s: { {k: round(v, 3) for k, v in sorted(rho_s.items())} }")


def main(config_path: str) -> None:
    cfg = load_run_config(config_path)
    run_tag = cfg["run_tag"]
    out_dir = _ROOT / "results" / run_tag
    processed_dir = out_dir / "processed"
    trace_df = pd.read_parquet(processed_dir / "fbo_convergence_traces.parquet")
    n_init = cfg["n_init"]

    # ============================================================================
    # Part 1: retrospective (through n=20) seed-level reanalysis, PRIMARY metric
    # ============================================================================
    print("=" * 90)
    print("PART 1: retrospective stagnation (cutoff=20, matches Part 6) vs AURC_star gain - seed-level")
    print("=" * 90)
    aurc_star = compute_aurc(trace_df, metric_col="regret_star")
    aurc_ind = aurc_star[aurc_star["method"] == "independent"].set_index(["seed", "client_id"])["aurc"]
    aurc_sim = aurc_star[aurc_star["method"] == "similarity"].set_index(["seed", "client_id"])["aurc"]
    gain_star = (aurc_ind - aurc_sim).rename("G_i").reset_index()

    D_i_20 = compute_stagnation(trace_df, method="independent", cutoff=20)
    merged_retro = D_i_20.merge(gain_star, on=["seed", "client_id"])
    report_association(merged_retro, "D_i", "G_i", label="D_i(cutoff=20) vs G_i=AURC_star_ind-AURC_star_sim")

    print("\n  -- family-controlled (residualized within seed x family cells) --")
    merged_retro["family"] = merged_retro["client_id"].apply(_family_of)
    merged_retro["D_i_resid"] = merged_retro.groupby(["seed", "family"])["D_i"].transform(lambda s: s - s.mean())
    merged_retro["G_i_resid"] = merged_retro.groupby(["seed", "family"])["G_i"].transform(lambda s: s - s.mean())
    rho_resid, p_resid = stats.spearmanr(merged_retro["D_i_resid"], merged_retro["G_i_resid"])
    print(f"  pooled Spearman on (seed x family)-demeaned D_i, G_i: rho={rho_resid:.3f}, p={p_resid:.4g}, n={len(merged_retro)}")
    print("\n  -- per-family pooled (across all seeds) --")
    for fam, g in merged_retro.groupby("family"):
        rho_f, p_f = stats.spearmanr(g["D_i"], g["G_i"])
        print(f"    {fam}: rho={rho_f:.3f}, p={p_f:.4g}, n={len(g)}")

    merged_retro.to_csv(processed_dir / "stagnation_seed_level_retrospective.csv", index=False)
    print(f"\nWrote {processed_dir / 'stagnation_seed_level_retrospective.csv'}")

    # ============================================================================
    # Part 2: prospective test - D_i(c) vs G_i^future(c), c in {11, 13, 15}
    # ============================================================================
    print("\n" + "=" * 90)
    print("PART 2: PROSPECTIVE test - stagnation through c predicting FUTURE benefit after c")
    print("=" * 90)
    all_prospective_rows = []
    for c in PROSPECTIVE_CUTOFFS:
        print(f"\n--- cutoff c={c} (stagnation uses n in [{n_init}, {c}]; future gain sums n in ({c}, {FINAL_N}]) ---")
        D_c = compute_stagnation(trace_df, method="independent", cutoff=c)
        LD_c = compute_L_and_delta(trace_df, method="independent", cutoff=c, n_init=n_init, h=WINDOW_H)
        G_future = compute_future_gain(trace_df, cutoff=c)

        merged_c = D_c.merge(LD_c, on=["seed", "client_id"]).merge(G_future, on=["seed", "client_id"])
        merged_c["cutoff"] = c
        all_prospective_rows.append(merged_c)

        report_association(merged_c, "D_i", "G_future", label=f"D_i(c={c}) [fraction non-improving]")
        report_association(merged_c, "L_i", "G_future", label=f"L_i(c={c}) [current no-improvement streak]")
        report_association(merged_c, "delta_i", "G_future", label=f"Delta_i^(h={WINDOW_H})(c={c}) [windowed rel. improvement] (note: sign-flipped - LOWER delta_i = more stagnant, so expect NEGATIVE rho if the mechanism holds)")

    prospective_df = pd.concat(all_prospective_rows, ignore_index=True)
    prospective_df.to_csv(processed_dir / "stagnation_prospective.csv", index=False)
    print(f"\nWrote {processed_dir / 'stagnation_prospective.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/runs/phase2_fbo_comparison_campaign_v3_c1cds05.yaml")
    args = parser.parse_args()
    main(args.config)
