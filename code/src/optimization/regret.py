"""
Normalized simple regret - the primary medium-run decision metric (user correction,
2026-08-29): r_i(n) = (J_i^best(n) - J_i^star) / (J_i^base - J_i^star). J_i^star
(oracle) and J_i^base (baseline controller) are used ONLY here, post-hoc, for analysis
- never inside `src.optimization.fleet_bo`'s training set, acquisition function, or
infeasibility penalty (that would be leakage no client could actually have at
deployment time).
"""
from __future__ import annotations

import numpy as np


def normalized_simple_regret(j_best, j_oracle, j_base, *, eps: float = 1e-9) -> np.ndarray:
    """Vectorized; denominator clipped to `eps` so a client whose oracle happens to
    equal (or, from sampling noise, slightly exceed) its baseline doesn't produce a
    division blow-up or sign flip."""
    j_best = np.asarray(j_best, dtype=float)
    j_oracle = np.asarray(j_oracle, dtype=float)
    j_base = np.asarray(j_base, dtype=float)
    denom = np.clip(j_base - j_oracle, eps, None)
    return (j_best - j_oracle) / denom


def oracle_relative_regret(j_best, j_oracle, *, eps: float = 1e-9) -> np.ndarray:
    """New PRIMARY metric for the T_lc-redesigned campaign (Part 8's recommendation,
    docs/phase2_diagnostic_findings.md, 2026-09-03): r_i^star(n) = (J_i^best(n) -
    J_i^star) / J_i^star - "fraction above the oracle," denominated by the oracle's own
    scale rather than `J_i^base - J_i^star`. `normalized_simple_regret`'s denominator
    can collapse toward zero for a client whose baseline happens to sit very close to
    its oracle (confirmed still present after the Step C baseline-normalization fix -
    Part 7's validation-rerun outlier audit), which inflates small absolute RMSE
    fluctuations into large regret swings; this metric doesn't have that failure mode
    since `J_i^star` is never expected to be near zero. Kept alongside (not replacing)
    `normalized_simple_regret`, which stays a secondary continuity metric so the new
    campaign's numbers remain comparable to Parts 2-7's."""
    j_best = np.asarray(j_best, dtype=float)
    j_oracle = np.asarray(j_oracle, dtype=float)
    denom = np.clip(j_oracle, eps, None)
    return (j_best - j_oracle) / denom
