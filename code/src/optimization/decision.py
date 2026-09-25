"""
Go/no-go decision for the Phase-2 medium-scale (30-client/5-seed) FBO run (user
correction, 2026-08-29). Translates the user's stated decision tree into explicit,
testable criteria - a heuristic operationalization, not a claim that these exact
thresholds are the only reasonable ones:

  - similarity-weighted FBO "beats" independent BO if its regret is lower at a
    majority of the {10, 15, 20}-evaluation checkpoints, or it reaches within 5% of
    the oracle in fewer evaluations on average (early/mid-budget separation matters
    more than final-convergence separation, since if all methods eventually converge
    the paper's contribution is sample efficiency, not final quality).
  - global pooling is "harmful" if it has MORE infeasible evaluations than
    similarity-weighted FBO, reaches 5%-of-oracle less often, or ends up no better than
    independent BO by the final checkpoint.
  - recipient-aware FBO "earns its place" over similarity-only if it wins the same
    early/mid-budget majority-of-checkpoints or evals-to-5% comparison against
    similarity-only, WITHOUT its final-checkpoint regret being worse by more than
    `tolerance` (relative).

Verdict order mirrors the user's stated priority: stop if neither similarity nor
recipient-aware beats independent BO; otherwise prefer recipient-aware if it earns its
place over similarity-only; otherwise simplify to similarity-only.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

METHODS_ORDER = ["independent", "global", "similarity", "recipient_aware"]


def evals_to_threshold_mean(evals_to_threshold, budget: int) -> float:
    """Mean evaluations-to-threshold, imputing `budget` for a client/seed that never
    reached the threshold (NaN) instead of excluding it. A plain `.mean()` over the raw
    values silently drops non-convergent cases via pandas' NaN-skipping default, which
    lets a method with a LOWER reach rate look faster on average by having its hardest
    (never-converged) cases excluded rather than penalized - the bug found in the
    2026-08-29/30 medium-run decision (`recipient_aware` had both the worst
    frac_reached_5pct and, via this bug, the best evals_to_5pct_mean)."""
    arr = np.asarray(evals_to_threshold, dtype=float)
    arr = np.where(np.isnan(arr), float(budget), arr)
    return float(arr.mean())


def _regret_wins(metrics: Dict[str, Dict[str, float]], method_a: str, method_b: str, checkpoints) -> bool:
    wins = 0
    for n in checkpoints:
        ra = metrics[method_a].get(f"regret_n{n}")
        rb = metrics[method_b].get(f"regret_n{n}")
        if ra is not None and rb is not None and np.isfinite(ra) and np.isfinite(rb) and ra < rb:
            wins += 1
    return wins >= (len(checkpoints) // 2 + 1)


def _evals_to_5pct_better(metrics: Dict[str, Dict[str, float]], method_a: str, method_b: str) -> bool:
    ea = metrics[method_a].get("evals_to_5pct_mean")
    eb = metrics[method_b].get("evals_to_5pct_mean")
    return ea is not None and eb is not None and np.isfinite(ea) and np.isfinite(eb) and ea < eb


def _beats(metrics: Dict[str, Dict[str, float]], method_a: str, method_b: str, checkpoints) -> bool:
    return _regret_wins(metrics, method_a, method_b, checkpoints) or _evals_to_5pct_better(metrics, method_a, method_b)


def _final_not_worse(metrics: Dict[str, Dict[str, float]], method_a: str, method_b: str, final_n: int, tolerance: float) -> bool:
    ra = metrics[method_a].get(f"regret_n{final_n}")
    rb = metrics[method_b].get(f"regret_n{final_n}")
    if ra is None or rb is None or not (np.isfinite(ra) and np.isfinite(rb)):
        return True  # insufficient data to judge - don't let this alone block a verdict
    return ra <= rb * (1.0 + tolerance)


def decide_fbo_verdict(
    metrics: Dict[str, Dict[str, float]],
    *,
    checkpoints=(10, 15, 20),
    final_n: int = 20,
    tolerance: float = 0.10,
) -> Tuple[str, str]:
    """`metrics[method]` must provide `regret_n{n}` for each `n` in `checkpoints`,
    plus `evals_to_5pct_mean`, `frac_reached_5pct`, `n_infeasible_mean`. Returns
    (verdict, human-readable detail string). Verdicts:
    `implement_recipient_aware_fbo`, `implement_similarity_fbo_only`,
    `stop_independent_sufficient`, `inconclusive`."""
    missing = [m for m in METHODS_ORDER if m not in metrics]
    if missing:
        raise ValueError(f"metrics is missing method(s): {missing}")

    similarity_beats_independent = _beats(metrics, "similarity", "independent", checkpoints)
    recipient_beats_independent = _beats(metrics, "recipient_aware", "independent", checkpoints)
    recipient_beats_similarity = _beats(metrics, "recipient_aware", "similarity", checkpoints) and _final_not_worse(
        metrics, "recipient_aware", "similarity", final_n, tolerance
    )
    global_harmful = (
        metrics["global"]["n_infeasible_mean"] > metrics["similarity"]["n_infeasible_mean"]
        or metrics["global"]["frac_reached_5pct"] < metrics["similarity"]["frac_reached_5pct"]
        or not _final_not_worse(metrics, "independent", "global", final_n, 0.0)
    )

    if not similarity_beats_independent and not recipient_beats_independent:
        verdict = "stop_independent_sufficient"
    elif recipient_beats_similarity:
        verdict = "implement_recipient_aware_fbo"
    elif similarity_beats_independent:
        verdict = "implement_similarity_fbo_only"
    else:
        verdict = "inconclusive"

    detail = (
        f"similarity_beats_independent={similarity_beats_independent}, "
        f"recipient_beats_independent={recipient_beats_independent}, "
        f"recipient_beats_similarity={recipient_beats_similarity}, "
        f"global_harmful={global_harmful}"
    )
    return verdict, detail
