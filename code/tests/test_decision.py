"""Validation tests for `src.optimization.decision.decide_fbo_verdict` - synthetic
metrics dicts exercising each of the four verdict branches against the user's stated
decision tree (2026-08-29 correction)."""
from __future__ import annotations

import numpy as np
import pytest

from src.optimization.decision import decide_fbo_verdict, evals_to_threshold_mean


def test_evals_to_threshold_mean_imputes_budget_for_non_convergent_runs():
    # 2 clients reach at 5 and 10 evals; a 3rd never reaches (NaN) - a plain
    # nanmean would silently drop it and report 7.5, hiding the non-convergence.
    assert evals_to_threshold_mean([5, 10, np.nan], budget=20) == pytest.approx((5 + 10 + 20) / 3)


def test_evals_to_threshold_mean_penalizes_lower_reach_rate():
    # Same reach-when-it-reaches speed, but method B fails more often - B must not
    # look "faster on average" once non-convergence is penalized instead of excluded.
    method_a = [8, 9, 10, 11]
    method_b = [8, 9, np.nan, np.nan]
    assert evals_to_threshold_mean(method_b, budget=20) > evals_to_threshold_mean(method_a, budget=20)


def _metrics(regret_by_method, evals_to_5pct_mean, frac_reached_5pct, n_infeasible_mean):
    """regret_by_method: {method: {10: r10, 15: r15, 20: r20}}"""
    out = {}
    for method, regrets in regret_by_method.items():
        out[method] = {f"regret_n{n}": r for n, r in regrets.items()}
        out[method]["evals_to_5pct_mean"] = evals_to_5pct_mean[method]
        out[method]["frac_reached_5pct"] = frac_reached_5pct[method]
        out[method]["n_infeasible_mean"] = n_infeasible_mean[method]
    return out


def test_stop_when_neither_pooled_method_beats_independent():
    regrets = {
        "independent": {10: 0.5, 15: 0.3, 20: 0.1},
        "global": {10: 0.9, 15: 0.8, 20: 0.7},
        "similarity": {10: 0.6, 15: 0.4, 20: 0.2},
        "recipient_aware": {10: 0.6, 15: 0.4, 20: 0.2},
    }
    evals = {"independent": 12, "global": 18, "similarity": 14, "recipient_aware": 14}
    frac = {"independent": 1.0, "global": 0.7, "similarity": 1.0, "recipient_aware": 1.0}
    infeas = {"independent": 1.0, "global": 2.0, "similarity": 1.0, "recipient_aware": 1.0}
    metrics = _metrics(regrets, evals, frac, infeas)
    verdict, _ = decide_fbo_verdict(metrics)
    assert verdict == "stop_independent_sufficient"


def test_recipient_aware_wins_when_it_beats_both_independent_and_similarity():
    regrets = {
        "independent": {10: 0.5, 15: 0.3, 20: 0.1},
        "global": {10: 0.9, 15: 0.8, 20: 0.7},
        "similarity": {10: 0.4, 15: 0.2, 20: 0.08},
        "recipient_aware": {10: 0.3, 15: 0.15, 20: 0.07},
    }
    evals = {"independent": 16, "global": 20, "similarity": 13, "recipient_aware": 11}
    frac = {"independent": 1.0, "global": 0.6, "similarity": 1.0, "recipient_aware": 1.0}
    infeas = {"independent": 1.0, "global": 3.0, "similarity": 1.0, "recipient_aware": 0.5}
    metrics = _metrics(regrets, evals, frac, infeas)
    verdict, detail = decide_fbo_verdict(metrics)
    assert verdict == "implement_recipient_aware_fbo"
    assert "recipient_beats_similarity=True" in detail


def test_similarity_only_when_recipient_does_not_improve_on_similarity():
    regrets = {
        "independent": {10: 0.5, 15: 0.3, 20: 0.1},
        "global": {10: 0.9, 15: 0.8, 20: 0.7},
        "similarity": {10: 0.3, 15: 0.15, 20: 0.07},
        # recipient_aware is essentially identical to similarity - no early/mid
        # separation, no evals-to-5% advantage.
        "similarity_dup_for_recipient": None,
    }
    regrets["recipient_aware"] = dict(regrets["similarity"])
    del regrets["similarity_dup_for_recipient"]
    evals = {"independent": 16, "global": 20, "similarity": 12, "recipient_aware": 12}
    frac = {"independent": 1.0, "global": 0.6, "similarity": 1.0, "recipient_aware": 1.0}
    infeas = {"independent": 1.0, "global": 3.0, "similarity": 1.0, "recipient_aware": 1.0}
    metrics = _metrics(regrets, evals, frac, infeas)
    verdict, _ = decide_fbo_verdict(metrics)
    assert verdict == "implement_similarity_fbo_only"


def test_missing_method_raises():
    with pytest.raises(ValueError):
        decide_fbo_verdict({"independent": {}, "global": {}, "similarity": {}})
