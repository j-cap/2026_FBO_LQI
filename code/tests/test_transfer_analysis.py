"""
Unit tests for `decide_next_step`'s five-way decision logic, added after Phase 1A
found a case the original four-way version couldn't represent: no mean-level
(fleet-average) cluster benefit, but a significant transfer-loss/dynamics-distance
relation AND a real worst-case/pairwise transfer risk (docs/phase1_diagnostic_findings.md).
"""
from __future__ import annotations

from src.transfer_analysis import decide_next_step


def test_cluster_benefit_and_transfer_relation_implements_fbo():
    decision = decide_next_step(G_cluster=0.05, individual_vs_cluster_gap=0.01, transfer_correlation=0.5)
    assert decision == "implement_dynamics_informed_fbo"


def test_cluster_benefit_without_transfer_relation_tests_descriptor():
    decision = decide_next_step(G_cluster=0.05, individual_vs_cluster_gap=0.01, transfer_correlation=0.1)
    assert decision == "test_control_aware_descriptor"


def test_individual_gap_without_cluster_benefit_rethinks_granularity():
    decision = decide_next_step(G_cluster=0.0, individual_vs_cluster_gap=0.05, transfer_correlation=0.1)
    assert decision == "rethink_federation_granularity"


def test_no_mean_benefit_but_real_worst_case_risk_pursues_safety_screening():
    """This is the Phase 1A case (phase1a_matched_model_full_v2): G_cluster=0.6%,
    individual_vs_cluster_gap=1.3%, transfer rho=0.377 with CI (0.272, 0.490) - below
    the 0.4 magnitude threshold but a tight, zero-excluding (statistically significant)
    interval - and a real worst-case transfer risk (some regime pairs at 26-40%
    relative RMSE increase)."""
    decision = decide_next_step(
        G_cluster=0.006, individual_vs_cluster_gap=0.013, transfer_correlation=0.377,
        transfer_correlation_ci=(0.272, 0.490), worst_case_transfer_risk=0.20,
    )
    assert decision == "pursue_dynamics_informed_safety_screening"


def test_ci_significance_qualifies_even_below_magnitude_threshold():
    """A tight, zero-excluding CI should count as a transfer relation even when the
    point estimate alone wouldn't clear transfer_corr_threshold - this is what the
    magnitude-only version of this function got wrong for Phase 1A's actual result."""
    decision = decide_next_step(
        G_cluster=0.10, individual_vs_cluster_gap=0.0, transfer_correlation=0.2,
        transfer_correlation_ci=(0.05, 0.35),
    )
    assert decision == "implement_dynamics_informed_fbo"


def test_wide_ci_crossing_zero_does_not_count_as_significant():
    decision = decide_next_step(
        G_cluster=0.0, individual_vs_cluster_gap=0.0, transfer_correlation=0.377,
        transfer_correlation_ci=(-0.05, 0.6), worst_case_transfer_risk=0.20,
    )
    assert decision == "stop_direction"


def test_nothing_significant_stops_direction():
    decision = decide_next_step(
        G_cluster=0.0, individual_vs_cluster_gap=0.0, transfer_correlation=0.05,
        worst_case_transfer_risk=0.02,
    )
    assert decision == "stop_direction"


def test_nan_worst_case_risk_does_not_crash_and_falls_back_to_stop():
    decision = decide_next_step(G_cluster=0.0, individual_vs_cluster_gap=0.0, transfer_correlation=0.05)
    assert decision == "stop_direction"
