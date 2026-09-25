"""
Integration smoke tests for `src.optimization.fleet_bo` - tiny fleet, tiny budgets, all
four Phase-2 FBO methods, checking the orchestration produces well-formed, monotone
convergence traces rather than checking specific numeric outcomes (which would be a
brittle re-implementation of the loop itself).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.caching import make_cached_evaluate
from src.episodes import build_episode_bank
from src.fleet_families import generate_fixed_speed_fleet
from src.optimization.fleet_bo import (
    METHODS,
    TRIGGERED_SIMILARITY,
    WARMSTART_METHODS,
    FrozenPeerRecord,
    compute_baseline_reference,
    compute_oracle_reference,
    run_fbo_method,
    run_target_warmstart,
)
from src.similarity import median_distance_lengthscale, similarity_matrix

BASELINE_XI = [0.0, 0.30103, 0.30103]

TS = 0.05
T_TOTAL = 3.0
T0 = 0.5
R_MAX = 0.6108652381980153
TFILTER = 0.1
XI_LOWER = [-2.0, -2.0, -2.0]
XI_UPPER = [2.0, 2.0, 2.0]
STATE_SCALE = [0.053229, 0.575573, 0.139862]
FIXED_R = 1.0
EVAL_KWARGS = dict(u_min=-0.34906585, u_max=0.34906585, noise_std=0.0, process_noise=(0.0, 0.0), plant_mode="linear")


@pytest.fixture
def tiny_fleet_and_banks():
    fleet = generate_fixed_speed_fleet(
        Ts=TS, n_per_family=1, variability=0.025, seed=7, delta=1.0, lambda_f=0.99, lane_change_time=[1.0, 1.5],
    )
    banks = {
        c.client_id: build_episode_bank(
            c, Ts=TS, T_total=T_TOTAL, T0=T0, r_max=R_MAX, n_calibration=1, n_test=1, base_seed=123, tfilter=TFILTER,
        )
        for c in fleet
    }
    return fleet, banks


@pytest.fixture
def dummy_similarity(tiny_fleet_and_banks):
    fleet, _ = tiny_fleet_and_banks
    n = len(fleet)
    # A synthetic but structurally valid symmetric distance matrix (identification is
    # not exercised here - this test is about the BO orchestration, not dynamics
    # distance correctness, which has its own tests).
    rng = np.random.default_rng(0)
    A = rng.random((n, n))
    D2 = (A + A.T) / 2
    np.fill_diagonal(D2, 0.0)
    l_d = median_distance_lengthscale(D2)
    return similarity_matrix(D2, l_d)


@pytest.mark.parametrize("method", METHODS)
def test_run_fbo_method_produces_monotone_nondecreasing_best_so_far(
    method, tiny_fleet_and_banks, dummy_similarity, tmp_path
):
    fleet, banks = tiny_fleet_and_banks
    cache_evaluate = make_cached_evaluate(str(tmp_path / "cache"))
    oracle_rmse = compute_oracle_reference(
        fleet, banks, cache_evaluate, xi_lower=XI_LOWER, xi_upper=XI_UPPER, state_scale=STATE_SCALE,
        fixed_R=FIXED_R, eval_kwargs=EVAL_KWARGS, n_oracle_sobol=8, seed=0,
    )
    assert set(oracle_rmse) == {c.client_id for c in fleet}
    baseline_rmse = compute_baseline_reference(
        fleet, banks, cache_evaluate, baseline_xi=BASELINE_XI, state_scale=STATE_SCALE, fixed_R=FIXED_R,
        eval_kwargs=EVAL_KWARGS,
    )
    assert set(baseline_rmse) == {c.client_id for c in fleet}

    result = run_fbo_method(
        fleet, banks, method, cache_evaluate=cache_evaluate, xi_lower=XI_LOWER, xi_upper=XI_UPPER,
        state_scale=STATE_SCALE, fixed_R=FIXED_R, beta_max=np.deg2rad(6.0), eval_kwargs=EVAL_KWARGS,
        n_init=3, n_iters=2, n_candidates=16, similarity_S=dummy_similarity, baseline_rmse=baseline_rmse,
        oracle_rmse=oracle_rmse, seed=0,
    )

    assert result.method == method
    assert set(result.client_traces) == {c.client_id for c in fleet}
    for trace in result.client_traces.values():
        assert trace.n_local_evals == sorted(trace.n_local_evals)
        # "best-so-far" must never get worse as more evaluations accumulate.
        rmses = trace.best_calibration_rmse
        assert all(rmses[k] <= rmses[k - 1] + 1e-9 for k in range(1, len(rmses)))
        assert trace.n_infeasible >= 0
        assert trace.n_degrading_proposals >= 0


def test_independent_method_never_reports_degrading_proposals(tiny_fleet_and_banks, dummy_similarity, tmp_path):
    fleet, banks = tiny_fleet_and_banks
    cache_evaluate = make_cached_evaluate(str(tmp_path / "cache"))
    oracle_rmse = compute_oracle_reference(
        fleet, banks, cache_evaluate, xi_lower=XI_LOWER, xi_upper=XI_UPPER, state_scale=STATE_SCALE,
        fixed_R=FIXED_R, eval_kwargs=EVAL_KWARGS, n_oracle_sobol=8, seed=0,
    )
    baseline_rmse = compute_baseline_reference(
        fleet, banks, cache_evaluate, baseline_xi=BASELINE_XI, state_scale=STATE_SCALE, fixed_R=FIXED_R,
        eval_kwargs=EVAL_KWARGS,
    )
    result = run_fbo_method(
        fleet, banks, "independent", cache_evaluate=cache_evaluate, xi_lower=XI_LOWER, xi_upper=XI_UPPER,
        state_scale=STATE_SCALE, fixed_R=FIXED_R, beta_max=np.deg2rad(6.0), eval_kwargs=EVAL_KWARGS,
        n_init=3, n_iters=2, n_candidates=16, similarity_S=dummy_similarity, baseline_rmse=baseline_rmse,
        oracle_rmse=oracle_rmse, seed=0,
    )
    assert all(t.n_degrading_proposals == 0 for t in result.client_traces.values())


def test_baseline_reference_normalizes_training_y_by_own_baseline(tiny_fleet_and_banks, tmp_path):
    """Every EvalRecord.training_y should equal calibration_rmse / J_base for that
    SAME client (or the fixed infeasible penalty) - the core of the oracle-leakage fix:
    normalization uses only a per-client baseline-controller reference, never the
    oracle."""
    from src.optimization.fleet_bo import INFEASIBLE_PENALTY_NORM

    fleet, banks = tiny_fleet_and_banks
    cache_evaluate = make_cached_evaluate(str(tmp_path / "cache"))
    baseline_rmse = compute_baseline_reference(
        fleet, banks, cache_evaluate, baseline_xi=BASELINE_XI, state_scale=STATE_SCALE, fixed_R=FIXED_R,
        eval_kwargs=EVAL_KWARGS,
    )
    oracle_rmse = {c.client_id: float("nan") for c in fleet}  # deliberately unused by training_y
    S = np.eye(len(fleet))
    result = run_fbo_method(
        fleet, banks, "independent", cache_evaluate=cache_evaluate, xi_lower=XI_LOWER, xi_upper=XI_UPPER,
        state_scale=STATE_SCALE, fixed_R=FIXED_R, beta_max=np.deg2rad(6.0), eval_kwargs=EVAL_KWARGS,
        n_init=3, n_iters=1, n_candidates=16, similarity_S=S, baseline_rmse=baseline_rmse,
        oracle_rmse=oracle_rmse, seed=0,
    )
    for rec in result.all_records:
        if rec.feasible:
            expected = rec.calibration_rmse / baseline_rmse[rec.client_id]
            assert np.isclose(rec.training_y, expected)
        else:
            assert rec.training_y == INFEASIBLE_PENALTY_NORM


def test_oracle_feasibility_out_matches_returned_rmse(tiny_fleet_and_banks, tmp_path):
    """Step D instrumentation (docs/phase2_diagnostic_findings.md, 2026-09-02):
    `feasibility_out` must be populated for every client, reuse the SAME Sobol scan
    (n_sobol == n_oracle_sobol), and phi must be the exact feasible fraction - and
    passing it must not change the returned oracle_rmse at all."""
    fleet, banks = tiny_fleet_and_banks
    cache_evaluate = make_cached_evaluate(str(tmp_path / "cache"))
    kwargs = dict(
        xi_lower=XI_LOWER, xi_upper=XI_UPPER, state_scale=STATE_SCALE, fixed_R=FIXED_R,
        eval_kwargs=EVAL_KWARGS, n_oracle_sobol=8, seed=0,
    )
    oracle_rmse_plain = compute_oracle_reference(fleet, banks, cache_evaluate, **kwargs)

    feasibility: dict = {}
    oracle_rmse_instrumented = compute_oracle_reference(fleet, banks, cache_evaluate, feasibility_out=feasibility, **kwargs)

    assert oracle_rmse_instrumented == oracle_rmse_plain
    assert set(feasibility) == {c.client_id for c in fleet}
    for cid, stats in feasibility.items():
        assert stats["n_sobol"] == 8
        assert 0 <= stats["n_feasible"] <= 8
        assert stats["phi"] == pytest.approx(stats["n_feasible"] / stats["n_sobol"])
        # oracle_rmse is finite exactly when at least one Sobol point was feasible.
        assert np.isfinite(oracle_rmse_instrumented[cid]) == (stats["n_feasible"] > 0)


def _run_triggered(
    fleet, banks, cache_evaluate, dummy_similarity, baseline_rmse, oracle_rmse, *, trigger_cutoff, activated, seed=0,
):
    return run_fbo_method(
        fleet, banks, TRIGGERED_SIMILARITY, cache_evaluate=cache_evaluate, xi_lower=XI_LOWER, xi_upper=XI_UPPER,
        state_scale=STATE_SCALE, fixed_R=FIXED_R, beta_max=np.deg2rad(6.0), eval_kwargs=EVAL_KWARGS,
        n_init=3, n_iters=4, n_candidates=16, similarity_S=dummy_similarity, baseline_rmse=baseline_rmse,
        oracle_rmse=oracle_rmse, seed=seed, trigger_cutoff=trigger_cutoff, activated=activated,
    )


def test_triggered_similarity_requires_cutoff_and_activated(tiny_fleet_and_banks, dummy_similarity, tmp_path):
    fleet, banks = tiny_fleet_and_banks
    cache_evaluate = make_cached_evaluate(str(tmp_path / "cache"))
    baseline_rmse = {c.client_id: 1.0 for c in fleet}
    oracle_rmse = {c.client_id: 0.5 for c in fleet}
    with pytest.raises(ValueError, match="trigger_cutoff and activated"):
        run_fbo_method(
            fleet, banks, TRIGGERED_SIMILARITY, cache_evaluate=cache_evaluate, xi_lower=XI_LOWER, xi_upper=XI_UPPER,
            state_scale=STATE_SCALE, fixed_R=FIXED_R, beta_max=np.deg2rad(6.0), eval_kwargs=EVAL_KWARGS,
            n_init=3, n_iters=2, n_candidates=16, similarity_S=dummy_similarity, baseline_rmse=baseline_rmse,
            oracle_rmse=oracle_rmse, seed=0,
        )


def test_triggered_similarity_never_activated_matches_pure_independent(
    tiny_fleet_and_banks, dummy_similarity, tmp_path
):
    """activated=all-False must be byte-identical to `independent` at every round, both
    before AND after the cutoff - a client that never triggers should behave exactly
    like it was never given the option to collaborate."""
    fleet, banks = tiny_fleet_and_banks
    cache_evaluate = make_cached_evaluate(str(tmp_path / "cache"))
    baseline_rmse = compute_baseline_reference(
        fleet, banks, cache_evaluate, baseline_xi=BASELINE_XI, state_scale=STATE_SCALE, fixed_R=FIXED_R,
        eval_kwargs=EVAL_KWARGS,
    )
    oracle_rmse = compute_oracle_reference(
        fleet, banks, cache_evaluate, xi_lower=XI_LOWER, xi_upper=XI_UPPER, state_scale=STATE_SCALE,
        fixed_R=FIXED_R, eval_kwargs=EVAL_KWARGS, n_oracle_sobol=8, seed=0,
    )
    result_ind = run_fbo_method(
        fleet, banks, "independent", cache_evaluate=cache_evaluate, xi_lower=XI_LOWER, xi_upper=XI_UPPER,
        state_scale=STATE_SCALE, fixed_R=FIXED_R, beta_max=np.deg2rad(6.0), eval_kwargs=EVAL_KWARGS,
        n_init=3, n_iters=4, n_candidates=16, similarity_S=dummy_similarity, baseline_rmse=baseline_rmse,
        oracle_rmse=oracle_rmse, seed=0,
    )
    activated = {c.client_id: False for c in fleet}
    result_trig = _run_triggered(
        fleet, banks, cache_evaluate, dummy_similarity, baseline_rmse, oracle_rmse,
        trigger_cutoff=4, activated=activated, seed=0,
    )
    for cid in [c.client_id for c in fleet]:
        np.testing.assert_allclose(
            result_ind.client_traces[cid].best_calibration_rmse, result_trig.client_traces[cid].best_calibration_rmse,
        )


def test_triggered_similarity_pre_cutoff_matches_independent_even_when_activated(
    tiny_fleet_and_banks, dummy_similarity, tmp_path
):
    """Even a client marked `activated=True` must behave as pure independent for every
    round up to and including the cutoff - activation only takes effect afterward
    (Part 12's two-stage design: the decision is about FUTURE rounds, not a retroactive
    reweighting of the pool already used)."""
    fleet, banks = tiny_fleet_and_banks
    cache_evaluate = make_cached_evaluate(str(tmp_path / "cache"))
    baseline_rmse = compute_baseline_reference(
        fleet, banks, cache_evaluate, baseline_xi=BASELINE_XI, state_scale=STATE_SCALE, fixed_R=FIXED_R,
        eval_kwargs=EVAL_KWARGS,
    )
    oracle_rmse = compute_oracle_reference(
        fleet, banks, cache_evaluate, xi_lower=XI_LOWER, xi_upper=XI_UPPER, state_scale=STATE_SCALE,
        fixed_R=FIXED_R, eval_kwargs=EVAL_KWARGS, n_oracle_sobol=8, seed=0,
    )
    result_ind = run_fbo_method(
        fleet, banks, "independent", cache_evaluate=cache_evaluate, xi_lower=XI_LOWER, xi_upper=XI_UPPER,
        state_scale=STATE_SCALE, fixed_R=FIXED_R, beta_max=np.deg2rad(6.0), eval_kwargs=EVAL_KWARGS,
        n_init=3, n_iters=4, n_candidates=16, similarity_S=dummy_similarity, baseline_rmse=baseline_rmse,
        oracle_rmse=oracle_rmse, seed=0,
    )
    # trigger_cutoff=n_init+2=5 (never reached within this 3-init/4-iter run's LAST
    # round, n_local max=7, so only rounds up to n_local=5 stay pre-cutoff) - activate
    # every client, but since the cutoff is only reached partway through, compare just
    # the pre-cutoff checkpoints.
    activated = {c.client_id: True for c in fleet}
    result_trig = _run_triggered(
        fleet, banks, cache_evaluate, dummy_similarity, baseline_rmse, oracle_rmse,
        trigger_cutoff=5, activated=activated, seed=0,
    )
    for cid in [c.client_id for c in fleet]:
        trace_ind, trace_trig = result_ind.client_traces[cid], result_trig.client_traces[cid]
        pre_cutoff_idx = [k for k, n in enumerate(trace_ind.n_local_evals) if n <= 5]
        assert pre_cutoff_idx  # sanity: the fixture's budget must actually reach n_local=5
        np.testing.assert_allclose(
            [trace_ind.best_calibration_rmse[k] for k in pre_cutoff_idx],
            [trace_trig.best_calibration_rmse[k] for k in pre_cutoff_idx],
        )


def test_triggered_similarity_activated_client_pools_after_cutoff(tiny_fleet_and_banks, dummy_similarity, tmp_path):
    """An activated client's post-cutoff weight for a peer record must equal s_ij
    (matching plain `similarity`'s weighting exactly), not 0 - the direct unit-level
    check of the two-stage weight rule itself, independent of any full BO run."""
    from src.optimization.fleet_bo import EvalRecord, _weight_for_record

    fleet, _ = tiny_fleet_and_banks
    recipient_idx = 0
    peer_idx = 1
    peer_record = EvalRecord(
        uid=0, client_id=fleet[peer_idx].client_id, client_idx=peer_idx, xi=np.zeros(3), round_idx=1,
        source="bo", feasible=True, calibration_rmse=1.0, training_y=1.0,
    )
    common_kwargs = dict(
        recipient_idx=recipient_idx, record=peer_record, similarity_S=dummy_similarity,
        recipient_client=fleet[recipient_idx], beta_max=np.deg2rad(6.0), bank_recipient=None,
        state_scale=STATE_SCALE, fixed_R=FIXED_R, gate_cache={},
    )
    expected_s_ij = float(dummy_similarity[recipient_idx, peer_idx])

    w_pre = _weight_for_record(TRIGGERED_SIMILARITY, post_trigger=False, recipient_activated=True, **common_kwargs)
    w_post_not_activated = _weight_for_record(
        TRIGGERED_SIMILARITY, post_trigger=True, recipient_activated=False, **common_kwargs
    )
    w_post_activated = _weight_for_record(
        TRIGGERED_SIMILARITY, post_trigger=True, recipient_activated=True, **common_kwargs
    )
    assert w_pre == 0.0
    assert w_post_not_activated == 0.0
    assert w_post_activated == pytest.approx(expected_s_ij)


def test_baseline_feasibility_out_matches_fallback(tiny_fleet_and_banks, tmp_path):
    """`feasibility_out[client_id]` must be False exactly when baseline_rmse fell back
    to the 1.0 penalty, and True exactly when it holds a real evaluated RMSE."""
    fleet, banks = tiny_fleet_and_banks
    cache_evaluate = make_cached_evaluate(str(tmp_path / "cache"))
    feasibility: dict = {}
    baseline_rmse = compute_baseline_reference(
        fleet, banks, cache_evaluate, baseline_xi=BASELINE_XI, state_scale=STATE_SCALE, fixed_R=FIXED_R,
        eval_kwargs=EVAL_KWARGS, feasibility_out=feasibility,
    )
    assert set(feasibility) == {c.client_id for c in fleet}
    for cid, is_feasible in feasibility.items():
        assert isinstance(is_feasible, bool)
        if is_feasible:
            assert baseline_rmse[cid] != 1.0
        else:
            assert baseline_rmse[cid] == 1.0


# ---- run_target_warmstart (Part 14, frozen mature-fleet leave-one-out) --------------------


def test_weight_frozen_peer_matrix():
    from src.optimization.fleet_bo import _weight_frozen_peer

    assert _weight_frozen_peer("independent", is_own=True, s_ij=0.3) == 1.0
    assert _weight_frozen_peer("global", is_own=True, s_ij=0.3) == 1.0
    assert _weight_frozen_peer("similarity", is_own=True, s_ij=0.3) == 1.0
    assert _weight_frozen_peer("independent", is_own=False, s_ij=0.7) == 0.0
    assert _weight_frozen_peer("global", is_own=False, s_ij=0.7) == 1.0
    assert _weight_frozen_peer("similarity", is_own=False, s_ij=0.7) == pytest.approx(0.7)
    with pytest.raises(ValueError):
        _weight_frozen_peer("bogus", is_own=False, s_ij=0.5)


def _build_frozen_peer_records(fleet, banks, cache_evaluate, peer_client, n_points=3):
    """A handful of REAL evaluated (xi, training_y) pairs for one peer - mirrors what a
    slice of a peer's completed `independent` history would look like, without needing a
    full BO run just to test the warm-start loop."""
    from src.candidate_eval import evaluate

    bank = banks[peer_client.client_id]
    xis = [[0.0, 0.3, 0.3], [1.0, -0.5, 0.2], [-1.0, 1.0, -1.0]][:n_points]
    records = []
    for xi in xis:
        ev = cache_evaluate(
            peer_client, xi, bank, state_scale=STATE_SCALE, fixed_R=FIXED_R, episodes=[e for e in bank.episodes if e.role == "calibration"],
            **EVAL_KWARGS,
        )
        training_y = float(ev.tracking_rmse) if (ev.feasible and ev.tracking_rmse) else 3.0
        records.append(FrozenPeerRecord(client_id=peer_client.client_id, xi=np.asarray(xi, float), training_y=training_y))
    return records


@pytest.fixture
def frozen_peers_and_target(tiny_fleet_and_banks, tmp_path):
    fleet, banks = tiny_fleet_and_banks
    cache_evaluate = make_cached_evaluate(str(tmp_path / "cache"))
    target, peers = fleet[0], fleet[1:]
    frozen_records = []
    for peer in peers:
        frozen_records.extend(_build_frozen_peer_records(fleet, banks, cache_evaluate, peer))
    peer_similarity = {peer.client_id: 0.6 for peer in peers}
    return target, banks[target.client_id], cache_evaluate, frozen_records, peer_similarity


def test_warmstart_unknown_method_raises(frozen_peers_and_target):
    target, bank, cache_evaluate, frozen_records, peer_similarity = frozen_peers_and_target
    with pytest.raises(ValueError, match="warm-start method"):
        run_target_warmstart(
            target, bank, "bogus", cache_evaluate=cache_evaluate, xi_lower=XI_LOWER, xi_upper=XI_UPPER,
            state_scale=STATE_SCALE, fixed_R=FIXED_R, eval_kwargs=EVAL_KWARGS, n_init=2, n_iters=1, n_candidates=8,
            frozen_peer_records=frozen_records, peer_similarity=peer_similarity, baseline_rmse=1.0, oracle_rmse=0.5,
        )


@pytest.mark.parametrize("method", WARMSTART_METHODS)
def test_warmstart_produces_monotone_well_formed_trace(method, frozen_peers_and_target):
    target, bank, cache_evaluate, frozen_records, peer_similarity = frozen_peers_and_target
    trace = run_target_warmstart(
        target, bank, method, cache_evaluate=cache_evaluate, xi_lower=XI_LOWER, xi_upper=XI_UPPER,
        state_scale=STATE_SCALE, fixed_R=FIXED_R, eval_kwargs=EVAL_KWARGS, n_init=2, n_iters=3, n_candidates=8,
        frozen_peer_records=frozen_records, peer_similarity=peer_similarity, baseline_rmse=1.0, oracle_rmse=0.5, seed=0,
    )
    # ONLY the target's own evaluations count toward n_local_evals - frozen peer data
    # (available at time zero) never inflates this count.
    assert trace.n_local_evals == [2, 3, 4, 5]
    assert trace.n_local_evals == sorted(trace.n_local_evals)
    rmses = trace.best_calibration_rmse
    assert all(rmses[k] <= rmses[k - 1] + 1e-9 for k in range(1, len(rmses)))
    assert trace.n_infeasible >= 0


def test_warmstart_independent_ignores_frozen_peer_content(frozen_peers_and_target):
    """method="independent" must be byte-identical whether the frozen peer pool is full
    or empty - peer weight is always 0, so the content must never leak into the target's
    proposals."""
    target, bank, cache_evaluate, frozen_records, peer_similarity = frozen_peers_and_target
    kwargs = dict(
        cache_evaluate=cache_evaluate, xi_lower=XI_LOWER, xi_upper=XI_UPPER, state_scale=STATE_SCALE,
        fixed_R=FIXED_R, eval_kwargs=EVAL_KWARGS, n_init=2, n_iters=3, n_candidates=8,
        peer_similarity=peer_similarity, baseline_rmse=1.0, oracle_rmse=0.5, seed=0,
    )
    trace_with_peers = run_target_warmstart(target, bank, "independent", frozen_peer_records=frozen_records, **kwargs)
    trace_without_peers = run_target_warmstart(target, bank, "independent", frozen_peer_records=[], **kwargs)
    np.testing.assert_allclose(trace_with_peers.best_calibration_rmse, trace_without_peers.best_calibration_rmse)
    for xi_a, xi_b in zip(trace_with_peers.best_xi, trace_without_peers.best_xi):
        np.testing.assert_allclose(xi_a, xi_b)
