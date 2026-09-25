"""
Regression tests for `generate_fixed_speed_fleet_v2_baseline_feasible` (Step C,
docs/phase2_diagnostic_findings.md, 2026-09-02) - the versioned, baseline-feasible
fleet generator. `generate_fixed_speed_fleet` (v1) is untouched and keeps its own
tests in test_fleet_families.py; these tests are specifically about the v2 admission
criterion and the per-attempt independent-RNG design that makes resampling safe.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

import src.fleet_families as fleet_families
from src.candidate_eval import EvaluationResult
from src.episodes import build_episode_bank
from src.fleet_families import generate_fixed_speed_fleet_v2_baseline_feasible
from src.optimization.fleet_bo import compute_baseline_reference

TS = 0.05
T_TOTAL = 3.0
T0 = 0.5
R_MAX = 0.6108652381980153
TFILTER = 0.1
STATE_SCALE = [0.053229, 0.575573, 0.139862]
FIXED_R = 1.0
BASELINE_XI = [0.0, 0.30103, 0.30103]
EVAL_KWARGS = dict(u_min=-0.34906585, u_max=0.34906585, noise_std=0.0, process_noise=(0.0, 0.0), plant_mode="linear")
N_PER_FAMILY = 2

GEN_KWARGS = dict(
    Ts=TS, n_per_family=N_PER_FAMILY, variability=0.025, seed=42, delta=1.0, lambda_f=0.99,
    lane_change_time=(1.0, 1.5), baseline_xi=BASELINE_XI, T_total=T_TOTAL, T0=T0, r_max=R_MAX,
    n_calibration_episodes=1, n_test_episodes=1, episode_base_seed=123, tfilter=TFILTER,
    state_scale=STATE_SCALE, fixed_R=FIXED_R, eval_kwargs=EVAL_KWARGS,
)


def _fleet():
    clients, provenance = generate_fixed_speed_fleet_v2_baseline_feasible(**GEN_KWARGS)
    return clients, provenance


def test_v2_same_seed_gives_identical_accepted_fleet():
    clients_a, prov_a = _fleet()
    clients_b, prov_b = _fleet()
    assert [c.client_id for c in clients_a] == [c.client_id for c in clients_b]
    for ca, cb in zip(clients_a, clients_b):
        np.testing.assert_allclose(ca.Ad_true, cb.Ad_true)
        np.testing.assert_allclose(ca.params.Cf, cb.params.Cf)
    pd_a = prov_a[prov_a["accepted"]].reset_index(drop=True)
    pd_b = prov_b[prov_b["accepted"]].reset_index(drop=True)
    assert pd_a["attempt"].tolist() == pd_b["attempt"].tolist()


def test_v2_all_accepted_clients_have_a_feasible_baseline_no_fallback():
    """The core Step C guarantee, checked the SAME way _prepare_seed_context does:
    compute_baseline_reference on the accepted fleet must report feasible=True for
    every client - zero 1.0-fallbacks."""
    clients, provenance = _fleet()
    banks = {
        c.client_id: build_episode_bank(
            c, Ts=TS, T_total=T_TOTAL, T0=T0, r_max=R_MAX, n_calibration=1, n_test=1,
            base_seed=123, tfilter=TFILTER,
        )
        for c in clients
    }

    def _direct_evaluate(client, xi, bank, **kwargs):
        from src.candidate_eval import evaluate

        return evaluate(client, xi, bank, **kwargs)

    feasibility: dict = {}
    baseline_rmse = compute_baseline_reference(
        clients, banks, _direct_evaluate, baseline_xi=BASELINE_XI, state_scale=STATE_SCALE, fixed_R=FIXED_R,
        eval_kwargs=EVAL_KWARGS, feasibility_out=feasibility,
    )
    assert all(feasibility.values()), f"fallback triggered for: {[c for c, ok in feasibility.items() if not ok]}"
    assert all(v != 1.0 for v in baseline_rmse.values())

    # Cross-check against the generator's own provenance for the accepted attempt.
    accepted_rows = provenance[provenance["accepted"]].set_index("client_id")
    for c in clients:
        assert accepted_rows.loc[c.client_id, "baseline_rmse"] == pytest.approx(baseline_rmse[c.client_id])


def test_v2_forced_rejection_of_one_slot_does_not_alter_other_slots():
    """Monkeypatch `evaluate` so payload_00's FIRST attempt is forced infeasible
    (forcing a resample) while every other call behaves normally - every OTHER
    client must come out byte-identical to an unpatched run."""
    from src.candidate_eval import evaluate as real_evaluate

    call_count = {"n": 0}

    def _patched_evaluate(client, xi, bank, **kwargs):
        if client.client_id == "payload_00" and call_count["n"] == 0:
            call_count["n"] += 1
            return EvaluationResult(
                client_id=client.client_id, cluster_id=client.cluster.name, xi=np.asarray(xi),
                controller_design=None, feasible=False, failure_reason="forced-for-test", tracking_rmse=None,
            )
        return real_evaluate(client, xi, bank, **kwargs)

    baseline_clients, _ = _fleet()
    with patch.object(fleet_families, "evaluate", side_effect=_patched_evaluate):
        forced_clients, forced_provenance = generate_fixed_speed_fleet_v2_baseline_feasible(**GEN_KWARGS)

    baseline_by_id = {c.client_id: c for c in baseline_clients}
    for c in forced_clients:
        if c.client_id == "payload_00":
            continue
        other = baseline_by_id[c.client_id]
        np.testing.assert_allclose(c.Ad_true, other.Ad_true)
        np.testing.assert_allclose(c.params.Cf, other.params.Cf)

    payload_00_attempts = forced_provenance[forced_provenance["client_id"] == "payload_00"]
    assert len(payload_00_attempts) >= 2  # attempt 0 (forced-infeasible) + at least one retry
    assert not payload_00_attempts.iloc[0]["accepted"]
    assert payload_00_attempts.iloc[-1]["accepted"]


def test_v2_exhausting_max_attempts_raises_loudly():
    def _always_infeasible(client, xi, bank, **kwargs):
        return EvaluationResult(
            client_id=client.client_id, cluster_id=client.cluster.name, xi=np.asarray(xi),
            controller_design=None, feasible=False, failure_reason="forced-for-test", tracking_rmse=None,
        )

    with patch.object(fleet_families, "evaluate", side_effect=_always_infeasible):
        with pytest.raises(RuntimeError, match="baseline-feasible draw"):
            generate_fixed_speed_fleet_v2_baseline_feasible(**{**GEN_KWARGS, "max_attempts": 3})


def test_v2_physical_parameters_stay_positive():
    clients, _ = _fleet()
    for c in clients:
        assert c.params.m > 0 and c.params.Iz > 0 and c.params.Cf > 0 and c.params.Cr > 0, c.client_id
        assert np.isfinite(c.Ad_true).all() and np.isfinite(c.Bd_true).all()
