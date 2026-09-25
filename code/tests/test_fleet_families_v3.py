"""
Regression tests for `generate_fixed_speed_fleet_v3_common_task` and
`build_multi_maneuver_episode_bank` (Part 8, docs/phase2_diagnostic_findings.md,
2026-09-03) - the common-task-distribution redesign that removes T_lc as a hidden
task confound. `generate_fixed_speed_fleet` (v1) and
`generate_fixed_speed_fleet_v2_baseline_feasible` (v2) are untouched; these tests are
specifically about v3's "every client sees every maneuver" admission/evaluation path.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

import src.fleet_families as fleet_families
from src.candidate_eval import EvaluationResult, evaluate
from src.episodes import build_multi_maneuver_episode_bank, calibration_episodes
from src.episodes import test_episodes as get_test_episodes
from src.fleet_families import generate_fixed_speed_fleet_v3_common_task
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
LANE_CHANGE_TIME = (1.0, 1.5)

GEN_KWARGS = dict(
    Ts=TS, n_per_family=N_PER_FAMILY, variability=0.025, seed=42, delta=1.0, lambda_f=0.99,
    lane_change_time=LANE_CHANGE_TIME, baseline_xi=BASELINE_XI, T_total=T_TOTAL, T0=T0, r_max=R_MAX,
    n_calibration_episodes_per_maneuver=1, n_test_episodes_per_maneuver=1, episode_base_seed=123, tfilter=TFILTER,
    state_scale=STATE_SCALE, fixed_R=FIXED_R, eval_kwargs=EVAL_KWARGS,
)


def _fleet():
    return generate_fixed_speed_fleet_v3_common_task(**GEN_KWARGS)


def test_v3_same_seed_gives_identical_accepted_fleet():
    clients_a, prov_a = _fleet()
    clients_b, prov_b = _fleet()
    assert [c.client_id for c in clients_a] == [c.client_id for c in clients_b]
    for ca, cb in zip(clients_a, clients_b):
        np.testing.assert_allclose(ca.Ad_true, cb.Ad_true)
        np.testing.assert_allclose(ca.params.Cf, cb.params.Cf)
    pd_a = prov_a[prov_a["accepted"]].reset_index(drop=True)
    pd_b = prov_b[prov_b["accepted"]].reset_index(drop=True)
    assert pd_a["attempt"].tolist() == pd_b["attempt"].tolist()


def test_v3_client_T_lc_is_uniform_not_slot_dependent():
    """The whole point of v3: no client's identity in the calibration objective
    depends on which half of its family it's in - `client.T_lc` (used only as the
    fixed identification maneuver, never by `build_multi_maneuver_episode_bank`) is
    the same for every client, unlike v1/v2's slot-index split."""
    clients, _ = _fleet()
    assert all(c.T_lc == LANE_CHANGE_TIME[0] for c in clients)


def test_v3_all_accepted_clients_have_a_feasible_baseline_no_fallback():
    clients, provenance = _fleet()
    banks = {
        c.client_id: build_multi_maneuver_episode_bank(
            c, Ts=TS, T_total=T_TOTAL, T0=T0, r_max=R_MAX, lane_change_time=LANE_CHANGE_TIME,
            n_calibration_per_maneuver=1, n_test_per_maneuver=1, base_seed=123, tfilter=TFILTER,
        )
        for c in clients
    }

    def _direct_evaluate(client, xi, bank, **kwargs):
        return evaluate(client, xi, bank, **kwargs)

    feasibility: dict = {}
    baseline_rmse = compute_baseline_reference(
        clients, banks, _direct_evaluate, baseline_xi=BASELINE_XI, state_scale=STATE_SCALE, fixed_R=FIXED_R,
        eval_kwargs=EVAL_KWARGS, feasibility_out=feasibility,
    )
    assert all(feasibility.values()), f"fallback triggered for: {[c for c, ok in feasibility.items() if not ok]}"
    assert all(v != 1.0 for v in baseline_rmse.values())

    accepted_rows = provenance[provenance["accepted"]].set_index("client_id")
    for c in clients:
        assert accepted_rows.loc[c.client_id, "baseline_rmse"] == pytest.approx(baseline_rmse[c.client_id])


def test_v3_forced_rejection_of_one_slot_does_not_alter_other_slots():
    call_count = {"n": 0}

    def _patched_evaluate(client, xi, bank, **kwargs):
        if client.client_id == "payload_00" and call_count["n"] == 0:
            call_count["n"] += 1
            return EvaluationResult(
                client_id=client.client_id, cluster_id=client.cluster.name, xi=np.asarray(xi),
                controller_design=None, feasible=False, failure_reason="forced-for-test", tracking_rmse=None,
            )
        return evaluate(client, xi, bank, **kwargs)

    baseline_clients, _ = _fleet()
    with patch.object(fleet_families, "evaluate", side_effect=_patched_evaluate):
        forced_clients, forced_provenance = generate_fixed_speed_fleet_v3_common_task(**GEN_KWARGS)

    baseline_by_id = {c.client_id: c for c in baseline_clients}
    for c in forced_clients:
        if c.client_id == "payload_00":
            continue
        other = baseline_by_id[c.client_id]
        np.testing.assert_allclose(c.Ad_true, other.Ad_true)
        np.testing.assert_allclose(c.params.Cf, other.params.Cf)

    payload_00_attempts = forced_provenance[forced_provenance["client_id"] == "payload_00"]
    assert len(payload_00_attempts) >= 2
    assert not payload_00_attempts.iloc[0]["accepted"]
    assert payload_00_attempts.iloc[-1]["accepted"]


def test_v3_exhausting_max_attempts_raises_loudly():
    def _always_infeasible(client, xi, bank, **kwargs):
        return EvaluationResult(
            client_id=client.client_id, cluster_id=client.cluster.name, xi=np.asarray(xi),
            controller_design=None, feasible=False, failure_reason="forced-for-test", tracking_rmse=None,
        )

    with patch.object(fleet_families, "evaluate", side_effect=_always_infeasible):
        with pytest.raises(RuntimeError, match="baseline-feasible draw"):
            generate_fixed_speed_fleet_v3_common_task(**{**GEN_KWARGS, "max_attempts": 3})


def test_v3_physical_parameters_stay_positive():
    clients, _ = _fleet()
    for c in clients:
        assert c.params.m > 0 and c.params.Iz > 0 and c.params.Cf > 0 and c.params.Cr > 0, c.client_id
        assert np.isfinite(c.Ad_true).all() and np.isfinite(c.Bd_true).all()


def test_multi_maneuver_bank_has_disjoint_seeds_per_maneuver_and_role():
    clients, _ = _fleet()
    bank = build_multi_maneuver_episode_bank(
        clients[0], Ts=TS, T_total=T_TOTAL, T0=T0, r_max=R_MAX, lane_change_time=LANE_CHANGE_TIME,
        n_calibration_per_maneuver=3, n_test_per_maneuver=3, base_seed=5000, tfilter=TFILTER,
    )
    assert set(bank.refs.keys()) == set(LANE_CHANGE_TIME)
    calib = calibration_episodes(bank)
    held_out = get_test_episodes(bank)
    assert len(calib) == 3 * len(LANE_CHANGE_TIME)
    assert len(held_out) == 3 * len(LANE_CHANGE_TIME)
    all_seeds = [e.seed for e in bank.episodes]
    assert len(all_seeds) == len(set(all_seeds)), "episode seeds must be pairwise disjoint"
    for T_lc in LANE_CHANGE_TIME:
        assert sum(1 for e in calib if e.T_lc == T_lc) == 3
        assert sum(1 for e in held_out if e.T_lc == T_lc) == 3


def test_evaluate_uses_per_episode_T_lc_reference_not_bank_fallback():
    """The core `evaluate()` change: an episode tagged with a non-first T_lc must
    roll out against ITS OWN reference trajectory, not the bank-level t/r_ref
    fallback (which is fixed to lane_change_time[0])."""
    clients, _ = _fleet()
    client = clients[0]
    bank = build_multi_maneuver_episode_bank(
        client, Ts=TS, T_total=T_TOTAL, T0=T0, r_max=R_MAX, lane_change_time=LANE_CHANGE_TIME,
        n_calibration_per_maneuver=1, n_test_per_maneuver=0, base_seed=5000, tfilter=TFILTER,
    )
    calib = calibration_episodes(bank)
    assert {e.T_lc for e in calib} == set(LANE_CHANGE_TIME)
    t1, r1 = bank.refs[LANE_CHANGE_TIME[0]]
    t2, r2 = bank.refs[LANE_CHANGE_TIME[1]]
    assert not np.allclose(r1, r2), "the two maneuvers must actually differ for this test to be meaningful"

    for ep in calib:
        ev = evaluate(
            client, BASELINE_XI, bank, state_scale=STATE_SCALE, fixed_R=FIXED_R, episodes=[ep], **EVAL_KWARGS,
        )
        assert ev.feasible
        expected_t, expected_r = bank.refs[ep.T_lc]
        # Re-run directly against the single-maneuver reference and confirm the RMSE
        # matches evaluate()'s per-episode lookup, not the (wrong) bank fallback.
        from src.interfaces.plant_client import PlantClient

        plant = PlantClient(client)
        client.Kx, client.Ki, client.k_r = ev.controller_design.Kx, ev.controller_design.Ki, ev.controller_design.k_r
        rollout = plant.rollout(
            expected_t, expected_r, seed=ep.seed, noise_std=0.0, process_noise=(0.0, 0.0),
            u_min=EVAL_KWARGS["u_min"], u_max=EVAL_KWARGS["u_max"], plant_mode=EVAL_KWARGS["plant_mode"], mu=1.0,
        )
        expected_rmse = float(np.sqrt(np.mean((expected_r[: len(rollout.y)] - rollout.y) ** 2)))
        assert ev.tracking_rmse == pytest.approx(expected_rmse, rel=1e-6)
