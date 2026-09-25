"""
Validation tests for `src.fleet_families` (Next-steps fleet redesign): three
plant-property families (nominal, payload, tire-degraded), all at ~U=15 m/s.
"""
from __future__ import annotations

import numpy as np

from src.fleet_families import NOMINAL, PAYLOAD, TIRE_DEGRADED, generate_fixed_speed_fleet
from src.ifac_bridge import nominal_params

TS = 0.05
N_PER_FAMILY = 10


def _fleet():
    return generate_fixed_speed_fleet(
        Ts=TS, n_per_family=N_PER_FAMILY, variability=0.025, seed=42, delta=1.0, lambda_f=0.99,
        lane_change_time=(1.0, 2.0),
    )


def test_fleet_size_and_family_labels():
    fleet = _fleet()
    assert len(fleet) == 3 * N_PER_FAMILY
    labels = {c.cluster.name for c in fleet}
    assert labels == {"NOMINAL", "PAYLOAD", "TIRE_DEGRADED"}
    for label, count in [(NOMINAL, N_PER_FAMILY), (PAYLOAD, N_PER_FAMILY), (TIRE_DEGRADED, N_PER_FAMILY)]:
        assert sum(1 for c in fleet if c.cluster.name == label.name) == count


def test_every_client_stays_near_15_ms_no_deliberate_speed_shift():
    """No family deliberately shifts U (unlike the old LOW_SPEED=10/HIGH_SPEED=20) -
    only the same +/-2.5% within-family noise that also perturbs m/Iz/Cf/Cr applies."""
    fleet = _fleet()
    for c in fleet:
        assert 15.0 * 0.95 <= c.params.U <= 15.0 * 1.05, f"{c.client_id}: U={c.params.U}"


def test_nominal_family_is_unperturbed_baseline_on_average():
    fleet = _fleet()
    p0 = nominal_params()
    nominal_clients = [c for c in fleet if c.cluster.name == "NOMINAL"]
    mean_m = np.mean([c.params.m for c in nominal_clients])
    mean_Iz = np.mean([c.params.Iz for c in nominal_clients])
    mean_Cf = np.mean([c.params.Cf for c in nominal_clients])
    mean_Cr = np.mean([c.params.Cr for c in nominal_clients])
    # +/-2.5% per-client noise around p0, mean over 10 clients should stay well within it
    assert abs(mean_m - p0.m) / p0.m < 0.05
    assert abs(mean_Iz - p0.Iz) / p0.Iz < 0.05
    assert abs(mean_Cf - p0.Cf) / p0.Cf < 0.05
    assert abs(mean_Cr - p0.Cr) / p0.Cr < 0.05


def test_payload_family_matches_src2_payload_prototype():
    fleet = _fleet()
    p0 = nominal_params()
    payload_clients = [c for c in fleet if c.cluster.name == "PAYLOAD"]
    mean_m = np.mean([c.params.m for c in payload_clients])
    mean_Iz = np.mean([c.params.Iz for c in payload_clients])
    # src2 cluster_prototype(Cluster.PAYLOAD, p0): m*1.20, Iz*1.25
    assert abs(mean_m - 1.20 * p0.m) / p0.m < 0.05
    assert abs(mean_Iz - 1.25 * p0.Iz) / p0.Iz < 0.05


def test_tire_degraded_family_matches_src2_aging_tires_prototype():
    fleet = _fleet()
    p0 = nominal_params()
    tire_clients = [c for c in fleet if c.cluster.name == "TIRE_DEGRADED"]
    mean_Cf = np.mean([c.params.Cf for c in tire_clients])
    mean_Cr = np.mean([c.params.Cr for c in tire_clients])
    # src2 cluster_prototype(Cluster.AGING_TIRES, p0): Cf*0.75, Cr unchanged
    assert abs(mean_Cf - 0.75 * p0.Cf) / p0.Cf < 0.05
    assert abs(mean_Cr - p0.Cr) / p0.Cr < 0.05


def test_clients_have_working_gains_and_rls():
    """Every client should come out with LQI gains already set (reset_gains) and a
    fresh RLS instance ready for identification - the same postconditions
    generate_fleet's own clients have."""
    fleet = _fleet()
    for c in fleet:
        assert c.Kx is not None and c.Ki is not None and c.k_r is not None
        assert np.all(np.isfinite(c.Kx)) and np.all(np.isfinite(c.Ki)) and np.isfinite(c.k_r)
        assert c.rls is not None
        assert c.round_counter == 0


def test_deterministic_given_seed():
    fleet_a, fleet_b = _fleet(), _fleet()
    assert fleet_a[0].client_id == fleet_b[0].client_id
    np.testing.assert_allclose(fleet_a[0].Ad_true, fleet_b[0].Ad_true)
    np.testing.assert_allclose(fleet_a[0].params.Cf, fleet_b[0].params.Cf)
