"""
Validation tests for `src.recipient_screen.predict_recipient_margins` (Phase 1D
methodological response to Phase 1C's asymmetry finding).
"""
from __future__ import annotations

import numpy as np

from src.episodes import build_episode_bank
from src.ifac_bridge import Cluster, generate_fleet
from src.recipient_screen import predict_recipient_margins

TS = 0.05
T_TOTAL = 6.0
T0 = 1.0
R_MAX = 0.6108652381980153
TFILTER = 0.1


def _tiny_fleet():
    return generate_fleet(
        Ts=TS,
        avail_clusters=[Cluster.LOW_SPEED, Cluster.PAYLOAD],
        n_per_cluster=2,
        variability=0.025,
        seed=2025,
        set_gains=True,
        compute_gains=False,
        delta=1.0,
        lambda_f=0.99,
        lane_change_time=[1.0, 2.0],
    )


def _bank_for(client):
    return build_episode_bank(
        client, Ts=TS, T_total=T_TOTAL, T0=T0, r_max=R_MAX,
        n_calibration=1, n_test=0, base_seed=5000, tfilter=TFILTER,
    )


def test_margins_are_finite_for_a_reasonable_xi():
    client = _tiny_fleet()[0]
    bank = _bank_for(client)
    margins = predict_recipient_margins(client, [0.0, 0.30103, 0.30103], bank.t, bank.r_ref)
    assert margins.nominal_feasible
    assert 0.0 <= margins.spectral_radius < 1.0
    assert margins.max_abs_beta_hat is not None and np.isfinite(margins.max_abs_beta_hat)
    assert margins.max_abs_delta_hat is not None and np.isfinite(margins.max_abs_delta_hat)
    assert margins.max_abs_delta_rate_hat is not None and np.isfinite(margins.max_abs_delta_rate_hat)


def test_infeasible_xi_returns_all_none(monkeypatch):
    """Which specific xi/state_scale reliably triggers a Riccati failure is a
    numerically fragile thing to pin down exactly (see
    test_lqi_synthesizer_handles_bare_valueerror_from_riccati_solve) - monkeypatch
    `synthesize` directly instead, so this test is deterministic."""
    import src.recipient_screen as recipient_screen_mod
    from src.interfaces.lqi_synthesizer import ControllerDesign

    def _infeasible_design(*args, **kwargs):
        return ControllerDesign(None, None, None, None, 1.0, None, False, "riccati_failure: mocked")

    monkeypatch.setattr(recipient_screen_mod, "synthesize", _infeasible_design)

    client = _tiny_fleet()[0]
    bank = _bank_for(client)
    margins = predict_recipient_margins(client, [0.0, 0.0, 0.0], bank.t, bank.r_ref)
    assert not margins.nominal_feasible
    assert margins.spectral_radius is None
    assert margins.max_abs_beta_hat is None
    assert margins.max_abs_delta_hat is None
    assert margins.max_abs_delta_rate_hat is None


def test_does_not_mutate_the_real_client():
    client = _tiny_fleet()[0]
    bank = _bank_for(client)
    Ad_true_before = client.Ad_true.copy()
    Bd_true_before = client.Bd_true.copy()
    Kx_before, Ki_before, k_r_before = client.Kx.copy(), client.Ki.copy(), client.k_r

    predict_recipient_margins(client, [0.5, 0.5, 0.5], bank.t, bank.r_ref)

    np.testing.assert_array_equal(client.Ad_true, Ad_true_before)
    np.testing.assert_array_equal(client.Bd_true, Bd_true_before)
    np.testing.assert_array_equal(client.Kx, Kx_before)
    np.testing.assert_array_equal(client.Ki, Ki_before)
    assert client.k_r == k_r_before


def test_uses_identified_not_true_model():
    """If Ad_hat/Bd_hat differ from Ad_true/Bd_true, the predicted margins must track
    the IDENTIFIED model - this is the whole point (recipient-conditioned on what the
    recipient itself believes, not omniscient knowledge of its true plant)."""
    client = _tiny_fleet()[0]
    bank = _bank_for(client)
    xi = [0.0, 0.30103, 0.30103]

    margins_identified = predict_recipient_margins(client, xi, bank.t, bank.r_ref)

    # Swap in a deliberately different "identified" model and confirm the margins change.
    client.Ad_hat = client.Ad_hat * 1.5
    margins_perturbed = predict_recipient_margins(client, xi, bank.t, bank.r_ref)

    assert margins_identified.nominal_feasible
    if margins_perturbed.nominal_feasible:
        assert margins_identified.spectral_radius != margins_perturbed.spectral_radius
