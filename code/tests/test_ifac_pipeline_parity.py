"""
Phase-0 parity test (doc §9.2): the adapters in `src/interfaces` and
`src/candidate_eval` must reproduce exactly what calling the underlying IFAC pipeline
(`src2`) directly produces, for a fixed seed - proving the wrappers introduce no
behavioral change. Definition of done (doc §9.2): reproducible through the new
wrappers, seeds deterministic, no BO code required to exercise it.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.candidate_eval import evaluate
from src.episodes import build_episode_bank
from src.ifac_bridge import Cluster, design_lqi, generate_fleet, run_CL_client_model, run_episode_update_rls
from src.interfaces.identifier import fit as identifier_fit
from src.interfaces.lqi_synthesizer import synthesize

SEED = 2025
TS = 0.05
T_TOTAL = 6.0
T0 = 1.0
R_MAX = 0.6108652381980153
TFILTER = 0.1
U_MIN, U_MAX = -0.34906585, 0.34906585
NOISE_STD = 1e-4


def _tiny_fleet():
    return generate_fleet(
        Ts=TS,
        avail_clusters=[Cluster.LOW_SPEED, Cluster.PAYLOAD],
        n_per_cluster=2,
        variability=0.025,
        seed=SEED,
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


def test_fleet_generation_is_deterministic():
    """Sanity check the parity tests below rest on: two independent `generate_fleet`
    calls with the same seed produce identical clients, so comparing a wrapped call on
    one fresh fleet against a direct call on another fresh fleet is a fair comparison."""
    fleet_a, fleet_b = _tiny_fleet(), _tiny_fleet()
    assert fleet_a[0].client_id == fleet_b[0].client_id
    np.testing.assert_allclose(fleet_a[0].Ad_true, fleet_b[0].Ad_true)
    np.testing.assert_allclose(fleet_a[0].Bd_true, fleet_b[0].Bd_true)
    np.testing.assert_allclose(fleet_a[0].Kx, fleet_b[0].Kx)


def test_identifier_matches_direct_rls_call():
    client_a, client_b = _tiny_fleet()[0], _tiny_fleet()[0]
    bank = _bank_for(client_a)
    seed = bank.episodes[0].seed

    result = identifier_fit(client_a, bank.r_ref, seeds=[seed], noise_std=NOISE_STD, u_min=U_MIN, u_max=U_MAX)

    np.random.seed(seed)
    run_episode_update_rls(client_b, bank.r_ref, noise_std=NOISE_STD, U_min=U_MIN, U_max=U_MAX)

    np.testing.assert_allclose(result.A_hat, client_b.Ad_hat)
    np.testing.assert_allclose(result.B_hat, client_b.Bd_hat)
    np.testing.assert_allclose(result.precision, client_b.W_raw)


def test_identifier_multi_episode_matches_repeated_direct_calls():
    """fit(..., seeds=[s1, s2, s3]) must equal three sequential direct
    run_episode_update_rls calls - not just re-run the first seed three times."""
    client_a, client_b = _tiny_fleet()[0], _tiny_fleet()[0]
    bank = _bank_for(client_a)
    seeds = [101, 202, 303]

    result = identifier_fit(client_a, bank.r_ref, seeds=seeds, noise_std=NOISE_STD, u_min=U_MIN, u_max=U_MAX)

    for seed in seeds:
        np.random.seed(seed)
        run_episode_update_rls(client_b, bank.r_ref, noise_std=NOISE_STD, U_min=U_MIN, U_max=U_MAX)

    np.testing.assert_allclose(result.A_hat, client_b.Ad_hat)
    np.testing.assert_allclose(result.B_hat, client_b.Bd_hat)
    assert client_b.round_counter == 3


def test_lqi_synthesizer_matches_direct_design_lqi():
    client = _tiny_fleet()[0]

    design = synthesize(client.Ad_hat, client.Bd_hat, xi=[0.0, 0.0, 0.0], state_scale=(1.0, 1.0, 1.0), fixed_R=1.0)
    Kx_direct, Ki_direct, k_r_direct = design_lqi(
        client.Ad_hat, client.Bd_hat, q_beta=1.0, q_r=1.0, q_int=1.0, rho=1.0
    )

    assert design.nominal_feasible
    np.testing.assert_allclose(design.Kx, Kx_direct)
    np.testing.assert_allclose(design.Ki, Ki_direct)
    assert design.k_r == pytest.approx(k_r_direct)


def test_lqi_synthesizer_state_scale_divides_not_multiplies():
    """state_scale=(beta_s,r_s,z_s) are DIVISORS (x_tilde=x/s): q_i(xi) = 10**xi_i /
    s_i**2. A non-unit scale must change Kx/Ki relative to identity scale, matching a
    direct design_lqi call with q pre-divided by s**2 - guards against silently
    inverting the normalization direction (q_raw = s**2 * 10**xi would be wrong)."""
    client = _tiny_fleet()[0]
    state_scale = (2.0, 0.5, 1.0)

    design = synthesize(client.Ad_hat, client.Bd_hat, xi=[0.0, 0.0, 0.0], state_scale=state_scale, fixed_R=1.0)
    q_expected = 1.0 / (np.asarray(state_scale) ** 2)  # 10**0 == 1
    Kx_direct, Ki_direct, k_r_direct = design_lqi(
        client.Ad_hat, client.Bd_hat, q_beta=q_expected[0], q_r=q_expected[1], q_int=q_expected[2], rho=1.0
    )

    assert design.nominal_feasible
    np.testing.assert_allclose(design.Q_diag_raw, q_expected)
    np.testing.assert_allclose(design.Kx, Kx_direct)
    np.testing.assert_allclose(design.Ki, Ki_direct)
    assert design.k_r == pytest.approx(k_r_direct)

    design_identity = synthesize(client.Ad_hat, client.Bd_hat, xi=[0.0, 0.0, 0.0], state_scale=(1.0, 1.0, 1.0))
    assert not np.allclose(design.Kx, design_identity.Kx)


def test_lqi_synthesizer_handles_bare_valueerror_from_riccati_solve(monkeypatch):
    """scipy.linalg.solve_discrete_are (inside design_lqi) can fail two different ways
    for a numerically bad (A, B): np.linalg.LinAlgError for a singular/degenerate case,
    or a bare ValueError from its internal `ordqz` generalized-Schur reordering when
    the problem is "too ill-conditioned" to reorder reliably - hit for real running
    Phase 1B (an under-fit RLS estimate from mismatched nonlinear-plant identification
    data produced an (A, B) that crashed the whole run before this fix, since only
    LinAlgError was caught). Both must be treated as an ordinary synthesis failure, not
    propagate and crash the caller."""
    import src.interfaces.lqi_synthesizer as lqi_synthesizer_mod

    def _raise_bare_value_error(*args, **kwargs):
        raise ValueError("Reordering of (A, B) failed because the transformed matrix pair (A, B) would be too far from generalized Schur form")

    monkeypatch.setattr(lqi_synthesizer_mod, "design_lqi", _raise_bare_value_error)

    client = _tiny_fleet()[0]
    design = synthesize(client.Ad_hat, client.Bd_hat, xi=[0.0, 0.0, 0.0])

    assert not design.nominal_feasible
    assert design.synthesis_failure_reason is not None
    assert "riccati_failure" in design.synthesis_failure_reason
    assert design.Kx is None and design.Ki is None and design.k_r is None


def test_candidate_eval_matches_direct_rollout():
    client_a, client_b = _tiny_fleet()[0], _tiny_fleet()[0]
    bank = _bank_for(client_a)
    seed = bank.episodes[0].seed
    # 10**log10(2) == 2.0 up to float roundoff, so this exercises the SAME weights
    # (Q_BETA=1, Q_R=2, Q_INT=2) as paper_UncertaintyAwareClustering/exp_02.ipynb's
    # original hand-tuned LQIConfig, without hand-rounding log10(2) in the test itself.
    xi = [0.0, math.log10(2.0), math.log10(2.0)]

    result = evaluate(client_a, xi, bank, u_min=U_MIN, u_max=U_MAX, noise_std=NOISE_STD)
    assert result.feasible

    Kx_direct, Ki_direct, k_r_direct = design_lqi(
        client_b.Ad_hat, client_b.Bd_hat, q_beta=1.0, q_r=2.0, q_int=2.0, rho=1.0
    )
    client_b.Kx, client_b.Ki, client_b.k_r = Kx_direct, Ki_direct, k_r_direct
    np.random.seed(seed)
    out = run_CL_client_model(client_b, bank.r_ref, noise_std=NOISE_STD, u_min=U_MIN, u_max=U_MAX)
    y_direct = out["y_hist"][:-1]
    r_direct = bank.r_ref[: len(y_direct)]
    rmse_direct = float(np.sqrt(np.mean((r_direct - y_direct) ** 2)))

    assert result.tracking_rmse == pytest.approx(rmse_direct)
