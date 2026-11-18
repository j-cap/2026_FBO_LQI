import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from src.actuator_antiwindup import integral_update,evaluate


@pytest.mark.parametrize('raw',[-2.,2.])
@pytest.mark.parametrize('ki',[.1,10.,1000.])
def test_tracking_moves_command_toward_both_limits_independent_of_gain(raw,ki):
    eta=-raw/ki;applied=np.clip(raw,-1,1)
    baseline=.998*eta
    updated=integral_update(eta,0.,raw,applied,ki,.05)
    np.testing.assert_allclose(-ki*(updated-baseline),.25*(applied-raw))
    assert abs(-ki*updated-applied)<abs(raw-applied)


def test_tracking_coordinate_equivalence():
    ts=.05;ki_eta=100.;ki_z=ts*ki_eta;z=.3;raw=-2.;applied=-1.;error=.1
    eta_next=integral_update(ts*z,error,raw,applied,ki_eta,ts)
    z_next=.998*z+error+ts*(raw-applied)/(ki_z*.2)
    np.testing.assert_allclose(eta_next,ts*z_next)


def test_no_saturation_identical_in_all_modes():
    values=[integral_update(.3,.2,.1,.1,100.,.05,m) for m in ['legacy','sign_only','tracking']]
    assert values[0]==values[1]==values[2]


def test_recovery_after_error_removed():
    eta=-.2;ki=100.
    for _ in range(100):
        raw=-ki*eta
        eta=integral_update(eta,0.,raw,np.clip(raw,-1,1),ki,.05)
    assert abs(ki*eta)<1.
    old=-.2
    for _ in range(10):
        raw=-ki*old
        old=integral_update(old,0.,raw,np.clip(raw,-1,1),ki,.05,'legacy')
    assert abs(old)>.2


def test_bad_tracking_parameters_rejected():
    with pytest.raises(ValueError):integral_update(0,0,1,0,0,.05)
    with pytest.raises(ValueError):integral_update(0,0,1,0,10,.05,tracking_time=.01)


def test_legacy_evaluator_parity():
    from types import SimpleNamespace
    from src.ifac.utils import nominal_params,build_bicycle_beta_r
    from src.episodes import build_multi_maneuver_episode_bank
    from src.actuator_bandwidth import evaluate as legacy
    from dataclasses import asdict
    client=SimpleNamespace(client_id='nominal_00',params=nominal_params())
    ac,bc,*_=build_bicycle_beta_r(client.params)
    cc=dict(state_scale=[.053229,.575573,.139862],fixed_R=1.,noise_std=.01,
        process_noise=[0.,0.],u_min=-.05,u_max=.05,u_rate_max=.2618,beta_max_deg=6.,tire_mu=1.)
    bank=build_multi_maneuver_episode_bank(client,Ts=.05,T_total=6.,T0=1.,r_max=.2,lane_change_time=[1.,2.],
        n_calibration_per_maneuver=1,n_test_per_maneuver=0,base_seed=123,tfilter=.1)
    args=(client,ac,bc,.2,[0.,0.,-1.],bank,'calibration',cc,.05,5)
    a,b=asdict(legacy(*args)),asdict(evaluate(*args,mode='legacy'))
    for key in a:
        if isinstance(a[key],(float,np.floating)):np.testing.assert_allclose(a[key],b[key],equal_nan=True)
        else:assert a[key]==b[key]
