import sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'scripts')]
from types import SimpleNamespace
from dataclasses import asdict,replace
import numpy as np
from src.actuator_mechanism import episode,gains
from src.actuator_antiwindup import evaluate
from src.ifac.utils import nominal_params,build_bicycle_beta_r
from src.episodes import build_multi_maneuver_episode_bank
from actuator_mechanism import controller_arms


def setup():
    client=SimpleNamespace(client_id='nominal_00',params=nominal_params())
    a,b,*_=build_bicycle_beta_r(client.params)
    cc=dict(state_scale=[.053229,.575573,.139862],fixed_R=1.,noise_std=.01,process_noise=[0.,0.],
        u_min=-.34906585,u_max=.34906585,u_rate_max=.2618,beta_max_deg=6.,tire_mu=1.)
    bank=build_multi_maneuver_episode_bank(client,Ts=.05,T_total=6.,T0=1.,r_max=.2,lane_change_time=[1.,2.],
        n_calibration_per_maneuver=1,n_test_per_maneuver=0,base_seed=123,tfilter=.1)
    return client,a,b,cc,bank


def test_measured_score_parity_and_exact_mse_decomposition():
    client,a,b,cc,bank=setup();ep=bank.episodes[0];xi=[0.,0.,-1.]
    for steps in [5,10]:
        g=gains(a,b,.2,xi,cc,.05)
        result=episode(client,a,b,.2,g,bank,ep,cc,.05,steps)
        old=asdict(evaluate(client,a,b,.2,xi,replace(bank,episodes=[ep]),'calibration',cc,.05,steps))
        for key,value in old.items():
            if isinstance(value,(float,np.floating)):np.testing.assert_allclose(result[key],value,equal_nan=True,rtol=1e-12)
            else:assert result[key]==value
        np.testing.assert_allclose(result['measured_mse'],result['true_mse']+result['noise_mse']+result['cross_mse'],atol=1e-15)


def test_zero_noise_scores_identical_and_failed_episode_not_scored():
    client,a,b,cc,bank=setup();cc['noise_std']=0.;g=gains(a,b,.2,[0.,0.,-1.],cc,.05)
    r=episode(client,a,b,.2,g,bank,bank.episodes[0],cc,.05)
    assert r['true_cost']==r['cost'] and r['noise_mse']==0
    cc['beta_max_deg']=.00001
    r=episode(client,a,b,.2,g,bank,bank.episodes[0],cc,.05)
    assert not r['feasible'] and r['true_cost'] is None and np.isnan(r['cost'])


def test_gain_freezing_is_independent_of_target_bandwidth():
    _,a,b,cc,_=setup();models={(355238,'nominal_00'):(a,b),(987946,'nominal_00'):(a*1.01,b)}
    first=controller_arms((355238,'nominal_00'),models,.2,[0.,0.,-1.],cc,.05)
    second=controller_arms((355238,'nominal_00'),models,.4,[0.,0.,-1.],cc,.05)
    assert first['target_redesign']==first['target_frozen_tau']
    for mode in ['target_frozen_tau','donor_frozen']:assert first[mode]==second[mode]
    assert first['target_redesign']!=second['target_redesign']
