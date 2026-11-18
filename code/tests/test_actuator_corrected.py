import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import actuator_corrected as run


def ev(cost=1.,feasible=True):return dict(cost=cost,feasible=feasible,reason='' if feasible else 'beta_limit')


def test_no_repeats_and_resume_equals_uninterrupted(monkeypatch):
    monkeypatch.setattr(run,'fit_weighted_gp',lambda *a,**k:SimpleNamespace(gp=object()))
    monkeypatch.setattr(run,'expected_improvement',lambda gp,x,*a,**k:np.ones(len(x)))
    score=lambda x:ev(1+float(np.sum(x*x)))
    method=dict(kind='bo',penalty=10.,margin=.01)
    full=run.optimize(score,method,10,-np.ones(3),np.ones(3),101,16,1.)
    partial=run.optimize(score,method,5,-np.ones(3),np.ones(3),101,16,1.)
    resumed=run.optimize(score,method,10,-np.ones(3),np.ones(3),101,16,1.,partial)
    assert full==resumed
    assert len({tuple(r['xi']) for r in full})==10
    baseline=run.optimize(score,dict(kind='sobol'),10,-np.ones(3),np.ones(3),101,16,1.)
    assert [r['xi'] for r in full[:2]]==[r['xi'] for r in baseline[:2]]


def test_margin_and_penalty_are_isolated(monkeypatch):
    margins=[];values=[]
    def fit(x,y,*a,**k):values.append(y.copy());return SimpleNamespace(gp=object())
    monkeypatch.setattr(run,'fit_weighted_gp',fit)
    def ei(gp,x,best,xi):margins.append(xi);return np.ones(len(x))
    monkeypatch.setattr(run,'expected_improvement',ei)
    r=run.optimize(lambda x:ev(float('nan'),False),dict(kind='bo',penalty=1.,margin=0.),3,
                   -np.ones(3),np.ones(3),101,8,1.)
    assert margins==[0.] and np.all(values[0]==1.) and r[-1]['training_y']==1.


def test_diversity_keeps_conservative_candidates_and_excludes_infeasible():
    records=[dict(xi=[x]*3,calibration=ev(1+x)) for x in [0.,.001,.002,.003,.5,1.]]
    records.append(dict(xi=[2.]*3,calibration=ev(.1,False)))
    chosen=run.diverse_shortlist(records,4,np.zeros(3),np.ones(3)*2)
    assert chosen[0]['xi']==[0.]*3
    assert any(r['xi']==[1.]*3 for r in chosen)
    assert all(r['calibration']['feasible'] for r in chosen)


def test_trace_reproduces_single_episode_cost():
    from src.ifac.utils import nominal_params,build_bicycle_beta_r
    from src.episodes import build_multi_maneuver_episode_bank
    from src.actuator_antiwindup import evaluate
    from dataclasses import replace
    client=SimpleNamespace(client_id='nominal_00',params=nominal_params())
    ac,bc,*_=build_bicycle_beta_r(client.params)
    cc=dict(state_scale=[.053229,.575573,.139862],fixed_R=1.,noise_std=.01,
            process_noise=[0.,0.],u_min=-.34906585,u_max=.34906585,u_rate_max=.2618,beta_max_deg=6.,tire_mu=1.)
    bank=build_multi_maneuver_episode_bank(client,Ts=.05,T_total=6.,T0=1.,r_max=.2,lane_change_time=[1.,2.],
        n_calibration_per_maneuver=1,n_test_per_maneuver=0,base_seed=123,tfilter=.1)
    ep=bank.episodes[0];one=replace(bank,episodes=[ep]);x=[0.,0.,-1.]
    result=evaluate(client,ac,bc,.2,x,one,'calibration',cc,.05,5)
    trace=run.trace_episode(client,ac,bc,.2,x,bank,ep,cc,.05,5)
    assert result.feasible
    np.testing.assert_allclose(result.cost,np.sqrt(np.mean([r['measured_error']**2 for r in trace])),rtol=1e-12)


def test_full_stages_with_fake_physics(tmp_path,monkeypatch):
    import json
    from src.actuator_bandwidth import Score
    from dataclasses import asdict
    monkeypatch.setattr(run,'ROOT',tmp_path)
    client=SimpleNamespace(client_id='nominal_00',params=None)
    monkeypatch.setattr(run,'context',lambda *a:([client],{'nominal_00':(None,None)}))
    monkeypatch.setattr(run,'evaluate',lambda *a:Score(1+.001*float(np.sum(np.array(a[4])**2)),True,''))
    monkeypatch.setattr(run,'trace_episode',lambda *a:[])
    monkeypatch.setattr(run,'fit_weighted_gp',lambda *a,**k:SimpleNamespace(gp=None))
    old=dict(base={},fleet=dict(Ts=.05,lane_change_time=[1.,2.]),controller=dict(xi_lower=[-1.]*3,xi_upper=[1.]*3,
        r_max=.6,T_total=6.,T0=1.),actuator=dict(source_tau=.2,target_taus=[.2],substeps=5,n_candidates=8),
        config=dict(episodes_per_maneuver={r:1 for r in ['legacy','calibration','validation','test']},
        episode_seeds=dict(legacy=5000,calibration=1200000,validation=1300000,test=1400000)))
    cfg=dict(repair_results='repair',previous_results='previous',progress_seconds=30,budgets=[2,3],optimizer_seeds=[101],
        methods=dict(sobol=dict(kind='sobol')),reference_grid=4,reference_shortlist=4,
        validation_per_maneuver=1,validation_seed=1500000,test_per_maneuver=1,test_seed=1600000)
    prev=tmp_path/'previous/seeds/1';r=dict(xi=[0.,0.,0.],score=ev())
    run.write_json(prev/'sources/nominal_00_calibration_101.json',dict(records=[r]*3))
    run.write_json(prev/'references/nominal_00_0.2.json',dict(selected=r,shortlist=[r]))
    run.write_json(tmp_path/'repair/seeds/1/nominal_00/references/0.2.json',dict(shortlist=[r]))
    out=tmp_path/'output'
    for stage in ['audit','references']:
        run.run_vehicle(1,'nominal_00',stage,cfg,old,out)
    # No test_new cache entries exist until explicit test stage (role is in key,
    # assert directly by counting final-assessment artifacts).
    assert not (out/'seeds/1/nominal_00/assessment').exists()
    run.run_vehicle(1,'nominal_00','test',cfg,old,out)
    run.analyze(out,[(1,'nominal_00')],cfg,old)
    assert (out/'processed/optimizer_assessment.csv').exists()
    monkeypatch.setattr(run,'evaluate',lambda *a:(_ for _ in ()).throw(AssertionError('resume evaluated')))
    for stage in ['audit','references','test']:run.run_vehicle(1,'nominal_00',stage,cfg,old,out)
