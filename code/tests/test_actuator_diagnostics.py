import sys
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import actuator_diagnostics as run


def score(cost=1.,feasible=True,reason=''):
    return dict(cost=cost,feasible=feasible,reason=reason)


def test_unresolved_is_not_failure_and_negative_regret_retained():
    assert run.classify(score(),None,False)==('unresolved_reference',None)
    assert run.classify(score(feasible=False),None,False)[0]=='infeasible'
    status, regret=run.classify(score(.8),score(),True)
    assert status=='success' and np.isclose(regret,-.2)


def test_failed_trajectory_integration_is_not_cost_convergence():
    ev=score(float('nan'),False,'beta_limit')
    result=run.integration_check(ev,ev,.001)
    assert result['status']=='failure_reproduced' and not result['cost_check_pass']
    assert run.integration_check(score(),ev,.001)['status']=='unresolved'


def test_reference_keeps_intermediate_best_and_never_reads_test(monkeypatch):
    calls=[]
    def evaluate(x,role):
        calls.append(role)
        assert role in ['calibration','validation']
        return score(1+float(np.sum(np.asarray(x)**2)))
    def minimize(fun,x,*args,**kwargs):
        fun(np.zeros(3))
        return SimpleNamespace(x=np.ones(3),success=False,nfev=1,message='budget')
    monkeypatch.setattr(run,'minimize',minimize)
    cfg=dict(reference_grid_sizes=[4,8],reference_starts=1,reference_maxfev=4,
             validation_shortlist=4,reference_tolerance=.01)
    ref=run.search_reference(evaluate,-np.ones(3),np.ones(3),11,cfg)
    assert ref['best_calibration']['xi']==[0.,0.,0.]
    assert ref['selected']['xi']==[0.,0.,0.]
    assert not ref['optimizer_diagnostics'][0]['success']


def test_banks_disjoint_and_legacy_preserved():
    cfg=dict(episodes_per_maneuver=dict(legacy=3,calibration=10,validation=20,test=40),
             episode_seeds=dict(legacy=5000,calibration=1200000,validation=1300000,test=1400000))
    fc=dict(Ts=.05,lane_change_time=[1.,2.]);cc=dict(T_total=6.,T0=1.,r_max=.6)
    banks=run.make_banks(SimpleNamespace(client_id='nominal_00'),cfg,fc,cc)
    assert [len(b.episodes) for b in banks.values()]==[6,20,40,80]
    from src.episodes import build_multi_maneuver_episode_bank
    old=build_multi_maneuver_episode_bank(SimpleNamespace(client_id='nominal_00'),Ts=.05,T_total=6.,T0=1.,r_max=.6,
        lane_change_time=[1.,2.],n_calibration_per_maneuver=3,n_test_per_maneuver=0,base_seed=5000)
    assert old.episodes==banks['legacy'].episodes


def test_staged_protocol_selection_and_resume(tmp_path,monkeypatch):
    from src.actuator_bandwidth import Score
    clients=[SimpleNamespace(client_id=f'{f}_00') for f in ['nominal','payload','tire_degraded']]
    monkeypatch.setattr(run,'context',lambda *a:(clients,{c.client_id:(None,None) for c in clients}))
    monkeypatch.setattr(run,'evaluate',lambda *a:Score(1+.001*float(np.sum(np.array(a[4])**2)),True,''))
    import actuator_bandwidth_experiment as original
    monkeypatch.setattr(original,'propose_next',lambda *a,**kw:(0,None))
    cfg=dict(fleet_seeds=[1],target_slots=['00'],optimizer_seeds=[101,202],budgets=[2,3],source_banks=['legacy','calibration'],
        episodes_per_maneuver={r:1 for r in ['legacy','calibration','validation','test']},
        episode_seeds=dict(legacy=5000,calibration=1200000,validation=1300000,test=1400000),
        reference_grid_sizes=[4,8],reference_starts=1,reference_maxfev=4,validation_shortlist=4,
        reference_tolerance=.01,integration_tolerance=.001)
    acfg=dict(source_tau=.2,target_taus=[.2,.4],substeps=5,n_candidates=8)
    fc=dict(Ts=.05,lane_change_time=[1.,2.]);cc=dict(xi_lower=[-1.]*3,xi_upper=[1.]*3,r_max=.6,T_total=6.,T0=1.)
    for stage in ['sources','references','test']:
        run.run_seed(1,stage,cfg,acfg,{},fc,cc,tmp_path)
    run.analyze(tmp_path,cfg,acfg)
    source=json.loads((tmp_path/'seeds/1/sources/nominal_00_calibration_101.json').read_text())
    assert len(source['records'])==3
    tests=json.loads((tmp_path/'seeds/1/tests/nominal_00.json').read_text())
    assert len(tests['sources'])==8 and len(tests['references'])==2
    monkeypatch.setattr(run,'evaluate',lambda *a:(_ for _ in ()).throw(AssertionError('evaluated on resume')))
    for stage in ['sources','references','test']:
        run.run_seed(1,stage,cfg,acfg,{},fc,cc,tmp_path)
