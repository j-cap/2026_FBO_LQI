import numpy as np
from scipy.linalg import expm, solve_discrete_are
from types import SimpleNamespace
from src.actuator_bandwidth import continuous_vehicle,augment_discrete,design,step


def test_zoh_embedding_round_trip():
    ac=np.array([[-2.,.4],[-.1,-3.]])
    bc=np.array([[1.],[2.]])
    block=np.zeros((3,3));block[:2,:2]=ac;block[:2,2:]=bc
    discrete=expm(block*.05)
    a,b=continuous_vehicle(discrete[:2,:2],discrete[:2,2:],.05)
    np.testing.assert_allclose(a,ac,atol=1e-12);np.testing.assert_allclose(b,bc,atol=1e-12)


def test_actuator_exact_pole_and_same_interval_vehicle_response():
    a,b=augment_discrete(np.diag([-2.,-3.]),np.ones((2,1)),.2,.05)
    np.testing.assert_allclose(a[2,2],np.exp(-.25))
    np.testing.assert_allclose(b[2,0],1-np.exp(-.25))
    assert (b[:2]>0).all()  # no accidental one-sample delay


def test_integrator_coordinate_change_preserves_gain():
    ts=.05; a,b=augment_discrete(np.diag([-2.,-3.]),np.ones((2,1)),.2,ts)
    xi=np.array([0.,.2,-.4]);scales=np.array([.1,.5,.2])
    k,kr,rho=design(a,b,xi,scales,ts)
    ai=np.block([[a,np.zeros((3,1))],[np.array([[0.,1.,0.]]),np.array([[.998]])]])
    bi=np.vstack([b,[[0.]]]); q=10**xi/scales**2
    P=solve_discrete_are(ai,bi,np.diag([q[0],q[1],0,q[2]]),np.eye(1))
    legacy=np.linalg.solve(1+bi.T@P@bi,bi.T@P@ai).ravel()
    np.testing.assert_allclose(k[:3],legacy[:3],rtol=1e-8)
    np.testing.assert_allclose(k[3]*ts,legacy[3],rtol=1e-8)
    assert rho<1 and np.isfinite(kr)


def test_coupled_rk4_matches_actuator_step_response():
    p=SimpleNamespace(m=1500.,b=1.5,a=1.2,U=15.,Cf=70000.,Cr=80000.,Iz=2500.)
    x=np.zeros(3)
    for _ in range(100):x=step(x,.02,p,.2,.001,1.)
    np.testing.assert_allclose(x[2],.02*(1-np.exp(-.5)),rtol=1e-9)
    assert abs(x[1])>0


def test_bo_counts_transfer_evaluation_and_uses_calibration_only(monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import actuator_bandwidth_experiment as run
    calls=[]
    def score(x):
        calls.append(np.array(x))
        return dict(cost=float(1+np.sum(np.asarray(x)**2)),feasible=True,reason='')
    monkeypatch.setattr(run,'propose_next',lambda *args,**kwargs:(0,None))
    initial=[.2,.3,.4]
    records,recs=run.bo(score,4,np.zeros(3),np.ones(3),5,8,1.,initial=initial)
    assert len(calls)==4 and len(recs)==4
    np.testing.assert_array_equal(calls[0],initial)
    assert all(any(np.array_equal(x,r['xi']) for r in records[:n]) for n,x in enumerate(recs,1))


def test_checkpoint_supports_numpy_boolean(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from actuator_bandwidth_experiment import write_json
    import json
    path=tmp_path/'score.json'
    write_json(path,{'feasible':np.bool_(True)})
    assert json.loads(path.read_text())=={'feasible':True}


def test_stage_sequence_and_resume_with_controlled_evaluator(tmp_path,monkeypatch):
    import sys,json
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import actuator_bandwidth_experiment as run
    from src.actuator_bandwidth import Score
    clients=[SimpleNamespace(client_id='nominal_00',Ad_hat=np.diag([.8,.7]),Bd_hat=np.ones((2,1))*.1,W_raw=np.eye(6)),
             SimpleNamespace(client_id='nominal_01',Ad_hat=np.diag([.81,.71]),Bd_hat=np.ones((2,1))*.1,W_raw=np.eye(6))]
    models={c.client_id:(np.diag([-2.,-3.]),np.ones((2,1))) for c in clients}
    monkeypatch.setattr(run,'context',lambda *args:(clients,models))
    monkeypatch.setattr(run,'evaluate',lambda client,ac,bc,tau,xi,*args:Score(1+.001*float(np.sum(np.asarray(xi)**2)),True,''))
    monkeypatch.setattr(run,'propose_next',lambda *args,**kwargs:(0,None))
    cfg=dict(source_tau=.2,target_taus=[.2,.4],source_budget=3,target_budget=3,n_candidates=8,
             substeps=3,landscape_sizes=[4,8],reference_starts=1,reference_maxfev=8,reference_tolerance=.01)
    base=dict(n_calibration_episodes=1,episode_base_seed=5000)
    fc=dict(Ts=.05,lane_change_time=[1.,2.])
    cc=dict(xi_lower=[-1.]*3,xi_upper=[1.]*3,r_max=.1,T_total=6.,T0=1.)
    for stage in ['sources','diagnostic','bo']:
        run.one_seed(1,stage,cfg,base,fc,cc,tmp_path,['00'],[.2,.4])
    files=list((tmp_path/'seeds/1').glob('bo_*.json'));assert len(files)==1
    rows=json.loads(files[0].read_text())['rows'];assert len(rows)==18
    for tau in [.2,.4]:
        starts=[r['xi'] for r in rows if r['tau']==tau and r['n_local_evals']==1 and r['method']!='local']
        assert starts[0]==starts[1]
    monkeypatch.setattr(run,'context',lambda *args:(_ for _ in ()).throw(AssertionError('resume rebuilt context')))
    run.one_seed(1,'bo',cfg,base,fc,cc,tmp_path,['00'],[.2,.4])
