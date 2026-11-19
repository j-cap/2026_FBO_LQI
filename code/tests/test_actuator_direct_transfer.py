import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import actuator_direct_transfer as run


def test_plan_excludes_target_and_uses_only_source_calibration():
    targets=[(1,'a'),(1,'b'),(2,'a')];histories={}
    for t in targets:
        for method in ['sobol','bo_lower_penalty']:
            for seed in [101,202,303]:
                histories[(t,method,seed)]=[dict(xi=[i,0,0],score=dict(cost=100-i,feasible=True)) for i in range(75)]
    # Target history must never be inspected.
    for k in list(histories):
        if k[0]==targets[0]:del histories[k]
    plan=run.transfer_plan(targets[0],targets,histories)
    assert len(plan)==24
    assert {(p['donor_seed'],p['donor_client']) for p in plan}==set(targets[1:])
    assert all(p['xi'][0]==p['budget']-1 for p in plan)


def test_no_infeasible_or_unresolved_success_and_negative_gap_retained():
    good=dict(feasible=True,integration_pass=True,cost=1.)
    assert run.classify(dict(good,cost=.9),good)['gap']<0
    for field,status in [('feasible','infeasible'),('integration_pass','unresolved_integration')]:
        result=run.classify(dict(good,**{field:False}),good)
        assert result['status']==status and not result['success_5']
    assert run.classify(good,dict(good,feasible=False))['status']=='unresolved_reference'


def test_episode_failure_cannot_be_averaged_away():
    good=dict(cost=1.,feasible=True);bad=dict(cost=float('nan'),feasible=False)
    rows=[dict(coarse=good,fine=good,integration=dict(status='pass')),
          dict(coarse=good,fine=bad,integration=dict(status='unresolved'))]
    summary=run.summarize(rows)
    assert not summary['feasible'] and summary['cost'] is None
    assert summary['failed_episodes']==1 and summary['unresolved_episodes']==1
