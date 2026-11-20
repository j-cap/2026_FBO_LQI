import sys
from pathlib import Path
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from coverage_stage3 import schedule,adjust,predict

class BudgetChecks(unittest.TestCase):
    def test_nested_unique_budget(self):
        full=schedule(16,[1.5,4.,7.])
        self.assertEqual(len(set(full)),16)
        for n in [0,4,8,16]:self.assertEqual(schedule(n,[1.5,4.,7.]),full[:n])
        self.assertLess(max(k for k,d in full),6)
    def test_zero_budget_preserves_prior(self):
        prior=lambda d:dict(E=1.2,S=1.1)
        c=adjust([],prior,1.)
        for v in c.values():np.testing.assert_array_equal(v,np.zeros(2))
        row=dict(candidate=0,demand=4.,prediction=dict(physical=True,E=2.,S=3.))
        self.assertAlmostEqual(predict([row],prior,c)[0]['E'],2.4)
    def test_same_fit_for_equal_prior_and_data(self):
        prior=lambda d:dict(E=1.,S=1.)
        rs=[dict(demand=d,prediction=dict(physical=True,E=1.,S=1.),plant=dict(physical=True,E=1.2,S=1.1),numerical_resolved=True) for d in [1.5,4.,7.]]
        a=adjust(rs,prior,1.);b=adjust(rs,prior,1.)
        for m in a:np.testing.assert_allclose(a[m],b[m])
        row=dict(candidate=0,demand=4.,prediction=dict(physical=True,E=1.,S=1.),plant=dict(E=999.))
        x=predict([row],prior,a);row['plant']['E']=0.
        self.assertEqual(x,predict([row],prior,a))

if __name__=='__main__':unittest.main()
