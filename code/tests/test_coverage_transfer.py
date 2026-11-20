import sys
from pathlib import Path
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from coverage_transfer import local_summary, factors, training_sets, expected_cases

class CoverageTests(unittest.TestCase):
    def test_summary_interpolation(self):
        def rows(mult):return [dict(plant=dict(physical=True,E=mult,S=2*mult),prediction=dict(physical=True,E=1.,S=2.),numerical_resolved=True) for _ in range(3)]
        s={1.:local_summary(rows(1.)),3.:local_summary(rows(4.))}
        self.assertAlmostEqual(factors(s,np.sqrt(5))['E'],2.)
        self.assertAlmostEqual(factors({1.:s[1.]},3.)['E'],1.)
        with self.assertRaises(ValueError):factors(s,4.)
    def test_equal_budget_and_no_target_leakage(self):
        cfg=dict(endpoints=[1.5,7.],target_demands=[1.5,2.75,4.25,5.5,7.],calibration_count=24,per_endpoint=12,evaluation_count=64)
        rs=[dict(vehicle=v,candidate=k,demand=d) for v,k,d in expected_cases(cfg)]
        self.assertEqual(len(rs),536)
        for name,g in training_sets(rs,cfg).items():
            flat=sum(g.values(),[]);self.assertEqual(len(flat),24)
            if name!='target_local_diagnostic':self.assertTrue(all(r['vehicle']!='target' for r in flat))
            self.assertTrue(all(r['demand'] in cfg['endpoints'] for r in flat))
    def test_insufficient_valid_data_abstains(self):
        self.assertIsNone(factors({1.:dict(n=2,sums=dict(E=0.,S=0.))},2.))

if __name__=='__main__':unittest.main()
