import sys
from pathlib import Path
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from coverage_stage3_margin import calibration_margin

class MarginChecks(unittest.TestCase):
    def row(self,k,d,ratio):
        return dict(candidate=k,demand=d,prediction=dict(physical=True,E=1.,S=1.),
                    plant=dict(physical=True,E=ratio,S=1.),differences=dict(E=1e-8,S=0.),numerical_resolved=True)
    def test_zero_budget_has_no_claimed_margin(self):
        m,f=calibration_margin([],lambda d:dict(E=1.,S=1.),1.)
        self.assertEqual(m,1.);self.assertEqual(f,[])
    def test_all_contexts_of_weight_excluded_together(self):
        rows=[self.row(0,d,1.2) for d in [1.5,4.,7.]]+[self.row(1,1.5,1.1)]
        m,folds=calibration_margin(rows,lambda d:dict(E=1.,S=1.),1.)
        self.assertEqual(len(folds),4);self.assertGreaterEqual(m,1.)
        for f in folds:
            self.assertNotIn(f['candidate'],f['training_candidates'])
            self.assertEqual(f['training_episodes'],1 if f['candidate']==0 else 3)
        self.assertAlmostEqual(m,max([1.]+[f['ratio'] for f in folds]))
    def test_invalid_observation_not_used_as_margin(self):
        row=self.row(0,4.,100.);row['numerical_resolved']=False
        m,folds=calibration_margin([row],lambda d:dict(E=1.,S=1.),1.)
        self.assertEqual(m,1.);self.assertEqual(folds,[])

if __name__=='__main__':unittest.main()
