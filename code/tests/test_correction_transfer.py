import sys
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from correction_transfer import correction, choose, predicted

class TransferChecks(unittest.TestCase):
    def test_target_outcomes_do_not_enter_predictions(self):
        r=dict(candidate=0,prediction=dict(physical=True,E=1.,S=2.),plant=dict(E=999,S=999))
        a=predicted([r],dict(E=1.2,S=1.1))
        r['plant']=dict(E=0.,S=0.)
        self.assertEqual(a,predicted([r],dict(E=1.2,S=1.1)))
        self.assertIsNone(choose(a,1.1));self.assertEqual(choose(a,1.3),0)
    def test_median_ratios_and_invalid_exclusion(self):
        rows=[dict(prediction=dict(physical=True,E=2.,S=4.),plant=dict(physical=True,E=2*x,S=8.),numerical_resolved=True) for x in [1.,1.2,1.4]]
        self.assertEqual(correction(rows),dict(E=1.2,S=2.))
        rows[-1]['numerical_resolved']=False
        self.assertIsNone(correction(rows))
    def test_requirement_selects_minimum_activity(self):
        y={0:dict(E=.9,S=4),1:dict(E=1.1,S=2),2:dict(E=.8,S=5)}
        self.assertEqual(choose(y,1.),0);self.assertEqual(choose(y,1.2),1)

if __name__=='__main__':unittest.main()
