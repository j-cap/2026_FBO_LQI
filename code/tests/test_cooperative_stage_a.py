"""Scientific invariants for exact alignment and conditional transfer."""
import importlib.util
from pathlib import Path
import unittest
import math
spec=importlib.util.spec_from_file_location('audit',Path(__file__).resolve().parents[1]/'scripts/cooperative_stage_a.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

class AlignmentTests(unittest.TestCase):
    def test_alignment_uses_weights_not_order(self):
        c,d=audit.aligned_curves({'a':{(1,):1.,(2,):3.},'b':{(2,):1.,(1,):2.}})
        at_two=next(x for x in c if x['threshold']==2.)
        self.assertEqual(at_two['joint'],1)
        self.assertEqual(next(x for x in d if x['threshold']==2. and x['donor']=='b')['conditional_transfer'],.5)
    def test_disjoint_grids_are_not_joined(self):
        self.assertEqual(audit.aligned_curves({'a':{(1,):1.},'b':{(2,):1.}}),([],[]))
    def test_failed_or_unknown_points_cannot_pass(self):
        c,d=audit.aligned_curves({'a':{(1,):1.,(2,):2.},'b':{(1,):float('inf'),(2,):float('nan')}})
        self.assertTrue(all(x['joint']==0 for x in c))
        self.assertTrue(all(math.isnan(x['conditional_transfer']) for x in d if x['donor']=='b'))
    def test_subset_overlap_keeps_exact_denominator(self):
        c,_=audit.aligned_curves({'a':{(1,):1.,(2,):2.},'b':{(2,):1.,(3,):1.}})
        self.assertEqual(c[-1]['n_shared'],1)
        self.assertEqual(c[-1]['joint_fraction'],1.)

if __name__=='__main__':unittest.main()
