import importlib.util
from pathlib import Path
import unittest
import numpy as np
spec=importlib.util.spec_from_file_location('validation',Path(__file__).resolve().parents[1]/'scripts/cooperative_validation.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)

class ValidationTests(unittest.TestCase):
    def test_seed_bank_disjoint_from_legacy_and_reproducible(self):
        a=v.noise_seeds(1,'nominal',1.,1000)
        self.assertEqual(a,v.noise_seeds(1,'nominal',1.,1000))
        self.assertEqual(len(set(a)),1000)
        self.assertTrue(all(2**31<=s<2**32 for s in a))
        self.assertFalse(set(a)&set(v.noise_seeds(1,'nominal',1.,1000,True)))
        self.assertFalse(set(a)&set(v.noise_seeds(2,'nominal',1.,1000)))
    def test_family_correction_widens_interval(self):
        x=np.linspace(.90,.94,100)
        one=v.mean_interval(x,.92,family=1);many=v.mean_interval(x,.92,family=120)
        self.assertLess(many['lower'],one['lower']);self.assertGreater(many['upper'],one['upper'])
        self.assertEqual(many['mean_status'],'unresolved')
    def test_accept_reject_unresolved(self):
        self.assertEqual(v.mean_interval([.8]*100,.92)['mean_status'],'supported')
        self.assertEqual(v.mean_interval([1.]*100,.92)['mean_status'],'exceeds')
        self.assertEqual(v.mean_interval([.8,1.04]*50,.92)['mean_status'],'unresolved')
    def test_nonfinite_cannot_be_accepted(self):
        with self.assertRaises(ValueError):v.mean_interval([.8,float('nan')],.92)

if __name__=='__main__':unittest.main()
