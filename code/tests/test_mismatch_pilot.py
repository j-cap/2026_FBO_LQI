"""Checks for the numerical and identification mechanisms used by the pilot."""
import sys
from pathlib import Path
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from mismatch_pilot import identify
from src.ifac_bridge import nominal_params, build_bicycle_beta_r
from src.actuator_bandwidth import step, augment_discrete, design
from src.actuator_antiwindup import integral_update

class PilotChecks(unittest.TestCase):
    def test_small_signal_plant_matches_linear_model(self):
        p=nominal_params();a,b,_,_=build_bicycle_beta_r(p)
        ad,bd=augment_discrete(a,b,.2,.05)
        x=np.array([1e-7,2e-7,1e-7]);u=2e-7;actual=x.copy()
        for _ in range(40):actual=step(actual,u,p,.2,.05/40,1.)
        np.testing.assert_allclose(actual,ad@x+bd[:,0]*u,rtol=1e-6,atol=1e-12)

    def test_gentle_output_error_identification(self):
        cfg=dict(ts=.05,identification_duration=20.,identification_amplitude_deg=.4,tau_true=.2)
        p=nominal_params();a,b,tau,fit,_,_=identify(cfg,p)
        self.assertLess(abs(tau-.2),.002)
        self.assertLess(abs(fit['Cf']/p.Cf-1),.03)
        self.assertLess(abs(fit['Cr']/p.Cr-1),.03)
        self.assertLess(fit['normalized_fit_rmse'],.02)

    def test_zero_reference_equilibrium(self):
        p=nominal_params();a,b,_,_=build_bicycle_beta_r(p)
        ad,bd=augment_discrete(a,b,.2,.05)
        K,kr,_=design(ad,bd,[0,0,0],[.053229,.575573,.139862],.05)
        x=np.zeros(3);eta=0.
        for _ in range(20):
            raw=-K[:3]@x-K[3]*eta
            eta=integral_update(eta,x[1],raw,raw,K[3],.05)
            x=step(x,raw,p,.2,.005,1.)
        np.testing.assert_array_equal(x,np.zeros(3));self.assertEqual(eta,0.)

if __name__=='__main__':unittest.main()
