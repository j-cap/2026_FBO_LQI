import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from limited_excitation import (actual_ay, measurement, separation, whiten_ar1,
                                acceptable_sets, nominal_params, controller_rollout)
from mismatch_pilot import rollout
from src.actuator_bandwidth import augment_discrete, design
from src.ifac_bridge import build_bicycle_beta_r

class LimitedExcitationChecks(unittest.TestCase):
    def test_ar1_distance_matches_full_covariance(self):
        n=8;rho=.7;sigma=np.array([.1,.2,.3]);z=np.arange(n*3).reshape(n,3)/100
        cov=rho**abs(np.arange(n)[:,None]-np.arange(n)[None,:])
        expected=sum((z[:,j]/sigma[j])@np.linalg.solve(cov,z[:,j]/sigma[j]) for j in range(3))
        self.assertAlmostEqual(np.sum(whiten_ar1(z,sigma,rho)**2),expected)
        self.assertEqual(separation(z,z,sigma,rho)['ideal_equal_prior_error'],.5)

    def test_actual_acceleration_not_yaw_equivalent(self):
        p=nominal_params();x=np.array([0.,0.,.01])
        self.assertGreater(actual_ay(x,p,1.),0.)
        self.assertEqual(x[1]*p.U,0.)
        # Same tangent stiffness despite different saturation levels.
        x[2]=1e-7
        self.assertAlmostEqual(actual_ay(x,p,.8),actual_ay(x,p,1.2),places=10)

    def test_common_sets_distinguish_feasibility_and_effort(self):
        d=pd.DataFrame([dict(mu=mu,candidate=k,E=1.,S=s,physical=True,resolved=True)
                        for mu,ss in [(0.8,[1.,2.]),(1.2,[2.,1.])] for k,s in enumerate(ss)])
        a,f,b=acceptable_sets(d,[.8,1.2],1.1,10.)
        self.assertFalse(a[.8]&a[1.2]);self.assertEqual(f[.8]&f[1.2],{0,1})
        d.loc[(d.mu==.8)&(d.candidate==0),'resolved']=False
        a,f,b=acceptable_sets(d,[.8,1.2],1.1,10.)
        self.assertEqual(f[.8],{1})

    def test_mu_one_reproduces_corrected_evaluator(self):
        import yaml
        cc=yaml.safe_load((ROOT/'config/controller.yaml').read_text())
        p=nominal_params();a,b,_,_=build_bicycle_beta_r(p);cfg={'ts':.05,'tau_true':.2}
        ad,bd=augment_discrete(a,b,.2,.05)
        K,kr,_=design(ad,bd,[0.,.3,.3],cc['state_scale'],.05,cc['fixed_R'])
        ref=.03*np.sin(np.arange(30)*.2)
        old=rollout(p,cfg,cc,a,b,.2,K,kr,ref,True,10)
        new=controller_rollout(p,cfg,cc,K,kr,ref,1.,10)
        self.assertEqual(old['physical'],new['physical'])
        for k in ['E','S']:self.assertAlmostEqual(old[k],new[k],places=12)

if __name__=='__main__':unittest.main()
