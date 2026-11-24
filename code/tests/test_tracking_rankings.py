"""Ranking decomposition, factual replay and a matched exact-kernel reference."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import analyze_tracking_rankings as diagnostic
from test_tracking_hierarchical import fixture
from src.tracking_hierarchical import cost_posterior,run_fleet


class RankingTests(unittest.TestCase):
    def test_score_terms_masks_and_exploration_are_distinct(self):
        cfg=dict(confidence_multiplier=2.,feasibility_penalty=1.,probability_floor=1e-6)
        mean=np.array([0.,1.,-100.]);variance=np.array([0.,4.,100.]);p=np.array([.5,.5,0.])
        terms=diagnostic.ranked_terms(mean,variance,p,np.array([True,True,False]),cfg)
        self.assertEqual(terms['ranks']['mean'][0],1)
        self.assertEqual(terms['ranks']['score'][1],1)
        self.assertTrue(np.isnan(terms['ranks']['score'][2]))
        self.assertNotIn(2,terms['ties'])
        margin=terms['score'][0]-terms['score'][1]
        components=sum(terms[t][0]-terms[t][1] for t in ('mean','exploration','penalty'))
        self.assertAlmostEqual(margin,components)
        self.assertTrue(np.isfinite(terms['penalty']).all())

    def test_exact_kernel_control_matches_feature_model_when_kernel_is_identical(self):
        rng=np.random.default_rng(12)
        phi=rng.normal(size=(8,5));phi[:,0]=1.
        gp=dict(amplitude=1.,constant_variance=.25,noise_std_log_J=.05)
        cfg=dict(sharing_fractions=[0.,.5,1.],prior_weights=[1,1,1])
        h=[[dict(action=a,eligible=True,J=float(np.exp(rng.normal()))) for a in ids]
           for ids in [[0,1,3],[0,2,6],[]]]
        h[0].append(dict(action=7,eligible=False,J=None))
        F=phi*np.r_[.5,np.ones(4)]
        exact=diagnostic.exact_kernel_posterior(h,F@F.T,cfg,gp)
        finite=cost_posterior(h,phi,gp,cfg)
        for key in ('mean','variance','sharing_weights','expected_sharing_fraction'):
            np.testing.assert_allclose(exact[key],finite[key],rtol=1e-7,atol=1e-8)

    def test_validation_join_cannot_change_predictions_or_ranks(self):
        data=pd.DataFrame([dict(plant_id='plant',context='context',candidate='gain',mean=.3,std=.5,
            score=-.6,score_rank=2.)])
        before=data.copy(deep=True)
        ref={('plant','context'):dict(J=1.)}
        a=diagnostic.attach_validation(data,(ref,{('plant','context','gain'):dict(eligible=True,J=1.)}))
        b=diagnostic.attach_validation(data,(ref,{('plant','context','gain'):dict(eligible=False,J=100.)}))
        pd.testing.assert_frame_equal(data,before)
        pd.testing.assert_frame_equal(a[before.columns],b[before.columns])
        self.assertTrue(a.validation_near_reference.iloc[0]);self.assertFalse(b.validation_near_reference.iloc[0])

    def test_both_factual_models_replay_while_shadow_fits_never_advance_history(self):
        cfg,domain,clients,oracle=fixture(all_methods=True)
        state=run_fleet(domain,cfg,clients,1101,oracle)
        job=dict(condition='homogeneous',seed=1101,clients=clients,state=state)
        context=domain.contexts[0]
        specification=dict(cfg=cfg,domain=domain,reference_candidates={('target',context):domain.candidates[1]})
        before=deepcopy(job)
        with patch.object(diagnostic,'CONTEXT',context):
            for source,method in diagnostic.ARMS.items():
                cases,candidates,_,checked=diagnostic.fit_source(specification,job,source)
                expected=sum(e['mode']=='feasibility_penalized_LCB' for e in state['methods'][method]['events'])
                self.assertEqual(checked,expected)
                self.assertFalse(cases.empty);self.assertFalse(candidates.empty)
                self.assertNotIn('validation_J',candidates)
                if source=='D':self.assertIn(diagnostic.EXACT,cases.model.unique())
        self.assertEqual(before,job)


if __name__=='__main__':unittest.main()
