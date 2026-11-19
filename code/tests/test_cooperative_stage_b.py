import importlib.util
from pathlib import Path
import unittest
import numpy as np
spec=importlib.util.spec_from_file_location('b',Path(__file__).resolve().parents[1]/'scripts/cooperative_stage_b.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
CFG=dict(search_query_cap=6,source_query_budget=2,repair_query_budget=2,gp_prior_std=.5,gp_length_scale=.3,lcb_multiplier=2.)
class PolicyTests(unittest.TestCase):
    def policy(self,value):
        calls=[]
        def query(c,k,phase,index):
            calls.append((c,k));return dict(condition=c,candidate=k,score=value(c,k),variance=.001,valid=True,rollouts=40)
        p=b.Policy(np.array([[.1,.2,.3],[.3,.4,.5],[.8,.9,.6]]),2,CFG,query)
        return p,calls
    def test_cache_does_not_charge_twice(self):
        p,calls=self.policy(lambda c,k:.8);p.get(0,0,'one');p.get(0,0,'two')
        self.assertEqual(len(calls),1);self.assertEqual(len(p.obs),1)
    def test_all_methods_respect_cap_and_fully_screen_success(self):
        for method in b.METHODS:
            p,calls=self.policy(lambda c,k: .8 if k==1 else 1.2)
            k,state=p.run(method,0)
            self.assertLessEqual(len(calls),CFG['search_query_cap'])
            self.assertEqual(len(calls),len(set(calls)))
            if state=='pass':self.assertTrue(all(p.cache[c,k]['score']<=1 for c in range(2)))
    def test_repair_queries_failing_condition(self):
        p,calls=self.policy(lambda c,k: .5 if c==0 and k==0 else 1.2 if c==1 and k==0 else .8)
        k,state=p.run('repair',0)
        self.assertEqual(k,1);self.assertEqual(state,'pass');self.assertIn((1,1),calls)
    def test_seed_separation(self):
        s=b.bank_seeds(701,0,'search',100,set());a=b.bank_seeds(701,0,'audit',100,set())
        self.assertFalse(set(s)&set(a));self.assertEqual(len(set(s)),100)
        self.assertNotIn(s[0],b.bank_seeds(701,0,'search',100,{s[0]}))
    def test_trial_plan_counts(self):
        cfg=dict(optimizer_seeds=[701,702,703],fleet_seeds=[1,2],clients=['a','b','c'])
        self.assertEqual(len(b.trial_plans(cfg,'pilot')),14);self.assertEqual(len(b.trial_plans(cfg,'full')),42)
if __name__=='__main__':unittest.main()
