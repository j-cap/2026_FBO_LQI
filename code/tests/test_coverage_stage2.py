import sys
from pathlib import Path
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from coverage_stage2 import merge_summaries,interpolate,summarize,allocate

class Stage2Checks(unittest.TestCase):
    def test_distributed_matches_centralized(self):
        def row(d,x):return dict(demand=d,plant=dict(physical=True,E=x,S=2*x),prediction=dict(physical=True,E=1.,S=2.),numerical_resolved=True)
        a=[row(1.5,1.01)]*8+[row(4.,1.1)]*4
        b=[row(4.,1.2)]*4+[row(7.,1.3)]*8
        dist=merge_summaries([summarize(a),summarize(b)]);central=summarize(a+b)
        for d in [1.5,2.75,4.,5.5,7.]:
            for m in ['E','S']:self.assertAlmostEqual(interpolate(dist,d)[m],interpolate(central,d)[m])
    def test_nearest_extension_and_middle_bin(self):
        s={1.:dict(n=3,sums=dict(E=0.,S=0.)),2.:dict(n=3,sums=dict(E=3*np.log(2),S=0.))}
        self.assertAlmostEqual(interpolate(s,7.)['E'],2.)
        self.assertAlmostEqual(interpolate(s,.5)['E'],1.)
        self.assertIsNone(interpolate({1.:dict(n=2,sums=dict(E=0.,S=0.))},1.))
    def test_allocations_fixed_budget_and_target_excluded(self):
        import json
        cfg=json.loads((ROOT/'config/runs/coverage_stage2.json').read_text())
        lookup={(v,k,d):dict(vehicle=v,candidate=k,demand=d) for v in ['a','b','target'] for k in range(16) for d in cfg['training_demands']}
        for pair in cfg['patterns'].values():
            pooled=[]
            for v,counts in zip(['a','b'],pair):
                rows=allocate(lookup,v,counts,cfg['training_demands']);pooled+=rows
                self.assertEqual(len(allocate(lookup,v,[2*n for n in counts],cfg['training_demands'])),24)
            self.assertEqual(len(pooled),24);self.assertTrue(all(r['vehicle']!='target' for r in pooled))

if __name__=='__main__':unittest.main()
