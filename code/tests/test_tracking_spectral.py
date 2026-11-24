"""Matched spectral families, pinned public coordinates and A-first scoring."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from test_tracking_hierarchical import fixture as original_fixture
from src import tracking_hierarchical as joint
from src import tracking_spectral as spectral
from src.tracking_kernel_features import basis_bank
from src.tracking_peer_cost_report import evaluate,eligibility_times
from src.tracking_spectral_report import benchmark_readout,paired_comparisons,write_report
import tracking_spectral as runner


def fixture(legacy=False):
    cfg,domain,clients,oracle=original_fixture()
    cfg.update(spectral_ranks=[2,4],primary_rank=2,basis_family='spectral')
    cfg['methods']=(spectral.LEGACY if legacy else [])+spectral.spectral_methods(cfg['spectral_ranks'])
    cfg['comparison_methods']=spectral.LEGACY+spectral.spectral_methods(cfg['spectral_ranks'])
    _,bank=basis_bank(domain,cfg['gp'],cfg['model'],[dict(family='spectral',rank=r) for r in cfg['spectral_ranks']])
    phis={r:bank[f'spectral_{r}']['F']/np.r_[np.sqrt(cfg['gp']['constant_variance']),np.full(r,cfg['gp']['amplitude'])]
          for r in cfg['spectral_ranks']}
    return cfg,domain,clients,oracle,phis


class PinnedBasisTests(unittest.TestCase):
    def test_public_basis_round_trip_and_tamper_rejection(self):
        cfg,domain,_,_,phis=fixture()
        expected={str(r):spectral.basis_descriptor(phi) for r,phi in phis.items()}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'public_bases.npz'
            spectral.save_bases(path,phis,domain)
            recovered=spectral.load_bases(path,cfg['spectral_ranks'],domain,expected)
            for r in phis:np.testing.assert_array_equal(phis[r],recovered[r])
            changed={r:p.copy() for r,p in phis.items()};changed[2][0,1]+=.01
            spectral.save_bases(path,changed,domain)
            with self.assertRaisesRegex(RuntimeError,'hash'):spectral.load_bases(path,cfg['spectral_ranks'],domain,expected)


class SpectralProtocolTests(unittest.TestCase):
    def test_legacy_methods_remain_identical_and_each_spectral_arm_obeys_common_protocol(self):
        cfg,domain,clients,oracle,phis=fixture(legacy=True)
        old=joint.run_fleet(domain,dict(cfg,methods=spectral.LEGACY),clients,1101,oracle)
        new=spectral.run_fleet(domain,cfg,clients,1101,oracle,phis)
        spectral.validate_state(new,domain,cfg,clients,1101,phis,complete=True)
        spectral.replay_methods(new,domain,cfg,clients,1101,phis)
        for method in spectral.LEGACY:self.assertEqual(old['methods'][method],new['methods'][method])
        for rank in cfg['spectral_ranks']:
            for method in [spectral.shared(rank),spectral.local(rank)]:
                data=new['methods'][method]
                self.assertEqual(len(data['events']),18)
                self.assertTrue(all(e['method']==method for e in data['events']))
                self.assertTrue(any(e['mode']=='peer_gain_verification' for e in data['events']))
                self.assertEqual(data['snapshots']['18']['model_payload_bytes'],data['learning'][-2]['cumulative_model_payload_bytes'])
                self.assertEqual(data['model_payload_bytes']>0,method==spectral.shared(rank))
                self.assertEqual(len(data['weights']),rank+1)

    def test_resume_matches_uninterrupted_run_and_future_posteriors_are_rejected(self):
        cfg,domain,clients,oracle,phis=fixture()
        full=spectral.run_fleet(domain,cfg,clients,1100,oracle,phis)
        partial=spectral.run_fleet(domain,cfg,clients,1100,oracle,phis,stop_round=3)
        partial=json.loads(json.dumps(partial,allow_nan=False))
        resumed=spectral.run_fleet(domain,cfg,clients,1100,oracle,phis,state=partial)
        self.assertEqual(full,resumed)
        bad=deepcopy(partial)
        bad['methods'][spectral.shared(2)]['cost_posterior']['information_round']=99
        with self.assertRaisesRegex(RuntimeError,'future'):spectral.validate_state(bad,domain,cfg,clients,1100,phis)
        wrong={r:p.copy() for r,p in phis.items()};wrong[2][:,1]*=1.1
        with self.assertRaisesRegex(RuntimeError,'posterior'):spectral.validate_state(partial,domain,cfg,clients,1100,wrong)

    def test_no_cross_arm_cost_information_and_peer_priority_precedes_model(self):
        cfg,domain,clients,oracle,phis=fixture()
        state=spectral.run_fleet(domain,cfg,clients,1101,oracle,phis,stop_round=1)
        a=spectral.plan_round(deepcopy(state),domain,cfg,clients,1101,phis)
        changed=deepcopy(state)
        data=changed['methods'][spectral.shared(2)]
        data['cost_posterior']['mean']=[[1e6]*len(domain.actions) for _ in clients]
        for h in data['histories']:h[0]['J']=1000.
        b=spectral.plan_round(changed,domain,cfg,clients,1101,phis)
        for method in [spectral.local(2),spectral.shared(4),spectral.local(4)]:
            self.assertEqual([p for p in a if p['method']==method],[p for p in b if p['method']==method])
        self.assertTrue(any(p['mode']=='peer_gain_verification' for p in b))
        spectral.validate_state(state,domain,cfg,clients,1101,phis)

    def test_failed_verifications_are_charged_and_all_failed_histories_are_valid(self):
        cfg,domain,clients,oracle,phis=fixture()
        def failures(plans,cache):
            observations=oracle(plans,cache)
            for p,o in zip(plans,observations):
                if p['client']!=1:o.update(eligible=False,J=None,feasible=False)
            return observations
        data=spectral.run_fleet(domain,cfg,clients,1101,failures,phis)
        spectral.validate_state(data,domain,cfg,clients,1101,phis,complete=True)
        for record in data['methods'].values():
            self.assertEqual(record['snapshots']['18']['recommendations'][0],[None]*2)
            self.assertTrue(any(e['mode']=='peer_gain_verification' and not e['eligible'] for e in record['events']))
        def no_labels(plans,cache):return [dict(eligible=False,J=None,complete=True,feasible=False,
            integration_resolved=True,substeps=2) for _ in plans]
        empty=spectral.run_fleet(domain,cfg,clients,1100,no_labels,phis)
        spectral.validate_state(empty,domain,cfg,clients,1100,phis,complete=True)
        self.assertTrue(all(d['payload_bytes']==0 for d in empty['methods'].values()))


class SpectralScoringTests(unittest.TestCase):
    def test_primary_rank_cannot_be_replaced_by_better_accuracy_control_result(self):
        cfg,domain,clients,oracle,phis=fixture(legacy=True)
        data=spectral.run_fleet(domain,cfg,clients,1101,oracle,phis)
        job=dict(condition='homogeneous',seed=1101,clients=clients,state=data)
        refs={('target',c):dict(J=1.,eligible=True) for c in domain.contexts}
        val={('target',c,k):dict(J=1.,eligible=True) for c,k in domain.actions}
        trials,contexts,queries,learning=evaluate([job],domain,cfg,(refs,val))
        last=trials.fleet_budget==18
        trials.loc[last&(trials.method==spectral.LEGACY[0]),'context_success']=.75
        trials.loc[last&(trials.method==spectral.shared(2)),'context_success']=.5
        trials.loc[last&(trials.method==spectral.shared(4)),'context_success']=1.
        paired=paired_comparisons(trials,cfg);seeds=paired_comparisons(trials,cfg,True)
        readout=benchmark_readout(paired,seeds,18)
        primary=readout[readout.role=='primary'].iloc[0]
        self.assertEqual(primary.method,spectral.shared(2));self.assertFalse(primary.primary_context_metric_improved)
        accuracy=readout[(readout.role=='accuracy_control')&(readout.baseline==spectral.LEGACY[0])].iloc[0]
        self.assertTrue(accuracy.primary_context_metric_improved)
        self.assertEqual(accuracy.seeds_with_positive_primary,1)
        times,_=eligibility_times([job],domain,cfg)
        with tempfile.TemporaryDirectory() as directory:
            write_report(Path(directory),trials,contexts,queries,learning,times,cfg,'smoke',len(domain.actions))
            html=(Path(directory)/'report.html').read_text()
            self.assertLess(html.index('Primary: D256'),html.index('Accuracy control: D384'))
            self.assertTrue((Path(directory)/'benchmark_readout.csv').exists())

    def test_smoke_or_missing_results_cannot_be_collected(self):
        with self.assertRaisesRegex(RuntimeError,'Smoke'):runner.collect(dict(specification={'mode':'smoke'}))
        with patch.object(runner,'analyze',side_effect=RuntimeError('Missing checkpoint')):
            with patch.object(runner.subprocess,'run') as command:
                with self.assertRaisesRegex(RuntimeError,'Missing'):runner.collect(dict(specification={'mode':'full'}))
                command.assert_not_called()


if __name__=='__main__':unittest.main()
