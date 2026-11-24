"""Fresh fleet draws, unchanged decisions, held-out evaluation and cluster units."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import tracking_confirmation as runner
from src.tracking_confirmation import METHODS, aggregate_candidates, calibration_reference, design, evaluate_job, paired_tables, pinned_design
from src.tracking_online import features
from src.tracking_spectral import basis_descriptor, run_fleet
from src.tracking_task import fleet
from src import tracking_peer_cost as peer
from test_tracking_hierarchical import fixture


def small_run():
    cfg,domain,clients,oracle=fixture()
    cfg.update(methods=METHODS,comparison_methods=METHODS,spectral_ranks=[256],primary_rank=256)
    phis={256:features(domain,dict(cfg['model'],features=256))}
    def observe(plans,cache):
        observations=oracle(plans,cache)
        for p,o in zip(plans,observations):
            key=f'{p["client"]}__{p["local_evaluation"]}__{p["context"]}__{p["candidate"]}'
            cache[key]=dict(noise_seed=p['noise_seed'],client=p['client'],local_evaluation=p['local_evaluation'],
                context=p['context'],result=dict(candidate=p['candidate'],**o))
        return observations
    return cfg,domain,clients,observe,phis


class FreshDesignTests(unittest.TestCase):
    def test_pinned_values_survive_platform_roundoff_but_reject_physical_changes(self):
        current=dict(groups=[dict(mass=1200.,id='client_00')],jobs=[1,2,3])
        saved=deepcopy(current);saved['groups'][0]['mass']+=1e-11
        self.assertEqual(pinned_design(current,saved),saved)
        changed=deepcopy(saved);changed['groups'][0]['mass']+=.01
        with self.assertRaisesRegex(RuntimeError,'values'):pinned_design(current,changed)
        changed=deepcopy(saved);changed['groups'][0]['id']='other'
        with self.assertRaisesRegex(RuntimeError,'identity'):pinned_design(current,changed)

    def test_independent_draws_retain_composition_and_consistent_geometry(self):
        physics=runner.read(ROOT/'config/runs/tracking_diagnostic.json')
        base=fleet(physics);requested=runner.read(runner.CONFIG)
        groups,jobs,draws=design(base,requested)
        again=design(base,requested)
        self.assertEqual((groups,jobs,draws),again)
        self.assertEqual(len(jobs),40);self.assertEqual(len(groups),11)
        self.assertEqual(len(draws),90)
        self.assertEqual(len({j['seed'] for j in jobs}),40)
        self.assertFalse({j['seed'] for j in jobs}&{1100,1101,1102})
        self.assertTrue(all(len(g)==9 for g in groups.values()))
        self.assertEqual(len({json.dumps(c['params'],sort_keys=True) for c in groups['homogeneous']}),1)
        self.assertNotEqual(groups['fleet_00'][0]['params'],groups['fleet_01'][0]['params'])
        for draw in draws:
            b=next(v['params'] for v in base if v['id']==draw['base_vehicle'])
            p,z=draw['params'],draw['draw']
            length_ratio=(p['l_f']+p['l_r'])/(b['l_f']+b['l_r'])
            self.assertAlmostEqual(length_ratio,np.exp(z['log_wheelbase_bound']))
            self.assertAlmostEqual(p['I_z']/b['I_z'],p['m']/b['m']*length_ratio**2*np.exp(z['log_inertia_shape_bound']))
            self.assertGreater(min(p['l_f'],p['l_r'],p['m'],p['I_z']),0)


class ConfirmationProtocolTests(unittest.TestCase):
    def test_A_B_remain_identical_all_four_replay_and_fleet_identity_is_checked(self):
        cfg,domain,clients,oracle,phis=small_run()
        old=peer.run_fleet(domain,dict(cfg,methods=peer.METHODS[:2]),clients,1100,oracle)
        data=run_fleet(domain,cfg,clients,1100,oracle,phis)
        for m in peer.METHODS[:2]:self.assertEqual(old['methods'][m],data['methods'][m])
        job=dict(condition='heterogeneous',fleet_id='f',replicate=0,seed=1100)
        state=dict(cfg=cfg,domain=domain,groups={'f':clients},signature='test',phis=phis,
            public_bases={'256':basis_descriptor(phis[256])})
        result=dict(**job,signature='test',clients=clients,public_bases=state['public_bases'],state=data)
        runner.validate_job(state,job,result,complete=True,replay=True)
        forged=deepcopy(result);forged['fleet_id']='another'
        with self.assertRaisesRegex(RuntimeError,'identity'):runner.validate_job(state,job,forged)
        bad=deepcopy(result)
        key=next(k for k,v in bad['state']['oracle_cache'].items() if v['result']['eligible'])
        bad['state']['oracle_cache'][key]['result']['J']*=2
        with self.assertRaisesRegex(RuntimeError,'cost'):runner.validate_job(state,job,bad)
        checkpoint=run_fleet(domain,cfg,clients,1100,oracle,phis,stop_round=2)
        resumed=run_fleet(domain,cfg,clients,1100,oracle,phis,state=json.loads(json.dumps(checkpoint)))
        self.assertEqual(data,resumed)

    def test_validation_union_contains_early_incumbents_and_calibration_winner(self):
        state=dict(domain=type('D',(),{'contexts':['c']})())
        result=dict(fleet_id='f',clients=[dict(plant_id='p')],state=dict(methods={
            'm':dict(snapshots={'1':dict(recommendations=[['early']]),'2':dict(recommendations=[['final']])})}))
        chosen={('f','p','c'):'reference'}
        selection=runner.validation_selection(state,[result],chosen)
        self.assertEqual(selection[('f','p','c')],{'early','final','reference'})

    def test_missing_reference_retains_context_and_policy_denominators(self):
        cfg,domain,clients,_,phis=small_run()
        def observed(plans,cache):return [dict(eligible=True,J=1.,complete=True,feasible=True,integration_resolved=True,substeps=2) for p in plans]
        data=run_fleet(domain,cfg,clients,1100,observed,phis)
        plant=clients[0]['plant_id'];contexts=domain.contexts
        refs={(plant,c):dict(candidate=domain.candidates[0],eligible=i!=0,J=1.) for i,c in enumerate(contexts)}
        val={(plant,c,k):dict(eligible=True,J=1.) for c,k in domain.actions}
        job=dict(condition='homogeneous',fleet_id='h',replicate=0,seed=1100)
        trials,cells,_,_=evaluate_job(job,dict(clients=clients,state=data),domain,cfg,refs,val)
        final=trials[trials.fleet_budget==cfg['fleet_budgets'][-1]]
        self.assertTrue(final.fully_feasible.all())
        self.assertTrue((final.context_success==.5).all())
        self.assertTrue((final.reference_coverage==.5).all())
        self.assertEqual(len(cells),len(cfg['fleet_budgets'])*len(METHODS)*len(clients)*len(contexts))
        missing=cells[cells.context==contexts[0]]
        self.assertFalse(missing.success.any());self.assertTrue(missing.relative_gap.isna().all())

    def test_reference_is_calibration_only_and_failed_repeat_is_not_dropped(self):
        batches=[dict(rows=[dict(candidate='a',complete=True,feasible=True,integration_resolved=True,J=1.),
            dict(candidate='b',complete=True,feasible=True,integration_resolved=True,J=2.)]),
            dict(rows=[dict(candidate='a',complete=True,feasible=False,integration_resolved=True,J=.5),
            dict(candidate='b',complete=True,feasible=True,integration_resolved=True,J=2.)])]
        aggregated=aggregate_candidates(batches)
        self.assertFalse(aggregated['a']['eligible'])
        self.assertEqual(calibration_reference(aggregated),'b')
        self.assertIsNone(calibration_reference({'a':dict(eligible=False,J=None)}))

    def test_incomplete_or_smoke_campaign_cannot_be_collected(self):
        with self.assertRaisesRegex(RuntimeError,'Smoke'):runner.collect(dict(specification={'mode':'smoke'}),1)
        with patch.object(runner,'analyze',side_effect=RuntimeError('Missing checkpoint')):
            with patch.object(runner.subprocess,'run') as command:
                with self.assertRaisesRegex(RuntimeError,'Missing'):runner.collect(dict(specification={'mode':'full'}),1)
                command.assert_not_called()


class ReplicationUnitTests(unittest.TestCase):
    def test_three_runs_and_all_clients_are_clustered_within_each_fleet(self):
        records=[]
        for condition in ['homogeneous','heterogeneous']:
            for fleet in range(10):
                for rep in range(3 if condition=='heterogeneous' else 1):
                    for client in range(2):
                        for method in METHODS:
                            records.append(dict(condition=condition,fleet_budget=216,method=method,
                                fleet_id=f'f{fleet}' if condition=='heterogeneous' else 'homogeneous',
                                seed=100*fleet+rep,client=f'c{client}',context_success=.6 if method==METHODS[2] else .4,
                                fully_feasible=True,J=.9 if method==METHODS[2] else 1.,reference_coverage=1.))
        cfg=dict(fleet_budgets=[216]);requested=dict(bootstrap_seed=2,bootstrap_repeats=10000)
        paired,units=paired_tables(pd.DataFrame(records),cfg,requested,'full')
        self.assertTrue((paired.replication_units==10).all())
        self.assertTrue(paired.confirmation_supported.all())
        self.assertTrue((units[units.condition=='heterogeneous'].programme_runs==3).all())
        self.assertTrue((units[units.condition=='homogeneous'].programme_runs==1).all())
        np.testing.assert_allclose(paired.unit_bootstrap_lower,.2)
        early=pd.DataFrame(records);early['fleet_budget']=108
        both=pd.concat([early,pd.DataFrame(records)],ignore_index=True)
        budgets,_=paired_tables(both,dict(fleet_budgets=[108,216]),requested,'full')
        self.assertFalse(budgets[budgets.fleet_budget==108].confirmation_supported.any())
        self.assertTrue(budgets[budgets.fleet_budget==216].confirmation_supported.all())
        bad=pd.DataFrame(records);bad.loc[0,'reference_coverage']=.5
        limited,_=paired_tables(bad,cfg,requested,'full')
        self.assertFalse(limited[(limited.condition=='homogeneous')&(limited.baseline==METHODS[0])].iloc[0].confirmation_supported)


if __name__=='__main__':unittest.main()
