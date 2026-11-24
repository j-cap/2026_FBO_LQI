"""Common peer priority, independent fleet histories and matched offline scoring."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
from test_tracking_online_lcb import fixture as old_fixture
from src import tracking_online_lcb as legacy
from src import tracking_peer_cost as peer
from src.tracking_peer_cost_report import evaluate, mechanism_tables, paired_comparisons, summarize
import tracking_peer_cost as runner


def fixture():
    cfg, domain, clients, oracle = old_fixture()
    cfg['methods'] = peer.METHODS.copy()
    return cfg, domain, clients, oracle


class PeerProtocolTests(unittest.TestCase):
    def test_gp_reference_reproduces_frozen_gain_sharing(self):
        cfg, domain, clients, oracle = fixture()
        old = legacy.run_fleet(domain, cfg, clients, 1101, 0, oracle)
        new = peer.run_fleet(domain, cfg, clients, 1101, oracle)
        record = runner.compare_reference(dict(condition='homogeneous', seed=1101, clients=clients, state=new),
            dict(clients=clients, state=old))
        self.assertTrue(record['matched'])
        np.testing.assert_array_equal(new['methods']['peer_gp_bo']['gp_parameters'],
                                      old['methods']['online_gain_sharing']['gp_parameters'])

    def test_every_arm_uses_peer_priority_even_with_extreme_shared_predictions(self):
        cfg, domain, clients, oracle = fixture(); phi = peer.features(domain, cfg['model'])
        state = peer.fresh_state(domain, cfg, clients)
        first = peer.plan_round(state, domain, cfg, clients, 1101, phi)
        peer.commit_round(state, first, oracle(first, {}), domain, cfg, clients, phi)
        state['methods']['peer_federated_cost_bo']['weights'] = [1e6]*phi.shape[1]
        with patch.object(peer, 'choose', wraps=peer.choose) as choose:
            plans = peer.plan_round(state, domain, cfg, clients, 1101, phi)
        candidates = []
        for method in peer.METHODS:
            verified = [p for p in plans if p['method'] == method and p['mode'] == 'peer_gain_verification']
            self.assertTrue(verified)
            candidates.append([(p['client'], p['candidate'], p['peer_donor']) for p in verified])
            for p in verified:
                self.assertEqual(p['peer_donor_query'], 1)
                self.assertEqual(p['peer_source_mode'], 'factory_initialization')
        self.assertEqual(candidates[0], candidates[1]); self.assertEqual(candidates[1], candidates[2])
        self.assertEqual(len(choose.call_args_list), sum(p['mode'] == 'feasibility_penalized_LCB' for p in plans))

    def test_fallback_mean_is_available_in_round_two(self):
        cfg, domain, clients, oracle = fixture(); phi = peer.features(domain, cfg['model'])
        state = peer.fresh_state(domain, cfg, clients)
        plans = peer.plan_round(state, domain, cfg, clients, 1100, phi)
        peer.commit_round(state, plans, oracle(plans, {}), domain, cfg, clients, phi)
        expected = phi@np.asarray(state['methods']['peer_federated_cost_bo']['weights'])
        self.assertGreater(np.linalg.norm(expected), .01)
        with patch.object(peer, 'choose', wraps=peer.choose) as choose:
            second = peer.plan_round(state, domain, cfg, clients, 1100, phi)
        self.assertTrue(all(p['mode'] == 'feasibility_penalized_LCB' for p in second))
        for plan, call in zip(second, choose.call_args_list):
            self.assertEqual(len(call.args[0]), 1)
            self.assertEqual(plan['information_round'], 1)
            if plan['method'] == 'peer_federated_cost_bo':
                np.testing.assert_array_equal(call.args[3], expected)
            elif plan['method'] == 'peer_gp_bo':
                np.testing.assert_array_equal(call.args[3], 0.)

    def test_other_arm_records_and_weights_cannot_change_gp_arm_choices(self):
        cfg, domain, clients, oracle = fixture(); phi = peer.features(domain, cfg['model'])
        state = peer.fresh_state(domain, cfg, clients)
        first = peer.plan_round(state, domain, cfg, clients, 1100, phi)
        peer.commit_round(state, first, oracle(first, {}), domain, cfg, clients, phi)
        a = peer.plan_round(deepcopy(state), domain, cfg, clients, 1100, phi)
        changed = deepcopy(state)
        changed['methods']['peer_federated_cost_bo']['weights'] = [10.]*phi.shape[1]
        for h in changed['methods']['peer_federated_cost_bo']['histories']:
            for observation in h:
                observation['J'] = 100.
        b = peer.plan_round(changed, domain, cfg, clients, 1100, phi)
        self.assertEqual([p for p in a if p['method'] == 'peer_gp_bo'],
                         [p for p in b if p['method'] == 'peer_gp_bo'])

    def test_peer_verification_observations_enter_federated_training(self):
        cfg, domain, clients, oracle = fixture(); phi = peer.features(domain, cfg['model'])
        state = peer.fresh_state(domain, cfg, clients)
        for _ in range(2):
            plans = peer.plan_round(state, domain, cfg, clients, 1101, phi)
            observed = oracle(plans, {})
            for p, o in zip(plans, observed):
                if p['mode'] == 'peer_gain_verification':
                    o['J'] = .25
            peer.commit_round(state, plans, observed, domain, cfg, clients, phi)
        data = state['methods']['peer_federated_cost_bo']
        self.assertEqual(data['learning'][-1]['eligible_training_counts'], [2]*3)
        self.assertTrue(any(o['J'] == .25 for h in data['histories'] for o in h))
        expected_models = [peer.LocalCostModel(h, phi, cfg['model']['ridge']) for h in data['histories']]
        self.assertEqual(sum(m.count for m in expected_models), 6)
        self.assertGreater(data['learning'][-1]['model_payload_bytes'], 0)

    def test_failed_verifications_are_charged_and_never_become_recommendations(self):
        cfg, domain, clients, oracle = fixture()
        def failed_target(plans, cache):
            observations = oracle(plans, cache)
            for p, o in zip(plans, observations):
                if p['client'] != 1:
                    o.update(eligible=False, J=None, feasible=False)
            return observations
        state = peer.run_fleet(domain, cfg, clients, 1101, failed_target)
        peer.validate_state(state, domain, cfg, clients, 1101, complete=True)
        for data in state['methods'].values():
            self.assertEqual(len(data['events']), 18)
            self.assertTrue(any(e['mode'] == 'peer_gain_verification' and not e['eligible'] for e in data['events']))
            self.assertEqual(data['snapshots']['18']['recommendations'][0], [None]*2)
            self.assertEqual(data['snapshots']['18']['recommendations'][2], [None]*2)

    def test_empty_eligible_history_does_not_delay_bo_or_create_free_gains(self):
        cfg, domain, clients, _ = fixture()
        def failed(plans, cache):
            return [dict(eligible=False, J=None, complete=True, feasible=False,
                integration_resolved=True, substeps=2) for _ in plans]
        state = peer.run_fleet(domain, cfg, clients, 1100, failed)
        peer.validate_state(state, domain, cfg, clients, 1100, complete=True)
        for data in state['methods'].values():
            self.assertEqual(len(data['events']), 18)
            self.assertEqual(data['payload_bytes'], 0)
            self.assertTrue(all(e['mode'] == 'feasibility_penalized_LCB' for e in data['events'] if e['round'] == 2))
            self.assertTrue(all(c is None for row in data['snapshots']['18']['recommendations'] for c in row))

    def test_payload_components_and_query_snapshots_exclude_unused_training(self):
        cfg, domain, clients, oracle = fixture()
        state = peer.run_fleet(domain, cfg, clients, 1101, oracle)
        peer.validate_state(state, domain, cfg, clients, 1101, complete=True)
        for method, data in state['methods'].items():
            self.assertEqual(data['snapshots']['3']['queries_per_client'], [1]*3)
            self.assertEqual(data['snapshots']['3']['payload_bytes'], 0)
            self.assertEqual(data['snapshots']['18']['round'], 6)
            self.assertEqual(data['payload_bytes'], data['peer_payload_bytes']+data['model_payload_bytes'])
            self.assertEqual(data['snapshots']['18']['payload_bytes'], data['learning'][-2]['cumulative_payload_bytes'])
            self.assertEqual(data['model_payload_bytes'] > 0, method == 'peer_federated_cost_bo')

    def test_json_checkpoint_resume_matches_uninterrupted_run(self):
        cfg, domain, clients, oracle = fixture()
        full = peer.run_fleet(domain, cfg, clients, 1101, oracle)
        partial = []
        def stop(state):
            if state['round'] == 3:
                partial.append(json.loads(json.dumps(state, allow_nan=False)))
                raise InterruptedError('checkpoint')
        with self.assertRaises(InterruptedError):
            peer.run_fleet(domain, cfg, clients, 1101, oracle, stop)
        peer.validate_state(partial[0], domain, cfg, clients, 1101)
        resumed = peer.run_fleet(domain, cfg, clients, 1101, oracle, state=partial[0])
        self.assertEqual(full, resumed)

    def test_forged_peer_sources_future_information_and_snapshots_are_rejected(self):
        cfg, domain, clients, oracle = fixture()
        state = peer.run_fleet(domain, cfg, clients, 1101, oracle)
        for key, value in [('peer_donor_query', 100), ('information_round', 100), ('mode', 'feasibility_penalized_LCB')]:
            forged = deepcopy(state)
            event = next(e for e in forged['methods']['peer_federated_cost_bo']['events'] if e['mode'] == 'peer_gain_verification')
            event[key] = value
            with self.assertRaises(RuntimeError):
                peer.validate_state(forged, domain, cfg, clients, 1101, complete=True)
        forged = deepcopy(state)
        forged['methods']['peer_local_cost_bo']['snapshots']['3']['model_payload_bytes'] = 1
        with self.assertRaisesRegex(RuntimeError, 'snapshot'):
            peer.validate_state(forged, domain, cfg, clients, 1101, complete=True)


class ScoringAndCollectionTests(unittest.TestCase):
    def test_early_missing_contexts_and_paired_cost_denominators_are_retained(self):
        cfg, domain, clients, oracle = fixture()
        data = peer.run_fleet(domain, cfg, clients, 1101, oracle)
        reference = {('target', c): dict(J=1., eligible=True) for c in domain.contexts}
        val = {('target', c, k): dict(J=1., eligible=True) for c, k in domain.actions}
        job = dict(condition='homogeneous', seed=1101, clients=clients, state=data)
        before = deepcopy(data)
        trials, cells, queries, _ = evaluate([job], domain, cfg, (reference, val))
        self.assertEqual(len(trials), 3*3*5); self.assertEqual(len(cells), 3*3*5*2)
        self.assertTrue((trials[trials.fleet_budget == 3].context_success == .5).all())
        self.assertTrue(trials[trials.fleet_budget == 3].J.isna().all())
        comparisons = paired_comparisons(trials)
        self.assertTrue((comparisons[comparisons.fleet_budget == 3].jointly_feasible_pairs == 0).all())
        self.assertTrue((comparisons[comparisons.fleet_budget == 18].jointly_feasible_pairs == 3).all())
        self.assertIn('mean_model_payload_bytes', summarize(trials))
        modes, peers, transfers = mechanism_tables(queries)
        self.assertEqual(int(modes.queries.sum()), 54)
        self.assertTrue((peers.peer_donor_query < peers['round']).all())
        self.assertTrue((transfers.first_target_round > transfers.peer_donor_query).all())
        self.assertEqual(before, data)

    def test_reference_forgery_and_incomplete_smoke_collection_do_not_stage(self):
        cfg, domain, clients, oracle = fixture()
        old = legacy.run_fleet(domain, cfg, clients, 1101, 0, oracle)
        new = peer.run_fleet(domain, cfg, clients, 1101, oracle)
        new['methods']['peer_gp_bo']['events'][0]['candidate'] = 'g4'
        with self.assertRaisesRegex(RuntimeError, 'reference differs'):
            runner.compare_reference(dict(clients=clients, state=new), dict(clients=clients, state=old))
        with self.assertRaisesRegex(RuntimeError, 'Smoke'):
            runner.collect(dict(specification={'mode': 'smoke'}))
        with patch.object(runner, 'analyze', side_effect=RuntimeError('Missing checkpoint')):
            with patch.object(runner.subprocess, 'run') as proc:
                with self.assertRaisesRegex(RuntimeError, 'Missing'):
                    runner.collect(dict(specification={'mode': 'full'}))
                proc.assert_not_called()


if __name__ == '__main__':
    unittest.main()
