"""Cold-start information barriers, FedAvg algebra and fair fleet accounting."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
from test_tracking_budget import toy
from src.tracking_online import (METHODS, LocalCostModel, centralized_fit, features, federated_fit,
    fresh_state, optimization_gap, peer_candidate, program, run_fleet, seed_for, validate_state)
from src.tracking_task import measurement_noise
from src.tracking_online_report import evaluate
import tracking_online as runner


def fixture():
    cfg, domain, _, _ = toy()
    model = dict(features=8, feature_seed=314159, length_scales=cfg['gp']['length_scales'],
                 ridge=.02, learning_rate=.2, local_steps=1, server_rounds=8)
    cfg.update(queries_per_client=6, fleet_budgets=[6, 12, 18], scout_search_queries=2,
               model=model, methods=METHODS.copy())
    clients = [dict(id=f'client_{i:02d}', plant_id='target', family='A', params={}) for i in range(3)]
    def oracle(plans, cache):
        return [dict(eligible=True, J=float(1.+.2*(p['action'] % 5-2)**2), complete=True,
                     feasible=True, integration_resolved=True, substeps=2) for p in plans]
    return cfg, domain, clients, oracle


class RegressionTests(unittest.TestCase):
    def test_features_are_public_deterministic_and_data_independent(self):
        cfg, domain, _, _ = fixture()
        a = features(domain, cfg['model'])
        np.testing.assert_array_equal(a, features(domain, cfg['model']))
        self.assertEqual(a.shape, (len(domain.actions), 9))
        np.testing.assert_array_equal(a[:, 0], 1.)

    def test_one_step_fedavg_equals_weighted_pooled_gradient(self):
        cfg, domain, _, _ = fixture(); phi = features(domain, cfg['model'])
        histories = [[dict(action=i, eligible=True, J=1.+i/10) for i in indices]
                     for indices in [[0, 1], [2, 3, 4]]]
        models = [LocalCostModel(h, phi, cfg['model']['ridge']) for h in histories]
        conf = dict(cfg['model'], server_rounds=1)
        start = np.arange(phi.shape[1])/10
        actual, payload = federated_fit(models, start, conf)
        A = sum(m.count*m.A for m in models)/5
        b = sum(m.count*m.b for m in models)/5
        expected = start-2*conf['learning_rate']*(A@start-b)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
        self.assertEqual(payload, 2*2*phi.shape[1]*8)

    def test_central_reference_matches_same_regression_objective(self):
        cfg, domain, _, _ = fixture(); phi = features(domain, cfg['model'])
        h = [dict(action=i, eligible=True, J=1.+i/10) for i in range(5)]
        split = [LocalCostModel(h[:2], phi, .02), LocalCostModel(h[2:], phi, .02)]
        pooled = LocalCostModel(h, phi, .02).fit()
        np.testing.assert_allclose(centralized_fit(split), pooled, atol=1e-12)
        self.assertAlmostEqual(optimization_gap(split, pooled), 0., places=12)
        fitted, _ = federated_fit(split, np.zeros(phi.shape[1]), dict(cfg['model'], server_rounds=1500))
        np.testing.assert_allclose(phi@fitted, phi@pooled, atol=1e-7)

    def test_failed_costs_do_not_enter_regression_and_empty_model_is_neutral(self):
        cfg, domain, _, _ = fixture(); phi = features(domain, cfg['model'])
        a = LocalCostModel([], phi, .02)
        b = LocalCostModel([dict(action=0, eligible=False, J=None)], phi, .02)
        np.testing.assert_array_equal(a.fit(), np.zeros(phi.shape[1]))
        np.testing.assert_array_equal(a.A, b.A)
        np.testing.assert_array_equal(a.b, b.b)
        w, payload = federated_fit([a, b], np.ones(phi.shape[1]), cfg['model'])
        np.testing.assert_array_equal(w, 0.); self.assertEqual(payload, 0)


class OnlineProtocolTests(unittest.TestCase):
    def test_programmes_are_balanced_and_noise_is_independent_between_clients(self):
        cfg, domain, clients, _ = fixture()
        for client in clients:
            contexts, gains = program(domain, cfg, 1100, client['id'])
            for j in range(0, cfg['queries_per_client'], len(domain.contexts)):
                self.assertEqual(set(contexts[j:j+2]), set(domain.contexts))
            self.assertEqual(gains[0], domain.candidates[0])
        seeds = [seed_for('episode', 1100, c['id'], q) for c in clients for q in range(6)]
        self.assertEqual(len(seeds), len(set(seeds)))
        noise_cfg = dict(noise_sigma_state=[.02, .02, .001, .05, 0., .001])
        self.assertFalse(np.array_equal(measurement_noise(noise_cfg, 'slow', seeds[0], 5),
                                        measurement_noise(noise_cfg, 'slow', seeds[6], 5)))

    def test_histories_start_empty_and_final_budget_includes_scout_followers(self):
        cfg, domain, clients, oracle = fixture()
        initial = fresh_state(domain, cfg, clients)
        self.assertTrue(all(not h for m in initial['methods'].values() for h in m['histories']))
        data = run_fleet(domain, cfg, clients, 1100, 0, oracle)
        validate_state(data, domain, cfg, clients, 1100, 0, complete=True)
        for method, value in data['methods'].items():
            self.assertEqual(len(value['events']), 18)
            self.assertEqual(value['snapshots']['18']['queries_per_client'], [6, 6, 6])
            self.assertEqual(value['snapshots']['18']['round'], 8 if method == 'single_scout' else 6)
        self.assertEqual(len(data['methods']['single_scout']['histories'][0]), 6)

    def test_first_round_contains_no_shared_current_round_results(self):
        cfg, domain, clients, oracle = fixture()
        batches = []
        def checked(plans, cache):
            self.assertEqual(len({p['round'] for p in plans}), 1)
            self.assertTrue(all(p['information_round'] == p['round']-1 for p in plans))
            batches.append(deepcopy(plans))
            return oracle(plans, cache)
        run_fleet(domain, cfg, clients, 1100, 0, checked)
        self.assertEqual(set(p['method'] for p in batches[0]), set(METHODS))
        self.assertEqual(sum(p['method'] == 'single_scout' for p in batches[0]), 1)

    def test_round_checkpoint_resume_is_identical_to_uninterrupted_execution(self):
        cfg, domain, clients, oracle = fixture()
        full = run_fleet(domain, cfg, clients, 1100, 0, oracle)
        partial = []
        def stop(data):
            if data['round'] == 3:
                partial.append(json.loads(json.dumps(data)))
                raise InterruptedError('test interruption')
        with self.assertRaises(InterruptedError):
            run_fleet(domain, cfg, clients, 1100, 0, oracle, stop)
        resumed = run_fleet(domain, cfg, clients, 1100, 0, oracle, state=partial[0])
        self.assertEqual(full, resumed)

    def test_peer_sharing_exposes_only_incumbents_and_still_requires_target_query(self):
        cfg, domain, _, _ = fixture()
        histories = [[], [dict(action=0, eligible=True, J=2.), dict(action=1, eligible=True, J=1.)], []]
        action = peer_candidate(histories, domain, 0, 'slow')
        self.assertEqual(action, 1)
        histories[0].append(dict(action=1, eligible=True, J=5.))
        self.assertIsNone(peer_candidate(histories, domain, 0, 'slow'))
        histories[1].append(dict(action=2, eligible=False, J=None))
        self.assertIsNone(peer_candidate(histories, domain, 0, 'slow'))

    def test_future_labels_forged_recommendations_and_incomplete_runs_are_rejected(self):
        cfg, domain, clients, oracle = fixture()
        data = run_fleet(domain, cfg, clients, 1100, 0, oracle)
        for field, value in [('information_round', 100), ('noise_seed', 123), ('recommendations', ['g4', 'g4'])]:
            changed = deepcopy(data); changed['methods']['federated_cost_bo']['events'][0][field] = value
            with self.assertRaises(RuntimeError):
                validate_state(changed, domain, cfg, clients, 1100, 0, complete=True)
        changed = deepcopy(data); changed['methods']['independent_bo']['events'].pop()
        with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
            validate_state(changed, domain, cfg, clients, 1100, 0, complete=True)


class OracleAndEvaluationTests(unittest.TestCase):
    def test_only_requested_counterfactual_gains_are_simulated_and_cache_is_paired(self):
        cfg, domain, clients, _ = fixture()
        state = dict(pool=[dict(id=f'g{i}', gains=[i]*3) for i in range(5)], refs=[dict(id='slow')], physics={})
        plans = [dict(client=i, local_evaluation=1, context='slow', candidate=c,
                      noise_seed=100+i) for i, c in [(0, 'g0'), (0, 'g0'), (0, 'g1'), (1, 'g0')]]
        calls = []
        def fake(params, ref, pool, physics, seed):
            calls.append((seed, [p['id'] for p in pool]))
            return [dict(candidate=p['id'], J=float(seed), complete=True, feasible=True,
                         integration_resolved=True, substeps=2) for p in pool], {}
        cache = {}
        with patch.object(runner, 'resolved_rollout', side_effect=fake):
            observed = runner.observe(plans, cache, state, clients)
            self.assertEqual(calls, [(100, ['g0', 'g1']), (101, ['g0'])])
            self.assertEqual(observed[0], observed[1]); self.assertNotEqual(observed[0], observed[3])
            runner.observe(plans, cache, state, clients)
            self.assertEqual(len(calls), 2)

    def test_validation_rescores_frozen_recommendations_without_changing_learning(self):
        cfg, domain, clients, oracle = fixture()
        data = run_fleet(domain, cfg, clients, 1100, 0, oracle)
        reference = {('target', c): dict(J=1., eligible=True) for c in domain.contexts}
        val = {('target', c, k): dict(J=1., eligible=True) for c, k in domain.actions}
        job = dict(condition='homogeneous', seed=1100, clients=clients, state=data)
        before = deepcopy(data)
        rows, _, _, _ = evaluate([job], domain, cfg, (reference, val))
        self.assertTrue(rows[rows.fleet_budget == max(cfg['fleet_budgets'])].all_contexts_success.all())
        # At an early matched fleet budget, waiting scout followers may still
        # lack recommendations. They must remain in the success denominator.
        self.assertFalse(rows[(rows.method == 'single_scout') &
                              (rows.fleet_budget == min(cfg['fleet_budgets']))].all_contexts_success.all())
        damaged = {key: dict(J=3., eligible=False) for key in val}
        other, _, _, _ = evaluate([job], domain, cfg, (reference, damaged))
        self.assertFalse(other.all_contexts_success.any())
        self.assertEqual(data, before)

    def test_smoke_and_incomplete_collect_do_not_stage_results(self):
        with self.assertRaisesRegex(RuntimeError, 'Smoke'):
            runner.collect(dict(specification={'mode': 'smoke'}))
        with patch.object(runner, 'analyze', side_effect=RuntimeError('Missing checkpoint')):
            with patch.object(runner.subprocess, 'run') as proc:
                with self.assertRaisesRegex(RuntimeError, 'Missing'):
                    runner.collect(dict(specification={'mode': 'full'}))
                proc.assert_not_called()


if __name__ == '__main__':
    unittest.main()
