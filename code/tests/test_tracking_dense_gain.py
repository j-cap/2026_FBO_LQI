"""Exact GP equivalence, learner isolation, paid replay and checkpoint resume."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import tracking_dense_gain as runner
from src.tracking_dense_gain import ExactCostSnapshot, METHODS, D, E, run_fleet, validate_state
from src.tracking_dense_gain_check import representation_check


class DenseGainTests(unittest.TestCase):
    def test_public_pool_geometry_and_no_evaluator_access(self):
        with patch.object(runner.base, 'setup', side_effect=AssertionError('No old full-grid basis')):
            state = runner.setup()
        self.assertEqual(len(state['pool']), 1292)
        self.assertEqual(len(state['domain'].actions), 23256)
        self.assertEqual(state['pool'][:71], runner.base.read(ROOT / 'results/tracking_gain_domain_v1/run_manifest.json')['pool'][:71])
        self.assertTrue(any(c['gains'][0] > .05 for c in state['pool']))
        self.assertTrue(all(c['gains'][1] <= 1 and c['gains'][2] <= 1 for c in state['pool']))
        self.assertGreater(state['domain'].X[:, 0].max(), 1.)
        self.assertFalse(set(runner.payload(state)) & {'specification', 'reference_input_hashes', 'calibration', 'validation', 'reference_selection'})

    def test_exact_shared_local_posterior_against_direct_gp(self):
        state = runner.setup(smoke=True)
        # Use a compact subset for the regression test; full preflight probes
        # the complete pilot grid with 9 clients and up to 24 observations each.
        domain = deepcopy(state['domain'])
        domain.X = domain.X[:90]
        domain.actions = domain.actions[:90]
        domain.contexts = [domain.actions[0][0]]
        domain.candidates = [a[1] for a in domain.actions]
        result = representation_check(domain, state['cfg'], probe_count=32)
        self.assertTrue(result['all_checks_passed'])
        self.assertEqual(len(result['checks']), 8)

    def test_local_model_is_independent_of_other_clients_and_remainder_is_retained(self):
        state = runner.setup(smoke=True)
        histories = [[dict(action=0, eligible=True, J=1.)], [dict(action=1, eligible=True, J=2.)], []]
        changed = deepcopy(histories)
        changed[1][0]['J'] = 100.
        models = [ExactCostSnapshot(h, state['domain'], state['cfg']['gp'], state['cfg']['hierarchical'], False)
            for h in (histories, changed)]
        queries = [0, 7, len(state['domain'].actions) - 1]
        np.testing.assert_array_equal(models[0].predict(0, queries), models[1].predict(0, queries))
        shared = ExactCostSnapshot(histories, state['domain'], state['cfg']['gp'], state['cfg']['hierarchical'], True)
        _, variance = shared.predict(2, [len(state['domain'].actions) - 1])
        self.assertGreater(variance[0], .1)

    def test_round_resume_and_logged_fit_replay_without_simulation_or_optimization(self):
        state = runner.setup(smoke=True)
        cfg, domain, clients, seed = state['cfg'], state['domain'], state['clients'], state['jobs'][0]['seed']
        def observe(plans, cache):
            return [dict(eligible=True, J=float(1 + (p['action'] % 17) / 20), complete=True,
                feasible=True, integration_resolved=True, substeps=4) for p in plans]
        first = run_fleet(domain, cfg, clients, seed, observe, stop_round=4)
        before = deepcopy(first)
        resumed = run_fleet(domain, cfg, clients, seed, observe, state=first)
        self.assertEqual(first, before)
        self.assertEqual(len(resumed['methods'][D]['events']), len(clients) * cfg['queries_per_client'])
        with patch('src.tracking_budget.minimize', side_effect=AssertionError('No optimizer during replay')), \
                patch('src.tracking_six_speed.minimize', side_effect=AssertionError('No recovery optimizer')):
            validate_state(resumed, domain, cfg, clients, seed, complete=True, replay=True)
        altered = deepcopy(resumed)
        event = next(e for e in altered['methods'][D]['events'] if e['predicted_log_J'] is not None)
        event['predicted_log_J'] += .1
        with self.assertRaises(RuntimeError):
            validate_state(altered, domain, cfg, clients, seed, complete=True, replay=True)

    def test_fresh_validation_union_and_smoke_collection_guard(self):
        state = runner.setup(smoke=True)
        selection = {c: set(state['domain'].candidates[:3]) for c in state['domain'].contexts}
        tasks = runner.evaluation_tasks(state, selection)
        self.assertEqual(sum(len(t['candidates']) for t in tasks), 18 * 3 * 2)
        self.assertEqual(len({t['noise_seed'] for t in tasks}), 2)
        self.assertTrue(all(t['role'] == 'validation' for t in tasks))
        with self.assertRaisesRegex(RuntimeError, 'Smoke'):
            runner.collect(state, 1)


if __name__ == '__main__':
    unittest.main()
