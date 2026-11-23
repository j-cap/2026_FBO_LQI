"""Immediate adaptation, cost-aware cold starts and censored early evaluation."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
from test_tracking_online import fixture as original_fixture
from src import tracking_online_lcb as online
from src.tracking_online import program as frozen_program
from src.tracking_online_lcb_report import eligibility_times, evaluate, summarize, summarize_times
import tracking_online_lcb as runner


def fixture():
    cfg, domain, clients, oracle = original_fixture()
    cfg.update(initialization_tests=1, fleet_budgets=[3, 6, 9, 12, 18],
        acquisition=dict(confidence_multiplier=2., feasibility_penalty=1., probability_floor=1e-6))
    return cfg, domain, clients, oracle


class AcquisitionTests(unittest.TestCase):
    def test_cost_mean_changes_query_without_any_eligible_incumbent(self):
        cfg, domain, _, _ = fixture()
        params = np.log([*cfg['gp']['length_scales'], cfg['gp']['amplitude']])
        # The other context has even smaller predicted cost but is unavailable.
        prior = np.full(len(domain.actions), -100.)
        prior[:5] = [10., -2., 10., 10., 10.]
        for history in ([], [dict(action=0, eligible=False, J=None)]):
            def select(mean):
                return online.choose(history, domain, 'slow', mean, cfg['gp'], cfg['acquisition'],
                    params, 0, np.random.default_rng(1))
            a = select(prior)
            self.assertEqual(a[0], 1)
            self.assertEqual(a[3], 'feasibility_penalized_LCB')
            changed = prior.copy(); changed[1], changed[2] = 10., -2.
            self.assertEqual(select(changed)[0], 2)

    def test_exploration_feasibility_and_zero_probability_floor(self):
        cfg, _, _, _ = fixture()
        values = online.acquisition_values(np.array([0., 0., 0., 0.]),
            np.array([0., 1., 1., 1.]), np.array([.5, .5, .1, 0.]), cfg['acquisition'])
        self.assertLess(values[1], values[0])  # exploration bonus
        self.assertGreater(values[2], values[1])  # lower eligibility is penalized
        self.assertGreater(values[3], values[2])
        self.assertTrue(np.isfinite(values).all())
        self.assertAlmostEqual(values[3], -2.-np.log(1e-6))

    def test_exhausted_scheduled_context_does_not_query_another_context(self):
        cfg, domain, _, _ = fixture()
        history = [dict(action=i, eligible=False, J=None) for i in range(5)]
        params = np.log([*cfg['gp']['length_scales'], cfg['gp']['amplitude']])
        with self.assertRaisesRegex(RuntimeError, 'No unqueried'):
            online.choose(history, domain, 'slow', np.zeros(len(domain.actions)), cfg['gp'],
                cfg['acquisition'], params, 0, np.random.default_rng(1))


class ImmediateProtocolTests(unittest.TestCase):
    def test_round_two_uses_shared_model_and_only_prior_round_observations(self):
        cfg, domain, clients, oracle = fixture()
        phi = online.features(domain, cfg['model'])
        state = online.fresh_state(domain, cfg, clients)
        first = online.plan_round(state, domain, cfg, clients, 1100, 0, phi)
        self.assertTrue(all(p['mode'] == 'factory_initialization' for p in first))
        online.commit_round(state, first, oracle(first, {}), domain, cfg, clients, 0, phi)
        shared = phi@np.asarray(state['methods']['federated_cost_bo']['weights'])
        self.assertGreater(np.linalg.norm(shared), .01)
        with patch.object(online, 'choose', wraps=online.choose) as choose:
            second = online.plan_round(state, domain, cfg, clients, 1100, 0, phi)
        adaptive = [p for p in second if p['mode'] == 'feasibility_penalized_LCB']
        self.assertEqual(len(adaptive), len(choose.call_args_list))
        for plan, call in zip(adaptive, choose.call_args_list):
            self.assertEqual(len(call.args[0]), 1)
            self.assertEqual(plan['information_round'], 1)
            if plan['method'] == 'federated_cost_bo':
                np.testing.assert_array_equal(call.args[3], shared)
        for plan in second:
            self.assertEqual(plan['local_evaluation'], 2)
            self.assertNotEqual(plan['mode'], 'factory_initialization')

    def test_failed_factory_tests_are_charged_and_do_not_delay_adaptation(self):
        cfg, domain, clients, oracle = fixture()
        def failed(plans, cache):
            return [dict(eligible=False, J=None, complete=True, feasible=False,
                         integration_resolved=True, substeps=2) for _ in plans]
        state = online.run_fleet(domain, cfg, clients, 1100, 0, failed)
        online.validate_state(state, domain, cfg, clients, 1100, 0, complete=True)
        for method, data in state['methods'].items():
            self.assertEqual(len(data['events']), 18)
            self.assertTrue(all(c is None for row in data['snapshots']['18']['recommendations'] for c in row))
            self.assertTrue(all(not o['eligible'] for h in data['histories'] for o in h))
            for event in data['events']:
                if event['local_evaluation'] == 2 and (method != 'single_scout' or event['client'] == 0):
                    self.assertEqual(event['mode'], 'feasibility_penalized_LCB')

    def test_one_common_factory_test_early_budgets_and_scout_waiting(self):
        cfg, domain, clients, oracle = fixture()
        state = online.run_fleet(domain, cfg, clients, 1100, 0, oracle)
        online.validate_state(state, domain, cfg, clients, 1100, 0, complete=True)
        initial = None
        for method, data in state['methods'].items():
            self.assertEqual(len(data['events']), 18)
            self.assertEqual(data['snapshots']['18']['queries_per_client'], [6]*3)
            self.assertEqual(data['snapshots']['18']['round'], 8 if method == 'single_scout' else 6)
            if method != 'single_scout':
                self.assertEqual(data['snapshots']['3']['queries_per_client'], [1]*3)
                events = [e for e in data['events'] if e['mode'] == 'factory_initialization']
                self.assertEqual(len(events), 3)
                paired = [(e['action'], e['noise_seed'], e['J']) for e in events]
                initial = paired if initial is None else initial
                self.assertEqual(paired, initial)
        scout = state['methods']['single_scout']
        self.assertEqual(sum(e['mode'] == 'factory_initialization' for e in scout['events']), 1)
        self.assertEqual(sum(e['mode'] == 'scout_gain_verification' for e in scout['events']), 4)

    def test_peer_verification_can_begin_in_round_two(self):
        cfg, domain, clients, oracle = fixture()
        # This programme supplies peer observations in the next scheduled context.
        state = online.run_fleet(domain, cfg, clients, 1101, 0, oracle)
        second = [e for e in state['methods']['online_gain_sharing']['events'] if e['round'] == 2]
        self.assertTrue(any(e['mode'] == 'peer_gain_verification' for e in second))
        online.validate_state(state, domain, cfg, clients, 1101, 0, complete=True)

    def test_context_and_noise_programmes_match_frozen_pilot(self):
        cfg, domain, clients, _ = fixture()
        for client in clients:
            self.assertEqual(online.program(domain, cfg, 1100, client['id']),
                             frozen_program(domain, cfg, 1100, client['id']))

    def test_json_checkpoint_resume_is_identical(self):
        cfg, domain, clients, oracle = fixture()
        full = online.run_fleet(domain, cfg, clients, 1100, 0, oracle)
        partial = []
        def stop(state):
            if state['round'] == 1:
                partial.append(json.loads(json.dumps(state)))
                raise InterruptedError('stop immediately before adaptive querying')
        with self.assertRaises(InterruptedError):
            online.run_fleet(domain, cfg, clients, 1100, 0, oracle, stop)
        online.validate_state(partial[0], domain, cfg, clients, 1100, 0)
        resumed = online.run_fleet(domain, cfg, clients, 1100, 0, oracle, state=partial[0])
        self.assertEqual(full, resumed)

    def test_forged_delayed_initialization_future_labels_and_early_snapshot_are_rejected(self):
        cfg, domain, clients, oracle = fixture()
        full = online.run_fleet(domain, cfg, clients, 1100, 0, oracle)
        for field, value in [('mode', 'factory_initialization'), ('information_round', 2), ('noise_seed', 123)]:
            changed = deepcopy(full)
            event = next(e for e in changed['methods']['federated_cost_bo']['events'] if e['local_evaluation'] == 2)
            event[field] = value
            with self.assertRaises(RuntimeError):
                online.validate_state(changed, domain, cfg, clients, 1100, 0, complete=True)
        changed = deepcopy(full)
        changed['methods']['federated_cost_bo']['snapshots']['3']['queries_per_client'][0] = 6
        with self.assertRaisesRegex(RuntimeError, 'snapshot'):
            online.validate_state(changed, domain, cfg, clients, 1100, 0, complete=True)


class EvaluationAndRunnerTests(unittest.TestCase):
    def test_missing_contexts_remain_in_early_denominators_and_validation_is_separate(self):
        cfg, domain, clients, oracle = fixture()
        state = online.run_fleet(domain, cfg, clients, 1100, 0, oracle)
        reference = {('target', c): dict(J=1., eligible=True) for c in domain.contexts}
        val = {('target', c, k): dict(J=1., eligible=True) for c, k in domain.actions}
        job = dict(condition='homogeneous', seed=1100, clients=clients, state=state)
        before = deepcopy(state)
        trials, cells, _, _ = evaluate([job], domain, cfg, (reference, val))
        early = trials[(trials.fleet_budget == 3) & (trials.method != 'single_scout')]
        self.assertTrue((early.eligible_context_coverage == .5).all())
        self.assertTrue((early.context_success == .5).all())
        self.assertFalse(early.fully_feasible.any())
        self.assertTrue(early.J.isna().all())
        self.assertEqual(len(cells[cells.fleet_budget == 3]), 6*3*2)
        damaged = {key: dict(J=3., eligible=False) for key in val}
        changed, _, _, _ = evaluate([job], domain, cfg, (reference, damaged))
        self.assertTrue((changed.eligible_context_coverage == trials.eligible_context_coverage).all())
        self.assertFalse(changed.validation_eligible_context_coverage.any())
        self.assertEqual(state, before)
        summary = summarize(trials)
        self.assertEqual(summary[(summary.method == 'federated_cost_bo') &
                                 (summary.fleet_budget == 3)].eligible_context_coverage.iloc[0], .5)

    def test_first_eligibility_times_retain_censoring_and_scout_delay(self):
        cfg, domain, clients, oracle = fixture()
        def partial(plans, cache):
            observations = oracle(plans, cache)
            for plan, observation in zip(plans, observations):
                if plan['client'] == 2 or plan['context'] == 'fast' or plan['local_evaluation'] == 1:
                    observation.update(eligible=False, J=None, feasible=False)
            return observations
        state = online.run_fleet(domain, cfg, clients, 1100, 0, partial)
        job = dict(condition='homogeneous', seed=1100, clients=clients, state=state)
        times, context_times = eligibility_times([job], domain, cfg)
        self.assertEqual(len(times), 6*3)
        self.assertEqual(len(context_times), 6*3*2)
        censored = times[times.client == 'client_02']
        self.assertFalse(censored.any_eligible_observed.any())
        self.assertTrue(censored.first_eligible_query.isna().all())
        self.assertFalse(times.all_contexts_observed.any())
        self.assertTrue((context_times[context_times.context == 'fast'].censor_query == 6).all())
        scout = times[(times.method == 'single_scout') & (times.client != 'client_00') & times.any_eligible_observed]
        self.assertTrue((scout.first_eligible_round == scout.first_eligible_query+2).all())
        timing = summarize_times(times)
        self.assertTrue((timing.first_eligible_censored == 1).all())
        self.assertTrue((timing.all_contexts_censored == 3).all())

    def test_smoke_and_incomplete_collection_never_stage_results(self):
        with self.assertRaisesRegex(RuntimeError, 'Smoke'):
            runner.collect(dict(specification={'mode': 'smoke'}))
        with patch.object(runner, 'analyze', side_effect=RuntimeError('Missing checkpoint')):
            with patch.object(runner.subprocess, 'run') as proc:
                with self.assertRaisesRegex(RuntimeError, 'Missing'):
                    runner.collect(dict(specification={'mode': 'full'}))
                proc.assert_not_called()


if __name__ == '__main__':
    unittest.main()
