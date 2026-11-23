"""Causal-channel isolation and matched factorial-analysis checks."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
from test_tracking_budget import toy
from src.tracking_budget import TargetOracle, evaluate_job, prepare_source, run_method
from src.tracking_information import (ARMS, METHODS, control_comparison,
                                      factorial_effects, masked_source, run_suite)
import tracking_information as runner


def fixture():
    cfg, domain, cal, val = toy()
    cfg['methods'] = METHODS.copy()
    source = prepare_source(cal, domain, 'target', 'leave_vehicle_out', cfg['gp'])
    return cfg, domain, cal, val, source


class ChannelIsolationTests(unittest.TestCase):
    def test_disabled_channels_are_not_read_and_source_is_not_modified(self):
        _, domain, _, _, source = fixture()
        before = deepcopy(source)
        for arm, (cost, feasibility) in ARMS.items():
            masked = masked_source(source, domain, arm)
            np.testing.assert_array_equal(masked['prior_mean'], source['prior_mean'] if cost else 0.)
            np.testing.assert_array_equal(masked['prior_feasibility'], source['prior_feasibility'] if feasibility else .5)
            masked['catalogue']['slow'] = 'changed'
        self.assertEqual(source, before)
        broken = deepcopy(source); broken['prior_mean'] = None
        np.testing.assert_array_equal(masked_source(broken, domain, 'feasibility_only')['prior_mean'], 0.)
        broken = deepcopy(source); broken['prior_feasibility'] = None
        np.testing.assert_array_equal(masked_source(broken, domain, 'cost_only')['prior_feasibility'], .5)
        broken['prior_mean'] = [np.nan]*len(domain.actions)
        with self.assertRaises(ValueError):
            masked_source(broken, domain, 'cost_only')

    def test_control_arms_exactly_match_unchanged_optimizer(self):
        cfg, domain, cal, _, source = fixture()
        current = run_suite(cal, domain, 'target', source, cfg, 1000, 101)
        for arm, previous in [('no_extra_prior', 'catalogue_warmstart_bo'), ('both_priors', 'pooled_bo')]:
            old = run_method(previous, TargetOracle(cal, 'target', 101, domain),
                             domain, source, cfg['gp'], 1000, max(cfg['budgets']))
            new = {key: value for key, value in current[arm].items() if key != 'channels'}
            self.assertEqual(new, old)
        for arm in ARMS:
            self.assertEqual(current[arm]['steps'][:len(domain.contexts)], current['catalogue']['steps'])
            self.assertEqual(current[arm]['actual_rollouts'], max(cfg['budgets']))

    def test_poisoning_disabled_feasibility_prior_cannot_change_cost_only_trace(self):
        cfg, domain, cal, _, source = fixture()
        other = deepcopy(source); other['prior_feasibility'] = [np.nan]*len(domain.actions)
        def execute(s):
            return run_method('pooled_bo', TargetOracle(cal, 'target', 101, domain), domain,
                              masked_source(s, domain, 'cost_only'), cfg['gp'], 1000, 4)
        self.assertEqual(execute(source), execute(other))

    def test_validation_is_not_used_by_learning(self):
        cfg, domain, cal, val, source = fixture()
        with patch('src.tracking_budget.dense_references', side_effect=AssertionError('Validation leak')):
            result = run_suite(cal, domain, 'target', source, cfg, 1000, 101)
        self.assertEqual(set(result), set(METHODS))
        with self.assertRaisesRegex(ValueError, 'validation'):
            run_suite(pd.concat([cal, val]), domain, 'target', source, cfg, 1000, 101)


class FactorialAnalysisTests(unittest.TestCase):
    def rows(self):
        rows = []
        for target in ['a', 'b']:
            for arm, quality, failed, cost in [('no_extra_prior', .2, 5, 2.),
                                              ('cost_only', .5, 4, 1.7),
                                              ('feasibility_only', .3, 2, 1.9),
                                              ('both_priors', .8, 1, 1.5)]:
                rows.append(dict(protocol='leave_vehicle_out', target=target, seed=0, budget=4,
                                 method=arm, success_fraction=quality, joint_success=False,
                                 fully_feasible=True, failed_queries=failed, J=cost))
        return pd.DataFrame(rows)

    def test_conditional_effects_main_effects_and_interaction(self):
        _, summary, _ = factorial_effects(self.rows())
        def value(metric, contrast):
            return summary[(summary.metric == metric) & (summary.contrast == contrast)].iloc[0].mean_difference
        self.assertAlmostEqual(value('success_fraction', 'cost_with_feasibility_off'), .3)
        self.assertAlmostEqual(value('success_fraction', 'cost_with_feasibility_on'), .5)
        self.assertAlmostEqual(value('success_fraction', 'feasibility_with_cost_off'), .1)
        self.assertAlmostEqual(value('success_fraction', 'cost_main_effect'), .4)
        self.assertAlmostEqual(value('success_fraction', 'feasibility_main_effect'), .2)
        self.assertAlmostEqual(value('success_fraction', 'interaction'), .2)
        self.assertAlmostEqual(value('failed_queries', 'feasibility_main_effect'), -3.)
        self.assertAlmostEqual(value('J', 'interaction'), -.1)

    def test_cost_contrasts_retain_their_own_feasible_denominators(self):
        frame = self.rows()
        bad = (frame.target == 'b') & (frame.method == 'both_priors')
        frame.loc[bad, 'fully_feasible'] = False
        frame.loc[bad, 'J'] = None
        _, effects, _ = factorial_effects(frame)
        row = effects[(effects.metric == 'J') & (effects.contrast == 'interaction')].iloc[0]
        self.assertEqual(row.matched_trials, 2); self.assertEqual(row.eligible_trials, 1)
        row = effects[(effects.metric == 'J') & (effects.contrast == 'cost_with_feasibility_off')].iloc[0]
        self.assertEqual(row.eligible_trials, 2)
        row = effects[(effects.metric == 'success_fraction') & (effects.contrast == 'interaction')].iloc[0]
        self.assertEqual(row.eligible_trials, 2)

    def test_unmatched_or_duplicated_arms_are_rejected(self):
        frame = self.rows()
        with self.assertRaisesRegex(ValueError, 'Unmatched'):
            factorial_effects(frame.iloc[1:])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            factorial_effects(pd.concat([frame, frame.iloc[:1]]))

    def test_initial_budget_effects_are_zero_before_any_adaptive_query(self):
        cfg, domain, cal, val, source = fixture()
        methods = run_suite(cal, domain, 'target', source, cfg, 1000, 101)
        job = dict(target='target', protocol='leave_vehicle_out', seed=1000, methods=methods)
        rows, _ = evaluate_job(job, cal, val, domain, cfg)
        trials, _, _ = factorial_effects(pd.DataFrame(rows))
        np.testing.assert_allclose(trials[trials.budget == 2].difference, 0., atol=1e-12)


class ProtocolAndCheckpointTests(unittest.TestCase):
    def make_job(self):
        cfg, domain, cal, val, source = fixture()
        expected = dict(target='target', protocol='leave_vehicle_out', seed=1000,
                        noise_seed=101, fold='leave_vehicle_out__target')
        methods = run_suite(cal, domain, 'target', source, cfg, 1000, 101)
        result = dict(**expected, signature='sig', source_ids=source['source_ids'], methods=methods)
        state = dict(signature='sig', domain=domain, calibration=cal, cfg=cfg)
        return state, expected, result

    def test_channel_labels_and_common_initialization_are_validated(self):
        state, expected, result = self.make_job()
        runner.validate_job(state, expected, result)
        changed = deepcopy(result)
        changed['methods']['cost_only']['channels']['feasibility_prior'] = True
        with self.assertRaisesRegex(RuntimeError, 'channel'):
            runner.validate_job(state, expected, changed)
        changed = deepcopy(result)
        changed['methods']['feasibility_only']['steps'][0]['recommendations'] = ['g4', 'g4']
        with self.assertRaisesRegex(RuntimeError, 'incumbent'):
            runner.validate_job(state, expected, changed)

    def test_control_audit_compares_corresponding_legacy_arms(self):
        _, _, result = self.make_job()
        previous = dict(methods={'catalogue': result['methods']['catalogue'],
            'catalogue_warmstart_bo': result['methods']['no_extra_prior'],
            'pooled_bo': result['methods']['both_priors']})
        audit = control_comparison(result, previous)
        self.assertTrue(all(r['action_trace_match'] and r['recommendation_trace_match'] for r in audit))

    def test_hyperparameter_changes_are_not_allowed_in_the_attribution_test(self):
        cfg = runner.base.read(runner.CONFIG)
        cfg['gp']['noise_std_log_J'] *= 2
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'changed.json'; path.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, 'mismatch in gp'):
                runner.setup(path)

    def test_incomplete_and_smoke_results_cannot_be_staged(self):
        state, _, _ = self.make_job()
        with tempfile.TemporaryDirectory() as tmp:
            state.update(out=Path(tmp), specification={'mode': 'full'}, targets=['target'],
                         archive={'configuration': {'calibration_seeds': [101, 102]}})
            state['cfg'].update(protocols=['leave_vehicle_out'], optimizer_seeds=[1000])
            runner.base.save(Path(tmp)/'run_manifest.json', {'signature': 'sig'})
            with patch.object(runner.base.subprocess, 'run') as proc:
                with self.assertRaisesRegex(RuntimeError, 'Missing information'):
                    runner.collect(state)
                proc.assert_not_called()
            state['specification']['mode'] = 'smoke'
            with self.assertRaisesRegex(RuntimeError, 'Smoke'):
                runner.collect(state)


if __name__ == '__main__':
    unittest.main()
