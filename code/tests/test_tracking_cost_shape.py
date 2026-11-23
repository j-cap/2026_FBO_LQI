"""Prior projection, matched controls, and experiment integrity checks."""
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
from src.tracking_budget import evaluate_job, prepare_source
from src.tracking_information import run_suite as previous_suite
from src.tracking_cost_shape import (ARMS, METHODS, PRIOR_DEFINITION, control_comparison,
                                     paired_effects, run_suite, source_for_arm)
import tracking_cost_shape as runner


def fixture():
    cfg, domain, cal, val = toy()
    cfg.update(methods=METHODS.copy(), prior_definition=PRIOR_DEFINITION)
    source = prepare_source(cal, domain, 'target', 'leave_vehicle_out', cfg['gp'])
    return cfg, domain, cal, val, source


class PriorProjectionTests(unittest.TestCase):
    def test_level_is_constant_within_context_and_preserves_uniform_log_mean(self):
        _, domain, _, _, source = fixture()
        # Deliberately different means and variations in each context.
        source['prior_mean'] = [float(i*i) for i in range(len(domain.actions))]
        before = deepcopy(source)
        level = source_for_arm(source, domain, 'context_level')
        for context in domain.contexts:
            indices = [i for i, (c, _) in enumerate(domain.actions) if c == context]
            expected = np.mean(np.asarray(source['prior_mean'])[indices])
            np.testing.assert_array_equal(np.asarray(level['prior_mean'])[indices], expected)
        self.assertEqual(source, before)
        level['catalogue']['slow'] = 'changed'
        self.assertEqual(source, before)

    def test_projection_uses_action_context_labels_rather_than_reshape_order(self):
        _, domain, _, _, source = fixture()
        source['prior_mean'] = list(range(len(domain.actions)))
        order = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
        shuffled = deepcopy(domain)
        shuffled.actions = [domain.actions[i] for i in order]
        shuffled_source = deepcopy(source)
        shuffled_source['prior_mean'] = [source['prior_mean'][i] for i in order]
        a = source_for_arm(source, domain, 'context_level')['prior_mean']
        b = source_for_arm(shuffled_source, shuffled, 'context_level')['prior_mean']
        np.testing.assert_array_equal(b, np.asarray(a)[order])

    def test_disabled_information_is_never_read(self):
        _, domain, _, _, source = fixture()
        source['prior_feasibility'] = None
        for arm in ARMS:
            np.testing.assert_array_equal(source_for_arm(source, domain, arm)['prior_feasibility'], .5)
        source['prior_mean'] = None
        np.testing.assert_array_equal(source_for_arm(source, domain, 'no_extra_prior')['prior_mean'], 0.)
        for bad in ([np.nan]*len(domain.actions), [0.], None):
            source['prior_mean'] = bad
            for arm in ['context_level', 'full_cost_prior']:
                with self.assertRaises(ValueError):
                    source_for_arm(source, domain, arm)

    def test_controls_match_previous_no_extra_and_cost_only_arms(self):
        cfg, domain, cal, _, source = fixture()
        current = run_suite(cal, domain, 'target', source, cfg, 1000, 101)
        oldcfg = deepcopy(cfg)
        oldcfg['methods'] = ['catalogue', 'no_extra_prior', 'cost_only', 'feasibility_only', 'both_priors']
        old = previous_suite(cal, domain, 'target', source, oldcfg, 1000, 101)
        for new, previous in [('catalogue', 'catalogue'), ('no_extra_prior', 'no_extra_prior'),
                               ('full_cost_prior', 'cost_only')]:
            for key in ['steps', 'actual_rollouts', 'final_gp_parameters']:
                self.assertEqual(current[new][key], old[previous][key])
        for arm in ARMS:
            self.assertEqual(current[arm]['steps'][:2], current['catalogue']['steps'])
            self.assertEqual(current[arm]['actual_rollouts'], max(cfg['budgets']))
            actions = [s['action'] for s in current[arm]['steps']]
            self.assertEqual(len(actions), len(set(actions)))
        audit = control_comparison(dict(target='target', protocol='leave_vehicle_out', seed=1000,
                                         methods=current), dict(methods=old))
        self.assertTrue(all(r['action_trace_match'] and r['recommendation_trace_match'] for r in audit))

    def test_constant_full_prior_produces_identical_level_and_full_traces(self):
        cfg, domain, cal, _, source = fixture()
        # Binary-exact constants avoid a floating-point averaging discrepancy.
        source['prior_mean'] = [.5 if c == 'slow' else 1. for c, _ in domain.actions]
        result = run_suite(cal, domain, 'target', source, cfg, 1000, 101)
        for key in ['steps', 'actual_rollouts', 'final_gp_parameters']:
            self.assertEqual(result['context_level'][key], result['full_cost_prior'][key])

    def test_validation_cannot_enter_acquisition(self):
        cfg, domain, cal, val, source = fixture()
        with patch('src.tracking_budget.dense_references', side_effect=AssertionError('Validation leak')):
            result = run_suite(cal, domain, 'target', source, cfg, 1000, 101)
        self.assertEqual(set(result), set(METHODS))
        with self.assertRaisesRegex(ValueError, 'validation'):
            run_suite(pd.concat([cal, val]), domain, 'target', source, cfg, 1000, 101)


class PairedAnalysisTests(unittest.TestCase):
    def rows(self):
        return pd.DataFrame([dict(protocol='leave_vehicle_out', target=target, seed=0, budget=4,
            method=arm, success_fraction=success, joint_success=False, fully_feasible=True,
            failed_queries=failed, J=cost) for target in ['a', 'b'] for arm, success, failed, cost in [
                ('no_extra_prior', .2, 5, 2.), ('context_level', .5, 2, 1.8),
                ('full_cost_prior', .8, 1, 1.5)]])

    def test_effect_signs_and_per_vehicle_denominators(self):
        _, summary, vehicles = paired_effects(self.rows())
        def value(metric, contrast):
            return summary[(summary.metric == metric) & (summary.contrast == contrast)].iloc[0].mean_difference
        self.assertAlmostEqual(value('success_fraction', 'level_minus_none'), .3)
        self.assertAlmostEqual(value('success_fraction', 'full_minus_level'), .3)
        self.assertAlmostEqual(value('J', 'full_minus_level'), -.3)
        self.assertAlmostEqual(value('failed_queries', 'full_minus_none'), -4.)
        self.assertTrue((summary.matched_trials == 2).all())
        self.assertTrue((vehicles.matched_trials == 1).all())

    def test_each_cost_pair_keeps_its_own_feasible_denominator(self):
        frame = self.rows()
        mask = (frame.target == 'b') & (frame.method == 'full_cost_prior')
        frame.loc[mask, 'fully_feasible'] = False; frame.loc[mask, 'J'] = None
        _, effects, _ = paired_effects(frame)
        for contrast, count in [('level_minus_none', 2), ('full_minus_level', 1), ('full_minus_none', 1)]:
            row = effects[(effects.metric == 'J') & (effects.contrast == contrast)].iloc[0]
            self.assertEqual(row.matched_trials, 2); self.assertEqual(row.eligible_trials, count)
        self.assertTrue((effects[effects.metric == 'success_fraction'].eligible_trials == 2).all())

    def test_unmatched_or_duplicate_trials_are_rejected(self):
        frame = self.rows()
        with self.assertRaisesRegex(ValueError, 'Unmatched'):
            paired_effects(frame.iloc[1:])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            paired_effects(pd.concat([frame, frame.iloc[:1]]))

    def test_initial_budget_has_zero_treatment_effects(self):
        cfg, domain, cal, val, source = fixture()
        methods = run_suite(cal, domain, 'target', source, cfg, 1000, 101)
        rows, _ = evaluate_job(dict(target='target', protocol='leave_vehicle_out', seed=1000,
                                   methods=methods), cal, val, domain, cfg)
        effects, _, _ = paired_effects(pd.DataFrame(rows))
        np.testing.assert_allclose(effects[effects.budget == 2].difference, 0., atol=1e-12)


class IntegrityTests(unittest.TestCase):
    def make_job(self):
        cfg, domain, cal, _, source = fixture()
        expected = dict(target='target', protocol='leave_vehicle_out', seed=1000,
                        noise_seed=101, fold='leave_vehicle_out__target')
        result = dict(**expected, signature='sig', source_ids=source['source_ids'],
                      methods=run_suite(cal, domain, 'target', source, cfg, 1000, 101))
        return dict(signature='sig', domain=domain, calibration=cal, cfg=cfg), expected, result

    def test_arm_labels_observations_and_incumbents_are_validated(self):
        state, expected, result = self.make_job()
        runner.validate_job(state, expected, result)
        changed = deepcopy(result); changed['methods']['context_level']['prior_mode'] = 'full'
        with self.assertRaisesRegex(RuntimeError, 'prior labels'):
            runner.validate_job(state, expected, changed)
        changed = deepcopy(result)
        changed['methods']['full_cost_prior']['steps'][0]['recommendations'] = ['g4', 'g4']
        with self.assertRaisesRegex(RuntimeError, 'incumbent'):
            runner.validate_job(state, expected, changed)

    def test_changed_gp_or_prior_definition_is_rejected(self):
        for field in ['gp', 'prior_definition']:
            cfg = runner.base.read(runner.CONFIG)
            if field == 'gp':
                cfg['gp']['noise_std_log_J'] *= 2
            else:
                cfg[field] = 'target_derived_mean'
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'changed.json'; path.write_text(json.dumps(cfg))
                with self.assertRaises(ValueError):
                    runner.setup(path)

    def test_incomplete_or_smoke_results_are_never_staged(self):
        state, _, _ = self.make_job()
        state.update(specification={'mode': 'smoke'})
        with self.assertRaisesRegex(RuntimeError, 'Smoke'):
            runner.collect(state)
        state['specification']['mode'] = 'full'
        with patch.object(runner, 'analyze', side_effect=RuntimeError('Missing checkpoint')):
            with patch.object(runner.base.subprocess, 'run') as proc:
                with self.assertRaisesRegex(RuntimeError, 'Missing checkpoint'):
                    runner.collect(state)
                proc.assert_not_called()


if __name__ == '__main__':
    unittest.main()
