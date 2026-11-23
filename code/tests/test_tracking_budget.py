"""Leakage, budget accounting, acquisition math, and deterministic tie checks."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
from src.tracking_budget import (Posterior, TargetOracle, dense_references,
    domain_from_manifest, evaluate_job, incumbents, initial_design,
    lognormal_ei, matern, prepare_source, run_suite, source_ids)
from src.tracking_budget_report import paired_comparisons, summarize
from src.tracking_context import nearest_reference, stable_minimum
import tracking_budget as runner


def toy():
    cfg = json.loads((ROOT/'config/runs/tracking_budget.json').read_text())
    cfg['budgets'] = [2, 4]
    manifest = dict(configuration={'gains': {'log10_multiplier_bounds': [-1., 1.]}},
                    references=[dict(id='slow', speed=20., frequency=.1), dict(id='fast', speed=30., frequency=.6)],
                    candidates=[dict(id=f'g{i}', multipliers=[s]*3) for i, s in enumerate([1., .2, .5, 2., 8.])],
                    vehicles=[dict(id=v, family=f) for v, f in [('target', 'A'), ('sibling', 'A'), ('b', 'B'), ('c', 'C')]])
    domain = domain_from_manifest(manifest)
    rows = []
    for v in domain.families:
        for role, seeds in [('calibration', [101, 102]), ('validation', [201, 202])]:
            for seed in seeds:
                for context in domain.contexts:
                    for i, candidate in enumerate(domain.candidates):
                        cost = 1.+.2*(i-2)**2+(.05 if context == 'fast' else 0.)
                        rows.append(dict(vehicle=v, context=context, role=role, seed=seed,
                                         candidate=candidate, complete=True, feasible=True,
                                         integration_resolved=True, J=cost))
    data = pd.DataFrame(rows)
    return cfg, domain, data[data.role == 'calibration'].copy(), data[data.role == 'validation'].copy()


class SourceAndObservationTests(unittest.TestCase):
    def test_target_and_whole_family_are_excluded_from_source_statistics(self):
        cfg, domain, cal, _ = toy()
        self.assertEqual(source_ids(domain, 'target', 'leave_vehicle_out'), ['b', 'c', 'sibling'])
        self.assertEqual(source_ids(domain, 'target', 'leave_family_out'), ['b', 'c'])
        first = prepare_source(cal, domain, 'target', 'leave_family_out', cfg['gp'])
        damaged = cal.copy()
        mask = damaged.vehicle.isin(['target', 'sibling'])
        damaged.loc[mask, 'J'] *= 10000
        damaged.loc[mask, 'feasible'] = False
        self.assertEqual(first, prepare_source(damaged, domain, 'target', 'leave_family_out', cfg['gp']))

    def test_validation_rows_cannot_enter_sources_or_target_oracle(self):
        cfg, domain, cal, val = toy()
        both = pd.concat([cal, val])
        with self.assertRaisesRegex(ValueError, 'calibration'):
            prepare_source(both, domain, 'target', 'leave_vehicle_out', cfg['gp'])
        with self.assertRaisesRegex(ValueError, 'validation'):
            TargetOracle(both, 'target', 101, domain)

    def test_infeasible_observation_has_no_cost_and_duplicate_queries_fail(self):
        _, domain, cal, _ = toy()
        mask = (cal.vehicle == 'target') & (cal.context == 'slow') & (cal.candidate == 'g0')
        cal.loc[mask, 'feasible'] = False
        oracle = TargetOracle(cal, 'target', 101, domain)
        obs = oracle.observe(0)
        self.assertFalse(obs['eligible']); self.assertIsNone(obs['J'])
        self.assertEqual(incumbents([obs], domain), [None, None])
        with self.assertRaisesRegex(ValueError, 'twice'):
            oracle.observe(0)


class AcquisitionTests(unittest.TestCase):
    def test_lognormal_ei_agrees_with_direct_gaussian_quadrature(self):
        for mu, variance, best in [(0., .25, 1.), (-.3, .7, 2.), (.8, .1, 1.5)]:
            sigma = np.sqrt(variance)
            top = (np.log(best)-mu)/sigma
            expected = quad(lambda z: (best-np.exp(mu+sigma*z))*np.exp(-z*z/2)/np.sqrt(2*np.pi),
                            -14, top, epsabs=1e-11)[0]
            actual = lognormal_ei(np.array([mu]), np.array([variance]), np.array([best]))[0]
            self.assertAlmostEqual(actual, expected, places=9)

    def test_kernel_positive_definite_and_posterior_uses_revealed_points(self):
        cfg, domain, _, _ = toy()
        K = matern(domain.X, domain.X, cfg['gp']['length_scales'])
        self.assertGreater(np.linalg.eigvalsh(K).min(), -1e-12)
        observations = [dict(action=i, eligible=True, J=2.) for i in [0, 2, 5, 7]]
        params = np.log([*cfg['gp']['length_scales'], cfg['gp']['amplitude']])
        gp = Posterior(domain, observations, np.zeros(len(domain.actions)), cfg['gp'], params, True)
        mean, var = gp.predict()
        self.assertTrue(np.isfinite(mean).all())
        self.assertTrue((var > 0).all())
        np.testing.assert_allclose(mean[[0, 2, 5, 7]], np.log(2.), atol=.01)
        self.assertTrue((var[[0, 2, 5, 7]] < .01).all())

    def test_budget_accounting_shared_initialization_and_validation_blindness(self):
        cfg, domain, cal, _ = toy()
        source = prepare_source(cal, domain, 'target', 'leave_vehicle_out', cfg['gp'])
        with patch('src.tracking_budget.dense_references', side_effect=AssertionError('Validation leak')):
            suite = run_suite(cal, domain, 'target', source, cfg, 1000, 101)
        for method, result in suite.items():
            count = 2 if method == 'catalogue' else 4
            self.assertEqual(result['actual_rollouts'], count)
            self.assertEqual(len({s['action'] for s in result['steps']}), count)
            self.assertEqual({s['context'] for s in result['steps'][:2]}, set(domain.contexts))
        for method in ['catalogue_warmstart_bo', 'pooled_bo']:
            self.assertEqual(suite['catalogue']['steps'], suite[method]['steps'][:2])
        self.assertEqual([s['action'] for s in suite['random']['steps'][:2]],
                         [s['action'] for s in suite['independent_bo']['steps'][:2]])
        changed = deepcopy(source)
        changed['catalogue'] = dict.fromkeys(domain.contexts, 'g4')
        changed['prior_mean'] = [50.]*len(domain.actions)
        other = run_suite(cal, domain, 'target', changed, cfg, 1000, 101)
        self.assertEqual(suite['independent_bo'], other['independent_bo'])


class EvaluationAndCheckpointTests(unittest.TestCase):
    def make_job(self):
        cfg, domain, cal, val = toy()
        source = prepare_source(cal, domain, 'target', 'leave_vehicle_out', cfg['gp'])
        suite = run_suite(cal, domain, 'target', source, cfg, 1000, 101)
        expected = dict(target='target', protocol='leave_vehicle_out', seed=1000,
                        noise_seed=101, fold='leave_vehicle_out__target')
        job = dict(**expected, methods=suite, signature='sig', source_ids=source['source_ids'])
        state = dict(signature='sig', domain=domain, calibration=cal, cfg=cfg)
        return cfg, domain, cal, val, expected, job, state

    def test_validation_changes_scores_but_not_frozen_decisions_or_cost_accounting(self):
        cfg, domain, cal, val, _, job, _ = self.make_job()
        before = deepcopy(job)
        rows, cells = evaluate_job(job, cal, val, domain, cfg)
        self.assertEqual(job, before)
        cat = [r for r in rows if r['method'] == 'catalogue']
        self.assertEqual([r['actual_rollouts'] for r in cat], [2, 2])
        broken = val.copy(); broken.loc[broken.vehicle == 'target', 'feasible'] = False
        later, _ = evaluate_job(job, cal, broken, domain, cfg)
        self.assertTrue(all(r['success_fraction'] == 0 and r['J'] is None for r in later))
        self.assertEqual(job, before)
        self.assertEqual(len(cells), len(rows)*len(domain.contexts))

    def test_later_success_is_censored_at_an_earlier_budget(self):
        cfg, domain, cal, val = toy()
        oracle = TargetOracle(cal, 'target', 101, domain)
        observed, steps = [], []
        for i, action in enumerate([domain.action('slow', 'g0'), domain.action('fast', 'g0'),
                                    domain.action('slow', 'g2'), domain.action('fast', 'g2')]):
            observed.append(oracle.observe(action))
            steps.append(dict(evaluation=i+1, **observed[-1], recommendations=incumbents(observed, domain)))
        job = dict(target='target', protocol='leave_vehicle_out', seed=0,
                   methods={'test': {'steps': steps}})
        rows, _ = evaluate_job(job, cal, val, domain, cfg)
        self.assertIsNone(rows[0]['first_success'])
        self.assertEqual(rows[0]['restricted_evals_to_success'], 2)
        self.assertEqual(rows[1]['first_success'], 4)
        self.assertEqual(rows[1]['success_fraction'], 1.)

    def test_recommendation_forgery_and_incomplete_checkpoints_are_rejected(self):
        _, _, _, _, expected, job, state = self.make_job()
        runner.validate_job(state, expected, job)
        altered = deepcopy(job)
        altered['methods']['independent_bo']['steps'][0]['recommendations'] = ['g4', 'g4']
        with self.assertRaisesRegex(RuntimeError, 'incumbent'):
            runner.validate_job(state, expected, altered)
        altered = deepcopy(job); altered['source_ids'].append('target')
        with self.assertRaisesRegex(RuntimeError, 'leaked'):
            runner.validate_job(state, expected, altered)
        altered = deepcopy(job); altered['methods']['pooled_bo']['steps'].pop()
        with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
            runner.validate_job(state, expected, altered)

    def test_censoring_and_conditional_costs_do_not_drop_failed_trials(self):
        rows = []
        for target in ['a', 'b']:
            for method in ['pooled_bo', 'catalogue']:
                feasible = not (target == 'b' and method == 'pooled_bo')
                rows.append(dict(protocol='leave_vehicle_out', method=method, budget=4, target=target,
                                 seed=0, fully_feasible=feasible, J=1. if feasible else None,
                                 success_fraction=1. if feasible else 0., joint_success=feasible,
                                 failed_queries=0 if feasible else 1, unresolved_queries=0,
                                 actual_rollouts=4, restricted_evals_to_success=2 if feasible else 4))
        frame = pd.DataFrame(rows); summary = summarize(frame)
        pooled = summary[summary.method == 'pooled_bo'].iloc[0]
        self.assertTrue(pd.isna(pooled.mean_J_all_trials))
        self.assertEqual(pooled.context_success_rate, .5)
        self.assertEqual(pooled.restricted_mean_evals_to_joint_success, 3.)
        pair = paired_comparisons(frame).query("method == 'pooled_bo' and baseline == 'catalogue'").iloc[0]
        self.assertEqual(pair.paired_trials, 2); self.assertEqual(pair.jointly_feasible_pairs, 1)
        self.assertEqual(pair.context_success_delta, -.5)

    def test_missing_or_smoke_collection_does_not_stage_anything(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = dict(out=Path(tmp), signature='sig', specification={'mode': 'full'},
                         cfg={'protocols': ['leave_vehicle_out'], 'optimizer_seeds': [1000]},
                         targets=['target'], domain=toy()[1],
                         archive={'configuration': {'calibration_seeds': [101, 102]}})
            runner.save(Path(tmp)/'run_manifest.json', {'signature': 'sig'})
            with patch.object(runner.subprocess, 'run') as proc:
                with self.assertRaisesRegex(RuntimeError, 'Missing replay'):
                    runner.collect(state)
                proc.assert_not_called()
            state['specification']['mode'] = 'smoke'
            with self.assertRaisesRegex(RuntimeError, 'Smoke'):
                runner.collect(state)
            runner.save(Path(tmp)/'run_manifest.json', {'signature': 'changed'})
            with self.assertRaisesRegex(RuntimeError, 'Different BO'):
                runner.initialize(state)


class StableReferenceTests(unittest.TestCase):
    def test_roundoff_and_input_order_do_not_change_twenty_five_m_s_assignment(self):
        controls = [dict(id='s30', speed=30., spectral_centroid_hz=.3+1e-16),
                    dict(id='s20', speed=20., spectral_centroid_hz=.3)]
        reference = dict(speed=25., spectral_centroid_hz=.22)
        self.assertEqual(nearest_reference(reference, controls), 's20')
        self.assertEqual(nearest_reference(reference, controls[::-1]), 's20')
        controls[0]['spectral_centroid_hz'] -= 2e-16
        self.assertEqual(nearest_reference(reference, controls), 's20')
        self.assertEqual(stable_minimum([1.+1e-15, 1.], ['a', 'b']), 0)
        with self.assertRaises(ValueError):
            stable_minimum([np.nan], ['a'])


if __name__ == '__main__':
    unittest.main()
