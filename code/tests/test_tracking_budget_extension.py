"""Immutable prefixes, horizon independence, continuation replay and long-history GP fidelity."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import tracking_budget_extension as runner
from src import tracking_budget_extension as policy
from src.tracking_budget import matern
from src.tracking_dense_gain import A, D, E, METHODS, ExactCostSnapshot
from src.tracking_kernel_features import query_posterior
from src.tracking_budget_extension_report import evaluate, verify_24_checkpoint


class BudgetExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = runner.setup(smoke=True)

    def test_programme_prefix_and_suffix_are_horizon_independent(self):
        s = self.state
        prefixes = s['source_state']['specification']['programmes']
        longer = policy.make_programmes(prefixes, s['cfg']['context_metadata'], maximum=130)
        metadata = {r['id']: r for r in s['cfg']['context_metadata']}
        for seed, clients in prefixes.items():
            for client, prefix in clients.items():
                actual = s['cfg']['programmes'][seed][client]
                self.assertEqual(actual[:24], prefix)
                self.assertEqual(actual, longer[seed][client][:100])
                for start in range(24, 97, 3):
                    self.assertEqual(sorted(metadata[c]['frequency'] for c in actual[start:start + 3]), [.1, .3, .6])

    def test_import_keeps_full_state_and_live_payload_has_no_evaluator_labels(self):
        s = dict(self.state)
        with tempfile.TemporaryDirectory() as directory, patch.object(runner.base, 'resolved_rollout', runner.prohibited):
            s['out'] = Path(directory)
            runner.import_prefixes(s)
            job = s['jobs'][0]
            actual = runner.base.read(runner.run_path(s, job))
            original = runner.base.read(s['prefix_paths'][str(job['seed'])])
            self.assertEqual(actual['state'], original['state'])
            self.assertEqual(actual['added_elapsed_seconds'], 0)
        self.assertFalse(set(runner.payload(s)) & {'source_state', 'prefix_paths', 'validation_reuse', 'reference_input_hashes', 'specification'})
        full = runner.setup()
        self.assertEqual(len(full['jobs']), 10)
        self.assertEqual(full['specification']['checkpoints'], [24, 25, 50, 100])
        self.assertEqual(len(full['jobs']) * len(full['clients']) * 76 * len(METHODS), 27360)
        with self.assertRaisesRegex(RuntimeError, 'Smoke'):
            runner.collect(s, 1)

    @staticmethod
    def fake_observe(plans, cache):
        rows = []
        for plan in plans:
            if plan['local_evaluation'] <= 24:
                raise AssertionError('Continuation spent a frozen prefix query again.')
            row = dict(candidate=plan['candidate'], J=1 + (plan['action'] % 17) / 20,
                complete=True, feasible=True, integration_resolved=True, substeps=4)
            cache[runner.base.cache_key(plan)] = dict(client=plan['client'], local_evaluation=plan['local_evaluation'],
                context=plan['context'], noise_seed=plan['noise_seed'], result=row)
            rows.append(dict(eligible=True, **{k: row[k] for k in ('J', 'complete', 'feasible', 'integration_resolved', 'substeps')}))
        return rows

    def test_query_25_resume_replay_and_prefix_tamper_rejection(self):
        s = dict(self.state)
        with tempfile.TemporaryDirectory() as directory:
            s['out'] = Path(directory)
            runner.import_prefixes(s)
            job = s['jobs'][0]
            original = runner.base.read(runner.run_path(s, job))
            args = s['domain'], s['cfg'], s['clients'], job['seed'], 1
            first = policy.run_fleet(*args, self.fake_observe, state=original['state'], stop_round=25)
            prefix_before = runner.body_hash(policy.prefix_view(original['state'], s['clients']))
            self.assertEqual(runner.body_hash(policy.prefix_view(first, s['clients'])), prefix_before)
            complete = policy.run_fleet(*args, self.fake_observe, state=json.loads(json.dumps(first)))
            result = dict(original, state=complete)
            runner.validate_job(runner.payload(s), job, result, complete=True)
            runner.base.save(runner.run_path(s, job), result)
            runner.base.init_worker(runner.payload(s) | dict(prefix_paths=s['prefix_paths']))
            with patch.object(runner.base, 'resolved_rollout', runner.prohibited), patch('src.tracking_budget.minimize', runner.prohibited), patch('src.tracking_six_speed.minimize', runner.prohibited):
                runner.replay_worker(job)
            changed = deepcopy(result)
            changed['state']['methods'][D]['events'][0]['J'] = 999.
            with self.assertRaisesRegex(RuntimeError, 'frozen prefix'):
                runner.validate_job(runner.payload(s), job, changed)
            self.assertEqual(complete['methods'][D]['snapshots']['225']['queries_per_client'], [25] * 9)

    def test_fixed_context_metric_does_not_change_when_an_uncovered_context_arrives(self):
        s = deepcopy(self.state)
        job = s['jobs'][0]
        original = runner.base.read(s['prefix_paths'][str(job['seed'])])
        reused = runner.old_validation_bank(s)
        grouped = {}
        for (context, repeat, candidate), row in reused.items():
            grouped.setdefault(context, {}).setdefault(repeat, []).append(row)
        validation = {c: runner.aggregate_candidates([dict(rows=rows, repeat=r) for r, rows in reps.items()]) for c, reps in grouped.items()}
        import pandas as pd
        frame = pd.read_csv(s['source_state']['out'] / 'empirical_references.csv')
        refs = {r['context']: {k: v for k, v in r.items() if k != 'context'} for r in frame.to_dict('records')}
        s['specification']['checkpoints'] = [24]
        _, units, _ = evaluate(s, [original], validation, refs)
        verify_24_checkpoint(s, units)
        s['specification']['checkpoints'] = [24, 25]
        changed = deepcopy(original)
        for i, client in enumerate(s['clients']):
            contexts = s['cfg']['programmes'][str(job['seed'])][client['id']]
            contexts[24] = next(c for c in s['domain'].contexts if c not in contexts[:24])
        for data in changed['state']['methods'].values():
            data['snapshots']['225'] = deepcopy(data['snapshots']['216'])
            data['snapshots']['225']['round'] = 25
            data['snapshots']['225']['queries_per_client'] = [25] * 9
        _, units, _ = evaluate(s, [changed], validation, refs)
        fixed = units[units.scope == 'fixed_24_contexts']
        np.testing.assert_array_equal(fixed[fixed.queries == 24].context_success, fixed[fixed.queries == 25].context_success)
        self.assertTrue((units[units.scope == 'full_18_contexts'].contexts == 18).all())

    def test_evaluator_only_simulates_missing_rows_and_caches_stably(self):
        s = dict(self.state)
        reuse = runner.old_validation_bank(s)
        context, repeat, first = next(iter(reuse))
        new = next(c for c in s['domain'].candidates if (context, repeat, c) not in reuse)
        task = dict(role='validation', context=context, repeat=repeat, noise_seed=s['specification']['validation_seeds'][repeat], chunk=0, candidates=[first, new])
        def simulate(plant, ref, candidates, physics, noise):
            self.assertEqual([c['id'] for c in candidates], [new])
            self.assertEqual(noise, task['noise_seed'])
            row = dict(reuse[context, repeat, first], candidate=new)
            return [row], None
        with tempfile.TemporaryDirectory() as directory:
            s['out'] = Path(directory)
            runner.base.init_worker(runner.payload(s) | dict(validation_reuse=reuse))
            with patch.object(runner.base, 'resolved_rollout', side_effect=simulate) as mock:
                runner.evaluation_worker(task)
                self.assertEqual(mock.call_count, 1)
            batch = runner.base.read(runner.evaluation_path(s, task))
            self.assertEqual((batch['new_rollouts'], batch['reused_rollouts']), (1, 1))
            self.assertEqual(batch['rows'][0], reuse[context, repeat, first])
            with patch.object(runner.base, 'resolved_rollout', runner.prohibited):
                runner.evaluation_worker(task)

    def test_exact_gp_supports_100_local_observations_without_a_feature_truncation(self):
        s = self.state
        rng = np.random.default_rng(17390)
        histories = []
        for client in range(9):
            actions = rng.choice(150, 100, replace=False)
            histories.append([dict(action=int(a), eligible=True,
                J=float(np.exp(.2 * s['domain'].X[a, 0] + .03 * client))) for a in actions])
        model = ExactCostSnapshot(histories, s['domain'], s['cfg']['gp'], s['cfg']['hierarchical'], True)
        queries = [151, 333, 887]
        inputs = sorted({o['action'] for h in histories for o in h} | set(queries))
        positions = {a: i for i, a in enumerate(inputs)}
        kernel = matern(s['domain'].X[inputs], s['domain'].X[inputs], s['cfg']['gp']['length_scales'],
            s['cfg']['gp']['amplitude'], s['cfg']['gp']['constant_variance'])
        local = [[dict(o, action=positions[o['action']]) for o in h] for h in histories]
        direct = query_posterior(local, kernel, s['cfg']['hierarchical'], s['cfg']['gp'], [0] * 3, [positions[a] for a in queries])
        mean, variance = model.predict(0, queries)
        np.testing.assert_allclose(mean, direct['mean'], rtol=2e-8, atol=2e-8)
        np.testing.assert_allclose(variance, direct['variance'], rtol=2e-8, atol=2e-8)


if __name__ == '__main__':
    unittest.main()
