"""Bound-gradient recovery and audited, interruption-safe checkpoint provenance repair."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_tracking_six_speed import fixture, runner, six
from src import tracking_budget
from src.tracking_online_lcb import choose


class BoundedFitTests(unittest.TestCase):
    def test_gradient_stays_in_bounds_at_both_endpoints_and_does_not_mutate_input(self):
        bounds = np.asarray([[-2., 0.], [.5, 2.]])
        point = np.asarray([np.nextafter(-2., -np.inf), np.nextafter(2., np.inf)])
        original = point.copy()
        def loss(x):
            self.assertTrue(np.all(x >= bounds[:, 0]) and np.all(x <= bounds[:, 1]))
            return float(np.sum((x - [-1., 1.])**2))
        actual = six._bounded_gradient(loss, point, bounds)
        np.testing.assert_allclose(actual, [-2., 2.], atol=1e-6)
        np.testing.assert_array_equal(point, original)

    def test_ordinary_fit_is_unchanged_and_only_the_specific_error_is_recovered(self):
        cfg, domain, _, _, _ = fixture()
        history = [dict(action=a, eligible=True, J=.5 + .1 * a) for a in (0, 9, 24, 32)]
        prior = np.zeros(len(domain.actions))
        params = np.log([*cfg['gp']['length_scales'], cfg['gp']['amplitude']]).tolist()
        context = domain.contexts[-1]
        old = choose(history, domain, context, prior, cfg['gp'], cfg['acquisition'], params, 0, np.random.default_rng(8))
        new = six._gp_proposal(history, domain, context, prior, cfg, params, 0, np.random.default_rng(8))
        self.assertEqual(old[0], new[0])
        np.testing.assert_array_equal(old[1], new[1])
        self.assertEqual(old[2], new[2])
        self.assertNotIn('gp_optimizer_recovery', new[3])
        with patch('src.tracking_budget.minimize', side_effect=ValueError('unrelated optimizer error')):
            with patch.object(six, '_bounded_refit') as fallback:
                with self.assertRaisesRegex(ValueError, 'unrelated'):
                    six._gp_proposal(history, domain, context, prior, cfg, params, 0, np.random.default_rng(8))
                fallback.assert_not_called()

    def test_reported_failure_recovers_and_is_replayed_without_repeating_the_fit(self):
        cfg, domain, clients, phi, oracle = fixture()
        original = tracking_budget.minimize
        failures = []
        def fail_once(*args, **kwargs):
            if not failures:
                failures.append(True)
                raise ValueError('`x0` violates bound constraints.')
            return original(*args, **kwargs)
        with patch('src.tracking_budget.minimize', side_effect=fail_once):
            data = six.run_fleet(domain, cfg, clients, 5100, oracle, phi)
        recovered = [e for m in data['methods'].values() for e in m['events'] if e.get('gp_optimizer_recovery')]
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0]['gp_optimizer_recovery'], 'bounded_finite_difference_v1')
        bounds = np.log([cfg['gp']['length_bounds']] * 5 + [cfg['gp']['amplitude_bounds']])
        params = np.asarray(recovered[0]['gp_fit_parameters'])
        self.assertTrue(np.all(params >= bounds[:, 0]) and np.all(params <= bounds[:, 1]))
        with patch('src.tracking_budget.minimize', side_effect=AssertionError('no fresh optimization')):
            with patch.object(six, 'minimize', side_effect=AssertionError('no recovery optimization')):
                six.validate_state(data, domain, cfg, clients, 5100, phi, complete=True, replay=True)


def repair_fixture(out):
    cfg, domain, clients, phi, oracle = fixture()
    def cached(plans, cache):
        rows = oracle(plans, cache)
        for p, o in zip(plans, rows):
            cache[runner.cache_key(p)] = dict(client=p['client'], local_evaluation=p['local_evaluation'],
                context=p['context'], noise_seed=p['noise_seed'], result=dict(candidate=p['candidate'], **o))
        return rows
    sources = dict(runner.LEGACY_BOUND_SOURCES, **{'code/src/unchanged.py': 'unchanged'})
    old_spec = dict(configuration=cfg, source_hashes=sources)
    signature = lambda spec: hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    new_spec = deepcopy(old_spec)
    for name in runner.LEGACY_BOUND_SOURCES:
        new_spec['source_hashes'][name] = runner.digest(runner.REPO / name)
    job = dict(condition='homogeneous', fleet_id='homogeneous', replicate=0, seed=5100)
    state = dict(cfg=cfg, domain=domain, clients=clients, phi=phi, jobs=[job], out=out,
                 specification=new_spec, signature=signature(new_spec))
    data = six.run_fleet(domain, cfg, clients, 5100, cached, phi, stop_round=5)
    checkpoint = dict(**job, signature=signature(old_spec), clients=clients, state=data)
    runner.save(out / 'run_manifest.json', dict(**old_spec, signature=signature(old_spec), runtime={'python': 'original'}))
    runner.save(runner.run_path(state, job), checkpoint)
    return state, job, checkpoint, cached


class RepairTests(unittest.TestCase):
    def test_repair_preserves_payload_backs_up_originals_and_continues_the_partial_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            state, job, old, oracle = repair_fixture(out)
            path = runner.run_path(state, job)
            original_bytes = path.read_bytes()
            with patch.object(runner, 'resolved_rollout', side_effect=AssertionError('no simulation during repair')):
                with patch('src.tracking_budget.minimize', side_effect=AssertionError('no refits during repair')):
                    runner.repair_resume(state)
            repaired = runner.read(path)
            self.assertEqual(repaired['signature'], state['signature'])
            self.assertEqual(repaired['state'], old['state'])
            backup = out / 'repairs' / runner.REPAIR_NAME / 'backup/runs/5100.json'
            self.assertEqual(backup.read_bytes(), original_bytes)
            self.assertTrue(runner.read(out / 'repairs' / runner.REPAIR_NAME / 'repair.json')['complete'])
            mtime = path.stat().st_mtime_ns
            runner.repair_resume(state)
            self.assertEqual(path.stat().st_mtime_ns, mtime)
            resumed = six.run_fleet(state['domain'], state['cfg'], state['clients'], 5100, oracle,
                                    state['phi'], state=repaired['state'])
            direct = six.run_fleet(state['domain'], state['cfg'], state['clients'], 5100, oracle, state['phi'])
            six.assert_equivalent(resumed['methods'], direct['methods'])

    def test_interrupted_repair_refuses_normal_run_then_resumes_without_changing_histories(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            state, job, old, _ = repair_fixture(out)
            save = runner.save
            def interrupted(path, value):
                if path == out / 'run_manifest.json' and value['signature'] == state['signature']:
                    raise RuntimeError('simulated interruption')
                return save(path, value)
            with patch.object(runner, 'save', side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    runner.repair_resume(state)
            with self.assertRaisesRegex(RuntimeError, 'Interrupted GP bounds repair'):
                runner.initialize(state)
            runner.repair_resume(state)
            self.assertEqual(runner.read(runner.run_path(state, job))['state'], old['state'])
            self.assertEqual(runner.read(out / 'run_manifest.json')['signature'], state['signature'])

    def test_unrelated_source_change_is_rejected_before_any_checkpoint_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            state, job, _, _ = repair_fixture(out)
            path = runner.run_path(state, job)
            before = path.read_bytes()
            state['specification']['source_hashes']['code/src/unchanged.py'] = 'different'
            with self.assertRaisesRegex(RuntimeError, 'source transition'):
                runner.repair_resume(state)
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse((out / 'repairs').exists())


if __name__ == '__main__':
    unittest.main()
