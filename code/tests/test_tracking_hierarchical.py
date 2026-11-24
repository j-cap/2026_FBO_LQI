"""Joint posterior algebra, local information barriers and frozen peer protocol."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
from test_tracking_peer_cost import fixture as original_fixture
from src import tracking_hierarchical as joint
from src import tracking_peer_cost as peer
from src.tracking_peer_cost_report import evaluate, eligibility_times
from src.tracking_hierarchical_report import benchmark_readout, paired_comparisons, write_report
import tracking_hierarchical as runner


def fixture(all_methods=False):
    cfg, domain, clients, oracle = original_fixture()
    cfg.update(methods=(peer.METHODS if all_methods else [])+joint.METHODS,
        comparison_methods=peer.METHODS+joint.METHODS,
        conditions=['homogeneous', 'heterogeneous'],
        hierarchical=dict(sharing_fractions=[0., .5, 1.], prior_weights=[1, 1, 1]))
    return cfg, domain, clients, oracle


def dense_reference(histories, F, fraction, noise):
    """Independent GP conditioning over input/client pairs (no message formulas)."""
    observations = [(i, o['action'], np.log(o['J'])) for i, h in enumerate(histories) for o in h if o['eligible']]
    indices = [a for _, a, _ in observations]
    tasks = np.asarray([i for i, _, _ in observations])
    y = np.asarray([y for _, _, y in observations])
    K = (F[indices]@F[indices].T)*(fraction+(1-fraction)*(tasks[:, None] == tasks))
    K.flat[::len(K)+1] += noise
    factor = cho_factor(K, lower=True)
    alpha = cho_solve(factor, y)
    predictions = []
    for i in range(len(histories)):
        cross = (F@F[indices].T)*(fraction+(1-fraction)*(tasks == i))
        predictions.append((cross@alpha, np.sum(F*F, axis=1)-np.einsum('ij,ji->i', cross, cho_solve(factor, cross.T))))
    evidence = -.5*(y@alpha+2*np.log(np.diag(factor[0])).sum()+len(y)*np.log(2*np.pi))
    return predictions, evidence


class PosteriorTests(unittest.TestCase):
    def test_messages_match_dense_joint_conditioning_including_endpoints_and_evidence(self):
        rng = np.random.default_rng(12)
        F = rng.normal(size=(7, 4))
        histories = [[dict(action=a, eligible=True, J=float(np.exp(rng.normal()))) for a in actions]
                     for actions in [[0, 1, 3], [0, 2], []]]
        for rho in [0., .2, .8, 1.]:
            blocks = [joint.ClientBlock(h, F, rho, .05**2) for h in histories]
            mu, covariance, evidence = joint.aggregate_messages([b.information_message() for b in blocks], F.shape[1])
            expected, expected_evidence = dense_reference(histories, F, rho, .05**2)
            self.assertAlmostEqual(evidence, expected_evidence, places=7)
            for block, (mean, variance) in zip(blocks, expected):
                actual = block.predict(F, mu, covariance)
                np.testing.assert_allclose(actual[0], mean, rtol=1e-8, atol=1e-8)
                np.testing.assert_allclose(actual[1], variance, rtol=1e-8, atol=1e-8)

    def test_mixture_moments_include_between_component_uncertainty(self):
        cfg, domain, _, _ = fixture(); phi = joint.features(domain, cfg['model'])
        histories = [[dict(action=0, eligible=True, J=.3)], [dict(action=0, eligible=True, J=2.)]]
        posterior = joint.cost_posterior(histories, phi, cfg['gp'], cfg['hierarchical'])
        F = phi*np.r_[np.sqrt(cfg['gp']['constant_variance']), np.full(phi.shape[1]-1, cfg['gp']['amplitude'])]
        components = [dense_reference(histories, F, rho, cfg['gp']['noise_std_log_J']**2+1e-9)
                      for rho in cfg['hierarchical']['sharing_fractions']]
        evidences = np.asarray([c[1] for c in components])
        weights = np.exp(evidences-logsumexp(evidences))
        means = np.asarray([[p[0] for p in c[0]] for c in components])
        variances = np.asarray([[p[1] for p in c[0]] for c in components])
        mean = np.einsum('k,kia->ia', weights, means)
        variance = np.einsum('k,kia->ia', weights, variances+(means-mean[None])**2)
        np.testing.assert_allclose(posterior['sharing_weights'], weights, atol=1e-9)
        np.testing.assert_allclose(posterior['mean'], mean, atol=1e-8)
        np.testing.assert_allclose(posterior['variance'], variance, atol=1e-8)

    def test_one_vehicle_cannot_identify_sharing_and_each_label_is_conditioned_once(self):
        cfg, domain, _, _ = fixture(); phi = joint.features(domain, cfg['model'])
        histories = [[dict(action=a, eligible=True, J=.5+a*.1) for a in [0, 1, 3]]]
        shared = joint.cost_posterior(histories, phi, cfg['gp'], cfg['hierarchical'])
        local = joint.cost_posterior(histories, phi, cfg['gp'], cfg['hierarchical'], shared=False)
        np.testing.assert_allclose(shared['sharing_weights'], [1/3]*3, atol=1e-9)
        np.testing.assert_allclose(shared['mean'], local['mean'], atol=1e-9)
        np.testing.assert_allclose(shared['variance'], local['variance'], atol=1e-9)

    def test_local_control_cannot_use_other_client_costs_and_failed_costs_are_excluded(self):
        cfg, domain, _, _ = fixture(); phi = joint.features(domain, cfg['model'])
        histories = [[dict(action=0, eligible=True, J=.5)], [dict(action=0, eligible=True, J=2.)]]
        a = joint.cost_posterior(histories, phi, cfg['gp'], cfg['hierarchical'], shared=False)
        changed = deepcopy(histories); changed[1][0]['J'] = 100.
        b = joint.cost_posterior(changed, phi, cfg['gp'], cfg['hierarchical'], shared=False)
        np.testing.assert_array_equal(a['mean'][0], b['mean'][0])
        for h in histories:
            h.append(dict(action=4, eligible=False, J=None))
        failed = joint.cost_posterior(histories, phi, cfg['gp'], cfg['hierarchical'], shared=False)
        np.testing.assert_array_equal(a['mean'], failed['mean'])
        self.assertEqual(a['model_payload_bytes'], 0)
        empty = joint.cost_posterior([[], []], phi, cfg['gp'], cfg['hierarchical'])
        self.assertEqual(empty['model_payload_bytes'], 0)
        np.testing.assert_array_equal(empty['mean'], 0.)
        self.assertTrue(np.isfinite(empty['variance']).all())


class ProtocolTests(unittest.TestCase):
    def test_frozen_arms_are_unchanged_and_new_methods_obey_peer_priority(self):
        cfg, domain, clients, oracle = fixture(all_methods=True)
        old = peer.run_fleet(domain, dict(cfg, methods=peer.METHODS), clients, 1101, oracle)
        new = joint.run_fleet(domain, cfg, clients, 1101, oracle)
        joint.validate_state(new, domain, cfg, clients, 1101, complete=True)
        joint.replay_new_methods(new, domain, cfg, clients, 1101)
        for method in peer.METHODS:
            self.assertEqual(old['methods'][method], new['methods'][method])
        for method in joint.METHODS:
            data = new['methods'][method]
            self.assertEqual(len(data['events']), 18)
            self.assertTrue(any(e['mode'] == 'peer_gain_verification' for e in data['events']))
            self.assertEqual(data['snapshots']['3']['model_payload_bytes'], 0)
            self.assertEqual(data['snapshots']['18']['model_payload_bytes'], data['learning'][-2]['cumulative_model_payload_bytes'])
            self.assertEqual(data['model_payload_bytes'] > 0, method == joint.SHARED)

    def test_resume_matches_full_run_and_stale_posterior_and_payload_forgery_are_rejected(self):
        cfg, domain, clients, oracle = fixture()
        full = joint.run_fleet(domain, cfg, clients, 1100, oracle)
        partial = joint.run_fleet(domain, cfg, clients, 1100, oracle, stop_round=3)
        partial = json.loads(json.dumps(partial, allow_nan=False))
        resumed = joint.run_fleet(domain, cfg, clients, 1100, oracle, state=partial)
        self.assertEqual(full, resumed)
        for key, value in [('information_round', 99), ('model_payload_bytes', 123)]:
            bad = deepcopy(partial)
            bad['methods'][joint.SHARED]['cost_posterior'][key] = value
            with self.assertRaises(RuntimeError):
                joint.validate_state(bad, domain, cfg, clients, 1100)
        bad = deepcopy(full)
        bad['methods'][joint.SHARED]['learning'][0]['sharing_weights'] = [1., 0., 0.]
        with self.assertRaisesRegex(RuntimeError, 'learning replay'):
            joint.replay_new_methods(bad, domain, cfg, clients, 1100)

    def test_failed_verifications_cost_budget_and_no_eligible_labels_remain_valid(self):
        cfg, domain, clients, oracle = fixture()
        def failures(plans, cache):
            observations = oracle(plans, cache)
            for p, o in zip(plans, observations):
                if p['client'] != 1:
                    o.update(eligible=False, J=None, feasible=False)
            return observations
        state = joint.run_fleet(domain, cfg, clients, 1101, failures)
        joint.validate_state(state, domain, cfg, clients, 1101, complete=True)
        for data in state['methods'].values():
            self.assertEqual(len(data['events']), 18)
            self.assertTrue(any(e['mode'] == 'peer_gain_verification' and not e['eligible'] for e in data['events']))
            self.assertEqual(data['snapshots']['18']['recommendations'][0], [None]*2)
        def no_labels(plans, cache):
            return [dict(eligible=False, J=None, complete=True, feasible=False,
                         integration_resolved=True, substeps=2) for _ in plans]
        empty = joint.run_fleet(domain, cfg, clients, 1100, no_labels)
        joint.validate_state(empty, domain, cfg, clients, 1100, complete=True)
        self.assertTrue(all(d['payload_bytes'] == 0 for d in empty['methods'].values()))

    def test_other_arm_records_cannot_change_shared_model_decisions(self):
        cfg, domain, clients, oracle = fixture(all_methods=True)
        state = joint.run_fleet(domain, cfg, clients, 1100, oracle, stop_round=1)
        phi = joint.features(domain, cfg['model'])
        a = joint.plan_round(deepcopy(state), domain, cfg, clients, 1100, phi)
        changed = deepcopy(state)
        changed['methods'][peer.METHODS[2]]['weights'] = [100.]*phi.shape[1]
        for h in changed['methods'][peer.METHODS[2]]['histories']:
            h[0]['J'] = 1000.
        b = joint.plan_round(changed, domain, cfg, clients, 1100, phi)
        self.assertEqual([p for p in a if p['method'] == joint.SHARED], [p for p in b if p['method'] == joint.SHARED])


class EvaluationTests(unittest.TestCase):
    def test_a_is_primary_and_a_denominators_are_not_replaced_by_favorable_b_comparison(self):
        cfg, domain, clients, oracle = fixture(all_methods=True)
        data = joint.run_fleet(domain, cfg, clients, 1101, oracle)
        refs = {('target', c): dict(J=1., eligible=True) for c in domain.contexts}
        val = {('target', c, k): dict(J=1., eligible=True) for c, k in domain.actions}
        job = dict(condition='homogeneous', seed=1101, clients=clients, state=data)
        trials, cells, queries, learning = evaluate([job], domain, cfg, (refs, val))
        # Deliberately let D beat B and lose to A, retaining one missing D policy.
        last = trials.fleet_budget == 18
        trials.loc[last & (trials.method == peer.METHODS[0]), 'context_success'] = 1.
        trials.loc[last & (trials.method == peer.METHODS[1]), 'context_success'] = .25
        trials.loc[last & (trials.method == joint.SHARED), 'context_success'] = .5
        missing = last & (trials.method == joint.SHARED) & (trials.client == clients[0]['id'])
        trials.loc[missing, 'fully_feasible'] = False
        trials.loc[missing, 'J'] = np.nan
        paired, per_seed = paired_comparisons(trials), paired_comparisons(trials, per_seed=True)
        readout = benchmark_readout(paired, per_seed, 18)
        a = readout[readout.baseline == peer.METHODS[0]].iloc[0]
        b = readout[readout.baseline == peer.METHODS[1]].iloc[0]
        self.assertFalse(a.primary_context_metric_improved)
        self.assertFalse(a.all_three_aggregate_metrics_support_improvement)
        self.assertTrue(b.primary_context_metric_improved)
        self.assertEqual(a.jointly_feasible_pairs, 2)
        self.assertEqual(a.eligible_policies_method, 2)
        times, _ = eligibility_times([job], domain, cfg)
        with tempfile.TemporaryDirectory() as tmp:
            write_report(Path(tmp), trials, cells, queries, learning, times, cfg, 'smoke')
            html = (Path(tmp)/'report.html').read_text()
            self.assertLess(html.index('Primary: D minus A'), html.index('Secondary: D minus B'))
            self.assertTrue((Path(tmp)/'sharing_strength.csv').exists())

    def test_smoke_and_missing_results_cannot_be_collected(self):
        with self.assertRaisesRegex(RuntimeError, 'Smoke'):
            runner.collect(dict(specification={'mode': 'smoke'}))
        with patch.object(runner, 'analyze', side_effect=RuntimeError('Missing checkpoint')):
            with patch.object(runner.subprocess, 'run') as process:
                with self.assertRaisesRegex(RuntimeError, 'Missing'):
                    runner.collect(dict(specification={'mode': 'full'}))
                process.assert_not_called()


if __name__ == '__main__':
    unittest.main()
