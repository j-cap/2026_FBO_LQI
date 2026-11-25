"""Sparse context coverage, lazy joint inference, portable replay and failure accounting."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from src.tracking_budget import Domain
from src.tracking_online import features
from src import tracking_hierarchical as joint
from src import tracking_six_speed as six
from src.tracking_six_speed_report import evaluate, paired_comparisons
import tracking_six_speed as runner


def fixture(failure=False):
    baseline = json.loads((ROOT / 'config/runs/tracking_peer_cost.json').read_text())
    metadata = [dict(id=f's{v}_f{f}', speed=v, frequency=f) for v in (20, 22, 24, 26, 28, 30) for f in (.1, .3, .6)]
    candidates = [f'g{i:03d}' for i in range(8)]
    actions = [(r['id'], g) for r in metadata for g in candidates]
    rng = np.random.default_rng(42)
    gain_features = rng.uniform(size=(8, 3))
    X = np.asarray([np.r_[gain_features[g], (r['speed'] - 20) / 10, np.log(r['frequency'] / .1) / np.log(6)]
                    for r in metadata for g in range(8)])
    domain = Domain(X, actions, [r['id'] for r in metadata], candidates,
                    [r['frequency'] for r in metadata], {}, 0)
    cfg = dict(methods=six.METHODS.copy(), gp=deepcopy(baseline['gp']), model=deepcopy(baseline['model']),
        acquisition=baseline['acquisition'], relative_tolerance=.05, queries_per_client=12,
        fleet_budgets=[3, 12, 24, 36], context_metadata=metadata,
        hierarchical=dict(sharing_fractions=[0., .5, 1.], prior_weights=[1, 1, 1]))
    cfg['model']['features'] = 8
    cfg['gp']['optimizer_maxiter'] = 5
    clients = [dict(id=f'client_{i:02d}') for i in range(3)]
    phi = features(domain, cfg['model'])
    def observe(plans, cache):
        observations = []
        for p in plans:
            good = not (failure and p['client'] == 0 and p['local_evaluation'] == 1)
            J = float(.6 + .2 * p['client'] + np.sum((domain.X[p['action'], :3] - .4)**2))
            observations.append(dict(eligible=good, J=J if good else None, complete=True,
                                     feasible=good, integration_resolved=True, substeps=2))
        return observations
    return cfg, domain, clients, phi, observe


class SixSpeedTests(unittest.TestCase):
    def test_programme_repeats_speeds_balances_frequencies_and_is_counterfactually_common(self):
        cfg, domain, clients, _, _ = fixture()
        meta = {r['id']: r for r in cfg['context_metadata']}
        programmes = [six.program(domain, cfg, 5100, c['id']) for c in clients]
        self.assertEqual(programmes, [six.program(domain, cfg, 5100, c['id']) for c in clients])
        self.assertNotEqual(programmes[0], programmes[1])
        for p in programmes:
            self.assertEqual(len(p), 12)
            self.assertLess(len(set(p)), 18)
            for start in range(0, 12, 3):
                self.assertEqual({meta[c]['frequency'] for c in p[start:start + 3]}, {.1, .3, .6})
        self.assertTrue(any(meta[a]['speed'] == meta[b]['speed'] for p in programmes for a, b in zip(p, p[1:])))

    def test_lazy_predictions_match_full_frozen_message_model_and_local_is_independent(self):
        cfg, domain, clients, phi, _ = fixture()
        histories = [[dict(action=a, eligible=True, J=.7 + .1 * a) for a in aa] for aa in ([0, 9], [3, 24], [])]
        for shared in (True, False):
            lazy = six.CostSnapshot(histories, phi, cfg['gp'], cfg['hierarchical'], shared)
            original = joint.cost_posterior(histories, phi, cfg['gp'], cfg['hierarchical'], shared)
            for i in range(len(clients)):
                indices = [5, 1, 10, 71]
                mean, variance = lazy.predict(i, indices)
                np.testing.assert_allclose(mean, np.asarray(original['mean'])[i, indices], atol=2e-10)
                np.testing.assert_allclose(variance, np.asarray(original['variance'])[i, indices], atol=2e-10)
            np.testing.assert_allclose(lazy.weights, original['sharing_weights'], atol=1e-12)
            self.assertEqual(lazy.diagnostics['model_payload_bytes'], original['model_payload_bytes'])
        changed = deepcopy(histories)
        changed[1][0]['J'] *= 20
        a = six.CostSnapshot(histories, phi, cfg['gp'], cfg['hierarchical'], False).predict(0, [0, 5])
        b = six.CostSnapshot(changed, phi, cfg['gp'], cfg['hierarchical'], False).predict(0, [0, 5])
        np.testing.assert_array_equal(a, b)

    def test_resume_and_logged_fit_replay_preserve_all_decisions_without_optimization(self):
        cfg, domain, clients, phi, oracle = fixture(failure=True)
        full = six.run_fleet(domain, cfg, clients, 5100, oracle, phi)
        prefix = six.run_fleet(domain, cfg, clients, 5100, oracle, phi, stop_round=4)
        resumed = six.run_fleet(domain, cfg, clients, 5100, oracle, phi, state=prefix)
        six.assert_equivalent(full['methods'], resumed['methods'])
        with patch('src.tracking_budget.minimize', side_effect=AssertionError('No refits during audit')):
            six.validate_state(full, domain, cfg, clients, 5100, phi, complete=True, replay=True)
        for m in six.METHODS:
            record = full['methods'][m]
            self.assertEqual(len(record['events']), 36)
            self.assertTrue(all(len(h) == 12 for h in record['histories']))
            for e in record['events']:
                self.assertEqual(e['information_round'], e['round'] - 1)
                self.assertEqual(e['mode'] == 'factory_initialization', e['local_evaluation'] == 1)
        self.assertTrue(any(e.get('gp_fit_count', 0) for e in full['methods'][six.METHODS[0]]['events']))

    def test_peer_priority_and_provenance_are_checked(self):
        cfg, domain, clients, phi, oracle = fixture()
        full = six.run_fleet(domain, cfg, clients, 5101, oracle, phi)
        e = next(e for e in full['methods'][six.SHARED]['events'] if e['mode'] == 'peer_gain_verification')
        self.assertLess(e['peer_donor_query'], e['local_evaluation'])
        broken = deepcopy(full)
        event = next(e for e in broken['methods'][six.SHARED]['events'] if e['mode'] == 'peer_gain_verification')
        event['peer_donor_query'] += 1
        with self.assertRaisesRegex(RuntimeError, 'Peer priority'):
            six.validate_state(broken, domain, cfg, clients, 5101, phi)

    def test_logged_parameter_carryover_and_schedule_tampering_are_rejected(self):
        cfg, domain, clients, phi, oracle = fixture()
        full = six.run_fleet(domain, cfg, clients, 5100, oracle, phi)
        event = next(e for e in full['methods'][six.METHODS[0]]['events'] if e['mode'] == 'feasibility_penalized_LCB')
        event['gp_fit_count'] += 1
        with self.assertRaisesRegex(RuntimeError, 'refit schedule'):
            six.validate_state(full, domain, cfg, clients, 5100, phi, replay=True)

    def test_resume_rejects_changed_gp_state_and_visit_metadata(self):
        cfg, domain, clients, phi, oracle = fixture()
        original = six.run_fleet(domain, cfg, clients, 5100, oracle, phi, stop_round=4)
        bad = deepcopy(original)
        bad['methods'][six.METHODS[0]]['gp_parameters'][0][0] += .1
        with self.assertRaisesRegex(RuntimeError, 'Checkpoint GP state'):
            six.validate_state(bad, domain, cfg, clients, 5100, phi)
        bad = deepcopy(original)
        bad['methods'][six.SHARED]['events'][0]['speed_visit'] = 2
        with self.assertRaisesRegex(RuntimeError, 'Visit metadata'):
            six.validate_state(bad, domain, cfg, clients, 5100, phi)

    def test_unvisited_contexts_are_separate_and_failed_first_visits_stay_in_denominator(self):
        cfg, domain, clients, phi, oracle = fixture(failure=True)
        data = six.run_fleet(domain, cfg, clients, 5100, oracle, phi)
        refs = {c: dict(candidate=domain.candidates[0], eligible=True, J=1.) for c in domain.contexts}
        val = {c: {g: dict(eligible=True, J=1.) for g in domain.candidates} for c in domain.contexts}
        trials, cells, visits, _ = evaluate([dict(seed=5100, clients=clients, state=data)], domain, cfg, refs, val)
        final = trials[trials.fleet_budget == 36]
        self.assertTrue((final.encountered_contexts + final.unvisited_contexts == 18).all())
        self.assertTrue((final.unvisited_contexts >= 6).all())
        self.assertFalse(final.complete_eligible_on_full_grid.any())
        self.assertEqual(int(final.encountered_contexts.sum()), len(cells[cells.fleet_budget == 36]))
        bad = visits[(visits.client == clients[0]['id']) & (visits.local_evaluation == 1)]
        self.assertEqual(len(bad), 4)
        self.assertFalse(bad.requested_eligible.any())
        self.assertFalse(bad.requested_success.any())
        comparisons, _ = paired_comparisons(trials)
        self.assertEqual(len(comparisons), 12)

    def test_worker_cannot_access_evaluator_or_old_results_and_smoke_cannot_collect(self):
        state = {k: None for k in ('cfg', 'physics', 'clients', 'pool', 'refs', 'domain', 'phi', 'out', 'signature')}
        state.update(old_histories='secret', validation='secret', references='secret', report='secret')
        self.assertFalse({'old_histories', 'validation', 'references', 'report'} & runner.payload(state).keys())
        with self.assertRaisesRegex(RuntimeError, 'Smoke'):
            runner.collect(dict(specification=dict(mode='smoke')), 1)


if __name__ == '__main__':
    unittest.main()
