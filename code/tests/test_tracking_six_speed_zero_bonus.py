"""Causal matching, separate histories, online acquisition and restart evidence."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_tracking_six_speed import fixture
from src import tracking_six_speed as reference
from src import tracking_six_speed_zero_bonus as zero
from src.tracking_six_speed_report import evaluate
from src.tracking_six_speed_zero_bonus_report import paired_comparisons, write_report, CONTRASTS, LABELS
import tracking_six_speed_zero_bonus as runner


def design(failure=False):
    cfg, domain, clients, phi, observe = fixture(failure)
    cfg.update(tag='unit_test', methods=zero.METHODS.copy(), bonus_by_method=zero.BONUSES.copy())
    return cfg, domain, clients, phi, observe


class MatchedZeroBonusTests(unittest.TestCase):
    def test_four_reference_arms_are_unchanged_by_the_ablation_runner(self):
        cfg, domain, clients, phi, oracle = design(failure=True)
        matched = zero.run_fleet(domain, cfg, clients, 6100, oracle, phi)
        control_cfg = dict(cfg, methods=reference.METHODS.copy())
        controls = reference.run_fleet(domain, control_cfg, clients, 6100, oracle, phi)
        for method in reference.METHODS:
            reference.assert_equivalent(matched['methods'][method], controls['methods'][method])
        for method in zero.METHODS:
            data = matched['methods'][method]
            self.assertEqual(len(data['events']), 36)
            self.assertTrue(all(len(h) == 12 for h in data['histories']))
            self.assertTrue(all(e['information_round'] == e['round'] - 1 for e in data['events']))
            self.assertEqual(sum(not e['eligible'] for e in data['events']), 1)
        self.assertEqual(sum(len(r['events']) for r in matched['methods'].values()), 216)

    def test_zero_bonus_is_variance_invariant_but_preserves_eligibility_penalty(self):
        cfg, domain, clients, phi, oracle = design()
        state = zero.run_fleet(domain, cfg, clients, 6100, oracle, phi, stop_round=1)
        class FakeModel:
            def __init__(self, huge):
                self.huge = huge
            def predict(self, client, actions):
                n = len(actions)
                variance = np.zeros(n)
                variance[-1] = 1e6 if self.huge else 0.
                return np.arange(n) * .05, variance
        probability = np.full(len(domain.actions), .5)
        for context in domain.contexts:
            probability[domain.action(context, domain.candidates[0])] = .001
        with patch('src.tracking_six_speed.feasibility_probability', return_value=probability), \
                patch('src.tracking_six_speed_zero_bonus.feasibility_probability', return_value=probability), \
                patch('src.tracking_peer_cost.peer_proposal', return_value=(None, dict.fromkeys(reference.peer.PEER_FIELDS))):
            a = zero.plan_round(deepcopy(state), domain, cfg, clients, 6100,
                {m: FakeModel(False) for m in zero.SPECTRAL}, phi)
            b = zero.plan_round(deepcopy(state), domain, cfg, clients, 6100,
                {m: FakeModel(True) for m in zero.SPECTRAL}, phi)
        for method in (zero.D0, zero.E0):
            aa = [p for p in a if p['method'] == method]
            bb = [p for p in b if p['method'] == method]
            self.assertEqual([p['action'] for p in aa], [p['action'] for p in bb])
            self.assertTrue(all(p['candidate'] == domain.candidates[1] for p in bb))
            for p in bb:
                self.assertAlmostEqual(p['acquisition_score'], p['predicted_log_J'] - np.log(.5))
                self.assertEqual(p['predicted_variance'], 0.)
        for method in (zero.D2, zero.E2):
            self.assertNotEqual([p['action'] for p in a if p['method'] == method],
                [p['action'] for p in b if p['method'] == method])

    def test_resume_and_full_replay_need_neither_optimizer_nor_rollouts(self):
        cfg, domain, clients, phi, oracle = design(failure=True)
        full = zero.run_fleet(domain, cfg, clients, 6100, oracle, phi)
        prefix = zero.run_fleet(domain, cfg, clients, 6100, oracle, phi, stop_round=4)
        resumed = zero.run_fleet(domain, cfg, clients, 6100, oracle, phi, state=prefix)
        reference.assert_equivalent(full['methods'], resumed['methods'])
        with patch('src.tracking_budget.minimize', side_effect=AssertionError('No optimizer during replay')), \
                patch.object(runner.base, 'resolved_rollout', side_effect=AssertionError('No rollout during replay')):
            zero.validate_state(full, domain, cfg, clients, 6100, phi, complete=True, replay=True)
        self.assertTrue(any(e['mode'] == 'peer_gain_verification' for e in full['methods'][zero.D0]['events']))
        broken = deepcopy(full)
        e = next(e for e in broken['methods'][zero.D0]['events'] if e['mode'] == 'peer_gain_verification')
        e['peer_donor_query'] += 1
        with self.assertRaisesRegex(RuntimeError, 'Peer priority'):
            zero.validate_state(broken, domain, cfg, clients, 6100, phi)

    def test_zero_bonus_score_tampering_is_rejected_by_replay(self):
        cfg, domain, clients, phi, oracle = design()
        full = zero.run_fleet(domain, cfg, clients, 6100, oracle, phi)
        e = next(e for e in full['methods'][zero.D0]['events'] if e['mode'] == 'feasibility_penalized_LCB')
        e['acquisition_score'] += 1
        with self.assertRaisesRegex(RuntimeError, 'Replay numeric'):
            zero.validate_state(full, domain, cfg, clients, 6100, phi, replay=True)

    def test_numerical_reuse_only_for_equal_fitting_inputs_and_never_shares_labels(self):
        cfg, domain, clients, phi, oracle = design()
        state = zero.run_fleet(domain, cfg, clients, 6100, oracle, phi, stop_round=1)
        models = zero.fit_models(state, phi, cfg)
        self.assertIs(models[zero.D0], models[zero.D2])
        self.assertIs(models[zero.E0], models[zero.E2])
        untouched = models[zero.D2].predict(0, [1, 2])
        original = deepcopy(state['methods'][zero.D2]['histories'])
        state['methods'][zero.D0]['histories'][0][0]['J'] *= 10
        changed = zero.fit_models(state, phi, cfg)
        self.assertIsNot(changed[zero.D0], changed[zero.D2])
        np.testing.assert_array_equal(changed[zero.D2].predict(0, [1, 2]), untouched)
        self.assertEqual(state['methods'][zero.D2]['histories'], original)
        self.assertFalse(np.allclose(changed[zero.D0].predict(0, [1, 2]), untouched))

    def test_reports_cover_all_contrasts_and_keep_failed_encounters(self):
        cfg, domain, clients, phi, oracle = design(failure=True)
        data = zero.run_fleet(domain, cfg, clients, 6100, oracle, phi)
        refs = {c: dict(candidate=domain.candidates[0], eligible=True, J=1.) for c in domain.contexts}
        validation = {c: {g: dict(eligible=True, J=1.) for g in domain.candidates} for c in domain.contexts}
        frames = evaluate([dict(seed=6100, clients=clients, state=data)], domain, cfg, refs, validation)
        trials, cells, visits, learning = frames
        self.assertEqual(len(visits), 216)
        first_bad = visits[(visits.client == clients[0]['id']) & (visits.local_evaluation == 1)]
        self.assertEqual(len(first_bad), 6)
        self.assertFalse(first_bad.requested_success.any())
        self.assertTrue((trials.encountered_contexts + trials.unvisited_contexts == 18).all())
        comparisons, units = paired_comparisons(trials)
        self.assertEqual(len(comparisons), len(cfg['fleet_budgets']) * len(CONTRASTS))
        self.assertEqual(set(comparisons.role), {r for _, _, r in CONTRASTS})
        self.assertTrue((comparisons.programme_seeds == 1).all())
        self.assertTrue(np.isfinite(units.success_delta_pp).all())
        with tempfile.TemporaryDirectory() as directory:
            state = dict(out=Path(directory), cfg=cfg, signature='test',
                specification=dict(mode='smoke', basis_accuracy=dict(
                    relative_kernel_Frobenius_error=.0052, max_relative_diagonal_error=.0511)))
            write_report(state, *frames, refs)
            html = (Path(directory) / 'report.html').read_text()
            for method in zero.METHODS:
                self.assertIn(LABELS[method], html)
            self.assertIn('software verification only', html)
            self.assertEqual(len(list(Path(directory).glob('*.csv'))), 11)

    def test_pinned_basis_fresh_programmes_worker_isolation_and_collection_guard(self):
        state = runner.setup()
        frozen = runner.base.setup()
        np.testing.assert_array_equal(state['phi'], frozen['phi'])
        np.testing.assert_array_equal(state['domain'].X, frozen['domain'].X)
        self.assertFalse({j['seed'] for j in state['jobs']} & {j['seed'] for j in frozen['jobs']})
        self.assertEqual(len(state['clients']) * len(state['jobs']) * 24 * len(zero.METHODS), 3888)
        enriched = dict(state, validation='private', old_histories='private', report='private')
        payload = runner.base.payload(enriched)
        self.assertFalse({'validation', 'old_histories', 'report', 'specification'} & payload.keys())
        self.assertNotIn('results', json.dumps(payload['cfg']))
        with self.assertRaisesRegex(RuntimeError, 'Smoke'):
            runner.collect(dict(specification=dict(mode='smoke')), 1)


if __name__ == '__main__':
    unittest.main()
