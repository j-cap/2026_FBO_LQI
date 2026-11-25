"""Matched controls, globally fixed exploration, replay and honest pilot readout."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from test_tracking_six_speed import fixture
from src import tracking_six_speed as reference
from src import tracking_six_speed_zero_bonus as zero
from src import tracking_six_speed_small_bonus as small
from src.tracking_six_speed_report import evaluate
from src.tracking_six_speed_small_bonus_report import CONTRASTS, LABELS, paired_comparisons, pilot_readout, write_report
import tracking_six_speed_small_bonus as runner


def design(failure=False):
    cfg, domain, clients, phi, oracle = fixture(failure)
    cfg.update(tag='unit_test', methods=small.METHODS.copy(), bonus_by_method=small.BONUSES.copy())
    return cfg, domain, clients, phi, oracle


class SmallBonusTests(unittest.TestCase):
    def test_A_B_and_zero_bonus_controls_match_the_frozen_implementation(self):
        cfg, domain, clients, phi, oracle = design(failure=True)
        current = small.run_fleet(domain, cfg, clients, 7100, oracle, phi)
        previous_cfg = dict(cfg, methods=zero.METHODS.copy(), bonus_by_method=zero.BONUSES.copy())
        previous = zero.run_fleet(domain, previous_cfg, clients, 7100, oracle, phi)
        for method in (small.A, small.B, small.D0, small.E0):
            reference.assert_equivalent(current['methods'][method], previous['methods'][method])
        self.assertEqual(sum(len(v['events']) for v in current['methods'].values()), 216)
        for method in small.METHODS:
            data = current['methods'][method]
            self.assertTrue(all(len(h) == 12 for h in data['histories']))
            self.assertEqual(sum(not e['eligible'] for e in data['events']), 1)
            for event in data['events']:
                self.assertEqual(event['information_round'], event['round'] - 1)
                if event['mode'] == 'feasibility_penalized_LCB':
                    penalty = -np.log(max(event['eligibility_probability'], 1e-6))
                    expected = event['predicted_log_J'] - small.BONUSES[method] * np.sqrt(event['predicted_variance']) + penalty
                    self.assertAlmostEqual(event['acquisition_score'], expected)

    def test_beta_half_applies_in_every_context_with_the_same_eligibility_penalty(self):
        cfg, domain, clients, phi, oracle = design()
        prefix = small.run_fleet(domain, cfg, clients, 7100, oracle, phi, stop_round=1)
        class Model:
            def predict(self, client, actions):
                candidates = np.asarray([domain.candidates.index(domain.actions[a][1]) for a in actions])
                return .05 * candidates, np.where(candidates == 7, 16., 0.)
        probability = np.full(len(domain.actions), .5)
        for context in domain.contexts:
            probability[domain.action(context, domain.candidates[0])] = .001
        models = {m: Model() for m in small.SPECTRAL}
        with patch('src.tracking_six_speed.feasibility_probability', return_value=probability), \
                patch('src.tracking_six_speed_small_bonus.feasibility_probability', return_value=probability), \
                patch('src.tracking_peer_cost.peer_proposal', return_value=(None, dict.fromkeys(reference.peer.PEER_FIELDS))):
            for context in domain.contexts:
                with patch('src.tracking_six_speed.program', return_value=[context] * 12):
                    plans = small.plan_round(deepcopy(prefix), domain, cfg, clients, 7100, models, phi)
                for plan in plans:
                    if plan['method'] in (small.D0, small.E0):
                        self.assertEqual(plan['candidate'], domain.candidates[1])
                    elif plan['method'] in (small.D_SMALL, small.E_SMALL):
                        self.assertEqual(plan['candidate'], domain.candidates[7])
                        self.assertAlmostEqual(plan['acquisition_score'], .35 - .5 * 4 - np.log(.5))

    def test_restart_and_replay_preserve_barriers_peer_rules_and_portable_fits(self):
        cfg, domain, clients, phi, oracle = design(failure=True)
        full = small.run_fleet(domain, cfg, clients, 7100, oracle, phi)
        prefix = small.run_fleet(domain, cfg, clients, 7100, oracle, phi, stop_round=4)
        resumed = small.run_fleet(domain, cfg, clients, 7100, oracle, phi, state=prefix)
        reference.assert_equivalent(full['methods'], resumed['methods'])
        with patch('src.tracking_budget.minimize', side_effect=AssertionError('No refits')), \
                patch('src.tracking_six_speed.minimize', side_effect=AssertionError('No recovery refits')), \
                patch.object(runner.base, 'resolved_rollout', side_effect=AssertionError('No rollouts')):
            small.validate_state(full, domain, cfg, clients, 7100, phi, complete=True, replay=True)
        broken = deepcopy(full)
        event = next(e for e in broken['methods'][small.D_SMALL]['events'] if e['mode'] == 'peer_gain_verification')
        event['peer_donor_query'] += 1
        with self.assertRaisesRegex(RuntimeError, 'Peer priority'):
            small.validate_state(broken, domain, cfg, clients, 7100, phi)
        broken = deepcopy(full)
        event = next(e for e in broken['methods'][small.D_SMALL]['events'] if e['mode'] == 'feasibility_penalized_LCB')
        event['acquisition_score'] += 1
        with self.assertRaisesRegex(RuntimeError, 'Replay numeric'):
            small.validate_state(broken, domain, cfg, clients, 7100, phi, replay=True)

    def test_model_reuse_does_not_leak_another_arms_observations(self):
        cfg, domain, clients, phi, oracle = design()
        state = small.run_fleet(domain, cfg, clients, 7100, oracle, phi, stop_round=1)
        models = small.fit_models(state, phi, cfg)
        self.assertIs(models[small.D0], models[small.D_SMALL])
        self.assertIs(models[small.E0], models[small.E_SMALL])
        original = models[small.D0].predict(0, [1, 2])
        state['methods'][small.D_SMALL]['histories'][0][0]['J'] *= 10
        changed = small.fit_models(state, phi, cfg)
        self.assertIsNot(changed[small.D0], changed[small.D_SMALL])
        np.testing.assert_array_equal(changed[small.D0].predict(0, [1, 2]), original)
        self.assertFalse(np.allclose(changed[small.D_SMALL].predict(0, [1, 2]), original))

    def test_quality_gains_cannot_hide_extra_failures_or_lost_eligible_policies(self):
        summary = pd.DataFrame([dict(method=m, fleet_budget=216, seeds=3,
            failed_paid_requests=10 if m == small.D_SMALL else 5,
            complete_eligible_on_encountered=17 if m == small.D_SMALL else 18) for m in small.METHODS])
        paired = pd.DataFrame([dict(role=r, fleet_budget=216, success_delta_pp=5.) for _, _, r in CONTRASTS])
        frequency = pd.DataFrame([dict(method=m, fleet_budget=216, frequency=.3,
            context_success_percent=30. if m == small.D_SMALL else 0.) for m in small.METHODS])
        discoveries = pd.DataFrame([dict(method=small.D_SMALL, frequency=.3, seed=s,
            near_reference_requests=1) for s in (7100, 7101, 7102)])
        readout = pilot_readout(summary, paired, dict(frequency_outcomes=frequency,
            frequency_discoveries=discoveries), 216)
        self.assertTrue(readout['criteria']['middle_frequency_success_above_zero_D'])
        self.assertTrue(readout['criteria']['primary_success_above_A'])
        self.assertFalse(readout['criteria']['failed_paid_requests_not_above_zero_D'])
        self.assertFalse(readout['criteria']['complete_eligible_policies_not_below_zero_D'])
        self.assertFalse(readout['all_descriptive_criteria_met'])
        self.assertFalse(readout['confirmation_claim'])
        self.assertFalse(readout['clean_improvement_over_zero_D'])
        missing_seed = pilot_readout(summary, paired, dict(frequency_outcomes=frequency,
            frequency_discoveries=discoveries.iloc[:2]), 216)
        self.assertFalse(missing_seed['criteria']['middle_frequency_discovery_in_every_seed'])

    def test_reports_keep_all_failures_contrasts_and_frequency_specific_denominators(self):
        cfg, domain, clients, phi, oracle = design(failure=True)
        data = small.run_fleet(domain, cfg, clients, 7100, oracle, phi)
        refs = {c: dict(candidate=domain.candidates[0], eligible=True, J=1.) for c in domain.contexts}
        validation = {c: {g: dict(eligible=True, J=1.) for g in domain.candidates} for c in domain.contexts}
        frames = evaluate([dict(seed=7100, clients=clients, state=data)], domain, cfg, refs, validation)
        trials, cells, visits, learning = frames
        visits['confidence_multiplier'] = visits.method.map(small.BONUSES)
        paired, _ = paired_comparisons(trials)
        self.assertEqual(set(paired.role), {r for _, _, r in CONTRASTS})
        first = paired[paired.fleet_budget == cfg['fleet_budgets'][0]]
        self.assertEqual(list(zip(first.method, first.baseline, first.role)), CONTRASTS)
        self.assertTrue((trials.encountered_contexts + trials.unvisited_contexts == 18).all())
        self.assertEqual(len(visits), 216)
        bad = visits[(visits.client == clients[0]['id']) & visits.is_factory]
        self.assertEqual(len(bad), 6)
        self.assertFalse(bad.requested_success.any())
        with tempfile.TemporaryDirectory() as directory:
            state = dict(out=Path(directory), cfg=cfg, signature='test', specification=dict(mode='smoke'))
            saved = write_report(state, *frames, refs)
            frequency = saved['frequency_outcomes']
            final = frequency[frequency.fleet_budget == 36]
            self.assertEqual(len(final), 18)
            self.assertEqual(int(final.encountered_contexts.sum()), len(cells[cells.fleet_budget == 36]))
            html = (Path(directory) / 'report.html').read_text()
            self.assertTrue(all(label in html for label in LABELS.values()))
            self.assertIn('software verification only', html)
            self.assertTrue((Path(directory) / 'pilot_readout.json').exists())
            self.assertTrue((Path(directory) / 'consolidation_inputs.md').exists())

    def test_fresh_pinned_setup_isolated_payload_and_smoke_collection_guard(self):
        state, previous = runner.setup(), runner.previous.setup()
        np.testing.assert_array_equal(state['phi'], previous['phi'])
        np.testing.assert_array_equal(state['domain'].X, previous['domain'].X)
        self.assertFalse({j['seed'] for j in state['jobs']} & {j['seed'] for j in previous['jobs']})
        self.assertEqual(len(state['clients']) * len(state['jobs']) * 24 * len(small.METHODS), 3888)
        payload = runner.base.payload(dict(state, validation='private', old_histories='private'))
        self.assertFalse({'validation', 'old_histories', 'specification'} & payload.keys())
        self.assertEqual(state['specification']['historical_training_records'], 0)
        self.assertEqual(state['requested']['small_bonus_multiplier'], .5)
        with self.assertRaisesRegex(RuntimeError, 'Smoke'):
            runner.collect(dict(specification=dict(mode='smoke')), 1)


if __name__ == '__main__':
    unittest.main()
