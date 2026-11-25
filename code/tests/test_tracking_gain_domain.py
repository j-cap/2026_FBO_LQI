"""Nested physical domains, offline holdout selection, source reuse and caches."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import tracking_gain_domain as runner
from src.tracking_gain_domain import choose_stages, validation_selection, STAGES
from src.tracking_gain_domain_report import tables, write_report
from src.tracking_confirmation import aggregate_candidates


def row(candidate, J=1., eligible=True):
    return dict(candidate=candidate, J=J if eligible else None, complete=eligible,
        feasible=eligible, integration_resolved=True, failure='' if eligible else 'numeric_or_state_guard')


def synthetic_outcomes(state):
    dense = next(c for c in state['members']['dense_original'] if c not in state['members']['original'])
    outside = next(c['id'] for c in state['pool'] if state['labels'][c['id']]['outside_original'])
    calibration = {c['id']: {p['id']: dict(eligible=True, J=.6 if p['id'] == outside else
        .8 if p['id'] == dense else 1. if p['id'] == 'g000' else 2.) for p in state['pool']}
        for c in state['contexts']}
    choices = choose_stages(calibration, state['members'])
    selection = validation_selection(choices, state['profiles'], state['contexts'])
    buckets = {c: [dict(repeat=rep, rows=[row(k, .9 if k == 'g000' else 1.1, k != outside)
        for k in sorted(ids)]) for rep in range(len(state['validation_seeds']))] for c, ids in selection.items()}
    validation = {c: aggregate_candidates(batches) for c, batches in buckets.items()}
    return calibration, choices, validation, buckets, dense, outside


class GainDomainTests(unittest.TestCase):
    def test_original_pool_preserved_and_real_upper_gain_bounds_extended(self):
        state = runner.setup()
        self.assertEqual(state['pool'][:71], state['source']['pool'])
        self.assertEqual({k: len(v) for k, v in state['members'].items()},
            dict(original=71, dense_original=1033, expanded=2011))
        members = [set(state['members'][stage]) for stage in STAGES]
        self.assertTrue(members[0] < members[1] < members[2])
        points = np.asarray([c['gains'] for c in state['pool']])
        np.testing.assert_allclose(points.max(axis=0), [.15, 3., 3.])
        self.assertTrue(np.any(points[:, 1] > 1.) and np.any(points[:, 2] > 1.))
        for candidate in state['pool']:
            label = state['labels'][candidate['id']]
            outside = bool(np.any(np.asarray(candidate['multipliers']) > 10))
            self.assertEqual(label['outside_original'], outside)
            self.assertEqual(candidate['id'] in members[1], not outside)
            if outside:
                self.assertGreater(max(label['original_normalized_coordinates']), 1.)

    def test_profiles_hold_other_physical_gains_fixed_and_old_face_is_not_extension(self):
        state = runner.setup(smoke=True)
        lookup = {c['id']: c for c in state['pool']}
        for profile in state['profiles']:
            source, new = lookup[profile['candidate']], lookup[profile['candidate_id']]
            j = profile['coordinate']
            for k in range(3):
                if k != j:
                    self.assertEqual(source['gains'][k], new['gains'][k])
                    self.assertEqual(source['multipliers'][k], new['multipliers'][k])
            self.assertEqual(profile['outside_original'], profile['multiplier'] > 10)
            self.assertEqual(profile['candidate_id'] in state['members']['dense_original'], profile['multiplier'] == 10)

    def test_no_kernel_construction_new_calibration_excludes_old_rows_and_holdout_is_fresh(self):
        with patch.object(runner.base, 'basis_bank', side_effect=AssertionError('No full-grid feature basis')), \
                patch('src.tracking_budget.Posterior', side_effect=AssertionError('No BO')):
            state = runner.setup()
        calibration_tasks = runner.tasks(state, 'calibration')
        self.assertEqual(sum(len(t['candidates']) for t in calibration_tasks), 69840)
        self.assertEqual(sum(len(t['candidates']) for t in runner.tasks(state, 'source_check')), 108)
        old = set(state['members']['original'])
        for task in calibration_tasks:
            self.assertFalse(set(task['candidates']) & old)
            self.assertLessEqual(len(task['candidates']), 128)
        old_noise = {b['noise_seed'] for b in state['source_batches'].values()}
        self.assertFalse(old_noise & set(state['validation_seeds']))
        self.assertEqual(len(state['validation_seeds']), 4)
        self.assertNotIn('source_batches', runner.payload(state))
        self.assertNotIn('phi', runner.payload(state))

    def test_calibration_eligibility_and_fresh_validation_never_reselect_winner(self):
        state = runner.setup(smoke=True)
        calibration, choices, validation, buckets, dense, outside = synthetic_outcomes(state)
        selected, comparisons, summary, _ = tables(state, calibration, choices, validation, buckets)
        self.assertTrue((selected[selected.stage == 'expanded'].candidate == outside).all())
        self.assertFalse(selected[selected.stage == 'expanded'].validation_eligible.any())
        extension = comparisons[comparisons.effect == 'extension']
        self.assertFalse(extension.material_improvement.any())
        self.assertTrue(extension.validation_improvement_percent.isna().all())
        self.assertEqual(int(summary[summary.stage == 'expanded'].validation_eligible_winners.iloc[0]), 0)
        for rows in calibration.values():
            rows[outside]['eligible'] = False
        changed = choose_stages(calibration, state['members'])
        self.assertTrue(all(x['expanded'] == dense for x in changed.values()))

    def test_source_parity_requires_identity_classification_and_numeric_agreement(self):
        state = runner.setup(smoke=True)
        with tempfile.TemporaryDirectory() as directory:
            state = dict(state, out=Path(directory))
            tasks = runner.tasks(state, 'source_check')
            for task in tasks:
                old = {r['candidate']: r for r in state['source_batches'][task['context'], task['repeat']]['rows']}
                runner.base.save(runner.job_path(state, task), dict(signature=state['signature'], **task,
                    rows=[deepcopy(old[c]) for c in task['candidates']]))
            self.assertEqual(runner.verify_source_checks(state), 18)
            path = runner.job_path(state, tasks[0])
            original = runner.base.read(path)
            altered = deepcopy(original)
            altered['noise_seed'] += 1
            runner.base.save(path, altered)
            with self.assertRaisesRegex(RuntimeError, 'identity'):
                runner.verify_source_checks(state)
            altered = deepcopy(original)
            altered['rows'][0]['peak_y'] += .01
            runner.base.save(path, altered)
            with self.assertRaisesRegex(RuntimeError, 'numerical outcome'):
                runner.verify_source_checks(state)

    def test_report_and_collection_keep_smoke_separate(self):
        state = runner.setup(smoke=True)
        calibration, choices, validation, buckets, _, _ = synthetic_outcomes(state)
        with tempfile.TemporaryDirectory() as directory:
            state = dict(state, out=Path(directory))
            write_report(state, calibration, choices, validation, buckets)
            html = (Path(directory) / 'report.html').read_text()
            self.assertTrue('software verification only' in html)
            self.assertTrue('Expanded box' in html)
            self.assertEqual(len(list(Path(directory).glob('*.csv'))), 6)
            with self.assertRaisesRegex(RuntimeError, 'Smoke'):
                runner.collect(state)


if __name__ == '__main__':
    unittest.main()
