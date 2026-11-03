"""Audit regressions; runnable with unittest, without additional dependencies."""
import importlib.util
import itertools
from pathlib import Path
import tempfile
import unittest

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/audit_review_implementation.py'
spec = importlib.util.spec_from_file_location('review_audit', SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ReviewAuditTests(unittest.TestCase):
    def test_real_constraint_function_boundaries(self):
        rows, info = audit.steering_checks({'Ts': .05}, {'u_rate_max': .2618})
        self.assertTrue(all(r['passed'] for r in rows))
        self.assertAlmostEqual(info['equivalent_rate_deg_s'], 300., places=2)

    def test_actual_selection_distinguishes_denominators(self):
        constants, program, _ = audit.selection_source()
        rows, cap = audit.selection_checks(constants, program)
        self.assertEqual(cap, 13)
        self.assertTrue(all(r['passed'] for r in rows))
        sparse = next(r for r in rows if r['n_feasible'] == 64)
        self.assertEqual(sparse['actual_count'], 13)
        self.assertEqual(sparse['feasible_fraction_rule_count'], 7)

    def test_archive_integrity_and_missing_archive(self):
        cfg = {'seeds': [11, 22, 33, 44, 55]}
        constants = {'N_STUDY_SEEDS': 5}
        fc = {'clients_per_regime': 10}
        rows = [dict(seed=s, client_i=f'c{i}', client_j=f'c{j}',
                     overlap_10pct=.75, rho_J=.99)
                for s in cfg['seeds'] for i, j in itertools.combinations(range(30), 2)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'pairs.csv'
            result, _ = audit.archive_checks(path, cfg, constants, fc)
            self.assertEqual(result['status'], 'NOT_AVAILABLE')
            pd.DataFrame(rows).to_csv(path, index=False)
            result, per_fleet = audit.archive_checks(path, cfg, constants, fc)
            self.assertEqual(result['status'], 'PASS')
            self.assertEqual(result['n_pairs'], 2175)
            self.assertEqual(len(per_fleet), 5)
            rows[1] = rows[0].copy()  # same row count, but duplicate unordered pair
            pd.DataFrame(rows).to_csv(path, index=False)
            result, _ = audit.archive_checks(path, cfg, constants, fc)
            self.assertEqual(result['status'], 'FAIL')

    def test_malformed_archive_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'pairs.csv'
            path.write_text('seed,unrelated\n1,2\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Missing archive columns'):
                audit.archive_checks(path, {}, {}, {})


if __name__ == '__main__':
    unittest.main()
