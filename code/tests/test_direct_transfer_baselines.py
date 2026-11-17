"""Information separation and failure accounting for the direct-transfer audit."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from direct_transfer_baselines import select_sources, pooled_recommendation, score_fields


def history():
    return pd.DataFrame(dict(client_id=['target', 'a', 'a', 'b'], training_y=[-100., .8, .1, .6],
        feasible=[True, True, False, True], xi_1=[0., 1., 2., 3.], xi_2=[0., 1., 2., 3.], xi_3=[0., 1., 2., 3.]))


def test_source_selection_excludes_target_and_infeasible_candidates():
    got = select_sources(history(), 'target')
    assert got.client_id.tolist() == ['a', 'b']
    assert got.training_y.tolist() == [.8, .6]


def test_missing_feasible_source_is_not_silently_removed():
    h = history()
    h.loc[h.client_id == 'b', 'feasible'] = False
    with pytest.raises(ValueError, match='No feasible source'):
        select_sources(h, 'target')


def test_pooled_selection_cannot_see_target_records(monkeypatch):
    import direct_transfer_baselines as mod
    seen = []
    class GP:
        def predict(self, x):
            return x[:, 0]
    def fit(x, y, weights, **kwargs):
        seen.append((x.copy(), y.copy()))
        return type('Fit', (), {'gp': GP()})()
    monkeypatch.setattr(mod, 'fit_weighted_gp', fit)
    h = history()
    a = pooled_recommendation(h, 'target', np.zeros(3), np.ones(3)*4, 7, 8)
    h.loc[h.client_id == 'target', ['training_y', 'xi_1']] = [9999, 9999]
    b = pooled_recommendation(h, 'target', np.zeros(3), np.ones(3)*4, 7, 8)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(seen[0][0], seen[1][0])
    assert len(seen[0][1]) == 3  # infeasible source penalties remain in the pooled GP


def test_failure_does_not_become_a_success_or_disappear():
    assert score_fields(.1, False, 1.)['regret'] == np.inf
    assert not score_fields(None, False, 1.)['success']
    assert score_fields(.99, True, 1.)['success']
