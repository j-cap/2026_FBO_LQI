"""Validation tests for `src.optimization.regret.normalized_simple_regret`."""
from __future__ import annotations

import numpy as np

from src.optimization.regret import normalized_simple_regret


def test_regret_is_zero_at_the_oracle():
    r = normalized_simple_regret(j_best=[0.05], j_oracle=[0.05], j_base=[0.10])
    assert np.isclose(r[0], 0.0)


def test_regret_is_one_at_the_baseline():
    r = normalized_simple_regret(j_best=[0.10], j_oracle=[0.05], j_base=[0.10])
    assert np.isclose(r[0], 1.0)


def test_regret_is_between_zero_and_one_for_intermediate_performance():
    r = normalized_simple_regret(j_best=[0.075], j_oracle=[0.05], j_base=[0.10])
    assert 0.0 < r[0] < 1.0


def test_regret_handles_degenerate_denominator_without_blowing_up():
    r = normalized_simple_regret(j_best=[0.05], j_oracle=[0.05], j_base=[0.05])
    assert np.isfinite(r[0])


def test_regret_is_vectorized():
    r = normalized_simple_regret(j_best=[0.05, 0.10, 0.20], j_oracle=[0.05, 0.05, 0.05], j_base=[0.10, 0.10, 0.10])
    assert r.shape == (3,)
    np.testing.assert_allclose(r, [0.0, 1.0, 3.0])
