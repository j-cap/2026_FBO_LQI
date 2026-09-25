"""Validation tests for `src.similarity` (Phase 1D "Layer 1" symmetric prior)."""
from __future__ import annotations

import numpy as np

from src.similarity import median_distance_lengthscale, similarity_matrix


def test_median_distance_lengthscale_matches_manual_median():
    D2 = np.array([[0.0, 4.0, 16.0], [4.0, 0.0, 1.0], [16.0, 1.0, 0.0]])
    # off-diagonal sqrt distances: sqrt(4)=2, sqrt(16)=4, sqrt(1)=1 -> median=2
    assert median_distance_lengthscale(D2) == 2.0


def test_median_distance_lengthscale_degenerate_single_client():
    D2 = np.zeros((1, 1))
    assert median_distance_lengthscale(D2) == 1.0


def test_similarity_matrix_diagonal_is_one():
    D2 = np.array([[0.0, 4.0], [4.0, 0.0]])
    S = similarity_matrix(D2, lengthscale=1.0)
    assert S[0, 0] == 1.0 and S[1, 1] == 1.0


def test_similarity_matrix_decreases_with_distance():
    D2 = np.array([[0.0, 1.0, 100.0], [1.0, 0.0, 50.0], [100.0, 50.0, 0.0]])
    S = similarity_matrix(D2, lengthscale=2.0)
    assert S[0, 1] > S[0, 2]
    assert np.all((S >= 0.0) & (S <= 1.0))


def test_similarity_matrix_is_symmetric():
    D2 = np.array([[0.0, 4.0, 9.0], [4.0, 0.0, 25.0], [9.0, 25.0, 0.0]])
    S = similarity_matrix(D2, lengthscale=3.0)
    np.testing.assert_allclose(S, S.T)
