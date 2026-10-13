"""Unit tests for the opt-in RGPE transfer-BO baseline."""
from __future__ import annotations

import numpy as np

from src.optimization.rgpe import (
    combine_rgpe_marginals,
    compute_ranking_loss,
    fit_rgpe_task_model,
    propose_next_rgpe,
    rank_weights_from_losses,
)


def test_ranking_loss_is_zero_for_correct_order_and_positive_for_reverse():
    y = np.array([0.0, 1.0, 2.0])
    perfect = np.tile(y, (4, 1))
    reverse = np.tile(y[::-1], (4, 1))
    assert np.all(compute_ranking_loss(perfect, y) == 0)
    assert np.all(compute_ranking_loss(reverse, y) > 0)


def test_weight_dilution_prunes_bad_source_and_target_wins_ties():
    source_losses = np.array(
        [
            [0, 0, 0, 0],  # useful source
            [5, 5, 5, 5],  # clearly worse than target -> pruned
        ],
        dtype=float,
    )
    target_losses = np.array([0, 1, 0, 1], dtype=float)
    weights, active = rank_weights_from_losses(
        source_losses,
        target_losses,
        rng=np.random.default_rng(1),
        prune_quantile=0.95,
    )
    assert active.tolist() == [True, False]
    np.testing.assert_allclose(weights, [0.5, 0.0, 0.5])
    assert np.isclose(weights.sum(), 1.0)


def _fit_toy_model(shift: float, seed: int):
    X = np.array(
        [
            [0.05, 0.10, 0.15],
            [0.30, 0.25, 0.20],
            [0.55, 0.60, 0.50],
            [0.85, 0.80, 0.90],
        ]
    )
    y = np.sum((X - shift) ** 2, axis=1)
    return fit_rgpe_task_model(X, y, seed=seed)


def test_ensemble_marginals_have_expected_shape_and_finite_values():
    models = [_fit_toy_model(0.2, 1), _fit_toy_model(0.6, 2)]
    Xq = np.array([[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]])
    mean, std = combine_rgpe_marginals(models, np.array([0.25, 0.75]), Xq)
    assert mean.shape == (2,)
    assert std.shape == (2,)
    assert np.all(np.isfinite(mean))
    assert np.all(std > 0)


def test_rgpe_proposal_is_deterministic_and_returns_valid_candidate():
    source_models = [_fit_toy_model(0.15, 3), _fit_toy_model(0.70, 4)]
    X_target = np.array([[0.10, 0.10, 0.10], [0.60, 0.55, 0.50]])
    y_target = np.array([0.8, 0.3])
    candidates = np.random.default_rng(5).random((32, 3))

    a = propose_next_rgpe(
        source_models,
        X_target,
        y_target,
        candidates,
        num_samples=16,
        seed=17,
    )
    b = propose_next_rgpe(
        source_models,
        X_target,
        y_target,
        candidates,
        num_samples=16,
        seed=17,
    )
    assert 0 <= a.candidate_index < len(candidates)
    assert a.candidate_index == b.candidate_index
    np.testing.assert_allclose(a.weights.weights, b.weights.weights)
    assert np.isclose(a.weights.weights.sum(), 1.0)
