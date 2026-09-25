"""Validation tests for `src.optimization.gate.recipient_gate` (Phase 1D "Layer 2")."""
from __future__ import annotations

import numpy as np

from src.optimization.gate import recipient_gate
from src.recipient_screen import RecipientMargins


def test_infeasible_margins_give_zero_gate():
    margins = RecipientMargins(False, None, None, None, None)
    assert recipient_gate(margins, beta_max=0.1) == 0.0


def test_beta_well_within_bound_gives_gate_near_one():
    margins = RecipientMargins(True, 0.5, 0.001, 0.1, 0.1)
    g = recipient_gate(margins, beta_max=0.1)
    assert g > 0.99


def test_beta_at_bound_gives_gate_near_exp_minus_one():
    margins = RecipientMargins(True, 0.5, 0.1, 0.1, 0.1)
    g = recipient_gate(margins, beta_max=0.1)
    assert np.isclose(g, np.exp(-1.0), atol=1e-9)


def test_beta_well_beyond_bound_gives_gate_near_zero():
    margins = RecipientMargins(True, 0.5, 1.0, 0.1, 0.1)
    g = recipient_gate(margins, beta_max=0.1)
    assert g < 1e-3


def test_gate_is_monotonically_decreasing_in_beta():
    betas = [0.01, 0.05, 0.1, 0.2, 0.5]
    gates = [recipient_gate(RecipientMargins(True, 0.5, b, 0.1, 0.1), beta_max=0.1) for b in betas]
    assert all(gates[i] > gates[i + 1] for i in range(len(gates) - 1))
