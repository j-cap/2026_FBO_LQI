"""
Recipient-side transfer gate g_i(xi) in [0,1] (Phase 1D "Layer 2"), built on top of
`src.recipient_screen.predict_recipient_margins`. Phase 1D's diagnostic
(docs/phase1_diagnostic_findings.md, Part 5) found `max_abs_beta_hat` - predicted peak
sideslip when a candidate xi is rolled out on the RECIPIENT's own identified model -
the most visibly associated recipient-side margin with true-plant transfer loss
(rho=+0.255, p=1e-53, n=3537): higher predicted sideslip on the recipient's own model
predicts worse real-world transfer. This ties the gate to the SAME physical quantity
(beta) already governing true-plant feasibility via `beta_max_deg` (candidate_eval's
`beta_max` constraint), rather than introducing a new, separately-tuned threshold.
"""
from __future__ import annotations

import numpy as np

from src.recipient_screen import RecipientMargins


def recipient_gate(margins: RecipientMargins, *, beta_max: float) -> float:
    """
    g_i(xi) = 0 if the candidate isn't even nominally stable on client i's own
    identified model (hard cutoff - Layer 2's cheapest, most decisive signal); else
    exp(-(max_abs_beta_hat / beta_max)^2), so a candidate whose predicted sideslip sits
    at the client's own feasibility boundary is already strongly discounted (g~=0.37)
    and one well past it is discounted to ~0, while a candidate well within the
    boundary is barely discounted (g~=1).
    """
    if not margins.nominal_feasible or margins.max_abs_beta_hat is None:
        return 0.0
    ratio = margins.max_abs_beta_hat / max(float(beta_max), 1e-12)
    return float(np.exp(-(ratio**2)))
