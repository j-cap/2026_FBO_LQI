"""
Fleet-level federated BO orchestration (Next-steps item 4 / Phase 1D "6."): the same
weighted-GP-EI loop run four ways, differing only in what weight each pooled
cross-client observation gets when fit into a given client's surrogate - a clean,
two-step ablation rather than four unrelated implementations:

  - independent:      w_{i<-j} = 1 if j==i else 0            (no sharing at all)
  - global:            w_{i<-j} = 1                            (share everything, unweighted)
  - similarity:        w_{i<-j} = s_ij                         (symmetric dynamics prior only)
  - recipient_aware:   w_{i<-j}(xi) = s_ij * g_i(xi)            (+ directional recipient-side gate)

All four use the SAME per-client sequential BO loop (`src.optimization.gp_surrogate`)
and the SAME true-plant evaluation function (`src.candidate_eval.evaluate`, via a
cache) - only the training-set weighting passed to the GP differs. "Communication
rounds" are synchronous: every client's proposal in round r is computed from the pool
as it stood at the END of round r-1 (a frozen snapshot), so no client sees another
client's round-r point before proposing its own - avoiding an arbitrary
processing-order advantage.

Calibration episodes drive optimization decisions (acquisition + incumbent tracking);
held-out test episodes are used ONLY to report the convergence curve, never to select a
candidate (doc §5.2's calibration/test separation, correction-plan item 2). The SAME
`EpisodeBank` (fixed calibration + fixed held-out test seeds) is built once per
(fleet seed, client) by the caller and passed in unchanged for every round and every
method - the held-out convergence curve is therefore a PAIRED comparison across
methods (same trajectories/noise), even though the candidate xi evaluated on that fixed
bank differs round to round.

Objective normalization (post-Phase-2-smoke correction, see
docs/phase1_diagnostic_findings.md Part 6): every true-plant evaluation is normalized by
that SAME client's own pre-optimization baseline-controller performance,
J_tilde_i(xi) = J_i(xi) / J_i^base (`compute_baseline_reference` below evaluates
`hand_tuned_reference_xi` once per client - a reference every client can actually
obtain without touching an oracle or another client's data). GPs are fit on J_tilde,
never raw RMSE, both so pooled cross-client observations are on a comparable scale
(client j's "1.2x its own baseline" means the same thing to client i as client i's own
"1.2x its own baseline" - very different RAW rmse could mean the same relative quality)
and so the infeasibility penalty no longer needs the oracle (which would never be
available at deployment time): infeasible candidates are simply assigned
J_tilde_infeas=INFEASIBLE_PENALTY_NORM. The oracle (`compute_oracle_reference`) is
still computed, but used ONLY post-hoc/as a monitoring metric (evals-to-5%-of-oracle,
and the driver script's normalized-regret analysis) - never inside the training set,
the acquisition function, or the infeasibility penalty.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import qmc

from src.episodes import EpisodeBank, calibration_episodes, test_episodes
from src.ifac_bridge import Client
from src.optimization.gate import recipient_gate
from src.optimization.gp_surrogate import propose_next
from src.recipient_screen import predict_recipient_margins

METHODS = ["independent", "global", "similarity", "recipient_aware"]

# Diagnostic-triggered adaptive sharing (Part 11/12, docs/phase2_diagnostic_findings.md,
# 2026-09-04): a 5th weighting rule, deliberately kept OUT of `METHODS` (decision.py's
# `decide_fbo_verdict` and the main driver's per-method loop are written specifically
# around the original 4-method ablation's roles - "recipient_aware", in particular, is
# referenced by name). `run_fbo_method(method=TRIGGERED_SIMILARITY, ...)` is a separate,
# explicitly-opted-into code path; a dedicated driver (`scripts/phase2_triggered_dev.py`)
# calls it directly rather than going through the generic METHODS loop.
TRIGGERED_SIMILARITY = "triggered_similarity"

# Fixed, dimensionless penalty for an infeasible candidate in NORMALIZED (J/J_base)
# units - replaces the earlier oracle-derived penalty (3 * oracle_rmse), which leaked
# information no client could actually have at deployment time.
INFEASIBLE_PENALTY_NORM = 3.0

_uid_counter = count()


@dataclass
class EvalRecord:
    uid: int
    client_id: str
    client_idx: int
    xi: np.ndarray
    round_idx: int
    source: str  # "sobol_init" | "bo"
    feasible: bool
    calibration_rmse: Optional[float]  # raw (physical units), None if nominally infeasible
    training_y: float  # NORMALIZED J_tilde = calibration_rmse / J_base (or INFEASIBLE_PENALTY_NORM)


@dataclass
class ClientTrace:
    client_id: str
    n_local_evals: List[int] = field(default_factory=list)
    best_calibration_rmse: List[float] = field(default_factory=list)
    best_held_out_rmse: List[float] = field(default_factory=list)
    best_xi: List[np.ndarray] = field(default_factory=list)
    n_infeasible: int = 0
    # Diagnostic only, NOT a decision metric (a BO algorithm intentionally explores
    # points worse than its incumbent - this alone isn't proof of harmful transfer):
    # count of feasible proposals from a pooled-data method that landed worse than the
    # client's own pre-round-best by more than `negative_transfer_threshold`.
    n_degrading_proposals: int = 0
    baseline_rmse: Optional[float] = None
    oracle_rmse: Optional[float] = None
    evals_to_5pct_of_oracle: Optional[int] = None


@dataclass
class FBOComparisonResult:
    method: str
    client_traces: Dict[str, ClientTrace]
    all_records: List[EvalRecord]


def _normalize(xi: np.ndarray, xi_lower: np.ndarray, xi_upper: np.ndarray) -> np.ndarray:
    return (xi - xi_lower) / (xi_upper - xi_lower)


def _denormalize(xi_norm: np.ndarray, xi_lower: np.ndarray, xi_upper: np.ndarray) -> np.ndarray:
    return xi_lower + xi_norm * (xi_upper - xi_lower)


def _true_eval_to_record(
    ev, client_id: str, client_idx: int, xi: np.ndarray, round_idx: int, source: str, J_base: float
) -> EvalRecord:
    feasible = bool(ev.feasible and ev.tracking_rmse is not None)
    raw_rmse = float(ev.tracking_rmse) if ev.tracking_rmse is not None else None
    training_y = (raw_rmse / J_base) if feasible else INFEASIBLE_PENALTY_NORM
    return EvalRecord(
        uid=next(_uid_counter), client_id=client_id, client_idx=client_idx, xi=np.asarray(xi, float),
        round_idx=round_idx, source=source, feasible=feasible, calibration_rmse=raw_rmse, training_y=training_y,
    )


def _weight_for_record(
    method: str, recipient_idx: int, record: EvalRecord, *, similarity_S: np.ndarray,
    recipient_client: Client, beta_max: float, bank_recipient: EpisodeBank,
    state_scale: Sequence[float], fixed_R: float, gate_cache: Dict[tuple, float],
    post_trigger: bool = False, recipient_activated: bool = False,
) -> float:
    if record.client_idx == recipient_idx:
        return 1.0
    if method == "global":
        return 1.0
    if method == "independent":
        return 0.0
    s_ij = float(similarity_S[recipient_idx, record.client_idx])
    if method == "similarity":
        return s_ij
    if method == TRIGGERED_SIMILARITY:
        # Pre-trigger (n_local <= trigger_cutoff): always independent, regardless of
        # `recipient_activated` - the trigger decision only takes effect from the round
        # immediately after the cutoff (Part 12's two-stage design). Post-trigger: s_ij
        # if this recipient was activated, else stays independent (a_i=0 clients never
        # collaborate - the whole point of the "who needs it" gate).
        if not post_trigger or not recipient_activated:
            return 0.0
        return s_ij
    # recipient_aware
    key = (recipient_idx, record.uid)
    if key not in gate_cache:
        margins = predict_recipient_margins(
            recipient_client, record.xi, bank_recipient.t, bank_recipient.r_ref,
            state_scale=state_scale, fixed_R=fixed_R,
        )
        gate_cache[key] = recipient_gate(margins, beta_max=beta_max)
    return s_ij * gate_cache[key]


def compute_oracle_reference(
    fleet: List[Client],
    banks: Dict[str, EpisodeBank],
    cache_evaluate: Callable,
    *,
    xi_lower: Sequence[float],
    xi_upper: Sequence[float],
    state_scale: Sequence[float],
    fixed_R: float,
    eval_kwargs: dict,
    n_oracle_sobol: int,
    seed: int = 0,
    feasibility_out: Optional[Dict[str, dict]] = None,
) -> Dict[str, float]:
    """Per-client best feasible calibration RMSE over an independent Sobol reference
    set (NOT reused as BO training data) - the "oracle" denominator for the
    evals-to-5%-of-oracle secondary metric. Uses the same calibration episodes/cache as
    the BO loop, so a Sobol point that happens to coincide with one already evaluated
    during BO is a cheap cache hit, not a leak (calibration performance is calibration
    performance regardless of which scan proposed the point).

    `feasibility_out`, if given, is populated in place with
    `{client_id: {"n_feasible": int, "n_sobol": int, "phi": float}}` - a coarse
    feasible-support estimate (Step D, docs/phase2_diagnostic_findings.md, 2026-09-02)
    reusing this SAME Sobol scan rather than a new one. `phi` = n_feasible/n_sobol is
    only as fine-grained as `n_oracle_sobol` allows (32 in the existing campaigns - a
    coarse diagnostic estimate, not a high-quality feasible-volume estimate)."""
    xi_lower_a, xi_upper_a = np.asarray(xi_lower, float), np.asarray(xi_upper, float)
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed + 97)
    Xi = qmc.scale(sampler.random(n_oracle_sobol), xi_lower_a, xi_upper_a)

    oracle_rmse: Dict[str, float] = {}
    for client in fleet:
        bank = banks[client.client_id]
        best = float("inf")
        n_feasible = 0
        for xi in Xi:
            ev = cache_evaluate(
                client, xi, bank, state_scale=state_scale, fixed_R=fixed_R,
                episodes=calibration_episodes(bank), **eval_kwargs,
            )
            if ev.feasible and ev.tracking_rmse is not None:
                best = min(best, float(ev.tracking_rmse))
                n_feasible += 1
        oracle_rmse[client.client_id] = best if np.isfinite(best) else float("nan")
        if feasibility_out is not None:
            feasibility_out[client.client_id] = {
                "n_feasible": n_feasible, "n_sobol": len(Xi), "phi": n_feasible / len(Xi),
            }
    return oracle_rmse


def compute_baseline_reference(
    fleet: List[Client],
    banks: Dict[str, EpisodeBank],
    cache_evaluate: Callable,
    *,
    baseline_xi: Sequence[float],
    state_scale: Sequence[float],
    fixed_R: float,
    eval_kwargs: dict,
    feasibility_out: Optional[Dict[str, bool]] = None,
) -> Dict[str, float]:
    """Per-client baseline-controller calibration RMSE (`J_i^base`) - the SAME
    `baseline_xi` (the hand-tuned reference point, `controller.yaml`'s
    `hand_tuned_reference_xi`) evaluated on each client's own true plant. Unlike the
    oracle, this is something every client can actually compute before optimization
    starts, with no cross-client or oracle information - the legitimate normalization
    reference `run_fbo_method` divides every observation by. Falls back to 1.0 (i.e.
    un-normalized) for a client on which the baseline itself is infeasible, which
    should not occur for a sane `hand_tuned_reference_xi` but is handled rather than
    left to crash or silently divide by None.

    KNOWN CONFOUND when this fallback fires (Step D/C,
    docs/phase2_diagnostic_findings.md, 2026-09-02): a fallback client's training_y
    becomes `raw_rmse / 1.0` (~0.01-0.02), while a normal client's is `raw_rmse /
    J_base` (~0.5-5) - wildly different scales feeding the SAME pooled GP for
    `similarity`/`global`/`recipient_aware`, sometimes at high `s_ij` weight between
    same-family clients. `feasibility_out`, if given, is populated in place with
    `{client_id: bool}` so callers can detect/count this rather than only see it in a
    printed warning - Step C's planned fix is to reject/resample any client for whom
    this is False at fleet-generation time, making the fallback an anomaly again."""
    baseline_rmse: Dict[str, float] = {}
    for client in fleet:
        bank = banks[client.client_id]
        ev = cache_evaluate(
            client, np.asarray(baseline_xi, float), bank, state_scale=state_scale, fixed_R=fixed_R,
            episodes=calibration_episodes(bank), **eval_kwargs,
        )
        feasible = bool(ev.feasible and ev.tracking_rmse is not None and ev.tracking_rmse > 0)
        if feasible:
            baseline_rmse[client.client_id] = float(ev.tracking_rmse)
        else:
            print(
                f"[fleet_bo] WARNING: baseline_xi infeasible for client {client.client_id!r} - "
                "falling back to J_base=1.0 (unnormalized) for this client."
            )
            baseline_rmse[client.client_id] = 1.0
        if feasibility_out is not None:
            feasibility_out[client.client_id] = feasible
    return baseline_rmse


def run_fbo_method(
    fleet: List[Client],
    banks: Dict[str, EpisodeBank],
    method: str,
    *,
    cache_evaluate: Callable,
    xi_lower: Sequence[float],
    xi_upper: Sequence[float],
    state_scale: Sequence[float],
    fixed_R: float,
    beta_max: float,
    eval_kwargs: dict,
    n_init: int,
    n_iters: int,
    n_candidates: int,
    similarity_S: np.ndarray,
    baseline_rmse: Dict[str, float],
    oracle_rmse: Dict[str, float],
    negative_transfer_threshold: float = 0.10,
    seed: int = 0,
    trigger_cutoff: Optional[int] = None,
    activated: Optional[Dict[str, bool]] = None,
) -> FBOComparisonResult:
    """`baseline_rmse` (from `compute_baseline_reference`) is the normalization
    reference EVERY observation is divided by before it ever reaches a GP or an
    infeasibility penalty. `oracle_rmse` (from `compute_oracle_reference`) is used
    ONLY to populate `ClientTrace.oracle_rmse` / `evals_to_5pct_of_oracle` - a
    monitoring metric, never fed back into acquisition, training, or the penalty.

    `trigger_cutoff`/`activated` are REQUIRED when `method=TRIGGERED_SIMILARITY`
    (Part 12) and ignored otherwise. `activated` is a precomputed `{client_id: bool}`
    decision (the caller's responsibility, NOT computed inside this function) - keeps
    "how do we decide who's triggered" (stagnation-based, using an already-completed
    `independent` run's own trajectory, or a random-count-matched control) decoupled
    from "how does the BO loop behave once activation is known", which is all this
    function does. The decision, once given, is STICKY for the whole run (no
    re-evaluation at any later round) - matching the frozen two-stage design."""
    if method not in METHODS and method != TRIGGERED_SIMILARITY:
        raise ValueError(f"Unknown method {method!r}, expected one of {METHODS + [TRIGGERED_SIMILARITY]}")
    if method == TRIGGERED_SIMILARITY and (trigger_cutoff is None or activated is None):
        raise ValueError(f"method={TRIGGERED_SIMILARITY!r} requires both trigger_cutoff and activated")

    xi_lower_a = np.asarray(xi_lower, float)
    xi_upper_a = np.asarray(xi_upper, float)
    n = len(fleet)
    client_idx = {c.client_id: i for i, c in enumerate(fleet)}

    rng = np.random.default_rng(seed)
    init_sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    Xi_init_norm = init_sampler.random(n_init)
    cand_sampler = qmc.Sobol(d=3, scramble=True, seed=seed + 1)
    Xi_cand_norm = cand_sampler.random(n_candidates)

    all_evals: List[EvalRecord] = []
    own_evals: Dict[str, List[EvalRecord]] = {c.client_id: [] for c in fleet}
    traces: Dict[str, ClientTrace] = {
        c.client_id: ClientTrace(
            client_id=c.client_id, baseline_rmse=baseline_rmse.get(c.client_id), oracle_rmse=oracle_rmse.get(c.client_id)
        )
        for c in fleet
    }
    gate_cache: Dict[tuple, float] = {}

    def _record_round_summary(client_id: str, n_local: int, round_idx: int, client: Client, bank: EpisodeBank):
        trace = traces[client_id]
        feasible_own = [r for r in own_evals[client_id] if r.feasible]
        if not feasible_own:
            return
        best_rec = min(feasible_own, key=lambda r: r.calibration_rmse)
        ev_test = cache_evaluate(
            client, best_rec.xi, bank, state_scale=state_scale, fixed_R=fixed_R,
            episodes=test_episodes(bank), **eval_kwargs,
        )
        held_out_rmse = float(ev_test.tracking_rmse) if ev_test.tracking_rmse is not None else float("nan")
        trace.n_local_evals.append(n_local)
        trace.best_calibration_rmse.append(best_rec.calibration_rmse)
        trace.best_held_out_rmse.append(held_out_rmse)
        trace.best_xi.append(best_rec.xi)
        ref = trace.oracle_rmse
        if (
            trace.evals_to_5pct_of_oracle is None
            and ref is not None
            and np.isfinite(ref)
            and ref > 0
            and best_rec.calibration_rmse <= 1.05 * ref
        ):
            trace.evals_to_5pct_of_oracle = n_local

    # ---- round 0: common Sobol init, evaluated per-client on its own true plant -------
    for client in fleet:
        cid = client.client_id
        bank = banks[cid]
        for k in range(n_init):
            xi = _denormalize(Xi_init_norm[k], xi_lower_a, xi_upper_a)
            ev = cache_evaluate(
                client, xi, bank, state_scale=state_scale, fixed_R=fixed_R,
                episodes=calibration_episodes(bank), **eval_kwargs,
            )
            rec = _true_eval_to_record(ev, cid, client_idx[cid], xi, 0, "sobol_init", baseline_rmse[cid])
            if not rec.feasible:
                traces[cid].n_infeasible += 1
            own_evals[cid].append(rec)
            all_evals.append(rec)
        _record_round_summary(cid, n_init, 0, client, bank)

    # ---- rounds 1..n_iters: synchronous BO proposals ----------------------------------
    for r in range(1, n_iters + 1):
        # Trigger state as of the START of this round (Part 12): round r's proposal is
        # computed from the pool as it stood at the end of round r-1, i.e. after
        # n_init+r-1 local evaluations - post-trigger weighting applies from the first
        # round where that count has already reached trigger_cutoff.
        post_trigger = trigger_cutoff is not None and (n_init + r - 1) >= trigger_cutoff
        pool_snapshot = list(all_evals)
        new_records = []
        for client in fleet:
            cid = client.client_id
            i = client_idx[cid]
            bank = banks[cid]
            recipient_activated = bool(activated.get(cid, False)) if activated is not None else False

            weights = np.array(
                [
                    _weight_for_record(
                        method, i, rec, similarity_S=similarity_S, recipient_client=client, beta_max=beta_max,
                        bank_recipient=bank, state_scale=state_scale, fixed_R=fixed_R, gate_cache=gate_cache,
                        post_trigger=post_trigger, recipient_activated=recipient_activated,
                    )
                    for rec in pool_snapshot
                ]
            )
            X_train_norm = np.array([_normalize(rec.xi, xi_lower_a, xi_upper_a) for rec in pool_snapshot])
            y_train = np.array([rec.training_y for rec in pool_snapshot])

            own_feasible = [rec.training_y for rec in own_evals[cid] if rec.feasible]
            best_f = min(own_feasible) if own_feasible else float(np.min(y_train)) if len(y_train) else 1.0

            cand_idx, _ = propose_next(
                X_train_norm, y_train, weights, Xi_cand_norm, best_f, seed=seed + r, rng=rng
            )
            xi_proposed = _denormalize(Xi_cand_norm[cand_idx], xi_lower_a, xi_upper_a)

            ev = cache_evaluate(
                client, xi_proposed, bank, state_scale=state_scale, fixed_R=fixed_R,
                episodes=calibration_episodes(bank), **eval_kwargs,
            )
            rec = _true_eval_to_record(ev, cid, i, xi_proposed, r, "bo", baseline_rmse[cid])
            pooling_active = method not in ("independent", TRIGGERED_SIMILARITY) or (
                method == TRIGGERED_SIMILARITY and post_trigger and recipient_activated
            )
            if not rec.feasible:
                traces[cid].n_infeasible += 1
            elif pooling_active and best_f < float("inf"):
                if rec.training_y > (1.0 + negative_transfer_threshold) * best_f:
                    traces[cid].n_degrading_proposals += 1
            new_records.append(rec)
            own_evals[cid].append(rec)

        all_evals.extend(new_records)
        for client in fleet:
            cid = client.client_id
            _record_round_summary(cid, n_init + r, r, client, banks[cid])

    return FBOComparisonResult(method=method, client_traces=traces, all_records=all_evals)


# Frozen mature-fleet leave-one-out warm start (Part 14, docs/phase2_diagnostic_findings.md,
# 2026-09-07) - the answer to "how many new local experiments does a NEW client need,
# given what the fleet already knows", not "when should an already-participating client
# start sharing" (Part 12's trigger). A single LIVE client (the held-out target) runs its
# own fresh BO loop against a FIXED, never-growing pool of another 29 clients' already-
# complete `independent`-method evaluation records - deliberately `independent` histories,
# not `similarity`/`global`, so the frozen "fleet knowledge" isn't itself already
# federated. Peer weighting is by CLIENT IDENTITY (`is_own` bool / peer client_id), not
# by numeric `client_idx` the way `_weight_for_record` does - reusing that function here
# would risk an index collision (the target's own index in a 1-client "fleet" is always
# 0, which could coincide with a frozen peer record's ORIGINAL index from the 30-client
# fleet the histories came from), so this gets its own small weight function instead.
WARMSTART_METHODS = ["independent", "global", "similarity"]


@dataclass
class FrozenPeerRecord:
    """Minimal (xi, training_y) pair from a peer's already-completed `independent`-method
    history - already normalized by THAT peer's own baseline (same convention as
    `EvalRecord.training_y`), so pooling it into the target's GP needs no rescaling."""
    client_id: str
    xi: np.ndarray
    training_y: float


def _weight_frozen_peer(method: str, *, is_own: bool, s_ij: float) -> float:
    if is_own:
        return 1.0
    if method == "independent":
        return 0.0
    if method == "global":
        return 1.0
    if method == "similarity":
        return s_ij
    raise ValueError(f"Unknown warm-start method {method!r}, expected one of {WARMSTART_METHODS}")


def run_target_warmstart(
    target: Client,
    target_bank: EpisodeBank,
    method: str,
    *,
    cache_evaluate: Callable,
    xi_lower: Sequence[float],
    xi_upper: Sequence[float],
    state_scale: Sequence[float],
    fixed_R: float,
    eval_kwargs: dict,
    n_init: int,
    n_iters: int,
    n_candidates: int,
    frozen_peer_records: List[FrozenPeerRecord],
    peer_similarity: Dict[str, float],
    baseline_rmse: float,
    oracle_rmse: float,
    seed: int = 0,
) -> ClientTrace:
    """`frozen_peer_records` never changes across rounds - it's the target's ENTIRE
    peer prior, available at time zero, not counted against the target's own local
    evaluation budget. Only the target's own `n_init` Sobol points + `n_iters` BO
    proposals count as `n_local_evals`. `peer_similarity` maps a peer's `client_id` to
    `s_{target,peer}` (precomputed by the caller - one row of the fleet's similarity
    matrix); irrelevant when `method="independent"`."""
    if method not in WARMSTART_METHODS:
        raise ValueError(f"Unknown warm-start method {method!r}, expected one of {WARMSTART_METHODS}")

    xi_lower_a = np.asarray(xi_lower, float)
    xi_upper_a = np.asarray(xi_upper, float)
    cid = target.client_id

    rng = np.random.default_rng(seed)
    init_sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    Xi_init_norm = init_sampler.random(n_init)
    cand_sampler = qmc.Sobol(d=3, scramble=True, seed=seed + 1)
    Xi_cand_norm = cand_sampler.random(n_candidates)

    peer_weights = np.array(
        [_weight_frozen_peer(method, is_own=False, s_ij=peer_similarity.get(rec.client_id, 0.0)) for rec in frozen_peer_records]
    )
    peer_X_norm = np.array([_normalize(rec.xi, xi_lower_a, xi_upper_a) for rec in frozen_peer_records])
    peer_y = np.array([rec.training_y for rec in frozen_peer_records])

    own_evals: List[EvalRecord] = []
    trace = ClientTrace(client_id=cid, baseline_rmse=baseline_rmse, oracle_rmse=oracle_rmse)

    def _record_round_summary(n_local: int):
        feasible_own = [r for r in own_evals if r.feasible]
        if not feasible_own:
            return
        best_rec = min(feasible_own, key=lambda r: r.calibration_rmse)
        ev_test = cache_evaluate(
            target, best_rec.xi, target_bank, state_scale=state_scale, fixed_R=fixed_R,
            episodes=test_episodes(target_bank), **eval_kwargs,
        )
        held_out_rmse = float(ev_test.tracking_rmse) if ev_test.tracking_rmse is not None else float("nan")
        trace.n_local_evals.append(n_local)
        trace.best_calibration_rmse.append(best_rec.calibration_rmse)
        trace.best_held_out_rmse.append(held_out_rmse)
        trace.best_xi.append(best_rec.xi)
        if (
            trace.evals_to_5pct_of_oracle is None
            and oracle_rmse is not None and np.isfinite(oracle_rmse) and oracle_rmse > 0
            and best_rec.calibration_rmse <= 1.05 * oracle_rmse
        ):
            trace.evals_to_5pct_of_oracle = n_local

    # ---- round 0: target's own fresh Sobol init (peers contribute nothing here - they
    # are a fixed prior, not co-participants) --------------------------------------------
    for k in range(n_init):
        xi = _denormalize(Xi_init_norm[k], xi_lower_a, xi_upper_a)
        ev = cache_evaluate(
            target, xi, target_bank, state_scale=state_scale, fixed_R=fixed_R,
            episodes=calibration_episodes(target_bank), **eval_kwargs,
        )
        rec = _true_eval_to_record(ev, cid, 0, xi, 0, "sobol_init", baseline_rmse)
        if not rec.feasible:
            trace.n_infeasible += 1
        own_evals.append(rec)
    _record_round_summary(n_init)

    # ---- rounds 1..n_iters: target's own sequential BO, pooled against the FROZEN peer
    # prior from round 1 onward (available at time zero, per the design) -----------------
    for r in range(1, n_iters + 1):
        own_X_norm = np.array([_normalize(rec.xi, xi_lower_a, xi_upper_a) for rec in own_evals])
        own_y = np.array([rec.training_y for rec in own_evals])
        own_weights = np.ones(len(own_evals))

        if len(peer_X_norm):
            X_train_norm = np.vstack([peer_X_norm, own_X_norm])
            y_train = np.concatenate([peer_y, own_y])
            weights = np.concatenate([peer_weights, own_weights])
        else:
            X_train_norm, y_train, weights = own_X_norm, own_y, own_weights

        own_feasible = [rec.training_y for rec in own_evals if rec.feasible]
        best_f = min(own_feasible) if own_feasible else float(np.min(y_train)) if len(y_train) else 1.0

        cand_idx, _ = propose_next(X_train_norm, y_train, weights, Xi_cand_norm, best_f, seed=seed + r, rng=rng)
        xi_proposed = _denormalize(Xi_cand_norm[cand_idx], xi_lower_a, xi_upper_a)

        ev = cache_evaluate(
            target, xi_proposed, target_bank, state_scale=state_scale, fixed_R=fixed_R,
            episodes=calibration_episodes(target_bank), **eval_kwargs,
        )
        rec = _true_eval_to_record(ev, cid, 0, xi_proposed, r, "bo", baseline_rmse)
        if not rec.feasible:
            trace.n_infeasible += 1
        elif method != "independent" and best_f < float("inf"):
            if rec.training_y > 1.10 * best_f:
                trace.n_degrading_proposals += 1
        own_evals.append(rec)
        _record_round_summary(n_init + r)

    return trace
