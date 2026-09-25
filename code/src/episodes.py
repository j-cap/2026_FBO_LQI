"""
Calibration / test episode banks for the oracle landscape study (doc §5.2, §10.2).

The IFAC trajectory generator (`lc_yaw_rate_ref`) is deterministic given a client's
lane-change half-duration `T_lc` - the only per-episode randomness available without
touching that generator is the measurement/process-noise realization consumed by
`closed_loop_step_lqi` (seeded via `np.random.seed` right before each rollout, see
`src.interfaces.plant_client.PlantClient.rollout`). So "3 episode seeds per candidate"
here means 3 independent noise realizations of the SAME reference maneuver, not 3
different maneuvers - documented explicitly since the plan doc's language ("episode
definitions") could otherwise be read as implying trajectory diversity that doesn't
exist yet in the underlying generator.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.ifac_bridge import Client, lc_yaw_rate_ref


@dataclass
class EpisodeSpec:
    seed: int
    role: str  # "calibration" | "test"
    # Which maneuver this episode uses (Part 8 common-task-distribution redesign,
    # docs/phase2_diagnostic_findings.md, 2026-09-03). None for every v1/v2 bank
    # (`build_episode_bank`) - those episodes all share the bank-level `t`/`r_ref`
    # exactly as before. Set only by `build_multi_maneuver_episode_bank`, whose
    # episodes look up their own maneuver in `EpisodeBank.refs` instead.
    T_lc: Optional[float] = None


@dataclass
class EpisodeBank:
    t: np.ndarray
    r_ref: np.ndarray
    episodes: List[EpisodeSpec]
    # Multi-maneuver reference trajectories, keyed by T_lc (Part 8 redesign). None for
    # a v1/v2 bank - `evaluate()` falls back to `t`/`r_ref` above whenever an episode's
    # `T_lc` is None, so this is purely additive and never changes v1/v2 behavior.
    refs: Optional[Dict[float, Tuple[np.ndarray, np.ndarray]]] = None


def deterministic_client_seed(base_seed: int, client_id: str) -> int:
    """Deterministic across processes/reruns - unlike Python's built-in hash(), which
    is randomized per-process (PYTHONHASHSEED) and would break joblib-worker parity."""
    return (base_seed + zlib.crc32(client_id.encode("utf-8"))) % (2**31 - 1)


def build_episode_bank(
    client: Client,
    *,
    Ts: float,
    T_total: float,
    T0: float,
    r_max: float,
    n_calibration: int,
    n_test: int,
    base_seed: int,
    tfilter: Optional[float] = None,
) -> EpisodeBank:
    """One fixed reference maneuver per client (from its own T_lc), n_calibration +
    n_test disjoint noise seeds derived deterministically from `base_seed` and
    `client.client_id`. Calibration and test episodes are kept in disjoint seed pools
    so nothing evaluated during Phase-1/BO calibration ever reappears as a held-out
    test episode (doc §5.2's dataset-separation rule)."""
    t, r_ref = lc_yaw_rate_ref(Ts, T_total, T0, client.T_lc, r_max, tfilter=tfilter)
    client_seed = deterministic_client_seed(base_seed, client.client_id)
    rng = np.random.default_rng(client_seed)
    seeds = rng.integers(0, 2**31 - 1, size=n_calibration + n_test)
    episodes = [EpisodeSpec(seed=int(s), role="calibration") for s in seeds[:n_calibration]] + [
        EpisodeSpec(seed=int(s), role="test") for s in seeds[n_calibration:]
    ]
    return EpisodeBank(t=t, r_ref=r_ref, episodes=episodes)


def build_multi_maneuver_episode_bank(
    client: Client,
    *,
    Ts: float,
    T_total: float,
    T0: float,
    r_max: float,
    lane_change_time: Sequence[float],
    n_calibration_per_maneuver: int,
    n_test_per_maneuver: int,
    base_seed: int,
    tfilter: Optional[float] = None,
) -> EpisodeBank:
    """Common-task-distribution episode bank (Part 8 T_lc redesign,
    docs/phase2_diagnostic_findings.md, 2026-09-03) - EVERY client gets calibration/test
    episodes across ALL `lane_change_time` values, not just its own slot-assigned T_lc
    (`build_episode_bank`/v1's `T_lc = lane_change_time[0 or 1]` split). `evaluate()`
    already averages `tracking_rmse` over whatever `episodes` list it's given, so
    calling it with `calibration_episodes(this_bank)` (spanning every maneuver)
    directly gives f_i(xi) = (1/N_omega) * sum_omega J_i(xi; omega) - no change needed
    in `src.optimization.fleet_bo`, which only ever passes `calibration_episodes(bank)`/
    `test_episodes(bank)` through unchanged.

    Independent of `client.T_lc` entirely (a client's identity here is just its
    client_id, not which half of its family it's in - the mechanism Part 8 found
    confounds `s_ij`, which only ever sees identified dynamics, with the task itself).

    Calibration/test seed pools stay disjoint PER maneuver (so no calibration episode
    from ANY maneuver ever reappears as a held-out test episode for the SAME maneuver),
    and seeds are derived independently per (client_id, T_lc) pair via a distinct
    `deterministic_client_seed` string - adding/removing a maneuver from
    `lane_change_time` never perturbs any other maneuver's seed draw for this client."""
    refs: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    episodes: List[EpisodeSpec] = []
    for T_lc in lane_change_time:
        t, r_ref = lc_yaw_rate_ref(Ts, T_total, T0, T_lc, r_max, tfilter=tfilter)
        refs[T_lc] = (t, r_ref)
        client_seed = deterministic_client_seed(base_seed, f"{client.client_id}_Tlc{T_lc}")
        rng = np.random.default_rng(client_seed)
        seeds = rng.integers(0, 2**31 - 1, size=n_calibration_per_maneuver + n_test_per_maneuver)
        episodes.extend(
            EpisodeSpec(seed=int(s), role="calibration", T_lc=T_lc) for s in seeds[:n_calibration_per_maneuver]
        )
        episodes.extend(
            EpisodeSpec(seed=int(s), role="test", T_lc=T_lc) for s in seeds[n_calibration_per_maneuver:]
        )
    # `t`/`r_ref` kept as the first maneuver's arrays purely so `EpisodeBank`'s
    # required fields are populated - every episode built here carries its own T_lc, so
    # `evaluate()` always resolves via `refs[ep.T_lc]`, never these fallback fields.
    t0, r_ref0 = refs[lane_change_time[0]]
    return EpisodeBank(t=t0, r_ref=r_ref0, episodes=episodes, refs=refs)


def identification_episode_seeds(client_id: str, n_episodes: int, base_seed: int) -> List[int]:
    """n_episodes noise seeds for repeated identification rounds on one client (doc
    §5.2's "identification data" role), derived the same deterministic way as
    `build_episode_bank`'s calibration/test seeds but from a separate `base_seed` so
    the two seed pools never collide."""
    client_seed = deterministic_client_seed(base_seed, client_id)
    rng = np.random.default_rng(client_seed)
    return [int(s) for s in rng.integers(0, 2**31 - 1, size=n_episodes)]


def calibration_episodes(bank: EpisodeBank) -> List[EpisodeSpec]:
    return [e for e in bank.episodes if e.role == "calibration"]


def test_episodes(bank: EpisodeBank) -> List[EpisodeSpec]:
    return [e for e in bank.episodes if e.role == "test"]
