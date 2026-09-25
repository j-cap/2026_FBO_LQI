"""
Disk cache around `candidate_eval.evaluate`, keyed on (client_id, rounded_xi,
episode_seeds, plant_version, eval_kwargs) (doc §27.1), so the oracle scan, later BO
baselines, and safety analysis never re-simulate the same (client, xi, episode) triple.

The heavy `client`/`bank` objects are excluded from the cache key (joblib's `ignore=`)
so hashing stays cheap - correctness of the key still rests on `client_id` +
`episode_seeds` uniquely identifying the client/bank content, which holds for one run
(fixed fleet + fixed episode banks for the run's lifetime).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from joblib import Memory

from src.candidate_eval import EvaluationResult, evaluate
from src.episodes import EpisodeBank, calibration_episodes
from src.ifac_bridge import Client

_XI_ROUND_DECIMALS = 6


def make_cached_evaluate(cache_dir: str, *, plant_version: str = "v1", verbose: int = 0) -> Callable:
    memory = Memory(Path(cache_dir), verbose=verbose)

    @memory.cache(ignore=["client", "bank"])
    def _cached(
        client_id: str,
        rounded_xi: tuple,
        episode_seeds: tuple,
        plant_version: str,
        kwargs: dict,
        client: Client,
        bank: EpisodeBank,
    ) -> EvaluationResult:
        return evaluate(client, np.asarray(rounded_xi), bank, **kwargs)

    def cached_evaluate(client: Client, xi, bank: EpisodeBank, **kwargs) -> EvaluationResult:
        rounded_xi = tuple(np.round(np.asarray(xi, float), _XI_ROUND_DECIMALS).tolist())
        episodes = kwargs.get("episodes") or calibration_episodes(bank)
        episode_seeds = tuple(e.seed for e in episodes)
        return _cached(client.client_id, rounded_xi, episode_seeds, plant_version, kwargs, client, bank)

    return cached_evaluate
