"""
Dense, search/score-separated offline reference J_i^ref for the held-out warm-start
targets - a robustness follow-up to the confirmatory campaign
(`phase2_warmstart_confirmatory_c1cds02`/`...c1cds05`, Part 21,
docs/phase2_diagnostic_findings.md). Replaces the 32-point `compute_oracle_reference`
Sobol scan (`src/optimization/fleet_bo.py`), which (a) is thin for a 3-D box and (b)
scores every candidate on the SAME calibration-episode bank BO itself queries - the two
weaknesses a skeptical reviewer would reasonably raise about a metric defined as
"within 5% of oracle." Per the frozen "freeze the algorithm, strengthen the evaluation
layer only" principle: this script runs no BO, does not touch `fleet_bo.py`'s
acquisition/GP/weighting code, and its output is consumed only by
`scripts/rescore_bo_trajectories.py` for a post-hoc, no-rerun-needed re-evaluation of
the already-completed confirmatory campaign.

Three-stage pipeline per held-out target system, matching the offline-reference design
discussed with the user:
  Stage 1 - dense global search: a Sobol design of `--n-sobol` points in the same
    xi_lower/xi_upper box the BO methods use, scored on the target's own SEARCH bank
    (`calibration_episodes(bank)` - the same episodes BO queries; still fine to reuse,
    since this is a global scan, not BO training data).
  Stage 2 - local refinement: bounded Powell from the top `--n-refine-starts` feasible
    Stage-1 points, still scored on the SEARCH bank only.
  Stage 3 - independent final scoring: the single best Stage-1/2 candidate is evaluated
    on a NEW, independent REFERENCE-TEST bank (`build_multi_maneuver_episode_bank` with
    a disjoint `base_seed`, `n_calibration_per_maneuver=0`) - never touched by Stage
    1/2, so the reported J_i^ref is not contaminated by the same noise realizations used
    to find it.

Two modes:
  --mode convergence: nested-Sobol-size convergence check (256/512/1024/2048/4096 -
    free extra output of ONE dense N=4096 scan via prefix cummin, no extra simulation)
    on a small diagnostic subset (default: target slot "00", all 3 families, first
    config seed only) - meant to justify the `--n-sobol` used in `--mode full`, not to
    be run at full scale. Writes `processed/convergence.csv`.
  --mode full: the complete 3-stage pipeline for every (seed, target_slot, family) in
    the config, at a fixed `--n-sobol` (should be chosen AFTER inspecting the
    convergence-mode output, not before). Writes `processed/dense_reference.csv`.

Usage:
    python scripts/generate_offline_reference.py --config config/runs/phase2_warmstart_confirmatory_c1cds02.yaml --mode convergence
    python scripts/generate_offline_reference.py --config config/runs/phase2_warmstart_confirmatory_c1cds02.yaml --mode full --n-sobol 1024
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.optimize import Bounds, minimize  # noqa: E402
from scipy.stats import qmc  # noqa: E402

from src.caching import make_cached_evaluate  # noqa: E402
from src.episodes import (  # noqa: E402
    build_multi_maneuver_episode_bank,
    calibration_episodes,
    deterministic_client_seed,
    test_episodes,
)
from src.io_utils import deep_update, load_run_config, load_yaml, write_run_manifest  # noqa: E402

from phase2_fbo_comparison import _prepare_seed_context, _resolve  # noqa: E402
from phase2_medium_analysis import _family_of  # noqa: E402
from phase2_warmstart_dev import FAMILIES  # noqa: E402

# Disjoint from every episode_base_seed used anywhere else in this project (5000 for
# calibration/test banks, 1000-offset for identification) - guarantees the Stage-3
# reference-test bank's noise seeds cannot coincide with anything Stage 1/2 or the BO
# campaign itself ever evaluated on.
REFERENCE_TEST_BASE_SEED = 900_000
REFERENCE_SOBOL_BASE_SEED = 800_000

N_SOBOL_CONVERGENCE = 4096
CONVERGENCE_CHECKPOINTS = (256, 512, 1024, 2048, 4096)
DEFAULT_N_SOBOL_FULL = 1024
DEFAULT_N_REFINE_STARTS = 8
DEFAULT_N_REF_TEST_PER_MANEUVER = 20
# A local-refinement candidate that lands outside feasibility gets this multiple of the
# best Stage-1 feasible value - large enough that Powell always prefers any feasible
# point, small enough not to be a numerically degenerate objective value.
INFEASIBLE_PENALTY_MULT = 10.0


def _search_objective(xi, client, bank, search_episodes, cached_evaluate, *, state_scale, fixed_R,
                       eval_kwargs, penalty):
    ev = cached_evaluate(client, xi, bank, state_scale=state_scale, fixed_R=fixed_R,
                          episodes=search_episodes, **eval_kwargs)
    if ev.feasible and ev.tracking_rmse is not None and ev.tracking_rmse > 0:
        return float(ev.tracking_rmse)
    return penalty


def _reference_for_target(
    seed: int, target, target_bank, cached_evaluate, *, n_sobol: int, n_refine_starts: int,
    n_ref_test_per_maneuver: int, xi_lower: np.ndarray, xi_upper: np.ndarray, state_scale, fixed_R,
    eval_kwargs: dict, fleet_cfg: dict, controller_cfg: dict, record_convergence: bool,
) -> tuple[dict, list[dict]]:
    """Runs Stage 1-3 for ONE target system. Returns (reference_row, convergence_rows)
    - `convergence_rows` is empty unless `record_convergence` (cheap either way, since
    Stage 1 is already run at `n_sobol` - convergence checkpoints are just prefix
    cummins of the same scan, no extra simulation)."""
    search_episodes = calibration_episodes(target_bank)

    # ---- Stage 1: dense global Sobol search on the SEARCH bank ------------------------
    sobol_seed = deterministic_client_seed(REFERENCE_SOBOL_BASE_SEED, f"{seed}_{target.client_id}")
    sampler = qmc.Sobol(d=3, scramble=True, seed=sobol_seed)
    Xi = qmc.scale(sampler.random(n_sobol), xi_lower, xi_upper)

    J = np.full(n_sobol, np.nan)
    feasible_mask = np.zeros(n_sobol, dtype=bool)
    for m, xi in enumerate(Xi):
        ev = cached_evaluate(target, xi, target_bank, state_scale=state_scale, fixed_R=fixed_R,
                              episodes=search_episodes, **eval_kwargs)
        if ev.feasible and ev.tracking_rmse is not None and ev.tracking_rmse > 0:
            J[m] = float(ev.tracking_rmse)
            feasible_mask[m] = True

    convergence_rows: list[dict] = []
    if record_convergence:
        running_best = np.minimum.accumulate(np.where(feasible_mask, J, np.inf))
        final_best = running_best[-1]
        for cp in CONVERGENCE_CHECKPOINTS:
            if cp > n_sobol:
                continue
            j_cp = float(running_best[cp - 1])
            convergence_rows.append({
                "seed": seed, "client_id": target.client_id, "family": _family_of(target.client_id),
                "n_sobol_checkpoint": cp, "J_search_best_at_checkpoint": j_cp,
                "J_search_best_final": float(final_best),
                "rel_dev_from_final": (j_cp - final_best) / final_best if np.isfinite(final_best) and final_best > 0 else np.nan,
            })

    feas_idx = np.where(feasible_mask)[0]
    if len(feas_idx) == 0:
        # No feasible Sobol point at all - cannot build a reference for this target.
        return (
            {
                "seed": seed, "client_id": target.client_id, "family": _family_of(target.client_id),
                "n_sobol": n_sobol, "n_feasible_sobol": 0, "phi_search": 0.0,
                "xi_ref_1": np.nan, "xi_ref_2": np.nan, "xi_ref_3": np.nan,
                "J_stage1_only_search": np.nan, "J_search_best": np.nan, "J_ref_test": np.nan,
                "n_refine_starts": n_refine_starts, "reference_feasible": False,
            },
            convergence_rows,
        )

    order = feas_idx[np.argsort(J[feas_idx])]
    stage1_best_idx = order[0]
    stage1_best_xi = Xi[stage1_best_idx]
    stage1_best_J = float(J[stage1_best_idx])

    # ---- Stage 2: bounded local refinement from the top-K feasible Stage-1 points -----
    bounds = Bounds(xi_lower, xi_upper)
    penalty = INFEASIBLE_PENALTY_MULT * stage1_best_J
    best_xi, best_J = stage1_best_xi, stage1_best_J
    starts = order[: min(n_refine_starts, len(order))]
    for idx in starts:
        x0 = Xi[idx]

        def _obj(xi):
            return _search_objective(
                xi, target, target_bank, search_episodes, cached_evaluate,
                state_scale=state_scale, fixed_R=fixed_R, eval_kwargs=eval_kwargs, penalty=penalty,
            )

        res = minimize(_obj, x0, method="Powell", bounds=bounds,
                        options={"xtol": 1e-4, "ftol": 1e-5, "maxiter": 100})
        if res.fun < best_J:
            best_J, best_xi = float(res.fun), np.clip(res.x, xi_lower, xi_upper)

    # ---- Stage 3: independent final scoring on the disjoint reference-test bank -------
    ref_test_bank = build_multi_maneuver_episode_bank(
        target, Ts=fleet_cfg["Ts"], T_total=controller_cfg["T_total"], T0=controller_cfg["T0"],
        r_max=controller_cfg["r_max"], lane_change_time=fleet_cfg["lane_change_time"],
        n_calibration_per_maneuver=0, n_test_per_maneuver=n_ref_test_per_maneuver,
        base_seed=REFERENCE_TEST_BASE_SEED, tfilter=controller_cfg.get("tfilter"),
    )
    ref_test_episodes = test_episodes(ref_test_bank)
    ev_ref = cached_evaluate(target, best_xi, ref_test_bank, state_scale=state_scale, fixed_R=fixed_R,
                              episodes=ref_test_episodes, **eval_kwargs)
    reference_feasible = bool(ev_ref.feasible and ev_ref.tracking_rmse is not None and ev_ref.tracking_rmse > 0)
    J_ref_test = float(ev_ref.tracking_rmse) if reference_feasible else float("nan")

    row = {
        "seed": seed, "client_id": target.client_id, "family": _family_of(target.client_id),
        "n_sobol": n_sobol, "n_feasible_sobol": int(feasible_mask.sum()), "phi_search": float(feasible_mask.mean()),
        "xi_ref_1": float(best_xi[0]), "xi_ref_2": float(best_xi[1]), "xi_ref_3": float(best_xi[2]),
        "J_stage1_only_search": stage1_best_J, "J_search_best": best_J, "J_ref_test": J_ref_test,
        "n_refine_starts": n_refine_starts, "reference_feasible": reference_feasible,
    }
    return row, convergence_rows


def _run_one_seed(
    seed: int, cfg: dict, fleet_cfg: dict, controller_cfg: dict, *, mode: str, n_sobol: int,
    n_refine_starts: int, n_ref_test_per_maneuver: int, xi_lower: np.ndarray, xi_upper: np.ndarray,
    state_scale, fixed_R, eval_kwargs: dict, cache_dir: Path, convergence_slots: set,
) -> tuple[list[dict], list[dict]]:
    ctx = _prepare_seed_context(
        seed, cfg, fleet_cfg, controller_cfg, cache_dir=cache_dir, xi_lower=xi_lower, xi_upper=xi_upper,
        state_scale=state_scale, fixed_R=fixed_R, eval_kwargs=eval_kwargs,
    )
    fleet, banks = ctx["fleet"], ctx["banks"]
    client_idx = {c.client_id: i for i, c in enumerate(fleet)}
    cached_evaluate = make_cached_evaluate(str(ctx["seed_cache_dir"]))

    target_slots = cfg.get("target_slots", ["00"])
    ref_rows, conv_rows = [], []
    for family in FAMILIES:
        for slot in target_slots:
            target_id = f"{family}_{slot}"
            if target_id not in client_idx:
                raise ValueError(f"seed={seed}: expected target client {target_id!r} not found in fleet")
            target = fleet[client_idx[target_id]]
            target_bank = banks[target_id]
            record_conv = (mode == "convergence") and (target_id in convergence_slots)
            if mode == "convergence" and target_id not in convergence_slots:
                continue
            print(f"[offline_ref] seed={seed}: target={target_id!r} mode={mode!r} n_sobol={n_sobol} ...")
            row, cr = _reference_for_target(
                seed, target, target_bank, cached_evaluate, n_sobol=n_sobol, n_refine_starts=n_refine_starts,
                n_ref_test_per_maneuver=n_ref_test_per_maneuver, xi_lower=xi_lower, xi_upper=xi_upper,
                state_scale=state_scale, fixed_R=fixed_R, eval_kwargs=eval_kwargs, fleet_cfg=fleet_cfg,
                controller_cfg=controller_cfg, record_convergence=record_conv,
            )
            row["oracle_rmse_weak_32pt"] = ctx["oracle_rmse"].get(target_id, float("nan"))
            ref_rows.append(row)
            conv_rows.extend(cr)
    print(f"[offline_ref] seed={seed}: done ({len(ref_rows)} targets)")
    return ref_rows, conv_rows


def main(config_path: str, mode: str, n_sobol: int, n_refine_starts: int, n_ref_test_per_maneuver: int,
         seeds_override, n_jobs: int) -> None:
    cfg = load_run_config(config_path)
    fleet_cfg = deep_update(load_yaml(_resolve(cfg["fleet_config"])), cfg.get("fleet_overrides", {}))
    controller_cfg = deep_update(load_yaml(_resolve(cfg["controller_config"])), cfg.get("controller_overrides", {}))

    beta_max = np.deg2rad(controller_cfg["beta_max_deg"])
    xi_lower = np.asarray(controller_cfg["xi_lower"], float)
    xi_upper = np.asarray(controller_cfg["xi_upper"], float)
    state_scale, fixed_R = controller_cfg["state_scale"], controller_cfg["fixed_R"]
    eval_kwargs = dict(
        u_min=controller_cfg["u_min"], u_max=controller_cfg["u_max"], u_rate_max=controller_cfg.get("u_rate_max"),
        beta_max=beta_max, noise_std=controller_cfg["noise_std"], process_noise=controller_cfg["process_noise"],
        plant_mode=controller_cfg.get("plant_mode", "linear"), mu=controller_cfg.get("tire_mu", 1.0),
    )

    reference_tag = cfg.get("offline_reference_tag", "offline_reference")
    out_dir = _ROOT / "results" / reference_tag
    (out_dir / "processed").mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    seeds = seeds_override if seeds_override else cfg["seeds"]
    if mode == "convergence":
        n_sobol = N_SOBOL_CONVERGENCE
        seeds = seeds[:1] if not seeds_override else seeds
        target_slots = cfg.get("target_slots", ["00"])
        convergence_slots = {f"{fam}_{target_slots[0]}" for fam in FAMILIES}
        print(f"[offline_ref] CONVERGENCE mode: seeds={seeds}, targets={sorted(convergence_slots)}, "
              f"n_sobol={n_sobol} (checkpoints={CONVERGENCE_CHECKPOINTS})")
    else:
        convergence_slots = set()
        print(f"[offline_ref] FULL mode: {len(seeds)} seeds, n_sobol={n_sobol}, "
              f"n_refine_starts={n_refine_starts}, n_ref_test_per_maneuver={n_ref_test_per_maneuver}")

    if n_jobs == 1:
        results = [
            _run_one_seed(
                seed, cfg, fleet_cfg, controller_cfg, mode=mode, n_sobol=n_sobol, n_refine_starts=n_refine_starts,
                n_ref_test_per_maneuver=n_ref_test_per_maneuver, xi_lower=xi_lower, xi_upper=xi_upper,
                state_scale=state_scale, fixed_R=fixed_R, eval_kwargs=eval_kwargs, cache_dir=cache_dir,
                convergence_slots=convergence_slots,
            )
            for seed in seeds
        ]
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
            delayed(_run_one_seed)(
                seed, cfg, fleet_cfg, controller_cfg, mode=mode, n_sobol=n_sobol, n_refine_starts=n_refine_starts,
                n_ref_test_per_maneuver=n_ref_test_per_maneuver, xi_lower=xi_lower, xi_upper=xi_upper,
                state_scale=state_scale, fixed_R=fixed_R, eval_kwargs=eval_kwargs, cache_dir=cache_dir,
                convergence_slots=convergence_slots,
            )
            for seed in seeds
        )

    ref_rows = [row for rr, _ in results for row in rr]
    conv_rows = [row for _, cr in results for row in cr]

    write_run_manifest(str(out_dir), {
        "config": cfg, "fleet_config": fleet_cfg, "controller_config": controller_cfg,
        "mode": mode, "n_sobol": n_sobol, "n_refine_starts": n_refine_starts,
        "n_ref_test_per_maneuver": n_ref_test_per_maneuver, "seeds": seeds,
    })

    if mode == "convergence":
        conv_df = pd.DataFrame(conv_rows)
        conv_path = out_dir / "processed" / "convergence.csv"
        conv_df.to_csv(conv_path, index=False)
        print(f"\n[offline_ref] wrote {conv_path} ({len(conv_df)} rows)")
        print("\n" + "=" * 90)
        print("Convergence: |J_search_best(N) - J_search_best(4096)| / J_search_best(4096), by checkpoint")
        print("=" * 90)
        print(conv_df.groupby("n_sobol_checkpoint")["rel_dev_from_final"].agg(["mean", "median", "max"]).round(5))
        print("\n(if the 1024-or-2048 row is already ~0, that N is safe to use for --mode full)")
    else:
        ref_df = pd.DataFrame(ref_rows)
        ref_path = out_dir / "processed" / "dense_reference.csv"
        ref_df.to_csv(ref_path, index=False)
        print(f"\n[offline_ref] wrote {ref_path} ({len(ref_df)} rows)")
        print("\n" + "=" * 90)
        print("New dense/independent J_ref_test vs. old 32-point oracle_rmse_weak_32pt")
        print("=" * 90)
        valid = ref_df[ref_df["reference_feasible"]]
        ratio = valid["J_ref_test"] / valid["oracle_rmse_weak_32pt"]
        print(f"J_ref_test / oracle_rmse_weak_32pt: mean={ratio.mean():.4f}, median={ratio.median():.4f} "
              f"(< 1 means the new reference is a TIGHTER/better bound than the old 32-pt one, as expected)")
        print(f"phi_search (feasible fraction of the dense Sobol scan): "
              f"{ref_df.groupby('family')['phi_search'].mean().round(4).to_dict()}")
        n_infeasible = (~ref_df["reference_feasible"]).sum()
        if n_infeasible:
            print(f"WARNING: {n_infeasible}/{len(ref_df)} targets had NO feasible Sobol point - "
                  f"inspect these before using dense_reference.csv downstream.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/runs/phase2_warmstart_confirmatory_c1cds02.yaml")
    parser.add_argument("--mode", choices=["convergence", "full"], default="convergence")
    parser.add_argument("--n-sobol", type=int, default=DEFAULT_N_SOBOL_FULL,
                         help="Only used in --mode full; convergence mode always uses 4096.")
    parser.add_argument("--n-refine-starts", type=int, default=DEFAULT_N_REFINE_STARTS)
    parser.add_argument("--n-ref-test-per-maneuver", type=int, default=DEFAULT_N_REF_TEST_PER_MANEUVER)
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                         help="Override cfg['seeds'] (e.g. for a quick smoke run on 1-2 seeds).")
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    main(args.config, args.mode, args.n_sobol, args.n_refine_starts, args.n_ref_test_per_maneuver,
         args.seeds, args.n_jobs)
