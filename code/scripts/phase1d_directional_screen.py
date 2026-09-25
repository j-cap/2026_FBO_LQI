"""
Phase 1D: directional-asymmetry diagnostic (methodological response to Phase 1C's
finding that transfer risk L_{i<-j} is asymmetric while d_ij^dyn is symmetric by
construction). Two things, using ONLY Phase 1C's already-computed
transfer_loss_matrix.parquet / oracle_optima.parquet - no new true-plant simulation, no
new fleet, no full oracle campaign:

  1. Confirms the directional structure quantitatively: the 3x3 donor->recipient
     family matrix L_bar_{a->b} and the pairwise asymmetry A_ij = L_{i<-j} - L_{j<-i}.
  2. Tests whether a cheap recipient-side model screen (src/recipient_screen.py -
     evaluate the transferred xi on the RECIPIENT's own identified model, not the true
     plant, not the donor's data) explains that asymmetry: for every existing
     cross-client transfer xi_j* -> i, predict margins using client i's own
     (Ad_hat_i, Bd_hat_i), and check whether high-transfer-loss cases show
     systematically worse margins.

The fleet + identification are regenerated deterministically (same config, same seeds)
to recover each client's Ad_hat/Bd_hat - these aren't saved directly in the Phase-1
parquet outputs. Identification uses the SAME plant_mode as the source run (Phase 1C
used plant_mode=nonlinear_tanh even for identification episodes, per correction-plan
item 5 - only the CONTROLLER-DESIGN model stays linear, not the data it's fit on).

Usage:
    python scripts/phase1d_directional_screen.py --config config/runs/phase1c_fixed_speed_n60.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src.episodes import build_episode_bank, identification_episode_seeds  # noqa: E402
from src.fleet_families import generate_fixed_speed_fleet  # noqa: E402
from src.interfaces.identifier import fit as identifier_fit  # noqa: E402
from src.io_utils import deep_update, load_run_config, load_yaml, require_keys, set_global_seed  # noqa: E402
from src.recipient_screen import predict_recipient_margins  # noqa: E402

FEATURE_COLS = ["spectral_radius", "max_abs_beta_hat", "max_abs_delta_hat", "max_abs_delta_rate_hat"]


def _resolve(rel_path: str) -> str:
    p = Path(rel_path)
    return str(p if p.is_absolute() else _ROOT / p)


def _regime(client_id: str) -> str:
    return client_id.rsplit("_", 1)[0]


def regenerate_fleet_and_identification(cfg, fleet_cfg, controller_cfg):
    """Reproduces the source run's fleet + identified models exactly (same seed,
    same fleet_kind, same plant_mode for identification)."""
    set_global_seed(cfg["seed"])
    fleet_kind = fleet_cfg.get("fleet_kind", "cluster_enum")
    if fleet_kind != "fixed_speed_families":
        raise NotImplementedError(
            f"This diagnostic currently only supports fleet_kind='fixed_speed_families' (got {fleet_kind!r})."
        )
    fleet = generate_fixed_speed_fleet(
        Ts=fleet_cfg["Ts"], n_per_family=fleet_cfg["clients_per_regime"],
        variability=fleet_cfg["variability"], seed=fleet_cfg["seed"],
        delta=fleet_cfg["delta"], lambda_f=fleet_cfg["lambda_f"],
        lane_change_time=fleet_cfg["lane_change_time"],
    )

    banks = {}
    for client in fleet:
        banks[client.client_id] = build_episode_bank(
            client, Ts=fleet_cfg["Ts"], T_total=controller_cfg["T_total"], T0=controller_cfg["T0"],
            r_max=controller_cfg["r_max"], n_calibration=cfg["n_calibration_episodes"],
            n_test=cfg.get("n_test_episodes", 0), base_seed=cfg.get("episode_base_seed", 5000),
            tfilter=controller_cfg.get("tfilter"),
        )

    plant_mode = controller_cfg.get("plant_mode", "linear")
    tire_mu = controller_cfg.get("tire_mu", 1.0)
    id_seed_offset = cfg.get("identification_seed_offset", 1000)
    n_id_episodes = cfg.get("n_identification_episodes", 1)
    for client in fleet:
        bank = banks[client.client_id]
        id_seeds = identification_episode_seeds(client.client_id, n_id_episodes, id_seed_offset)
        identifier_fit(
            client, bank.r_ref, seeds=id_seeds, noise_std=controller_cfg["noise_std"],
            process_noise=controller_cfg["process_noise"], u_min=controller_cfg["u_min"],
            u_max=controller_cfg["u_max"], plant_mode=plant_mode, mu=tire_mu, Ts=fleet_cfg["Ts"],
        )
    return fleet, banks


def main(config_path: str) -> None:
    cfg = load_run_config(config_path)
    require_keys(cfg, ["run_tag", "fleet_config", "controller_config", "seed"])
    fleet_cfg = deep_update(load_yaml(_resolve(cfg["fleet_config"])), cfg.get("fleet_overrides", {}))
    controller_cfg = deep_update(load_yaml(_resolve(cfg["controller_config"])), cfg.get("controller_overrides", {}))

    source_dir = _ROOT / "results" / cfg["run_tag"] / "processed"
    out_dir = _ROOT / "results" / "phase1d_directional_screen"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[phase1d] regenerating fleet + identification for {cfg['run_tag']!r} (deterministic)...")
    fleet, banks = regenerate_fleet_and_identification(cfg, fleet_cfg, controller_cfg)
    client_by_id = {c.client_id: c for c in fleet}
    print(f"[phase1d] regenerated {len(fleet)} clients")

    transfer_df = pd.read_parquet(source_dir / "transfer_loss_matrix.parquet")
    transfer_df = transfer_df[transfer_df["client_i"] != transfer_df["client_j"]].copy()
    oracle_optima = pd.read_parquet(source_dir / "oracle_optima.parquet")
    ind_xi = oracle_optima[oracle_optima["level"] == "individual"].set_index("group")[["xi_1", "xi_2", "xi_3"]]

    # ---- 1) 3x3 donor(row) -> recipient(col) family matrix + asymmetry ----------------
    transfer_df["regime_i"] = transfer_df["client_i"].map(_regime)
    transfer_df["regime_j"] = transfer_df["client_j"].map(_regime)
    family_matrix = transfer_df.groupby(["regime_j", "regime_i"])["transfer_loss"].mean().unstack()
    # reindex to a shared, sorted (donor, recipient) axis order without letting
    # reindex's target-index adopt the wrong .name for display (it would otherwise
    # silently relabel the row axis as "regime_i" too, since that's family_matrix
    # .columns' index name - purely cosmetic, the VALUES are already correctly
    # oriented, but worth not printing a misleading axis label).
    order = sorted(family_matrix.columns)
    family_matrix = family_matrix.reindex(index=order, columns=order)
    family_matrix.index.name = "donor"
    family_matrix.columns.name = "recipient"
    print("\n[phase1d] donor(row) -> recipient(col) mean transfer loss L_bar_a->b:")
    print(family_matrix.round(4).to_string())

    asymmetry = family_matrix - family_matrix.T
    print("\n[phase1d] family-level asymmetry (L_bar_a->b - L_bar_b->a):")
    print(asymmetry.round(4).to_string())

    family_matrix.to_csv(out_dir / "family_donor_recipient_matrix.csv")
    asymmetry.to_csv(out_dir / "family_asymmetry_matrix.csv")

    paired = transfer_df.merge(
        transfer_df, left_on=["client_i", "client_j"], right_on=["client_j", "client_i"], suffixes=("", "_rev"),
    )
    paired["A_ij"] = paired["transfer_loss"] - paired["transfer_loss_rev"]
    paired = paired[["client_i", "client_j", "transfer_loss", "transfer_loss_rev", "A_ij"]]
    paired.to_parquet(out_dir / "pairwise_asymmetry.parquet", index=False)
    mean_abs_A = float(paired["A_ij"].abs().mean())
    max_abs_A = float(paired["A_ij"].abs().max())
    print(f"\n[phase1d] pairwise asymmetry: mean|A_ij|={mean_abs_A:.4f}, max|A_ij|={max_abs_A:.4f}")

    # ---- 2) recipient-side margins for every existing cross-client transfer -----------
    rows = []
    for _, row in transfer_df.iterrows():
        i, j = row["client_i"], row["client_j"]
        if j not in ind_xi.index or i not in client_by_id:
            continue
        xi_j_star = ind_xi.loc[j].to_numpy(dtype=float)
        client_i = client_by_id[i]
        bank_i = banks[i]
        margins = predict_recipient_margins(
            client_i, xi_j_star, bank_i.t, bank_i.r_ref,
            state_scale=tuple(controller_cfg["state_scale"]), fixed_R=controller_cfg["fixed_R"],
        )
        rows.append(
            {
                "client_i": i, "client_j": j, "transfer_loss": row["transfer_loss"],
                "nominal_feasible": margins.nominal_feasible,
                "spectral_radius": margins.spectral_radius,
                "max_abs_beta_hat": margins.max_abs_beta_hat,
                "max_abs_delta_hat": margins.max_abs_delta_hat,
                "max_abs_delta_rate_hat": margins.max_abs_delta_rate_hat,
            }
        )
    margins_df = pd.DataFrame(rows)
    margins_df.to_parquet(out_dir / "recipient_margins_vs_transfer_loss.parquet", index=False)
    feasible_frac = float(margins_df["nominal_feasible"].mean()) if len(margins_df) else float("nan")
    print(
        f"\n[phase1d] computed recipient-side margins for {len(margins_df)} cross-client transfers "
        f"({feasible_frac:.1%} nominally feasible on the recipient's own model)"
    )

    # ---- 3) does a poor true-plant transfer show a poor recipient-side margin? --------
    valid = margins_df.dropna(subset=["transfer_loss"])
    print("\n[phase1d] Spearman(transfer_loss, recipient-side margin):")
    correlations = {}
    for col in FEATURE_COLS:
        sub = valid.dropna(subset=[col])
        if len(sub) >= 8:
            rho, p_value = spearmanr(sub["transfer_loss"], sub[col])
            correlations[col] = float(rho)
            print(f"  {col:25s} rho={rho:+.3f} (p={p_value:.2e}, n={len(sub)})")
        else:
            correlations[col] = float("nan")
            print(f"  {col:25s} insufficient data (n={len(sub)})")

    # ---- 4) figure -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(FEATURE_COLS), figsize=(4.5 * len(FEATURE_COLS), 4))
    for ax, col in zip(axes, FEATURE_COLS):
        sub = valid.dropna(subset=[col])
        ax.scatter(sub[col], sub["transfer_loss"], s=10, alpha=0.4, color="#0072B2")
        ax.set_xlabel(col)
        ax.set_title(f"rho={correlations[col]:.2f}" if np.isfinite(correlations[col]) else "n/a")
    axes[0].set_ylabel(r"transfer loss $L_{i \leftarrow j}$")
    fig.suptitle("Phase 1D - transfer loss vs recipient-side model margins (computed on client i's OWN identified model)")
    fig.tight_layout()
    fig.savefig(out_dir / "figD1_transfer_loss_vs_recipient_margins.png", dpi=150)
    plt.close(fig)

    # ---- 5) success criterion: does any margin clearly track transfer loss? -----------
    finite_rhos = [v for v in correlations.values() if np.isfinite(v)]
    best_rho = max(finite_rhos, key=abs) if finite_rhos else float("nan")
    proceed = np.isfinite(best_rho) and abs(best_rho) >= 0.3
    verdict = "PROCEED_TO_GATED_FBO" if proceed else "START_SYMMETRIC_FBO_ONLY"
    print(f"\n[phase1d] strongest |margin correlation| = {best_rho:.3f} -> verdict: {verdict}")

    verdict_text = (
        "High-transfer-loss cases show systematically worse recipient-side margins - "
        "implement w_{i<-j}(xi) = s_ij * g_i(xi) (recipient-aware dynamics FBO) as a 4th "
        "method alongside independent / global / similarity-weighted BO."
        if proceed
        else (
            "No recipient-side margin clearly predicts transfer loss at this scale - start "
            "FBO with symmetric similarity weighting only (independent / global / "
            "similarity-weighted) and treat the observed asymmetry as a documented "
            "limitation rather than engineering a directional gate now."
        )
    )

    summary_lines = [
        "# Phase 1D - directional-asymmetry diagnostic",
        "",
        f"Source run: `{cfg['run_tag']}`",
        "",
        "## 3x3 donor(row) -> recipient(col) mean transfer loss",
        "",
        "```",
        family_matrix.round(4).to_string(),
        "```",
        "",
        "## Family-level asymmetry (L_bar_a->b - L_bar_b->a)",
        "",
        "```",
        asymmetry.round(4).to_string(),
        "```",
        "",
        f"Pairwise asymmetry: mean|A_ij|={mean_abs_A:.4f}, max|A_ij|={max_abs_A:.4f}",
        "",
        "## Recipient-side margin correlations with transfer loss",
        "",
    ]
    for col, rho in correlations.items():
        summary_lines.append(f"- {col}: rho={rho:.3f}" if np.isfinite(rho) else f"- {col}: n/a")
    summary_lines += ["", f"## Verdict: `{verdict}`", "", verdict_text]
    (out_dir / "phase1d_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"[phase1d] wrote summary to {out_dir / 'phase1d_summary.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/runs/phase1c_fixed_speed_n60.yaml")
    args = parser.parse_args()
    main(args.config)
