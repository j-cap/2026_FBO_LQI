"""Direct weight transfer audit; no source BO or target BO is rerun.

Run from any directory. See docs/direct_transfer_experiment.md.
Selection uses source calibration records only. Test/reference data never select weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import qmc

from src.caching import make_cached_evaluate
from src.dynamics_distance import pairwise_uncertainty_aware_distance
from src.episodes import build_multi_maneuver_episode_bank, test_episodes
from src.ifac_bridge import pack_theta_AB
from src.io_utils import deep_update, load_run_config, load_yaml, set_global_seed
from src.optimization.gp_surrogate import fit_weighted_gp
from phase2_fbo_comparison import build_fleet_and_identify, _resolve
from generate_offline_reference import REFERENCE_TEST_BASE_SEED

XI = ['xi_1', 'xi_2', 'xi_3']


def select_sources(history, target_id):
    """Stable, feasible calibration minimum per peer; never silently drop a peer."""
    peers = history[history.client_id != target_id].copy()
    rows = []
    for cid, group in peers.groupby('client_id', sort=True):
        eligible = group[group.feasible & np.isfinite(group.training_y)]
        if eligible.empty:
            raise ValueError(f'No feasible source calibration for {cid}; define a fallback before running')
        rows.append(eligible.sort_values('training_y', kind='stable').iloc[0])
    if not rows:
        raise ValueError('No eligible sources')
    return pd.DataFrame(rows).reset_index(drop=True)


def pooled_recommendation(history, target_id, lower, upper, seed, n_candidates):
    peers = history[history.client_id != target_id]
    x = (peers[XI].to_numpy(float) - lower) / (upper - lower)
    fit = fit_weighted_gp(x, peers.training_y.to_numpy(float), np.ones(len(peers)), seed=seed)
    if fit.gp is None:
        raise ValueError('Insufficient source data for pooled GP')
    # Same deterministic candidate grid as target BO; mean minimization, not EI.
    grid = qmc.Sobol(d=3, scramble=True, seed=seed + 1).random(n_candidates)
    return lower + grid[np.argmin(fit.gp.predict(grid))] * (upper - lower)


def score_fields(cost, feasible, reference):
    feasible = bool(feasible and cost is not None and np.isfinite(cost) and cost > 0)
    regret = float(cost / reference - 1) if feasible else np.inf
    return dict(feasible=feasible, J_test=float(cost) if feasible else np.nan,
                regret=regret, success=bool(feasible and regret <= 0.05))


def run_seed(seed, history, traces, references, cfg, fleet_cfg, controller_cfg, out, signature, test_n, smoke):
    dest = out / 'seeds' / f'{seed}.json'
    if dest.exists():
        saved = json.loads(dest.read_text())
        if saved['signature'] != signature:
            raise ValueError(f'Stale checkpoint {dest}; use a new output tag')
        print(f'seed={seed}: reused checkpoint', flush=True)
        return saved['rows']
    set_global_seed(seed)
    ev_kw = dict(u_min=controller_cfg['u_min'], u_max=controller_cfg['u_max'],
                 u_rate_max=controller_cfg.get('u_rate_max'), beta_max=np.deg2rad(controller_cfg['beta_max_deg']),
                 noise_std=controller_cfg['noise_std'], process_noise=controller_cfg['process_noise'],
                 plant_mode=controller_cfg.get('plant_mode', 'linear'), mu=controller_cfg.get('tire_mu', 1.0))
    scale, fixed_r = controller_cfg['state_scale'], controller_cfg['fixed_R']
    # Recreate identification only; do not repeat the old oracle/source optimization.
    fleet, _ = build_fleet_and_identify(deep_update(fleet_cfg, {'seed': fleet_cfg['seed'] + seed}),
                                      controller_cfg, cfg, eval_kwargs=ev_kw, state_scale=scale, fixed_R=fixed_r)
    idx = {c.client_id: k for k, c in enumerate(fleet)}
    distance = pairwise_uncertainty_aware_distance([pack_theta_AB(c.Ad_hat, c.Bd_hat) for c in fleet],
                                                  [c.W_raw for c in fleet])
    evaluate = make_cached_evaluate(str(out / 'cache' / signature / f'seed_{seed}'))
    lower, upper = np.asarray(controller_cfg['xi_lower']), np.asarray(controller_cfg['xi_upper'])
    target_ids = sorted(traces.client_id.unique())
    if smoke:
        target_ids = target_ids[:1]
    rows = []
    for target_id in target_ids:
        target = fleet[idx[target_id]]
        ref = references[references.client_id == target_id]
        if len(ref) != 1 or not bool(ref.iloc[0].reference_feasible) or not (ref.iloc[0].J_ref_test > 0):
            raise ValueError(f'Missing/invalid reference for {seed}/{target_id}')
        jref = float(ref.iloc[0].J_ref_test)
        bank = build_multi_maneuver_episode_bank(target, Ts=fleet_cfg['Ts'], T_total=controller_cfg['T_total'],
            T0=controller_cfg['T0'], r_max=controller_cfg['r_max'], lane_change_time=fleet_cfg['lane_change_time'],
            n_calibration_per_maneuver=0, n_test_per_maneuver=test_n,
            base_seed=REFERENCE_TEST_BASE_SEED, tfilter=controller_cfg.get('tfilter'))
        memo = {}
        def score(xi):
            key = tuple(float(v) for v in xi)
            if key not in memo:
                ev = evaluate(target, np.asarray(xi), bank, state_scale=scale, fixed_R=fixed_r,
                              episodes=test_episodes(bank), **ev_kw)
                memo[key] = dict(**score_fields(ev.tracking_rmse, ev.feasible, jref), failure_reason=ev.failure_reason)
            return memo[key]
        base = dict(seed=int(seed), client_id=target_id, family=target_id.rsplit('_', 1)[0], J_ref_test=jref)
        selected = select_sources(history, target_id)
        nearest = min(selected.client_id, key=lambda cid: (distance[idx[target_id], idx[cid]], cid))
        for source in selected.itertuples():
            xi = [getattr(source, k) for k in XI]
            pair = dict(**base, method='direct_random', source_id=source.client_id,
                        source_training_y=float(source.training_y), dynamics_distance=float(distance[idx[target_id], idx[source.client_id]]),
                        n_local_evals=0, target_validation_evals=1, probability=1 / len(selected),
                        **dict(zip(XI, xi)), **score(xi))
            rows.append(pair)
            if source.client_id == nearest:
                rows.append(dict(pair, method='direct_dynamics', probability=1.0))
        xi = pooled_recommendation(history, target_id, lower, upper, seed, cfg['n_candidates'])
        rows.append(dict(**base, method='source_pooled', source_id='', n_local_evals=0,
                         target_validation_evals=1, probability=1.0, **dict(zip(XI, xi)), **score(xi)))
        for record in traces[traces.client_id == target_id].itertuples():
            xi = [getattr(record, k) for k in XI]
            rows.append(dict(**base, method='bo_' + record.method, source_id='', n_local_evals=int(record.n_local_evals),
                             target_validation_evals=1, probability=1.0, **dict(zip(XI, xi)), **score(xi)))
        print(f'seed={seed} target={target_id}: {len(memo)} distinct controllers scored', flush=True)
    # Atomic per-seed checkpoint; interrupted seeds can reuse their evaluation cache.
    payload = dict(signature=signature, rows=rows)
    temp = dest.with_suffix('.tmp')
    temp.write_text(json.dumps(payload, indent=2))
    temp.replace(dest)
    return rows


def summarize(rows, out):
    df = pd.DataFrame(rows)
    processed = out / 'processed'
    processed.mkdir(exist_ok=True)
    df.to_csv(processed / 'controller_scores.csv', index=False)
    # Uniform-source expectations within target, then equal-weight targets within fleet.
    keys = ['seed', 'client_id', 'method', 'n_local_evals']
    target = df.groupby(keys).agg(success_rate=('success', 'mean'), feasible_rate=('feasible', 'mean'),
                                 mean_regret=('regret', lambda x: float(np.mean(x.to_numpy())))).reset_index()
    target.to_csv(processed / 'target_metrics.csv', index=False)
    fleet = target.groupby(['seed', 'method', 'n_local_evals'])[['success_rate', 'feasible_rate', 'mean_regret']].mean().reset_index()
    fleet.to_csv(processed / 'fleet_metrics.csv', index=False)
    summary = []
    for (method, budget), g in fleet.groupby(['method', 'n_local_evals']):
        rng = np.random.default_rng(20261117)
        vals = g.success_rate.to_numpy()
        boot = rng.choice(vals, size=(5000, len(vals)), replace=True).mean(axis=1)
        observations = df[(df.method == method) & (df.n_local_evals == budget)].regret.to_numpy()
        ordered = np.sort(observations)
        # Inverse empirical CDF retains failures at +infinity without interpolation artifacts.
        quantile = lambda q: ordered[max(0, int(np.ceil(q * len(ordered))) - 1)]
        summary.append(dict(median_regret=quantile(.5), p90_regret=quantile(.9), method=method, n_local_evals=budget, n_fleets=len(g),
                            success_rate=vals.mean(), success_ci_low=np.quantile(boot, .025),
                            success_ci_high=np.quantile(boot, .975), feasible_rate=g.feasible_rate.mean()))
    summary = pd.DataFrame(summary)
    summary.to_csv(processed / 'summary.csv', index=False)
    # Paired differences: positive success delta means BO helped; negative regret delta means BO helped.
    paired = []
    for baseline in ['direct_random', 'direct_dynamics', 'source_pooled']:
        a = target[target.method == baseline]
        for method in sorted(m for m in target.method.unique() if m.startswith('bo_')):
            b = target[target.method == method]
            m = b.merge(a, on=['seed', 'client_id'], suffixes=('_bo', '_direct'), validate='many_to_one')
            for r in m.itertuples():
                paired.append(dict(seed=r.seed, client_id=r.client_id, method=method, baseline=baseline,
                    n_local_evals=r.n_local_evals_bo, success_delta=r.success_rate_bo-r.success_rate_direct,
                    regret_delta=r.mean_regret_bo-r.mean_regret_direct))
    pd.DataFrame(paired).to_csv(processed / 'paired_differences.csv', index=False)
    paired_fleet = pd.DataFrame(paired).groupby(['seed', 'method', 'baseline', 'n_local_evals'])['success_delta'].mean().reset_index()
    paired_fleet.to_csv(processed / 'paired_fleet_success.csv', index=False)
    # Each fleet gets its own matrix: do not pool unrelated client IDs across seeds.
    pairs = df[df.method == 'direct_random']
    for seed, g in pairs.groupby('seed'):
        g.pivot(index='client_id', columns='source_id', values='regret').to_csv(processed / f'transfer_matrix_{seed}.csv')
    make_figures(df, summary, out)
    print(summary.to_string(index=False))


def make_figures(df, summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = out / 'figures'
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 3.8), layout='constrained')
    labels = {'bo_global': 'Fleet-informed BO', 'bo_independent': 'Local BO',
              'bo_similarity': 'Dynamics-weighted BO', 'direct_random': 'Uniform source transfer',
              'direct_dynamics': 'Dynamics-selected transfer', 'source_pooled': 'Source-only pooled recommendation'}
    colors = {'bo_global': '#0072B2', 'bo_independent': '#666666', 'bo_similarity': '#56B4E9',
              'direct_random': '#009E73', 'direct_dynamics': '#D55E00', 'source_pooled': '#CC79A7'}
    for method, group in summary.groupby('method'):
        if method.startswith('bo_'):
            ax.plot(group.n_local_evals, group.success_rate, marker='o', ms=3, label=labels[method], color=colors[method])
        else:
            ax.axhline(group.success_rate.iloc[0], linestyle='--', label=labels[method], color=colors[method])
    ax.set(xlabel='Target optimization evaluations (one final validation additional)',
           ylabel='Fraction within 5% of reference', ylim=(-.02, 1.02), xlim=(0, df.n_local_evals.max()))
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, 1.30), ncol=2)
    ax.grid(alpha=.2)
    fig.savefig(figures / 'success_vs_budget.png', dpi=180)
    fig.savefig(figures / 'success_vs_budget.pdf')
    plt.close(fig)
    pairs = df[df.method == 'direct_random']
    for seed, group in pairs.groupby('seed'):
        matrix = group.pivot(index='client_id', columns='source_id', values='regret')
        values = matrix.to_numpy(float)
        # Grey cells are excluded self-transfer, crimson crosses mark failed controllers.
        shown = np.where(np.isposinf(values), .25, values)
        fig, ax = plt.subplots(figsize=(10, 3.5), layout='constrained')
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad('#dddddd')
        im = ax.imshow(shown, aspect='auto', vmin=0, vmax=.25, cmap=cmap)
        yy, xx = np.where(np.isposinf(values))
        ax.scatter(xx, yy, marker='x', color='crimson', s=25)
        ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(len(matrix.index)), matrix.index, fontsize=8)
        ax.set(xlabel='Source', ylabel='Target', title=f'Fleet {seed}: direct-transfer relative regret')
        fig.colorbar(im, ax=ax, label='Relative regret (display clipped to [0, 0.25])')
        fig.savefig(figures / f'transfer_matrix_{seed}.png', dpi=160)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['preflight', 'smoke', 'run', 'analyze'])
    p.add_argument('--config', default='config/runs/fleet_maturity_sweep_c1cds02.yaml')
    p.add_argument('--campaign-tag', default=None)
    p.add_argument('--output-tag', default=None)
    p.add_argument('--n-jobs', type=int, default=1)
    a = p.parse_args()
    cfg = load_run_config(_resolve(a.config))
    cfg.pop('_config_path', None)
    campaign = ROOT / 'results' / (a.campaign_tag or cfg['run_tag'])
    maturity = (campaign / 'processed/nested_source_histories.parquet').exists()
    hp = campaign / 'processed' / ('nested_source_histories.parquet' if maturity else 'mature_fleet_records.parquet')
    tp = campaign / 'processed' / ('maturity_target_traces.parquet' if maturity else 'target_convergence_traces.parquet')
    rp = ROOT / 'results/offline_reference/processed/dense_reference.csv'
    mp = ROOT / 'results/offline_reference/run_manifest.yaml'
    for path in [hp, tp, rp, mp]:
        if not path.exists():
            raise FileNotFoundError(f'Required existing result missing: {path}. Restore the original campaign; do not regenerate sources.')
    h, t, ref = pd.read_parquet(hp), pd.read_parquet(tp), pd.read_csv(rp)
    if maturity:
        h = h[h.eval_index <= 25].copy()
        t = t[(t.method == 'independent') | ((t.method == 'global') & (t.maturity == 25))].copy()
    fc = deep_update(load_yaml(_resolve(cfg['fleet_config'])), cfg.get('fleet_overrides', {}))
    cc = deep_update(load_yaml(_resolve(cfg['controller_config'])), cfg.get('controller_overrides', {}))
    manifest = load_yaml(str(mp))['resolved_config']
    # The reference may have been generated via the maturity campaign, but its physical/task configuration must match.
    for name, value in [('fleet_config', fc), ('controller_config', cc)]:
        if name in manifest and manifest[name] != value:
            raise ValueError(f'Reference {name} differs from this experiment')
    for key in ['episode_base_seed', 'identification_seed_offset', 'n_identification_episodes', 'n_calibration_episodes']:
        if manifest['config'].get(key) != cfg.get(key):
            raise ValueError(f'Reference protocol differs at {key}')
    seeds = sorted(int(s) for s in cfg['seeds'])
    expected_targets = {f'{family}_{slot}' for family in ['nominal', 'payload', 'tire_degraded'] for slot in cfg['target_slots']}
    for seed in seeds:
        hs, ts = h[h.seed == seed], t[t.seed == seed]
        if len(hs.client_id.unique()) != 3 * fc['clients_per_regime'] or not hs.groupby('client_id').size().eq(25).all():
            raise ValueError(f'Incomplete 25-evaluation source histories: seed {seed}')
        if set(ts.method) != ({'independent', 'global'} if maturity else {'independent', 'global', 'similarity'}):
            raise ValueError(f'Unexpected BO methods for seed {seed}')
        for _, group in ts.groupby(['client_id', 'method']):
            if sorted(group.n_local_evals) != list(range(2, 3 + cfg['target_n_iters'])):
                raise ValueError(f'Incomplete BO budget trace for seed {seed}')
        if set(ts.client_id) != expected_targets or ts.duplicated(['client_id', 'method', 'n_local_evals']).any():
            raise ValueError(f'Incomplete/duplicated target traces: seed {seed}')
        for target_id in expected_targets:
            select_sources(hs, target_id)
        rr = ref[(ref.seed == seed) & ref.client_id.isin(expected_targets)]
        if len(rr) != len(expected_targets) or not rr.reference_feasible.all() or not (rr.J_ref_test > 0).all():
            raise ValueError(f'Incomplete references: seed {seed}')
    if not np.isfinite(h[XI + ['training_y']].to_numpy(float)).all():
        raise ValueError('Nonfinite source records')
    print(f'Preflight OK: {len(seeds)} fleets, {len(expected_targets)} targets/fleet, 25 source evaluations. maturity={maturity}')
    if a.mode == 'preflight':
        return
    out = ROOT / 'results' / ((a.output_tag or (cfg['run_tag'] + '_direct_transfer')) + ('_smoke' if a.mode == 'smoke' else ''))
    signature_data = dict(config=cfg, fleet=fc, controller=cc,
        files={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in [hp, tp, rp, mp]},
        code_hashes={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(list((ROOT / 'src').rglob('*.py')) +
                     [Path(__file__), ROOT / 'scripts/phase2_fbo_comparison.py'])}, smoke=a.mode == 'smoke',
        reference_test_seed=REFERENCE_TEST_BASE_SEED)
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'seeds').mkdir(exist_ok=True)
    protocol_path = out / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != signature_data:
        raise ValueError('Output protocol differs; choose a new --output-tag')
    protocol_path.write_text(json.dumps(signature_data, indent=2))
    if a.mode == 'smoke':
        seeds = seeds[:1]
    if a.mode == 'analyze':
        rows = []
        for seed in seeds:
            saved = json.loads((out / 'seeds' / f'{seed}.json').read_text())
            if saved['signature'] != signature:
                raise ValueError('Checkpoint signature mismatch')
            rows.extend(saved['rows'])
    else:
        results = Parallel(n_jobs=a.n_jobs, backend='loky', inner_max_num_threads=1)(delayed(run_seed)(
            seed, h[h.seed == seed], t[t.seed == seed], ref[ref.seed == seed], cfg, fc, cc, out, signature,
            int(manifest['n_ref_test_per_maneuver']), a.mode == 'smoke') for seed in seeds)
        rows = [r for group in results for r in group]
    summarize(rows, out)


if __name__ == '__main__':
    main()
