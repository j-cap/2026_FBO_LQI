"""Read-only cost-landscape and feature-support review using completed archives.

No new physical rollout, BO proposal, model variant or tuning campaign is run.
Dense-bank decompositions are evaluator-only diagnostics, not online features.
"""
from pathlib import Path
import json

import analyze_tracking_peer_cost as audit
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.tracking_online import features

runner = audit.runner
OUT = runner.REPO/'docs/analysis/heterogeneous_cost_model_review'


def landscape_review(state):
    raw = pd.read_csv(state['archive']/'controlled_evaluations.csv', float_precision='round_trip')
    cal = raw[raw.role == 'calibration'].copy()
    cal['eligible'] = cal.complete & cal.feasible & cal.integration_resolved
    cal['log_J'] = np.log(cal.J.where(cal.eligible))
    table = cal.groupby(['vehicle', 'context', 'candidate'], sort=False).agg(
        repeats=('seed', 'size'), eligible=('eligible', 'all'), mean_J=('J', 'mean'),
        mean_log_J=('log_J', 'mean'), repeat_log_variance=('log_J', 'var')).reset_index()
    if not (table.repeats == 2).all():
        raise RuntimeError('The balanced calibration bank must have two repeats per cell.')
    vehicles = [v['id'] for v in state['all_vehicles']]
    families = {v['id']: v['family'] for v in state['all_vehicles']}
    summaries, offsets, pairwise, regions, shape_blocks = [], [], [], [], []
    for context in state['domain'].contexts:
        group = table[table.context == context]
        eligibility = group.pivot(index='vehicle', columns='candidate', values='eligible').loc[vehicles]
        common = eligibility.columns[eligibility.all(axis=0)].tolist()
        for vehicle in vehicles:
            regions.append(dict(context=context, vehicle=vehicle, family=families[vehicle],
                eligible_candidates=int(eligibility.loc[vehicle].sum()), total_candidates=len(eligibility.columns),
                common_eligible_candidates=len(common)))
        if len(common) < 2:
            raise RuntimeError('Too few common eligible gains for the cost-shape diagnostic.')
        used = group[group.candidate.isin(common)]
        Y = used.pivot(index='vehicle', columns='candidate', values='mean_log_J').loc[vehicles, common].to_numpy()
        J = used.pivot(index='vehicle', columns='candidate', values='mean_J').loc[vehicles, common].to_numpy()
        # A full common-pool lookup is deliberately more flexible than the RFF mean.
        pooled = Y.mean(axis=0, keepdims=True)
        deviation = Y-pooled
        shift = deviation.mean(axis=1, keepdims=True)
        residual = deviation-shift
        np.testing.assert_allclose(residual.sum(axis=0), 0., atol=1e-10)
        np.testing.assert_allclose(residual.sum(axis=1), 0., atol=1e-10)
        # Give each context equal weight in the joint task-shape diagnostic.
        shape_blocks.append(residual/np.sqrt(len(common)))
        total = float(np.sum(deviation**2))
        after_shift = float(np.sum(residual**2))
        np.testing.assert_allclose(total, after_shift+len(common)*float(np.sum(shift**2)), atol=1e-10)
        family_residual = Y.copy()
        for family in sorted(set(families.values())):
            indices = [i for i, v in enumerate(vehicles) if families[v] == family]
            family_residual[indices] -= Y[indices].mean(axis=0, keepdims=True)
        family_ss = float(np.sum(family_residual**2))
        shape_singular = np.linalg.svd(residual, compute_uv=False)
        shape_variance = shape_singular**2
        correlations = []
        for i in range(len(vehicles)):
            offsets.append(dict(context=context, vehicle=vehicles[i], family=families[vehicles[i]],
                common_eligible_candidates=len(common), local_log_cost_offset=float(shift[i, 0]),
                multiplicative_cost_level=float(np.exp(shift[i, 0]))))
            for j in range(i+1, len(vehicles)):
                rho = float(spearmanr(Y[i], Y[j]).statistic)
                correlations.append(rho)
                pairwise.append(dict(context=context, first=vehicles[i], second=vehicles[j],
                    same_family=families[vehicles[i]] == families[vehicles[j]],
                    common_eligible_candidates=len(common), cost_rank_correlation=rho,
                    log_cost_difference_RMS=float(np.sqrt(np.mean((Y[i]-Y[j])**2))),
                    centered_shape_difference_RMS=float(np.sqrt(np.mean((residual[i]-residual[j])**2)))))
        noise_variance = float(used.repeat_log_variance.mean())
        near = np.any(J <= 1.05*J.min(axis=1, keepdims=True), axis=0)
        near_Y = Y[:, near]
        near_deviation = near_Y-near_Y.mean(axis=0, keepdims=True)
        near_residual = near_deviation-near_deviation.mean(axis=1, keepdims=True)
        near_total = float(np.sum(near_deviation**2))
        summaries.append(dict(context=context, common_eligible_candidates=len(common),
            union_eligible_candidates=int(eligibility.any(axis=0).sum()),
            min_vehicle_eligible_candidates=int(eligibility.sum(axis=1).min()),
            max_vehicle_eligible_candidates=int(eligibility.sum(axis=1).max()),
            between_vehicle_log_cost_RMS=float(np.sqrt(np.mean(deviation**2))),
            residual_after_context_offset_RMS=float(np.sqrt(np.mean(residual**2))),
            fraction_between_vehicle_SS_removed_by_context_offset=1-after_shift/total if total else None,
            fraction_between_vehicle_SS_removed_by_family_lookup=1-family_ss/total if total else None,
            centered_shape_fraction_explained_by_first_component=float(shape_variance[:1].sum()/shape_variance.sum())
                if shape_variance.sum() else None,
            centered_shape_fraction_explained_by_two_components=float(shape_variance[:2].sum()/shape_variance.sum())
                if shape_variance.sum() else None,
            repeat_noise_log_cost_SD=float(np.sqrt(noise_variance)),
            median_pairwise_cost_rank_correlation=float(np.median(correlations)),
            min_pairwise_cost_rank_correlation=float(np.min(correlations)),
            distinct_common_pool_best_gains=len(set(np.argmin(J, axis=1))),
            near_reference_union_candidates=int(near.sum()),
            near_region_between_vehicle_log_cost_RMS=float(np.sqrt(np.mean(near_deviation**2))),
            near_region_offset_removed_fraction=(1-float(np.sum(near_residual**2))/near_total)
                if near.sum() >= 2 and near_total else None))
    joint_singular = np.linalg.svd(np.concatenate(shape_blocks, axis=1), compute_uv=False)
    joint_variance = joint_singular**2
    fractions = joint_variance/joint_variance.sum()
    spectrum = pd.DataFrame(dict(component=np.arange(1, len(fractions)+1),
        variance_fraction=fractions, cumulative_variance_fraction=np.cumsum(fractions)))
    return (*tuple(pd.DataFrame(rows) for rows in (summaries, offsets, pairwise, regions)), table, spectrum)


def cost_slice_figure(state, table):
    """Illustrative fixed-gain scaling slice, not an extra controller search."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    matplotlib.rcParams['svg.hashsalt'] = 'fedbo-heterogeneous-cost-review'
    colors = dict(compact='#238365', sport='#3478b2', suv='#ba623d')
    styles = dict(nominal='-', payload='--', front_tire=':')
    anchors = {p['id']: p['multipliers'][0] for p in state['pool'] if p['kind'] == 'scaling'}
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for ax, context in zip(axes.flat, state['domain'].contexts):
        for vehicle in state['all_vehicles']:
            g = table[(table.vehicle == vehicle['id']) & (table.context == context) & table.candidate.isin(anchors)].copy()
            g['gain_multiplier'] = g.candidate.map(anchors)
            g = g.sort_values('gain_multiplier')
            ax.plot(g.gain_multiplier, g.mean_J.where(g.eligible), color=colors[vehicle['family']],
                linestyle=styles[vehicle['variant']], marker='o', markersize=3, linewidth=1.2, alpha=.85)
        ax.set(xscale='log', yscale='log', title=context,
               xlabel='Common multiplier of factory gains', ylabel='Mean rollout J')
        ax.set_xticks([.5, 1, 2, 4, 8], ['0.5', '1', '2', '4', '8'])
        ax.grid(alpha=.2)
    legend = [Line2D([0], [0], color=color, label=family) for family, color in colors.items()]
    legend += [Line2D([0], [0], color='#555555', linestyle=style, label=variant.replace('_', ' '))
               for variant, style in styles.items()]
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, 1.06), ncol=6, fontsize=9)
    vector = OUT/'common_scaling_cost_slices.svg'
    fig.savefig(vector, bbox_inches='tight', metadata={'Date': None})
    # Matplotlib emits spaces at SVG path line ends; keep the tracked text clean.
    vector.write_text('\n'.join(line.rstrip() for line in vector.read_text().splitlines())+'\n')
    preview = runner.ROOT/'figures'/'heterogeneous_cost_model_review.png'
    preview.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview, dpi=140, bbox_inches='tight')
    plt.close(fig)


def online_support(state, results):
    phi = features(state['domain'], state['cfg']['model'])
    rows, clients = [], []
    for job in results:
        data = job['state']['methods']['peer_federated_cost_bo']
        for round_id in [1, 2, 3, 6, 12, 23, 24]:
            observations = [e for e in data['events'] if e['round'] <= round_id and e['eligible']]
            X = phi[[e['action'] for e in observations]]
            if len(X):
                singular = np.linalg.svd(X, compute_uv=False)
                rank = int(np.sum(singular > max(X.shape)*np.finfo(float).eps*singular[0]))
                G = X.T@X/len(X)
                eigenvalues = np.maximum(np.linalg.eigvalsh(G), 0.)
                effective_df = float(np.sum(eigenvalues/(eigenvalues+state['cfg']['model']['ridge'])))
                ys = pd.DataFrame(dict(action=[e['action'] for e in observations],
                                       log_J=np.log([e['J'] for e in observations])))
                error = ys.log_J-ys.groupby('action').log_J.transform('mean')
                disagreement = float(np.sqrt(np.mean(error**2)))
            else:
                rank, effective_df, disagreement = 0, 0., 0.
            counts = [sum(e['client'] == i for e in observations) for i in range(len(job['clients']))]
            if not (rank <= len(set(e['action'] for e in observations)) <= len(observations)
                    and effective_df <= rank+1e-8 and max(counts) <= round_id):
                raise RuntimeError('Invalid online feature-support or eligible-query accounting.')
            rows.append(dict(condition=job['condition'], seed=job['seed'], round=round_id,
                used_by_later_query=round_id < 24, attempted_queries=len(job['clients'])*round_id,
                eligible_cost_records=len(observations),
                distinct_gain_context_inputs=len(set(e['action'] for e in observations)),
                distinct_gain_vectors=len(set(e['candidate'] for e in observations)),
                mean_coefficients=phi.shape[1], unregularized_design_rank=rank,
                ridge_effective_degrees_of_freedom=effective_df,
                pooled_same_input_log_label_disagreement_RMS=disagreement,
                min_local_eligible_records=min(counts), max_local_eligible_records=max(counts)))
            for client, count in zip(job['clients'], counts):
                clients.append(dict(condition=job['condition'], seed=job['seed'], round=round_id,
                    client=client['id'], plant_id=client['plant_id'], family=client['family'],
                    eligible_cost_records=count, sample_weight=count/len(observations) if observations else 0.))
    return pd.DataFrame(rows), pd.DataFrame(clients)


def claim_ledger(paired, per_seed):
    rows = []
    for _, row in paired[paired.fleet_budget == 216].iterrows():
        seeds = per_seed[(per_seed.fleet_budget == 216) & (per_seed.condition == row.condition) &
                         (per_seed.method == row.method) & (per_seed.baseline == row.baseline)]
        rows.append(dict(condition=row.condition, contrast=row.method+' minus '+row.baseline,
            context_success_delta=row.context_success_delta,
            eligible_policies_method=int(row.eligible_policies_method),
            eligible_policies_baseline=int(row.eligible_policies_baseline),
            jointly_eligible_pairs=int(row.jointly_feasible_pairs),
            relative_reduction_of_joint_mean_J=row.relative_reduction_of_joint_mean_J,
            positive_context_success_seeds=int((seeds.context_success_delta > 1e-12).sum()),
            negative_context_success_seeds=int((seeds.context_success_delta < -1e-12).sum()),
            equal_context_success_seeds=int((seeds.context_success_delta.abs() <= 1e-12).sum()),
            interpretation=('Matched primary model-sharing contrast' if row.baseline == 'peer_local_cost_bo'
                            else 'Practical/architecture control; read cost and coverage together')))
    return pd.DataFrame(rows)


def main():
    state, results, _, _, _, queries, _, _, _, paired, per_seed = audit.verify()
    landscapes, offsets, pairs, regions, table, spectrum = landscape_review(state)
    support, weights = online_support(state, results)
    OUT.mkdir(parents=True, exist_ok=True)
    cost_slice_figure(state, table)
    for name, frame in [('balanced_landscape_decomposition', landscapes), ('vehicle_context_offsets', offsets),
        ('pairwise_common_pool_shapes', pairs), ('eligibility_regions', regions),
        ('online_feature_support', support), ('online_client_weights', weights),
        ('joint_task_shape_spectrum', spectrum),
        ('condition_dependent_claims', claim_ledger(paired, per_seed))]:
        frame.to_csv(OUT/(name+'.csv'), index=False)
    paths = [state['archive']/'controlled_evaluations.csv', state['out']/'run_manifest.json',
             state['out']/'completion.json', state['out']/'paired_comparisons.csv']
    paths += [runner.path_for(state, job) for job in runner.jobs(state)]
    paths += [runner.ROOT/'src'/name for name in ('tracking_budget.py', 'tracking_online.py', 'tracking_online_lcb.py',
                                                'tracking_peer_cost.py')]
    runner.save(OUT/'verification.json', dict(result_signature=state['signature'], jobs=len(results),
        charged_queries=len(queries), completed_campaign_revalidated=True,
        dense_role='calibration only, two frozen repeats per vehicle/context/gain',
        dense_bank_use='Evaluator-only descriptive decomposition; never online training or context features',
        online_support_method='Actual C histories; no new labels or BO decisions',
        new_simulator_rollouts=0, new_BO_proposals=0, new_model_variants=0,
        analysis_source_sha256=runner.digest(Path(__file__)),
        input_hashes={p.relative_to(runner.REPO).as_posix(): runner.digest(p) for p in paths}))
    print('Balanced calibration-bank decomposition:')
    print(landscapes.to_string(index=False))
    print('Actual C feature support before the final query:')
    print(support[support['round'] == 23].to_string(index=False))
    print('Review diagnostics: '+str(OUT))


if __name__ == '__main__':
    main()
