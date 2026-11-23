"""Verify frozen tracking results and export consolidation tables; no optimization.

Run from the repository root: python code/scripts/consolidate_tracking_evidence.py
The interpretation is maintained in docs/analysis/tracking_consolidation/report.md.
"""
import json
from pathlib import Path

import tracking_cost_shape as runner  # Sets BLAS threads before NumPy imports.
import numpy as np
import pandas as pd

from src.tracking_budget import dense_references, evaluate_job
from src.tracking_budget_report import summarize
from src.tracking_cost_shape import control_comparison, paired_effects, source_for_arm


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent/'docs/analysis/tracking_consolidation'


def verify():
    state = runner.setup()
    manifest = runner.base.read(state['out']/'run_manifest.json')
    completion = runner.base.read(state['out']/'completion.json')
    if manifest['signature'] != state['signature'] or completion['signature'] != state['signature']:
        raise RuntimeError('The completed experiment fingerprint differs.')
    sources = runner.frozen_sources(state)
    results = runner.load_jobs(state)
    if len(results) != completion['jobs'] or len(results) != 360:
        raise RuntimeError('The prescribed 360-job experiment is incomplete.')
    tables = dense_references(state['calibration'], state['validation'], state['domain'])
    if not all(v['eligible'] for v in tables[0].values()):
        raise RuntimeError('An empirical reference fails validation.')
    rows, cells, controls = [], [], []
    for result in results:
        r, c = evaluate_job(result, state['calibration'], state['validation'], state['domain'],
                            state['cfg'], tables)
        rows.extend(r); cells.extend(c)
        previous = runner.base.read(runner.base.checkpoint(state['reference_state'], result))
        expected = {k: result[k] for k in ('target', 'protocol', 'seed', 'noise_seed', 'fold')}
        runner.previous_runner.validate_job(state['reference_state'], expected, previous)
        controls.extend(control_comparison(result, previous))
    frame, contexts = pd.DataFrame(rows), pd.DataFrame(cells)

    def compare(current, name, keys):
        saved = pd.read_csv(state['out']/name)
        a = current.sort_values(keys).reset_index(drop=True)
        b = saved[a.columns].sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-12)

    compare(frame, 'budget_trials.csv', ['protocol', 'target', 'seed', 'method', 'budget'])
    compare(contexts, 'budget_contexts.csv', ['protocol', 'target', 'seed', 'method', 'budget', 'context'])
    summary = summarize(frame)
    compare(summary, 'cost_shape_summary.csv', ['protocol', 'method', 'budget'])
    effects, paired, _ = paired_effects(frame)
    compare(paired, 'paired_effects.csv', ['protocol', 'budget', 'metric', 'contrast'])
    audit = pd.DataFrame(controls)
    compare(audit, 'previous_control_audit.csv', ['protocol', 'target', 'seed', 'method'])
    if not audit[['action_trace_match', 'recommendation_trace_match']].all().all():
        raise RuntimeError('Previous controls differ; cross-campaign summary requires a separate audit.')
    if not np.allclose(effects[effects.budget == 6].difference, 0., atol=1e-12):
        raise RuntimeError('A treatment effect occurs before adaptive acquisition.')
    for value in sources.values():
        full = np.asarray(value['prior_mean'])
        level = np.asarray(source_for_arm(value, state['domain'], 'context_level')['prior_mean'])
        for context in state['domain'].contexts:
            mask = np.array([c == context for c, _ in state['domain'].actions])
            if not np.all(level[mask] == level[mask][0]) or not np.isclose(
                    full[mask].mean(), level[mask][0], rtol=0., atol=1e-14):
                raise RuntimeError('Invalid context-level projection.')
    verification = dict(signature=state['signature'], jobs=len(results), source_folds=len(sources),
        queries=sum(len(t['steps']) for j in results for t in j['methods'].values()),
        budget_rows=len(frame), context_rows=len(contexts), source_exclusions_verified=True,
        validation_rescoring_matches=True, previous_controls_exact=True,
        shared_initialization_verified=True, source_level_projection_verified=True,
        initial_treatment_effects_zero=True)
    return state, frame, contexts, summary, paired, verification


def export(state, frame, contexts, summary, paired, verification):
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT/'latest_summary.csv', index=False)
    paired.to_csv(OUT/'paired_effects.csv', index=False)
    last = frame[frame.budget == 24].copy()
    last.groupby(['protocol', 'target', 'method']).agg(context_success=('success_fraction', 'mean'),
        all_contexts_success=('joint_success', 'mean'), mean_J=('J', 'mean'),
        mean_failed_queries=('failed_queries', 'mean')).reset_index().to_csv(OUT/'vehicle_outcomes.csv', index=False)
    misses = contexts[(contexts.budget == 24) & (contexts.method == 'full_cost_prior') & ~contexts.success]
    misses.groupby(['protocol', 'target', 'context']).agg(repeated_optimizer_trials=('seed', 'size'),
        min_relative_gap=('relative_gap', 'min'), max_relative_gap=('relative_gap', 'max')).reset_index().to_csv(
            OUT/'remaining_misses.csv', index=False)
    sensitivity = []
    for tolerance in (.01, .05, .10):
        g = contexts[contexts.budget == 24].copy()
        g['passes'] = g.eligible & (g.relative_gap <= tolerance+1e-12)
        for (protocol, method), block in g.groupby(['protocol', 'method']):
            joint = block.groupby(['target', 'seed']).passes.all()
            sensitivity.append(dict(protocol=protocol, method=method, tolerance=tolerance,
                context_success_rate=float(block.passes.mean()), all_contexts_success_rate=float(joint.mean()),
                retrospective_sensitivity=True))
    pd.DataFrame(sensitivity).to_csv(OUT/'threshold_sensitivity.csv', index=False)
    # Query order varies with optimizer seed; independent measurement data do not.
    noise = state['archive']['configuration']['calibration_seeds']
    last['noise_seed'] = last.seed.map(lambda s: noise[s % len(noise)])
    last.groupby(['protocol', 'target', 'noise_seed', 'method'])[
        ['success_fraction', 'joint_success', 'failed_queries', 'J']].nunique().rename(
        columns=lambda c: c+'_distinct_values').reset_index().to_csv(OUT/'seed_outcome_diversity.csv', index=False)
    old = pd.read_csv(ROOT/'results/tracking_budget_v1/budget_summary.csv')
    cold = old[old.method == 'independent_bo'].copy()
    # Previous no-extra-prior controls match exactly across the archived chain.
    old_controls = pd.read_csv(ROOT/'results/tracking_information_v1/previous_control_audit.csv')
    if not old_controls[['action_trace_match', 'recommendation_trace_match']].all().all():
        raise RuntimeError('Information-ablation controls differ from the original budget campaign.')
    combined = pd.concat([cold.assign(campaign='tracking_budget_v1'),
                          summary.assign(campaign='tracking_cost_shape_v1')], ignore_index=True)
    combined.to_csv(OUT/'budget_comparison.csv', index=False)
    inputs = [ROOT/'results/tracking_diagnostic_v1'/name for name in
              ('run_manifest.json', 'q1_summary.csv', 'q2_context_overlap.csv', 'q3_donor_summary.csv')]
    for tag, names in {
        'tracking_budget_v1': ['run_manifest.json', 'completion.json', 'budget_summary.csv'],
        'tracking_information_v1': ['run_manifest.json', 'completion.json', 'information_summary.csv',
                                    'previous_control_audit.csv'],
        'tracking_cost_shape_v1': ['run_manifest.json', 'completion.json', 'budget_trials.csv',
                                   'budget_contexts.csv', 'source_level_audit.csv'],
    }.items():
        inputs.extend(ROOT/'results'/tag/name for name in names)
    inputs.extend([ROOT/'results/limited_excitation_v1/summary.json',
                   ROOT/'results/cooperative_validation_v1/processed/summary.json',
                   ROOT/'results/cooperative_stage_b_v1/processed/pilot_trials.csv',
                   ROOT.parent/'docs/analysis/corrected_transfer_consolidation.md'])
    verification.update(input_hashes={p.relative_to(ROOT.parent).as_posix(): runner.base.digest(p) for p in inputs},
        analysis_source_sha256=runner.base.digest(Path(__file__)),
        result_commit=runner.base.subprocess.check_output(['git', 'log', '-1', '--format=%H', '--',
            'code/results/tracking_cost_shape_v1/completion.json'], cwd=ROOT.parent, text=True).strip(),
        no_optimization_or_new_simulation=True)
    runner.base.save(OUT/'verification.json', verification)
    plot(combined)


def plot(combined):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods = ['independent_bo', 'catalogue', 'no_extra_prior', 'context_level', 'full_cost_prior']
    labels = ['Cold BO\n24 tests', 'Catalogue\n6 tests', 'Catalogue +\nlocal BO', 'Catalogue +\nlevel prior', 'Catalogue +\nfull prior']
    colors = ['#a6acb4', '#373f47', '#9b7a54', '#377caf', '#258465']
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.1), constrained_layout=True)
    for row, protocol in enumerate(['leave_vehicle_out', 'leave_family_out']):
        block = combined[(combined.protocol == protocol) & (combined.budget == 24)].set_index('method').loc[methods]
        values = [100*block.context_success_rate, block.mean_failed_queries]
        for col, y in enumerate(values):
            ax = axes[row, col]
            ax.bar(range(len(methods)), y, color=colors, width=.65)
            ax.set_xticks(range(len(methods)), labels, fontsize=9)
            ax.set_ylim(0, 112 if col == 0 else 8.3)
            ax.spines[['top', 'right']].set_visible(False)
            ax.yaxis.grid(alpha=.2); ax.set_axisbelow(True)
            for i, value in enumerate(y):
                ax.text(i, value+(1 if col == 0 else .14), f'{value:.1f}', ha='center', fontsize=10)
        axes[row, 0].set_title(protocol.replace('_', ' '), loc='left')
        axes[row, 0].set_ylabel('Contexts within 5% [%]')
        axes[row, 1].set_ylabel('Mean failed target tests')
    fig.suptitle('Historical calibration helps most through the catalogue; cost shape guides remaining search', fontsize=12)
    fig.savefig(OUT/'evidence.png', dpi=160)
    plt.close(fig)


def main():
    state, frame, contexts, summary, paired, verification = verify()
    export(state, frame, contexts, summary, paired, verification)
    print(f'Verified {verification["jobs"]} jobs and {verification["queries"]:,} queries. Tables: {OUT}')


if __name__ == '__main__':
    main()
