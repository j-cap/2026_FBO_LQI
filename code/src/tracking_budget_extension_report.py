"""Budget curves with an immutable context denominator and discovery diagnostics."""
import base64
import io
import json

import numpy as np
import pandas as pd

from .tracking_dense_gain import A, B, D, E, METHODS
from .tracking_six_speed_report import _outcome
from .tracking_delta_report import bootstrap_effects, document, table

LABELS = dict(zip(METHODS, ['A', 'B', 'D', 'E']))


def evaluate(state, results, validation, refs):
    cells, trials, traffic = [], [], []
    domain, cfg = state['domain'], state['cfg']
    for result in results:
        seed = result['seed']
        for method, data in result['state']['methods'].items():
            for queries in state['specification']['checkpoints']:
                budget = queries * len(state['clients'])
                snapshot = data['snapshots'][str(budget)]
                events = data['events'][:budget]
                discovered, local_discovered = set(), set()
                for event in events:
                    outcome = _outcome(event['context'], event['candidate'], refs, validation,
                        cfg['relative_tolerance'], event['eligible'])
                    if outcome['success']:
                        discovered.add(event['context'])
                        local_discovered.add((event['client'], event['context']))
                common = dict(seed=seed, method=method, queries=queries, fleet_budget=budget)
                traffic.append(dict(**common,
                    **{k: snapshot[k] for k in ('payload_bytes', 'peer_payload_bytes', 'model_payload_bytes', 'packet_deliveries')},
                    distinct_inputs_tried=len({e['action'] for e in events}),
                    distinct_eligible_inputs=len({e['action'] for e in events if e['eligible']}),
                    paid_BO_requests=sum(e['mode'] == 'feasibility_penalized_LCB' for e in events),
                    paid_peer_requests=sum(e['mode'] == 'peer_gain_verification' for e in events),
                    failed_requests=sum(not e['eligible'] for e in events)))
                for i, client in enumerate(state['clients']):
                    programme = cfg['programmes'][str(seed)][client['id']]
                    fixed, encountered = set(programme[:24]), set(programme[:queries])
                    own_events = [e for e in events if e['client'] == i]
                    for scope, contexts in [('fixed_24_contexts', fixed), ('encountered_by_budget', encountered), ('full_18_contexts', set(domain.contexts))]:
                        outcomes = []
                        for context in domain.contexts:
                            if context not in contexts:
                                continue
                            candidate = snapshot['recommendations'][i][domain.contexts.index(context)]
                            outcome = _outcome(context, candidate, refs, validation, cfg['relative_tolerance'])
                            row = dict(**common, client=client['id'], scope=scope, context=context, candidate=candidate,
                                encountered=context in encountered, locally_verified=candidate is not None,
                                local_ever_near_reference=(i, context) in local_discovered,
                                fleet_ever_near_reference=context in discovered, **outcome)
                            cells.append(row)
                            outcomes.append(row)
                        complete = all(row['eligible'] for row in outcomes)
                        trials.append(dict(**common, client=client['id'], scope=scope, contexts=len(outcomes),
                            context_success=float(np.mean([row['success'] for row in outcomes])),
                            eligible_coverage=float(np.mean([row['eligible'] for row in outcomes])),
                            local_ever_success=float(np.mean([row['local_ever_near_reference'] for row in outcomes])),
                            fleet_ever_success=float(np.mean([row['fleet_ever_near_reference'] for row in outcomes])),
                            complete_eligible=complete, J=float(np.mean([row['J'] for row in outcomes])) if complete else None,
                            failed_requests=sum(not e['eligible'] for e in own_events),
                            encountered_contexts=len(encountered), unvisited_contexts=len(domain.contexts) - len(encountered)))
    return pd.DataFrame(cells), pd.DataFrame(trials), pd.DataFrame(traffic)


def comparisons(trials):
    pairs, seeds = [], []
    fixed = trials[trials.scope == 'fixed_24_contexts']
    for queries, group in fixed.groupby('queries', sort=True):
        for baseline in (A, B, E):
            target, control = (group[group.method == m].set_index(['seed', 'client']).sort_index() for m in (D, baseline))
            if not target.index.equals(control.index) or not target.contexts.equals(control.contexts):
                raise RuntimeError('Method comparison contexts differ.')
            effects = (target.context_success - control.context_success).groupby('seed').mean()
            common = target.complete_eligible & control.complete_eligible
            jt, jc = (float(f.loc[common, 'J'].mean()) if common.any() else None for f in (target, control))
            pairs.append(dict(queries=queries, method=D, baseline=baseline,
                success_delta_pp=100 * float(effects.mean()), positive_programmes=int((effects > 1e-12).sum()),
                negative_programmes=int((effects < -1e-12).sum()), common_complete_policies=int(common.sum()),
                paired_J_change_percent=100 * (jt / jc - 1) if jt is not None else None))
            seeds.extend(dict(queries=queries, method=D, baseline=baseline, seed=int(seed),
                success_delta_pp=100 * float(effect)) for seed, effect in effects.items())
    changes = []
    for method, group in fixed.groupby('method', sort=False):
        reference = group[group.queries == 24].set_index(['seed', 'client']).sort_index()
        for queries in sorted(set(group.queries) - {24}):
            target = group[group.queries == queries].set_index(['seed', 'client']).sort_index()
            if not target.index.equals(reference.index) or not target.contexts.equals(reference.contexts):
                raise RuntimeError('Budget comparison contexts differ.')
            effects = (target.context_success - reference.context_success).groupby('seed').mean()
            changes.extend(dict(method=method, queries=queries, baseline_queries=24, seed=int(seed),
                success_delta_pp=100 * float(effect)) for seed, effect in effects.items())
    return pd.DataFrame(pairs), pd.DataFrame(seeds), pd.DataFrame(changes)


def verify_24_checkpoint(state, trials):
    original = pd.read_csv(state['source_state']['out'] / 'budget_trials.csv').query('interval == 1 and fleet_budget == 216')
    actual = trials[(trials.scope == 'fixed_24_contexts') & (trials.queries == 24)]
    original = original[original.seed.isin(actual.seed.unique())]
    keys = ['seed', 'client', 'method']
    old, new = (f.sort_values(keys).reset_index(drop=True) for f in (original, actual))
    if not old[keys].equals(new[keys]):
        raise RuntimeError('The 24-query comparison units differ.')
    np.testing.assert_array_equal(old.encountered_contexts, new.contexts)
    np.testing.assert_allclose(old.context_success, new.context_success, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(old.validation_eligible_context_coverage, new.eligible_coverage, rtol=1e-12, atol=1e-12)


def write_report(state, results, validation, refs, audit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = state['out']
    cells, trials, traffic = evaluate(state, results, validation, refs)
    verify_24_checkpoint(state, trials)
    summary = trials.groupby(['scope', 'method', 'queries'], sort=False).agg(
        client_runs=('client', 'size'), programmes=('seed', 'nunique'),
        context_success_percent=('context_success', lambda x: 100 * x.mean()),
        eligible_coverage_percent=('eligible_coverage', lambda x: 100 * x.mean()),
        local_ever_success_percent=('local_ever_success', lambda x: 100 * x.mean()),
        fleet_ever_success_percent=('fleet_ever_success', lambda x: 100 * x.mean()),
        complete_eligible_policies=('complete_eligible', 'sum'), failed_paid_requests=('failed_requests', 'sum'),
        conditional_mean_J=('J', 'mean'), mean_encountered_contexts=('encountered_contexts', 'mean')).reset_index()
    pairs, seed_pairs, changes = comparisons(trials)
    uncertainty = bootstrap_effects(seed_pairs, state['requested'], ['queries', 'method', 'baseline'])
    change_uncertainty = bootstrap_effects(changes, state['requested'], ['method', 'queries', 'baseline_queries'])
    metadata = pd.DataFrame(state['cfg']['context_metadata'])[['id', 'frequency', 'speed']]
    context = cells.merge(metadata, left_on='context', right_on='id', validate='many_to_one')
    frequency_units = context.groupby(['scope', 'method', 'queries', 'seed', 'client', 'frequency'], sort=False).agg(
        success=('success', 'mean'), coverage=('eligible', 'mean')).reset_index()
    frequency = frequency_units.groupby(['scope', 'method', 'queries', 'frequency'], sort=False).agg(
        success_percent=('success', lambda x: 100 * x.mean()), eligible_coverage_percent=('coverage', lambda x: 100 * x.mean())).reset_index()
    gaps = []
    fixed = cells[cells.scope == 'fixed_24_contexts']
    for tolerance in (.05, .1, .2, .5, 1.):
        frame = fixed.assign(hit=fixed.eligible & fixed.reference_available & (fixed.relative_gap <= tolerance))
        unit = frame.groupby(['method', 'queries', 'seed', 'client']).hit.mean().reset_index()
        for (method, queries), group in unit.groupby(['method', 'queries'], sort=False):
            gaps.append(dict(method=method, queries=queries, diagnostic_tolerance=tolerance,
                success_percent=100 * float(group.hit.mean()), declared_primary=tolerance == .05))
    frames = dict(context_outcomes=cells, client_outcomes=trials, budget_summary=summary,
        paired_comparisons=pairs, per_programme_comparisons=seed_pairs,
        programme_bootstrap=uncertainty, changes_from_24=changes, change_programme_bootstrap=change_uncertainty,
        communication_and_query_allocation=traffic, frequency_units=frequency_units, frequency_summary=frequency,
        gap_threshold_diagnostics=pd.DataFrame(gaps),
        empirical_references=pd.DataFrame([dict(context=c, **r) for c, r in refs.items()]))
    for name, frame in frames.items():
        frame.to_csv(out / (name + '.csv'), index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    colors = dict(zip(METHODS, ['#64748b', '#b58136', '#007d75', '#8265ad']))
    for method in METHODS:
        s = summary[(summary.scope == 'fixed_24_contexts') & (summary.method == method)].sort_values('queries')
        axes[0].plot(s.queries, s.context_success_percent, marker='o', color=colors[method], label=LABELS[method])
        s = summary[(summary.scope == 'full_18_contexts') & (summary.method == method)].sort_values('queries')
        axes[1].plot(s.queries, s.eligible_coverage_percent, marker='o', color=colors[method])
    for frequency_value, color in zip((.1, .3, .6), ['#a7772d', '#667d9d', '#007d75']):
        s = frequency[(frequency.scope == 'fixed_24_contexts') & (frequency.method == D) & (frequency.frequency == frequency_value)].sort_values('queries')
        axes[2].plot(s.queries, s.success_percent, marker='o', label=f'{frequency_value:g} Hz', color=color)
    axes[0].set(ylabel='Success on original context sets [%]')
    axes[1].set(ylabel='Eligible coverage on all 18 contexts [%]')
    axes[2].set(ylabel='D success by frequency [%]')
    for ax in axes:
        ax.set(xlabel='Total paid tests per client/arm', xticks=state['specification']['checkpoints'], ylim=(-2, 102))
        ax.grid(alpha=.2)
    axes[0].legend(ncol=4, frameon=False)
    axes[2].legend(frameon=False)
    fig.savefig(out / 'budget_curves.png', dpi=150)
    image = io.BytesIO()
    fig.savefig(image, format='png', dpi=120)
    plt.close(fig)
    body = '<p>One continuation per original every-round programme. The initial 24 tests, GP fit state, paid observations, lossless caches and communication counters are retained exactly. The factory test is included in every total budget.</p>'
    body += '<p>Primary budget curves use each client\'s original 24-test context set. Unvisited full-grid contexts and evolving encountered sets are separate views. Missing or ineligible gains remain failures. All models, priors, beta and peer-first rules stay fixed.</p>'
    body += '<img alt="Fixed-context budget curves and coverage" src="data:image/png;base64,' + base64.b64encode(image.getvalue()).decode() + '">'
    body += '<h2>Same task at every budget</h2>' + table(summary[summary.scope == 'fixed_24_contexts'])
    body += '<h2>D versus A, then B and matched E</h2>' + table(pairs) + table(uncertainty)
    body += '<h2>Changes from 24 tests</h2>' + table(change_uncertainty)
    body += '<p>Bootstrap intervals resample programme means, not clients. They are descriptive on the existing programme cohort. This continuation is not a fresh independent confirmation. Common-complete cost denominators remain explicit.</p>'
    body += '<h2>All 18 contexts</h2>' + table(summary[summary.scope == 'full_18_contexts'])
    body += '<h2>Evolving encountered-context metric</h2>' + table(summary[summary.scope == 'encountered_by_budget'])
    body += '<p>The original 24-test success metric is reproduced exactly. The evolving-context denominator can grow later, so its changes are not the fixed-task learning curve.</p>'
    body += '<h2>Discovery versus local verification</h2><p>Local-ever and fleet-ever success are retrospective held-out oracle diagnostics of gains already tested by that budget. They never guide a proposal. Their gap from the selected recommendation separates discovery, local verification and selection limitations.</p>'
    body += '<h2>Frequency detail</h2>' + table(frequency[frequency.scope == 'fixed_24_contexts'])
    body += '<h2>Traffic and query allocation per programme</h2>' + table(traffic)
    body += '<p>Checkpoint traffic includes every contact available to its decisions and excludes the publication after its final decision. That publication becomes useful when continuation resumes. Identical D fits are memoized within each simulated round; wall times are not a per-client CPU benchmark.</p>'
    body += '<h2>Evaluation provenance</h2><p>The original four held-out noise seeds and archived candidate rows are reused only by the evaluator, after live decisions finish. Newly encountered gain/context pairs receive missing repeats under those same seeds. No new dense calibration bank is created. All failed queries remain charged.</p>'
    body += '<p>Prefix replay: <a href="../' + audit['out'].name + '/completion.json">verification</a>. Signature: <code>' + state['signature'] + '</code>.</p>'
    title = '25/50/100-test matched budget continuation' if state['specification']['mode'] != 'smoke' else 'Budget continuation SMOKE: software verification only'
    (out / 'report.html').write_text(document(title, body), encoding='utf-8')
    (out / 'budget_readout.json').write_text(json.dumps(dict(confirmation_claim=False,
        primary_scope='fixed_24_contexts', checkpoints=state['specification']['checkpoints'],
        original_24_scores_reproduced=True, prefix_gate_signature=audit['audit_signature'],
        next_step='Analyse absolute quality and D/A/B/E budget effects before choosing another experiment.'), indent=2) + '\n')
    return frames
