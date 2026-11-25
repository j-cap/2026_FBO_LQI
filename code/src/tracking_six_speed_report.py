"""Encountered-context and paid-visit evaluation; no learner reads this module's labels."""
import base64
from html import escape
import io

import numpy as np
import pandas as pd

from .tracking_six_speed import METHODS, PAIRS

LABELS = dict(zip(METHODS, ['A', 'B', 'D256', 'E256']))


def _outcome(context, candidate, refs, validation, tolerance, online_eligible=True):
    ref = refs[context]
    val = validation.get(context, {}).get(candidate) if candidate is not None else None
    eligible = bool(online_eligible and val is not None and val['eligible'])
    available = bool(ref['eligible'] and ref['J'] is not None and ref['J'] > 0)
    gap = float(val['J'] / ref['J'] - 1) if eligible and available else None
    return dict(eligible=eligible, reference_available=available, relative_gap=gap,
                J=float(val['J']) if eligible else None, success=gap is not None and gap <= tolerance)


def evaluate(results, domain, cfg, refs, validation):
    trials, cells, visits, learning = [], [], [], []
    for result in results:
        seed = result['seed']
        for method, data in result['state']['methods'].items():
            common = dict(condition='homogeneous', seed=seed, method=method)
            learning.extend(dict(**common, **row) for row in data['learning'])
            for event in data['events']:
                context = event['context']
                incumbent = event['recommendations'][domain.contexts.index(context)]
                requested = _outcome(context, event['candidate'], refs, validation, cfg['relative_tolerance'], event['eligible'])
                best = _outcome(context, incumbent, refs, validation, cfg['relative_tolerance'])
                fields = {k: v for k, v in event.items() if k not in ('recommendations', 'method', 'client')}
                visits.append(dict(**common, **fields, client=result['clients'][event['client']]['id'],
                    is_factory=event['mode'] == 'factory_initialization',
                    first_context_visit=event['context_visit'] == 1, first_speed_visit=event['speed_visit'] == 1,
                    incumbent_candidate=incumbent, **{'requested_' + k: v for k, v in requested.items()},
                    **{'incumbent_' + k: v for k, v in best.items()}))
            for budget in cfg['fleet_budgets']:
                snapshot = data['snapshots'][str(budget)]
                for i, client in enumerate(result['clients']):
                    events = [e for e in data['events'][:budget] if e['client'] == i]
                    encountered = {e['context'] for e in events}
                    # The programme fixes the denominator independently of success or failures.
                    outcomes = []
                    for context in domain.contexts:
                        if context not in encountered:
                            continue
                        candidate = snapshot['recommendations'][i][domain.contexts.index(context)]
                        outcome = _outcome(context, candidate, refs, validation, cfg['relative_tolerance'])
                        cell = dict(**common, client=client['id'], fleet_budget=budget, context=context,
                            candidate=candidate, local_visits=sum(e['context'] == context for e in events),
                            has_local_eligible_gain=candidate is not None, **outcome)
                        cells.append(cell)
                        outcomes.append(cell)
                    if not outcomes:
                        raise RuntimeError('Declared snapshot has an unvisited client.')
                    complete = all(c['eligible'] for c in outcomes)
                    trials.append(dict(**common, client=client['id'], fleet_budget=budget,
                        actual_client_queries=len(events), encountered_contexts=len(outcomes),
                        encountered_speeds=len({e['speed'] for e in events}),
                        unvisited_contexts=len(domain.contexts) - len(outcomes),
                        context_success=float(np.mean([c['success'] for c in outcomes])),
                        validation_eligible_context_coverage=float(np.mean([c['eligible'] for c in outcomes])),
                        reference_coverage=float(np.mean([c['reference_available'] for c in outcomes])),
                        complete_eligible_on_encountered=complete,
                        complete_eligible_on_full_grid=complete and len(outcomes) == len(domain.contexts),
                        J=float(np.mean([c['J'] for c in outcomes])) if complete else None,
                        ineligible_queries=sum(not e['eligible'] for e in events),
                        unresolved_queries=sum(not e['integration_resolved'] for e in events),
                        **{k: snapshot[k] for k in ('payload_bytes', 'peer_payload_bytes', 'model_payload_bytes')}))
    return tuple(pd.DataFrame(rows) for rows in (trials, cells, visits, learning))


def paired_comparisons(trials):
    comparisons, unit_rows = [], []
    for budget, frame in trials.groupby('fleet_budget', sort=True):
        for method, baseline, role in PAIRS:
            a = frame[frame.method == method].set_index(['seed', 'client']).sort_index()
            b = frame[frame.method == baseline].set_index(['seed', 'client']).sort_index()
            if not a.index.equals(b.index) or not a.encountered_contexts.equals(b.encountered_contexts):
                raise RuntimeError('Counterfactual encountered-context denominators differ.')
            delta = a.context_success - b.context_success
            common = a.complete_eligible_on_encountered & b.complete_eligible_on_encountered
            effects = delta.groupby(level='seed').mean()
            ma = float(a.loc[common, 'J'].mean()) if common.any() else None
            mb = float(b.loc[common, 'J'].mean()) if common.any() else None
            comparisons.append(dict(fleet_budget=budget, method=method, baseline=baseline, role=role,
                programme_seeds=len(effects), matched_client_runs=len(a), success_delta_pp=100 * float(effects.mean()),
                positive_seeds=int((effects > 1e-12).sum()), negative_seeds=int((effects < -1e-12).sum()),
                tied_seeds=int((abs(effects) <= 1e-12).sum()), worst_seed_delta_pp=100 * float(effects.min()),
                eligible_method=int(a.complete_eligible_on_encountered.sum()),
                eligible_baseline=int(b.complete_eligible_on_encountered.sum()), common_eligible_pairs=int(common.sum()),
                paired_mean_J_method=ma, paired_mean_J_baseline=mb,
                paired_J_change_percent=100 * (ma / mb - 1) if ma is not None else None))
            unit_rows.extend(dict(fleet_budget=budget, method=method, baseline=baseline, role=role,
                                 seed=int(seed), success_delta_pp=100 * float(effect)) for seed, effect in effects.items())
    return pd.DataFrame(comparisons), pd.DataFrame(unit_rows)


def visit_tables(visits):
    groups = [('all_requests', np.ones(len(visits), dtype=bool)),
              ('first_context_visit', visits.first_context_visit),
              ('first_context_visit_after_factory', visits.first_context_visit & ~visits.is_factory),
              ('first_speed_visit_after_factory', visits.first_speed_visit & ~visits.is_factory)]
    for number in (2, 3):
        groups.append((f'context_visit_{number}', visits.context_visit == number))
    groups.append(('context_visit_4_plus', visits.context_visit >= 4))
    rows = []
    for label, mask in groups:
        for (method, seed), group in visits[mask].groupby(['method', 'seed'], sort=False):
            rows.append(dict(group=label, method=method, seed=seed, requests=len(group),
                online_eligible_requests=int(group.eligible.sum()), requested_validation_eligible=int(group.requested_eligible.sum()),
                requested_success_percent=100 * float(group.requested_success.mean()),
                incumbent_success_percent=100 * float(group.incumbent_success.mean()),
                incumbent_validation_eligible=int(group.incumbent_eligible.sum()),
                conditional_requested_J=float(group.requested_J.mean()),
                peer_verifications=int((group['mode'] == 'peer_gain_verification').sum()),
                BO_requests=int((group['mode'] == 'feasibility_penalized_LCB').sum())))
    return pd.DataFrame(rows)


def write_report(state, trials, cells, visits, learning, refs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = state['out']
    final = state['cfg']['fleet_budgets'][-1]
    paired, units = paired_comparisons(trials)
    visit_summary = visit_tables(visits)
    summary = trials.groupby(['method', 'fleet_budget'], sort=False).agg(
        client_runs=('client', 'size'), seeds=('seed', 'nunique'),
        context_success_percent=('context_success', lambda x: 100 * x.mean()),
        complete_eligible_on_encountered=('complete_eligible_on_encountered', 'sum'),
        complete_eligible_on_full_grid=('complete_eligible_on_full_grid', 'sum'),
        conditional_mean_J=('J', 'mean'), mean_encountered_contexts=('encountered_contexts', 'mean'),
        mean_unvisited_contexts=('unvisited_contexts', 'mean'),
        mean_ineligible_requests=('ineligible_queries', 'mean'), reference_coverage=('reference_coverage', 'mean')).reset_index()
    counts = cells.groupby(['method', 'fleet_budget'], sort=False).agg(encountered_context_outcomes=('success', 'size'),
        successes=('success', 'sum'), pooled_context_success_percent=('success', lambda x: 100 * x.mean())).reset_index()
    summary = summary.merge(counts, on=['method', 'fleet_budget'], validate='one_to_one')
    modes = visits.groupby(['method', 'seed', 'mode'], sort=False).agg(requests=('eligible', 'size'),
        online_eligible=('eligible', 'sum'), new_incumbents=('became_incumbent', 'sum')).reset_index()
    context = cells[cells.fleet_budget == final].groupby(['method', 'context'], sort=False).agg(
        encounters=('success', 'size'), success_percent=('success', lambda x: 100 * x.mean()),
        eligible=('eligible', 'sum'), mean_local_visits=('local_visits', 'mean')).reset_index()
    frames = dict(budget_trials=trials, budget_contexts=cells, paid_visits=visits, learning_diagnostics=learning,
        online_summary=summary, paired_comparisons=paired, per_seed_comparisons=units,
        visit_summary=visit_summary, query_modes=modes, final_context_outcomes=context,
        empirical_references=pd.DataFrame([dict(context=c, **row) for c, row in refs.items()]))
    for name, frame in frames.items():
        frame.to_csv(out / (name + '.csv'), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for method, color in zip(METHODS, ['#66748a', '#a77a39', '#007d75', '#7362ad']):
        rows = summary[summary.method == method]
        axes[0].plot(rows.fleet_budget, rows.context_success_percent, marker='o', label=LABELS[method], color=color)
        axes[1].plot(rows.fleet_budget, rows.mean_encountered_contexts, marker='o', color=color)
    axes[0].set(ylabel='Success over encountered contexts [%]', ylim=(-2, 102))
    axes[1].set(ylabel='Mean contexts encountered per vehicle', ylim=(0, 18.5))
    for ax in axes:
        ax.set(xlabel='Paid fleet requests', xticks=state['cfg']['fleet_budgets'])
        ax.grid(alpha=.2)
    axes[0].legend(ncol=2)
    fig.savefig(out / 'budget_curves.png', dpi=150)
    image = io.BytesIO()
    fig.savefig(image, format='png', dpi=120)
    plt.close(fig)
    def table(frame):
        frame = frame.copy()
        for col in ('method', 'baseline'):
            if col in frame:
                frame[col] = frame[col].map(LABELS)
        return '<div class="table">' + frame.to_html(index=False, float_format=lambda x: f'{x:.4g}', na_rep='unavailable') + '</div>'
    mode = state['specification']['mode']
    title = 'Six-speed homogeneous pilot' if mode == 'pilot' else 'Six-speed SMOKE — software verification only'
    accuracy = state['specification']['basis_accuracy']
    html = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    html += '<title>' + title + '</title><style>body{font:16px/1.6 system-ui;max-width:1250px;margin:35px auto;padding:0 22px;color:#203747}'
    html += 'table{border-collapse:collapse;font-size:12px}td,th{padding:8px;border:1px solid #d0dce2}.table{overflow-x:auto;margin:18px 0}'
    html += 'h2{margin-top:35px}img{max-width:100%}.note{background:#edf6f2;padding:18px;border-left:4px solid #007d75}</style></head><body>'
    html += '<h1>' + title + '</h1><p>Six speeds 20–30 m/s; three frequencies; one externally assigned context per trial. '
    html += 'Speeds are sampled independently with replacement. Frequencies use balanced shuffled blocks. All four methods '
    html += 'use the same paid peer-first rule, one charged factory test, cold histories and the same local cap.</p>'
    html += '<p class="note"><strong>Exploratory pilot, not confirmation.</strong> Three programme/noise seeds on one compact plant '
    html += 'do not establish heterogeneous-fleet performance. Main scores first average encountered-context success within each vehicle, '
    html += 'then average vehicles and seeds equally. Unvisited contexts are reported separately. Missing or ineligible recommendations '
    html += 'within encountered contexts remain failures. Raw pooled context rates use a different weighting and are also reported.</p>'
    html += '<h2>Primary: D256 versus A</h2>' + table(paired[(paired.fleet_budget == final) & (paired.role == 'primary')])
    html += '<h2>Secondary: D256 versus B</h2>' + table(paired[(paired.fleet_budget == final) & (paired.role == 'secondary')])
    html += '<h2>Matched sharing control: D256 versus E256</h2>' + table(paired[(paired.fleet_budget == final) & (paired.role == 'sharing_control')])
    html += '<p>Success changes are percentage points. Paired cost changes use the common eligible complete policies over encountered '
    html += 'contexts. Eligibility and paired denominators remain explicit. No three-seed confidence interval or confirmation flag is assigned.</p>'
    html += '<img alt="Success over encountered contexts and mean context coverage by budget" src="data:image/png;base64,' + base64.b64encode(image.getvalue()).decode() + '">'
    html += '<p>The encountered-context denominator grows with budget, so these curves are not fixed-grid learning curves.</p>'
    html += '<h2>Final outcomes and coverage</h2>' + table(summary[summary.fleet_budget == final])
    html += '<h2>First and subsequent visits</h2>' + table(visit_summary)
    html += '<p>Requested-controller success requires both the paid online test and both held-out repeats to be eligible. '
    html += 'Incumbent success evaluates the locally verified recommendation after that visit. These are different outcomes; '
    html += 'a failed exploratory query can retain a successful earlier incumbent. Factory tests are included unless the group explicitly excludes them.</p>'
    html += '<h2>Peer channel and BO allocation</h2>' + table(modes)
    html += '<h2>Per-seed comparisons</h2>' + table(units[units.fleet_budget == final])
    html += '<h2>Final encountered speed/frequency contexts</h2>' + table(context)
    html += '<h2>Representation and audit scope</h2><p>Public grid: 1,278 inputs; feature width: 257. Relative kernel Frobenius error '
    html += f'{accuracy["relative_kernel_Frobenius_error"]:.3%}; maximum relative diagonal error {accuracy["max_relative_diagonal_error"]:.3%}. '
    html += 'The earlier targets are 2% and 5%. Rank 256 is retained; its slight diagonal-target miss is reported, not tuned away.</p>'
    html += '<p>Only scheduled-context predictions are evaluated; the Gaussian messages, evidence mixture and total variance are unchanged. '
    html += 'Reference communication counts the same per-component float64 matrices/vectors and per-recipient broadcasts. Public-basis '
    html += f'distribution ({len(state["clients"])*state["phi"].nbytes:,} bytes per method/fleet if distributed to every client) is separate. '
    html += 'This is not a communication experiment. Unused end-of-final-round training is excluded from decision snapshots.</p>'
    html += '<p>All paid choices are audited before loading evaluator outcomes. A/B audits restore logged pre-query GP parameters '
    html += 'and refit counters; they do not claim identical fresh optimization across numerical platforms. D/E predictions and '
    html += 'choices are reconstructed from their own earlier histories. Online failures and unresolved attempts remain charged.</p>'
    html += '<p>Dense calibration and held-out validation are fresh evaluator-only work. All methods share a held-out bank for the common '
    html += 'physical plant, conditional on the stated programme/noise replication scope. Source hashes, pinned basis, programmes, '
    html += 'online raw simulator records and evaluator batches accompany this report.</p>'
    html += '<p>Configuration: <code>' + escape(state['cfg']['tag']) + '</code>.</p></body></html>'
    (out / 'report.html').write_text(html, encoding='utf-8')
