"""Online calibration with one factory test and immediate cost-aware acquisition.

Frozen regression, exchange and simulator interfaces are reused from the first
pilot. No historical outcomes or dense reference costs enter this module.
"""
from copy import deepcopy

import numpy as np

from .tracking_budget import Posterior, feasibility_probability, incumbents
from .tracking_online import (METHODS, LocalCostModel, commit_round, features,
    fresh_state, peer_candidate, program, seed_for)


def acquisition_values(mean, variance, probability, cfg):
    """Minimize cost LCB plus a negative-log eligibility penalty, on log J."""
    return (np.asarray(mean)-cfg['confidence_multiplier']*np.sqrt(np.maximum(variance, 0.))
            -cfg['feasibility_penalty']*np.log(np.maximum(probability, cfg['probability_floor'])))


def choose(history, domain, context, prior, gp, acquisition, params, last_fit, rng):
    count = sum(o['eligible'] for o in history)
    optimize = count >= 4 and count-last_fit >= gp['optimize_every']
    posterior = Posterior(domain, history, prior, gp, params, optimize)
    mean, variance = posterior.predict()
    probability = feasibility_probability(history, domain, np.full(len(domain.actions), .5), gp)
    values = acquisition_values(mean, variance, probability, acquisition)
    allowed = np.asarray([c == context for c, _ in domain.actions])
    allowed[[o['action'] for o in history]] = False
    values[~allowed] = np.inf
    best = float(values.min())
    if not np.isfinite(best):
        raise RuntimeError('No unqueried candidate in the scheduled context.')
    tied = np.flatnonzero(values <= best+1e-12*max(1., abs(best)))
    return (int(rng.choice(tied)), posterior.params.tolist(), count if optimize else last_fit,
            'feasibility_penalized_LCB')


def plan_round(state, domain, cfg, clients, seed, scout, phi):
    next_round = state['round']+1
    plans = []
    for method in METHODS:
        data = state['methods'][method]
        for i, client in enumerate(clients):
            history = data['histories'][i]
            n = len(history)
            if n >= cfg['queries_per_client'] or (method == 'single_scout' and i != scout
                                                  and next_round <= cfg['scout_search_queries']):
                continue
            contexts, initial = program(domain, cfg, seed, client['id'])
            context = contexts[n]
            if method == 'single_scout' and i != scout and n < len(domain.contexts):
                candidate = data['scout_catalogue'][domain.contexts.index(context)] or domain.candidates[0]
                action, mode = domain.action(context, candidate), 'scout_gain_verification'
            elif n == 0:
                action, mode = domain.action(context, initial[0]), 'factory_initialization'
            else:
                action = (peer_candidate(data['histories'], domain, i, context)
                          if method == 'online_gain_sharing' else None)
                if action is not None:
                    mode = 'peer_gain_verification'
                else:
                    prior = np.zeros(len(domain.actions))
                    if method == 'local_cost_bo':
                        prior = phi@LocalCostModel(history, phi, cfg['model']['ridge']).fit()
                    elif method in ('federated_cost_bo', 'centralized_cost_bo'):
                        prior = phi@np.asarray(data['weights'])
                    action, params, last_fit, mode = choose(history, domain, context, prior,
                        cfg['gp'], cfg['acquisition'], data['gp_parameters'][i], data['last_gp_fit'][i],
                        np.random.default_rng(seed_for('acquisition', seed, client['id'], n)))
                    data['gp_parameters'][i], data['last_gp_fit'][i] = params, last_fit
            plans.append(dict(method=method, client=i, local_evaluation=n+1, round=next_round,
                action=action, context=context, candidate=domain.actions[action][1], mode=mode,
                noise_seed=seed_for('episode', seed, client['id'], n), information_round=state['round']))
    return plans


def run_fleet(domain, cfg, clients, seed, scout, observe_batch, checkpoint=None, state=None):
    phi = features(domain, cfg['model'])
    state = deepcopy(state) if state is not None else fresh_state(domain, cfg, clients)
    while any(len(m['events']) < len(clients)*cfg['queries_per_client'] for m in state['methods'].values()):
        plans = plan_round(state, domain, cfg, clients, seed, scout, phi)
        if not plans:
            raise RuntimeError('Calibration stopped before the fleet budget was consumed.')
        observations = observe_batch(plans, state['oracle_cache'])
        if len(observations) != len(plans):
            raise RuntimeError('Oracle returned the wrong number of observations.')
        commit_round(state, plans, observations, domain, cfg, clients, scout, phi)
        if checkpoint is not None:
            checkpoint(state)
    return state


def validate_state(state, domain, cfg, clients, seed, scout, complete=False):
    if set(state['methods']) != set(METHODS):
        raise RuntimeError('Missing online method.')
    common_initial = None
    for method, data in state['methods'].items():
        if complete and len(data['events']) != len(clients)*cfg['queries_per_client']:
            raise RuntimeError('Incomplete online calibration.')
        histories = [[] for _ in clients]
        snapshots = {}
        previous_round, last_round = deepcopy(histories), 0
        for n, event in enumerate(data['events']):
            if event['round'] < last_round:
                raise RuntimeError('Events are not ordered by calibration round.')
            if event['round'] != last_round:
                previous_round, last_round = deepcopy(histories), event['round']
            i = event['client']; history = histories[i]
            if event['fleet_evaluation'] != n+1 or event['local_evaluation'] != len(history)+1:
                raise RuntimeError('Invalid online query accounting.')
            if event['information_round'] != event['round']-1 or event['round'] > state['round']:
                raise RuntimeError('A future or current-round observation entered acquisition.')
            expected_round = len(history)+1
            if method == 'single_scout' and i != scout:
                expected_round += cfg['scout_search_queries']
            if event['round'] != expected_round:
                raise RuntimeError('A client performed extra tests or skipped a scheduled calibration round.')
            if method == 'single_scout' and i != scout and event['round'] <= cfg['scout_search_queries']:
                raise RuntimeError('A follower ran before scout calibration finished.')
            contexts, initial = program(domain, cfg, seed, clients[i]['id'])
            if event['context'] != contexts[len(history)] or domain.actions[event['action']] != (
                    event['context'], event['candidate']):
                raise RuntimeError('Context programme or action label changed.')
            if method == 'single_scout' and i != scout and len(history) < len(domain.contexts):
                frozen = incumbents(data['histories'][scout][:cfg['scout_search_queries']], domain)
                expected = frozen[domain.contexts.index(event['context'])] or domain.candidates[0]
                if event['candidate'] != expected or event['mode'] != 'scout_gain_verification':
                    raise RuntimeError('Scout verification changed.')
            elif not history:
                if event['candidate'] != initial[0] or event['mode'] != 'factory_initialization':
                    raise RuntimeError('The single factory initialization changed.')
            else:
                peer = (peer_candidate(previous_round, domain, i, event['context'])
                        if method == 'online_gain_sharing' else None)
                expected_mode = 'peer_gain_verification' if peer is not None else 'feasibility_penalized_LCB'
                if event['mode'] != expected_mode or (peer is not None and event['action'] != peer):
                    raise RuntimeError('Adaptive acquisition or prior-round peer verification changed.')
            if any(o['action'] == event['action'] for o in history):
                raise RuntimeError('Repeated target gain/context query.')
            if event['noise_seed'] != seed_for('episode', seed, clients[i]['id'], len(history)):
                raise RuntimeError('Incorrect client-specific measurement seed.')
            if event['eligible'] != bool(event['complete'] and event['feasible'] and event['integration_resolved']):
                raise RuntimeError('Invalid eligibility label.')
            if (event['J'] is None) == event['eligible'] or (event['eligible'] and (
                    not np.isfinite(event['J']) or event['J'] <= 0)):
                raise RuntimeError('Invalid online objective observation.')
            observation = {k: event[k] for k in ('action', 'eligible', 'J', 'complete', 'feasible',
                                                'integration_resolved', 'substeps')}
            history.append(observation)
            if event['recommendations'] != incumbents(history, domain):
                raise RuntimeError('Unobserved or ineligible gain recommendation.')
            if n+1 in cfg['fleet_budgets']:
                snapshots[str(n+1)] = dict(round=event['round'],
                    recommendations=[incumbents(h, domain) for h in histories],
                    queries_per_client=[len(h) for h in histories])
        for budget, snapshot in snapshots.items():
            if any(data['snapshots'].get(budget, {}).get(k) != v for k, v in snapshot.items()):
                raise RuntimeError('An online budget snapshot contains later information.')
        if set(data['snapshots']) != set(snapshots):
            raise RuntimeError('Unexpected or missing online budget snapshot.')
        # History order must agree with event accounting, including numerical labels.
        for expected, actual in zip(histories, data['histories']):
            if expected != actual:
                raise RuntimeError('Client history differs from its event trace.')
        if method != 'single_scout':
            initial = [[(e['action'], e['noise_seed'], e['J'], e['eligible']) for e in data['events']
                        if e['client'] == i and e['local_evaluation'] == 1]
                       for i in range(len(clients))]
            if common_initial is None:
                common_initial = initial
            elif initial != common_initial:
                raise RuntimeError('Non-scout methods have different initialization observations.')
