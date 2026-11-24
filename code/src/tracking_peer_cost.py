"""Online fleet calibration with common peer priority and three BO mean models.

All information comes from this method's paid earlier-round observations.
The frozen simulator is injected; reference/validation archives are not inputs.
"""
from copy import deepcopy

import numpy as np

from .tracking_budget import incumbents
from .tracking_online import (LocalCostModel, features, federated_fit,
    optimization_gap, peer_candidate, program, seed_for)
from .tracking_online_lcb import choose


METHODS = ['peer_gp_bo', 'peer_local_cost_bo', 'peer_federated_cost_bo']
PEER_FIELDS = ('peer_donor', 'peer_donor_query', 'peer_donor_J', 'peer_source_mode')
OBSERVATION_FIELDS = ('action', 'eligible', 'J', 'complete', 'feasible',
                      'integration_resolved', 'substeps')


def fresh_state(domain, cfg, clients):
    if not cfg['methods'] or not set(cfg['methods']).issubset(METHODS):
        raise ValueError('Unknown peer-assisted method.')
    params = np.log([*cfg['gp']['length_scales'], cfg['gp']['amplitude']]).tolist()
    return dict(round=0, oracle_cache={}, methods={method: dict(
        histories=[[] for _ in clients], gp_parameters=[params.copy() for _ in clients],
        last_gp_fit=[0]*len(clients), weights=[0.]*(cfg['model']['features']+1),
        events=[], snapshots={}, learning=[], payload_bytes=0,
        peer_payload_bytes=0, model_payload_bytes=0) for method in cfg['methods']})


def peer_proposal(histories, events, domain, client, context):
    """Use the frozen peer rule; extra fields are audit provenance, not GP data."""
    action = peer_candidate(histories, domain, client, context)
    if action is None:
        return None, dict.fromkeys(PEER_FIELDS)
    donors = []
    for j, history in enumerate(histories):
        if j == client:
            continue
        good = [(q, o) for q, o in enumerate(history) if o['eligible']
                and domain.actions[o['action']][0] == context]
        if good:
            q, best = min(good, key=lambda item: (item[1]['J'], domain.actions[item[1]['action']][1]))
            if best['action'] == action:
                donors.append((best['J'], j, q+1))
    cost, donor, query = min(donors)
    source = events[(query-1)*len(histories)+donor]
    return action, dict(peer_donor=donor, peer_donor_query=query,
                        peer_donor_J=cost, peer_source_mode=source['mode'])


def plan_round(state, domain, cfg, clients, seed, phi):
    plans = []
    for method in cfg['methods']:
        data = state['methods'][method]
        for i, client in enumerate(clients):
            history = data['histories'][i]
            n = len(history)
            if n >= cfg['queries_per_client']:
                continue
            contexts, initial = program(domain, cfg, seed, client['id'])
            context = contexts[n]
            metadata = dict.fromkeys(PEER_FIELDS)
            if n == 0:
                action, mode = domain.action(context, initial[0]), 'factory_initialization'
            else:
                action, metadata = peer_proposal(data['histories'], data['events'], domain, i, context)
                if action is not None:
                    mode = 'peer_gain_verification'
                else:
                    prior = np.zeros(len(domain.actions))
                    if method == 'peer_local_cost_bo':
                        prior = phi@LocalCostModel(history, phi, cfg['model']['ridge']).fit()
                    elif method == 'peer_federated_cost_bo':
                        prior = phi@np.asarray(data['weights'])
                    action, params, last_fit, mode = choose(history, domain, context, prior,
                        cfg['gp'], cfg['acquisition'], data['gp_parameters'][i], data['last_gp_fit'][i],
                        np.random.default_rng(seed_for('acquisition', seed, client['id'], n)))
                    data['gp_parameters'][i], data['last_gp_fit'][i] = params, last_fit
            plans.append(dict(method=method, client=i, local_evaluation=n+1, round=state['round']+1,
                action=action, context=context, candidate=domain.actions[action][1], mode=mode,
                noise_seed=seed_for('episode', seed, client['id'], n),
                information_round=state['round'], **metadata))
    return plans


def commit_round(state, plans, observations, domain, cfg, clients, phi):
    for plan, observation in zip(plans, observations):
        data = state['methods'][plan['method']]
        history = data['histories'][plan['client']]
        previous = incumbents(history, domain)
        history.append(dict(action=plan['action'], **observation))
        recommendations = incumbents(history, domain)
        k = domain.contexts.index(plan['context'])
        event = dict(**plan, **observation, fleet_evaluation=len(data['events'])+1,
            recommendations=recommendations, became_incumbent=previous[k] != recommendations[k])
        data['events'].append(event)
        budget = len(data['events'])
        if budget in cfg['fleet_budgets']:
            data['snapshots'][str(budget)] = dict(round=plan['round'],
                recommendations=[incumbents(h, domain) for h in data['histories']],
                queries_per_client=[len(h) for h in data['histories']],
                **{key: data[key] for key in ('payload_bytes', 'peer_payload_bytes', 'model_payload_bytes')})
    state['round'] += 1
    for method in cfg['methods']:
        data = state['methods'][method]
        # Same incumbent-pair publication/broadcast estimate in every method.
        pairs = sum(sum(c is not None for c in incumbents(h, domain)) for h in data['histories'])
        peer_bytes = pairs*12*(len(clients)+1)
        counts = [sum(o['eligible'] for o in h) for h in data['histories']]
        model_bytes, gap = 0, None
        if method == 'peer_federated_cost_bo':
            models = [LocalCostModel(h, phi, cfg['model']['ridge']) for h in data['histories']]
            weights, model_bytes = federated_fit(models, np.asarray(data['weights']), cfg['model'])
            data['weights'] = weights.tolist()
            gap = optimization_gap(models, weights)
        data['peer_payload_bytes'] += peer_bytes
        data['model_payload_bytes'] += model_bytes
        data['payload_bytes'] += peer_bytes+model_bytes
        data['learning'].append(dict(round=state['round'], eligible_training_counts=counts,
            peer_payload_bytes=peer_bytes, model_payload_bytes=model_bytes,
            payload_bytes=peer_bytes+model_bytes, cumulative_payload_bytes=data['payload_bytes'],
            cumulative_peer_payload_bytes=data['peer_payload_bytes'],
            cumulative_model_payload_bytes=data['model_payload_bytes'], optimization_gap=gap))


def run_fleet(domain, cfg, clients, seed, observe_batch, checkpoint=None, state=None):
    phi = features(domain, cfg['model'])
    state = deepcopy(state) if state is not None else fresh_state(domain, cfg, clients)
    if state['round']:
        validate_state(state, domain, cfg, clients, seed)
    while state['round'] < cfg['queries_per_client']:
        plans = plan_round(state, domain, cfg, clients, seed, phi)
        observations = observe_batch(plans, state['oracle_cache'])
        if len(observations) != len(plans):
            raise RuntimeError('Oracle returned the wrong number of observations.')
        commit_round(state, plans, observations, domain, cfg, clients, phi)
        if checkpoint is not None:
            checkpoint(state)
    return state


def validate_state(state, domain, cfg, clients, seed, complete=False):
    nclients = len(clients)
    if set(state['methods']) != set(cfg['methods']) or not 0 <= state['round'] <= cfg['queries_per_client']:
        raise RuntimeError('Invalid peer-assisted methods or round count.')
    if complete and state['round'] != cfg['queries_per_client']:
        raise RuntimeError('Incomplete peer-assisted calibration.')
    common_initial = None
    for method, data in state['methods'].items():
        if len(data['events']) != state['round']*nclients or len(data['histories']) != nclients:
            raise RuntimeError('Non-atomic round or invalid local/fleet query accounting.')
        histories = [[] for _ in clients]
        snapshots, previous_round = {}, None
        for number, event in enumerate(data['events']):
            round_id, i = divmod(number, nclients)
            if i == 0:
                previous_round = deepcopy(histories)
            history = histories[i]
            if (event['method'], event['client'], event['local_evaluation'], event['round'],
                event['fleet_evaluation'], event['information_round']) != (
                    method, i, round_id+1, round_id+1, number+1, round_id):
                raise RuntimeError('Query order or prior-round information barrier changed.')
            contexts, initial = program(domain, cfg, seed, clients[i]['id'])
            if event['context'] != contexts[round_id] or domain.actions[event['action']] != (
                    event['context'], event['candidate']):
                raise RuntimeError('Scheduled context or candidate changed.')
            if round_id == 0:
                action, metadata = domain.action(event['context'], initial[0]), dict.fromkeys(PEER_FIELDS)
                mode = 'factory_initialization'
            else:
                action, metadata = peer_proposal(previous_round, data['events'], domain, i, event['context'])
                mode = 'peer_gain_verification' if action is not None else 'feasibility_penalized_LCB'
            if event['mode'] != mode or (action is not None and event['action'] != action):
                raise RuntimeError('Common peer priority or single factory initialization changed.')
            if any(event.get(key) != value for key, value in metadata.items()):
                raise RuntimeError('Peer donor provenance differs from prior-round publications.')
            if any(o['action'] == event['action'] for o in history):
                raise RuntimeError('Repeated local gain/context query.')
            if event['noise_seed'] != seed_for('episode', seed, clients[i]['id'], round_id):
                raise RuntimeError('Client/query noise pairing changed.')
            if event['eligible'] != bool(event['complete'] and event['feasible'] and event['integration_resolved']):
                raise RuntimeError('Invalid eligibility label.')
            if (event['J'] is None) == event['eligible'] or (event['eligible'] and (
                    not np.isfinite(event['J']) or event['J'] <= 0)):
                raise RuntimeError('Invalid eligible objective observation.')
            old = incumbents(history, domain)
            history.append({key: event[key] for key in OBSERVATION_FIELDS})
            recommendations = incumbents(history, domain)
            k = domain.contexts.index(event['context'])
            if event['recommendations'] != recommendations or event['became_incumbent'] != (old[k] != recommendations[k]):
                raise RuntimeError('Unobserved or ineligible recommendation.')
            if number+1 in cfg['fleet_budgets']:
                snapshots[str(number+1)] = dict(round=round_id+1,
                    recommendations=[incumbents(h, domain) for h in histories],
                    queries_per_client=[len(h) for h in histories])
        if histories != data['histories'] or set(snapshots) != set(data['snapshots']):
            raise RuntimeError('History or budget snapshot differs from the event trace.')
        peer_total, model_total = 0, 0
        if len(data['learning']) != state['round']:
            raise RuntimeError('Missing learning round.')
        for r, log in enumerate(data['learning'], 1):
            prefix = [h[:r] for h in histories]
            counts = [sum(o['eligible'] for o in h) for h in prefix]
            peer_bytes = sum(sum(c is not None for c in incumbents(h, domain)) for h in prefix)*12*(nclients+1)
            model_bytes = (cfg['model']['server_rounds']*sum(c > 0 for c in counts)*2*
                           (cfg['model']['features']+1)*8 if method == 'peer_federated_cost_bo' else 0)
            peer_total += peer_bytes; model_total += model_bytes
            expected = dict(round=r, eligible_training_counts=counts, peer_payload_bytes=peer_bytes,
                model_payload_bytes=model_bytes, payload_bytes=peer_bytes+model_bytes,
                cumulative_peer_payload_bytes=peer_total, cumulative_model_payload_bytes=model_total,
                cumulative_payload_bytes=peer_total+model_total)
            if any(log.get(key) != value for key, value in expected.items()):
                raise RuntimeError('Training counts or peer/model payload accounting changed.')
        for budget, expected in snapshots.items():
            snapshot = data['snapshots'][budget]
            prior = data['learning'][expected['round']-2] if expected['round'] > 1 else {}
            expected.update(payload_bytes=prior.get('cumulative_payload_bytes', 0),
                peer_payload_bytes=prior.get('cumulative_peer_payload_bytes', 0),
                model_payload_bytes=prior.get('cumulative_model_payload_bytes', 0))
            if any(snapshot.get(key) != value for key, value in expected.items()):
                raise RuntimeError('Budget snapshot contains later information or incorrect payload.')
        if (data['peer_payload_bytes'], data['model_payload_bytes'], data['payload_bytes']) != (
                peer_total, model_total, peer_total+model_total):
            raise RuntimeError('Total peer/model payload accounting changed.')
        initial = [{key: event[key] for key in OBSERVATION_FIELDS if key != 'action'}
                   for event in data['events'][:nclients]]
        if common_initial is None:
            common_initial = initial
        elif initial != common_initial:
            raise RuntimeError('Methods have different factory observations.')
