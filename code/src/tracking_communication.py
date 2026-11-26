"""Compact eligible-cost relay and matched periodic catalogue synchronization."""
from copy import deepcopy
import time

import numpy as np

from . import tracking_peer_cost as peer
from . import tracking_six_speed as old
from .tracking_budget import incumbents, feasibility_probability
from .tracking_dense_gain import A, B, D, E, METHODS, BONUSES, ExactCostSnapshot
from .tracking_online import features, LocalCostModel, seed_for
from .tracking_online_lcb import acquisition_values
from .tracking_communication_wire import COST, PEER, MODES, exchange


def fresh_state(cfg, clients):
    template = old.fresh_state(cfg, clients)
    template['methods'] = {method: deepcopy(template['methods'][source]) for method, source in zip(METHODS, old.METHODS)}
    for data in template['methods'].values():
        data['network'] = dict(last_contact_round=0, summaries={}, catalogue=[], contacts=[],
            peer_body_bytes=0, model_body_bytes=0, packet_deliveries=0, coordinator_broadcasts=0)
    return template


def visible_histories(data, client, clients):
    """Only delivered peer records, with the target's own full eligible history.

    Sender/trial IDs preserve independent observations at repeated inputs.
    Failed tests never become artificial finite objective observations.
    """
    grouped = [[] for _ in range(clients)]
    for record in sorted(data['network']['summaries'].values(), key=lambda r: (r[0], r[1])):
        sender, trial, action, cost = record
        grouped[sender].append(dict(action=action, eligible=True, J=cost))
    grouped[client] = [deepcopy(o) for o in data['histories'][client] if o['eligible']]
    return grouped


def cached_peer(data, domain, client, context):
    tried = {o['action'] for o in data['histories'][client]}
    records = [r for r in data['network']['catalogue'] if r[0] != client and
        domain.actions[r[2]][0] == context and r[2] not in tried]
    if not records:
        return None, dict.fromkeys(peer.PEER_FIELDS)
    sender, trial, action, mode, cost = min(records,
        key=lambda r: (r[-1], domain.actions[r[2]][1], r[0], r[2]))
    return action, dict(peer_donor=sender, peer_donor_query=trial, peer_donor_J=cost, peer_source_mode=MODES[mode])


def publish(data, method, domain, clients, round_id):
    network = data['network']
    outgoing = []
    for client, history in enumerate(data['histories']):
        records = []
        for context in domain.contexts:
            good = [(trial + 1, o) for trial, o in enumerate(history) if o['eligible'] and domain.actions[o['action']][0] == context]
            if good:
                trial, observation = min(good, key=lambda item: (item[1]['J'], domain.actions[item[1]['action']][1]))
                event = data['events'][(trial - 1) * clients + client]
                records.append((client, trial, observation['action'], MODES.index(event['mode']), observation['J']))
        outgoing.append(records)
    catalogue, peer_stats = exchange(PEER, round_id, outgoing, clients, len(domain.actions))
    if len({(r[0], domain.actions[r[2]][0]) for r in catalogue}) != len(catalogue):
        raise RuntimeError('Repeated catalogue context.')
    network['catalogue'] = [list(record) for record in catalogue]
    cost_stats = None
    if method == D:
        outgoing = [[(client, trial + 1, o['action'], o['J'])
            for trial, o in enumerate(history) if trial >= network['last_contact_round'] and o['eligible']]
            for client, history in enumerate(data['histories'])]
        received, cost_stats = exchange(COST, round_id, outgoing, clients, len(domain.actions))
        for record in received:
            key = f'{record[0]}:{record[1]}'
            if key in network['summaries'] and list(network['summaries'][key]) != list(record):
                raise RuntimeError('Conflicting observation identity.')
            network['summaries'][key] = list(record)
    network['last_contact_round'] = round_id
    log = dict(round=round_id, peer=peer_stats, cost=cost_stats)
    network['contacts'].append(log)
    data['peer_payload_bytes'] += peer_stats['application_bytes']
    data['model_payload_bytes'] += cost_stats['application_bytes'] if cost_stats else 0
    data['payload_bytes'] = data['peer_payload_bytes'] + data['model_payload_bytes']
    network['peer_body_bytes'] += peer_stats['body_bytes']
    network['model_body_bytes'] += cost_stats['body_bytes'] if cost_stats else 0
    network['packet_deliveries'] += peer_stats['packet_deliveries'] + (cost_stats['packet_deliveries'] if cost_stats else 0)
    network['coordinator_broadcasts'] += 1 + bool(cost_stats)
    return log


def plan_round(state, domain, cfg, clients, seed, local_phi, recorded=None, prediction_hook=None, timing_hook=None):
    plans = []
    metadata = {r['id']: r for r in cfg['context_metadata']}
    for method in METHODS:
        data = state['methods'][method]
        acquisition = dict(cfg['acquisition'], confidence_multiplier=BONUSES[method])
        for client, vehicle in enumerate(clients):
            started = time.perf_counter()
            history, previous_round = data['histories'][client], state['round']
            if len(history) != previous_round:
                raise RuntimeError('Non-atomic local history.')
            context = old.program(domain, cfg, seed, vehicle['id'])[previous_round]
            rng = np.random.default_rng(seed_for('acquisition', seed, vehicle['id'], previous_round))
            donor = dict.fromkeys(peer.PEER_FIELDS)
            diagnostics = dict.fromkeys(('gp_fit_parameters', 'gp_fit_count', 'predicted_log_J',
                'predicted_variance', 'eligibility_probability', 'acquisition_score'))
            if previous_round == 0:
                action, mode = domain.action(context, domain.candidates[0]), 'factory_initialization'
            else:
                action, donor = cached_peer(data, domain, client, context)
                mode = 'peer_gain_verification'
                if action is None:
                    mode = 'feasibility_penalized_LCB'
                    if method in (A, B):
                        prior = np.zeros(len(domain.actions)) if method == A else (
                            local_phi @ LocalCostModel(history, local_phi, cfg['model']['ridge']).fit())
                        trace = None if recorded is None else recorded['methods'][method]['events'][previous_round * len(clients) + client]
                        action, parameters, fitted, diagnostics = old._gp_proposal(history, domain, context, prior, cfg,
                            data['gp_parameters'][client], data['last_gp_fit'][client], rng, trace)
                        data['gp_parameters'][client], data['last_gp_fit'][client] = parameters, fitted
                    else:
                        view = visible_histories(data, client, len(clients)) if method == D else [
                            history if j == client else [] for j in range(len(clients))]
                        model = ExactCostSnapshot(view, domain, cfg['gp'], cfg['hierarchical'], method == D)
                        tried = {o['action'] for o in history}
                        allowed = [a for a, (c, _) in enumerate(domain.actions) if c == context and a not in tried]
                        mean, variance = model.predict(client, allowed)
                        probability = feasibility_probability(history, domain, np.full(len(domain.actions), .5), cfg['gp'])[allowed]
                        scores = acquisition_values(mean, variance, probability, acquisition)
                        action = old._minimum(scores, allowed, rng)
                        index = allowed.index(action)
                        diagnostics.update(predicted_log_J=float(mean[index]), predicted_variance=float(variance[index]),
                            eligibility_probability=float(probability[index]), acquisition_score=float(scores[index]))
                        if prediction_hook:
                            prediction_hook(method, client, previous_round, allowed, mean, variance, model)
            last_contact = data['network']['last_contact_round']
            plans.append(dict(method=method, client=client, local_evaluation=previous_round + 1,
                round=previous_round + 1, information_round=previous_round,
                remote_information_round=last_contact, remote_age_rounds=previous_round - last_contact,
                action=action, context=context, candidate=domain.actions[action][1], mode=mode,
                noise_seed=seed_for('six_speed_episode', seed, vehicle['id'], previous_round),
                speed=metadata[context]['speed'], frequency=metadata[context]['frequency'],
                context_visit=1 + sum(domain.actions[o['action']][0] == context for o in history),
                speed_visit=1 + sum(metadata[domain.actions[o['action']][0]]['speed'] == metadata[context]['speed'] for o in history),
                **donor, **diagnostics))
            if timing_hook:
                timing_hook(dict(phase='proposal', method=method, round=previous_round + 1, client=client,
                    seconds=time.perf_counter() - started))
    return plans


def commit_round(state, plans, observations, domain, cfg, clients, interval, timing_hook=None):
    for plan, observation in zip(plans, observations):
        data = state['methods'][plan['method']]
        history = data['histories'][plan['client']]
        previous = incumbents(history, domain)
        if observation['eligible'] != bool(observation['complete'] and observation['feasible'] and observation['integration_resolved']):
            raise RuntimeError('Invalid paid eligibility.')
        if (observation['J'] is None) == observation['eligible'] or (observation['eligible'] and
                (not np.isfinite(observation['J']) or observation['J'] <= 0)):
            raise RuntimeError('Invalid paid objective.')
        if any(o['action'] == plan['action'] for o in history):
            raise RuntimeError('Repeated local gain/context test.')
        history.append(dict(action=plan['action'], **observation))
        current = incumbents(history, domain)
        event = dict(**plan, **observation, fleet_evaluation=len(data['events']) + 1,
            recommendations=current, became_incumbent=previous != current)
        data['events'].append(event)
        budget = len(data['events'])
        if budget in cfg['fleet_budgets']:
            data['snapshots'][str(budget)] = dict(round=plan['round'],
                recommendations=[incumbents(h, domain) for h in data['histories']],
                queries_per_client=[len(h) for h in data['histories']],
                **{k: data[k] for k in ('payload_bytes', 'peer_payload_bytes', 'model_payload_bytes')},
                packet_deliveries=data['network']['packet_deliveries'],
                coordinator_broadcasts=data['network']['coordinator_broadcasts'])
    state['round'] += 1
    for method, data in state['methods'].items():
        started = time.perf_counter()
        before = {k: data[k] for k in ('payload_bytes', 'peer_payload_bytes', 'model_payload_bytes')}
        contact = state['round'] % interval == 0
        if contact:
            publish(data, method, domain, len(clients), state['round'])
        network = data['network']
        data['learning'].append(dict(round=state['round'], contact=contact,
            remote_information_round=network['last_contact_round'],
            eligible_training_counts=[sum(o['eligible'] for o in h) for h in data['histories']],
            peer_payload_bytes=data['peer_payload_bytes'] - before['peer_payload_bytes'],
            model_payload_bytes=data['model_payload_bytes'] - before['model_payload_bytes'],
            payload_bytes=data['payload_bytes'] - before['payload_bytes'],
            cumulative_peer_payload_bytes=data['peer_payload_bytes'], cumulative_model_payload_bytes=data['model_payload_bytes'],
            cumulative_payload_bytes=data['payload_bytes'], cumulative_packet_deliveries=network['packet_deliveries'],
            delivered_eligible_records=len(network['summaries']),
            pending_eligible_records=sum(o['eligible'] for h in data['histories'] for o in h[network['last_contact_round']:]) if method == D else 0))
        if timing_hook and contact:
            timing_hook(dict(phase='communication_codec', method=method, round=state['round'], client=-1,
                seconds=time.perf_counter() - started))


def run_fleet(domain, cfg, clients, seed, interval, observe, checkpoint=None, state=None,
              stop_round=None, recorded=None, prediction_hook=None, timing_hook=None):
    if interval not in (1, 2, 4) or cfg['methods'] != METHODS or cfg['bonus_by_method'] != BONUSES:
        raise ValueError('Invalid matched communication protocol.')
    state = deepcopy(state) if state is not None else fresh_state(cfg, clients)
    limit = cfg['queries_per_client'] if stop_round is None else stop_round
    if not state['round'] <= limit <= cfg['queries_per_client']:
        raise ValueError('Invalid stopping round.')
    local_phi = features(domain, cfg['model'])
    while state['round'] < limit:
        plans = plan_round(state, domain, cfg, clients, seed, local_phi, recorded, prediction_hook, timing_hook)
        observations = observe(plans, state['oracle_cache'])
        if len(observations) != len(plans):
            raise RuntimeError('Oracle observation count differs.')
        commit_round(state, plans, observations, domain, cfg, clients, interval, timing_hook)
        if checkpoint:
            checkpoint(state)
    return state


def validate_state(state, domain, cfg, clients, seed, interval, complete=False, replay=False):
    n, rounds = len(clients), state['round']
    if set(state['methods']) != set(METHODS) or not 0 <= rounds <= cfg['queries_per_client'] or (complete and rounds != cfg['queries_per_client']):
        raise RuntimeError('Invalid method set or completion.')
    initial = None
    for method, data in state['methods'].items():
        if len(data['events']) != n * rounds or len(data['learning']) != rounds or any(len(h) != rounds for h in data['histories']):
            raise RuntimeError('Incomplete atomic round.')
        rebuilt = dict(histories=[[] for _ in clients], events=[], network=dict(last_contact_round=0,
            summaries={}, catalogue=[], contacts=[], peer_body_bytes=0, model_body_bytes=0,
            packet_deliveries=0, coordinator_broadcasts=0), peer_payload_bytes=0, model_payload_bytes=0, payload_bytes=0)
        for r in range(rounds):
            prior = deepcopy(rebuilt)
            for i in range(n):
                event = data['events'][r * n + i]
                context = old.program(domain, cfg, seed, clients[i]['id'])[r]
                action, donor = (domain.action(context, domain.candidates[0]), dict.fromkeys(peer.PEER_FIELDS)) if r == 0 else cached_peer(prior, domain, i, context)
                mode = 'factory_initialization' if r == 0 else 'peer_gain_verification' if action is not None else 'feasibility_penalized_LCB'
                contact_round = (r // interval) * interval
                if (event['method'], event['client'], event['round'], event['local_evaluation'], event['fleet_evaluation'],
                        event['context'], event['information_round'], event['remote_information_round'], event['remote_age_rounds'], event['noise_seed']) != (
                        method, i, r + 1, r + 1, r * n + i + 1, context, r, contact_round, r - contact_round,
                        seed_for('six_speed_episode', seed, clients[i]['id'], r)):
                    raise RuntimeError('Causal schedule or information age differs.')
                if event['mode'] != mode or (action is not None and event['action'] != action) or any(event[k] != v for k, v in donor.items()):
                    raise RuntimeError('Cached peer priority or provenance differs.')
                if domain.actions[event['action']] != (context, event['candidate']) or any(o['action'] == event['action'] for o in rebuilt['histories'][i]):
                    raise RuntimeError('Invalid or repeated local action.')
                rebuilt['histories'][i].append({k: event[k] for k in peer.OBSERVATION_FIELDS})
                rebuilt['events'].append(event)
                if event['recommendations'] != incumbents(rebuilt['histories'][i], domain):
                    raise RuntimeError('Untested or ineligible recommendation.')
                budget = r * n + i + 1
                if budget in cfg['fleet_budgets']:
                    expected = dict(round=r + 1, recommendations=[incumbents(h, domain) for h in rebuilt['histories']],
                        queries_per_client=[len(h) for h in rebuilt['histories']],
                        **{k: rebuilt[k] for k in ('payload_bytes', 'peer_payload_bytes', 'model_payload_bytes')},
                        packet_deliveries=rebuilt['network']['packet_deliveries'],
                        coordinator_broadcasts=rebuilt['network']['coordinator_broadcasts'])
                    if data['snapshots'][str(budget)] != expected:
                        raise RuntimeError('Snapshot includes future communication or recommendations.')
            if (r + 1) % interval == 0:
                publish(rebuilt, method, domain, n, r + 1)
        if rebuilt['histories'] != data['histories'] or rebuilt['network'] != data['network'] or any(rebuilt[k] != data[k] for k in ('payload_bytes', 'peer_payload_bytes', 'model_payload_bytes')):
            raise RuntimeError('Stored histories, packets, cache or byte accounting differ.')
        first = [{k: e[k] for k in peer.OBSERVATION_FIELDS} for e in data['events'][:n]]
        if initial is not None and first != initial:
            raise RuntimeError('Factory observations differ across arms.')
        initial = first
    if replay:
        def recorded_observe(plans, cache):
            observations = []
            for plan in plans:
                event = state['methods'][plan['method']]['events'][(plan['round'] - 1) * n + plan['client']]
                old.assert_equivalent(plan, {k: event[k] for k in plan})
                observations.append({k: event[k] for k in peer.OBSERVATION_FIELDS if k != 'action'})
            return observations
        rebuilt = run_fleet(domain, cfg, clients, seed, interval, recorded_observe,
            stop_round=rounds, recorded=state)
        old.assert_equivalent(rebuilt['methods'], state['methods'])
