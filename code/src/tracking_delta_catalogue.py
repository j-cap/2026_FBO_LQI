"""Lossless catalogue deltas with the frozen matched BO planning logic."""
from copy import deepcopy
import time

import numpy as np

from . import tracking_peer_cost as peer
from . import tracking_six_speed as old
from . import tracking_communication as frozen
from .tracking_budget import incumbents
from .tracking_dense_gain import D, METHODS, BONUSES
from .tracking_online import features, seed_for
from .tracking_communication_wire import COST, exchange
from .tracking_delta_cache import empty_cache, deliver

cached_peer = frozen.cached_peer
visible_histories = frozen.visible_histories


def fresh_state(cfg, clients):
    state = frozen.fresh_state(cfg, clients)
    for data in state['methods'].values():
        data['network'].update(empty_cache(len(clients)))
    return state


def publish(data, method, domain, clients, round_id):
    network = data['network']
    peer_stats = deliver(data, domain, clients, round_id)
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


plan_round = frozen.plan_round


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
        rebuilt['network'].update(empty_cache(n))
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
