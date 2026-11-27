"""Prefix-preserving extension of the frozen every-round lossless BO policy."""
from copy import deepcopy
import time

import numpy as np

from . import tracking_peer_cost as peer
from . import tracking_six_speed as old
from . import tracking_delta_catalogue as frozen
from .tracking_budget import feasibility_probability, incumbents
from .tracking_dense_gain import A, B, D, E, METHODS, BONUSES, ExactCostSnapshot
from .tracking_online import features, LocalCostModel, seed_for
from .tracking_online_lcb import acquisition_values
from .tracking_delta_cache import empty_cache

fresh_state = frozen.fresh_state
publish = frozen.publish
commit_round = frozen.commit_round
cached_peer = frozen.cached_peer
visible_histories = frozen.visible_histories


def make_programmes(prefixes, metadata, maximum=100, prefix_queries=24):
    """Append per-block frequencies/per-trial speeds; horizon-independent seeds."""
    lookup = {(r['speed'], r['frequency']): r['id'] for r in metadata}
    speeds = sorted({x[0] for x in lookup})
    frequencies = sorted({x[1] for x in lookup})
    programmes = deepcopy(prefixes)
    for seed, clients in programmes.items():
        for client, contexts in clients.items():
            if len(contexts) != prefix_queries:
                raise ValueError('Only the immutable 24-round prefix may initialize a programme.')
            for r in range(prefix_queries, maximum):
                block, offset = divmod(r - prefix_queries, len(frequencies))
                rng = np.random.default_rng(seed_for('budget_extension_frequency_v1', int(seed), client, block))
                frequency = float(rng.permutation(frequencies)[offset])
                rng = np.random.default_rng(seed_for('budget_extension_speed_v1', int(seed), client, r))
                speed = float(rng.choice(speeds))
                contexts.append(lookup[speed, frequency])
    return programmes


def program(domain, cfg, seed, client):
    return cfg['programmes'][str(seed)][client]


def prefix_view(state, clients, prefix_queries=24):
    """Immutable scientific prefix, excluding fields that legitimately evolve."""
    n = len(clients)
    arms = {}
    for method, data in state['methods'].items():
        arms[method] = dict(histories=[h[:prefix_queries] for h in data['histories']],
            events=data['events'][:n * prefix_queries], learning=data['learning'][:prefix_queries],
            snapshots={k: v for k, v in data['snapshots'].items() if int(k) <= n * prefix_queries},
            contacts=[c for c in data['network']['contacts'] if c['round'] <= prefix_queries],
            summaries={k: v for k, v in data['network']['summaries'].items() if v[1] <= prefix_queries})
    return dict(methods=arms, oracle_cache={k: v for k, v in state['oracle_cache'].items() if v['local_evaluation'] <= prefix_queries})


def plan_round(state, domain, cfg, clients, seed, local_phi, recorded=None, prediction_hook=None, timing_hook=None):
    plans, shared_model = [], None
    metadata = {r['id']: r for r in cfg['context_metadata']}
    for method in METHODS:
        data = state['methods'][method]
        acquisition = dict(cfg['acquisition'], confidence_multiplier=BONUSES[method])
        for client, vehicle in enumerate(clients):
            started = time.perf_counter()
            history, previous_round = data['histories'][client], state['round']
            if data['network']['last_contact_round'] != previous_round:
                raise RuntimeError('Every-round contact is required for this continuation.')
            if len(history) != previous_round:
                raise RuntimeError('Non-atomic local history.')
            context = program(domain, cfg, seed, vehicle['id'])[previous_round]
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
                        if method == D:
                            if shared_model is None:
                                shared_model = ExactCostSnapshot(view, domain, cfg['gp'], cfg['hierarchical'], True)
                            model = shared_model
                        else:
                            model = ExactCostSnapshot(view, domain, cfg['gp'], cfg['hierarchical'], False)
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


def run_fleet(domain, cfg, clients, seed, interval, observe, checkpoint=None, state=None,
              stop_round=None, recorded=None, prediction_hook=None, timing_hook=None):
    if interval != 1 or cfg['methods'] != METHODS or cfg['bonus_by_method'] != BONUSES:
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
            prior = dict(histories=[list(h) for h in rebuilt['histories']], network=dict(catalogue=rebuilt['network']['catalogue']))
            for i in range(n):
                event = data['events'][r * n + i]
                context = program(domain, cfg, seed, clients[i]['id'])[r]
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
