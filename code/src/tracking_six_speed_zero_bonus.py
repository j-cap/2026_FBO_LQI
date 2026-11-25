"""Matched six-speed ablation: frozen A/B/D/E plus D/E without the cost bonus.

Each arm owns its histories, peer catalogue and model. Only public features,
programmes and identical requested physical simulations are reusable across arms.
The frozen four-arm implementation remains the reference implementation.
"""
from copy import deepcopy

import numpy as np

from . import tracking_peer_cost as peer
from . import tracking_six_speed as reference
from .tracking_budget import feasibility_probability
from .tracking_online import features, seed_for
from .tracking_online_lcb import acquisition_values

A, B, D2, E2 = reference.METHODS
D0 = 'peer_spectral_shared_256_zero_bonus_bo'
E0 = 'peer_spectral_local_256_zero_bonus_bo'
METHODS = [A, B, D2, E2, D0, E0]
SPECTRAL = [D2, E2, D0, E0]
BONUSES = {A: 2., B: 2., D2: 2., E2: 2., D0: 0., E0: 0.}


def fresh_state(cfg, clients):
    state = reference.fresh_state(cfg, clients)
    for target, source in ((D0, D2), (E0, E2)):
        state['methods'][target] = deepcopy(state['methods'][source])
    return state


def fit_models(state, phi, cfg):
    models, identical_inputs = {}, {}
    for method in SPECTRAL:
        histories = state['methods'][method]['histories']
        shared = method in (D2, D0)
        key = (shared, tuple(tuple((o['action'], o['J']) for o in h if o['eligible']) for h in histories))
        if key not in identical_inputs:
            identical_inputs[key] = reference.CostSnapshot(histories, phi, cfg['gp'], cfg['hierarchical'], shared)
        # Read-only numerical reuse only for exactly equal fitting inputs. Messages
        # are still charged independently; no arm receives another arm's labels.
        models[method] = identical_inputs[key]
    return models


def plan_round(state, domain, cfg, clients, seed, models, local_phi, recorded=None):
    plans = reference.plan_round(state, domain, cfg, clients, seed, models, local_phi, recorded)
    metadata = {r['id']: r for r in cfg['context_metadata']}
    acquisition = dict(cfg['acquisition'], confidence_multiplier=0.)
    for method in (D0, E0):
        data = state['methods'][method]
        for i, client in enumerate(clients):
            history = data['histories'][i]
            n = len(history)
            context = reference.program(domain, cfg, seed, client['id'])[n]
            rng = np.random.default_rng(seed_for('acquisition', seed, client['id'], n))
            donor = dict.fromkeys(peer.PEER_FIELDS)
            diagnostics = dict.fromkeys(('gp_fit_parameters', 'gp_fit_count', 'predicted_log_J',
                'predicted_variance', 'eligibility_probability', 'acquisition_score'))
            if n == 0:
                action, mode = domain.action(context, domain.candidates[0]), 'factory_initialization'
            else:
                action, donor = peer.peer_proposal(data['histories'], data['events'], domain, i, context)
                mode = 'peer_gain_verification'
                if action is None:
                    # Legacy mode name also denotes the beta=0 limiting case.
                    mode = 'feasibility_penalized_LCB'
                    tried = {o['action'] for o in history}
                    allowed = [a for a, (c, _) in enumerate(domain.actions) if c == context and a not in tried]
                    mean, variance = models[method].predict(i, allowed)
                    probability = feasibility_probability(history, domain,
                        np.full(len(domain.actions), .5), cfg['gp'])[allowed]
                    values = acquisition_values(mean, variance, probability, acquisition)
                    action = reference._minimum(values, allowed, rng)
                    j = allowed.index(action)
                    diagnostics.update(predicted_log_J=float(mean[j]), predicted_variance=float(variance[j]),
                        eligibility_probability=float(probability[j]), acquisition_score=float(values[j]))
            speed = metadata[context]['speed']
            plans.append(dict(method=method, client=i, local_evaluation=n + 1, round=state['round'] + 1,
                action=action, context=context, candidate=domain.actions[action][1], mode=mode,
                noise_seed=seed_for('six_speed_episode', seed, client['id'], n), information_round=state['round'],
                speed=speed, frequency=metadata[context]['frequency'],
                context_visit=1 + sum(domain.actions[o['action']][0] == context for o in history),
                speed_visit=1 + sum(metadata[domain.actions[o['action']][0]]['speed'] == speed for o in history),
                **donor, **diagnostics))
    return plans


def commit_round(state, plans, observations, domain, cfg, clients, phi, local_phi):
    peer.commit_round(state, plans, observations, domain, cfg, clients, local_phi)
    models = fit_models(state, phi, cfg)
    for method, model in models.items():
        data = state['methods'][method]
        diagnostic = model.diagnostics
        count = diagnostic['model_payload_bytes']
        data['model_payload_bytes'] += count
        data['payload_bytes'] += count
        data['learning'][-1].update(**diagnostic, payload_bytes=data['learning'][-1]['peer_payload_bytes'] + count,
            cumulative_model_payload_bytes=data['model_payload_bytes'], cumulative_payload_bytes=data['payload_bytes'])
    return models


def run_fleet(domain, cfg, clients, seed, observe, phi, checkpoint=None, state=None,
              stop_round=None, recorded=None):
    if (list(cfg['methods']) != METHODS or cfg['bonus_by_method'] != BONUSES or
            cfg['acquisition']['confidence_multiplier'] != 2.):
        raise ValueError('The matched six-arm design and reference bonus must remain fixed.')
    state = deepcopy(state) if state is not None else fresh_state(cfg, clients)
    limit = cfg['queries_per_client'] if stop_round is None else stop_round
    if not state['round'] <= limit <= cfg['queries_per_client']:
        raise ValueError('Invalid stop round.')
    local_phi = features(domain, cfg['model'])
    models = fit_models(state, phi, cfg) if state['round'] else {}
    while state['round'] < limit:
        plans = plan_round(state, domain, cfg, clients, seed, models, local_phi, recorded)
        observations = observe(plans, state['oracle_cache'])
        if len(observations) != len(plans):
            raise RuntimeError('Oracle observation count differs.')
        models = commit_round(state, plans, observations, domain, cfg, clients, phi, local_phi)
        if checkpoint is not None:
            checkpoint(state)
    return state


def validate_state(state, domain, cfg, clients, seed, phi, complete=False, replay=False):
    if set(state['methods']) != set(METHODS):
        raise RuntimeError('Matched six-arm method set differs.')
    # Reuse the frozen structural audit for both spectral pairs. Aliases are
    # private copies and only used for structural checks, never for acquisition.
    for pair in ((D2, E2), (D0, E0)):
        aliases = {A: A, B: B, D2: pair[0], E2: pair[1]}
        view = dict(state, methods={m: deepcopy(state['methods'][source]) for m, source in aliases.items()})
        for method, data in view['methods'].items():
            for event in data['events']:
                event['method'] = method
        reference.validate_state(view, domain, cfg, clients, seed, phi, complete, replay=False)
    if replay:
        n = len(clients)
        def recorded_observe(plans, cache):
            outcomes = []
            for plan in plans:
                event = state['methods'][plan['method']]['events'][(plan['round'] - 1) * n + plan['client']]
                reference.assert_equivalent(plan, {k: event[k] for k in plan})
                outcomes.append({k: event[k] for k in peer.OBSERVATION_FIELDS if k != 'action'})
            return outcomes
        actual = run_fleet(domain, cfg, clients, seed, recorded_observe, phi,
            stop_round=state['round'], recorded=state)
        reference.assert_equivalent(actual['methods'], state['methods'])
