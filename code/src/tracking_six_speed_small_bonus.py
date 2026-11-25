"""Single beta=0.5 comparison with frozen A/B and beta=0 spectral controls."""
from copy import deepcopy

import numpy as np

from . import tracking_peer_cost as peer
from . import tracking_six_speed as reference
from .tracking_budget import feasibility_probability
from .tracking_online import LocalCostModel, features, seed_for
from .tracking_online_lcb import acquisition_values
from .tracking_six_speed_zero_bonus import A, B, D0, E0

D_SMALL = 'peer_spectral_shared_256_small_bonus_bo'
E_SMALL = 'peer_spectral_local_256_small_bonus_bo'
SMALL_BONUS = .5
METHODS = [A, B, D0, E0, D_SMALL, E_SMALL]
SPECTRAL = [D0, E0, D_SMALL, E_SMALL]
SHARED = (D0, D_SMALL)
BONUSES = {A: 2., B: 2., D0: 0., E0: 0., D_SMALL: SMALL_BONUS, E_SMALL: SMALL_BONUS}


def fresh_state(cfg, clients):
    state = reference.fresh_state(cfg, clients)
    original = state['methods']
    source = {A: A, B: B, D0: reference.SHARED, E0: reference.LOCAL,
              D_SMALL: reference.SHARED, E_SMALL: reference.LOCAL}
    state['methods'] = {method: deepcopy(original[template]) for method, template in source.items()}
    return state


def fit_models(state, phi, cfg):
    models, identical_inputs = {}, {}
    for method in SPECTRAL:
        histories = state['methods'][method]['histories']
        shared = method in SHARED
        key = (shared, tuple(tuple((o['action'], o['J']) for o in h if o['eligible']) for h in histories))
        if key not in identical_inputs:
            identical_inputs[key] = reference.CostSnapshot(histories, phi, cfg['gp'], cfg['hierarchical'], shared)
        # Identical numerical inputs may reuse a read-only fit. Histories and
        # reference message/query charges remain independent for every arm.
        models[method] = identical_inputs[key]
    return models


def plan_round(state, domain, cfg, clients, seed, models, local_phi, recorded=None):
    plans = []
    metadata = {r['id']: r for r in cfg['context_metadata']}
    for method in METHODS:
        data = state['methods'][method]
        acquisition = dict(cfg['acquisition'], confidence_multiplier=BONUSES[method])
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
                    mode = 'feasibility_penalized_LCB'
                    if method in (A, B):
                        prior = np.zeros(len(domain.actions)) if method == A else (
                            local_phi @ LocalCostModel(history, local_phi, cfg['model']['ridge']).fit())
                        trace = None if recorded is None else recorded['methods'][method]['events'][n * len(clients) + i]
                        action, parameters, fitted, diagnostics = reference._gp_proposal(history, domain, context,
                            prior, cfg, data['gp_parameters'][i], data['last_gp_fit'][i], rng, trace)
                        data['gp_parameters'][i], data['last_gp_fit'][i] = parameters, fitted
                    else:
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
        data, diagnostic = state['methods'][method], model.diagnostics
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
        raise ValueError('The six-arm design fixes beta=0/0.5 for D/E and beta=2 for A/B.')
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
    for shared, local in ((D0, E0), (D_SMALL, E_SMALL)):
        aliases = {A: A, B: B, reference.SHARED: shared, reference.LOCAL: local}
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
