"""Isolated six-speed programme and lazy predictions from the frozen Gaussian model.

No frozen experiment is edited. Gaussian messages and predictions delegate to
tracking_hierarchical; only the scheduled context's candidates are predicted.
Fit states for A/B are logged before each requested observation, enabling replay
without repeating platform-dependent hyperparameter optimization.
"""
from copy import deepcopy

import numpy as np
from scipy.special import logsumexp

from . import tracking_hierarchical as joint
from . import tracking_peer_cost as peer
from .tracking_budget import Posterior, feasibility_probability, incumbents
from .tracking_confirmation import METHODS, PAIRS
from .tracking_online import LocalCostModel, features, seed_for
from .tracking_online_lcb import acquisition_values, choose as gp_choose

SHARED, LOCAL = METHODS[2:]


def program(domain, cfg, seed, client):
    """Balanced frequency blocks; independent uniform speed draws with replacement."""
    rng = np.random.default_rng(seed_for('six_speed_programme', seed, client))
    metadata = cfg['context_metadata']
    speeds = sorted({r['speed'] for r in metadata})
    frequencies = sorted({r['frequency'] for r in metadata})
    lookup = {(r['speed'], r['frequency']): r['id'] for r in metadata}
    if set(lookup) != {(v, f) for v in speeds for f in frequencies}:
        raise ValueError('Programme requires the Cartesian speed/frequency grid.')
    scheduled = []
    while len(scheduled) < cfg['queries_per_client']:
        scheduled.extend(rng.permutation(frequencies).tolist())
    assigned = rng.choice(speeds, size=cfg['queries_per_client'], replace=True)
    return [lookup[float(v), float(f)] for v, f in zip(assigned, scheduled)]


class CostSnapshot:
    """Same messages, evidence and joint predictions; no full-grid prediction bank."""
    def __init__(self, histories, phi, gp, hierarchical, shared=True):
        self.F = phi * np.r_[np.sqrt(gp['constant_variance']),
                             np.full(phi.shape[1] - 1, gp['amplitude'])]
        fractions = hierarchical['sharing_fractions'] if shared else [0.]
        prior = np.asarray(hierarchical['prior_weights'] if shared else [1.], float)
        self.components, evidence = [], []
        for fraction in fractions:
            blocks = [joint.ClientBlock(h, self.F, fraction, gp['noise_std_log_J']**2 + 1e-9)
                      for h in histories]
            if shared:
                mean, covariance, value = joint.aggregate_messages(
                    [b.information_message() for b in blocks], self.F.shape[1])
            else:
                mean, covariance, value = np.zeros(self.F.shape[1]), np.eye(self.F.shape[1]), 0.
            self.components.append((blocks, mean, covariance))
            evidence.append(value)
        log_weights = np.log(prior / prior.sum()) + evidence
        self.weights = np.exp(log_weights - logsumexp(log_weights))
        counts = [sum(o['eligible'] for o in h) for h in histories]
        self.diagnostics = dict(sharing_fractions=list(fractions), sharing_weights=self.weights.tolist(),
            component_log_evidence=evidence, expected_sharing_fraction=float(self.weights @ fractions),
            eligible_training_counts=counts,
            model_payload_bytes=joint.model_payload_bytes(counts, phi.shape[1], len(fractions)) if shared else 0)

    def predict(self, client, actions):
        rows = self.F[np.asarray(actions, int)]
        predictions = [blocks[client].predict(rows, mean, covariance)
                       for blocks, mean, covariance in self.components]
        means, variances = (np.asarray([p[k] for p in predictions]) for k in (0, 1))
        mean = self.weights @ means
        variance = self.weights @ (variances + (means - mean)**2)
        return mean, variance


def fresh_state(cfg, clients):
    parameters = cfg.get('initial_gp_parameters',
                         np.log([*cfg['gp']['length_scales'], cfg['gp']['amplitude']]).tolist())
    return dict(round=0, oracle_cache={}, methods={m: dict(histories=[[] for _ in clients],
        gp_parameters=[parameters.copy() for _ in clients], last_gp_fit=[0] * len(clients),
        events=[], snapshots={}, learning=[], payload_bytes=0, peer_payload_bytes=0,
        model_payload_bytes=0) for m in METHODS})


def fit_models(state, phi, cfg):
    return {m: CostSnapshot(state['methods'][m]['histories'], phi, cfg['gp'], cfg['hierarchical'], m == SHARED)
            for m in (SHARED, LOCAL)}


def _minimum(values, actions, rng):
    best = float(np.min(values))
    if not np.isfinite(best):
        raise RuntimeError('No unqueried candidate in the scheduled context.')
    tied = np.asarray(actions)[np.asarray(values) <= best + 1e-12 * max(1., abs(best))]
    return int(rng.choice(tied))


def _gp_proposal(history, domain, context, prior, cfg, parameters, last_fit, rng, recorded=None):
    """Fresh fits during execution; fixed logged fit during portable audit."""
    count = sum(o['eligible'] for o in history)
    optimize = count >= 4 and count - last_fit >= cfg['gp']['optimize_every']
    expected_fit = count if optimize else last_fit
    if recorded is None:
        action, parameters, fitted, _ = gp_choose(history, domain, context, prior, cfg['gp'],
            cfg['acquisition'], parameters, last_fit, rng)
    else:
        previous_parameters = np.asarray(parameters).copy()
        parameters = np.asarray(recorded['gp_fit_parameters'], float)
        limits = np.log([cfg['gp']['length_bounds']] * domain.X.shape[1] + [cfg['gp']['amplitude_bounds']])
        if (parameters.shape != (domain.X.shape[1] + 1,) or not np.isfinite(parameters).all() or
                np.any(parameters < limits[:, 0] - 1e-12) or np.any(parameters > limits[:, 1] + 1e-12) or
                recorded['gp_fit_count'] != expected_fit):
            raise RuntimeError('Invalid logged GP fit or refit schedule.')
        if not optimize and not np.array_equal(parameters, previous_parameters):
            raise RuntimeError('Unscheduled GP parameter change.')
        fitted = expected_fit
        posterior = Posterior(domain, history, prior, cfg['gp'], parameters, optimize=False)
        mean, variance = posterior.predict()
        probability = feasibility_probability(history, domain, np.full(len(domain.actions), .5), cfg['gp'])
        values = acquisition_values(mean, variance, probability, cfg['acquisition'])
        allowed = [a for a, (c, _) in enumerate(domain.actions) if c == context and a not in {o['action'] for o in history}]
        action = _minimum(values[allowed], allowed, rng)
        parameters = parameters.tolist()
    if fitted != expected_fit:
        raise RuntimeError('Unexpected local GP refit counter.')
    posterior = Posterior(domain, history, prior, cfg['gp'], np.asarray(parameters), optimize=False)
    mean, variance = posterior.predict()
    probability = feasibility_probability(history, domain, np.full(len(domain.actions), .5), cfg['gp'])
    values = acquisition_values(mean, variance, probability, cfg['acquisition'])
    diagnostics = dict(gp_fit_parameters=list(parameters), gp_fit_count=fitted,
        predicted_log_J=float(mean[action]), predicted_variance=float(variance[action]),
        eligibility_probability=float(probability[action]), acquisition_score=float(values[action]))
    return action, parameters, fitted, diagnostics


def plan_round(state, domain, cfg, clients, seed, models, local_phi, recorded=None):
    plans = []
    metadata = {r['id']: r for r in cfg['context_metadata']}
    for method in METHODS:
        data = state['methods'][method]
        for i, client in enumerate(clients):
            history = data['histories'][i]
            n = len(history)
            context = program(domain, cfg, seed, client['id'])[n]
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
                    if method in METHODS[:2]:
                        prior = np.zeros(len(domain.actions)) if method == METHODS[0] else (
                            local_phi @ LocalCostModel(history, local_phi, cfg['model']['ridge']).fit())
                        trace = None
                        if recorded is not None:
                            trace = recorded['methods'][method]['events'][n * len(clients) + i]
                        action, parameters, fitted, diagnostics = _gp_proposal(history, domain, context, prior, cfg,
                            data['gp_parameters'][i], data['last_gp_fit'][i], rng, trace)
                        data['gp_parameters'][i], data['last_gp_fit'][i] = parameters, fitted
                    else:
                        tried = {o['action'] for o in history}
                        allowed = [a for a, (c, _) in enumerate(domain.actions) if c == context and a not in tried]
                        mean, variance = models[method].predict(i, allowed)
                        probability = feasibility_probability(history, domain, np.full(len(domain.actions), .5), cfg['gp'])[allowed]
                        values = acquisition_values(mean, variance, probability, cfg['acquisition'])
                        action = _minimum(values, allowed, rng)
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
    # This frozen function handles requests, incumbents, snapshots and peer bytes.
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


def assert_equivalent(a, b):
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise RuntimeError('Replay keys differ.')
        for k in a:
            assert_equivalent(a[k], b[k])
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            raise RuntimeError('Replay length differs.')
        for x, y in zip(a, b):
            assert_equivalent(x, y)
    elif isinstance(a, (bool, str, int)) or a is None:
        if a != b:
            raise RuntimeError('Replay decision or metadata differs.')
    elif not np.isclose(a, b, rtol=2e-8, atol=2e-9):
        raise RuntimeError('Replay numeric value differs.')


def validate_state(state, domain, cfg, clients, seed, phi, complete=False, replay=False):
    n, rounds = len(clients), state['round']
    if set(state['methods']) != set(METHODS) or not 0 <= rounds <= cfg['queries_per_client']:
        raise RuntimeError('Method set or round count differs.')
    if complete and rounds != cfg['queries_per_client']:
        raise RuntimeError('Incomplete online job.')
    first = None
    for method, data in state['methods'].items():
        if (len(data['events']) != n * rounds or len(data['learning']) != rounds or
                len(data['histories']) != n):
            raise RuntimeError('Non-atomic paid round.')
        histories, snapshots = [[] for _ in clients], {}
        for number, event in enumerate(data['events']):
            r, i = divmod(number, n)
            if i == 0:
                prior = deepcopy(histories)
            context = program(domain, cfg, seed, clients[i]['id'])[r]
            if (event['round'], event['information_round'], event['client'], event['local_evaluation'],
                event['fleet_evaluation'], event['context'], event['noise_seed']) != (
                r + 1, r, i, r + 1, number + 1, context, seed_for('six_speed_episode', seed, clients[i]['id'], r)):
                raise RuntimeError('Paid schedule, counter or round barrier differs.')
            action, donor = (domain.action(context, domain.candidates[0]), dict.fromkeys(peer.PEER_FIELDS)) if r == 0 else (
                peer.peer_proposal(prior, data['events'], domain, i, context))
            mode = 'factory_initialization' if r == 0 else 'peer_gain_verification' if action is not None else 'feasibility_penalized_LCB'
            if (event['method'] != method or event['mode'] != mode or
                    (action is not None and event['action'] != action) or
                    any(event[k] != v for k, v in donor.items()) or
                    domain.actions[event['action']] != (context, event['candidate']) or
                    any(o['action'] == event['action'] for o in histories[i])):
                raise RuntimeError('Peer priority, provenance or distinct local query differs.')
            if (event['eligible'] != bool(event['complete'] and event['feasible'] and event['integration_resolved']) or
                    (event['J'] is None) == event['eligible'] or
                    (event['eligible'] and (not np.isfinite(event['J']) or event['J'] <= 0))):
                raise RuntimeError('Invalid eligibility or observed cost.')
            metadata = {r['id']: r for r in cfg['context_metadata']}
            speed = metadata[context]['speed']
            if (event['speed'], event['frequency'], event['context_visit'], event['speed_visit']) != (
                    speed, metadata[context]['frequency'],
                    1 + sum(domain.actions[o['action']][0] == context for o in histories[i]),
                    1 + sum(metadata[domain.actions[o['action']][0]]['speed'] == speed for o in histories[i])):
                raise RuntimeError('Visit metadata differs from the paid history.')
            old = incumbents(histories[i], domain)
            histories[i].append({k: event[k] for k in peer.OBSERVATION_FIELDS})
            current = incumbents(histories[i], domain)
            c = domain.contexts.index(context)
            if event['recommendations'] != current or event['became_incumbent'] != (old[c] != current[c]):
                raise RuntimeError('Unobserved or ineligible recommendation.')
            if number + 1 in cfg['fleet_budgets']:
                snapshots[str(number + 1)] = dict(round=r + 1, recommendations=[incumbents(h, domain) for h in histories],
                    queries_per_client=[len(h) for h in histories])
        if histories != data['histories']:
            raise RuntimeError('Saved histories differ from paid events.')
        if method in METHODS[:2]:
            initial_parameters = cfg.get('initial_gp_parameters',
                np.log([*cfg['gp']['length_scales'], cfg['gp']['amplitude']]).tolist())
            expected_parameters, expected_fit = [], []
            for i in range(n):
                fits = [e for e in data['events'] if e['client'] == i and e['gp_fit_parameters'] is not None]
                expected_parameters.append(fits[-1]['gp_fit_parameters'] if fits else initial_parameters)
                expected_fit.append(fits[-1]['gp_fit_count'] if fits else 0)
            if data['gp_parameters'] != expected_parameters or data['last_gp_fit'] != expected_fit:
                raise RuntimeError('Checkpoint GP state differs from the logged fits.')
        peers = models = 0
        for r, log in enumerate(data['learning'], 1):
            prefix = [h[:r] for h in histories]
            counts = [sum(o['eligible'] for o in h) for h in prefix]
            peer_bytes = sum(sum(c is not None for c in incumbents(h, domain)) for h in prefix) * 12 * (n + 1)
            model_bytes = joint.model_payload_bytes(counts, phi.shape[1], len(cfg['hierarchical']['sharing_fractions'])) if method == SHARED else 0
            peers += peer_bytes
            models += model_bytes
            expected = dict(round=r, eligible_training_counts=counts, peer_payload_bytes=peer_bytes,
                model_payload_bytes=model_bytes, payload_bytes=peer_bytes + model_bytes,
                cumulative_peer_payload_bytes=peers, cumulative_model_payload_bytes=models,
                cumulative_payload_bytes=peers + models)
            if any(log[k] != v for k, v in expected.items()):
                raise RuntimeError('Learning counts or reference communication accounting differs.')
        if (data['peer_payload_bytes'], data['model_payload_bytes'], data['payload_bytes']) != (peers, models, peers + models):
            raise RuntimeError('Cumulative payload differs.')
        for key, snapshot in snapshots.items():
            earlier = data['learning'][snapshot['round'] - 2] if snapshot['round'] > 1 else {}
            snapshot.update(payload_bytes=earlier.get('cumulative_payload_bytes', 0),
                model_payload_bytes=earlier.get('cumulative_model_payload_bytes', 0),
                peer_payload_bytes=earlier.get('cumulative_peer_payload_bytes', 0))
        if snapshots != data['snapshots']:
            raise RuntimeError('Snapshot recommendation or information age differs.')
        initial = [{k: e[k] for k in peer.OBSERVATION_FIELDS} for e in data['events'][:n]]
        if first is not None and initial != first:
            raise RuntimeError('Factory observations are not matched across arms.')
        first = initial
    if replay:
        def recorded_observe(plans, cache):
            result = []
            for plan in plans:
                event = state['methods'][plan['method']]['events'][(plan['round'] - 1) * n + plan['client']]
                assert_equivalent(plan, {k: event[k] for k in plan})
                result.append({k: event[k] for k in peer.OBSERVATION_FIELDS if k != 'action'})
            return result
        actual = run_fleet(domain, cfg, clients, seed, recorded_observe, phi, stop_round=rounds, recorded=state)
        assert_equivalent(actual['methods'], state['methods'])
