"""Exact shared/local cost GP on a larger public gain pool.

Clients form the same Gaussian information messages in a whitened dictionary
of eligible paid inputs. Conditioning on every training input makes this
representation exact; the unobserved conditional GP variance is retained.
"""
from copy import deepcopy

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.special import logsumexp

from . import tracking_hierarchical as joint
from . import tracking_peer_cost as peer
from . import tracking_six_speed as old
from .tracking_budget import matern, feasibility_probability, incumbents
from .tracking_online import features, LocalCostModel, seed_for
from .tracking_online_lcb import acquisition_values

A, B = old.METHODS[:2]
D, E = 'peer_exact_shared_bo', 'peer_exact_local_bo'
METHODS = [A, B, D, E]
BONUSES = {A: 2., B: 2., D: .5, E: .5}
PAIRS = [(D, A, 'primary'), (D, B, 'secondary'), (D, E, 'sharing')]


def support(histories):
    return sorted({o['action'] for h in histories for o in h if o['eligible']})


def model_bytes(histories, cfg):
    counts = [sum(o['eligible'] for o in h) for h in histories]
    size = len(support(histories))
    # Full public action-ID dictionary publication/broadcast, plus the frozen
    # dense Gaussian message reference. No compression or network claim.
    return joint.model_payload_bytes(counts, size, len(cfg['hierarchical']['sharing_fractions'])) + (
        8 * size * (len(histories) + 1) if size else 0)


class ExactCostSnapshot:
    def __init__(self, histories, domain, gp, hierarchical, shared=True):
        self.domain, self.gp, self.shared = domain, gp, shared
        self.histories = histories
        self.fractions = hierarchical['sharing_fractions'] if shared else [0.]
        prior = np.asarray(hierarchical['prior_weights'] if shared else [1.], float)
        self.support = support(histories) if shared else []
        self.components, evidence = [], []
        self.factor = None
        if shared and self.support:
            self.factor = cholesky(self.kernel(domain.X[self.support], domain.X[self.support]), lower=True)
            positions = {a: j for j, a in enumerate(self.support)}
            local = [[dict(o, action=positions[o['action']]) for o in h if o['eligible']] for h in histories]
            for fraction in self.fractions:
                blocks = [joint.ClientBlock(h, self.factor, fraction, gp['noise_std_log_J']**2 + 1e-9) for h in local]
                mean, covariance, value = joint.aggregate_messages(
                    [b.information_message() for b in blocks], len(self.support))
                self.components.append((blocks, mean, covariance))
                evidence.append(value)
        else:
            evidence = [0.] * len(self.fractions)
        log_weights = np.log(prior / prior.sum()) + evidence
        self.weights = np.exp(log_weights - logsumexp(log_weights))
        counts = [sum(o['eligible'] for o in h) for h in histories]
        self.diagnostics = dict(sharing_fractions=list(self.fractions), sharing_weights=self.weights.tolist(),
            component_log_evidence=evidence, expected_sharing_fraction=float(self.weights @ self.fractions),
            eligible_training_counts=counts, active_support_size=len(self.support),
            representation='exact_active_input_dictionary',
            model_payload_bytes=model_bytes(histories, dict(hierarchical=hierarchical)) if shared else 0)
        self.local = {}

    def kernel(self, X, Z):
        return matern(X, Z, self.gp['length_scales'], self.gp['amplitude'], self.gp['constant_variance'])

    def predict(self, client, actions):
        X = self.domain.X[np.asarray(actions, int)]
        diagonal = self.gp['amplitude']**2 + self.gp['constant_variance']
        if not self.shared:
            if client not in self.local:
                h = [o for o in self.histories[client] if o['eligible']]
                ids = [o['action'] for o in h]
                if ids:
                    matrix = self.kernel(self.domain.X[ids], self.domain.X[ids])
                    matrix.flat[::len(matrix) + 1] += self.gp['noise_std_log_J']**2 + 1e-9
                    L = cholesky(matrix, lower=True)
                    alpha = solve_triangular(L.T, solve_triangular(L, np.log([o['J'] for o in h]), lower=True))
                    self.local[client] = ids, L, alpha
                else:
                    self.local[client] = ids, None, None
            ids, L, alpha = self.local[client]
            if not ids:
                return np.zeros(len(X)), np.full(len(X), diagonal)
            cross = self.kernel(X, self.domain.X[ids])
            projection = solve_triangular(L, cross.T, lower=True)
            return cross @ alpha, np.maximum(diagonal - np.einsum('ij,ij->j', projection, projection), 0.)
        if self.factor is None:
            return np.zeros(len(X)), np.full(len(X), diagonal)
        F = solve_triangular(self.factor, self.kernel(X, self.domain.X[self.support]).T, lower=True).T
        residual = diagonal - np.einsum('ij,ij->i', F, F)
        if residual.min(initial=0.) < -1e-8:
            raise RuntimeError('Invalid exact conditional GP variance.')
        predictions = [blocks[client].predict(F, mean, covariance) for blocks, mean, covariance in self.components]
        means = np.asarray([p[0] for p in predictions])
        variances = np.asarray([p[1] for p in predictions]) + np.maximum(residual, 0.)
        mean = self.weights @ means
        variance = self.weights @ (variances + (means - mean)**2)
        return mean, variance


def fresh_state(cfg, clients):
    original = old.fresh_state(cfg, clients)
    original['methods'] = {m: deepcopy(original['methods'][source]) for m, source in
        zip(METHODS, old.METHODS)}
    return original


def fit_models(state, domain, cfg):
    return {m: ExactCostSnapshot(state['methods'][m]['histories'], domain, cfg['gp'], cfg['hierarchical'], m == D)
        for m in (D, E)}


def plan_round(state, domain, cfg, clients, seed, models, local_phi, recorded=None):
    plans = []
    metadata = {r['id']: r for r in cfg['context_metadata']}
    for method in METHODS:
        data = state['methods'][method]
        acquisition = dict(cfg['acquisition'], confidence_multiplier=BONUSES[method])
        for i, client in enumerate(clients):
            history, n = data['histories'][i], len(data['histories'][i])
            context = old.program(domain, cfg, seed, client['id'])[n]
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
                        action, parameters, fitted, diagnostics = old._gp_proposal(history, domain, context,
                            prior, cfg, data['gp_parameters'][i], data['last_gp_fit'][i], rng, trace)
                        data['gp_parameters'][i], data['last_gp_fit'][i] = parameters, fitted
                    else:
                        tried = {o['action'] for o in history}
                        allowed = [a for a, (c, _) in enumerate(domain.actions) if c == context and a not in tried]
                        mean, variance = models[method].predict(i, allowed)
                        probability = feasibility_probability(history, domain, np.full(len(domain.actions), .5), cfg['gp'])[allowed]
                        values = acquisition_values(mean, variance, probability, acquisition)
                        action = old._minimum(values, allowed, rng)
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


def commit_round(state, plans, observations, domain, cfg, clients, local_phi):
    peer.commit_round(state, plans, observations, domain, cfg, clients, local_phi)
    models = fit_models(state, domain, cfg)
    for method, model in models.items():
        data, diagnostic = state['methods'][method], model.diagnostics
        count = diagnostic['model_payload_bytes']
        data['model_payload_bytes'] += count
        data['payload_bytes'] += count
        data['learning'][-1].update(**diagnostic, payload_bytes=data['learning'][-1]['peer_payload_bytes'] + count,
            cumulative_model_payload_bytes=data['model_payload_bytes'], cumulative_payload_bytes=data['payload_bytes'])
    return models


def run_fleet(domain, cfg, clients, seed, observe, checkpoint=None, state=None, stop_round=None, recorded=None):
    if cfg['methods'] != METHODS or cfg['bonus_by_method'] != BONUSES:
        raise ValueError('Fixed four-arm protocol differs.')
    state = deepcopy(state) if state is not None else fresh_state(cfg, clients)
    limit = cfg['queries_per_client'] if stop_round is None else stop_round
    if not state['round'] <= limit <= cfg['queries_per_client']:
        raise ValueError('Invalid stopping round.')
    local_phi = features(domain, cfg['model'])
    models = fit_models(state, domain, cfg) if state['round'] else {}
    while state['round'] < limit:
        plans = plan_round(state, domain, cfg, clients, seed, models, local_phi, recorded)
        observations = observe(plans, state['oracle_cache'])
        if len(observations) != len(plans):
            raise RuntimeError('Oracle observation count differs.')
        models = commit_round(state, plans, observations, domain, cfg, clients, local_phi)
        if checkpoint is not None:
            checkpoint(state)
    return state


def validate_state(state, domain, cfg, clients, seed, complete=False, replay=False):
    """Reuse the frozen schedule/peer/history checks, then check new messages.

    Only the validation view's payload fields are mapped to the old fixed-width
    reference. The real dictionary-based accounting is verified separately.
    """
    if set(state['methods']) != set(METHODS):
        raise RuntimeError('Four-arm method set differs.')
    view = deepcopy(state)
    view['methods'] = {alias: view['methods'][method] for method, alias in zip(METHODS, old.METHODS)}
    for method, alias in zip(METHODS, old.METHODS):
        actual, mapped = state['methods'][method], view['methods'][alias]
        model_total, peer_total, mapped_total = 0, 0, 0
        for r, (log, changed) in enumerate(zip(actual['learning'], mapped['learning']), 1):
            prefix = [h[:r] for h in actual['histories']]
            expected = model_bytes(prefix, cfg) if method == D else 0
            model_total += expected
            peer_total += log['peer_payload_bytes']
            if (log['model_payload_bytes'], log['cumulative_model_payload_bytes'],
                    log['payload_bytes'], log['cumulative_payload_bytes']) != (
                    expected, model_total, expected + log['peer_payload_bytes'], model_total + peer_total):
                raise RuntimeError('Exact dictionary message accounting differs.')
            if method in (D, E) and (log['representation'] != 'exact_active_input_dictionary' or
                    log['active_support_size'] != (len(support(prefix)) if method == D else 0)):
                raise RuntimeError('Exact dictionary support differs.')
            count = joint.model_payload_bytes([sum(o['eligible'] for o in h) for h in prefix], 1,
                len(cfg['hierarchical']['sharing_fractions'])) if method == D else 0
            mapped_total += count
            changed.update(model_payload_bytes=count, cumulative_model_payload_bytes=mapped_total,
                payload_bytes=count + log['peer_payload_bytes'], cumulative_payload_bytes=mapped_total + peer_total)
        if (actual['model_payload_bytes'], actual['payload_bytes']) != (model_total, model_total + peer_total):
            raise RuntimeError('Exact message total differs.')
        mapped.update(model_payload_bytes=mapped_total, payload_bytes=mapped_total + peer_total)
        for event in mapped['events']:
            event['method'] = alias
        for key, snapshot in actual['snapshots'].items():
            prior = actual['learning'][snapshot['round'] - 2] if snapshot['round'] > 1 else {}
            expected = prior.get('cumulative_model_payload_bytes', 0)
            if snapshot['model_payload_bytes'] != expected or snapshot['payload_bytes'] != expected + snapshot['peer_payload_bytes']:
                raise RuntimeError('Snapshot includes future model messages.')
            mapped_snapshot = mapped['snapshots'][key]
            mapped_prior = mapped['learning'][snapshot['round'] - 2] if snapshot['round'] > 1 else {}
            count = mapped_prior.get('cumulative_model_payload_bytes', 0)
            mapped_snapshot.update(model_payload_bytes=count, payload_bytes=count + snapshot['peer_payload_bytes'])
    old.validate_state(view, domain, cfg, clients, seed, np.ones((1, 1)), complete, replay=False)
    if replay:
        def recorded_observe(plans, cache):
            observations = []
            for plan in plans:
                event = state['methods'][plan['method']]['events'][(plan['round'] - 1) * len(clients) + plan['client']]
                old.assert_equivalent(plan, {k: event[k] for k in plan})
                observations.append({k: event[k] for k in peer.OBSERVATION_FIELDS if k != 'action'})
            return observations
        rebuilt = run_fleet(domain, cfg, clients, seed, recorded_observe, stop_round=state['round'], recorded=state)
        old.assert_equivalent(rebuilt['methods'], state['methods'])
