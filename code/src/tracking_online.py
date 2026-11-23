"""Cold-start fleet calibration with round-synchronous information exchange.

Only observations requested in this run enter a learner. The vehicle simulator
is injected as an oracle; validation/reference tables are never accepted here.
"""
from copy import deepcopy
import hashlib

import numpy as np
from scipy.linalg import solve

from .tracking_budget import Posterior, feasibility_probability, incumbents, lognormal_ei


METHODS = ['independent_bo', 'local_cost_bo', 'online_gain_sharing',
           'single_scout', 'federated_cost_bo', 'centralized_cost_bo']


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:4], 'little')


def program(domain, cfg, seed, client):
    """Same context programme/initial gains for a client's counterfactual methods."""
    rng = np.random.default_rng(seed_for('programme', seed, client))
    contexts = []
    while len(contexts) < cfg['queries_per_client']:
        contexts.extend(rng.permutation(domain.contexts).tolist())
    initial = [domain.candidates[0], *rng.choice(domain.candidates[1:],
        size=len(domain.contexts)-1, replace=False).tolist()]
    return contexts[:cfg['queries_per_client']], initial


def features(domain, cfg):
    """Public, data-independent random Fourier features for a Matern-5/2 kernel.

    A common chi-square denominator per frequency gives multivariate Student-t
    frequencies (five degrees of freedom), scaled by the fixed length scales.
    The trainable model is linear in these fixed features, including an intercept.
    """
    rng = np.random.default_rng(cfg['feature_seed'])
    count, dim = cfg['features'], domain.X.shape[1]
    omega = rng.normal(size=(count, dim))/np.sqrt(rng.chisquare(5, size=(count, 1))/5)
    omega /= np.asarray(cfg['length_scales'])
    phase = rng.uniform(0, 2*np.pi, size=count)
    return np.column_stack((np.ones(len(domain.actions)),
        np.sqrt(2/count)*np.cos(domain.X@omega.T+phase)))


class LocalCostModel:
    """A client's local quadratic loss; server aggregation uses only update()."""
    def __init__(self, history, phi, ridge):
        good = [o for o in history if o['eligible']]
        self.count = len(good)
        width = phi.shape[1]
        self.A, self.b = ridge*np.eye(width), np.zeros(width)
        if good:
            X = phi[[o['action'] for o in good]]
            y = np.log([o['J'] for o in good])
            self.A += X.T@X/self.count
            self.b = X.T@y/self.count

    def update(self, weights, steps, rate):
        local = np.asarray(weights).copy()
        for _ in range(steps):
            local -= 2*rate*(self.A@local-self.b)
        return local

    def fit(self):
        return solve(self.A, self.b, assume_a='pos', check_finite=False)


def federated_fit(local_models, previous, cfg):
    active = [m for m in local_models if m.count]
    if not active:
        return np.zeros_like(previous), 0
    proportions = np.asarray([m.count for m in active], float)
    proportions /= proportions.sum()
    weights = np.asarray(previous).copy()
    for _ in range(cfg['server_rounds']):
        updates = [m.update(weights, cfg['local_steps'], cfg['learning_rate']) for m in active]
        weights = sum(p*w for p, w in zip(proportions, updates))
    # Float64 model-vector payload, uplink + per-client broadcast; no transport overhead.
    payload = cfg['server_rounds']*len(active)*2*len(weights)*8
    return weights, int(payload)


def centralized_fit(local_models):
    active = [m for m in local_models if m.count]
    if not active:
        return np.zeros_like(local_models[0].b)
    total = sum(m.count for m in active)
    A = sum(m.count*m.A for m in active)/total
    b = sum(m.count*m.b for m in active)/total
    return solve(A, b, assume_a='pos', check_finite=False)


def optimization_gap(local_models, weights):
    """Offline diagnostic. Never influences federated updates or acquisition."""
    active = [m for m in local_models if m.count]
    if not active:
        return 0.
    optimum = centralized_fit(local_models)
    delta = weights-optimum
    return max(0., float(sum(m.count*(delta@m.A@delta) for m in active)/sum(m.count for m in active)))


def choose(history, domain, context, prior, gp, params, last_fit, rng):
    count = sum(o['eligible'] for o in history)
    optimize = count >= 4 and count-last_fit >= gp['optimize_every']
    posterior = Posterior(domain, history, prior, gp, params, optimize)
    mean, variance = posterior.predict()
    probability = feasibility_probability(history, domain, np.full(len(domain.actions), .5), gp)
    good = [o['J'] for o in history if o['eligible'] and domain.actions[o['action']][0] == context]
    if good:
        scores = probability*lognormal_ei(mean, variance, min(good))
        mode = 'feasibility_weighted_EI'
    else:
        scores = probability*(1+.05*np.sqrt(variance))
        mode = 'feasibility_search'
    allowed = np.asarray([c == context for c, _ in domain.actions])
    allowed[[o['action'] for o in history]] = False
    scores[~allowed] = -np.inf
    best = float(scores.max())
    if not np.isfinite(best):
        raise RuntimeError('No unqueried candidate in the scheduled context.')
    tied = np.flatnonzero(scores >= best-1e-12*max(1., abs(best)))
    return int(rng.choice(tied)), posterior.params.tolist(), count if optimize else last_fit, mode


def peer_candidate(histories, domain, client, context):
    """Only each peer's current context incumbent and its cost are shared."""
    tested = {o['action'] for o in histories[client]}
    proposals = []
    for peer, history in enumerate(histories):
        if peer == client:
            continue
        good = [o for o in history if o['eligible'] and domain.actions[o['action']][0] == context]
        if good:
            best = min(good, key=lambda o: (o['J'], domain.actions[o['action']][1]))
            if best['action'] not in tested:
                proposals.append((best['J'], domain.actions[best['action']][1], peer, best['action']))
    return min(proposals)[-1] if proposals else None


def fresh_state(domain, cfg, clients):
    params = np.log([*cfg['gp']['length_scales'], cfg['gp']['amplitude']]).tolist()
    return dict(round=0, oracle_cache={}, methods={method: dict(
        histories=[[] for _ in clients], gp_parameters=[params.copy() for _ in clients],
        last_gp_fit=[0]*len(clients), weights=[0.]*(cfg['model']['features']+1),
        events=[], snapshots={}, learning=[], payload_bytes=0,
        scout_catalogue=None) for method in METHODS})


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
            elif n < len(domain.contexts):
                action, mode = domain.action(context, initial[n]), 'blind_initialization'
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
                        cfg['gp'], data['gp_parameters'][i], data['last_gp_fit'][i],
                        np.random.default_rng(seed_for('acquisition', seed, client['id'], n)))
                    data['gp_parameters'][i], data['last_gp_fit'][i] = params, last_fit
            plans.append(dict(method=method, client=i, local_evaluation=n+1, round=next_round,
                action=action, context=context, candidate=domain.actions[action][1], mode=mode,
                noise_seed=seed_for('episode', seed, client['id'], n), information_round=state['round']))
    return plans


def commit_round(state, plans, observations, domain, cfg, clients, scout, phi):
    """All plans are frozen before any outcome in this round is exposed."""
    touched = set()
    for plan, observation in zip(plans, observations):
        data = state['methods'][plan['method']]
        event = dict(**plan, **observation, fleet_evaluation=len(data['events'])+1)
        history = data['histories'][plan['client']]
        history.append(dict(action=plan['action'], **observation))
        event['recommendations'] = incumbents(history, domain)
        data['events'].append(event)
        budget = len(data['events'])
        if budget in cfg['fleet_budgets']:
            data['snapshots'][str(budget)] = dict(round=plan['round'],
                recommendations=[incumbents(h, domain) for h in data['histories']],
                queries_per_client=[len(h) for h in data['histories']],
                payload_bytes=data['payload_bytes'])
        touched.add(plan['method'])
    state['round'] += 1
    for method in METHODS:
        if method not in touched:
            continue
        data = state['methods'][method]
        models = [LocalCostModel(h, phi, cfg['model']['ridge']) for h in data['histories']]
        payload, gap = 0, None
        if method == 'federated_cost_bo':
            weights, payload = federated_fit(models, np.asarray(data['weights']), cfg['model'])
            data['weights'] = weights.tolist()
            gap = optimization_gap(models, weights)
        elif method == 'centralized_cost_bo':
            data['weights'] = centralized_fit(models).tolist()
            # One newly observed record per active client: five inputs, cost, eligibility.
            payload = sum(p['method'] == method for p in plans)*7*8
        elif method == 'online_gain_sharing':
            pairs = sum(sum(c is not None for c in incumbents(h, domain)) for h in data['histories'])
            payload = pairs*12*(len(clients)+1)  # candidate int32 + cost float64.
        elif method == 'single_scout' and state['round'] == cfg['scout_search_queries']:
            data['scout_catalogue'] = incumbents(data['histories'][scout], domain)
            payload = sum(c is not None for c in data['scout_catalogue'])*12*(len(clients)-1)
        data['payload_bytes'] += payload
        data['learning'].append(dict(round=state['round'], eligible_training_counts=[m.count for m in models],
            payload_bytes=payload, cumulative_payload_bytes=data['payload_bytes'], optimization_gap=gap))


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
        for n, event in enumerate(data['events']):
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
            if len(history) < len(domain.contexts):
                expected = initial[len(history)]
                if method == 'single_scout' and i != scout:
                    frozen = incumbents(data['histories'][scout][:cfg['scout_search_queries']], domain)
                    expected = frozen[domain.contexts.index(event['context'])] or domain.candidates[0]
                if event['candidate'] != expected:
                    raise RuntimeError('Blind initialization or scout verification changed.')
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
                        if e['client'] == i and e['local_evaluation'] <= len(domain.contexts)]
                       for i in range(len(clients))]
            if common_initial is None:
                common_initial = initial
            elif initial != common_initial:
                raise RuntimeError('Non-scout methods have different initialization observations.')
