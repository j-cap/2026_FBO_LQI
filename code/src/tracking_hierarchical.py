"""Joint shared/local Bayesian cost learning under the frozen peer-first protocol.

Clients send Gaussian information messages, not histories. Each eligible label
is conditioned on once. This is exact federated Gaussian inference, not FedAvg.
Only this arm's paid earlier-round observations enter the cost posterior.
"""
from copy import deepcopy

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp

from . import tracking_peer_cost as peer
from .tracking_budget import feasibility_probability
from .tracking_online import features, program, seed_for
from .tracking_online_lcb import acquisition_values


SHARED = 'peer_hierarchical_bayes_bo'
LOCAL = 'peer_local_bayes_bo'
METHODS = [SHARED, LOCAL]


class ClientBlock:
    """Local likelihood after integrating a vehicle's random coefficients."""
    def __init__(self, history, F, fraction, noise_variance):
        good = [o for o in history if o['eligible']]
        self.X = F[[o['action'] for o in good]]
        self.y = np.log([o['J'] for o in good])
        self.count, self.width = len(good), F.shape[1]
        self.fraction = fraction
        self.factor = None
        if self.count:
            R = (1-fraction)*(self.X@self.X.T)
            R.flat[::self.count+1] += noise_variance
            self.factor = cho_factor(R, lower=True, check_finite=False)

    def information_message(self):
        width = self.width
        if not self.count:
            return None
        inverse_X = cho_solve(self.factor, self.X, check_finite=False)
        inverse_y = cho_solve(self.factor, self.y, check_finite=False)
        precision = self.fraction*(self.X.T@inverse_X)
        return dict(precision_upper=precision[np.triu_indices(width)],
            information=np.sqrt(self.fraction)*(self.X.T@inverse_y),
            quadratic=float(self.y@inverse_y),
            log_determinant=float(2*np.log(np.diag(self.factor[0])).sum()), count=self.count)

    def predict(self, F, global_mean, global_covariance):
        H = np.sqrt(self.fraction)*F
        mean = np.zeros(len(F))
        variance = (1-self.fraction)*np.sum(F*F, axis=1)
        if self.count:
            cross = (1-self.fraction)*(F@self.X.T)
            inverse_cross = cho_solve(self.factor, cross.T, check_finite=False)
            mean += cross@cho_solve(self.factor, self.y, check_finite=False)
            H -= cross@cho_solve(self.factor, np.sqrt(self.fraction)*self.X, check_finite=False)
            variance -= np.einsum('ij,ji->i', cross, inverse_cross)
        mean += H@global_mean
        variance += np.einsum('ij,ij->i', H@global_covariance, H)
        if np.min(variance) < -1e-8:
            raise RuntimeError('Negative joint predictive variance.')
        return mean, np.maximum(variance, 0.)


def aggregate_messages(messages, width):
    """Server interface: no local X, labels, identities, families or plant inputs."""
    upper = np.triu_indices(width)
    precision, information = np.eye(width), np.zeros(width)
    quadratic, log_determinant, count = 0., 0., 0
    for message in messages:
        if message is None:
            continue
        block = np.zeros((width, width))
        block[upper] = message['precision_upper']
        precision += block+block.T-np.diag(np.diag(block))
        information += message['information']
        quadratic += message['quadratic']
        log_determinant += message['log_determinant']
        count += message['count']
    factor = cho_factor(precision, lower=True, check_finite=False)
    mean = cho_solve(factor, information, check_finite=False)
    covariance = cho_solve(factor, np.eye(width), check_finite=False)
    log_evidence = -.5*(quadratic-information@mean+log_determinant+
        2*np.log(np.diag(factor[0])).sum()+count*np.log(2*np.pi))
    return mean, covariance, float(log_evidence)


def model_payload_bytes(counts, width, components):
    """Packed symmetric matrices + vectors/scalars; float64, no transport overhead.

    Uplink: precision triangle, information vector, quadratic, logdet, count.
    Downlink: covariance triangle, mean vector, component log evidence.
    All clients receive the broadcast, including those with no eligible cost.
    """
    active = sum(c > 0 for c in counts)
    if not active:
        return 0
    triangle = width*(width+1)//2
    return int(8*components*(active*(triangle+width+3)+len(counts)*(triangle+width+1)))


def cost_posterior(histories, phi, gp, cfg, shared=True):
    """Mixture over a public prior on sharing strength; no held-out tuning."""
    fractions = np.asarray(cfg['sharing_fractions'] if shared else [0.], float)
    prior = np.asarray(cfg['prior_weights'] if shared else [1.], float)
    if (not len(fractions) or len(fractions) != len(prior) or
        not np.isfinite(fractions).all() or np.any((fractions < 0) | (fractions > 1)) or
        not np.isfinite(prior).all() or np.any(prior <= 0)):
        raise ValueError('Invalid public sharing-strength prior.')
    scale = np.full(phi.shape[1], gp['amplitude'])
    scale[0] = np.sqrt(gp['constant_variance'])
    F = phi*scale
    noise_variance = gp['noise_std_log_J']**2+1e-9
    means, variances, evidence = [], [], []
    for fraction in fractions:
        blocks = [ClientBlock(h, F, fraction, noise_variance) for h in histories]
        if shared:
            mean, covariance, log_evidence = aggregate_messages(
                [b.information_message() for b in blocks], F.shape[1])
        else:
            # rho=0: the global coefficient is unused; no peer costs enter a fit.
            mean, covariance, log_evidence = np.zeros(F.shape[1]), np.eye(F.shape[1]), 0.
        predictions = [b.predict(F, mean, covariance) for b in blocks]
        means.append([p[0] for p in predictions]); variances.append([p[1] for p in predictions])
        evidence.append(log_evidence)
    means, variances = np.asarray(means), np.asarray(variances)
    log_weights = np.log(prior/prior.sum())+evidence
    weights = np.exp(log_weights-logsumexp(log_weights))
    mean = np.einsum('k,kia->ia', weights, means)
    # Total variance includes disagreement between sharing-strength components.
    variance = np.einsum('k,kia->ia', weights, variances+(means-mean[None, :, :])**2)
    counts = [sum(o['eligible'] for o in h) for h in histories]
    return dict(mean=mean.tolist(), variance=variance.tolist(), sharing_fractions=fractions.tolist(),
        sharing_weights=weights.tolist(), component_log_evidence=evidence,
        expected_sharing_fraction=float(weights@fractions), eligible_training_counts=counts,
        model_payload_bytes=model_payload_bytes(counts, phi.shape[1], len(fractions)) if shared else 0)


def fresh_state(domain, cfg, clients):
    allowed = peer.METHODS+METHODS
    if not cfg['methods'] or not set(cfg['methods']).issubset(allowed):
        raise ValueError('Unknown joint-cost experiment method.')
    template = peer.fresh_state(domain, dict(cfg, methods=[peer.METHODS[0]]), clients)
    data = template['methods'][peer.METHODS[0]]
    template['methods'] = {m: deepcopy(data) for m in cfg['methods']}
    for method in METHODS:
        if method in template['methods']:
            template['methods'][method]['cost_posterior'] = None
    return template


def choose(history, domain, context, posterior, client, gp, acquisition, rng):
    mean, variance = np.asarray(posterior['mean'][client]), np.asarray(posterior['variance'][client])
    probability = feasibility_probability(history, domain, np.full(len(domain.actions), .5), gp)
    values = acquisition_values(mean, variance, probability, acquisition)
    allowed = np.asarray([c == context for c, _ in domain.actions])
    allowed[[o['action'] for o in history]] = False
    values[~allowed] = np.inf
    best = float(values.min())
    if not np.isfinite(best):
        raise RuntimeError('No unqueried candidate in the scheduled context.')
    tied = np.flatnonzero(values <= best+1e-12*max(1., abs(best)))
    return int(rng.choice(tied))


def plan_round(state, domain, cfg, clients, seed, phi):
    old = [m for m in cfg['methods'] if m in peer.METHODS]
    plans = peer.plan_round(state, domain, dict(cfg, methods=old), clients, seed, phi) if old else []
    for method in [m for m in cfg['methods'] if m in METHODS]:
        data = state['methods'][method]
        for i, client in enumerate(clients):
            history = data['histories'][i]
            n = len(history)
            if n >= cfg['queries_per_client']:
                continue
            contexts, initial = program(domain, cfg, seed, client['id'])
            context = contexts[n]
            metadata = dict.fromkeys(peer.PEER_FIELDS)
            if not n:
                action, mode = domain.action(context, initial[0]), 'factory_initialization'
            else:
                action, metadata = peer.peer_proposal(data['histories'], data['events'], domain, i, context)
                mode = 'peer_gain_verification'
                if action is None:
                    if data['cost_posterior']['information_round'] != state['round']:
                        raise RuntimeError('Cost posterior violates the prior-round barrier.')
                    action = choose(history, domain, context, data['cost_posterior'], i, cfg['gp'],
                        cfg['acquisition'], np.random.default_rng(seed_for('acquisition', seed, client['id'], n)))
                    mode = 'feasibility_penalized_LCB'
            plans.append(dict(method=method, client=i, local_evaluation=n+1, round=state['round']+1,
                action=action, context=context, candidate=domain.actions[action][1], mode=mode,
                noise_seed=seed_for('episode', seed, client['id'], n),
                information_round=state['round'], **metadata))
    return plans


def commit_round(state, plans, observations, domain, cfg, clients, phi):
    # Reuse exact query accounting, incumbents, peer publications and old-arm fits.
    peer.commit_round(state, plans, observations, domain, cfg, clients, phi)
    for method in [m for m in cfg['methods'] if m in METHODS]:
        data = state['methods'][method]
        posterior = cost_posterior(data['histories'], phi, cfg['gp'], cfg['hierarchical'], method == SHARED)
        posterior['information_round'] = state['round']
        data['cost_posterior'] = posterior
        payload = posterior['model_payload_bytes']
        data['model_payload_bytes'] += payload; data['payload_bytes'] += payload
        log = data['learning'][-1]
        log.update(model_payload_bytes=payload, payload_bytes=log['peer_payload_bytes']+payload,
            cumulative_model_payload_bytes=data['model_payload_bytes'], cumulative_payload_bytes=data['payload_bytes'],
            expected_sharing_fraction=posterior['expected_sharing_fraction'],
            sharing_weights=posterior['sharing_weights'], component_log_evidence=posterior['component_log_evidence'])


def run_fleet(domain, cfg, clients, seed, observe_batch, checkpoint=None, state=None, stop_round=None):
    phi = features(domain, cfg['model'])
    state = deepcopy(state) if state is not None else fresh_state(domain, cfg, clients)
    if state['round']:
        validate_state(state, domain, cfg, clients, seed)
    limit = cfg['queries_per_client'] if stop_round is None else stop_round
    if not state['round'] <= limit <= cfg['queries_per_client']:
        raise ValueError('Invalid stopping round.')
    while state['round'] < limit:
        plans = plan_round(state, domain, cfg, clients, seed, phi)
        observations = observe_batch(plans, state['oracle_cache'])
        if len(observations) != len(plans):
            raise RuntimeError('Oracle returned the wrong number of observations.')
        commit_round(state, plans, observations, domain, cfg, clients, phi)
        if checkpoint is not None:
            checkpoint(state)
    return state


def validate_state(state, domain, cfg, clients, seed, complete=False):
    # The frozen validator already audits peer priority, noise pairing, every
    # charged request, eligibility, incumbents and pre-training budget snapshots.
    accounting = deepcopy(state)
    for method in METHODS:
        if method not in accounting['methods']:
            continue
        data = accounting['methods'][method]
        for snapshot in data['snapshots'].values():
            snapshot['model_payload_bytes'] = 0
            snapshot['payload_bytes'] = snapshot['peer_payload_bytes']
        for log in data['learning']:
            log.update(model_payload_bytes=0, payload_bytes=log['peer_payload_bytes'],
                cumulative_model_payload_bytes=0, cumulative_payload_bytes=log['cumulative_peer_payload_bytes'])
        data['model_payload_bytes'] = 0
        data['payload_bytes'] = data['peer_payload_bytes']
    peer.validate_state(accounting, domain, cfg, clients, seed, complete)
    phi = features(domain, cfg['model'])
    for method in [m for m in cfg['methods'] if m in METHODS]:
        data = state['methods'][method]
        total = 0
        for log in data['learning']:
            expected = (model_payload_bytes(log['eligible_training_counts'], phi.shape[1],
                len(cfg['hierarchical']['sharing_fractions'])) if method == SHARED else 0)
            total += expected
            if (log['model_payload_bytes'] != expected or log['payload_bytes'] != expected+log['peer_payload_bytes'] or
                log['cumulative_model_payload_bytes'] != total or
                log['cumulative_payload_bytes'] != total+log['cumulative_peer_payload_bytes']):
                raise RuntimeError('Joint-model information-message payload differs.')
        if data['model_payload_bytes'] != total or data['payload_bytes'] != total+data['peer_payload_bytes']:
            raise RuntimeError('Joint-model cumulative payload differs.')
        for snapshot in data['snapshots'].values():
            prior = data['learning'][snapshot['round']-2] if snapshot['round'] > 1 else {}
            expected = prior.get('cumulative_model_payload_bytes', 0)
            if snapshot['model_payload_bytes'] != expected or snapshot['payload_bytes'] != expected+snapshot['peer_payload_bytes']:
                raise RuntimeError('Snapshot includes unused/later model communication.')
        if state['round']:
            expected = cost_posterior(data['histories'], phi, cfg['gp'], cfg['hierarchical'], method == SHARED)
            actual = data['cost_posterior']
            if actual['information_round'] != state['round']:
                raise RuntimeError('Stale or future joint-cost posterior.')
            for key in expected:
                if not np.allclose(actual[key], expected[key], rtol=1e-10, atol=1e-10):
                    raise RuntimeError('Stored joint-cost posterior differs: '+key)


def replay_new_methods(state, domain, cfg, clients, seed):
    """Audit all new-arm choices against recorded labels; no physical rollout."""
    methods = [m for m in cfg['methods'] if m in METHODS]
    def recorded(plans, cache):
        observations = []
        for plan in plans:
            event = state['methods'][plan['method']]['events'][(plan['round']-1)*len(clients)+plan['client']]
            if any(plan[k] != event[k] for k in plan):
                raise RuntimeError('Recorded joint-model choice/provenance differs from replay.')
            observations.append({k: event[k] for k in peer.OBSERVATION_FIELDS if k != 'action'})
        return observations
    replay = run_fleet(domain, dict(cfg, methods=methods), clients, seed, recorded, stop_round=state['round'])
    for method in methods:
        if replay['methods'][method]['events'] != state['methods'][method]['events']:
            raise RuntimeError('Joint-model event replay differs.')
        for a, b in zip(replay['methods'][method]['learning'], state['methods'][method]['learning']):
            for key in ('expected_sharing_fraction', 'sharing_weights', 'component_log_evidence'):
                if not np.allclose(a[key], b[key], rtol=1e-10, atol=1e-10):
                    raise RuntimeError('Joint-model learning replay differs: '+key)
