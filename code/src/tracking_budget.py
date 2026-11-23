"""Blind finite-pool contextual-BO replay using calibration observations only.

All acquisition and incumbent choices are made before evaluation on held-out
noise. No vehicle simulator or new plant parameter distribution is introduced.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import ndtr

from .tracking_context import stable_minimum


@dataclass
class Domain:
    X: np.ndarray
    actions: list
    contexts: list
    candidates: list
    frequencies: list
    families: dict
    nominal_index: int

    def action(self, context, candidate):
        return self.actions.index((context, candidate))


def domain_from_manifest(manifest):
    refs, pool = manifest['references'], manifest['candidates']
    bounds = manifest['configuration']['gains']['log10_multiplier_bounds']
    mult = np.asarray([c['multipliers'] for c in pool])
    gx = (np.log10(mult)-bounds[0])/(bounds[1]-bounds[0])
    speed = np.asarray([r['speed'] for r in refs], float)
    freq = np.log(np.asarray([r['frequency'] for r in refs], float))
    cx = np.column_stack(((speed-speed.min())/max(np.ptp(speed), 1e-12),
                           (freq-freq.min())/max(np.ptp(freq), 1e-12)))
    actions = [(r['id'], c['id']) for r in refs for c in pool]
    X = np.array([np.r_[g, context] for context in cx for g in gx])
    return Domain(X, actions, [r['id'] for r in refs], [c['id'] for c in pool],
                  [r['frequency'] for r in refs],
                  {v['id']: v['family'] for v in manifest['vehicles']}, 0)


def source_ids(domain, target, protocol):
    if protocol == 'leave_vehicle_out':
        return sorted(v for v in domain.families if v != target)
    if protocol == 'leave_family_out':
        return sorted(v for v in domain.families if domain.families[v] != domain.families[target])
    raise ValueError('Unknown exclusion protocol.')


def matern(X, Z, lengths, amplitude=1., constant=.25):
    delta = (X[:, None, :]-Z[None, :, :])/np.asarray(lengths)
    rr = np.sqrt(np.maximum(np.sum(delta*delta, axis=2), 0.))
    sr = np.sqrt(5.)*rr
    return amplitude**2*(1+sr+5*rr*rr/3)*np.exp(-sr)+constant


class Posterior:
    def __init__(self, domain, observed, prior_mean, cfg, params, optimize=False):
        self.domain, self.cfg = domain, cfg
        self.prior_mean = np.asarray(prior_mean)
        good = [o for o in observed if o['eligible']]
        self.indices = np.array([o['action'] for o in good], int)
        self.residual = np.log([o['J'] for o in good])-self.prior_mean[self.indices]
        self.params = np.asarray(params, float).copy()
        if optimize and len(good) >= 4:
            bounds = [tuple(np.log(cfg['length_bounds']))]*domain.X.shape[1]
            bounds += [tuple(np.log(cfg['amplitude_bounds']))]
            starts = [self.params, np.log([*cfg['length_scales'], cfg['amplitude']])]
            results = [minimize(self._loss, start, method='L-BFGS-B', bounds=bounds,
                                options={'maxiter': cfg['optimizer_maxiter']})
                       for start in starts[:1+cfg['optimizer_restarts']]]
            valid = [r for r in results if np.isfinite(r.fun)]
            if valid:
                self.params = min(valid, key=lambda r: r.fun).x
        self.factor, self.alpha = self._factor(self.params) if len(good) else (None, None)

    def _kernel(self, A, B, params):
        theta = np.exp(params)
        return matern(A, B, theta[:-1], theta[-1], self.cfg['constant_variance'])

    def _factor(self, params):
        X = self.domain.X[self.indices]
        K = self._kernel(X, X, params)
        K.flat[::len(K)+1] += self.cfg['noise_std_log_J']**2+1e-9
        factor = cho_factor(K, lower=True, check_finite=False)
        return factor, cho_solve(factor, self.residual, check_finite=False)

    def _loss(self, params):
        try:
            factor, alpha = self._factor(params)
            return float(.5*self.residual@alpha+np.log(np.diag(factor[0])).sum()+
                         .5*len(alpha)*np.log(2*np.pi))
        except (ValueError, np.linalg.LinAlgError):
            return 1e30

    def predict(self):
        mean = self.prior_mean.copy()
        variance = np.full(len(mean), np.exp(self.params[-1])**2+self.cfg['constant_variance'])
        if self.factor is not None:
            cross = self._kernel(self.domain.X, self.domain.X[self.indices], self.params)
            mean += cross@self.alpha
            variance -= np.einsum('ij,ji->i', cross,
                                  cho_solve(self.factor, cross.T, check_finite=False))
        return mean, np.maximum(variance, 1e-12)


def lognormal_ei(mean, variance, best):
    """Expected improvement in original J units when log(J) is Gaussian."""
    sigma = np.sqrt(np.maximum(variance, 1e-12))
    z = (np.log(np.maximum(best, 1e-12))-mean)/sigma
    expected = np.exp(np.clip(mean+.5*variance, -700, 700))
    return np.maximum(best*ndtr(z)-expected*ndtr(z-sigma), 0.)


def _eligible(frame):
    return frame.complete & frame.feasible & frame.integration_resolved


def prepare_source(calibration, domain, target, protocol, cfg):
    """Only other vehicles' calibration data enter the catalogue or prior."""
    if not (calibration.role == 'calibration').all():
        raise ValueError('Source preparation accepts calibration rows only.')
    ids = source_ids(domain, target, protocol)
    if not ids:
        raise ValueError('No permitted sources.')
    src = calibration[calibration.vehicle.isin(ids)].copy()
    src['eligible'] = _eligible(src)
    summary = src.groupby(['context', 'candidate']).agg(
        eligible=('eligible', 'mean'), all_eligible=('eligible', 'all'),
        observations=('eligible', 'count'))
    usable = src[src.eligible].groupby(['context', 'candidate']).J.agg(['mean', 'var', 'count'])
    # Feasibility and objective are separate. Ineligible source trials never
    # masquerade as finite good costs, and their feasibility labels are retained.
    objective, noise, valid = [], [], []
    probability = np.array([summary.loc[action, 'eligible'] for action in domain.actions])
    for i, action in enumerate(domain.actions):
        if action in usable.index:
            row = usable.loc[action]
            objective.append(np.log(max(float(row['mean']), 1e-12)))
            sampling_var = (float(row['var'])/row['count']/max(row['mean']**2, 1e-12)
                            if row['count'] > 1 else 0.)
            noise.append(cfg['source_noise_std_log_J']**2+max(sampling_var, 0.))
            valid.append(i)
    if not valid:
        raise ValueError('No eligible source objectives.')
    K = matern(domain.X[valid], domain.X[valid], cfg['length_scales'],
               cfg['amplitude'], cfg['constant_variance'])
    K.flat[::len(K)+1] += np.asarray(noise)+1e-9
    alpha = cho_solve(cho_factor(K, lower=True), objective)
    mu = matern(domain.X, domain.X[valid], cfg['length_scales'],
                 cfg['amplitude'], cfg['constant_variance'])@alpha
    # Catalogue: one common gain per frequency, robust over both speeds and all
    # permitted sources/noise repeats. No target parameter or cost is used.
    catalogue = {}
    for frequency in sorted(set(domain.frequencies)):
        contexts = [c for c, f in zip(domain.contexts, domain.frequencies) if f == frequency]
        good = summary.loc[summary.index.get_level_values('context').isin(contexts)]
        good = good.groupby('candidate').all_eligible.all()
        candidates = sorted(good[good].index)
        costs = [float(src[(src.context.isin(contexts)) & (src.candidate == c)].J.mean())
                 for c in candidates]
        choice = candidates[stable_minimum(costs, candidates)] if candidates else None
        for c in contexts:
            catalogue[c] = choice
    return dict(source_ids=ids, source_observations=len(src),
                source_eligible_observations=int(src.eligible.sum()), catalogue=catalogue,
                prior_mean=mu.tolist(), prior_feasibility=probability.tolist())


class TargetOracle:
    """The learner can reveal one archived rollout at a time; no validation API."""
    def __init__(self, calibration, target, noise_seed, domain):
        if not (calibration.role == 'calibration').all():
            raise ValueError('An optimization oracle cannot contain validation rows.')
        rows = calibration[(calibration.vehicle == target) & (calibration.seed == noise_seed)]
        if len(rows) != len(domain.actions):
            raise ValueError('Target oracle is incomplete.')
        self.__rows = rows.set_index(['context', 'candidate'])
        self.__domain = domain
        self.__seen = set()

    def observe(self, action):
        if action in self.__seen:
            raise ValueError('A target rollout cannot be queried twice.')
        self.__seen.add(action)
        row = self.__rows.loc[self.__domain.actions[action]]
        eligible = bool(row.complete and row.feasible and row.integration_resolved)
        return dict(action=int(action), eligible=eligible,
                    J=float(row.J) if eligible else None,
                    complete=bool(row.complete), feasible=bool(row.feasible),
                    integration_resolved=bool(row.integration_resolved))


def initial_design(domain, seed):
    """One point per context: original gain first, then maximin gain-space points."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(domain.contexts))
    m = len(domain.candidates)
    selected, actions = [], []
    for j, ci in enumerate(order):
        if j == 0:
            candidate = domain.nominal_index
        else:
            dist = np.min(np.sum((domain.X[:m, None, :3]-domain.X[selected, :3])**2, axis=2), axis=1)
            tied = np.flatnonzero(dist >= dist.max()-1e-12)
            candidate = int(rng.choice(tied))
        selected.append(candidate)
        actions.append(int(ci*m+candidate))
    return actions


def incumbents(observed, domain):
    chosen = []
    for context in domain.contexts:
        good = [o for o in observed if o['eligible'] and domain.actions[o['action']][0] == context]
        if good:
            labels = [domain.actions[o['action']][1] for o in good]
            best = good[stable_minimum([o['J'] for o in good], labels)]
            chosen.append(domain.actions[best['action']][1])
        else:
            chosen.append(None)
    return chosen


def feasibility_probability(observed, domain, prior, cfg):
    strength = cfg['feasibility_prior_strength']
    p0 = np.clip(prior, *cfg['feasibility_prior_clip'])
    if not observed:
        return p0
    weights = matern(domain.X, domain.X[[o['action'] for o in observed]],
                     cfg['length_scales'], 1., 0.)
    labels = np.asarray([o['eligible'] for o in observed], float)
    return (strength*p0+weights@labels)/(strength+weights.sum(axis=1))


def propose(observed, domain, prior_mean, prior_feasibility, cfg, params, rng, optimize):
    posterior = Posterior(domain, observed, prior_mean, cfg, params, optimize)
    mean, var = posterior.predict()
    probability = feasibility_probability(observed, domain, prior_feasibility, cfg)
    best = {}
    for o in observed:
        if o['eligible']:
            context = domain.actions[o['action']][0]
            best[context] = min(best.get(context, float('inf')), o['J'])
    missing = set(domain.contexts)-set(best)
    if missing:
        score = probability*(1+.05*np.sqrt(var))
        score[[i for i, (c, _) in enumerate(domain.actions) if c not in missing]] = -np.inf
        mode = 'feasibility_search'
    else:
        thresholds = np.asarray([best[c] for c, _ in domain.actions])
        score = probability*lognormal_ei(mean, var, thresholds)
        mode = 'feasibility_weighted_EI'
    score[[o['action'] for o in observed]] = -np.inf
    maximum = float(np.max(score))
    if not np.isfinite(maximum):
        raise RuntimeError('No unqueried target action remains.')
    tied = np.flatnonzero(score >= maximum-1e-12*max(1., abs(maximum)))
    return int(rng.choice(tied)), posterior.params, mode


def run_method(method, oracle, domain, source, cfg, seed, max_budget):
    rng = np.random.default_rng(seed)
    initial = initial_design(domain, seed)
    if method in ('catalogue', 'catalogue_warmstart_bo', 'pooled_bo'):
        initial = [domain.action(domain.actions[a][0], source['catalogue'].get(domain.actions[a][0])
                                 or domain.candidates[domain.nominal_index]) for a in initial]
    prior = np.asarray(source['prior_mean']) if method == 'pooled_bo' else np.zeros(len(domain.actions))
    p0 = np.asarray(source['prior_feasibility']) if method == 'pooled_bo' else np.full(len(prior), .5)
    params = np.log([*cfg['length_scales'], cfg['amplitude']])
    observed, steps = [], []
    last_optimized = 0
    limit = len(domain.contexts) if method == 'catalogue' else max_budget
    for k in range(limit):
        if k < len(initial):
            action, mode = initial[k], 'source_initialization' if method in (
                'catalogue', 'catalogue_warmstart_bo', 'pooled_bo') else 'space_filling_initialization'
        elif method == 'random':
            counts = {c: sum(domain.actions[o['action']][0] == c for o in observed) for c in domain.contexts}
            minimum = min(counts.values())
            available = [i for i, (c, _) in enumerate(domain.actions) if counts[c] == minimum
                         and i not in {o['action'] for o in observed}]
            action, mode = int(rng.choice(available)), 'balanced_random'
        else:
            ngood = sum(o['eligible'] for o in observed)
            optimize = ngood >= 4 and ngood-last_optimized >= cfg['optimize_every']
            action, params, mode = propose(observed, domain, prior, p0, cfg, params, rng, optimize)
            if optimize:
                last_optimized = ngood
        obs = oracle.observe(action)
        observed.append(obs)
        steps.append(dict(evaluation=k+1, **obs, context=domain.actions[action][0],
                          candidate=domain.actions[action][1], mode=mode,
                          recommendations=incumbents(observed, domain)))
    return dict(steps=steps, actual_rollouts=len(steps), final_gp_parameters=params.tolist())


def run_suite(calibration, domain, target, source, cfg, seed, noise_seed):
    # Validation rows and dense target-reference values are never passed here.
    return {method: run_method(method, TargetOracle(calibration, target, noise_seed, domain),
                               domain, source, cfg['gp'], seed, max(cfg['budgets']))
            for method in cfg['methods']}


def dense_references(calibration, validation, domain):
    cal = calibration.copy(); cal['eligible'] = _eligible(cal)
    val = validation.copy(); val['eligible'] = _eligible(val)
    cg = cal.groupby(['vehicle', 'context', 'candidate']).agg(eligible=('eligible', 'all'), J=('J', 'mean'))
    vg = val.groupby(['vehicle', 'context', 'candidate']).agg(eligible=('eligible', 'all'), J=('J', 'mean'))
    reference = {}
    for vehicle in domain.families:
        for context in domain.contexts:
            g = cg.loc[(vehicle, context)]
            g = g[g.eligible]
            labels = sorted(g.index)
            if not labels:
                raise ValueError('No dense calibration reference in a target context.')
            candidate = labels[stable_minimum([g.loc[c, 'J'] for c in labels], labels)]
            v = vg.loc[(vehicle, context, candidate)]
            reference[vehicle, context] = dict(candidate=candidate, J=float(v.J), eligible=bool(v.eligible))
    return reference, {key: dict(eligible=bool(row.eligible), J=float(row.J))
                       for key, row in vg.iterrows()}


def evaluate_job(job, calibration, validation, domain, cfg, evaluation_tables=None):
    """Independent evaluator, called only after complete query traces are frozen."""
    reference, vg = evaluation_tables or dense_references(calibration, validation, domain)
    rows, context_rows = [], []
    for method, result in job['methods'].items():
        first_success = None
        evaluated = []
        for step in result['steps']:
            cells = []
            for context, candidate in zip(domain.contexts, step['recommendations']):
                ref = reference[job['target'], context]
                v = vg[(job['target'], context, candidate)] if candidate is not None else None
                eligible = v is not None and bool(v['eligible'])
                gap = float(v['J']/ref['J']-1) if eligible and ref['eligible'] else None
                cells.append(dict(context=context, candidate=candidate, eligible=eligible,
                                  J=float(v['J']) if eligible else None, relative_gap=gap,
                                  success=gap is not None and gap <= cfg['relative_tolerance']))
            joint = all(c['success'] for c in cells)
            if joint and first_success is None:
                first_success = step['evaluation']
            evaluated.append(cells)
        for budget in cfg['budgets']:
            n = min(budget, len(result['steps']))
            cells = evaluated[n-1]
            history = result['steps'][:n]
            all_eligible = all(c['eligible'] for c in cells)
            common = dict(target=job['target'], protocol=job['protocol'], seed=job['seed'],
                          method=method, budget=budget, actual_rollouts=n)
            rows.append(dict(**common, fully_feasible=all_eligible,
                             success_fraction=float(np.mean([c['success'] for c in cells])),
                             joint_success=all(c['success'] for c in cells),
                             J=float(np.mean([c['J'] for c in cells])) if all_eligible else None,
                             failed_queries=sum(not s['feasible'] or not s['complete'] for s in history),
                             unresolved_queries=sum(not s['integration_resolved'] for s in history),
                             first_success=first_success if first_success is not None and first_success <= budget else None,
                             restricted_evals_to_success=min(first_success or budget, budget)))
            context_rows.extend(dict(**common, **cell) for cell in cells)
    return rows, context_rows
