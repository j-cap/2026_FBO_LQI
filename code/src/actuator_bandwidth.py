"""Isolated actuator-aware LQI and nonlinear evaluator; legacy experiments unchanged."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm, logm, solve_discrete_are
from src.nonlinear_plant import nonlinear_bicycle_derivatives


def continuous_vehicle(ad, bd, ts):
    """Principal real ZOH embedding of the identified bare-vehicle model.

    Valid only for a model identified with actual held wheel angle as input.
    This is an equivalent local linear model, not identification of physical parameters.
    """
    block = np.block([[ad, bd], [np.zeros((1, 2)), np.ones((1, 1))]])
    gen = logm(block) / ts
    if not np.isfinite(gen).all() or np.max(np.abs(np.imag(gen))) > 1e-8:
        raise ValueError('Identified vehicle has no acceptable principal real ZOH embedding')
    gen = np.real(gen)
    if not np.allclose(expm(gen * ts), block, atol=1e-8, rtol=1e-8) or np.max(abs(gen[-1])) > 1e-8:
        raise ValueError('ZOH embedding reconstruction failed')
    return gen[:2, :2], gen[:2, 2:3]


def augment_discrete(ac, bc, tau, ts):
    if tau <= 0 or ts <= 0:
        raise ValueError('tau and sampling time must be positive')
    a = np.zeros((3, 3)); a[:2, :2] = ac; a[:2, 2:3] = bc; a[2, 2] = -1/tau
    b = np.array([[0.], [0.], [1/tau]])
    zoh = expm(np.block([[a, b], [np.zeros((1, 4))]]) * ts)
    return zoh[:3, :3], zoh[:3, 3:4]


def design(ad, bd, xi, scales, ts, fixed_r=1., leak=0.002):
    # eta = Ts * legacy sample-sum integrator; same sign (r-r_ref).
    q = 10. ** np.asarray(xi) / np.asarray(scales)**2
    Q = np.diag([q[0], q[1], 0., q[2]/ts**2])
    cr = np.array([[0., 1., 0.]])
    ai = np.block([[ad, np.zeros((3, 1))], [ts*cr, np.array([[1-leak]])]])
    bi = np.vstack([bd, [[0.]]])
    P = solve_discrete_are(ai, bi, Q, np.array([[fixed_r]]))
    K = np.linalg.solve(bi.T @ P @ bi + fixed_r, bi.T @ P @ ai)
    radius = float(max(abs(np.linalg.eigvals(ai-bi@K))))
    kr = float(1 / (cr @ np.linalg.solve(np.eye(3)-ad+bd@K[:, :3], bd)).item())
    if not np.isfinite(K).all() or not np.isfinite(kr) or radius >= 1-1e-8:
        raise ValueError('Invalid or unstable augmented feedback')
    return K.ravel(), kr, radius


@dataclass
class Score:
    cost: float
    feasible: bool
    reason: str
    max_beta: float = 0.
    max_delta: float = 0.
    max_rate: float = 0.
    saturation_fraction: float = 0.
    spectral_radius: float = float('nan')


def step(x, u, params, tau, dt, mu):
    """Coupled RK4: actual steering changes inside each integration interval."""
    def rhs(state):
        return np.r_[nonlinear_bicycle_derivatives(state[:2], state[2], params, mu=mu),
                     (u-state[2])/tau]
    k1=rhs(x); k2=rhs(x+dt*k1/2); k3=rhs(x+dt*k2/2); k4=rhs(x+dt*k3)
    return x+dt*(k1+2*k2+2*k3+k4)/6


def evaluate(client, ac, bc, tau, xi, bank, role, controller, ts, substeps=5):
    ad, bd = augment_discrete(ac, bc, tau, ts)
    try:
        K, kr, radius = design(ad, bd, xi, controller['state_scale'], ts, controller['fixed_R'])
    except (ValueError, np.linalg.LinAlgError) as e:
        return Score(float('nan'), False, 'synthesis: '+str(e))
    costs=[]; max_beta=max_delta=max_rate=0.; saturated=count=0
    beta_limit=np.deg2rad(controller['beta_max_deg'])
    # Legacy configuration is radians per sampling interval, convert to rad/s.
    rate_limit=controller['u_rate_max']/ts
    for ep in bank.episodes:
        if ep.role != role: continue
        _, ref = bank.refs[ep.T_lc] if bank.refs is not None else (bank.t, bank.r_ref)
        rng=np.random.RandomState(ep.seed)
        x=np.zeros(3); eta=kr*float(ref[0])/K[3]; errors=[]
        for rr in ref[:-1]:
            raw=-K[:3]@x-K[3]*eta+kr*rr
            u=float(np.clip(raw, controller['u_min'], controller['u_max']))
            y=x[1]+controller['noise_std']*rng.randn()
            errors.append((y-rr)**2)
            eta=(1-.002)*eta+ts*(y-rr+.2*(u-raw))
            saturated+=int(abs(u-raw)>1e-10);count+=1
            # First-order response has its peak derivative at the interval start.
            max_rate=max(max_rate,abs((u-x[2])/tau))
            for _ in range(substeps):
                x=step(x,u,client.params,tau,ts/substeps,controller.get('tire_mu',1.))
                max_beta=max(max_beta,abs(x[0]));max_delta=max(max_delta,abs(x[2]))
                if not np.isfinite(x).all() or max_beta > beta_limit:
                    return Score(float('nan'),False,'beta_limit',max_beta,max_delta,max_rate,
                                 saturated/count,radius)
            x[:2]+=rng.multivariate_normal(np.zeros(2),np.diag(controller['process_noise']))
        costs.append(float(np.sqrt(np.mean(errors))))
    if not costs: raise ValueError('No episodes for requested role')
    # Rate is a feasibility constraint, not an unannounced extra rate-limiter model.
    feasible=max_rate<=rate_limit and max_delta<=max(abs(controller['u_min']),abs(controller['u_max']))+1e-8
    return Score(float(np.mean(costs)),feasible,'' if feasible else 'actuator_limit',
                 max_beta,max_delta,max_rate,saturated/count,radius)
