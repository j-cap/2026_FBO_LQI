"""Corrected actuator evaluator for the paired anti-windup audit.

The legacy module remains immutable: fleet admission and old manifests depend on it.
All plant, noise, synthesis and scoring equations below match that evaluator.
Only the integral update changes. Do not use old optimized weights as corrected optima.
"""
import numpy as np
from src.actuator_bandwidth import augment_discrete, design, Score, step


def integral_update(eta, error, raw, applied, ki, ts, mode="tracking", tracking_time=.2):
    """u_raw = -Kx - ki*eta + kr*r; eta integrates measured yaw error.

    tracking: eta_dot = error + (raw-applied)/(ki*T_aw).
    Thus the integral contribution to raw moves T_s/T_aw toward the limit.
    sign_only isolates the historical sign error while retaining its gain .2.
    The leak .002 is retained per sample, exactly as in synthesis and legacy code.
    """
    if ts <= 0:
        raise ValueError("Sampling time must be positive")
    if mode == "tracking":
        if not np.isfinite(ki) or abs(ki) < 1e-12:
            raise ValueError("Back-calculation requires nonzero finite integral gain")
        if not np.isfinite(tracking_time) or tracking_time < ts:
            raise ValueError("Tracking time must be finite and at least one sample")
        correction=(raw-applied)/(ki*tracking_time)
    elif mode == "sign_only":
        correction=.2*(raw-applied)
    elif mode == "legacy":
        correction=.2*(applied-raw)
    else:
        raise ValueError(f"Unknown anti-windup mode: {mode}")
    return (1-.002)*eta+ts*(error+correction)


def evaluate(client, ac, bc, tau, xi, bank, role, controller, ts, substeps=5, mode="tracking", tracking_time=.2):
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
            eta=integral_update(eta, y-rr, raw, u, K[3], ts, mode, tracking_time)
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
