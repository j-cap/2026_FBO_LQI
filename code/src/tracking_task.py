"""Nonlinear tracking diagnostic, isolated from the legacy yaw-rate evaluators.

Controller and plant conventions follow the archived CFL-FF code. Selected
original scalar implementations are retained in tracking_reference for parity.
"""
from copy import deepcopy
import hashlib
import json

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import solve
from scipy.signal import cont2discrete
from scipy.stats import qmc

from .tracking_reference.cfl_trajectories import (
    TrajectoryGenConfig, TrajectoryGenerator, _lane_change_kappa_single,
    _lane_change_kappa_double, _slew_rate_limit)


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def fleet(cfg):
    """Three CFL anchors and two separate interventions per anchor (no scatter)."""
    anchors = [
        ('compact', 1220., 1500., 1.05, 1.45, (9., 1.27, .93), (10., 1.27, .93)),
        ('sport', 1520., 1800., 1.15, 1.35, (13.5, 1.38, .86), (14., 1.38, .86)),
        ('suv', 2550., 5200., 1.30, 1.55, (7.5, 1.22, .95), (8., 1.22, .95)),
    ]
    result = []
    for family, m, iz, lf, lr, front, rear in anchors:
        p = dict(m=m, I_z=iz, l_f=lf, l_r=lr, mu=.9, g=9.81,
                 delta_r_offset=0., eps_vx=1., alpha_max=np.deg2rad(25.),
                 ax_clip_min=-.9 * 9.81, ax_clip_max=6.,
                 pacejka_f_lat=dict(zip(('B', 'C', 'E'), front)),
                 pacejka_r_lat=dict(zip(('B', 'C', 'E'), rear)))
        for variant in ('nominal', 'payload', 'front_tire'):
            pp = deepcopy(p)
            if variant == 'payload':
                mp = cfg['fleet']['payload_fraction'] * m
                d = cfg['fleet']['payload_rearward_fraction_wheelbase'] * (lf + lr)
                shift = mp * d / (m + mp)
                pp.update(m=m + mp, l_f=lf + shift, l_r=lr - shift,
                          I_z=iz + m * mp / (m + mp) * d * d)
            elif variant == 'front_tire':
                pp['pacejka_f_lat']['B'] *= cfg['fleet']['front_B_multiplier']
            result.append(dict(id=family + '_' + variant, family=family,
                               variant=variant, params=pp))
    return result


def candidates(cfg, smoke=False):
    """One common log-Sobol pool plus an explicit common-scaling baseline."""
    cc = cfg['gains']
    power = cfg['smoke']['sobol_power'] if smoke else cc['sobol_power']
    lo, hi = cc['log10_multiplier_bounds']
    samples = qmc.Sobol(3, scramble=True, seed=cc['sobol_seed']).random_base2(power)
    multipliers = [[s] * 3 for s in cc['scaling_anchors']]
    kinds = ['scaling'] * len(multipliers)
    multipliers += (10 ** (lo + (hi - lo) * samples)).tolist()
    kinds += ['sobol'] * len(samples)
    result = []
    for mult, kind in zip(multipliers, kinds):
        if any(np.allclose(mult, r['multipliers'], rtol=0, atol=1e-12) for r in result):
            continue
        result.append(dict(id=f'g{len(result):03d}', multipliers=mult,
                           gains=(np.asarray(cc['nominal']) * mult).tolist(), kind=kind))
    return result


def integrate_reference(t, v, kappa):
    """Continuous cubic interpolation of sampled v,kappa; high-order pose quadrature.

    Integrate heading analytically from the cubic yaw-rate interpolant. Integrate
    translation on a 10x finer grid by cubic antiderivatives. This removes the
    old Euler pose error without a plant-dependent reference construction.
    """
    fine_t = np.linspace(t[0], t[-1], 10 * (len(t) - 1) + 1)
    vs = CubicSpline(t, v)
    yaw_integral = CubicSpline(t, v * kappa).antiderivative()
    psi_fine = yaw_integral(fine_t) - yaw_integral(t[0])
    xi = CubicSpline(fine_t, vs(fine_t) * np.cos(psi_fine)).antiderivative()
    yi = CubicSpline(fine_t, vs(fine_t) * np.sin(psi_fine)).antiderivative()
    return np.column_stack((xi(t) - xi(t[0]), yi(t) - yi(t[0]),
                            yaw_integral(t) - yaw_integral(t[0])))


def _reference(ref_id, t, v, kappa, **meta):
    pose = integrate_reference(t, v, kappa)
    return dict(id=ref_id, t=t, v=v, kappa=kappa, r=v*kappa,
                pose=pose, ay_ref=v*v*kappa, **meta)


def references(cfg, realism=False):
    t = np.arange(round(cfg['duration_s'] / cfg['dt']) + 1) * cfg['dt']
    result = []
    if realism:
        rc = cfg['realism']
        for v in rc['speeds_m_s']:
            # Guaranteed single/double/curve coverage, using the original CFL
            # shape helpers and caps; random generator draws remain a separate bank.
            sc = rc['structured']
            start = round(sc['start_s']/cfg['dt'])
            for kind in ('single', 'double', 'curve'):
                kap = np.zeros_like(t)
                if kind == 'single':
                    duration = sc['single_duration_s']
                    nn = round(duration/cfg['dt'])+1
                    shape = _lane_change_kappa_single(nn, cfg['dt'], v,
                                                      rc['generator_config']['lane_change_d_m'], duration)
                    cap = rc['generator_config']['ay_evt_max_highway']
                elif kind == 'double':
                    half = sc['double_half_duration_s']
                    nn = round(2*half/cfg['dt'])+1
                    shape = _lane_change_kappa_double(nn, cfg['dt'], v,
                                                      rc['generator_config']['lane_change_d_m'], half)
                    cap = rc['generator_config']['ay_evt_max_emergency']
                else:
                    duration = sc['curve_duration_s']
                    nn = round(duration/cfg['dt'])+1
                    tt = np.arange(nn)*cfg['dt']
                    shape = sc['curve_kappa_m_inv']*np.sin(np.pi*tt/duration)**2
                    cap = rc['generator_config']['ay_evt_max_highway']
                shape = np.clip(shape, -min(cap/v**2, rc['generator_config']['kappa_evt_max']),
                                min(cap/v**2, rc['generator_config']['kappa_evt_max']))
                kap[start:start+nn] = shape[:len(kap)-start]
                kap = _slew_rate_limit(kap, rc['generator_config']['kappa_dot_max'], cfg['dt'])
                result.append(_reference(f'cfl_{kind}_s{v:g}', t, np.full_like(t, v), kap,
                                         bank='realism', speed=v, frequency=None,
                                         trajectory_seed=None))
            gc = deepcopy(rc['generator_config'])
            gc['vx_choices'] = [v]
            for seed in rc['trajectory_seeds']:
                gen = TrajectoryGenerator(seed=seed, config=TrajectoryGenConfig(**gc))
                raw = gen.sample(len(t), cfg['dt'], dtype=np.float64)
                result.append(_reference(f'cfl_s{v:g}_seed{seed}', t, raw['v_ref'],
                                         raw['kappa_ref'], bank='realism', speed=v,
                                         trajectory_seed=seed, frequency=None))
    else:
        rc = cfg['references']
        start, end, ramp = rc['start_s'], rc['end_s'], rc['ramp_s']
        # C2 smoothstep ramps with a flat central section and straight recovery.
        def smooth(z):
            z = np.clip(z, 0, 1)
            return z**3 * (10 + z * (-15 + 6*z))
        window = smooth((t-start)/ramp) * smooth((end-t)/ramp)
        for v in rc['speeds_m_s']:
            for f in rc['frequencies_hz']:
                shape = window * np.sin(2*np.pi*f*(t-start))
                shape /= np.max(np.abs(shape))
                kap = rc['peak_ay_m_s2'] / v**2 * shape
                result.append(_reference(f's{v:g}_f{f:g}', t, np.full_like(t, v), kap,
                                         bank='controlled', speed=v, frequency=f,
                                         trajectory_seed=None))
    return result


def reference_metadata(ref, dt):
    power = np.abs(np.fft.rfft(ref['kappa']))**2
    freq = np.fft.rfftfreq(len(ref['t']), dt)
    return dict(id=ref['id'], bank=ref['bank'], speed=ref['speed'],
                frequency=ref['frequency'], trajectory_seed=ref['trajectory_seed'],
                peak_ay_ref=float(np.max(np.abs(ref['ay_ref']))),
                rms_ay_ref=float(np.sqrt(np.mean(ref['ay_ref']**2))),
                rms_kappa_dot=float(np.sqrt(np.mean(np.diff(ref['kappa'])**2))/dt),
                spectral_peak_hz=float(freq[np.argmax(power)]),
                spectral_centroid_hz=float(freq @ power / max(power.sum(), 1e-30)))


def rhs(x, u, p, auxiliary=False):
    """Vectorized exact CFL-FF dbm_ct force law, x=(batch,6), u=(batch,2)."""
    x, u = np.asarray(x), np.asarray(u)
    vx, vy, r = x[:, 3], x[:, 4], x[:, 5]
    delta = u[:, 0]
    ax = np.clip(u[:, 1], p['ax_clip_min'], p['ax_clip_max'])
    m, iz, lf, lr, mu = (p[k] for k in ('m', 'I_z', 'l_f', 'l_r', 'mu'))
    fzf = m*p['g']*lr/(lf+lr)
    fzr = m*p['g']*lf/(lf+lr)
    vxs = np.maximum(np.abs(vx), p['eps_vx']) * np.where(vx >= 0, 1., -1.)
    af = np.clip(np.arctan2(vy+lf*r, vxs)-delta, -p['alpha_max'], p['alpha_max'])
    ar = np.clip(np.arctan2(vy-lr*r, vxs)-p['delta_r_offset'],
                 -p['alpha_max'], p['alpha_max'])

    def force(alpha, fz, lat):
        b, c, e = (lat[k] for k in ('B', 'C', 'E'))
        fy = -mu*fz*np.sin(c*np.arctan(b*alpha*(1-e)+e*np.arctan(b*alpha)))
        fx = m*ax*fz/(fzf+fzr)
        scale = np.maximum(1., np.hypot(fx, fy)/(mu*fz))
        return fx/scale, fy/scale

    fxf, fyf = force(af, fzf, p['pacejka_f_lat'])
    fxr, fyr = force(ar, fzr, p['pacejka_r_lat'])
    c, s = np.cos(delta), np.sin(delta)
    cr, sr = np.cos(p['delta_r_offset']), np.sin(p['delta_r_offset'])
    fxbody = fxf*c-fyf*s+fxr*cr-fyr*sr
    fyfront, fyrear = fxf*s+fyf*c, fxr*sr+fyr*cr
    fybody = fyfront + fyrear
    cp, sp = np.cos(x[:, 2]), np.sin(x[:, 2])
    derivative = np.column_stack((vx*cp-vy*sp, vx*sp+vy*cp, r,
                                  fxbody/m+r*vy, fybody/m-r*vx,
                                  (lf*fyfront-lr*fyrear)/iz))
    if auxiliary:
        return derivative, dict(ay=fybody/m, beta=np.arctan2(vy, np.maximum(abs(vx), 1e-6)),
                                util=np.maximum(np.hypot(fxf, fyf)/(mu*fzf),
                                                np.hypot(fxr, fyr)/(mu*fzr)))
    return derivative


def errors(x, pose):
    dx, dy = pose[0]-x[:, 0], pose[1]-x[:, 1]
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return c*dx+s*dy, -s*dx+c*dy, wrap_angle(pose[2]-x[:, 2])


def control(x, ref, k, gains, previous, p, cfg):
    ex, ey, ep = errors(x, ref['pose'][k])
    raw = gains[:, 0]*ey+gains[:, 1]*ep+gains[:, 2]*(ref['r'][k]-x[:, 5])
    angle = np.deg2rad(cfg['limits']['steer_deg'])
    rate = np.deg2rad(cfg['limits']['steer_rate_deg_s']) * cfg['dt']
    fb = np.clip(raw, -angle, angle)
    ff = np.arctan((p['l_f']+p['l_r'])*ref['kappa'][k])
    total = fb+ff
    clipped = np.clip(total, -angle, angle)
    applied = previous + np.clip(clipped-previous, -rate, rate)
    kx, kv = cfg['longitudinal_gains']
    axraw = kx*ex+kv*(ref['v'][k]-x[:, 3])
    ax = np.clip(axraw, -cfg['limits']['decel_m_s2'], cfg['limits']['accel_m_s2'])
    flags = np.column_stack((abs(raw)>angle, abs(total)>angle,
                             abs(clipped-previous)>rate, ax != axraw))
    return np.column_stack((applied, ax)), flags


def measurement_noise(cfg, ref_id, seed, n):
    # Stable across Python processes/platforms. Candidate, vehicle and resolution
    # are deliberately absent: paired common random numbers within each context.
    key = int.from_bytes(hashlib.sha256(ref_id.encode()).digest()[:4], 'little')
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), key]))
    return rng.normal(size=(n, 6)) * cfg['noise_sigma_state']


def rollout(p, ref, pool, cfg, seed, substeps, trace_ids=()):
    count, n = len(pool), len(ref['t'])-1
    gains = np.asarray([c['gains'] for c in pool])
    x = np.tile([*ref['pose'][0], ref['v'][0], 0., ref['r'][0]], (count, 1))
    previous = np.zeros(count)
    noise = measurement_noise(cfg, ref['id'], seed, n)
    stopped = np.zeros(count, bool)
    reasons = [''] * count
    sumsq = np.zeros((count, 6))  # ey, ex, heading, speed, rate, steering
    peaks = np.zeros((count, 5))  # ey, beta, speed, ay, utilization
    clips = np.zeros((count, 4))
    trace_indices = [i for i, c in enumerate(pool) if c['id'] in trace_ids]
    trace = {pool[i]['id']: dict(state=[x[i].tolist()], control=[], ey=[], ay=[])
             for i in trace_indices}
    dt = cfg['dt']

    def monitor(pose, speed, u):
        nonlocal stopped
        _, ey, _ = errors(x, pose)
        deriv, aux = rhs(x, u, p, auxiliary=True)
        beta = np.rad2deg(abs(aux['beta']))
        numeric = (~np.isfinite(x).all(axis=1) | ~np.isfinite(deriv).all(axis=1)
                   | (beta > 40.) | (x[:, 3] < -2.) | (abs(x[:, 5]) > 5.))
        for i in np.flatnonzero(numeric & ~stopped):
            reasons[i] = 'numeric_or_state_guard'
        stopped |= numeric
        vals = np.column_stack((abs(ey), beta, abs(x[:, 3]-speed),
                                abs(aux['ay']), aux['util']))
        peaks[:] = np.maximum(peaks, np.nan_to_num(vals, nan=1e30, posinf=1e30))
        return ey, aux['ay']

    for k in range(n):
        ex, ey, ep = errors(x, ref['pose'][k])
        ev = ref['v'][k]-x[:, 3]
        u, flags = control(x+noise[k], ref, k, gains, previous, p, cfg)
        rate = (u[:, 0]-previous)/dt
        sumsq += np.column_stack((ey**2, ex**2, ep**2, ev**2, rate**2, u[:, 0]**2))
        clips += flags
        _, ay = monitor(ref['pose'][k], ref['v'][k], u)
        for i in trace_indices:
            tr = trace[pool[i]['id']]
            tr['ey'].append(float(ey[i])); tr['ay'].append(float(ay[i]))
            tr['control'].append(u[i].tolist())
        h = dt/substeps
        for j in range(substeps):
            # Diverged states are quarantined. Their partial scores are never used.
            x[stopped] = [0., 0., 0., ref['v'][k], 0., 0.]
            u[stopped] = 0.
            k1 = rhs(x, u, p)
            k2 = rhs(x+h*k1/2, u, p)
            k3 = rhs(x+h*k2/2, u, p)
            k4 = rhs(x+h*k3, u, p)
            x += h*(k1+2*k2+2*k3+k4)/6
            theta = (j+1)/substeps
            pose = (1-theta)*ref['pose'][k]+theta*ref['pose'][k+1]
            speed = (1-theta)*ref['v'][k]+theta*ref['v'][k+1]
            monitor(pose, speed, u)
        previous = u[:, 0].copy()
        for i in trace_indices:
            trace[pool[i]['id']]['state'].append(x[i].tolist())
    ex, ey, ep = errors(x, ref['pose'][-1])
    sumsq[:, :4] += np.column_stack((ey**2, ex**2, ep**2, (ref['v'][-1]-x[:, 3])**2))
    rms = np.sqrt(sumsq/np.array([n+1]*4+[n]*2))
    objective = (rms[:, 0]/cfg['score']['lateral_scale_m']
                 + np.rad2deg(rms[:, 4])/cfg['score']['steer_rate_scale_deg_s'])
    f = cfg['feasibility']
    task_ok = ((peaks[:, 0] <= f['peak_lateral_error_m']) &
               (peaks[:, 1] <= f['peak_beta_deg']) &
               (peaks[:, 2] <= f['peak_speed_error_m_s']))
    result = []
    for i, c in enumerate(pool):
        complete = not bool(stopped[i])
        violations = [name for name, value, bound in zip(
            ('lateral_error', 'sideslip', 'speed_error'), peaks[i, :3],
            (f['peak_lateral_error_m'], f['peak_beta_deg'], f['peak_speed_error_m_s'])) if value > bound]
        result.append(dict(candidate=c['id'], complete=complete,
                           feasible=bool(complete and task_ok[i]),
                           failure=reasons[i] or ','.join(violations), substeps=substeps,
                           J=float(objective[i]) if complete else None,
                           rmse_y=float(rms[i, 0]) if complete else None,
                           rmse_x=float(rms[i, 1]) if complete else None,
                           rmse_heading_deg=float(np.rad2deg(rms[i, 2])) if complete else None,
                           rmse_speed=float(rms[i, 3]) if complete else None,
                           rms_rate_deg_s=float(np.rad2deg(rms[i, 4])) if complete else None,
                           rms_steer_deg=float(np.rad2deg(rms[i, 5])) if complete else None,
                           peak_y=float(peaks[i, 0]), peak_beta_deg=float(peaks[i, 1]),
                           peak_speed_error=float(peaks[i, 2]), peak_ay=float(peaks[i, 3]),
                           peak_utilization=float(peaks[i, 4]),
                           fb_clip_fraction=float(clips[i, 0]/n),
                           total_clip_fraction=float(clips[i, 1]/n),
                           rate_clip_fraction=float(clips[i, 2]/n),
                           accel_clip_fraction=float(clips[i, 3]/n)))
    return result, trace


def _agreement(coarse, fine, cfg):
    if (coarse['complete'], coarse['feasible'], coarse['failure']) != (
            fine['complete'], fine['feasible'], fine['failure']):
        return False, None
    if not fine['complete']:
        return True, None  # same guard failure; never an eligible candidate
    keys = ('rmse_y', 'rms_rate_deg_s', 'peak_y', 'peak_beta_deg',
            'peak_speed_error', 'peak_ay')
    cc = cfg['integration']
    ratios = [abs(coarse[k]-fine[k]) / (cc['absolute_tolerance']+
              cc['relative_tolerance']*max(abs(fine[k]), 1e-3)) for k in keys]
    return bool(max(ratios) <= 1.), float(max(ratios))


def resolved_rollout(p, ref, pool, cfg, seed, trace_ids=()):
    """Adaptive per-candidate RK4 refinement with identical held controls/noise.

    Only unresolved candidates are rerun at the next resolution; all statistics
    and trace arrays are taken from that candidate's last computed resolution.
    """
    levels = cfg['integration']['substeps']
    coarse, _ = rollout(p, ref, pool, cfg, seed, levels[0])
    pending = list(range(len(pool)))
    output, traces = [None] * len(pool), {}
    for level in levels[1:]:
        current_pool = [pool[i] for i in pending]
        fine, tr = rollout(p, ref, current_pool, cfg, seed, level, trace_ids)
        again = []
        next_coarse = []
        for j, i in enumerate(pending):
            row = fine[j]
            ok, ratio = _agreement(coarse[j], row, cfg)
            row.update(integration_resolved=ok, integration_error_ratio=ratio)
            output[i] = row
            if row['candidate'] in tr:
                traces[row['candidate']] = tr[row['candidate']]
            if not ok:
                again.append(i); next_coarse.append(row)
        pending, coarse = again, next_coarse
        if not pending:
            break
    return output, traces


def lateral_linearization(p, speed):
    """Exact-parameter oracle tangent of the true nonlinear force law."""
    x = np.array([[0., 0., 0., speed, 0., 0.]])
    u = np.zeros((1, 2))
    h = 1e-5
    a = np.column_stack([(rhs(x+np.eye(6)[j]*h, u, p)[0, 4:6]-
                          rhs(x-np.eye(6)[j]*h, u, p)[0, 4:6])/(2*h) for j in (4, 5)])
    b = ((rhs(x, u+[[h, 0]], p)-rhs(x, u-[[h, 0]], p))/(2*h))[0, 4:6]
    return a, b


def baseline_tangent_model(p, v, cfg):
    """Sampled baseline feedback model with a ZOH curvature disturbance."""
    a, b = lateral_linearization(p, v)
    # e_y_dot = v*e_psi-v_y; e_psi_dot = v*kappa-r.
    az = np.zeros((4, 4)); az[0, 1] = v; az[0, 2] = -1; az[1, 3] = -1
    az[2:, 2:] = a
    bz = np.zeros((4, 2)); bz[1, 1] = v; bz[2:, 0] = b
    ad, bd, _, _, _ = cont2discrete((az, bz, np.eye(4), np.zeros((4, 2))), cfg['dt'])
    ky, kp, kr = cfg['gains']['nominal']
    feedback = np.array([ky, kp, 0., -kr])
    direct = p['l_f']+p['l_r']+kr*v
    acl = ad+np.outer(bd[:, 0], feedback)
    bc = bd[:, 1]+bd[:, 0]*direct
    return acl, bc, feedback, direct


def oracle_features(vehicle, ref, cfg):
    p, v = vehicle['params'], ref['speed']
    a, b = lateral_linearization(p, v)
    poles = np.linalg.eigvals(a)
    stable = bool(np.max(poles.real) < 0)
    dc = float((-np.linalg.solve(a, b))[1])
    tau = float(-1/np.max(poles.real)) if stable else None
    acl, bc, feedback, direct = baseline_tangent_model(p, v, cfg)
    closed_stable = bool(np.max(abs(np.linalg.eigvals(acl))) < 1.)
    frequencies = np.fft.rfftfreq(len(ref['t']), cfg['dt'])
    z = np.exp(2j*np.pi*frequencies*cfg['dt'])
    response = np.array([solve(zz*np.eye(4)-acl, bc) for zz in z])
    hy = response[:, 0]
    hd = response @ feedback+direct
    hd_rate = hd*(1-1/z)/cfg['dt']
    return dict(vehicle=vehicle['id'], context=ref['id'], dc_yaw=dc,
                normalized_dc_yaw=dc/(v/(p['l_f']+p['l_r'])), tau=tau,
                open_loop_stable=stable, baseline_closed_loop_stable=closed_stable,
                physical=[p['m'], p['I_z'], p['l_f'], p['l_r'],
                          *[p['pacejka_f_lat'][k] for k in ('B', 'C', 'E')],
                          *[p['pacejka_r_lat'][k] for k in ('B', 'C', 'E')]],
                hy=hy, hd_rate=hd_rate)


def descriptor_distances(va, vb, fa, fb, ref, cfg):
    physical = float(np.linalg.norm((np.asarray(fa['physical'])-fb['physical']) /
                                    cfg['transfer']['physical_feature_scales']))
    steady = abs(np.log(fa['normalized_dc_yaw']/fb['normalized_dc_yaw']))
    transient = (float(np.linalg.norm(np.array([
        np.log(fa['normalized_dc_yaw']/fb['normalized_dc_yaw']),
        np.log(fa['tau']/fb['tau'])]) / cfg['transfer']['transient_log_feature_scales']))
        if fa['open_loop_stable'] and fb['open_loop_stable'] else None)
    n = len(ref['kappa'])
    weight = abs(np.fft.rfft(ref['kappa']))**2 / n**2
    # Parseval weights for real signals: DC and (if present) Nyquist counted once.
    weight[1:] *= 2
    if n % 2 == 0:
        weight[-1] /= 2
    difference = (abs(fa['hy']-fb['hy'])**2/cfg['score']['lateral_scale_m']**2+
                  abs(fa['hd_rate']-fb['hd_rate'])**2/
                  np.deg2rad(cfg['score']['steer_rate_scale_deg_s'])**2)
    closed = (float(np.sqrt(weight @ difference)) if
              fa['baseline_closed_loop_stable'] and fb['baseline_closed_loop_stable'] else None)
    return dict(family=0. if va['family'] == vb['family'] else 1., physical=physical,
                steady=float(steady), transient=transient, closed_loop=closed)
