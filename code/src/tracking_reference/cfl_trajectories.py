import numpy as np
from typing import Optional, Any, Dict, Tuple
from dataclasses import dataclass

@dataclass
class TrajectoryGenConfig:
    # --- baseline curvature (highway-like) ---
    init_straight_s: float = 2.0                 # each trajectory starts straight
    base_dur_s: Tuple[float, float] = (8.0, 40.0)

    kappa_gentle_max: float = 2e-3               # gentle highway curvature
    p_straight: float = 0.75                     # baseline mixture
    p_gentle: float = 0.25

    kappa_dot_max: float = 3e-3                  # global curvature slew-rate limit (1/m/s)

    # --- event process (rare lane changes) ---
    lambda_evt: float = 0.03                     # events per second (Poisson in continuous time)

    p_evt_single: float = 0.75                   # event type mixture
    p_evt_double: float = 0.25

    lane_change_d_m: float = 3.5                 # lane width (m)

    # single lane change: "normal highway"
    lc_single_T_s: Tuple[float, float] = (3.0, 5.0)
    ay_evt_max_highway: float = 4.0              # cap for single LC (tune)
    kappa_evt_max: float = 0.07                  # hard cap from steering limit (1/m)

    # double lane change: "emergency"
    lc_double_half_T_s: Tuple[float, float] = (0.9, 1.4)   # total duration 2*T_half
    ay_evt_max_emergency: float = 7.0            # more aggressive than highway (tune)

    # --- speed profile ---
    vx_choices: Tuple[float, ...] = (20.0, 25.0, 30.0, 33.0)
    vx_seg_s: Tuple[float, float] = (20.0, 60.0)
    vx_tau_s: float = 3.0
    ax_max: float = 2.0

    # --- reference integration ---
    psi0: float = 0.0
    X0: float = 0.0
    Y0: float = 0.0

class TrajectoryGenerator:
    def __init__(self, seed: int = 0, config: Optional[TrajectoryGenConfig] = None):
        self.rng = np.random.default_rng(seed)
        self.cfg = config if config is not None else TrajectoryGenConfig()

    def sample(
        self,
        T: int,
        Ts: float,
        *,
        dtype: Any = np.float32,
        include_derivatives: bool = False,
        include_ay_ref: bool = True,
    ) -> Dict[str, np.ndarray]:
        T = int(T)
        Ts = float(Ts)
        if T <= 0 or Ts <= 0:
            raise ValueError("T and Ts must be positive")

        v_ref = self._sample_vx(T, Ts)
        kappa = self._sample_kappa(T, Ts, v_ref)

        X_ref, Y_ref, psi_ref = _integrate_reference(
            v_ref, kappa, Ts, psi0=self.cfg.psi0, X0=self.cfg.X0, Y0=self.cfg.Y0
        )

        time = np.arange(T, dtype=float) * Ts
        r_ref = v_ref * kappa

        out: Dict[str, np.ndarray] = {
            "time": np.asarray(time, dtype=dtype, order="C"),
            "v_ref": np.asarray(v_ref, dtype=dtype, order="C"),
            "kappa_ref": np.asarray(kappa, dtype=dtype, order="C"),
            "r_ref": np.asarray(r_ref, dtype=dtype, order="C"),
            "X_ref": np.asarray(X_ref, dtype=dtype, order="C"),
            "Y_ref": np.asarray(Y_ref, dtype=dtype, order="C"),
            "psi_ref": np.asarray(psi_ref, dtype=dtype, order="C"),
        }

        if include_ay_ref:
            out["ay_ref"] = np.asarray((v_ref * v_ref) * kappa, dtype=dtype, order="C")

        if include_derivatives:
            out["kappa_dot"] = np.asarray(_forward_diff(kappa, Ts), dtype=dtype, order="C")

        return out

    # --------------------------
    # speed profile (unchanged)
    # --------------------------
    def _sample_vx(self, T: int, Ts: float) -> np.ndarray:
        cfg = self.cfg
        lengths, vals, total = [], [], 0
        while total < T:
            dur = self.rng.uniform(cfg.vx_seg_s[0], cfg.vx_seg_s[1])
            n = max(1, int(np.round(dur / Ts)))
            lengths.append(n)
            vals.append(float(self.rng.choice(np.array(cfg.vx_choices, dtype=float))))
            total += n

        vx_target = np.repeat(np.array(vals, dtype=float), np.array(lengths, dtype=int))[:T]
        vx = _pt1_filter(vx_target, Ts, cfg.vx_tau_s)
        vx = _accel_limit(vx, cfg.ax_max, Ts)
        return vx

    # --------------------------
    # curvature profile (reworked)
    # --------------------------
    def _sample_kappa(self, T: int, Ts: float, vx: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        kappa = np.zeros(T, dtype=float)

        # --- helpers ---
        def sample_next_event_step(current_k: int) -> int:
            if cfg.lambda_evt <= 0:
                return T + 1
            dt_evt = self.rng.exponential(1.0 / cfg.lambda_evt)  # seconds
            n = max(1, int(np.ceil(dt_evt / Ts)))
            return current_k + n

        def nsteps(dur_lo_s: float, dur_hi_s: float) -> int:
            dur = self.rng.uniform(dur_lo_s, dur_hi_s)
            return max(1, int(np.round(dur / Ts)))

        def sample_baseline_segment(n: int) -> np.ndarray:
            # only straight + gentle, no ramp
            u = self.rng.random()
            if u < cfg.p_straight:
                return np.zeros(n)
            else:
                k0 = self.rng.uniform(-cfg.kappa_gentle_max, cfg.kappa_gentle_max)
                return k0 * np.ones(n)

        def choose_event_type() -> str:
            # normalize in case user doesn't sum to 1
            p1 = max(0.0, float(cfg.p_evt_single))
            p2 = max(0.0, float(cfg.p_evt_double))
            s = p1 + p2
            if s <= 0:
                return "single"
            u = self.rng.random() * s
            return "single" if u < p1 else "double"

        def sample_lane_change_event(n: int, k_start: int, kind: str) -> np.ndarray:
            # left/right
            sign = 1.0 if (self.rng.random() < 0.5) else -1.0
            d = cfg.lane_change_d_m * sign
            # Shape (quintic-smoothstep lateral offset -> curvature) is derived assuming a
            # locally-constant speed, estimated once from the first few samples -- reasonable
            # for the *shape*, since events are short relative to the speed profile's own time
            # constant (vx_tau_s).
            vx_local = float(np.mean(vx[k_start:k_start + max(1, min(n, 10))]))

            if kind == "single":
                T_lc = float(self.rng.uniform(*cfg.lc_single_T_s))
                # override n to match duration (caller may pass min(n, T-k))
                n_nom = max(1, int(np.round(T_lc / Ts)))
                n_use = min(n, n_nom)
                kap = _lane_change_kappa_single(n=n_use, dt=Ts, v=vx_local, d=d, T=T_lc)
                ay_cap = float(cfg.ay_evt_max_highway)
            else:  # "double"
                Th = float(self.rng.uniform(*cfg.lc_double_half_T_s))
                n_nom = max(1, int(np.round((2.0 * Th) / Ts)))
                n_use = min(n, n_nom)
                kap = _lane_change_kappa_double(n=n_use, dt=Ts, v=vx_local, d=d, T_half=Th)
                ay_cap = float(cfg.ay_evt_max_emergency)

            # Caps: curvature (fixed) and lateral acceleration (ay = v^2*kappa, so the
            # curvature bound implied by a fixed ay cap depends on speed). This MUST be
            # evaluated per-sample using each event sample's actual future speed, not a single
            # scalar estimated once at the event's start -- events can span 1-3s, during which
            # the speed profile (independently accel-limited at ax_max) can change by several
            # m/s, and holding kappa at a cap computed from the start-of-event speed lets the
            # realized ay = v_actual(t)^2 * kappa keep growing as v rises through the event,
            # silently exceeding ay_cap by the event's end (found via the trajectory-diagnostics
            # investigation: a "double" event capped at v~25 m/s reached ay~9.2 m/s^2, 31% over
            # ay_evt_max_emergency, because v had risen to ~29 m/s by the time the held-constant
            # kappa was still being applied).
            vx_event = np.asarray(vx[k_start:k_start + len(kap)], dtype=float)
            vx_event_safe = np.maximum(vx_event, 1e-3)
            kappa_ay_cap = ay_cap / (vx_event_safe * vx_event_safe)
            kappa_cap = np.minimum(float(cfg.kappa_evt_max), kappa_ay_cap)
            kap = np.clip(kap, -kappa_cap, kappa_cap)
            return kap

        # --- enforce initial straight segment ---
        n_init = max(0, int(np.round(cfg.init_straight_s / Ts)))
        n_init = min(n_init, T)
        kappa[:n_init] = 0.0

        k = n_init
        next_evt = sample_next_event_step(k)

        while k < T:
            if k >= next_evt:
                kind = choose_event_type()

                # allocate max writable length first; the sampler will pick its own nominal n internally
                n_write_max = T - k

                kap_evt = sample_lane_change_event(n_write_max, k, kind)
                n_evt = len(kap_evt)

                kappa[k:k + n_evt] = kap_evt
                k += n_evt
                next_evt = sample_next_event_step(k)
            else:
                n_base = nsteps(*cfg.base_dur_s)
                n_write = min(n_base, T - k, next_evt - k)
                kappa[k:k + n_write] = sample_baseline_segment(n_write)
                k += n_write

        # global slew-rate limit to keep transitions feasible
        kappa = _slew_rate_limit(kappa, cfg.kappa_dot_max, Ts)
        return kappa

def _integrate_reference(
    vx: np.ndarray,
    kappa: np.ndarray,
    dt: float,
    psi0: float = 0.0,
    X0: float = 0.0,
    Y0: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reference kinematics (path geometry):
        psi_dot = vx * kappa
        X_dot   = vx * cos(psi)
        Y_dot   = vx * sin(psi)
    """
    vx = np.asarray(vx, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    T = int(len(vx))

    psi = np.empty(T, dtype=float)
    X = np.empty(T, dtype=float)
    Y = np.empty(T, dtype=float)

    psi[0] = float(psi0)
    X[0] = float(X0)
    Y[0] = float(Y0)

    for k in range(1, T):
        psi_km1 = psi[k - 1]
        vx_km1 = vx[k - 1]
        kap_km1 = kappa[k - 1]
        psi[k] = psi_km1 + dt * vx_km1 * kap_km1
        X[k]   = X[k - 1] + dt * vx_km1 * np.cos(psi_km1)
        Y[k]   = Y[k - 1] + dt * vx_km1 * np.sin(psi_km1)

    return X, Y, psi

def _forward_diff(x: np.ndarray, dt: float) -> np.ndarray:
    """Cheap derivative: forward difference, last value repeated."""
    x = np.asarray(x, dtype=float)
    dx = np.empty_like(x)
    dx[:-1] = (x[1:] - x[:-1]) / float(dt)
    dx[-1] = dx[-2] if len(x) > 1 else 0.0
    return dx


def _accel_limit(v: np.ndarray, a_max: float, dt: float) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    y = np.empty_like(v)
    y[0] = v[0]
    dv_max = float(a_max) * float(dt)
    for k in range(1, len(v)):
        dv = v[k] - y[k - 1]
        dv = np.clip(dv, -dv_max, dv_max)
        y[k] = y[k - 1] + dv
    return y

def _pt1_filter(x: np.ndarray, dt: float, tau: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    tau = float(tau)
    dt = float(dt)
    if tau <= 0:
        return x.copy()
    alpha = float(np.exp(-dt / tau))
    y = np.empty_like(x)
    y[0] = x[0]
    for k in range(1, len(x)):
        y[k] = alpha * y[k - 1] + (1.0 - alpha) * x[k]
    return y

def _slew_rate_limit(x: np.ndarray, rate_max: float, dt: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.empty_like(x)
    y[0] = x[0]
    dx_max = float(rate_max) * float(dt)
    for k in range(1, len(x)):
        dx = x[k] - y[k - 1]
        dx = np.clip(dx, -dx_max, dx_max)
        y[k] = y[k - 1] + dx
    return y

def _quintic_s(s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quintic smoothstep:
      f(s)  = 10 s^3 - 15 s^4 +  6 s^5
      f'(s) = 30 s^2 - 60 s^3 + 30 s^4
      f''(s)= 60 s   -180 s^2 +120 s^3
    Returns (f, f', f'') for vector s.
    """
    f  = 10*s**3 - 15*s**4 +  6*s**5
    fp = 30*s**2 - 60*s**3 + 30*s**4
    fpp= 60*s    -180*s**2 +120*s**3
    return f, fp, fpp


def _kappa_from_y_t(y_dot: np.ndarray, y_ddot: np.ndarray, v: float) -> np.ndarray:
    """
    Convert time-derivatives of lateral profile y(t) into curvature kappa(t)
    under approx. constant longitudinal speed v:
      y_x  = y_dot / v
      y_xx = y_ddot / v^2
      kappa = y_xx / (1 + y_x^2)^(3/2)
    """
    v = max(float(v), 1e-3)
    y_x  = y_dot / v
    y_xx = y_ddot / (v*v)
    return y_xx / (1.0 + y_x*y_x)**1.5


def _lane_change_kappa_single(
    n: int, dt: float, v: float, d: float, T: float
) -> np.ndarray:
    """
    Single lane change: y: 0 -> d over duration T (smooth start/end).
    """
    n = int(n)
    if n <= 1:
        return np.zeros(max(n, 1), dtype=float)

    t = np.arange(n, dtype=float) * float(dt)
    s = np.clip(t / float(T), 0.0, 1.0)

    f, fp, fpp = _quintic_s(s)

    # y(t) = d f(s), with s=t/T:
    # y_dot  = d f'(s) * (1/T)
    # y_ddot = d f''(s) * (1/T^2)
    y_dot  = d * fp  / float(T)
    y_ddot = d * fpp / (float(T)*float(T))

    return _kappa_from_y_t(y_dot, y_ddot, v)


def _lane_change_kappa_double(
    n: int, dt: float, v: float, d: float, T_half: float
) -> np.ndarray:
    """
    Double lane change (moose-like): 0 -> d in T_half, then d -> 0 in T_half.
    Total duration is 2*T_half; caller should set n accordingly.
    """
    n = int(n)
    if n <= 1:
        return np.zeros(max(n, 1), dtype=float)

    T_half = float(T_half)
    T_tot  = 2.0 * T_half

    t = np.arange(n, dtype=float) * float(dt)

    y_dot  = np.zeros(n, dtype=float)
    y_ddot = np.zeros(n, dtype=float)

    # first half: 0 -> d
    mask1 = t <= T_half
    s1 = np.clip(t[mask1] / T_half, 0.0, 1.0)
    _, fp1, fpp1 = _quintic_s(s1)
    y_dot[mask1]  = d * fp1  / T_half
    y_ddot[mask1] = d * fpp1 / (T_half*T_half)

    # second half: d -> 0  (use mirrored time u=(t-T_half)/T_half)
    mask2 = t > T_half
    u = np.clip((t[mask2] - T_half) / T_half, 0.0, 1.0)
    _, fp2, fpp2 = _quintic_s(u)

    # y(t) = d*(1 - f(u))
    # y_dot  = -d f'(u) * (1/T_half)
    # y_ddot = -d f''(u) * (1/T_half^2)
    y_dot[mask2]  = -d * fp2  / T_half
    y_ddot[mask2] = -d * fpp2 / (T_half*T_half)

    return _kappa_from_y_t(y_dot, y_ddot, v)
