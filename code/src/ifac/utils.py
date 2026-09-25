from dataclasses import dataclass
from enum import Enum, auto
import numpy as np
from scipy import signal
from scipy.linalg import solve_discrete_are
from scipy.signal import butter, filtfilt

# ---------------------------------
# (1) Types with controller fields
# ---------------------------------
@dataclass(frozen=True)
class BikeParams:
    m: float; Iz: float; a: float; b: float; Cf: float; Cr: float; U: float

class Cluster(Enum):
    PAYLOAD = auto()
    HIGH_SPEED = auto()
    LOW_SPEED = auto()
    AGING_TIRES = auto()
    HIGH_GRIP = auto()
    MALICIOUS = auto()

@dataclass
class LQIConfig:
    q_beta: float = 1.0
    q_r: float    = 2.0
    q_int: float  = 2.0
    rho: float    = 5e-1
    preview: bool = False  # if you later use one-step preview, keep the flag here

# ----------------------
# (2) Model utilities
# ----------------------
def build_bicycle_beta_r(p: BikeParams):
    m, Iz, a, b, Cf, Cr, U = p.m, p.Iz, p.a, p.b, p.Cf, p.Cr, p.U
    A = np.array([
        [-(Cf+Cr)/(m*U), (-a*Cf + b*Cr)/(m*U*U) - 1],
        [(-a*Cf + b*Cr)/(Iz), -(a*a*Cf + b*b*Cr)/(Iz*U)]
    ], float)
    B = np.array([[Cf/(m*U)], [a*Cf/Iz]], float)
    C = np.eye(2); D = np.zeros((2,1))
    return A,B,C,D

def discretize(A,B,C,D, Ts):
    # Ad,Bd,Cd,Dd,_ = signal.cont2discrete((A,B,C,D), Ts, method='zoh')
    d_sys = signal.cont2discrete((A,B,C,D), Ts, method='zoh') 
    Ad, Bd, Cd, Dd = d_sys[0], d_sys[1], d_sys[2], d_sys[3]
    return Ad,Bd,Cd,Dd


def nominal_params() -> BikeParams:
    return BikeParams(m=1500.0, Iz=2700.0, a=1.25, b=1.35, Cf=80_000.0, Cr=90_000.0, U=15.0)

def cluster_prototype(cluster: Cluster, p0: BikeParams) -> BikeParams:
    if cluster is Cluster.AGING_TIRES:
        return BikeParams(p0.m, p0.Iz, p0.a, p0.b, 0.75*p0.Cf, 1.0*p0.Cr, p0.U)
    elif cluster is Cluster.PAYLOAD:
        return BikeParams(1.20*p0.m, 1.25*p0.Iz, p0.a, p0.b, p0.Cf, p0.Cr, p0.U)
    elif cluster is Cluster.LOW_SPEED:
        return BikeParams(p0.m, p0.Iz, p0.a, p0.b, p0.Cf, p0.Cr, 10.0)
    elif cluster is Cluster.HIGH_GRIP:
        return BikeParams(p0.m, p0.Iz, p0.a, p0.b, 1.15*p0.Cf, 1.15*p0.Cr, p0.U)
    elif cluster is Cluster.HIGH_SPEED:
        return BikeParams(p0.m, p0.Iz, p0.a, p0.b, p0.Cf, p0.Cr, 20.0)
    elif cluster is Cluster.MALICIOUS:
        return BikeParams(p0.m*2, 0.5*p0.Iz, 2*p0.a, 2*p0.b, 2*p0.Cf, 2*p0.Cr, 2*p0.U)
    raise ValueError("Unknown cluster")

def perturb_uniform(val, rng, rel): return float(val * (1.0 + rng.uniform(-rel, rel)))

def perturb_params(p: BikeParams, rng, rel_spread: float) -> BikeParams:
    return BikeParams(
        m=perturb_uniform(p.m, rng, rel_spread),
        Iz=perturb_uniform(p.Iz, rng, rel_spread),
        a=perturb_uniform(p.a, rng, 0),
        b=perturb_uniform(p.b, rng, 0),
        Cf=perturb_uniform(p.Cf, rng, rel_spread),
        Cr=perturb_uniform(p.Cr, rng, rel_spread),
        U=perturb_uniform(p.U, rng, rel_spread),
    )

def closed_loop_step_lqi(x, z, r_k, Ad, Bd, Cd, Kx, Ki, k_r,
                         u_min=-0.5, u_max=0.5, k_aw=0.2, leak=0.0, noise_std=0.0, process_noise=[0.0, 0.0]):
    """
    LQI law: u = -Kx x - Ki z + k_r r_k
    Anti-windup: back-calculation; leak ∈ [0, 1e-2] optional.
    Returns x_next, z_next, u, y (= Cd x)
    """
    u_unsat = - float((Kx @ x + Ki * z).item()) + float(k_r) * float(r_k)
    u = np.clip(u_unsat, u_min, u_max)
    y = Cd @ x + noise_std * np.random.randn(Cd.shape[0])
    if Cd.shape[0] == 1:
        e = y[0] - float(r_k)
    elif Cd.shape[0] == 2:
        e = y[1] - float(r_k)
    else:
        raise ValueError("Cd has unsupported number of outputs")
    # integrator update (with leak and back-calculation AW)
    z_next = ((1.0 - leak) * z + e + k_aw * (u - u_unsat)).item()
    x_next = Ad @ x + Bd.flatten() * u + np.random.multivariate_normal(np.zeros(2), np.diag(process_noise))
    return x_next, z_next, u, y

# ----------------------
# (3) LQI + prefilter
# ----------------------
def design_lqi(Ad, Bd, q_beta=1.0, q_r=2.0, q_int=2.0, rho=5e-1):
    """
    LQR on augmented system with integral on yaw-rate error.
    Returns Kx (1x2), Ki (1x1), k_r (scalar prefilter).
    """
    C_r = np.array([[0.0, 1.0]])  # track yaw rate
    # augmented plant
    A_aug = np.block([[Ad,  np.zeros((2,1))],
                      [C_r, np.eye(1)]])
    B_aug = np.vstack([Bd, np.zeros((1,1))])
    Q_aug = np.diag([q_beta, q_r, q_int])
    R = np.array([[rho]])

    P = solve_discrete_are(A_aug, B_aug, Q_aug, R)
    K = np.linalg.inv(B_aug.T @ P @ B_aug + R) @ (B_aug.T @ P @ A_aug)
    Kx = K[:, :2]
    Ki = K[:, 2:]

    # static prefilter for unity DC from r to y_r on this plant
    Acl = Ad - Bd @ Kx
    k_r = float((1.0 / (C_r @ np.linalg.inv(np.eye(2) - Acl) @ Bd)).item())
    return Kx, Ki, k_r


def init_integrator(Kx, Ki, k_r, x0, r_ref, preview=False):
    def get_initial_ref(r_ref, preview=False):
        if not preview:
            return float(r_ref[0])
        return float(r_ref[1] if len(r_ref) > 1 else r_ref[0])
    rr0 = get_initial_ref(r_ref, preview=preview)  # the first ref used by the controller
    # Ki is 1x1 for this problem; make it scalar
    Ki_scalar = float(Ki.squeeze())
    z0 = float((-(Kx @ x0).item() + k_r * rr0) / Ki_scalar)
    return z0

def refresh_gains(client):
    """Recompute LQI gains for the client's current model."""
    cfg = client.lqi_cfg
    Kx, Ki, k_r = design_lqi(client.Ad_hat, client.Bd_hat, cfg.q_beta, cfg.q_r, cfg.q_int, cfg.rho)
    client.Kx, client.Ki, client.k_r = Kx, Ki, k_r

def reset_gains(client, nominal_parms: BikeParams):
    """ Set LQI gains based on nominal model."""
    p0 = nominal_parms
    A0,B0,C0,D0 = build_bicycle_beta_r(p0)
    Ad0,Bd0,Cd0,Dd0 = discretize(A0,B0,C0,D0, Ts=0.05)  # match your Ts
    cfg = client.lqi_cfg
    Kx, Ki, k_r = design_lqi(Ad0, Bd0, cfg.q_beta, cfg.q_r, cfg.q_int, cfg.rho)
    client.Kx, client.Ki, client.k_r = Kx, Ki, k_r

def pack_theta_AB(Ad: np.ndarray, Bd: np.ndarray) -> np.ndarray:
    """Stack [Ad Bd] (2x2 and 2x1) -> theta (6,)."""
    Theta = np.hstack([Ad, Bd])          # (2x3)
    return Theta.reshape(-1)             # (6,)

def unpack_theta_AB(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """theta (6,) -> (Ad_hat (2x2), Bd_hat (2x1))."""
    T = theta.reshape(2, 3)
    return T[:, :2].copy(), T[:, 2:3].copy()

def lc_yaw_rate_ref(Ts, T_total: float, T0: float, T1: float, r_max: float, tfilter=None):
    t = np.arange(int(T_total/Ts)) * Ts
    r = np.zeros_like(t)
    # base DLC (smooth mid join)
    idx1 = (T0 <= t) & (t < T0 + T1)
    idx2 = (T0 + T1 <= t) & (t < T0 + 2*T1)
    r[idx1] = np.sin(np.pi * (t[idx1] - T0) / T1)
    r[idx2] = -np.sin(np.pi * (t[idx2] - (T0 + T1)) / T1)
    r *= r_max
    if tfilter is not None: 
        fc = 1.0          # Hz cutoff (tune vs. your Ts and bandwidth needs)
        b = butter(N=1, Wn=fc * tfilter, btype='low', analog=False)
        r = filtfilt(b[0], b[1], r)
    return t, r