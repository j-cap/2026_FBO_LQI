
import numpy as np
from typing import Dict, Any, Tuple

def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def althoff_fb(x: Dict[str, float], xd: Dict[str, float], gains: Dict[str, float], limits: Dict[str, Any]) -> Tuple[float, float, float]:
    """
    Minimal Althoff-style FB controller.

    Inputs:
      x  : current (measured/estimated) state dict with keys:
           "X","Y","psi","vx","r"
      xd : desired state dict with keys:
           "X","Y","psi","vx","r"

    Output:
      (delta_cmd [rad], ax_cmd [m/s^2], delta_cmd_raw [rad])

    delta_cmd_raw is the steering gain output before saturation -- needed by the iterative
    closed-loop learning protocol (docs/iterative_cfl_experimental_protocol.md section 5) as
    delta_fb_raw. delta_cmd is delta_cmd_raw clipped to +/- max_steer_angle_rad; ax_cmd is
    similarly clipped.

    gains:
      K1,K2,K3 (lateral), K5,K6 (longitudinal)

    limits:
      max_steer_angle_rad, max_lon_accel_mps2, max_lon_decel_mps2
    """
    X, Y, psi, vx, r = x["X"], x["Y"], x["psi"], x["vx"], x["r"]

    Xd, Yd, psid, vxd, rd = xd["X"], xd["Y"], xd["psi"], xd["vx"], xd["r"]

    dx = Xd - X
    dy = Yd - Y

    c, s = float(np.cos(psid)), float(np.sin(psid))

    # errors expressed in desired frame (Althoff & Dolan convention)
    eps_x   =  c * dx + s * dy
    eps_y   = -s * dx + c * dy
    eps_psi = wrap_angle(psid - psi)
    eps_r   = rd - r

    u_delta = float(gains["K1"]) * eps_y + float(gains["K2"]) * eps_psi + float(gains["K3"]) * eps_r
    a_x     = float(gains["K5"]) * eps_x + float(gains["K6"]) * (vxd - vx)

    # saturate
    delta_max = float(limits["max_steer_angle_rad"])
    ax_min = -float(limits["max_lon_decel_mps2"])
    ax_max =  float(limits["max_lon_accel_mps2"])

    delta = float(np.clip(u_delta, -delta_max, delta_max))
    ax    = float(np.clip(a_x, ax_min, ax_max))
    return delta, ax, float(u_delta)
