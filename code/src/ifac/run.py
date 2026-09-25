import numpy as np 
from .utils import (
    BikeParams, 
    build_bicycle_beta_r, discretize, 
    design_lqi, closed_loop_step_lqi, 
    pack_theta_AB
)
from .client import Client
from .server import Server
from .rls import make_precision_from_rls
from scipy.linalg import block_diag

def run_episode_update_rls(client, r_ref, init_x=None, use_preview=False, noise_std=0.0, 
                           process_noise=[0.0, 0.0], U_min=-0.5, U_max=0.5, debug=False):
    """
    Simulate one DLC episode for 'client', update local RLS on [Ad Bd] online.
    Returns a dict with logs and final (Θ_hat, P).
    """
    Ad, Bd = client.Ad_true, client.Bd_true # true physical system
    Ad_hat, Bd_hat = client.Ad_hat, client.Bd_hat # current estimates used for RLS
    C_r = np.array([[0.0, 1.0]])
    Kx, Ki, k_r = client.Kx, client.Ki, client.k_r
    Ki_s = float(Ki.squeeze())

    N = len(r_ref)
    x = np.zeros(2) if init_x is None else np.asarray(init_x, float).copy()
    # steady-state integrator init for the *first* reference the controller uses
    r0 = float(r_ref[1] if (use_preview and N > 1) else r_ref[0])
    z = (-(Kx @ x).item() + k_r * r0) / Ki_s

    rls_new = client.rls # use existing RLS instance
    rls_new.reset_theta(pack_theta_AB(Ad_hat, Bd_hat))  # reset to current estimate
    # logs
    if debug:
        y_hist = np.zeros(N); u_hist = np.zeros(N); z_hist = np.zeros(N)
        x_hist = np.zeros((N, 2)); theta_hist = np.zeros((N, 2, 3))
        x_hist[0] = x
        z_hist[0] = z
        theta_hist[0] = rls_new.get_theta()
    for k in range(N-1):
        rr = float(r_ref[k+1] if use_preview else r_ref[k])
        x_next, z_next, u, y = closed_loop_step_lqi(x, z, rr, Ad, Bd, C_r, Kx, Ki, k_r, 
                                                    u_min=U_min, u_max=U_max, k_aw=0.2, leak=0.002, 
                                                    noise_std=noise_std, process_noise=process_noise)
        rls_new.update(u_k=u, x_k=x, x_kp1=x_next)

        # log
        if debug:
            y_hist[k] = float(y.item()); u_hist[k] = float(u)
            x_hist[k+1] = x; z_hist[k+1] = z
            theta_hist[k+1] = rls_new.get_theta()
        x, z = x_next, z_next
    
    # Done, compute now the precision matrix
    Ad_hat, Bd_hat = rls_new.get_AB()
    P_hat = rls_new.P

    # Per-row innovation std (EWMA) with a safety floor
    std_rows = np.array([rls_new.var_rows[0].std, rls_new.var_rows[1].std], float)

    # Build a *conditioned* precision for clustering/GLS weighting
    # W_A, W_B = rls_new.get_precision_blocks(row_std=std_rows, mode="schur")
    # W_cond, wstats = make_precision_from_rls(P_hat, std_rows,
    #                                         eps=1e-10, kappa_max=1e6,
    #                                         sigma_min=1e-4, eig_ceiling=None)
    # W_raw = rls_new.get_W(mode="schur", row_std=std_rows)
    W_raw = rls_new.get_W()
    W_A = W_raw[:2, :2]
    W_B = W_raw[2:, 2:]
    # Persist everything needed by the server
    client.Ad_hat  = Ad_hat
    client.Bd_hat  = Bd_hat
    client.P_hat   = P_hat
    client.std_rows = std_rows         # useful for diagnostics
    client.W_A      = W_A              # unconditioned
    client.W_B      = W_B              # unconditioned
    client.W_raw    = W_raw           # already conditioned (but un-normalized)
    # client.W_trace  = wstats["trace"]  # server will do global rescale
    # client.W_cond   = wstats["cond"]
    client.theta_hist_RLS.append(rls_new.get_theta().copy())
    client.Ad_hist_RLS.append(Ad_hat.copy())
    client.Bd_hist_RLS.append(Bd_hat.copy())
    client.round_counter += 1
    return 

def run_CL_nominal_model(nominal_parms: BikeParams, client: Client, r_ref, init_x=None, 
                         use_preview=False, noise_std=0.0, process_noise=[0.,0.], 
                         u_min=-0.5, u_max=0.5, Ts=0.05):
    p0 = nominal_parms
    A0,B0,C0,D0 = build_bicycle_beta_r(p0)
    Ad0,Bd0,Cd0,Dd0 = discretize(A0,B0,C0,D0, Ts=Ts)  # match your Ts
    cfg = client.lqi_cfg
    Kx, Ki, k_r = design_lqi(Ad0, Bd0, cfg.q_beta, cfg.q_r, cfg.q_int, cfg.rho)

    Ad, Bd = client.Ad_true, client.Bd_true
    C_r = np.array([[0.0, 1.0]])

    N = len(r_ref)
    x = np.zeros(2) if init_x is None else np.asarray(init_x, float).copy()
    rr0 = float(r_ref[1] if (use_preview and N > 1) else r_ref[0])
    z = ((-(Kx @ x).item() + k_r * rr0) / Ki).item()

    y_hist = np.zeros(N)
    u_hist = np.zeros(N)
    z_hist = np.zeros(N)
    x_hist = np.zeros((N, 2))
    x_hist[0] = x
    z_hist[0] = z

    for k in range(N-1):
        rr = float(r_ref[k+1] if use_preview else r_ref[k])
        x_next, z_next, u, y = closed_loop_step_lqi(x, z, rr, Ad, Bd, C_r, Kx, Ki, k_r, 
                                                    u_min=u_min, u_max=u_max, k_aw=0.2, leak=0.002, 
                                                    noise_std=noise_std, process_noise=process_noise)
        y_hist[k] = float(y.item()); u_hist[k] = float(u.item())
        x, z = x_next, z_next
        x_hist[k+1] = x; z_hist[k+1] = z

    return {
        "x_hist": x_hist,
        "y_hist": y_hist,
        "u_hist": u_hist,
    }

# ---------- evaluation-only episode (no RLS updates) ----------
def run_CL_client_model(client, r_ref, init_x=None, use_preview=False, noise_std=0.0, 
                        process_noise=[0., 0.], u_min=-0.5, u_max=0.5):
    """
    Simulate one episode for 'client' with its CURRENT controller (Kx,Ki,k_r).
    No model updates; returns a result dict with time series for plotting.
    """
    Ad, Bd = client.Ad_true, client.Bd_true
    C_r = np.array([[0.0, 1.0]])
    Kx, Ki, k_r = client.Kx, client.Ki, client.k_r
    Ki_s = float(Ki)

    N = len(r_ref)
    x = np.zeros(2) if init_x is None else np.asarray(init_x, float).copy()
    rr0 = float(r_ref[1] if (use_preview and N > 1) else r_ref[0])
    z = (-(Kx @ x).item() + k_r * rr0) / Ki_s

    y_hist = np.zeros(N)
    u_hist = np.zeros(N)
    z_hist = np.zeros(N)
    x_hist = np.zeros((N, 2))
    x_hist[0, :] = x
    z_hist[0] = z

    for k in range(N-1):
        rr = float(r_ref[k+1] if use_preview else r_ref[k])
        x_next, z_next, u, y = closed_loop_step_lqi(x, z, rr, Ad, Bd, C_r, Kx, Ki, k_r, 
                                                    u_min=u_min, u_max=u_max, k_aw=0.2, leak=0.002, 
                                                    noise_std=noise_std, process_noise=process_noise)
        y_hist[k] = float(y.item()); u_hist[k] = float(u.item())
        x, z = x_next, z_next
        x_hist[k+1] = x; z_hist[k+1] = z

    return {
        "x_hist": x_hist,
        "y_hist": y_hist,
        "u_hist": u_hist,
    }


def run_fleet_training_round_single(fleet, server: Server, r_ref, use_preview=False):
    # 1) each client runs one episode with CURRENT controller and updates local ID
    for client in fleet:
        run_episode_update_rls(client, r_ref, init_x=np.zeros(2), use_preview=use_preview)
    # 2) server recomputes gains from each client's estimates and pushes down
    server.update_controllers_personalized(fleet)
    return 

def run_fleet_training_round_FedAvg(fleet, server: Server, r_ref, use_preview=False, noise_std:float=0.0, process_noise=[0.,0.]):
    # 1) each client runs one episode with CURRENT controller and updates local ID
    for client in fleet:
        run_episode_update_rls(client, r_ref, init_x=np.zeros(2), use_preview=use_preview, noise_std=noise_std, process_noise=process_noise)
    # 2) server recomputes gains from each client's estimates and pushes down
    server.update_controllers_fedavg(fleet, C_shared=None)
    return 

def run_fleet_training_round_mean(fleet, server: Server, r_ref, use_preview=False, noise_std:float=0.0, process_noise=[0.,0.]):
    # 1) each client runs one episode with CURRENT controller and updates local ID
    for client in fleet:
        if r_ref is None:
            r_ref = client.r_ref
        run_episode_update_rls(client, r_ref, init_x=np.zeros(2), use_preview=use_preview, noise_std=noise_std, process_noise=process_noise)
    # 2) server recomputes gains from each client's estimates and pushes down
    server.update_controllers_mean(fleet)
    return 

def run_fleet_training_round_Clustered_Euclidean(fleet, server: Server, K: int, r_ref=None, lam: float=0.995, 
                                       use_preview: bool=False, noise_std:float=0.0, 
                                       process_noise=[0.,0.], C_r=None):
    # 1) each client runs one episode with CURRENT controller and updates local ID
    for client in fleet:
        if r_ref is None:
            r_ref = client.r_ref
        run_episode_update_rls(client, r_ref, init_x=np.zeros(2), use_preview=use_preview, noise_std=noise_std, process_noise=process_noise)
    # 2) server recomputes gains from each client's estimates and pushes down
    server.update_controllers_clustered_euclidean(fleet, K=K, use_client_C=False, C_shared=C_r, 
                                                  max_iters=10, tol=1e-6, seed=2025, )
    return

def run_fleet_training_round_Clustered(fleet, server: Server, K: int, r_ref=None, lam: float=0.995, 
                                       use_preview: bool=False, noise_std:float=0.0, debug=False, 
                                       process_noise=[0.,0.], C_r=None):
    # 1) each client runs one episode with CURRENT controller and updates local ID
    for client in fleet:
        if r_ref is None:
            r_ref = client.r_ref
        run_episode_update_rls(client, r_ref, init_x=np.zeros(2), use_preview=use_preview, noise_std=noise_std, process_noise=process_noise)
    if debug:
        print("A error: ", np.linalg.norm(fleet[0].Ad_hat - fleet[0].Ad_true))
        print(f"B error: ", np.linalg.norm(fleet[0].Bd_hat - fleet[0].Bd_true))
    # 2) server recomputes gains from each client's estimates and pushes down
    server.update_controllers_clustered(fleet, K=K, use_client_C=False, C_shared=C_r, 
                                        max_iters=10, tol=1e-6, seed=2025, )
    if debug:
        print("A error: ", np.linalg.norm(fleet[0].Ad_hat - fleet[0].Ad_true))
        print(f"B error: ", np.linalg.norm(fleet[0].Bd_hat - fleet[0].Bd_true))
    return

def run_fleet_training_round_clustered_soft(fleet, server: Server, r_ref, K: int, lam: float=0.995,
                                            use_preview: bool = False, P0: float = 1e-2, max_iters: int = 50,
                                            tol: float = 1e-6, seed: int = 2025, controller_mode: str = "mixture", init: str = "kmeans"):
    """
    One round:
      (1) each client runs episode, updates RLS → (Ad_hat, Bd_hat, P_hat)
      (2) server runs EM soft clustering, designs per-cluster LQI
      (3) server pushes either mixture gains or winner gains to clients
    """
    # (1) local ID
    for client in fleet:
        run_episode_update_rls(client, r_ref, init_x=np.zeros(2), use_preview=use_preview)

    # (2) soft clustered update and push controllers
    server.update_controllers_clustered_soft(fleet, K=K, use_client_C=False, C_shared=None, max_iters=max_iters,
                                             tol=tol, seed=seed, init=init, controller_mode=controller_mode)
    return

