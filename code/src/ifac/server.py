import numpy as np 
from .clustering import mahalanobis_kmeans
from .rls import RLSAB, RLSAB_BlockCov
from .utils import (
    BikeParams, 
    build_bicycle_beta_r, discretize, 
    design_lqi, cluster_prototype, 
    pack_theta_AB, unpack_theta_AB, 
    perturb_params,
    LQIConfig, Cluster, refresh_gains, reset_gains, nominal_params,
)
from .client import Client
from typing import List
from sklearn.cluster import KMeans

def _spd_condition(W, eps=1e-9, max_cond=1e6):
    # symmetrize
    W = 0.5*(W + W.T)
    # eigen clip and cond-cap
    s, U = np.linalg.eigh(W)
    s = np.clip(s, eps, None)
    cond = float(s.max()/s.min())
    if cond > max_cond:
        smin = s.max()/max_cond
        s = np.clip(s, smin, None)
    return (U * s) @ U.T

def precision_from_P_hat(P_hat_6x6: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    # ensure symmetric, PD, and invert
    P = 0.5 * (P_hat_6x6 + P_hat_6x6.T)
    return np.linalg.inv(P + eps * np.eye(6))

def kkt_residual_relative(cluster_idx, assignments, thetas, precisions, mus):
    members = [i for i,a in enumerate(assignments) if a == cluster_idx]
    if not members:
        return 0.0, 0
    S = sum(precisions[i] for i in members)
    b = sum(precisions[i] @ thetas[i] for i in members)
    mu = mus[cluster_idx]
    r  = S @ mu - b
    denom = np.linalg.norm(S, 2)*np.linalg.norm(mu, 2) + np.linalg.norm(b, 2) + 1e-18
    return float(np.linalg.norm(r, 2) / denom), len(members)

def condition_precision(W: np.ndarray, eps: float = 1e-8, max_cond: float = 1e6) -> np.ndarray:
    # Symmetrize
    W = 0.5 * (W + W.T)
    # Eig-clip to enforce SPD and cap condition number
    s, U = np.linalg.eigh(W)
    s = np.clip(s, eps, None)
    cond = float(s.max() / s.min())
    if cond > max_cond:
        s_min_target = s.max() / max_cond
        s = np.clip(s, s_min_target, None)
    return U @ np.diag(s) @ U.T

def generate_fleet(Ts: float, avail_clusters: List[Cluster], n_per_cluster=20, variability=0.10, seed=1337,
                   lqi_cfg: LQIConfig | None = None, set_gains=True, compute_gains=False, 
                   delta=1e-2, lambda_f=0.995, lane_change_time:List[float] = [2.0]) -> list[Client]:
    rng = np.random.default_rng(seed)
    p0 = nominal_params()
    A_nom, B_nom, C_nom, D_nom = build_bicycle_beta_r(p0)
    Ad_nom, Bd_nom, Cd_nom, Dd_nom = discretize(A_nom, B_nom, C_nom, D_nom, Ts)
    theta_init = np.hstack([Ad_nom, Bd_nom])
    clients: list[Client] = []
    # T_lc_choices = np.random.choice(np.array(lane_change_time), size=n_per_cluster*len(avail_clusters), seed=seed)
    for cluster in avail_clusters:
        proto = cluster_prototype(cluster, p0)
        for i in range(n_per_cluster):
            p_i = perturb_params(proto, rng, variability)
            A,B,C,D = build_bicycle_beta_r(p_i)
            Ad,Bd,Cd,Dd = discretize(A,B,C,D, Ts)
            T_lc = lane_change_time[0]
            if i >= n_per_cluster/2:
                T_lc = lane_change_time[1]
            # T_lc = T_lc_choices[len(clients)]
            client = Client(
                client_id=f"{cluster.name.lower()}_{i:02d}",
                cluster=cluster, params=p_i,
                A=A,B=B,C=C,D=D, 
                Ad_hat=Ad_nom, Bd_hat=Bd_nom, Cd_hat=Cd_nom, Dd_hat=Dd_nom,
                Ad_true=Ad, Bd_true=Bd, Cd_true=Cd, Dd_true=Dd,
                lqi_cfg=(lqi_cfg if lqi_cfg is not None else LQIConfig()),
                rls=RLSAB_BlockCov(n=2, lambda_f=lambda_f, theta_init=theta_init, delta=delta), 
                T_lc=T_lc, W_raw=np.eye(6)
            )
            if compute_gains:
                refresh_gains(client)
            if set_gains:
                reset_gains(client, p0)
            clients.append(client)

    return clients

class Server:
    """Central server that (a) designs the nominal controller once,
       (b) after each episode recomputes per-client gains from their estimates."""
    def __init__(self, Ts: float, lqi_cfg: LQIConfig, nominal_params: BikeParams):
        self.Ts = Ts
        self.cfg = lqi_cfg
        # nominal plant
        self.nominal_params = nominal_params
        A0,B0,C0,D0 = build_bicycle_beta_r(nominal_params)
        Ad0,Bd0,Cd0,Dd0 = discretize(A0, B0, C0, D0, Ts)
        self.nominal = dict(Ad=Ad0, Bd=Bd0, C=C0, D=D0)
        # design nominal LQI once
        self.Kx0, self.Ki0, self.k_r0 = design_lqi(Ad0, Bd0,
                                                   q_beta=lqi_cfg.q_beta,
                                                   q_r=lqi_cfg.q_r,
                                                   q_int=lqi_cfg.q_int,
                                                   rho=lqi_cfg.rho)
        self.history = []

    def push_nominal_controller(self, fleet: list[Client]):
        """Initialize every client with the same nominal controller."""
        for c in fleet:
            c.Kx = self.Kx0.copy()
            c.Ki = self.Ki0.copy()
            c.k_r = float(self.k_r0)

    def update_controllers_personalized(self, fleet: list[Client]):
        """After an episode: compute per-client LQI from their current estimates."""
        for c in fleet:
            Ad_hat = getattr(c, "Ad_hat", None)
            Bd_hat = getattr(c, "Bd_hat", None)
            if Ad_hat is None or Bd_hat is None:
                # no estimate yet -> keep current gains
                continue
            Kx, Ki, k_r = design_lqi(Ad_hat, Bd_hat,
                                     q_beta=self.cfg.q_beta,
                                     q_r=self.cfg.q_r,
                                     q_int=self.cfg.q_int,
                                     rho=self.cfg.rho)
            c.Kx, c.Ki, c.k_r = Kx, Ki, float(k_r)
            
    def update_controllers_mean(self, fleet: list[Client]):
        """Compute the mean of all client estimates and design one LQI for all."""
        Ad_list, Bd_list = [], []
        for c in fleet:
            Ad_hat = getattr(c, "Ad_hat", None)
            Bd_hat = getattr(c, "Bd_hat", None)
            if Ad_hat is None or Bd_hat is None:
                continue
            Ad_list.append(Ad_hat)
            Bd_list.append(Bd_hat)
        if not Ad_list:
            return
        A_bar = np.mean(np.array(Ad_list), axis=0)
        B_bar = np.mean(np.array(Bd_list), axis=0)
        Kx, Ki, k_r = design_lqi(A_bar, B_bar,
                                 q_beta=self.cfg.q_beta,
                                 q_r=self.cfg.q_r,
                                 q_int=self.cfg.q_int,
                                 rho=self.cfg.rho)
        for c in fleet:
            c.Kx = Kx.copy()
            c.Ki = Ki.copy()
            c.k_r = float(k_r)
        self.mean_model = dict(Ad=A_bar, Bd=B_bar, C=self.nominal["C"], Kx=Kx, Ki=Ki, k_r=k_r)
        return
        
    def update_controllers_clustered_euclidean(self, fleet: list, K: int, use_client_C: bool = False,
                                 C_shared: np.ndarray | None = None, max_iters: int = 30,
                                 tol: float = 1e-6, seed: int = 1337):
        """
        Cluster clients in theta-space with Mahalanobis k-means using their RLS precisions.
        Compute GLS prototype per cluster, design one LQI per cluster, and push gains.

        Returns dict with assignments, prototypes, and per-cluster gains for logging.
        """
        # collect clients with estimates
        thetas, idxs = [], []
        for idx, client in enumerate(fleet):
            if not (hasattr(client, "Ad_hat") and hasattr(client, "Bd_hat")):
                continue    
            thetas.append(pack_theta_AB(client.Ad_hat, client.Bd_hat))
            idxs.append(idx)
        
        if not thetas:
            return {"assignments": None, "prototypes": None, "client_indices": []}
        # ----------------------------------------------------------------------

        # hard clustering
        rng = np.random.default_rng(seed)
        assignments = KMeans(n_clusters=K, max_iter=max_iters, tol=tol, random_state=seed).fit_predict(thetas)

        # compute LQI per cluster
        cluster_models = []
        cluster_gains  = []
        mus = [np.zeros_like(thetas[0]) for _ in range(K)]  # cluster prototypes
        # print(f"Assignments: ", assignments)
        for c_id in range(K):
            # overwrite the cluster prototype with the mean of the client estimates
            theta_idx = [i for i, a in enumerate(assignments) if a == c_id]
            theta_c = np.mean(np.array(thetas)[theta_idx], axis=0)
           
            mus[c_id] = theta_c
            # theta_c = mus[c_id]
            A_bar, B_bar = unpack_theta_AB(theta_c)

            # choose C for design
            if use_client_C:
                # average available C (optional; identity fallback)
                Cs = []
                for i_local, i_global in enumerate(idxs):
                    if assignments[i_local] == c_id:
                        C_i = getattr(fleet[i_global], "Cd_hat", None) or getattr(fleet[i_global], "Cd", None)
                        if C_i is not None:
                            Cs.append(C_i)
                C_use = np.mean(np.stack(Cs, axis=0), axis=0) if len(Cs) else np.eye(2)
            else:
                C_use = C_shared if C_shared is not None else np.eye(2)

            Kx, Ki, k_r = design_lqi(A_bar, B_bar,
                                    q_beta=self.cfg.q_beta,
                                    q_r=self.cfg.q_r,
                                    q_int=self.cfg.q_int,
                                    rho=self.cfg.rho)
            cluster_models.append({"Ad": A_bar, "Bd": B_bar, "C": C_use})
            cluster_gains.append({"Kx": Kx, "Ki": Ki, "k_r": float(k_r)})

        # push cluster gains to members (hard assignment)
        for i_local, i_global in enumerate(idxs):
            c_id = int(assignments[i_local])
            g = cluster_gains[c_id]
            client = fleet[i_global]
            client.Kx = g["Kx"].copy()
            client.Ki = g["Ki"].copy()
            client.k_r = g["k_r"]
            client.cluster_id_est = c_id   # <-- add this
            A_bar, B_bar = unpack_theta_AB(mus[c_id])
            client.Ad_hat = A_bar
            client.Bd_hat = B_bar
            client.theta_hist.append(mus[c_id])  # log the cluster prototype as well


        # (optional) store on server for inspection
        # stash for later (plots, MDS, ellipses)
        self.clustered = {
            "assignments": assignments,
            "prototypes": mus,
            "cluster_models": cluster_models,
            "cluster_gains": cluster_gains,
            "client_indices": idxs,
        }
        self.history.append(self.clustered)
        return 


    def update_controllers_clustered2(
        self, fleet: list, K: int, use_client_C: bool = False,
        C_shared: np.ndarray | None = None, max_iters: int = 30,
        tol: float = 1e-6, seed: int = 1337
    ):
        """
        Cluster clients in theta-space with Mahalanobis k-means using their RLS precisions.
        Compute robust GLS prototype per cluster, design one LQI per cluster, and push gains.

        Returns: dict with assignments, prototypes, per-cluster gains and precisions.
        """
        # -------- 1) Collect eligible clients (fix hasattr logic) --------
        thetas, weights, idxs = [], [], []
        for idx, client in enumerate(fleet):
            if not (hasattr(client, "Ad_hat") and hasattr(client, "Bd_hat")
                    and hasattr(client, "W_A") and hasattr(client, "W_B")):
                continue
            theta_i = pack_theta_AB(client.Ad_hat, client.Bd_hat)     # (d,)
            # block-diagonal precision: diag(W_A, W_B)
            # W_i = np.block([
            #     [client.W_A, np.zeros((client.W_A.shape[0], client.W_B.shape[1]))],
            #     [np.zeros((client.W_B.shape[0], client.W_A.shape[1])), client.W_B]
            # ])
            W_i = client.W_raw
            W_i = _spd_condition(W_i)  # ensure SPD, symmetrize & condition
            thetas.append(theta_i)
            weights.append(W_i)
            idxs.append(idx)

        if not thetas:
            out = {"assignments": None, "prototypes": None, "client_indices": []}
            self.clustered = out
            self.history.append(out)
            return out

        d = thetas[0].shape[0]
        N = len(thetas)

        # -------- 2) Light global rescale of weights (keeps μ unchanged) --------
        tau = np.median([np.trace(W) for W in weights])
        if tau > 0:
            weights = [W / tau for W in weights]

        # -------- 3) Hard clustering (Mahalanobis k-means) --------
        rng = np.random.default_rng(seed)
        assignments, mus = mahalanobis_kmeans(
            thetas, weights, K, max_iters=max_iters, tol=tol, rng=rng)

        # Optional sanity ping: KKT residual of current μ vs. member sums
        for c_id in range(K):
            members = [i for i, a in enumerate(assignments) if a == c_id]
            if not members:
                continue
            S = np.zeros((d, d)); b = np.zeros(d)
            for j in members:
                S += weights[j]
                b += weights[j] @ thetas[j]
            r = S @ mus[c_id] - b
            denom = (np.linalg.norm(S, 2) * (np.linalg.norm(mus[c_id]) + 1e-12)
                    + np.linalg.norm(b) + 1e-12)
            kkt = float(np.linalg.norm(r) / denom)
            if kkt > 1e-3:
                print(f"[warn] Cluster {c_id}: KKT residual after k-means = {kkt:.3e}")

        # -------- 4) Robust GLS prototype per cluster --------
        cluster_models = []
        cluster_gains  = []
        cluster_precisions = []   # store S_reg from robust GLS for plotting/metrics

        for c_id in range(K):
            members = [i for i, a in enumerate(assignments) if a == c_id]
            if not members:
                # keep previous centroid if empty
                theta_c = mus[c_id].copy()
                S_reg_c = 1e-6 * np.eye(d)
            else:
                thetas_c   = [thetas[j]  for j in members]
                weights_c  = [weights[j] for j in members]
                # robust GLS mean with equilibration + Cholesky+jitter
                theta_c, stats_c, S_c, b_c = gls_mean_robust(
                    thetas=thetas_c, precisions=weights_c,
                    ridge=1e-6, eps=1e-12, equilibrate_iters=1, return_S_b=True
                )
                S_reg_c = S_c  # already ridged/equilibrated in gls_mean_robust

            mus[c_id] = theta_c
            A_bar, B_bar = unpack_theta_AB(theta_c)

            # --- choose C for design ---
            if use_client_C:
                Cs = []
                for i_local, i_global in enumerate(idxs):
                    if assignments[i_local] == c_id:
                        C_i = getattr(fleet[i_global], "Cd_hat", None)
                        if C_i is None:
                            C_i = getattr(fleet[i_global], "Cd", None)
                        if C_i is not None:
                            Cs.append(C_i)
                C_use = np.mean(np.stack(Cs, axis=0), axis=0) if len(Cs) else (C_shared if C_shared is not None else np.eye(A_bar.shape[0]))
            else:
                C_use = C_shared if C_shared is not None else np.eye(A_bar.shape[0])

            # --- design LQI controller for the cluster prototype ---
            Kx, Ki, k_r = design_lqi(
                A_bar, B_bar,
                q_beta=self.cfg.q_beta,
                q_r=self.cfg.q_r,
                q_int=self.cfg.q_int,
                rho=self.cfg.rho
            )

            cluster_models.append({"Ad": A_bar, "Bd": B_bar, "C": C_use})
            cluster_gains.append({"Kx": Kx, "Ki": Ki, "k_r": float(k_r)})
            cluster_precisions.append(_spd_condition(S_reg_c))

        # -------- 5) Push cluster gains to members (hard assignment) --------
        for i_local, i_global in enumerate(idxs):
            c_id = int(assignments[i_local])
            g = cluster_gains[c_id]
            client = fleet[i_global]
            client.Kx = g["Kx"].copy()
            client.Ki = g["Ki"].copy()
            client.k_r = g["k_r"]
            client.cluster_id_est = c_id
            A_bar, B_bar = unpack_theta_AB(mus[c_id])
            client.Ad_hat = A_bar
            client.Bd_hat = B_bar
            if hasattr(client, "theta_hist"):
                client.theta_hist.append(mus[c_id])  # log prototype

        # -------- 6) Stash & return --------
        out = {
            "assignments": assignments,
            "prototypes": mus,
            "cluster_models": cluster_models,
            "cluster_gains": cluster_gains,
            "client_indices": idxs,
            "cluster_precisions": cluster_precisions,  # S_reg per cluster (SPD)
        }
        self.clustered = out
        self.history.append(out)
        return out


    def update_controllers_clustered(
        self, fleet: list, K: int, use_client_C: bool = False,
        C_shared: np.ndarray | None = None, max_iters: int = 30,
        tol: float = 1e-6, seed: int = 1337
    ):
        """
        Cluster clients in theta-space with Mahalanobis k-means using their RLS precisions.
        For each cluster, use the Mahalanobis k-means centroid as prototype, design one LQI,
        and push the cluster controller to all members (hard assignment).

        Returns
        -------
        out : dict
            {
                "assignments": np.ndarray of shape (N_used,),
                "prototypes": list of length K (each θ_c of shape (d,)),
                "cluster_models": list of dicts with keys {"Ad","Bd","C"},
                "cluster_gains":  list of dicts with keys {"Kx","Ki","k_r"},
                "client_indices": list of global fleet indices used in clustering,
                "cluster_precisions": list of (d x d) SPD matrices S_c_reg,
            }
        """
        # -------- 1) Collect eligible clients --------
        thetas, weights, idxs = [], [], []
        for idx, client in enumerate(fleet):
            if not (hasattr(client, "Ad_hat") and hasattr(client, "Bd_hat")
                    and hasattr(client, "W_A") and hasattr(client, "W_B")):
                continue

            theta_i = pack_theta_AB(client.Ad_hat, client.Bd_hat)   # (d,)

            # Precision matrix, assumed SPD from RLS but numerically conditioned once
            W_i = client.W_raw
            W_i = _spd_condition(W_i)
            thetas.append(theta_i)
            weights.append(W_i)
            idxs.append(idx)

        # No eligible clients → return empty result
        if not thetas:
            out = {
                "assignments": None,
                "prototypes": None,
                "cluster_models": [],
                "cluster_gains": [],
                "client_indices": [],
                "cluster_precisions": [],
            }
            self.clustered = out
            self.history.append(out)
            return out

        d = thetas[0].shape[0]
        N = len(thetas)

        # -------- 2) Light global rescale of weights (does not change GLS centroids) --------
        # tau = np.median([np.trace(W) for W in weights])
        # if tau > 0:
        #     weights = [W / tau for W in weights]

        # -------- 3) Hard clustering (Mahalanobis k-means) --------
        rng = np.random.default_rng(seed)
        assignments, mus = mahalanobis_kmeans(
            thetas, weights, K, max_iters=max_iters, tol=tol, rng=rng, print_debug=False
        )

        # -------- 4) Per-cluster statistics S_c, b_c and optional KKT check --------
        # Build cluster membership and S_c, b_c in a single pass
        cluster_members = [[] for _ in range(K)]
        cluster_S = [np.zeros((d, d)) for _ in range(K)]
        cluster_b = [np.zeros(d) for _ in range(K)]

        for i_local, c_id in enumerate(assignments):
            cluster_members[c_id].append(i_local)
            cluster_S[c_id] += weights[i_local]
            cluster_b[c_id] += weights[i_local] @ thetas[i_local]

        # Optional KKT sanity check: S_c μ_c ≈ b_c
        for c_id in range(K):
            members = cluster_members[c_id]
            if not members:
                continue
            S = cluster_S[c_id]
            b = cluster_b[c_id]
            r = S @ mus[c_id] - b
            denom = (
                np.linalg.norm(S, 2) * (np.linalg.norm(mus[c_id]) + 1e-12)
                + np.linalg.norm(b) + 1e-12
            )
            kkt = float(np.linalg.norm(r) / denom)
            if kkt > 1e-3:
                print(f"[warn] Cluster {c_id}: KKT residual after k-means = {kkt:.3e}")

        # -------- 5) Cluster prototypes, C selection, LQI design --------
        cluster_models = []
        cluster_gains = []
        cluster_precisions = []

        for c_id in range(K):
            members = cluster_members[c_id]

            if not members:
                # No members: keep centroid as-is, use small isotropic precision
                theta_c = mus[c_id].copy()
                S_reg_c = 1e-6 * np.eye(d)
            else:
                theta_c = mus[c_id]  # centroid from k-means (already GLS barycenter)
                S_c = 0.5 * (cluster_S[c_id] + cluster_S[c_id].T)
                mass = max(1.0, float(np.trace(S_c) / d))
                S_reg_c = S_c + (1e-6 * mass) * np.eye(d)

            # Store updated prototype in mus
            mus[c_id] = theta_c
            A_bar, B_bar = unpack_theta_AB(theta_c)

            # --- choose C for design ---
            if use_client_C and members:
                Cs = []
                for i_local in members:
                    i_global = idxs[i_local]
                    client = fleet[i_global]
                    C_i = getattr(client, "Cd_hat", None)
                    if C_i is None:
                        C_i = getattr(client, "Cd", None)
                    if C_i is not None:
                        Cs.append(C_i)

                if Cs:
                    C_use = np.mean(np.stack(Cs, axis=0), axis=0)
                else:
                    C_use = C_shared if C_shared is not None else np.eye(A_bar.shape[0])
            else:
                C_use = C_shared if C_shared is not None else np.eye(A_bar.shape[0])

            # --- design LQI controller for the cluster prototype ---
            Kx, Ki, k_r = design_lqi(
                A_bar, B_bar,
                q_beta=self.cfg.q_beta,
                q_r=self.cfg.q_r,
                q_int=self.cfg.q_int,
                rho=self.cfg.rho,
            )

            cluster_models.append({"Ad": A_bar, "Bd": B_bar, "C": C_use})
            cluster_gains.append({"Kx": Kx, "Ki": Ki, "k_r": float(k_r)})
            # Condition the cluster precision once for plotting/metrics if desired
            cluster_precisions.append(_spd_condition(S_reg_c))

        # -------- 6) Push cluster gains and prototypes to members --------
        for i_local, i_global in enumerate(idxs):
            c_id = int(assignments[i_local])
            g = cluster_gains[c_id]
            client = fleet[i_global]

            # Assign cluster controller
            client.Kx = g["Kx"].copy()
            client.Ki = g["Ki"].copy()
            client.k_r = g["k_r"]
            client.cluster_id_est = c_id

            # Overwrite identified model with cluster prototype
            # A_bar, B_bar = unpack_theta_AB(mus[c_id])
            # client.Ad_hat = A_bar
            # client.Bd_hat = B_bar

            # Optional logging of prototype trajectory
            if hasattr(client, "theta_hist"):
                client.theta_hist.append(mus[c_id])

        # -------- 7) Stash & return --------
        out = {
            "assignments": assignments,
            "prototypes": mus,
            "cluster_models": cluster_models,
            "cluster_gains": cluster_gains,
            "client_indices": idxs,
            "cluster_precisions": cluster_precisions,
        }
        self.clustered = out
        self.history.append(out)
        return out


import numpy as np
from typing import Iterable, Tuple, Dict, Optional

def gls_mean(
    thetas: Iterable[np.ndarray] | None = None,
    precisions: Iterable[np.ndarray] | None = None,
    *,
    clients: Iterable[object] | None = None,   # objects with .theta and .W
    ridge: float = 1e-6,                        # dimensionless ridge (scaled by trace(S)/d)
    eps: float = 1e-12,                         # tiny SPD floor
    return_S_b: bool = False
) -> Tuple[np.ndarray, Dict[str, float]] | Tuple[np.ndarray, Dict[str, float], np.ndarray, np.ndarray]:
    """
    GLS barycenter mu = argmin_mu sum_i (mu - theta_i)^T W_i (mu - theta_i).
    Inputs (choose ONE way):
      - (thetas, precisions): lists/arrays of shape (d,) and (d,d)
      - clients: iterable with .theta (d,) and .W (d,d)

    Returns:
      mu : (d,) GLS estimate
      stats : dict with cond(S), kkt_residual, mass, n, d
      [S, b] : (optional) accumulated precision and RHS used to compute mu
    """
    # ---- collect data ----
    if clients is not None:
        thetas = [np.asarray(pack_theta_AB(c.Ad_hat, c.Bd_hat), float).reshape(-1) for c in clients]
        precisions = [np.asarray(c.W_raw, float) for c in clients]
    else:
        if thetas is None or precisions is None:
            raise ValueError("Provide either (thetas, precisions) or clients=...")
        thetas = [np.asarray(t, float).reshape(-1) for t in thetas]
        precisions = [np.asarray(W, float) for W in precisions]

    if len(thetas) == 0:
        raise ValueError("Empty input.")

    d = thetas[0].shape[0]
    for t in thetas:
        if t.shape != (d,):
            raise ValueError("All theta vectors must have same shape (d,).")
    for W in precisions:
        if W.shape != (d, d):
            raise ValueError("All precision matrices must be (d,d).")

    # ---- accumulate S = sum W_i, b = sum W_i @ theta_i ----
    S = np.zeros((d, d), dtype=float)
    b = np.zeros(d, dtype=float)
    for t, W in zip(thetas, precisions):
        W = 0.5 * (W + W.T)                 # symmetrize
        S += W
        b += W @ t

    # ---- regularize (scaled ridge) ----
    mass = max(1.0, float(np.trace(S)) / d)  # scale ridge by "cluster mass"
    S_reg = S + (ridge * mass + eps) * np.eye(d)

    # ---- solve ----
    mu = np.linalg.solve(S_reg, b)

    # ---- diagnostics ----
    r = S @ mu - b
    denom = (np.linalg.norm(S, 2) * (np.linalg.norm(mu) + eps) + np.linalg.norm(b) + eps)
    stats = dict(
        cond=float(np.linalg.cond(S_reg)),
        kkt_residual=float(np.linalg.norm(r) / denom),
        mass=float(mass),
        n=len(thetas),
        d=d,
    )

    if return_S_b:
        return mu, stats, S_reg, b
    return mu, stats

import numpy as np
from typing import Iterable, Optional, Tuple, Dict
from scipy.linalg import cho_factor, cho_solve

def gls_mean_robust(
    thetas: Iterable[np.ndarray] = None,
    precisions: Iterable[np.ndarray] = None,
    *,
    ridge: float = 1e-6,
    eps: float = 1e-12,
    max_jitter_tries: int = 5,
    equilibrate_iters: int = 1,   # 0 = off; 1–2 usually enough
    return_S_b: bool = False
) -> Tuple[np.ndarray, Dict[str, float]] | Tuple[np.ndarray, Dict[str, float], np.ndarray, np.ndarray]:
    """GLS mean with diagonal equilibration and Cholesky + adaptive jitter."""
    thetas = [np.asarray(t, float).reshape(-1) for t in thetas]
    precisions = [0.5*(np.asarray(W,float)+np.asarray(W,float).T) for W in precisions]

    d = thetas[0].size
    S = np.zeros((d,d))
    b = np.zeros(d)
    for t, W in zip(thetas, precisions):
        S += W
        b += W @ t

    # ridge scaled by "mass"
    mass = max(1.0, float(np.trace(S))/d)
    S = 0.5*(S+S.T)
    S_reg = S + (ridge*mass + eps)*np.eye(d)

    # --- diagonal equilibration (Ruiz-lite) ---
    D = np.ones(d)
    for _ in range(max(0, equilibrate_iters)):
        diagS = np.maximum(np.diag(S_reg), eps)
        scale = 1.0/np.sqrt(diagS)
        D *= scale
        S_reg = (scale[:,None]*S_reg)*scale[None,:]
        b = scale * b

    # --- Cholesky with adaptive jitter ---
    jitter = 0.0
    for _ in range(max_jitter_tries):
        try:
            c, lower = cho_factor(S_reg + jitter*np.eye(d), overwrite_a=False, check_finite=False)
            y = cho_solve((c, lower), b, check_finite=False)
            break
        except np.linalg.LinAlgError:
            jitter = max(1e-12, (10.0 if jitter==0.0 else 10.0*jitter))
    else:
        # last resort
        y = np.linalg.lstsq(S_reg + (jitter+1e-9)*np.eye(d), b, rcond=None)[0]

    # unscale
    mu = D * y

    # diagnostics
    r = S @ mu - b / D  # note: b was scaled; divide back elementwise by D
    denom = (np.linalg.norm(S,2)*(np.linalg.norm(mu)+eps) + np.linalg.norm(b/D)+eps)
    stats = dict(
        cond=float(np.linalg.cond(S_reg)),
        mass=float(mass),
        kkt_residual=float(np.linalg.norm(r)/denom),
        jitter=float(jitter),
        equil_iters=int(equilibrate_iters),
        d=int(d),
        n=len(thetas)
    )
    if return_S_b:
        return mu, stats, S, b/D
    return mu, stats

