
"""
# Create an online-capable RLS identification module for A, B, C.
# - Handles SISO, n-state discrete-time systems.
# - Uses per-row scalar-output RLS filters (numerically simple & robust).
# - Supports forgetting factor (lambda_f) and ridge init (delta).
# - Provides convenience wrappers for 2-state systems.
#
# API:
#   - RLSRow: scalar-output RLS (theta, P) with .update(phi, y)
#   - RLSAB: maintains n parallel RLSRow to estimate [A B] from (x_k, u_k) -> x_{k+1}
#   - RLSC:   one RLSRow to estimate C from x_k -> y_k (SISO)
#   - helpers: pack/unpack, reset, snapshot
#
# Acknowledgment: Standard RLS with forgetting factor (Goodwin & Sin, 1984).

---------
Online (recursive) least-squares identification of discrete-time LTI system matrices (A, B, C)
for SISO systems with n states.

Models:
    x_{k+1} = A x_k + B u_k,   (A in R^{n x n}, B in R^{n x 1})
    y_k     = C x_k,           (C in R^{1 x n}, SISO; extend similarly for MIMO)

Estimators:
    - [A B]: n parallel scalar-output RLS filters, one for each state component (row).
    Each row j solves: x_{k+1}^{(j)} = theta_j^T * [x_k; u_k], theta_j in R^{n+1}.
    - C:     one scalar-output RLS filter: y_k = c^T x_k, c in R^{n}.

Features:
    - Forgetting factor lambda_f in (0,1], where lambda_f < 1 favors recent data.
    - Ridge initialization delta > 0: P0 = (1/delta) I (small delta -> large initial covariance).
    - Numerically simple per-row RLS; robust for online use.

"""
import numpy as np
from typing import Optional, Tuple, Dict
from scipy.linalg import block_diag


class RLSAB_BlockCov:
    """
    Matrix-RLS for x_{k+1} = Θ [x_k; u_k] with shared regressor covariance P_k.

    Public API mirrors your RLSAB:
      - __init__(n, lambda_f=0.995, delta=1e-2, ewma_alpha=0.01, theta_init=None)
      - update(x_k, u_k, x_kp1)
      - get_rows_std()
      - get_AB()
      - get_theta()
      - reset_theta(theta)
      - reset(theta=None, delta=None)

    Extras:
      - get_P_blocks() -> (P_xx, P_xu, P_ux, P_uu)
      - get_precision_blocks(mode='ignore'|'schur') -> (W_A, W_B)
        * 'ignore': W_A = I_n ⊗ P_xx^{-1}, W_B = I_n ⊗ P_uu^{-1}
        * 'schur' : uses Schur complements of P to remove cross-terms
    """
    def __init__(self, n: int, lambda_f: float = 0.995, delta: float = 1e-2,
                 ewma_alpha: float = 0.01, theta_init: Optional[np.ndarray] = None):
        assert 0 < lambda_f <= 1.0, "lambda_f must be in (0,1]"
        self.n = int(n)
        self.d = self.n + 1           # z = [x; u] for SISO input
        self.lambda_f = float(lambda_f)

        # Parameter matrix Θ (n x d)
        if theta_init is None:
            self.Theta = np.zeros((self.n, self.d), dtype=float)
        else:
            th = np.asarray(theta_init, dtype=float)
            self.Theta = th.reshape(self.n, self.d).copy()

        # Shared covariance in regressor space (d x d), ridge init
        self.P = (1.0 / float(delta)) * np.eye(self.d, dtype=float)

        # Per-row innovation variance trackers (for compatibility)
        self.var_rows = [OnlineVariance(alpha=ewma_alpha, init_var=1.0) for _ in range(self.n)]

        # book-keeping
        self.n_updates = 0

    # -------- core RLS update (matrix form) --------
    def update(self, x_k: np.ndarray, u_k: float, x_kp1: np.ndarray) -> None:
        x_k   = np.asarray(x_k,   dtype=float).reshape(self.n, 1)   # (n,1)
        x_kp1 = np.asarray(x_kp1, dtype=float).reshape(self.n, 1)   # (n,1)
        z = np.vstack([x_k, [[float(u_k)]]])                        # (d,1)

        # Gain in regressor space (d,1)
        denom = self.lambda_f + float((z.T @ self.P @ z).item())
        K = (self.P @ z) / denom

        # Innovation (n,1)
        E = x_kp1 - (self.Theta @ z)   # vector of row-wise innovations

        # Parameter update: Θ_k = Θ_{k-1} + E K^T
        self.Theta = self.Theta + (E @ K.T)

        # Covariance update in regressor space: P_k = (P_{k-1} - K z^T P_{k-1}) / λ
        self.P = (self.P - (K @ (z.T @ self.P))) / self.lambda_f

        # Track per-row variances (compatibility with your previous API)
        e = E.ravel()
        for j in range(self.n):
            self.var_rows[j].update(e[j])

        self.n_updates += 1

    # -------- convenience & compatibility methods --------
    def get_rows_std(self) -> np.ndarray:
        return np.array([vr.std for vr in self.var_rows], dtype=float)

    def get_theta(self) -> np.ndarray:
        return self.Theta.copy()

    def get_AB(self) -> Tuple[np.ndarray, np.ndarray]:
        A = self.Theta[:, :self.n].copy()
        B = self.Theta[:, self.n:].reshape(self.n, 1).copy()
        return A, B

    def reset_theta(self, theta: np.ndarray) -> None:
        theta = np.asarray(theta, dtype=float).reshape(self.n, self.d)
        self.Theta = theta.copy()

    def reset(self, theta: Optional[np.ndarray] = None, delta: Optional[float] = None) -> None:
        if theta is not None:
            self.reset_theta(theta)
        if delta is not None:
            self.P = (1.0 / float(delta)) * np.eye(self.d, dtype=float)
        else:
            self.P = np.eye(self.d, dtype=float)
        self.n_updates = 0
        for vr in self.var_rows:
            vr.var = 1.0

    # -------- block helpers aligned with the figure --------
    def get_P_blocks(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return P split as [[P_xx, P_xu],[P_ux, P_uu]] with xx first, then xu; ux then uu."""
        n = self.n
        P_xx = self.P[:n, :n]
        P_xu = self.P[:n, n:]
        P_ux = self.P[n:, :n]
        P_uu = self.P[n:, n:]
        return P_xx, P_xu, P_ux, P_uu
    
    # def get_P(self) -> np.ndarray:
    #     """Return full shared covariance P (d x d)."""
    #     P_theta = np.zeros((self.d*2, self.d*2), dtype=float)
    #     P_xx, P_xu, P_ux, P_uu = self.get_P_blocks()
    #     P_theta[:self.d, :self.d] = P_xx
    #     P_theta[:self.d, self.d:] = P_xu
    #     P_theta[self.d:, :self.d] = P_ux
    #     P_theta[self.d:, self.d:] = P_uu
    #     return P_theta

    # def get_precision_blocks(self, mode: str = "ignore", row_std: np.ndarray | None = None):
    #     n = self.n
    #     P_xx, P_xu, P_ux, P_uu = self.get_P_blocks()
    #     I_xx = np.eye(P_xx.shape[0]); I_uu = np.eye(P_uu.shape[0])

    #     if mode == "ignore":
    #         Pxx_inv = np.linalg.solve(P_xx, I_xx)
    #         Puu_inv = np.linalg.solve(P_uu, I_uu)  # scalar for SISO
    #     elif mode == "schur":
    #         Puu_inv = np.linalg.solve(P_uu, I_uu)
    #         Pxx_inv = np.linalg.solve(P_xx, I_xx)
    #         S_xx = P_xx - P_xu @ (Puu_inv @ P_ux)
    #         S_uu = P_uu - P_ux @ (Pxx_inv @ P_xu)   # scalar for SISO
    #         Pxx_inv = np.linalg.solve(S_xx, I_xx)
    #         Puu_inv = np.linalg.solve(S_uu, I_uu)
    #     else:
    #         raise ValueError("mode must be 'ignore' or 'schur'")

    #     W_A = np.kron(np.eye(n), Pxx_inv)
    #     W_B = np.kron(np.eye(n), Puu_inv)

    #     if row_std is not None:
    #         S = np.diag(1.0 / np.maximum(np.asarray(row_std, float).ravel(), 1e-8))
    #         W_A = (np.kron(S, np.eye(Pxx_inv.shape[0]))) @ W_A @ (np.kron(S, np.eye(Pxx_inv.shape[0])))
    #         W_B = (S) @ W_B @ (S)  # since B block is size 1 per row (SISO)

    #     return W_A, W_B

    # def get_W_old(self, mode: str = "ignore", row_std: np.ndarray | None = None) -> np.ndarray:
    #     W_A, W_B = self.get_precision_blocks(mode=mode, row_std=row_std)
    #     W = np.block([[W_A, np.zeros((W_A.shape[0], W_B.shape[1]))],
    #                   [np.zeros((W_B.shape[0], W_A.shape[1])), W_B]])
    #     return W
    
    def get_W(self): 
        D = np.diag(self.get_rows_std()**2) 
        P = self.P
        P_L = np.kron(D, P)
        W = np.linalg.pinv(P_L)
        return W
    
def ewma_var(prev_var, innov, alpha=0.01):
    """EWMA variance tracker for innovations (scalar)."""
    return (1 - alpha) * prev_var + alpha * (innov**2)

# --- Example of innovation-variance tracking per row (to feed into sigma_rows/sigma_y) ---
class OnlineVariance:
    """Simple online EWMA variance estimator."""
    def __init__(self, alpha=0.01, init_var=1.0):
        self.alpha = float(alpha)
        self.var = float(init_var)

    def update(self, innovation):
        self.var = ewma_var(self.var, float(innovation), alpha=self.alpha)
        return self.var
    
    @property 
    def std(self) -> float: 
        return float(self.var ** 0.5)

# -----------------------------
# Scalar-output RLS (one row)
# -----------------------------
class RLSRow:
    def __init__(self, d: int, lambda_f: float = 0.995, delta: float = 1e-2, theta_init=None):
        """
        d         : number of regressors (dimension of phi)
        lambda_f  : forgetting factor in (0,1]; closer to 1 = slower forgetting
        delta     : ridge init; initial P0 = (1/delta) * I_d
        """
        assert 0 < lambda_f <= 1.0, "lambda_f must be in (0,1]"
        self.d = int(d)
        self.lambda_f = float(lambda_f)
        self.theta = np.zeros(d, dtype=float) if theta_init is None else np.asarray(theta_init, dtype=float).reshape(d,)
        self.P = (1.0 / float(delta)) * np.eye(d)  # covariance
        # stats
        self.n_updates = 0

    def update(self, phi: np.ndarray, y: float) -> float:
        """
        One RLS update for scalar output y and regressor phi (shape (d,) or (d,1)).
        """
        phi = np.asarray(phi, dtype=float).reshape(self.d, 1)  # (d,1)
        y = float(y)
        # Gain
        denom = self.lambda_f + float((phi.T @ self.P @ phi).item())
        K = (self.P @ phi) / denom           # (d,1)
        # Prediction error
        y_hat = float(self.theta @ phi.ravel())
        e = y - y_hat
        # Update theta and P
        self.theta = self.theta + (K.ravel() * e)
        self.P = (self.P - (K @ (phi.T @ self.P))) / self.lambda_f
        self.n_updates += 1
        return e # return innovation for variance tracking

    def predict(self, phi: np.ndarray) -> float:
        phi = np.asarray(phi, dtype=float).reshape(self.d,)
        return float(self.theta @ phi)

    def reset(self, theta0: Optional[np.ndarray] = None, delta: Optional[float] = None) -> None:
        if theta0 is not None:
            theta0 = np.asarray(theta0, dtype=float).reshape(self.d,)
            self.theta = theta0.copy()
        if delta is not None:
            self.P = (1.0 / float(delta)) * np.eye(self.d)
        else:
            self.P = np.eye(self.d)

# ----------------------------------------
# Parallel RLS for [A B] (n rows in total)
# ----------------------------------------
class RLSAB:
    def __init__(self, n: int, lambda_f: float = 0.995, delta: float = 1e-2, ewma_alpha: float = 0.01, theta_init=None):
        """
        n         : number of states
        lambda_f  : forgetting factor
        delta     : ridge init for all rows
        """
        self.n = int(n)
        self.d = n + 1
        self.rows = [RLSRow(self.d, lambda_f=lambda_f, delta=delta, theta_init=theta_init[i]) for i in range(n)]
        self.var_rows = [OnlineVariance(alpha=ewma_alpha, init_var=1.0) for _ in range(n)] 

    def update(self, x_k: np.ndarray, u_k: float, x_kp1: np.ndarray) -> None:
        """
        Update all rows using one sample (x_k, u_k) -> x_{k+1}.
        """
        x_k = np.asarray(x_k, dtype=float).reshape(self.n,)
        x_kp1 = np.asarray(x_kp1, dtype=float).reshape(self.n,)
        phi = np.hstack([x_k, float(u_k)])    # (n+1,)
        for j in range(self.n):
            e_j = self.rows[j].update(phi, x_kp1[j])
            self.var_rows[j].update(e_j)

    def get_rows_std(self) -> np.ndarray:
        """Return estimated per-row noise std (shape (n,))."""
        return np.array([vr.std for vr in self.var_rows], dtype=float)

    def get_AB(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns estimated (A, B) where rows are from thetas.
        A: (n,n), B: (n,1)
        """
        Theta = self.get_theta()
        A = Theta[:, :self.n]
        B = Theta[:, self.n:].reshape(self.n, 1)
        return A, B
    
    def get_theta(self) -> np.ndarray:
        """Return stacked theta for all rows (shape (n, n+1))."""
        Theta = np.vstack([row.theta for row in self.rows])  # (n, n+1)
        return Theta

    def reset_theta(self, theta: np.ndarray) -> None:
        theta = np.asarray(theta, dtype=float).reshape(self.n, self.d)
        for i, row in enumerate(self.rows):
            row.theta = theta[i,:].copy()
        
    def reset(self, theta: Optional[np.ndarray] = None, delta: Optional[float] = None) -> None:
        for row in self.rows:
            row.reset(theta0=theta, delta=delta)



def make_precision_from_rls(P_hat_6x6: np.ndarray,
                            std_rows: np.ndarray,            # shape (2,) here
                            eps: float = 1e-10,
                            kappa_max: float = 1e6,
                            sigma_min: float = 1e-4,
                            eig_ceiling: float | None = None):
    """
    Build a precision W suitable for weighting in GLS:
        W_raw = T * (P_hat^{-1}) * T^T,   T = kron(diag(1/std_rows), I_3)
    Then condition (SPD floor + cond cap [+ optional eig ceiling]).
    Returns: W_cond (6x6), stats dict.
    """
    # inverse of 6x6 without explicit np.linalg.inv (more stable)
    I6 = np.eye(6)
    Pinv = np.linalg.solve(P_hat_6x6, I6)

    # 1/std with floor -> scales like 1/variance in W_raw
    std = np.maximum(np.asarray(std_rows, float).reshape(-1), sigma_min)
    inv_std = 1.0 / std
    T = np.kron(np.diag(inv_std), np.eye(3))  # (6x6)

    W_raw = T @ Pinv @ T.T

    # condition
    s, U = np.linalg.eigh(0.5 * (W_raw + W_raw.T))
    s = np.maximum(s, eps)
    if eig_ceiling is not None:
        s = np.minimum(s, eig_ceiling)
    cond = float(s.max() / s.min())
    if cond > kappa_max:
        s = np.maximum(s, s.max() / kappa_max)
    W_cond = U @ np.diag(s) @ U.T

    stats = dict(trace=float(np.trace(W_cond)),
                 cond=float(s.max()/s.min()),
                 min_eig=float(s.min()),
                 max_eig=float(s.max()))
    return W_cond, stats