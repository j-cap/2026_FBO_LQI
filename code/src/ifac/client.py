
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np
from .utils import (
    BikeParams, 
    Cluster, 
    LQIConfig
)
from .rls import RLSAB

@dataclass
class Client:
    client_id: str
    cluster: Cluster
    params: BikeParams
    A: np.ndarray; B: np.ndarray; C: np.ndarray; D: np.ndarray
    Ad_hat: np.ndarray; Bd_hat: np.ndarray; Cd_hat: np.ndarray; Dd_hat: np.ndarray
    Ad_true: np.ndarray; Bd_true: np.ndarray; Cd_true: np.ndarray; Dd_true: np.ndarray
    # Controller design spec + gains (filled later)
    lqi_cfg: LQIConfig = field(default_factory=LQIConfig)
    Kx: np.ndarray | None = None
    Ki: np.ndarray | None = None
    k_r: float | None = None
    rls: RLSAB | None = None
    theta_hist: list = field(default_factory=list)
    theta_hist_RLS: list = field(default_factory=list)
    Ad_hist_RLS: list = field(default_factory=list)
    Bd_hist_RLS: list = field(default_factory=list)
    W_raw: np.ndarray | None = None  # precision matrix for clustering/GLS
    std_rows: np.ndarray | None = None  # per-row innovation std dev
    round_counter: int = 0
    T_lc: float = 0.2 # half lane change duration used to generate reference trajectory
    t_ref: list[float] = field(default_factory=list)  # time vector for reference trajectory
    r_ref: list[float] = field(default_factory=list)  # reference trajectory for this client
    # RLS estimates (filled later)