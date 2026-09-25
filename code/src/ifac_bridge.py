"""
Single import point for the vendored IFAC uncertainty-aware clustered identification
pipeline (`src/ifac`, copied from paper_UncertaintyAwareClustering/src2 - see
`src/ifac/__init__.py`). The rest of paper_FBO_LQI imports these names from here, so the
vendored package's layout stays an implementation detail. No sys.path manipulation and
no dependency on any other experiment directory.
"""
from __future__ import annotations

from src.ifac import client as _ifac_client_mod
from src.ifac import clustering as _ifac_clustering_mod
from src.ifac import rls as _ifac_rls_mod
from src.ifac import run as _ifac_run_mod
from src.ifac import server as _ifac_server_mod
from src.ifac import utils as _ifac_utils_mod

# ---- re-exports used throughout paper_FBO_LQI/src --------------------------------
BikeParams = _ifac_utils_mod.BikeParams
Cluster = _ifac_utils_mod.Cluster
LQIConfig = _ifac_utils_mod.LQIConfig
cluster_prototype = _ifac_utils_mod.cluster_prototype
nominal_params = _ifac_utils_mod.nominal_params
perturb_params = _ifac_utils_mod.perturb_params
reset_gains = _ifac_utils_mod.reset_gains
build_bicycle_beta_r = _ifac_utils_mod.build_bicycle_beta_r
discretize = _ifac_utils_mod.discretize
design_lqi = _ifac_utils_mod.design_lqi
closed_loop_step_lqi = _ifac_utils_mod.closed_loop_step_lqi
init_integrator = _ifac_utils_mod.init_integrator
lc_yaw_rate_ref = _ifac_utils_mod.lc_yaw_rate_ref
pack_theta_AB = _ifac_utils_mod.pack_theta_AB
unpack_theta_AB = _ifac_utils_mod.unpack_theta_AB

Client = _ifac_client_mod.Client

RLSAB_BlockCov = _ifac_rls_mod.RLSAB_BlockCov
make_precision_from_rls = _ifac_rls_mod.make_precision_from_rls

mahalanobis_kmeans = _ifac_clustering_mod.mahalanobis_kmeans
clustering_agreement = _ifac_clustering_mod.clustering_agreement

generate_fleet = _ifac_server_mod.generate_fleet
gls_mean_robust = _ifac_server_mod.gls_mean_robust
spd_condition = _ifac_server_mod._spd_condition

run_episode_update_rls = _ifac_run_mod.run_episode_update_rls
run_CL_client_model = _ifac_run_mod.run_CL_client_model
