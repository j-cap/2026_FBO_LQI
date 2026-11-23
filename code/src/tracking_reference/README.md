# CFL-FF reference sources

Selected from Jakob Weber's `CFL_FF_AK_audit_minimal.zip`, 22 September 2026,
archive root `CFL_FF_AK_audit_minimal/`. Exact archive and original member SHA256
values are in `provenance.json`. No external package installation or access to
the archive is needed for the new experiment.

- `cfl_dynamics.py`: original six-state scalar dynamics, used as a parity reference.
- `cfl_controller.py`: original scalar feedback controller, used as a parity reference.
- `cfl_trajectories.py`: original reference generator and lane-change helpers, used
  to build the held-out bank. Its Euler pose integration is replaced by accurate
  integration in the new task; its sampled speed/curvature generation is preserved.

Changes to these copies are limited to LF/trailing-whitespace normalization, removal of the unused
dynamics utils import, and a local controller wrap-angle helper. The new vectorized
physics, payload-CG inertia correction, paired-noise experiment, finite gain pool,
reference banks, score, and selection analysis are defined separately in
`tracking_task.py`, `tracking_analysis.py`, and the explicit run configuration.
The original archive was not modified. No archived results or training data are
needed or copied into this repository.
