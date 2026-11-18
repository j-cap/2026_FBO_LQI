# Anti-windup correction and paired replay

This audit precedes any new optimization. It evaluates the same saved weights on the
same physical vehicles and noise realizations, with three integral updates. Results
are development diagnostics, not an independent confirmation of transfer performance.

## Mathematics

The controller uses `raw = -Kx*x - Ki*eta + kr*r_ref`, with
`eta = Ts*z` and yaw error `e = measured_r - r_ref`. The historical update was
`eta_next = .998*eta + Ts*(e + .2*(applied-raw))`.
For positive Ki, this correction pushes the raw command farther into saturation.
The same sign appears in `src/ifac/utils.py:closed_loop_step_lqi` and
`src/nonlinear_plant.py:closed_loop_step_lqi_nonlinear`. This identifies an affected
code path; it does not establish how many earlier experiments changed materially.
Without saturation, the three updates are identical.

The replay compares:

- `legacy`: exactly the original equation, checked against saved episode scores.
- `sign_only`: replace `(applied-raw)` by `(raw-applied)`, retain coefficient .2.
- `tracking`: `eta_next = .998*eta + Ts*(e + (raw-applied)/(Ki*T_aw))`.

For tracking, the anti-windup contribution to the raw command is
`Ts/T_aw*(applied-raw)`, independent of Ki. In sample-sum coordinates use
`z_next = .998*z + e + Ts/(Ki_z*T_aw)*(raw-applied)`, where `Ki_z = Ts*Ki`.
We fix T_aw = .2 seconds for every vehicle and actuator tau; it is a controller
setting, not an identified actuator parameter or an optimized value. At Ts=.05,
the saturated integrator coefficient is .748 with frozen state/reference/error.
This scalar check is not a nonlinear closed-loop stability proof. The sign-only
coefficient is `.998 - .01*Ki`; large Ki can make that correction too aggressive.

## Scope and provenance

The new `src/actuator_antiwindup.py` evaluator changes only the integral update.
Old evaluator and utilities remain unchanged for exact reproduction, including
fleet admission and identification. Old entry points therefore still run the
legacy controller; use the new audit entry point for corrected evaluations.
Do not silently replace the old helpers: this can change which vehicles are admitted.
A subsequent corrected optimization run must use fixed saved vehicles and explicitly
select the corrected evaluator. Its references must be recomputed.

The replay contains all 75 saved v3 episode reports (6,000 episodes): old/new
reference controllers across actuator bandwidths and the original source optimizer's
final controllers. It does not replay every optimizer-ablation incumbent or establish
corrected optimizer rankings. Each episode is evaluated using all three rules at
5 and 10 RK4 substeps. Legacy scores must reproduce saved values within numerical
tolerance, otherwise execution stops. Cost is reported only for wholly feasible banks;
failed episodes and unresolved integration checks remain explicit.

Inputs and code have SHA-256 manifests. Old result directories are untouched.
New output: `code/results/actuator_antiwindup_v4`; smoke output is separate.
Each completed episode is an atomic checkpoint. Repeating replay resumes it.
Progress prints at controller boundaries and roughly every 30 seconds between episodes.
Expect substantial runtime: 36,000 single-episode evaluations, with no GP fitting.

## Run from repository root (Windows CMD / activated environment)

```bat
git fetch origin
git switch --track origin/fix/antiwindup-replay
python -m pytest code/tests/test_actuator_antiwindup.py code/tests/test_actuator_repair.py code/tests/test_actuator_bandwidth.py -q
python code/scripts/actuator_antiwindup_audit.py preflight
python code/scripts/actuator_antiwindup_audit.py smoke --n-jobs 2
python code/scripts/actuator_antiwindup_audit.py replay --n-jobs 2
```

If the branch already exists locally, use `git switch fix/antiwindup-replay` and
`git pull --ff-only` instead of the tracking-branch creation command.
Replay automatically creates `processed/episodes.csv` and `processed/controllers.csv`.
To regenerate those summaries: `python code/scripts/actuator_antiwindup_audit.py analyze`.

Commit and push results:

```bat
python code/scripts/actuator_antiwindup_audit.py collect
git diff --cached --stat
git commit -m "Add paired anti-windup correction replay results"
git push -u origin fix/antiwindup-replay
```

`collect` checks completeness and force-adds only this run's manifest, episode
checkpoints and CSV summaries, avoiding ignored-path and wildcard problems.
Smoke results are not collected. No automatic commit or push is performed.

## Decision after replay

First check legacy reproduction, per-episode feasibility and numerical convergence.
Then compare sign-only and tracking to separate sign and gain effects. Quantify
how much the old references/source weights change and whether failures persist.
Only then fix the controller convention for subsequent runs, rebuild references,
and rerun calibration/transfer comparisons. Saved weights are not optima of the
corrected problem, and this audit cannot demonstrate a need for fleet-informed BO.
The older ideal-steering experiments require their own saturation-impact audit
before carrying their numerical claims into the paper.
