# Paired explanation diagnostics (v7)

Purpose: quantify the contribution of model-based gain redesign and distinguish
true-state tracking error from noise-inclusive measured error. This is a bounded
mechanistic follow-up, not another BO campaign or independent confirmation study.

## Four cases on each of six targets and five actuator time constants

All cases except the reference use the SAME retrospective common weight vector.
It is a diagnostic probe; no claim of independent weight selection is made.

| Case | Model used to design gains | Design actuator tau | Weights |
|---|---|---|---|
| target_redesign | target identified model | actual target tau | common |
| target_frozen_tau | target identified model | fixed .2 s | common |
| donor_frozen | same-family vehicle in opposite fleet realization | fixed .2 s | common |
| reference | target identified model | actual target tau | frozen v5 reference |

Donor assignment is deterministic and excludes the target. Using the same family
in the other fleet realization isolates a modest vehicle-model mismatch; this is
not an exhaustive cross-family fixed-gain transfer study. There is no donor selection
based on performance. The comparisons separate bandwidth redesign from vehicle
redesign but their effects need not be additive.

K and the reference prefilter kr are both frozen where specified. Integral coordinates
eta=Ts*z, leak=.002, tracking anti-windup T_aw=.2 s, noise, limits, reference and
nonlinear plant remain unchanged. Gain-scaled anti-windup uses the implemented Ki
in each case. Do not force the back-calculation coefficient to be numerically equal
across gains, which would change its command-space tracking time constant.

At target tau=.2, target_redesign and target_frozen_tau must be identical. Frozen
controllers are NOT rejected for an unstable target-model linear radius before
simulation: radius is reported alongside finite-horizon feasibility. The latter is
not a proof of asymptotic nonlinear stability. Constraint aborts remain failures;
truncated episodes do not receive a full-episode tracking score.

## Matched observations and checks

Reuse the v6 evaluation bank, seed 1900000, 40 episodes per maneuver (80 total).
Both true and measured errors are logged on the SAME noisy trajectories; measurement
noise is not switched off. Preserve the existing direct-state feedback and noisy
integrator convention. Report mean episode RMSE for each error and episode MSEs:

    measured_MSE = true_MSE + noise_MSE + 2*mean(true_error*noise).

The sample cross term is retained, not assumed zero. Do not subtract noise variance
from the squared mean RMSE. The reference is rescored using both metrics on matched
episodes; no reference reselection or retuning for true error occurs. Consequently,
it is still the measured-objective reference, not a true-error optimum.

Every episode runs with 5 and 10 RK4 substeps. Feasibility and convergence are reported
for both measured and true error. The redesigned common-weight case must reproduce
v6 score and state/rate maxima within tight numerical tolerance or the run stops.
Reconstructed target identified models are checked against the saved models.

Save trajectories for the first episode of each maneuver, selected before examining
results. All other episodes retain scalar scores, MSE components and constraint
metrics. Total: 120 controller/bandwidth/vehicle cases, 9600 paired episode records.
Progress reporting and per-episode atomic checkpoints support interruption/resume.
Code and inputs are hashed; old experiments remain unchanged.

## Run locally from repository root

```bat
git fetch origin
git switch --track origin/experiment/actuator-mechanism
python code/scripts/actuator_mechanism.py preflight
python code/scripts/actuator_mechanism.py smoke --n-jobs 2
python code/scripts/actuator_mechanism.py run --n-jobs 2
```

If the branch already exists, use `git switch experiment/actuator-mechanism` and
`git pull --ff-only`. Stop if a command fails. Smoke covers one vehicle, two taus,
four cases, and two episodes per case, in a separate directory. Repeat an interrupted
run command to reuse completed episode checkpoints. No optimization is performed.

Results: `code/results/actuator_mechanism_v7`. Summaries are generated automatically.
To regenerate them, use `python code/scripts/actuator_mechanism.py analyze`.

```bat
python code/scripts/actuator_mechanism.py collect
git commit -m "Add paired gain redesign and tracking error diagnostics"
git push -u origin experiment/actuator-mechanism
```

Collect requires all expected episode records and force-adds only this run's manifest,
controller gains, episode records/selected traces and CSV summaries. It excludes smoke.

## Short HTML after the results are available

Prepare one concise, self-contained explanation with embedded figures, using the
corrected evidence and these diagnostics. Suggested order:

1. Andreas's baseline question and the distinction between transferring weights
   and transferring a controller (one compact equation/diagram).
2. Direct-transfer success across tau from v6, with the primary 5% threshold and
   six-vehicle scope explicit (one figure).
3. Gain redesign effects and true versus measured error from v7, retaining failures
   and numerical flags (one paired figure, supplemented by a trajectory only if useful).
4. A short explanation of the overlapping good-weight regions from saved v5 grids,
   then supported conclusion, limitations and the remaining research decision.

Keep reference definitions and retrospective common-weight selection visible. Do
not present the 1800 correlated transfer rows as independent systems. Use plain
language; omit implementation details that do not affect interpretation. Do not
construct the explanatory figure around an assumed favorable redesign effect:
report whichever outcome the paired experiment shows.
