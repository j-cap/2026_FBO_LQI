# Small correction-transfer pilot

Purpose: test whether a simple performance correction learned on a nominal source
improves requirement-based LQI weight selection on a payload target. This is a
noise-free developmental mechanism study, not confirmation or an FL experiment.

## Fixed setup and budget

Reuse the mismatch pilot evaluator unchanged. Source: nominal vehicle at 15 m/s.
Target: mass x1.20, yaw inertia x1.25, same geometry, tire parameters and tau=0.2 s.
This is a controlled payload parameter change, not a detailed load distribution model.
Both vehicles identify their OWN linear tire + actuator model from a 20-second gentle
multisine. Preserve local gain redesign, nonlinear tire plant, antiwindup, numerical
checks and the three demands (1.5,4,7 m/s²). No random fleet, repetitions or noise.

Eight fresh Sobol calibration weights (seed1901) per vehicle are distinct from the
64 preceding-pilot evaluation weights (seed901). Each weight is evaluated at all
three demands. Total 2 x 72 x 3 = 432 paired model/plant cases, normally 864 nonlinear
integration rollouts plus 432 linear predictions; refinement can increase this.
These are computational evaluations. Method calibration budgets are separately
24 source or target plant experiments; the dense source comparator uses 216.
Local identification costs apply to every method and are listed separately.

Per demand and metric, fit a multiplier as the median actual/predicted ratio from
ONLY the eight calibration weights. Require >=3 resolved feasible pairs; otherwise
abstain. Multipliers correct E (yaw RMSE) and S (steering-rate RMS). A constant positive
S multiplier cannot change candidate ordering within a demand: the decision mechanism
is corrected feasibility prediction. No weight-dependent regression, neural network,
interpolation across demands, or hyperparameter search is involved.

## Comparisons

1. Model-only: target predictions, no plant calibration.
2. Fixed 25% tracking margin: predicted E x1.25, unchanged S, no plant calibration.
   This deliberately simple margin is motivated by the preceding pilot, not an
   independently justified uncertainty bound. It is not a guarantee.
3. Source correction: 24 source observations, no target calibration.
4. Direct transfer: pick among the SAME eight source-calibrated weights using observed
   source metrics for the requested limit; recompute gains on the target.
5. Target correction: 24 target observations; diagnostic for the correction model's
   adequacy, not a zero-shot method.
6. Dense source library: all72 source weights (216 observations); stronger, more
   expensive diagnostic comparator. No target outcomes in selection.

Model-based methods select among all72 weights; the extra64 are available via local
simulation, not free physical evaluations. Prediction-error analysis uses ONLY the
64 weights excluded from fitting. Selection uses no target plant outcomes except the
explicitly budgeted target correction. Target outcomes are used subsequently to score
selected weights and construct a finite-candidate hindsight reference over all72.

For every demand evaluate a fixed 1%..30% peak-yaw tracking-limit grid, step1%, including
strict requirements that may have no sampled solution. This grid is exploratory and
chosen after the previous pilot; do not claim independent confirmation. Reference
availability is recorded for each request. Correlated threshold counts are NOT
statistical success probabilities. Boundary cases within measured integration
uncertainty are separate. Report abstention, violation, physical/numerical failure,
and steering excess conditional on passing; fewer violations through abstention
alone are not a benefit. Zero feasible candidates does not prove true infeasibility.

## Outputs and interpretation

Self-contained report.html: outcome map across the entire fixed requirement grid,
outcome counts, steering excess. CSV: all decisions, held-out prediction errors.
JSON: learned factors, numerical/failure summary, manifest, individual cases and
identification fits; NPZ: identification trajectories. Resume is atomic per case;
input changes require a new tag. collect validates completion and the normalized
code/config signature before staging only this result directory.

Promising: source correction reduces target violations while retaining useful
coverage and steering quality, beyond direct transfer and the fixed margin.
If local correction works but source correction does not, transfer is the problem.
If neither works, a demand-only multiplier may be inadequate. If source correction
matches direct transfer or margin, the extra learner has little demonstrated value.
A successful single source still does NOT establish a need for a fleet or FL.
Do not expand seeds/fleet until this comparison is informative.

## Windows commands (repository root)

```bat
conda activate paper_FBO_LQI
git switch study/cooperative-calibration
git pull --ff-only origin study/cooperative-calibration
python code/scripts/correction_transfer.py smoke
python code/scripts/correction_transfer.py run
start code\results\correction_transfer_v1\report.html
python code/scripts/correction_transfer.py collect
git commit -m "Add small source-target correction transfer results"
git push origin study/cooperative-calibration
```

Smoke uses four calibration and four evaluation weights per vehicle in a separate
output directory. It is a plumbing check, not included in the reported experiment.
