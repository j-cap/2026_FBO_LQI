# Complementary coverage: small equal-budget pilot

Question: does combining corrections from sources with complementary operating
coverage improve weight selection on a third vehicle? Distinguish the value of
coverage from an intrinsic advantage of multiple vehicles.

## Frozen design

A is nominal; B has mass x1.20 and inertia x1.25; target has front cornering stiffness
x0.75. All at15 m/s, same tire law/mu=1 and actuator tau=0.2. Each identifies its own
linear model using the unchanged 20-second gentle experiment. These controlled
parameter changes are exploratory, not sampled real vehicle populations.

Sources calibrate at demands1.5 and7 m/s² only. Target evaluation includes these
endpoints and unseen demands2.75,4.25,5.5. Fresh weight pools:24 calibration weights
(first24 of32 scrambled Sobol points, seed2901),64 evaluation weights (seed3901).
All controller, integrator, numerical and physical checks come from mismatch_pilot.
Noise-free, no seed replications. 96 source cases +440 target cases =536 paired cases,
normally1072 nonlinear integration rollouts. Serial progress and atomic resume.

Every data-based policy receives24 calibration experiments:

| Policy | Data available |
|---|---|
| A low only |24 weights on A at1.5|
| B high only |24 weights on B at7|
| Pooled A low/B high |12 weights at each endpoint, different sources|
| Pooled B low/A high |Coverage-role swap, same24 total|
| A full coverage |12 weights at each endpoint on A|
| B full coverage |12 weights at each endpoint on B|
| Target local diagnostic |12 weights at each endpoint on target|
| Direct pooled library |Same24 source observations as primary pooled correction|

Model-only and fixed25% margin use zero calibration observations. Identification
costs apply separately: each vehicle has20 seconds of local identification; two-source
methods involve two source identification records. Do not call equal calibration
counts equal total acquisition cost without accounting for these records.

## Fixed simple correction, no neural learner

Each client sends counts and summed log(actual/predicted) E and S ratios per endpoint.
Coordinator computes geometric-mean multipliers and linearly interpolates their logs
in squared demand. This low-order interpolation is fixed before this run, motivated
by amplitude-dependent nonlinearity; no fitted hyperparameters or target tuning.
It replaces the preceding pilot's per-demand median because endpoint summaries now
must support unseen demands. At least3 valid pairs per endpoint are required.
Restricted single-endpoint baselines hold their multiplier constant. Thus coverage
advantage is explicitly tied to this limited predictor, not a universal impossibility
of predicting unseen demands from one source. Compare with full-coverage controls.

All correction methods select among88 target-model predictions. Fitting uses only
budgeted endpoint records; target plant outcomes are excluded except the explicitly
labeled target-local diagnostic. Prediction-error evaluation uses only the64 weights
not in calibration, including the three unseen demands. Counts/sums aggregation is
exactly equivalent to centralized fitting of this endpoint model; there is no claim
of a novel FL algorithm, formal privacy, or centralized superiority. Trajectories
stay conceptually local; this pilot simulates all clients in one process.

Direct pooled-library baseline interpolates log(E/demand), log(S/demand) in squared
demand for each of the12 source-tested weights, using A low/B high measurements,
then transfers the selected weights and recomputes target gains. It has no target
performance correction. This is a fixed simple source-library interpolation policy,
not an optimized source BO strategy or oracle across unobserved contexts.

## Scoring and interpretation

Same fixed1%..30% tracking-limit grid per demand:150 correlated requests per method.
Report endpoints and unseen interiors separately, abstentions, physical/numerical
unknowns and violations, feasible sampled-reference availability, and steering excess
conditional on success. Reference is best among88 evaluated target candidates, NOT
a global optimum. Same numerical boundary handling as correction_transfer.
The large plant table is offline scoring data, not free policy calibration data.

Success is fewer violations/less conservative steering at useful coverage. Pooling
beating restricted sources but matching full-coverage single sources demonstrates
complementary information access, not an intrinsic fleet advantage. Pooling losing to
full-coverage sources can expose heterogeneity or interpolation problems. Role swap
checks source identity/coverage confounding. Fixed25% margin remains a strong simple
baseline. Do not expand seeds or fleet before analyzing this result.

## Windows commands, repository root

```bat
conda activate paper_FBO_LQI
git switch study/cooperative-calibration
git pull --ff-only origin study/cooperative-calibration
python code/scripts/coverage_transfer.py smoke
python code/scripts/coverage_transfer.py run
start code\results\coverage_transfer_v1\report.html
python code/scripts/coverage_transfer.py collect
git commit -m "Add complementary source coverage pilot results"
git push origin study/cooperative-calibration
```

Smoke:6 calibration weights (3 per endpoint),4 evaluation weights,74 cases, separate
output directory. collect verifies case completeness and the CRLF-normalized input
signature and force-stages only this run. Changed inputs require a new run tag.
