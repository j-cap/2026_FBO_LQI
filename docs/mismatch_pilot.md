# Linear-model / nonlinear-plant opportunity pilot

Purpose: determine whether realistic local linearization mismatch changes the choice
of LQI weights. No fleet learner, optimizer comparison, or statistical confirmation.
Old experiments and their conclusions remain unchanged.

## Frozen small design

- One nominal vehicle, speed 15 m/s, existing tanh tire plant, mu=1, actuator tau=0.2 s.
- One 20-second gentle steering multisine (peak bounded by 0.4 degrees); noise-free full
  beta, yaw, actual steering observations. Known local mass, inertia, geometry and speed.
- Fit Cf, Cr and actuator tau by bounded output-error least squares using an exact ZOH
  linear bicycle + actuator predictor. Fixed initial guess (60k,60k,0.3); no true tire
  coefficients supplied to fitting. Record excitation and fit diagnostics.
- 64 fresh Sobol weights in the existing [-3,3]^3 domain, seed 901. No old optimum added.
- Same identified-model gains, integrator normalization, reference prefilter, limits,
  corrected antiwindup and initial conditions on both systems. This preserves the
  implementation's Q_integral = 10^xi_3 / (scale_3^2 * Ts^2).
- Six-second double-sine yaw maneuver, one-second half-duration. Filtered reference
  rescaled to peak yaw-equivalent lateral demands 1.5, 4, 7 m/s² (r_peak=demand/15).
  These are exploratory stresses, not validated vehicle operating limits. Actual
  lateral acceleration also contains beta_dot; demand is a reference descriptor.
- No noise, seed replications, speed sweep or fleet generation.
- Outputs: true yaw RMSE E [deg/s], actual steering-rate RMS S [deg/s], constraint peaks.
  S uses the exact squared-rate integral over each held-command actuator interval.
- Linear propagation is exact ZOH; nonlinear RK4 compares 10/20 substeps and, if needed,
  20/40. Require differences <= 1e-5 + 0.001*max(metric,0.001) for BOTH E and S.
  Unresolved integration is excluded from the plant reference, never penalized as a
  bad objective. Physical early exits are separately recorded; incomplete metrics
  are not treated as numerically converged. Identification checks 20 vs 40 substeps.

192 candidate/context pairs, 192 linear rollouts and normally 384 nonlinear rollouts
(576 maximum), plus identification. Serial execution avoids startup overhead. Each
candidate prints progress; atomic case files support reruns after interruption.
Smoke uses four candidates and a separate output directory.

## Decision test

For each demand, use exploratory yaw RMSE limits of 3%, 6%, 12% of peak reference yaw
rate. These are fixed diagnostic requests, not external safety/comfort specifications.
The old 0.92 deg/s noisy metric is not imported into this noiseless experiment.

Select minimum predicted S among predicted physically feasible candidates satisfying
E <= limit. Evaluate that exact choice on the nonlinear plant. Compare with minimum
actual S among numerically resolved, physically feasible sampled candidates satisfying
E <= limit. This is a finite-grid hindsight reference, NOT an optimum or deployment
procedure. Report no-solution, tracking failure, physical failure, numerical unknown,
and feasible selection with steering excess separately. Near-boundary differences
at the integration tolerance require inspection, not a strong conclusion.

Promising evidence: a missed requirement with a sampled feasible alternative, or an
appreciable steering excess (e.g. >5%, interpreted descriptively). Only prediction
bias without decision consequences is weak motivation. No feasible sampled solution
may indicate infeasible requirements, controller limitations, or sparse candidate
coverage. One promising vehicle cannot establish cross-vehicle shared discrepancies
or a benefit from FL. Next test, if warranted, is a tiny source/target correction
comparison, including stronger local modeling and one-source design-library transfer.

## Run from repository root (Windows / conda)

```bat
conda activate paper_FBO_LQI
git switch study/cooperative-calibration
git pull --ff-only origin study/cooperative-calibration
python code/scripts/mismatch_pilot.py smoke
python code/scripts/mismatch_pilot.py run
```

Open `code/results/mismatch_pilot_v1/report.html`. CSVs, raw cases, identified model,
excitation data and a CRLF-normalized input signature are saved alongside it.
Configuration/code changes require a new tag; interrupted identical runs resume.

```bat
python code/scripts/mismatch_pilot.py collect
git commit -m "Add linear-model nonlinear-plant mismatch pilot results"
git push origin study/cooperative-calibration
```

Collect regenerates the report, verifies completion and the input signature, and
force-stages only this result directory despite the repository's results ignore rule.
It does not include smoke results or unrelated files.
