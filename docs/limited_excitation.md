# Limited-excitation simulation study — Gate 1

**Research assumptions informed by vehicle-testing standard scopes. Not an ISO
compliance test, validated commissioning procedure, sensor specification, or claim
that fleet knowledge is necessary.** Prepared 20 November 2026.

## Question and scope

Do limited local observations leave uncertainty that actually changes the selection
of LQI design weights? Before another learner is built, test both:

1. whether the possible plants require materially different weight choices; and
2. whether their responses can be distinguished using the stipulated excitation
   and observation model.

This first gate does not fit a local estimator, train a fleet model, or claim a cost
advantage. It provides an optimistic diagnostic for local information: exact plant
hypotheses and their nuisance parameters are known when evaluating distinguishability.
If even that ideal test struggles, there may be an information gap for this signal.
If it succeeds, a local estimator is a necessary next comparator. Success of the
ideal test does not prove a practical estimator succeeds, while difficulty for one
input does not prove all permitted excitation is uninformative.

## What testing standards contribute

Public official scope descriptions checked 20 November 2026:

- ISO 15037-1:2019: general conditions for measurement, processing, environment,
  preparation/loading and reporting: https://www.iso.org/standard/70164.html
- ISO 7401:2011: open-loop lateral transient response tests:
  https://www.iso.org/standard/54144.html
- ISO 4138:2021: open-loop steady-state circular-driving behavior:
  https://www.iso.org/standard/81710.html
- ISO 22140:2021: simulation validation against measured transient-response tests:
  https://www.iso.org/standard/72680.html

We use these scopes to motivate controlled inputs, declared initial conditions,
separate characterization/evaluation, and reporting actual responses. We have not
implemented the full standards. The multisine, acceleration caps, durations, noise,
plant coefficients and acceptance thresholds below are OUR research assumptions.
A constant-speed steering multisine here is not an ISO 7401 compliance claim. The
source descriptions do not prescribe an allowable commissioning budget or prohibit
stronger excitation. Actual operational limits require application-owner input.

## Controlled plant family

Only the effective tire peak-force coefficient mu changes: 0.8, 1.0, 1.2, applied to
both axles in Fy=-mu*Fz*tanh(C*alpha/(mu*Fz)). These bracket the prior mu=1 model;
they are not measured tire families or a validated road-friction distribution.
Mass 1500 kg, inertia 2700 kg m², Cf=80 kN/rad, Cr=90 kN/rad, speed 15 m/s,
geometry and actuator tau=0.2 s stay identical. Small-slip tire stiffness therefore
stays identical. No load transfer, combined slip, additional tire relaxation, or
actuator mismatch is introduced.

All candidate gains are synthesized from the **exact common tangent model**, including
the actuator, with the existing normalized Q, R, prefilter and corrected anti-windup.
This deliberately favorable local-model assumption isolates nonlinear information.
There is no hidden use of mu in synthesis. It is NOT a measured identification result.
No source–target metadata relation is invented. If this gate is promising, that relation
must be specified before claiming personalized fleet inference.

## Limited excitation

- Same 20-second multisine shape used previously: frequencies 0.23, 0.71 and 1.3 Hz,
  relative amplitudes 1, 0.5 and 0.25; sampling interval 0.05 s.
- Three separate hypothetical permission envelopes: actual |ay| <= 1, 2, 3 m/s².
- Common input amplitude is computed from the tangent model at 80% of each cap;
  wheel command magnitude is additionally limited to 3 degrees.
- Actual |ay|, actual wheel angle <=3 degrees, wheel rate <=20 degrees/s and the
  existing 6-degree sideslip stop are monitored at integration substeps. Exceedance
  aborts the characterization and is reported. No free rescaling using target
  nonlinear responses or replacement episode is performed.
- These are alternative budgets, not 60 seconds cumulatively available to a policy.
- Record actual ay=U*(beta_dot+r), not U*r or the reference demand. This is the
  modeled CG acceleration in the planar bicycle approximation.
- Measurements for the information diagnostic: yaw [deg/s], CG ay [m/s²], actual
  wheel angle [deg]. No sideslip measurement.
- Engineering-assumption marginal measurement SDs: 0.1 deg/s, 0.1 m/s², 0.05 deg.
  Independent channels with stationary temporal AR(1) correlation rho=0.9 at 20 Hz.
  These are not vendor specifications. No bias/drift or process disturbance is modeled.

No random sensor samples are necessary for this first diagnostic. We compute the
exact Gaussian separation between the saved noise-free mean trajectories under
this stipulated observation noise. For each pair, whiten their difference with the
stationary AR(1) covariance to obtain D. For equal priors and TWO known hypotheses,
optimal binary classification error is Phi(-D/2), expected log likelihood ratio D²/2.
This accounts for repeated samples and temporal correlation. Per-sample differences
alone do not establish indistinguishability. It is not a three-class error or an
estimated success rate. Prior availability to an actual local method is not claimed.
A later practical local fit must use only its allowed information.

Compare 20/40 integration substeps for characterization. Both runs must complete and
maximum measurement discrepancy must be <=0.001 marginal noise SD in every channel.
All uncompleted/unresolved characterizations are excluded from pairwise separation.
Saved NPZ files contain means (truth), not noisy observations for a learner.

## Decision relevance

32 fresh Sobol weights (seed 8901), plus TWO predetermined anchors: the historical
common vector from the corrected study and the original hand-tuned vector. Total34.
The historical common vector was selected retrospectively in the earlier study; it
is frozen before this new study, not an independent validation result from that study.

Test each at demands 3, 5 and 7 m/s² (yaw-equivalent reference amplitude), six-second
maneuver, all three plants: **306 paired candidate/context/plant evaluations**.
These demand levels are diagnostic stresses, not validated operating limits. Some
may have no feasible candidate; do not interpret that as evidence for federation.

Closed-loop decision scoring remains noise-free full-state feedback, as in the
preceding pilot, to isolate the physical weight-selection question. The AR(1) model
applies ONLY to characterization observations, not controller feedback. No practical
noisy closed-loop reliability claim follows. Compare RK4 10/20 and, if needed,20/40
substeps for E and S; use existing tolerance 1e-5+0.001*max(abs(metric),0.001).
Synthesis failures, physical failures and numerical unknowns are separate.

Primary requests are tracking RMSE <=10%,20%,30% of peak reference yaw rate, with
minimum steering-rate RMS among feasible candidates. A weight is acceptable when
feasible and within10% of that plant's best sampled steering RMS. These are fixed
developmental criteria; no externally justified specification is claimed.

Output per request:
- feasible and acceptable sets and pairwise Jaccard overlaps;
- number of weights acceptable on ALL variants;
- best hindsight common feasible choice minimizing worst relative steering excess;
- transfer of each plant's best sampled choice to the others;
- performance of the two fixed anchors.

The hindsight common/donor choices access evaluation truth: they are diagnostic
upper benchmarks for simple alternatives, not deployable learned policies. Candidate
sets are finite. An empty common set is not proof that no common continuous solution
exists. Keep the full1–30% threshold sweep descriptive and do not choose a primary
threshold based on favorable outcomes. Threshold requests are correlated, not trials.

## Decision after Gate 1

- Common acceptable weights across primary requests: little motivation for personalized
  calibration under these assumptions; do not build a fleet learner yet.
- Different acceptable choices but clearly distinguished mean trajectories: strengthen
  local identification before introducing federation.
- Different choices AND substantial pairwise confusion: investigate this information
  gap, its sensitivity to a better permitted input, realistic sensor assumptions and
  the source–target link. Still no proof that FL adds value.
- No feasible candidates or many unresolved cases: diagnose task/pool/numerics first.

Do not retune mu, noise or thresholds after observing outcomes merely to obtain the
third case. Freeze a new version and state the reason for any scientific revision.
A useful gap must ultimately be tied to an actual test-access or cost restriction.

## Run locally (Windows, repository root)

```bat
conda activate paper_FBO_LQI
git switch study/cooperative-calibration
git pull --ff-only origin study/cooperative-calibration
python code/scripts/limited_excitation.py smoke
python code/scripts/limited_excitation.py run
start code\results\limited_excitation_v1\report.html
python code/scripts/limited_excitation.py collect
git commit -m "Add limited-excitation Gate 1 results"
git push origin study/cooperative-calibration
```

Smoke uses four Sobol weights plus the two anchors:54 cases, separate directory,
same characterization checks. Full run uses306 cases. Serial, explicit progress,
atomic case files, resume with unchanged inputs; no new dependencies. No need to
rerun earlier studies. Expected runtime is on the order of minutes, hardware dependent;
characterization prints after each plant/cap pair and evaluation after each weight.
Use `analyze` to regenerate only the full report. `collect` validates completeness,
rebuilds summaries and force-stages only the full result directory despite gitignore.
It does not stage smoke outputs or unrelated files.

Inputs/code hashes normalize CRLF for Windows. Changed inputs require a new run tag.
Artifacts: manifest, candidates, excitation means/diagnostics, raw cases, metrics,
requests, overlap, direct-transfer oracle, fixed anchors, summary, self-contained HTML.
