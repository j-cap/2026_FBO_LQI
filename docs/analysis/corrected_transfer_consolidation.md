# Corrected evidence: why LQI weight transfer succeeds

19 November 2026. Frozen experiment results: v4 `49b623a`, v5 `9ed48cb`, v6
`60b7aaa340265cb875e901fe9bc26d98e43e3067`. This note consolidates those results and
adds read-only diagnostics from saved models and reference-screening candidates.
No new optimization or plant simulation was performed for the explanatory analysis.
Reproduce its CSV/JSON tables from the repository root:

```bat
python code/scripts/consolidate_transfer_evidence.py
```

## What the corrected experiments establish

The original actuator evaluator's anti-windup term had the wrong sign for the
implemented negative integral feedback. In the paired v4 replay, sign reversal and
tracking back-calculation both eliminated all ten recorded infeasible episodes;
75/75 controller/bank combinations became feasible. The median cost improvement
among previously feasible banks was small. Rare severe failures were the principal
impact. Earlier failure patterns cannot be used as evidence of a physical limit to
transferability. The older ideal-steering runners contain the same sign and need a
separate impact audit before their numerical claims can be reused.

V 5 fixed tracking back-calculation at T_aw=.2 s, preserving the physical vehicles
and identification. All 30 rebuilt references passed feasibility and episode-level
integration checks. Both Sobol and lower-penalty BO met the 5% reference-gap criterion
in all 18 source runs even at budget 25. The 2700 source queries were all feasible.

V 6 transferred weights selected solely from each donor's calibration history to
other vehicles, with the target's own history excluded. Every target used all five
other physical vehicles as donors; no best donor was selected on target outcomes.
Gains were recomputed from the identified target model and its known actuator time
constant. Thirty frozen reference controllers were re-evaluated on the same new
80-episode banks used for transferred weights.

| Source method | Source budget | Within 5% | Within 1% | Median gap | Worst gap |
|---|---:|---:|---:|---:|---:|
| Sobol | 25 | 450/450 | 129/450 | 1.184% | 3.932% |
| Sobol | 75 | 450/450 | 145/450 | 1.106% | 2.359% |
| BO | 25 | 450/450 | 287/450 | 0.712% | 3.627% |
| BO | 75 | 450/450 | 337/450 | 0.044% | 1.816% |

All 537 distinct target/tau/weight evaluations, comprising 42960 episode records,
passed feasibility and 5-versus-10-substep integration checks. The 1800 transfer rows
retain donor provenance but contain repeated weights and strongly correlated data.
They are not 1800 independent systems: this is a six-vehicle developmental pilot
from two fleet realizations. The references are validated finite-search candidates,
not proven global optima. Negative gaps are retained.

The retrospectively identified common vector passed all 30 target/tau cases and
had a worst gap of 0.101%. This is exploratory, not independent source-only selection.

**Supported claim:** within this model family, common calibration task, known
actuator range tau=.1–.4 s and 5% tolerance, historical weights plus target-specific
model-based gain synthesis achieved the criterion without target weight optimization.

**Unsupported claims:** target BO is unnecessary in general; a universal controller
has been found; pooled historical observations form a valid target GP; real vehicles
need no target verification; the current finding alone constitutes a strong new paper.

## 1. The transferred object is a design preference, not the controller

Let theta denote the physical/model parameters and xi the log weight vector.
The actual controller is synthesized as

    K_theta(xi) = LQI(Ahat_theta, Bhat_theta, tau_theta, Q(xi), R).

The tested transfer cost is J_target(K_target(xi_source)). It is NOT
J_target(K_source(xi_source)). Identification, actuator-model augmentation, Riccati
solution, reference prefilter computation and gain-scaled anti-windup already supply
substantial target adaptation. The weight vector specifies preferences between
state penalties; preserving preferences across vehicles does not imply preserving
the feedback law.

For the same common xi and stored identified nominal model (fleet 355238):

| Actuator tau [s] | K_beta | K_r | K_delta | K_eta | Reference gain |
|---|---:|---:|---:|---:|---:|
| .10 | .771 | 4.599 | 3.473 | 43.643 | 5.475 |
| .20 | 1.383 | 8.027 | 7.070 | 75.564 | 9.609 |
| .40 | 2.508 | 13.776 | 13.784 | 127.618 | 16.675 |

Thus a fourfold actuator-time-constant change induces large gain changes while xi
stays fixed. The model-based redesign is an observed mechanism. Its quantitative
causal contribution relative to leaving gains unchanged has not yet been measured.
It does not cancel the physical actuator lag or guarantee equal closed-loop behavior.
Full table: shared_weights_gains.csv. Models are from the frozen v1 source artifacts,
using the unchanged identification/synthesis definitions retained in subsequent runs.

## 2. There is a shared region of acceptable weights

Using only the 256 common `new_sobol` candidates in each v5 reference screening,
compare the three families within each fleet realization at a fixed tau. There are
30 pairwise comparisons: two fleets × five taus × three family pairs.

- All 256 candidates are calibration-feasible in each of the 30 grids.
- Spearman rank correlation: minimum .9881, median .9937.
- Each grid has 38–79 candidates (14.8–30.9%) within 5% of its OWN best grid cost.
- The Jaccard overlap of those 5% sets has minimum .7722 and median .8637.
- At 1%, only 2–12 candidates per grid qualify.

These diagnostics explain why even modest source searches often find transferable
weights: acceptable candidates occupy a substantial fraction of the sampled domain,
and these regions overlap strongly between families. They also explain why BO can
improve fine precision without being necessary to reach the 5% criterion.

These are finite-grid observations, not a volume estimate, a proof of landscape
identity, or a correlation result across different tau values. The near-optimal sets
use each grid's own minimum rather than the validation-selected reference. Large
whole-domain rank correlation alone would not establish near-optimum transfer;
the set-overlap calculation and actual v6 transfers provide the additional evidence.
Tables: aligned_landscapes.csv and grid_near_optimal_counts.csv.

## 3. The task gives the vehicles similar tuning incentives

The fleet varies vehicle parameters, but it retains a common model structure,
controller architecture, sample time, state normalization, maneuver family and
performance definition. Payload and reduced tire stiffness are modeled; they do not
by themselves imply different optimal design preferences. Within-family parameter
spread is only ±2.5%. Actuator tau is known and explicitly included in synthesis.

The outer calibration objective is mean episode yaw-rate tracking RMSE, subject to
hard sideslip, steering and steering-rate constraints. It has no continuous penalty
for control effort, command variation, comfort, or sideslip below the constraint.
R=1 penalizes input inside the LQI DESIGN problem; it is not a separate control-effort
term in the OUTER simulation objective. Increasing state penalties can therefore
change that design tradeoff while the outer score continues to reward tracking.

The common vector is approximately xi=(-2.197,2.972,1.882): small normalized beta
weight, large yaw weight near its upper search bound, and strong integral weight.
This is consistent with a common preference for tight yaw tracking. It is a plausible
explanation, not proof of coordinate-wise monotonicity or a boundary optimum.
Do not interpret raw Q entries across differently scaled state coordinates directly.

Among the 537 evaluated controller settings, peak actual sideslip was 4.061deg
(limit 6deg), peak actual wheel angle 12.364deg (limit 20deg), and peak actual steering
rate 4.521rad/s (limit 5.236rad/s). These are substantial margins in sideslip and
actual angle, with a smaller rate margin. Command saturation still occurred in
20154/42960 episodes, and reached 15.97% of samples in one episode. Thus it would be
incorrect to say saturation or nonlinear effects are absent. Actual wheel-angle
margin does not imply that the commanded steering signal never reaches its clamp.
These margins concern tested transferred/reference controllers, not the whole domain.

## 4. The observation model also makes this a favorable transfer setting

Code inspection shows that state feedback uses the simulated states beta, r and
actual steering delta directly. The .01rad/s yaw measurement noise enters the
integrator update and the reported score, but not the instantaneous proportional
state-feedback term. Process noise is zero. There is no state observer or model of
state-estimation uncertainty in this evaluator. This limits the physical scope of
our claims and must be stated when describing the simulation.

The objective includes measurement noise: y=r+v. Since the current noise sample is
independent of the current true state/error in this update ordering,

    E[(y-r_ref)^2] = E[(r-r_ref)^2] + sigma_v^2.

Here sigma_v=.01rad/s. This creates a noise contribution shared across controllers
and can compress relative differences in the measured tracking score. The identity
is for expected MSE; the reported quantity is a MEAN OF EPISODE RMSEs, so subtracting
sigma_v squared from the squared reported mean is not an exact de-noising procedure.
Noise also affects future dynamics through the integrator. Its quantitative
contribution is not available from the saved aggregate RMSEs. It is an unquantified
contributor, not an established explanation for all small transfer gaps.

## What to do next, without expanding the experiment indefinitely

First discuss the consolidated result and changed claim with Andreas/Markus.
The current experiment answers the missing-baseline question convincingly within
its scope. It does not yet explain general transferability boundaries.

If a short mechanistic follow-up is needed, prioritize two paired diagnostics:

1. For one fixed weight vector, compare target-specific gain recomputation against
   retaining the source gains, with the same physical targets and episodes. This
   isolates the benefit of model-based redesign. Preserve controller coordinates,
   prefilter and anti-windup conventions; count all failures rather than dropping them.
2. Log true-state tracking error alongside measured error on the SAME rollouts,
   preserving the original measurement noise and controller. This quantifies the
   contribution of measurement noise without changing the experiment's dynamics.

The existing matched-grid analysis already addresses landscape similarity; another
large BO campaign is not needed for that purpose. Establish practical motivation
and independent-system confirmation before claiming a general theory or a new
transfer method. Do not tighten the primary tolerance or add physical effects solely
to create a BO advantage.

A possible research question, if pursued, is: when does model-based controller
redesign make calibration preferences transferable, and which departures from its
assumptions actually require retuning? The present evidence supplies a positive
transfer regime; it does not yet establish the boundary or novelty of that question.
