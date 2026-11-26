# FedBO: consolidation after the denser-domain online comparison

26 November 2026. Latest evidence: the
[read-only dense-gain audit](analysis/dense_gain/report.md).
The [25 November consolidation](fedbo_consolidation_2026_11_25.md) and
[24 November consolidation](fedbo_consolidation_2026_11_24.md) remain archived.

**Online cost-model sharing improves average near-reference tuning success
beyond the same peer-gain transfer mechanism in the tested homogeneous task.**
The exact shared/local method now has a verified, better-resolved search
reference. It leads A, B and matched E in aggregate, while programme dependence,
conditional cost comparisons and a small B coverage/failure tradeoff remain.

Retain this method as the statistical reference and start defining the
communication question. Keep the performance claim at its supported scope;
communication savings will not by themselves establish robust commissioning.

## The current problem and method

The controller tunes three feedback gains, K_y/K_psi/K_r, with kinematic
steering feedforward on the nonlinear Pacejka bicycle inherited from CFL-FF.
The physical cost is lateral RMSE divided by 0.25 m plus applied steering-rate
RMS divided by 10 deg/s. These are direct gains, rather than LQI Q/R weights.

One trial holds a known speed/frequency context and gain vector for a
25-second rollout. Six speeds and three frequencies give 18 contexts. Speeds
are drawn with replacement; learning occurs between trials. Nine homogeneous
clients receive 24 requests per method, including the factory test and all
failed tests. No archived cost labels initialize a learner.

The public action set is now 1,292 controllers, with physical bounds
[0.0005, 0.01, 0.01] to [0.15, 1, 1]. This retains all original 71 points,
densifies their box and extends K_y. The offline audit demonstrated useful
upper-domain headroom in K_y; no selected winner required K_psi/K_r above 1.
The reference remains finite-pool best-found, not a global optimum certificate.

All four methods use the same paid, exact-context peer-first rule. A is an
adaptive local GP, and B adds the original 65-coefficient locally learned
mean. D uses the exact contextual shared/local Gaussian cost model with
an evidence mixture over sharing strengths. E has the same marginal prior,
with independent cost models. A/B keep beta=2; matched D/E keep beta=0.5.
The eligibility predictor still uses only the target's own trial outcomes.

D now uses a whitened active-input dictionary and retains the exact
conditional GP remainder. Its basis grows with earlier eligible paid inputs;
it is no longer the fixed 257-coefficient spectral approximation. The
Gaussian information-message interface remains, with public dictionary IDs
also communicated. Direct checks of every recorded D/E BO prediction agree
with observation-space GP calculations to numerical precision.

## What the latest experiment establishes

| Method | Primary success | Complete eligible policies | Failed paid tests |
|---|---:|---:|---:|
| A | 20.76% | 13/27 | 69/648 |
| B | 21.81% | 15/27 | 60/648 |
| D | 29.07% | 14/27 | 61/648 |
| E | 20.91% | 8/27 | 70/648 |

The declared primary success averages encountered contexts within clients,
then clients and seeds equally. A complete eligible policy covers all
contexts that client encountered with validated feasible gains; it need
not meet the near-reference target everywhere.

D improves success over A/B/E by 8.31/7.26/8.16 percentage points. D/B is
positive in all three programmes; D/A and D/E are positive in two of three.
Its paired mean-J changes are -6.02%, -23.51% and -20.82% on 11, eight and
five common complete eligible policies, respectively. Those are conditional
comparisons with small, explicit denominators.

Seed 8101 exposes the distinction between objectives: D has nine complete
eligible policies and only three failed tests, but loses success to A/E.
Seed 8102 has large success gains, while its paired cost is above A on three
common complete policies. The positive-every-seed and practical-no-tradeoff
descriptive checks do not all pass. Uniform superiority is not established.

The difficult 0.1 Hz contexts remain weak: D succeeds in 2/126 encountered
contexts, and A/B/E in none. D makes its two useful exterior K_y discoveries
in rounds 23 and 24, leaving little opportunity for peer transfer. Most of
its advantage over A/B is at 0.6 Hz; it also improves 0.3 Hz over matched E.
Fifty-two of its 56 failed BO requests occur at 0.1 Hz. All seventeen missing
D context recommendations lack a local eligible trial; they do not result
from newly failing held-out validation of a selected incumbent.

The exact posterior favors fully shared cost information in all three
homogeneous programmes. This supports useful fleet information in this
condition. It does not settle the older heterogeneous D/E result, justify
removing personalization generally, or establish privacy.

## How the experiment chain fits together

The original LQI/yaw-rate task provided little transfer difficulty. The
trajectory-tracking work established context dependence and online
acquisition under a local experiment cap. Matching the peer rule isolated
the added value of sharing cost-model information beyond a controller catalogue.
Joint shared/local inference improved the earlier mean-only formulation,
and the spectral studies exposed representation and exploration tradeoffs.

The gain-domain audit then showed that the original 71-point reference was
coarse and excluded useful K_y values. The larger-domain follow-up retains
positive average D/E and A/B effects after correcting that reference and
removing fixed-rank kernel approximation. This strengthens the limited
average-information claim.

The old success percentages remain valid for their declared 71-point tasks.
They are not rescored against controllers outside their permitted action
sets. The new study changes the pool, representation, programme seeds and
number of validation repeats, so cross-study hit-rate changes cannot be
attributed solely to the wider gain range. The studies are an adaptively
developed experiment chain, not one pooled confirmation experiment.

## Readiness for communication work

The current exact method is a suitable reference for a scoped communication
study. Freeze its task, gain pool, prior, sharing mixture, bonus, budget and
peer rule. Preserve seed 8101 and the low-frequency failure/discovery cases
as development stress cases. More BO tuning is not required to define a
communication-preserving-inference question.

The supported control claim is an average homogeneous pilot advantage.
Fresh fixed-protocol confirmation is needed for broader robustness claims
and can be included in the eventual fixed communication evaluation. Do not
replace the declared 5% success metric after observing the outcomes; cost,
failure and complete-policy tradeoffs remain visible.

First determine the permitted information exchange. The learner's datum
is a scalar cost with a public gain/context ID, and the peer channel already
publishes incumbent gains and costs. Dense Gaussian messages are not yet
established as necessary, more private, or more efficient than compact
observation summaries. Where summaries are allowed, they are a required
communication reference.

Then ask how much communication can be reduced while preserving the exact
posterior and decision quality, followed by a matched online evaluation.
Use full D as the statistical reference and retain A/B/E and all paid-query
accounting. Posterior fidelity alone is not a new control-performance result,
and cheaper messages do not solve the remaining feasibility/discovery gap.
No communication protocol, compression or network result is implemented here.

## Reproducibility

The [dense-gain audit](analysis/dense_gain/verification.json) reconstructs
2,592 paid decisions and 13 saved tables, verifies 437 D/E BO predictions
with independent exact GP calculations, checks reference selection from
46,512 archived calibration observations and all 3,076 fresh validation
rollouts, and preserves every source-result fingerprint. No new rollout or
hyperparameter optimization is performed.

```bat
python code/scripts/analyze_tracking_dense_gain.py --workers 3
```

Earlier evidence remains available through the
[gain-domain audit](analysis/gain_domain/report.md),
[three-study six-speed consolidation](analysis/six_speed_consolidation/report.md),
[two-speed confirmation](analysis/spectral_confirmation/report.md) and
[25 November consolidation](fedbo_consolidation_2026_11_25.md).
