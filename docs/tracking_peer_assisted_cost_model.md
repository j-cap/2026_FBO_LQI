# Learned cost models under a common peer-gain verification rule

Proposed protocol, 24 November 2026. Branch: `plan/peer-assisted-cost-model`.
This document defines the next comparison. No new optimizer or physical
experiment is implemented by this plan.

## Question and hypotheses

Does a federated cost mean improve online controller calibration once direct
reuse of eligible peer gains is available to every method?

The main contrast is the same learned mean trained locally versus trained
federatively. Retain the current gain-sharing plus local-GP method as a third
control to measure the contribution of adding the mean model itself. All three
methods already learn a local GP: "additional learned cost model" here refers
specifically to the fixed-feature mean, not the existence of any cost learning.

## Definition of the peer-gain verification rule

At the beginning of round t, client i has a scheduled trajectory context c and
its own past tested gain/context pairs. Only outcomes completed before round t
may be used.

1. Each other client publishes its lowest-observed-cost **eligible** gain in
   context c, if it has one. Eligibility requires a completed, physically
   feasible and numerically resolved rollout. Each publication contains the
   context, candidate/gain vector, donor ID and observed donor cost.
2. Remove proposals whose gain has already been tested by client i in that
   context. Previous unsuccessful tests also count as tested.
3. Among remaining proposals, select the one with the lowest observed donor
   cost. Resolve ties by candidate ID and then donor ID, as in the current
   implementation. The ranking is not changed by a cost-model prediction,
   vehicle-family label or target incumbent value.
4. Run that gain on client i in context c. This consumes one full local rollout
   query, including when it fails or performs worse than the current controller.
5. Update client i's local observations. Only a locally eligible result may
   become a recommendation. The recommended gain remains the lowest-cost
   eligible controller actually tested on that client in that context.
6. If there is no untried peer incumbent for context c, select the next gain
   using the method's BO fallback.

This is a **peer-first** rule: an available peer proposal takes the query slot.
The learned model cannot veto, reorder or replace it in this experiment.
Each donor publishes only its current incumbent per context, not its entire
history or a ranked list of second-best gains. Peer costs are compared in the
same declared objective units. No cross-family normalization is introduced.

For example, a peer reports candidate g012 at 30 m/s and .3 Hz with J=.8. A
target that has not tested g012 in that context verifies it locally. If it
measures J=1.1 while its existing eligible gain has J=.9, it retains the latter.
The donor's .8 is not treated as a target observation. If verification fails,
the target still pays for the query and receives an ineligible label.

The factory gain is tested once at local query 1. Peer verification and BO
fallbacks begin at query 2. The rule is the existing `peer_candidate` selection
procedure in `code/src/tracking_online.py`, applied to the complete prior-round
fleet state of the method being evaluated.

## Methods

| Arm | Peer rule | Additional log-cost mean for BO | Interpretation |
|---|---|---|---|
| A: Peer gains + local GP | Common rule above | Zero | Current online gain-sharing reference |
| B: Peer gains + local cost model | Same rule | Same fixed-feature regressor fitted to own eligible observations | Architecture control |
| C: Peer gains + federated cost model | Same rule | Same regressor fitted through FedAvg to current fleet observations | Incremental effect of model sharing |

All BO fallbacks use a local residual GP and the frozen acquisition:

\[
 \theta=\arg\min_{\theta\;\mathrm{untried\ in}\;c}
 \left[\mu_i(\theta,c)-2\sigma_i(\theta,c)
 -\log\max\{10^{-6},\widehat p_{i,\mathrm{eligible}}(\theta,c)\}\right].
\]

The GP mean and standard deviation refer to log J. In B and C the GP learns a
local residual around the additional mean. Local eligibility learning, neutral
.5 prior, GP configuration, hyperparameter-refit policy and deterministic
acquisition ties remain unchanged. No eligible incumbent is required to run BO.

The regressor keeps 64 public random features plus intercept, ridge .02 and
the existing feature seed/length scales. B solves its local ridge problem;
C uses the existing weighted FedAvg settings (200 server iterations, one local
full-batch step, learning rate .2). Both train on all eligible observations
available to them, including peer-verification outcomes. C may therefore learn
from verification rounds even when no BO query occurs in that round.

The main comparison **C versus B** isolates sharing the mean beyond the same
architecture fitted locally. B versus A measures the additional local mean;
C versus A measures the complete shared-mean addition to the current practical
baseline. Keep the exact same-data ridge optimum as an offline FedAvg diagnostic;
another full centralized or scout arm is not needed for this focused pilot.

## Common rule and method-specific experience

Each method starts with empty histories and maintains its own fleet catalogue.
Only its own paid rollout queries enter its learners and publications. No
comparator may use trials generated by another comparator. Paired simulator
noise or computation caches do not create free learning observations.

The same peer procedure does not imply identical later peer gains or fallback
counts. Different BO choices can discover different incumbents, which then
become available to that method's peers. This feedback is part of the model's
total effect on online fleet calibration. Forcing a common externally supplied
catalogue would answer a different question about a fixed information source.

At each round all actions are frozen before outcomes are exposed. Training and
catalogue updates take effect in the next round. A proposed peer gain has no
target eligibility or target cost until the target verifies it.

## Fixed physical protocol and budget

Retain the one-test/LCB pilot settings in `tracking_online_lcb.json`: the same
nonlinear plants, controller and cost, six trajectory contexts, 71 candidates,
9 clients, 25-second episodes, integration checks, and homogeneous/heterogeneous
conditions. Do not add new dynamics, contexts or tuned acquisition coefficients
in this comparison.

Use seeds 1100/1101/1102 with the existing client/context programmes and
client/query measurement-noise pairing. Give every method 24 queries per client
and snapshots at fleet budgets 9/18/27/54/108/216. All three are concurrent and
finish in 24 calibration rounds. Failed and verification rollouts count equally.

The development pilot has 2 conditions x 3 seeds x 3 methods x 216 tests =
**3,888 charged rollout requests**. This is not independent-fleet confirmation.
Arm A should reproduce the existing gain-sharing action/outcome trace under
the unchanged settings, providing a control against an accidental protocol
change. Completed old experiments and signatures remain frozen.

## Evaluation fixed before execution

The primary contrast is C minus B in near-reference context success at the
final 216-query fleet budget, reported separately for each condition and seed.
Use the existing 5% empirical-reference threshold. Report complete eligible
client policies alongside it. Missing or ineligible recommendations are
unsuccessful; no client/context is dropped from those denominators.

Retain the full declared budget curves and prespecified 54/108 intermediate
comparisons. Also report failed/ineligible/unresolved tests, first context-
coverage times with censoring, and paired J differences with the number of
jointly eligible policies. Keep family/context breakdowns, including the SUV
and `s30_f0.1` cases, without selecting only favorable subgroups. Seed-level
results remain visible; repeated clients/contexts are not independent fleets.

Measure peer-publication payload in every arm and add model-vector payload
for C. Report whether any physical-query benefit justifies that extra channel;
no communication or privacy advantage is presumed. Actual gain/cost incumbent
pairs are shared by all three methods, so this is explicitly a hybrid of direct
gain sharing and model learning.

The dense reference and independent validation archive remain evaluator-only,
opened after complete query traces are frozen. Numerical failures stay charged
and cannot train cost models. No past archive or held-out cost table initializes
the live learners.

## Mechanism diagnostics and interpretation

Log the number and outcomes of peer-verification and BO-fallback queries, and
which BO discoveries become later peer incumbents. In the completed gain-sharing
reference, fallback BO accounts for 239/621 homogeneous and 242/621 heterogeneous
adaptive queries: roughly 38--39%. That is a real opportunity for model guidance;
the new arms need not have the same counts. Do not force extra BO queries solely
to create room for the learned model.

A direct comparison of mean observed costs only on each arm's fallback queries
is selection-dependent: the methods may encounter different fallback events.
The full paired campaign is the primary assessment of calibration performance.
If decision attribution is needed, add an offline diagnostic on one reference
arm's saved prior-round histories: fit both mean variants from those same
records and compare their suggested candidates at the same fallback events.
The frozen validation bank can score these suggestions after execution. Such a
diagnostic is a retrospective one-step comparison, not the performance of a new
complete online policy and not a same-noise counterfactual physical rollout.

Interpretation is fixed as follows:

| Observed result | Supported conclusion |
|---|---|
| C improves on B and A with comparable policy completion and a consistent seed pattern | Model sharing adds calibration value beyond direct gain reuse and the local mean |
| B improves on A, while C does not improve on B | The extra learned mean helps; federation adds no demonstrated benefit |
| C reduces paired J but leaves more incomplete policies | Conditional tracking refinement with a completion tradeoff |
| C helps one condition or family only | A condition-specific benefit, not universal fleet superiority |
| C does not improve on B or A | No added value demonstrated for this cost-model architecture under peer-first reuse |

If the main result is positive, the storyline is: a fleet shares tested
controller candidates for local verification, while learning a shared cost
landscape to guide the remaining experiments. If it is negative, consolidate
that direct reuse already captures the useful fleet information under the
current task and budget, before expanding the task or changing the mechanism.
