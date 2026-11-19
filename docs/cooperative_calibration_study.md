# Cooperative controller calibration: feasibility-study protocol

Version: 0.2, 19 November 2026.
Status: study design; not a frozen confirmation protocol or executable experiment.
Base: analysis/transfer-explanation at 84fad6e49b53e01a31245243ca3e4c8c5132442f.
Preserve all previous experiments and their limitations.

## Research question and decision

Can a heterogeneous fleet cooperatively discover common LQI design weights that
meet specified performance requirements, using fewer total closed-loop calibration
experiments than calibrating one vehicle and transferring its weights, while
retaining vehicle models and trajectories locally?

First establish an experimental-allocation benefit. Develop a new federated
surrogate method only if that benefit exists and a simpler scalar-sharing method
does not already solve the problem adequately.

Hypothesis H1: different vehicles exclude different candidate weights, but an
acceptable common region exists; selecting vehicle-weight experiments jointly
can reduce the cost of discovering and validating a point in that region.
Null explanation: local model-based gain redesign already aligns acceptable
regions sufficiently that one-vehicle calibration is adequate.

The existing easy-transfer experiment is a required negative control.
A negative result is a valid outcome. Do not alter requirements or physics after
seeing a comparison simply to obtain a cooperative advantage.

## Scope and mathematical target

Each vehicle i retains its identified model M_i and computes K_i(theta) locally.
The common variable theta is the existing three-dimensional log-weight
parameterization. Controller gains and reference prefilters remain vehicle-specific.
Do not average local optima or silently treat vehicle-dependent costs as one function.

For predefined maneuvers m, define measurable performance J_im(theta) and fixed
requirements b_m (vehicle-specific requirements only with an engineering rationale).
The desired set is:

    A = intersection over i,m of {theta: J_im(theta) <= b_m and constraints hold}.

First target the finite participating fleet and the specified maneuver distribution.
Generalization to unseen vehicles is a separate claim requiring independent fleets.
An empty sampled intersection is not proof that the continuous intersection is empty.

Primary performance: mean episode yaw-rate RMSE measured using the existing sensor
convention, checked separately for each maneuver so averages cannot hide failure.
Secondary diagnostic: true-state RMSE on the same trajectories.
Preserve existing sideslip, steering-angle and steering-rate constraints.
Finite test-horizon feasibility is not a stability guarantee.

Do not import the retrospective common weights, dense references, or historical
optimized samples into the from-scratch algorithms. They are diagnostic comparators.

## Provisional exploratory requirement: selected 19 November 2026

The user selected 0.92 deg/s for the next exploratory study after reviewing Stage A.
This is a benchmark assumption selected using development evidence, not an external
engineering specification or a preregistered threshold for the existing data.

For every vehicle, bandwidth and maneuver separately, require:

    E_noise[episode measured yaw-rate RMSE] <= 0.92 deg/s
                                           = 0.016057029118347832 rad/s.

The expectation is over repeated episodes under the fixed noise/task convention.
This is mean episode RMSE, not the square root of mean pooled squared error.
Do not average maneuvers, vehicles or bandwidths to obtain acceptance.
Continue to enforce existing physical constraints and numerical validity checks.
Observed constraint satisfaction is not a probabilistic safety guarantee.

Machine-readable specification: code/config/runs/cooperative_acceptance.json.
The same threshold applies to both current maneuvers. True-state error remains a
secondary diagnostic; it does not replace measured error for this benchmark.

The selected threshold is provisional for exploration and must remain unchanged
within the next validation/comparison. Any later revision creates a new protocol
version and is reported transparently. Existing threshold sensitivity results remain
retrospective. No claim of industrial relevance follows from choosing this value.

Immediate next step: validate the FROZEN common-weight and reference controllers
on fresh noise episodes across all six vehicles, five bandwidths and two maneuvers.
No retuning or new optimization in that validation. The retrospective common
weights are a feasibility probe, not free prior information for from-scratch methods.

A sample mean below 0.92 deg/s is descriptive acceptance only. Before running fresh
validation, fix its sample size and uncertainty rule (including simultaneous scope),
or a sequentially valid procedure. Do not repeatedly sample until an ordinary
confidence interval passes. An interval spanning the threshold means unresolved,
not accepted and not established infeasibility. The fresh-validation runner is now implemented as
code/scripts/cooperative_validation.py. It uses 1,000 fresh episodes per cell and
approximate two-sided Bonferroni Student-t intervals across 120 means, with no
adaptive extension. See docs/cooperative_validation.md for the frozen execution
and interpretation rules.

## Stage A: opportunity audit using corrected evidence

No new BO and no physical changes in this stage.

Inputs:
- corrected v5 matched landscape evaluations;
- v6 direct-transfer records;
- v7 gain-redesign / measurement diagnostics.
Use actual stored weight vectors to align grids, not row numbers.
Retain infeasible, missing and numerically unresolved evaluations explicitly.
Only evaluate a joint region where the relevant vehicles share evaluated candidates.

Separate within-family, cross-family and bandwidth comparisons.
The same six physical vehicles evaluated at five bandwidths are not 30 independent
vehicle draws. Do not treat repeated optimizers or noise episodes as independent fleets.
The bandwidth range is an illustrative stress range, not an empirical population model.

Required outputs:
1. Absolute measured and true RMSE distributions by vehicle and maneuver.
2. For a full reported threshold range, fraction of sampled weights acceptable
   on each vehicle and jointly on the fleet.
3. Conditional transfer success: among candidates acceptable on donor i, fraction
   acceptable on all other vehicles. This is a landscape diagnostic, not a BO policy.
4. For each source-calibration run, actual donor-selected candidate acceptance
   across the fleet, without selecting the donor retrospectively.
5. Which vehicles or maneuvers exclude otherwise promising candidates.
6. Fraction of records excluded for numerical inconsistency, separated from
   physical constraint failures.
7. A short conclusion: broad shared region, narrow shared region, no sampled
   shared region, or evidence inadequate.

Report dense-reference and retrospectively selected common-weight results separately.
Source-selected validation candidates must not use the final test bank for selection.
Where stored maneuver-level summaries are absent, report the limitation rather
than infer them from a combined average.

Stage A decision:
- Broad common region and reliable direct transfer: keep as negative control;
  no new federated algorithm justified by these data.
- Shared region with substantial donor-dependent rejection: proceed to Stage B
  after requirements are justified.
- No sampled shared region: investigate whether common weights are an appropriate
  product requirement before adding optimization machinery.
- Numerical/measurement uncertainty dominates: repair evidence before comparing policies.

If the current setting gives no opportunity, allow at most ONE separately documented
extension motivated by an engineering use case. Its mechanism, ranges and requirements
must be fixed before comparing algorithms. Do not sweep physics until BO wins.

## Stage B: bounded allocation pilot

Development fleet: the existing six vehicles, considered development data.
New noise banks must be distinct from earlier calibration and test banks.
Use three optimizer seeds for debugging only; these do not establish population evidence.
Keep the original controller architecture and corrected anti-windup implementation.
Local identification and simulation access must be equal across all methods.

A query is a chosen (vehicle, weights) pair, evaluated on the same fixed number of
episodes per maneuver for all methods. Count individual rollouts and simulated
or physical duration as well as queries.

Initial proposed development cap: 150 vehicle-weight calibration queries per method
and seed, matching six independent budgets of 25. This is a resource cap, not a claim
that 150 queries are necessary. Report anytime results from the first query.
Validation is additional, separately itemized, and included in the total cost.
Freeze episode counts and a validation cap in the executable protocol before running.

Compare:
B0. Calibrate one vehicle, transfer, and validate on the fleet. Run each of the six
    donor choices separately; each is a separate deployment strategy, not six donors
    acquired for free. Report all donors, the average, and the retrospective best as
    an explicitly oracle-selected diagnostic.
B1. Sequential repair: start as B0, then calibrate a failing vehicle and validate the
    new proposal. Keep previous information; a repair can break earlier acceptance.
    Use a deterministic failure-priority rule fixed before runs.
B2. Simple coordinated candidate testing: common Sobol sequence, deterministic rotating
    vehicle order, reject a candidate after a decisive failure and validate survivors.
    Use the same evidence rule and early rejection for all relevant baselines.
B3. Centralized selective-evaluation BO over vehicle-indexed scalar observations,
    using separate vehicle responses and a shared-feasibility acquisition objective.
    This is a literature-based prototype, not claimed methodological novelty.
B4. Independent calibration of all vehicles: reference for the cost of personalization;
    its output is different and it is not the primary shared-weight competitor.

B3 may receive weight vectors, scalar performance summaries and feasibility information.
Models and trajectories remain local. This already satisfies the stated information
restriction and must precede any more elaborate surrogate-exchange design.
Specify the chosen published acquisition and adaptation before implementation;
do not label a new heuristic "SOTA" without matching the original formulation.

Failure to find common weights at the cap is a failure, not a missing observation.
Do not report only successful runs or condition cost comparisons on eventual success.
No fleet-average objective can substitute for individual acceptance.

## Acceptance, validation and statistical safeguards

Separate optimization observations, online validation, and a sealed final audit.
Candidates rejected by online validation can guide further optimization.
The sealed audit is opened only for the final policy output and cannot trigger repair.

Every method uses the same acceptance rule, validation bank size and accounting.
Repeated online decisions require a sequentially valid rule or a finite, preregistered
number of checks with an appropriate multiplicity correction. A GP posterior
probability alone is not an empirical performance guarantee.

For an initial descriptive pilot, report observed acceptance and uncertainty without
claiming a certified failure probability. Do not promise a 95% fleet-wide guarantee
from a small per-vehicle test bank. Specify that guarantee and its sample size
separately if the application actually requires it.

Repeat the established coarse/fine integration check for the implemented controllers.
Numerically unresolved outcomes cannot count as accepted or as clean physical failures.
If the protocol rejects an unstable linearized design, apply that filter equally and
record it; otherwise disclose linear radius separately from horizon feasibility.

All methods start with the same prior information. Count any new fleet-wide
initialization and any historical data acquisition used by a method.
Diagnostic dense grids and fine-resolution numerical checks are research overhead:
report separately and give no policy access. Count normal validation rollouts in
deployment cost. Also report validation-free search cost to expose where savings arise.

## Stage C: independent confirmation, only after the pilot

Freeze requirements, algorithms, hyperparameters, source-choice rule and budgets.
Use at least ten newly generated fleet realizations from the justified distribution,
with three optimizer seeds per fleet averaged within fleet for inference.
Do not treat the resulting 30 optimizer runs as 30 independent fleets.
A final sample-size decision should use development variability and desired precision;
ten fleets are an initial planning minimum, not a power guarantee.

Use paired fleet-level comparisons and report:
- acceptance probability versus total rollout budget;
- total cost to acceptance, retaining failures as censored observations;
- fleet-level paired differences and uncertainty;
- per-vehicle performance and worst normalized requirement violation;
- false acceptance on the sealed audit;
- communication and wall-clock time separately from experiment count.

Provisional continuation criterion, to freeze before confirmation:
at least 20% lower median total rollout cost than both one-source transfer and
sequential repair, with no meaningful deterioration in final acceptance.
Use a 5-percentage-point acceptance noninferiority margin as a planning proposal,
subject to the application's actual tolerance. Do not claim noninferiority unless
the confidence interval supports it. Report all methods at the same total budgets.
The 20% margin is a project effort/value criterion, not a literature-derived standard.

Only if B3 shows useful savings should a federated surrogate implementation be added.
It must then compete with B3 under identical permissible messages, and with an
appropriate published collaborative BO method. Local raw-data retention is not
formal differential privacy. Privacy guarantees require a threat model and analysis.

## Deliverables and reproducibility

Stage A: opportunity report, aligned candidate tables, transfer matrix and threshold curves.
Stage B: immutable protocol, resumable runner, per-query ledger, scalar observations,
validation decisions, failures and paired budget curves.
Stage C: frozen confirmation report and a concise illustrated HTML explanation.

Runner requirements:
- preflight, smoke, run, analyze and collect commands;
- periodic flushed progress, atomic per-query checkpoints, safe resume;
- input/code hashes; refuse incompatible resume;
- collect force-adds only complete study outputs beneath the ignored results directory;
- Windows instructions always include commit and push, without git diff --cached --stat.
No new experiment command is supplied with this design-only version.

## Closest methods to resolve before coding Stage B

- Dai et al., Federated Bayesian Optimization via Thompson Sampling, NeurIPS 2020:
  https://proceedings.neurips.cc/paper/2020/hash/6dfe08eda761bd321f8a9b239f6f4ec3-Abstract.html
- Dai et al., Differentially Private Federated BO with Distributed Exploration, 2021:
  https://arxiv.org/abs/2110.14153
- Chen et al., Multi-Agent Collaborative BO via Constrained Gaussian Processes:
  https://doi.org/10.1080/00401706.2024.2365732
- Pelamatti et al., Coupling and selecting constraints in BO under uncertainties:
  https://doi.org/10.1007/s11081-023-09807-x
- Toscano-Palmerin and Frazier, BO with Expensive Integrands:
  https://doi.org/10.1137/19M1303125
- Chang et al., Collaborative Contextual BO, April 2026 preprint:
  https://arxiv.org/abs/2604.18912
- Savaia et al., Experimental automatic calibration of a semi-active suspension controller:
  https://doi.org/10.1016/j.conengprac.2021.104826
- Zhao et al., Efficient Learning of Vehicle Controller Parameters via Multi-Fidelity BO:
  https://doi.org/10.1109/IV64158.2025.11097398
