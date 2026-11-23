# Budgeted contextual controller calibration

Prepared 23 November 2026. Branch: `study/tracking-budget-comparison`.
Configuration: `code/config/runs/tracking_budget.json`.

## Purpose

The completed trajectory-tracking diagnostic shows substantial reference dependence,
little additional headroom for vehicle personalization, and a strong three-gain
catalogue baseline. This experiment asks whether historical fleet evaluations reduce
target calibration effort, and what sequential BO adds beyond reusing that catalogue.
It is a **retrospective finite-pool replay**, using the archived nonlinear rollouts.
It introduces no new plants, reference trajectories, controller structure, cost,
feasibility thresholds, or gain-search bounds.

Source input is `code/results/tracking_diagnostic_v1/`. Original sources, signatures,
checkpoints, Windows reference assignments, and results remain unchanged. The new
experiment has its own source/configuration/input fingerprints and result tag.

## Target, source split, and information

A target is an entire vehicle. Each target must be calibrated across all six
controlled references. Run two exclusion protocols:

| Protocol | Historical sources | What is excluded |
|---|---|---|
| leave_vehicle_out | Eight other vehicles | Every calibration and validation observation of the target vehicle |
| leave_family_out | Six vehicles from the other two families | Every observation of the target and its two same-family variants |

All 9 vehicles are targets, with 20 optimizer seeds per protocol. This gives
360 replay jobs. Source priors are cached once per distinct source set: nine
vehicle-exclusion folds and three family-exclusion folds. Source data never
accumulate from earlier target jobs; job order cannot leak held-out targets.

Use the **full source calibration archive**: each source contributes
`6 contexts × 71 candidates × 2 noise repeats = 852` historical rollouts.
The two protocols therefore supply 6,816 or 5,112 historical rollouts per fold.
This is a declared data-rich historical scenario, not evidence of efficiency
with sparse source histories. Do not weaken the catalogue by withholding useful
history only from that comparator.

The target optimization oracle exposes a single archived calibration rollout for
each requested `(reference, gain)` pair. An optimizer seed selects noise seed
101 or 102 by parity; methods use identical counterfactual observations for the
same target action. An action cannot be queried twice. A query consumes **one
rollout**, including failed and numerically unresolved trials. The two calibration
noise realizations are not averaged into a free target observation.

Twenty optimizer seeds repeat acquisition/initial-design randomness using these
two fixed noise realizations. They are not twenty independent physical fleets or
new measurement-noise realizations. No population-level confidence claim follows
from treating all target/seed/context rows as independent samples.

## Budget and methods

Budgets are **6, 12, and 24 target rollouts per vehicle in total**, shared across
the six contexts. They are not budgets per context. Each 25-second rollout gives
2.5, 5, or 10 minutes of prescribed driving across these budgets, excluding setup,
resetting and other operational costs. These are research budgets, not a prescribed
or validated commissioning procedure.

| Method | Initialization | Additional target search | Historical surrogate |
|---|---|---|---|
| random | One point per context, original gains first and then gain-space maximin points | Balanced random search | None |
| independent_bo | Same initial actions as random | Contextual feasibility-weighted EI | None |
| catalogue | Source-selected gain for each frequency, verified at both speeds | Stops after six target rollouts | None |
| catalogue_warmstart_bo | Same six catalogue-verification actions | Same target-only BO as independent | None |
| pooled_bo | Same six catalogue-verification actions | Contextual feasibility-weighted EI | Fixed source GP mean and source feasibility rates |

The first six contexts are visited in a seed-dependent order. The independent
initialization spans gain space without inspecting target outcomes. Each catalogue
gain minimizes mean source calibration J over both speeds at its frequency,
subject to all permitted source/noise observations being eligible. Ties use
candidate ID within a fixed numerical tolerance. If no catalogue gain exists,
the original hand-tuned gain is the declared fallback, with no presumed feasibility.

The fixed catalogue spends six verification rollouts and then stops. At budget
allowances 12/24, its actual spend remains six and its recommendation stays fixed.
It never receives free confirmation of an unqueried gain on the target.

Comparing catalogue-warmstart BO with the fixed catalogue isolates additional
target search. Comparing pooled BO with catalogue-warmstart BO isolates the
historical surrogate beyond the same source-derived initialization. Random
search is a check on the usefulness of the target-only BO implementation.

## Surrogate and acquisition

Each action has five input coordinates: three normalized log10 gain multipliers,
normalized speed, and normalized log reference frequency. Independent BO is one
joint contextual GP per target, not six disconnected local optimizers. This allows
it to share its own target observations across contexts.

The objective surrogate models `log(J)` with a Matérn-5/2 kernel plus a constant
kernel. Bounds, log-noise SD .05, initial length scales, and optimization settings
are fixed in the configuration. Every four additional feasible observations,
bounded marginal-likelihood optimization refits the target kernel using target
observations only. Ineligible observations update feasibility but never become
finite good objective values. The original applied-steering-rate penalty remains
part of J; an oscillatory controller can remain within rollout state bounds and
still be an expensive objective observation.

Source preparation pools eligible costs at common reference/gain points, fits a
source GP to the log of their arithmetic mean cost, and retains empirical
feasibility rates from **all** source observations. Variance-of-the-mean and a fixed
log-noise term regularize this GP. Its predictions are a prior mean for pooled BO;
the target GP learns residual corrections using the same target-kernel machinery
as independent BO. This is a simple pooled historical prior, not a learned
similarity gate or RGPE ensemble.

Feasibility weighting uses an explicit kernel-smoothed beta-prior estimate:
`p = (strength*p0 + sum(kernel_weight*eligible)) / (strength + sum(kernel_weight))`.
Independent and warmstart BO use `p0=.5`; pooled BO uses source rates clipped to
`.05..95`. It is an acquisition heuristic, not a calibrated failure probability
or a safe-BO guarantee. When any context lacks a feasible target incumbent,
acquisition first searches those contexts using feasibility and uncertainty.
After every context has an incumbent, it maximizes feasibility-weighted expected
improvement across the six contexts. EI is computed in original J units under
the lognormal objective posterior.

Recommendations are the best **observed eligible target** gain in each context.
Neither an unqueried source gain nor a favorable validation outcome can enter an
incumbent. A context without an eligible observation has no recommendation.

## Frozen evaluation

Workers receive calibration data and source priors only. They finish the entire
maximum-budget trace before an independent evaluator reads validation noise
201/202. Validation never influences acquisition, hyperparameters, stopping,
incumbent selection, or source weights. The evaluator reads both validation
repeats of each frozen recommendation and requires both to be eligible.

The empirical reference for each target/context is the eligible gain minimizing
the target's **two-repeat calibration mean**, evaluated on validation noise. It
uses the complete finite pool only inside the evaluator. It is not a global
optimum, a learner-accessible target model, or a best-on-validation choice.

Primary metrics at each budget:

- Fraction of target contexts with an eligible recommendation within 5% of
  their empirical-reference validation J. Every missing/failed context remains
  in the denominator.
- Fraction of vehicle/seed trials meeting that condition in **all six contexts**.
- Fraction with a fully feasible six-context recommendation, failed and unresolved
  query counts, and actual target rollout spend.
- Mean validation J only when all six recommendations are eligible. Aggregate
  costs are unavailable if any trial is ineligible; conditional means retain their
  explicit denominators. Paired cost comparisons use jointly eligible trials and
  report their count; paired success/failure comparisons retain every trial.
- Restricted mean evaluations to first joint success, censored at the allowance.
  Unsuccessful runs are never removed. First-hit evaluation is retrospective;
  learners do not stop after observing a validation-based threshold.

Validation is an offline performance assessment outside the commissioning-query
budget for **every** method. The study does not claim that prospective evaluation
of every intermediate recommendation would be physically free. New reference
shapes, unseen speeds and new plant draws are outside this replay's evidence.

The decision is whether sequential fleet-assisted BO adds useful success or cost
improvement over catalogue reuse and catalogue-initialized local BO, while accounting
for failed target tests. If the catalogue already supplies the useful performance,
report that result rather than attributing its initialization benefit to a new BO
algorithm. The complete source histories and limited nine-plant pool must remain
explicit in any interpretation.

## Deterministic reference assignment

`code/src/tracking_context.py` supplies a version-2 reference-only assignment.
Distances within `1e-12*max(1,abs(min_distance))` are tied and resolved by reference
ID. This fixes near-equal 20/30 m/s distances for 25 m/s references across platforms.
`reference_assignment_v2.json` records old versus canonical assignments using
archived spectral metadata, without regenerating trajectories or using outcomes.
The old diagnostic runner and its fingerprints remain unchanged. This budget
replay evaluates the six controlled contexts; it does not silently rescore the
old held-out bank or claim new 25 m/s performance.

## Run locally

Use the existing `paper_FBO_LQI` environment. No additional packages are required.
From the repository root on Windows:

```bat
git fetch origin
git switch study/tracking-budget-comparison
git pull --ff-only origin study/tracking-budget-comparison
conda activate paper_FBO_LQI
python -m unittest discover -s code/tests -p "test_tracking*.py" -v
python code/scripts/tracking_budget.py smoke --workers 4
python code/scripts/tracking_budget.py run --workers 4
start code\results\tracking_budget_v1\report.html
```

The smoke run covers three target configurations spanning all families, both
exclusion protocols, two optimizer seeds, and budgets 6/12: 12 replay jobs. It
still uses the full permitted source sets. Its purpose is software verification.
The full run has 360 jobs and 36,720 requested target observations across all
methods. These are cached accesses, not newly simulated vehicle episodes.

Jobs checkpoint atomically. Repeating `run` validates and resumes existing jobs.
Changing optimization code, configuration or source input files requires a new
tag. Presentation code is fingerprinted separately so figures can be regenerated
without invalidating optimization traces. Frozen source priors are reused rather
than recomputed under a different numerical platform.

```bat
python code/scripts/tracking_budget.py status
python code/scripts/tracking_budget.py collect
git status --short
git commit -m "Add budgeted contextual BO replay results"
git push origin study/tracking-budget-comparison
```

`collect` verifies every job, its observation count, source exclusions, replayed
observations and incumbents before staging only `code/results/tracking_budget_v1`.
It rejects smoke collection and incomplete results. Reports include per-trial,
per-context and per-vehicle CSVs, paired attribution comparisons, a self-contained
HTML report, source-prior records and complete query traces. No networked
federation, raw-trajectory exchange, or privacy mechanism is implemented here.
