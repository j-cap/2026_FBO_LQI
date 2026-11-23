# Which historical information helps controller calibration?

Prepared 23 November 2026. Branch: `study/tracking-information-ablation`.
Configuration: `code/config/runs/tracking_information.json`.

## Question and controlled design

The completed budget replay found that pooled BO improves the remaining
near-reference calibration cases and incurs far fewer unsuccessful target tests
than catalogue-initialized local BO. That comparison changed two historical
channels together: the objective prior mean and the feasibility prior.

This focused experiment independently switches those channels on and off in a
**paired 2×2 factorial design**. The existing catalogue, its six initial target
tests, the optimizer, target-learning rules, hyperparameter policy, historical
archive, and target budget remain fixed.

| Arm | Historical objective prior mean | Historical feasibility prior |
|---|---|---|
| 00 / no_extra_prior | Off: zero prior in log(J) | Off: .5 before target updates |
| 10 / cost_only | Frozen historical GP mean | Off: .5 before target updates |
| 01 / feasibility_only | Off: zero prior in log(J) | Frozen historical eligibility rates |
| 11 / both_priors | Frozen historical GP mean | Frozen historical eligibility rates |

All four arms continue learning objective values and feasibility from target
observations. Switching off a historical channel does not switch off the target
model. The fixed catalogue is retained as a separate six-query benchmark.

The questions are:

1. Does the cost prior improve recommendation quality or allocate target tests
   more effectively, conditional on whether the feasibility prior is available?
2. Does the feasibility prior reduce unsuccessful tests, conditional on whether
   the cost prior is available? Does avoiding poor regions also improve quality?
3. Does either channel already provide the useful benefit, or does their
   combination improve the quality–experiment-failure tradeoff further?

History size is fixed at the current full archive. Sparse histories are a
subsequent experiment, so information type and information amount are not changed
together in this test.

## What is held fixed

- Same nine deterministic vehicle configurations and six controlled references.
- Same 71 gain candidates, controller, cost, and rollout eligibility criteria.
- Both leave-vehicle-out and leave-family-out source exclusions.
- Same 20 optimizer seeds and calibration noise seed selected by parity.
- Budgets 6, 12 and 24 individual target rollouts **per vehicle across all six
  contexts**, not per context. Failed attempts consume the same budget as before.
- Same source-derived catalogue and identical six initial observations, in the
  same order, for every arm of each target/protocol/seed combination.
- Same target GP kernel, noise settings, marginal-likelihood refitting policy,
  acquisition formula, target-feasibility updates, and incumbent-selection rule.
- Same independent validation noise and calibration-selected empirical reference.

The runner rejects changes to the GP configuration, full budgets, seed list,
exclusion protocols and tolerances relative to the completed budget comparison.
The only optimization treatment is the pair of historical prior arrays.

This is again a **retrospective finite-pool replay**. There are no new nonlinear
vehicle simulations. The full study has 360 replay jobs, each with four 24-query
BO arms and the six-query catalogue: 36,720 requested cached observations.

## Implementation and isolation

The unchanged `src/tracking_budget.py::run_method` runs every factorial arm via
its existing pooled-BO path. The new wrapper passes either the frozen source
array or its prescribed neutral array for each historical channel. The catalogue
is passed unchanged in all cases. Disabled arrays are not read by the wrapper.

Consequently, under one numerical runtime:

- Arm 00 exactly reproduces catalogue-initialized local BO.
- Arm 11 exactly reproduces pooled BO.
- Arms 10 and 01 use that same optimizer with one historical channel neutralized.

Source arrays are copied from `tracking_budget_v1/sources/` without refitting.
The old source signature, source-file hash, excluded vehicle/family membership,
and all new experiment fingerprints are recorded. Source histories remain
6,816 or 5,112 calibration rollouts per fold. No target data or previous target
job is added to a source prior.

Every arm is rerun under the current runtime. The older control trajectories are
not mixed with newly generated single-channel arms. A separate control audit
compares arms 00/11 with the corresponding previous traces; numerical-library
changes can affect acquisition paths across runtimes. Exact old-trace agreement
is diagnostic, not a criterion for choosing which new results to report.

Workers receive calibration observations and source priors only. They freeze
complete query histories before validation scoring. Neither validation costs nor
the dense target references enter acquisition, fitting, stopping or source weights.
Recommendations must remain best observed eligible target gains.

The old diagnostic and budget-replay sources/results stay unchanged. The new
runner fingerprints the frozen optimizer, new wrapper, original input archive,
reference source files, reference configuration and reference control traces.
Presentation code is fingerprinted separately from optimization checkpoints.

## Metrics and attribution

Use the same per-budget outcomes as the preceding campaign: fraction of contexts
within 5% of the empirical reference, all-six-context success, final policy
eligibility, mean validation J, failed target queries, unresolved queries, and
actual rollout spend. Missing or failed recommendations remain in the success
denominator. The fixed catalogue spends only six queries at every allowance.

For a metric y, report these matched differences for each target/seed:

- Cost effect with feasibility off: `y10 - y00`.
- Cost effect with feasibility on: `y11 - y01`.
- Feasibility effect with cost off: `y01 - y00`.
- Feasibility effect with cost on: `y11 - y10`.
- Cost main effect: average of the two cost effects.
- Feasibility main effect: average of the two feasibility effects.
- Interaction: `y11 - y10 - y01 + y00`.

Higher success effects are better. Lower J and failed-query effects are better.
At the initial six-query checkpoint all contrasts must be zero: no adaptive
query has occurred and all arms have identical observations and recommendations.

Conditional effects are the primary attribution evidence. Interaction is
descriptive; its sign alone does not establish physical or algorithmic synergy,
particularly for bounded success fractions and failure counts with a zero floor.

Success and failure contrasts retain every matched trial. A J contrast is
available only when all arms entering that particular contrast have eligible
six-context recommendations. Its own eligible-trial count is reported; one
failed arm must not remove a trial from unrelated contrasts. Per-vehicle effects
are saved alongside aggregate effects. No pooled-row significance claim is made.

Interpret possible outcomes directly: a feasibility-only arm matching joint
performance would motivate a simpler feasibility-informed search; a cost-only
arm matching it would motivate keeping the objective prior alone. If conditional
effects show useful contributions from both, retain both channels. Judge these
tradeoffs across quality, failed tests and individual vehicles rather than
declaring equivalence from a small mean-J difference alone.

## Interpretation limits

All arms inherit historical cost and eligibility information through their
**common catalogue initialization**. The historical cost GP itself was fitted to
eligible source objectives. This experiment therefore isolates the two explicit
algorithmic channels after that shared initialization. It cannot show that useful
cost data could be obtained without any historical eligibility information.

The cost toggle removes the entire historical mean, including both its overall
level and gain-dependent variation. It does not distinguish those contributions.
Likewise, the feasibility toggle tests the complete historical rate map.

The histories are deliberately data-rich. The nine synthetic vehicles, two target
calibration-noise realizations and optimizer seeds are not independent population
samples. The finite gain pool and previously observed search-boundary limitations
remain. Eligibility refers to developmental rollout bounds, not a stability
certificate or a calibrated safe-BO guarantee. This test does not implement a
networked federated protocol, privacy protection, or new-reference validation.

## Run locally

Use the existing `paper_FBO_LQI` environment. From the repository root on Windows:

```bat
git fetch origin
git switch study/tracking-information-ablation
git pull --ff-only origin study/tracking-information-ablation
conda activate paper_FBO_LQI
python -m unittest discover -s code/tests -p "test_tracking*.py" -v
python code/scripts/tracking_information.py smoke --workers 4
python code/scripts/tracking_information.py run --workers 4
start code\results\tracking_information_v1\report.html
```

The smoke run uses three target configurations, both protocols, two optimizer
seeds and budgets 6/12: 12 jobs. It verifies software and cannot answer the
scientific questions. A full-budget canary can verify the 24-query path separately.

Repeating `run` validates and resumes completed checkpoints. Reports include
budget and context outcomes, every query observation, paired conditional/main
effects and interactions, per-vehicle effects, and the previous-control audit.

```bat
python code/scripts/tracking_information.py status
python code/scripts/tracking_information.py collect
git status --short
git commit -m "Add historical information attribution results"
git push origin study/tracking-information-ablation
```

`collect` rejects smoke or incomplete runs and verifies source exclusions,
observation counts, recorded observations, incumbents, channel labels and common
initialization before staging only `code/results/tracking_information_v1`.
