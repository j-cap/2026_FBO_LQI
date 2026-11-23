# Historical cost level versus gain dependence

Prepared 23 November 2026. Branch: `study/tracking-cost-level-shape`.
Configuration: `code/config/runs/tracking_cost_shape.json`.

## Purpose and stopping point

The completed information ablation found that the historical cost prior retains
the final-budget quality and failed-query benefits of the combined cost and
feasibility priors. A separate transferred feasibility-rate map adds little once
the cost prior is active. That finding leaves one attribution gap: the cost
prior changes both the GP's overall cost level and its gain-dependent landscape.

This final focused replay separates those effects. After its results, consolidate
the entire experimental chain before choosing further experiments. Do not extend
this study into sparse-history sweeps, additional plants or new optimizer variants.

## Three paired BO arms and a fixed benchmark

| Method | Historical log-cost mean | Historical feasibility prior |
|---|---|---|
| `no_extra_prior` | Zero | Neutral .5 before target updates |
| `context_level` | One source-derived constant per reference context | Neutral .5 before target updates |
| `full_cost_prior` | Frozen gain-dependent source GP mean | Neutral .5 before target updates |
| `catalogue` | No adaptive search; common six-query benchmark | No adaptive search |

Let the frozen historical mean for gains g and context c be mu_s(g,c), in
log(J) space. Define the context-level arm before any target query by

`mu_level(g,c) = (1/71) * sum over all candidate gains h of mu_s(h,c)`.

All 71 candidates in the fixed pool receive equal weight, including points whose
historical GP values are predictions in poorly observed or historically
ineligible regions. This is an average of the **frozen predicted log-cost means**,
not an average of raw observations, the best source cost, or a target-derived
offset. Exponentiating gives a geometric cost level. Each context's average
log-cost is preserved exactly, while variation over gains is removed.

The candidate pool is the existing Sobol-plus-anchor design, so its uniform
average is a finite-pool definition, not an integral over continuous gain space.
No weights, offsets or matching rules will be selected after seeing outcomes.

The unchanged target optimizer is called through its pooled-BO path with the
appropriate prior vector and constant .5 historical feasibility rates. Every BO
arm still learns cost and feasibility from its own target observations. All share
the identical six catalogue observations, in the same seeded order, before
adaptive acquisition. Best observed eligible gains remain the recommendations.

## What remains fixed

- Nine vehicle configurations and six speed/frequency contexts.
- The 71 gain candidates, nonlinear plant, fixed controller structure, cost,
  rollout eligibility rules and archived numerical-resolution decisions.
- Both leave-vehicle-out and leave-family-out exclusions.
- Twenty optimizer seeds, 1000 through 1019, with the same calibration-noise
  seed mapping and independent validation-noise repeats.
- Budgets 6, 12 and 24 **total target rollouts per vehicle across all six
  contexts**. Failed queries consume budget. The fixed catalogue spends six.
- GP kernel, noise levels, hyperparameter optimization policy, acquisition,
  target-feasibility learning, tie handling and success threshold.
- Full source histories, the source-derived catalogue, and the frozen source
  cost GP arrays. No source refitting or new nonlinear simulations.

The full study has 360 jobs. Each contains three 24-query BO arms and the
six-query catalogue: **28,080 requested cached target observations**. Historical
source counts remain 6,816 for leave-vehicle-out and 5,112 for leave-family-out.

## Reproducibility and leakage checks

The reference campaign is `code/results/tracking_information_v1`. Its source
arrays were originally copied from the completed budget replay. The new runner
copies those frozen arrays again without fitting, verifies signatures and source
exclusions, and records the source-file hashes. It checks the previous experiment
is complete and rejects changes to the full budget, seeds, protocols, GP settings,
tolerances or the declared context-level definition.

All three BO arms are rerun together under one runtime. Their controls are not
borrowed from previous result rows. A diagnostic compares the new no-extra-prior
and full-cost-prior traces with the earlier no-extra-prior and cost-only traces.
Numerical-library differences may change acquisition paths across machines;
exact old-trace agreement is an audit, not a result-selection rule. Unit tests
verify exact control equivalence under one runtime.

Workers receive calibration observations and permitted source arrays only.
Validation scoring and dense empirical-reference evaluation occur after complete
query traces are frozen. Checkpoints verify observation identities, unique query
accounting, source exclusions, best-observed incumbents, prior labels, and common
initial observations. Input and optimization-code fingerprints prevent mixing
different runs. Presentation code is fingerprinted separately.

`source_level_audit.csv` records each fold/context's preserved mean log-cost and
the standard deviation/range removed by the level projection. The existing
diagnostic, budget and information-ablation sources and results stay unchanged.

## Metrics and interpretation

Report the same budgeted metrics: contexts within 5% of the calibration-selected
finite-pool reference on validation noise, all-six-context success, policy
eligibility, validation J, failed/unresolved queries and actual rollout spend.
Retain unsuccessful or missing recommendations in the success denominator.

For each target/seed/budget, compute:

| Paired difference | Interpretation |
|---|---|
| `level_minus_none` | Effect of the context-specific historical cost level |
| `full_minus_level` | Additional effect of gain dependence, with context averages preserved |
| `full_minus_none` | Complete historical cost-prior effect |

Positive success differences are better; negative J or failed-query differences
are better. Each J contrast uses only pairs with eligible six-context policies
in both included arms and reports its own denominator. Success and failure
contrasts retain every matched trial. Report per-vehicle and per-context outcomes
alongside aggregate effects. All effects must be zero at the initial six-query
checkpoint. Optimizer seeds are repeated searches, not independent fleets, so
there is no pooled-row significance test.

If the full prior improves over the level prior, gain-dependent historical
information helps this fixed BO procedure. If the level prior reproduces the
benefit over no additional prior, the evidence supports useful centering rather
than a need for a detailed landscape. Mixed effects should be explained by
vehicle, success, cost and unsuccessful-query behavior. Similar averages alone
do not establish equivalence or prove that shape can never be useful.

Every arm still inherits historical costs and eligibility through the common
catalogue. Historical cost GPs were fitted to eligible source observations.
This is conditional attribution after that initialization, not a comparison with
an entirely history-free controller. It also does not establish that other
vehicles are necessary, that history is economical to collect, or that a
networked federated protocol outperforms transmitting a small catalogue.
The fixed gain pool, limited vehicle/noise diversity and boundary optima remain
limitations. Rollout eligibility is not a stability certificate.

## Run locally

From the repository root in Windows, using the existing environment:

```bat
git fetch origin
git switch study/tracking-cost-level-shape
git pull --ff-only origin study/tracking-cost-level-shape
conda activate paper_FBO_LQI
python -m unittest discover -s code/tests -p "test_tracking*.py" -v
python code/scripts/tracking_cost_shape.py smoke --workers 4
python code/scripts/tracking_cost_shape.py run --workers 4
start code\results\tracking_cost_shape_v1\report.html
```

Smoke runs use three targets, both protocols, two seeds and budgets 6/12: twelve
jobs. Smoke output is verification only and lives under a separate result tag.
Repeating `run` verifies and resumes completed checkpoints. `status` reports the
number complete; `analyze` regenerates the report from verified full checkpoints.

```bat
python code/scripts/tracking_cost_shape.py status
python code/scripts/tracking_cost_shape.py collect
git status --short
git commit -m "Add historical cost level and shape results"
git push origin study/tracking-cost-level-shape
```

`collect` rejects incomplete or smoke campaigns and stages only the new full
result directory after verification. Outputs include the HTML report, summary,
all budget/context outcomes and query observations, paired/per-vehicle effects,
source-level audit and previous-control audit.

## Consolidation after this experiment

Review the original yaw-rate task and the trajectory-tracking experiment chain
together. Organize the conclusions around:

1. Where controller tuning has practical headroom and whether gains depend mainly
   on the vehicle, trajectory context, or both.
2. How much a small transferred catalogue already provides, and what adaptive
   historical-prior BO adds under the same target budget.
3. Which historical information supports that remaining benefit: objective
   values, feasibility rates, context-specific cost level, or gain dependence.
4. Which claims about federated learning remain unsupported, the material
   limitations, and the scientific decision about continuing or changing task.

Separate supported findings, rejected explanations and unresolved questions.
Choose the research direction from that consolidated evidence before preparing
any further experimental campaign.
