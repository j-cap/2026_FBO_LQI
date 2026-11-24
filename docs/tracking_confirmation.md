# Frozen D256 confirmation on fresh seeds and physical fleets

Completed results and the independent audit are in the
[confirmation analysis](analysis/spectral_confirmation/report.md). The protocol
below was fixed before those results; its criteria and outcomes are retained.

24 November 2026. This protocol follows the completed
[matched spectral campaign](analysis/spectral_online_cost_model/report.md) and
the [seed-1102 discovery-chain inspection](analysis/spectral_discovery_chains/report.md).
It is fixed before running the full confirmation campaign. The development
result is retained with its unfavorable seeds and matched local controls.

## Questions, methods and frozen choices

The primary question remains whether D256 improves on practical comparator A.
Improvement on B is secondary. D256 versus E256 measures the added value of
cost-model sharing under the same peer-gain channel.

| Label | Method | Information |
|---|---|---|
| A | `peer_gp_bo` | Own adaptive GP and common paid peer verification |
| B | `peer_local_cost_bo` | Own fitted RFF64 mean/residual GP and the same peer rule |
| D256 | `peer_spectral_shared_256_bo` | Frozen shared/local spectral Bayesian model and the same peer rule |
| E256 | `peer_spectral_local_256_bo` | Matched independent spectral cost model and the same peer rule |

Freeze the plants' nonlinear force law, controller structure, desired
trajectories, six speed/frequency contexts, 71 gains, cost function, eligibility
guards, numerical refinement and noise amplitudes. Freeze D256's exact public
basis from the completed campaign, 257-coefficient width, covariance/noise
settings, sharing grid/prior, evidence update, acquisition and message algebra.
A/B retain their existing fitting and hyperparameter adaptation. No method is
retuned to the diagnostic, new fleet draws or confirmation outcomes.

Each nine-vehicle job starts from zero histories and one **charged** factory
test per vehicle. Peer-first verification, round barriers, target-only
eligibility learning and local recommendations remain unchanged. Every
requested test, including failures and unresolved integration, consumes the
same **24-test local cap**. Each context is scheduled four times. Retain
snapshots at 9/18/27/54/108/216 fleet requests. Experimental arms own separate
histories and peer catalogues. All four methods are rerun on the new jobs;
there are no reused online labels or baseline traces.

Communication is left at the current exact Gaussian reference. This campaign
does not compress, defer, approximate or otherwise change a message.

## Replication design

| Condition | Independent analysis units | Runs per unit | Live jobs |
|---|---:|---:|---:|
| Homogeneous | 10 fresh programme/noise seeds | 1 | 10 |
| Heterogeneous | 10 independently drawn physical fleets | 3 fresh programme/noise seeds | 30 |

The homogeneous condition uses the original common compact nominal plant
for nine clients, with programme seeds 2100--2109. Its ten analysis units vary
online context ordering and measurement noise on that declared plant; they
are not ten different physical homogeneous fleets.

Heterogeneous fleet seeds are 3100--3109. Each fleet retains the frozen
nine-slot roster: compact/sport/SUV, each with nominal, payload and front-tire
interventions. For every slot, independently sample the following bounded
manufacturing perturbations around that slot's frozen physical parameters.
Draws are independent between slots and fleets. All draws are retained.

| Draw | Distribution | Physical application |
|---|---|---|
| Log mass | Uniform[-.05,.05] | Multiply mass by exp(draw) |
| Log wheelbase | Uniform[-.03,.03] | Multiply total wheelbase by exp(draw) |
| CG fraction | Uniform[-.02,.02] | Add to the original front-distance/wheelbase fraction |
| Log inertia shape | Uniform[-.05,.05] | Multiply yaw inertia by mass factor x wheelbase factor squared x exp(draw) |
| Log front tire B | Uniform[-.05,.05] | Multiply front B by exp(draw) |
| Log rear tire B | Uniform[-.05,.05] | Multiply rear B by exp(draw) |

The axle distances sum to the new wheelbase and remain positive. Tire C/E,
friction and all other plant parameters retain their frozen values. The
payload/tire interventions are applied in the original slot prototypes before
this manufacturing scatter. This confirms performance within the same family
and intervention composition with physical scatter; it does not test new
vehicle families or a new operating envelope. Family/plant parameters are
simulator inputs and reporting metadata, never cost-model features or labels.

Fleet i uses programme seeds `4100 + 3*i + j`, j=0,1,2. Programmes and online
noise are common to the four counterfactual methods within a job. Fleets have
distinct programme/noise seeds. The ten physical fleets, rather than their
vehicles, contexts or three individual runs, are the heterogeneous replication
units. The manifest stores every physical draw and job before any rollout.
Resume and later analysis load those exact saved physical values, allowing
only floating-point roundoff in a comparison with the public seeded generator.
Thus moving Windows results to another platform does not regenerate a subtly
different physical fleet or public basis.

## Fresh evaluation with batched offline work

The completed development archives provide code/configuration provenance and
the label-free public basis only. No development rollout is used for training
or evaluation in the confirmation campaign.

After all online histories are complete, generate a new dense calibration
reference for every physical plant/context using all 71 gains and two fresh
noise repeats. Select the minimum average-J candidate eligible in both repeats,
using the original deterministic near-tie rule. Calibration selection never
reads held-out validation labels.

On two separate validation noise repeats, evaluate the union of:

- The calibration-only reference winner in each plant/context.
- Every gain recommended at any declared budget by any method/run on that plant.

Taking the union avoids repeating the same held-out simulation for identical
recommendations. It includes early incumbents, not only final controllers.
The dense calibration search and validation union are evaluator-only work and
do not consume or enlarge the online local cap. The validation workload is
smaller than evaluating all 71 gains unless every gain is needed. Vectorized
simulation batches requested gains within the same plant/context/noise episode.
Evaluator seeds are stable hashes of the confirmation namespace, fleet group,
calibration/validation role and repeat. Roles and physical fleets therefore
have separate noise streams. Homogeneous runs reuse one declared held-out bank
for their common physical plant, matching their conditional replication scope.

The full campaign contains:

- **34,560 charged online requests:** 40 jobs x 4 methods x 9 clients x 24 tests.
- **77,532 offline dense-reference rollouts:** 91 unique physical plants x
  6 contexts x 71 gains x 2 noise repeats.
- At most **77,532 offline validation rollouts**, usually fewer because only
  the union above is needed. The actual count is reported after completion.

The offline reference cost is explicit. These uncharged evaluator runs never
become historical training data or a catalogue for any online method.

## Metrics, uncertainty and failure interpretation

The declared final comparison remains near-reference context success at
216 fleet requests with the 5% reference threshold. Retain complete-policy
eligibility, mean J on **common eligible policies**, pair counts, failures,
intermediate budgets, every individual run, and family/context breakdowns.
Per-method conditional means retain their own eligible denominators.

For each contrast, average its context-success difference over three runs
within each heterogeneous fleet, then average the ten fleet effects equally.
For homogeneous results, each programme/noise seed is one unit. Display all
ten unit effects, positive/negative/tied counts and the worst unit. Compute a
95% percentile bootstrap interval by resampling those ten units 10,000 times
with the predeclared stable seed. No context- or vehicle-level bootstrap is used.
The interval describes variability across these declared replication units;
it is not a proof of performance on every fleet or programme.

The saved `confirmation_supported` flag for each final contrast requires all
of the following: ten completed units, full empirical-reference coverage,
a positive lower bootstrap bound for context-success improvement, no reduction
in the aggregate number of validation-eligible complete policies, and lower
mean J on common eligible policies. Read D256/A first, D256/B second, and the
sharing control separately. Report all component metrics even if the combined
flag is false. A success-rate gain with a mean-cost loss remains a tradeoff.
The flags are predeclared evidence criteria, not multiplicity-adjusted p-values.

No physical fleet, seed, context, failed verification or unresolved attempt is
dropped or redrawn after seeing performance. A missing recommendation or
validation-ineligible controller is unsuccessful. If no calibration-eligible
reference exists, or its held-out validation is ineligible, retain that context
in all success denominators, mark it unsuccessful with an unavailable gap,
report reference coverage, and prevent a confirmation flag. Never replace its
reference using validation outcomes. Complete-policy cost/eligibility still
describes the methods' own selected controllers independently of reference
availability. Missing jobs or evaluator batches are incomplete work and stop
analysis/collection until resumed; they are not silently omitted.

## Local execution

Use the same Python environment as the completed spectral experiment. Run from
the repository root:

```bat
git fetch origin
git switch experiment/spectral-confirmation
git pull --ff-only
python code/scripts/tracking_confirmation.py preflight
python code/scripts/tracking_confirmation.py run --workers 6
```

The single run command completes online jobs, dense calibration, selected
validation and audited analysis in that order. Up to six workers are used with
one BLAS thread each. All jobs and evaluator batches are saved atomically.
Repeating the command resumes unfinished work and reuses completed requested
episodes. Do not run two processes against the same output directory.

```bat
python code/scripts/tracking_confirmation.py status
python code/scripts/tracking_confirmation.py analyze --workers 6
```

The offline report is `code/results/tracking_confirmation_v1/report.html`.
The full design, physical draws, public basis, paid histories, independent
evaluation batches, paired effects, replication units and failure counts are
retained there. Analysis replays **all four methods'** recorded decisions before
reading evaluator labels.

After the complete run:

```bat
python code/scripts/tracking_confirmation.py collect --workers 6
git commit -m "Add frozen D256 confirmation results"
git push
```

`collect` validates/reanalyzes the complete campaign and stages only that result
directory. The software smoke check uses
`python code/scripts/tracking_confirmation.py smoke --workers 2`: two jobs,
three clients, eight requests/client, all four methods, 192 charged online
requests, and a fresh small evaluation bank. The full public 71-gain grid/basis
is retained; smoke has one evaluator repeat and is labeled software verification
only. It cannot satisfy confirmation or be collected as full results.
