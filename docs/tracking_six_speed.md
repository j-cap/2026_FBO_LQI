# Six discrete speeds: homogeneous contextual BO pilot

25 November 2026. This is a separate, exploratory follow-up to the frozen
[confirmation](tracking_confirmation.md) and [GP replay closure](analysis/gp_replay_closure/report.md).
It changes the operating-context programme, not the vehicle dynamics,
controller, objective, gain pool or communication rule. No old online outcomes
or evaluator labels are read by the new runner.

## Question and fixed design

Can the shared/local cost model help when a vehicle has few observations at
each encountered speed/frequency context? Read D256/A first, D256/B second,
and D256/E256 to isolate the extra cost-model-sharing contribution beyond
the common peer-gain channel. Improvement is a hypothesis, not a requirement
used to change the task after seeing outcomes.

| Choice | Pilot |
|---|---|
| Vehicles | Nine clients on the same nominal compact plant |
| Programme/noise seeds | 5100, 5101, 5102 |
| Speeds | 20, 22, 24, 26, 28, 30 m/s |
| Maneuver frequencies | Existing 0.1, 0.3, 0.6 Hz |
| Context grid | 18 speed/frequency pairs |
| Gain pool | Existing 71 vectors; 1,278 public gain/context inputs |
| Local cap | 24 charged requests, including one factory test |
| Fleet snapshots | 9, 18, 27, 54, 108, 216 requests per method |
| Methods | A, B, D256, E256, all simulated afresh |
| Initial histories | Empty |
| Reference / validation repeats | Two fresh noise repeats for each role |

Every trial has one fixed speed. For each client, speeds are sampled uniformly
with replacement, independently of other clients. The same speed can recur
on consecutive trials. Frequency uses a random permutation of the three
frequencies in each block, so each frequency appears eight times in 24 trials.
Speed/frequency contexts are not forced to have equal counts or complete
coverage. All programmes are generated without costs and saved in the manifest.

The context is externally assigned and known before gain selection. BO chooses
the gains, not the speed or frequency. Each trial is the existing 25-second
rollout with a fresh reset, unchanged smooth reference construction, peak
desired lateral acceleration of 2.5 m/s² and measurement-noise amplitudes.
Speeds remain inside the previously tested 20–30 m/s envelope. This first
pilot uses the same speed distribution at every client; it does not yet impose
different route distributions or physical heterogeneity.

After the charged factory test, all four methods first verify an untried,
prior-round eligible peer incumbent in the scheduled **exact** context when
available. The donor's cost ranks proposals; local eligibility and cost
determine the recipient's incumbent. Otherwise, the unchanged
eligibility-penalized LCB chooses an untested gain. Failed and unresolved
requests consume budget. There is no near-reference stopping rule.

All choices within a round precede that round's observations. Histories,
incumbent catalogues and learned states remain separate between methods.
Peer publications, target-only eligibility learning and the 5% evaluation
threshold retain their existing definitions.

## Statistical model and representation check

A retains the adaptive local Matérn GP. B retains the locally fitted RFF64 mean
and adaptive residual GP. D256 retains the shared/local Gaussian cost model,
six sharing components, priors, noise settings, evidence update and total
mixture variance. E256 fixes sharing to zero with the same marginal prior.
Both spectral vectors still have 257 coefficients.

The new public basis is computed on the larger grid and pinned before any
controller rollout. It uses only public gain/context coordinates. All
length scales and covariance settings remain fixed for D/E.

The measured label-free representation diagnostics on this grid are:

| Diagnostic | Six-speed rank 256 | Earlier target |
|---|---:|---:|
| Relative kernel Frobenius error | Approximately 0.521% | 2% |
| Maximum relative diagonal error | Approximately 5.112% | 5% |

The slight diagonal-target miss is retained explicitly in preflight, the
manifest and report. This pilot keeps D256/E256 for continuity; it does not
silently increase rank, renormalize the kernel or relax the earlier target.
The result is not a claim that all earlier posterior/acquisition accuracy
targets were re-established for this grid. D/E remain matched within the
same finite-feature model.

For efficiency, Gaussian components are fitted once after each round using
the original client information messages and server aggregation. Predictions
are calculated only for candidates in the scheduled context when BO needs
them. Tests compare these lazy predictions, evidence weights and payloads
against the original full-grid implementation. No message is compressed,
delayed or omitted relative to that Gaussian reference. The shared/local
dependence and between-component variance are preserved.

A/B now log fitted GP parameters and refit counters before receiving the
current query outcome. The audit restores these learned states rather than
assuming identical fresh numerical optimization across platforms. It checks
the refit schedule, bounds, non-refit carryover, acquisition values and chosen
actions. D/E predictions and choices are reconstructed from their own prior
histories. This does not claim fresh cross-platform GP optimization is identical.

## Evaluation and denominators

The **primary pilot outcome** is final near-reference success over each
vehicle's **encountered contexts**, averaged equally over vehicles and then
over programme seeds. The context programme fixes these denominators before
any costs are observed. A context counts as encountered even when every
attempt in it fails. A missing local incumbent or validation-ineligible
recommendation in an encountered context is unsuccessful.

Unvisited contexts are reported separately, not treated as calibrated and not
silently added as failures to this encountered-context score. Full-grid policy
eligibility is also reported separately. The number of encountered contexts
can vary across clients. Therefore the report displays both the primary
equal-client rate and the pooled success count/rate, whose weighting differs.
As the encountered denominator grows over time, budget curves are not
fixed-grid success curves.

Retain these additional readouts:

- Complete validation-eligible policy counts **over encountered contexts**.
- Mean J over common eligible complete policies, with paired counts.
- Actual paid-controller quality and eligibility on first context visits,
  first speed visits, and later context visits.
- The locally verified incumbent after each visit, separately from the
  requested controller, since a poor exploratory test can retain a good incumbent.
- First-visit summaries both including and excluding the common factory test.
- BO/peer request allocation, local failures, unresolved requests, every seed,
  and speed/frequency coverage.

For paid-controller success, the online request must itself be eligible and
the controller must be eligible in both held-out repeats with relative gap
at most 5%. A failed online request is unsuccessful even if the same gain is
eligible on held-out noise. Failed attempts never become finite cost-model
targets. Their eligibility labels remain available to the local classifier.

Dense reference selection uses all 71 gains on fresh calibration repeats,
requiring eligibility in both. Held-out validation evaluates the union of
the calibration winners and **every requested gain** across methods/seeds.
The latter also covers every incumbent needed at any budget or visit.
Calibration and validation seeds have separate namespaces from online tests
and from previous campaigns. No validation label selects a reference winner.

All methods share one evaluator bank for their common physical plant. Each
programme/noise seed, not its clients or contexts, is a replication unit.
This three-seed homogeneous pilot reports paired directions and individual
seed effects. It assigns no confirmation flag or three-seed confidence
interval, and it is not evidence on heterogeneous fleets.

## Workload, isolation and execution

The full pilot requests:

- **2,592 charged online tests:** 3 seeds × 4 methods × 9 clients × 24 tests.
- **2,556 offline dense-reference rollouts:** 18 contexts × 71 gains × 2 repeats.
- **At most 2,556 held-out validation rollouts**, usually fewer through the union.

Online jobs run independently per seed. Within a job, requested counterfactual
gains with the same client, episode and noise are batched and cached, while
every method is charged for each requested rollout. The common homogeneous
plant's evaluator is reused across programme seeds, with its conditional
replication scope explicit. Worker payloads exclude old histories, reference
winners, reports and evaluation tables. A single BLAS thread per worker
avoids CPU oversubscription. Three workers are a reasonable default.

Atomic checkpoints are written after each complete round; evaluator batches
are atomic as well. Repeating `run` resumes unfinished work. Source hashes,
programmes, physical values and public basis are checked on resume. Keep all
these files together when moving a run. Do not run two processes against the
same result directory. Completed old result directories are untouched.

From the repository root, use the existing environment:

```bat
git fetch origin
git switch experiment/six-speed-contextual-bo
git pull --ff-only
python code/scripts/tracking_six_speed.py preflight
python code/scripts/tracking_six_speed.py run --workers 3
```

`run` performs online testing, dense calibration, selected validation, paid
decision audit and reporting. It resumes automatically after an interruption.

```bat
python code/scripts/tracking_six_speed.py status
python code/scripts/tracking_six_speed.py analyze --workers 3
```

The report is **`code/results/tracking_six_speed_v1/report.html`**. Start with
the final primary comparison, then the secondary and sharing controls.
Inspect `paired_comparisons.csv`, `per_seed_comparisons.csv`, `visit_summary.csv`,
`query_modes.csv`, coverage columns in `online_summary.csv`, and `verification.json`.

The optional software smoke uses three clients and eight requests/client,
one evaluator repeat per role, the full public grid and all four methods:

```bat
python code/scripts/tracking_six_speed.py smoke --workers 2
```

Its 96 charged requests and separate `_smoke` directory are software checks,
not pilot evidence. Smoke and pilot evaluator noise have separate namespaces.
Smoke cannot be collected as the full pilot.

After the full run, validate and stage only its result directory:

```bat
python code/scripts/tracking_six_speed.py collect --workers 3
git commit -m "Add six-speed contextual BO pilot results"
git push
```
