# Matched continuation to 25, 50 and 100 total tests

The question is whether a larger local test budget improves absolute controller
quality, and whether frozen every-round lossless D retains its advantage over
A, then B, and matched E as that budget grows. This study continues the ten
completed every-round programmes from the
[lossless catalogue study](tracking_delta_catalogue.md). It does not introduce
another method or communication setting.

## Budget and matched protocol

Every total includes the single factory-controller test. The archived budget
was **24 total tests**, not 24 tests after initialization.

| Checkpoint per client and method | Factory tests | Other tests | Additional tests beyond the archived 24 |
|---|---:|---:|---:|
| 24, archived reference | 1 | 23 | 0 |
| 25 | 1 | 24 | 1 |
| 50 | 1 | 49 | 26 |
| 100 | 1 | 99 | 76 |

One continuation to 100 supplies all checkpoints; separate 25- and 50-test runs
are unnecessary. With ten programme seeds (10100–10109), nine clients and four
methods, **8,640 paid requests are reused**, **27,360 requests are added**, and
the complete study has **36,000 paid requests**. Each client's budget is spread
over its operating contexts; it is not 100 tests per context. Failed tests and
peer verification tests consume the same budget as BO proposals. Physical
duplicates cached within a job remain separate charged method requests.

All four methods use their original priors, hyperparameters, acquisition
bonuses and arm-specific peer-first rule:

| Method | Cost information used for BO | Acquisition bonus |
|---|---|---:|
| A | Local adaptive GP | 2 |
| B | Local 65-coefficient mean and adaptive GP | 2 |
| D | Exact shared/local contextual GP | 0.5 |
| E | Independent GP with D's matched marginal prior | 0.5 |

The public 1,292-controller pool, 18 speed/frequency contexts, homogeneous
vehicles, physics, trajectories, noise, objective and eligibility rules stay
fixed. Every arm receives lossless catalogue changes every round. D additionally
receives every newly eligible cost observation. Own data are current; remote
data for a decision come only from completed previous rounds. Failure labels
stay private. No historical bank or evaluator labels guide a new proposal.

## Preserve the original experiment

The first 24 context assignments, paid observations, choices, GP fit state,
recommendations, communication counters and sender/server/recipient caches
are imported from each archived checkpoint. No factory or prefix test is
repeated. The publication after round 24 was unused by the original final
decision; it is retained and becomes available for query 25. Snapshot traffic
excludes the contact after that snapshot's final decision.

The runner first replays all **8,640 archived requests** through the continuation
planner, with physical simulation and hyperparameter optimization disabled.
It verifies decisions and full method states, including lossless caches and
packets. An immutable prefix hash is also checked on import, resume and analysis.

The original schedule generator cannot simply be called with a longer horizon:
doing so would change its first 24 speed draws. The extension instead appends
a deterministic suffix. Each three-trial block contains all three frequencies
in randomized order; each trial independently draws one of six speeds with
replacement. Dedicated per-block/per-trial seeds make the suffix independent
of the requested stopping horizon. All methods share each client's programme
and physical noise seeds. Pauses at 25 or 50 do not reset any model or cache.

For practical simulation time, identical D fits are reused within a round.
With complete every-round delivery, all D clients have the same eligible
training view before that round; their client-specific predictions are still
computed separately. Prefix replay verifies unchanged decisions. This is
simulator computation reuse, not a per-client CPU or privacy claim. Local E
fits and both A/B adaptive fits remain independent.

## Evaluation and interpretation

The report keeps three context denominators separate:

| Scope | Contexts scored for a client | Purpose |
|---|---|---|
| `fixed_24_contexts` | Distinct contexts in that client's original 24 assignments | Primary learning curve on the same task at every budget |
| `encountered_by_budget` | Distinct contexts encountered by the current checkpoint | Original metric, with its growing denominator visible |
| `full_18_contexts` | All 18 declared contexts, including unvisited ones | Coverage of the full operating grid |

Success requires a locally tested recommendation that passes held-out
eligibility and costs at most 1.05 times the frozen empirical reference.
Missing and ineligible recommendations count as failures. Contexts receive
equal weight within a client, then clients and programmes receive equal
weight. The finite-pool empirical reference is not a certified global optimum.

The 24-test scores must reproduce the archived every-round report exactly.
This includes D's 40.11% encountered-context success on the full ten-programme
cohort. Cost averages retain explicit common-eligible denominators. Reports
also include failed requests, eligible coverage, frequency effects, looser
gap thresholds as diagnostics, query allocation, distinct tested inputs and
application traffic.

Retrospective local-ever and fleet-ever near-reference rates ask whether a
successful gain had already been tested locally or anywhere in that arm by a
checkpoint. These held-out oracle diagnostics help distinguish discovery,
local verification and recommendation selection; they never enter the planner.

Reference selection reuses the original archived calibration. Final evaluation
uses the same four held-out noise seeds as the archived confirmation. Identical
candidate/context/repeat rows are reused; only missing rows are simulated, after
all live queries finish. New proposals and intermediate checkpoints never see
those labels. This preserves the matched 24-test baseline and avoids another
dense calibration bank. The extra held-out simulation count depends on the
observed candidate union and is reported separately from paid online tests.

D/A, D/B and D/E differences and changes from 24 use paired programme-level
effects. Descriptive 95% bootstrap intervals resample the ten programmes, not
the 90 dependent clients. These programmes and evaluation seeds have already
been examined: this is a matched budget diagnostic, not a fresh independent
confirmation. Analyse absolute quality and budget dependence before selecting
any further experiment.

## Run locally

Use the existing environment, from the repository root. No new dependency is
needed.

```powershell
git fetch origin
git switch experiment/tracking-budget-100
git pull --ff-only
python code/scripts/tracking_budget_extension.py run --workers 3
```

Independent programme jobs run in a process pool with one BLAS thread per
worker. `--workers 6` is also supported if the machine has sufficient resources.
State is saved atomically after every completed round. Repeat the same command
after interruption: imported prefixes, completed rounds and valid held-out
batches are reused. An interrupted incomplete round or batch may need to be
recomputed. Keep the source results and hashed code unchanged while running.

Optional checks:

```powershell
python code/scripts/tracking_budget_extension.py audit --workers 3
python code/scripts/tracking_budget_extension.py preflight --workers 3
python code/scripts/tracking_budget_extension.py smoke --workers 3
```

`audit` and `preflight` require no new control rollouts. `smoke` uses the real
nine-client, full-pool seed 10100 prefix and continues only to 28 total tests:
864 reused requests and 144 additional requests, followed by missing held-out
rows. Smoke outputs are separate and cannot be collected as the 100-test study.

Main output: `code/results/tracking_budget_extension_v1/report.html`.
The principal tables are `budget_summary.csv`, `paired_comparisons.csv`,
`programme_bootstrap.csv`, `change_programme_bootstrap.csv` and
`communication_and_query_allocation.csv`. `verification.json` records reused
and new paid/evaluator counts; `budget_readout.json` records the analysis scope.

## Verify and upload completed results

```powershell
python code/scripts/tracking_budget_extension.py collect --workers 3
git diff --cached --stat
git commit -m "results: matched controller tuning at 25 50 and 100 tests"
git push origin HEAD
```

`collect` verifies the immutable prefix and replays all continuation decisions
and packets with simulation and optimization prohibited. It regenerates the
report using saved held-out rows and stages only the new main-study and
prefix-equivalence directories. It cannot silently finish a partial experiment
or stage smoke results.

Implementation checks:

```powershell
python -m unittest discover -s code/tests -p "test_tracking_budget_extension.py" -v
```

The checks cover exact import, horizon-independent schedules, query-25 resume,
immutable-prefix tampering, decision/packet replay, fixed metric denominators,
missing-row evaluation reuse and exact GP fidelity with 100 observations per
client.

Before release, all six tests pass, and the complete ten-programme prefix gate
replays 8,640 decisions exactly. The 28-test smoke completes its 144 new paid
requests, reuses 1,228 held-out rows and simulates 108 missing held-out rows.
Its choices and packets replay, and its original 24-test scores reproduce.
A cached rerun and the full 100-test preflight also pass with physical
simulation and both hyperparameter optimizers disabled. The full 100-test
continuation remains for the local run; smoke is software verification only.
