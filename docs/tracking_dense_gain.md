# Matched online BO on the denser K_y-extended controller pool

This follow-up tests whether D beats A and then B, and whether sharing adds
value over matched E, after the [gain-domain audit](analysis/gain_domain/report.md)
identified missing sampling resolution and a restrictive K_y cap. Analyse
and consolidate these results before starting communication experiments.

## Fixed public search task

| Gain | Nominal | Permitted physical range |
|---|---:|---:|
| K_y | 0.005 | 0.0005–0.15 |
| K_psi | 0.1 | 0.01–1 |
| K_r | 0.1 | 0.01–1 |

The pool is the geometry-only subset of the audited 2,011-point design that
respects these ranges: **1,292 controllers**, comprising all 1,033 points
inside the old box plus 259 exterior points with K_y above 0.05. Every
original 71-vector record and every expanded-audit winner is retained.
Candidates are not screened by feasibility or cost. The audit-derived
design includes the previously declared retrospective boundary probes.

Gain normalization remains `(log10(multiplier)+1)/2`; the K_y coordinate
can exceed 1. This preserves the physical meaning of the previous kernel
length scales. Six speeds (20/22/24/26/28/30 m/s) and three frequencies
(0.1/0.3/0.6 Hz) give **23,256 public gain/context inputs**. The same
homogeneous compact plant, trajectories, noise levels, 25-second rollout,
objective, actuator limits and integration/eligibility checks are retained.

## Four matched methods

| Method | Cost information and model | Cost bonus |
|---|---|---:|
| A | Own paid observations; adaptive local GP | 2 |
| B | Own paid observations; original 65-coefficient local cost mean plus adaptive GP | 2 |
| D | Online Gaussian cost-model messages; shared/local exact contextual GP and evidence mixture over sharing strength | 0.5 |
| E | Own paid observations; independent exact contextual GP with D's marginal prior | 0.5 |

All four have the same **paid, prior-round, exact-context peer-first rule**.
A/B therefore also transfer peer gains. E isolates sharing of cost-model
information beyond that common channel. Each arm owns its histories,
catalogue and incumbents; no arm borrows another arm's discoveries.

Nine clients each receive **24 paid requests**, including the single factory
test and all failed or numerically unresolved tests. Each client holds one
known speed/frequency context per trial. Speeds are independently drawn with
replacement and frequencies use balanced shuffled blocks, as before. Round
actions precede all round observations and communication. Context schedules
and measurement-noise seeds are paired across methods.

There are three fresh programme/noise seeds: **8100, 8101, 8102**. Thus the
pilot has **2,592 paid requests**. Histories start empty; there are no blind
six-test initializations or archived training labels. The fixed 0.5 bonus is
the previously tested small-bonus setting, with no new bonus sweep.

## Exact representation without a grid-sized kernel

The denser task makes representation error a material potential confound.
D/E therefore use the exact target Matérn covariance, including its constant
component, rather than retaining the old rank-256 approximation. The shared
plus client-specific Gaussian formulation, log-cost likelihood, noise level,
sharing-strength mixture and Gaussian information-message algebra remain.

For D, form the sorted dictionary U of distinct eligible inputs requested
in earlier rounds. Whiten K(U,U) with its Cholesky factor L. A query has
features `K(x,U) L^(-T)`, plus conditional GP variance
`K(x,x) - K(x,U) K(U,U)^(-1) K(U,x)`. Training observations are all in U,
so their conditional remainder is zero. Applying the unchanged Gaussian
client messages and server aggregation in these coordinates gives the exact
shared/local posterior. The full conditional query remainder is retained,
including at unseen gain/context inputs.

U has at most 216 entries at the final fit, and at most 207 before the last
round's decisions. No 23,256-by-23,256 kernel is built. D's coefficient basis
now changes with U; it is **not a fixed 257-coefficient spectral model**.
E fits each client's exact local GP without using other clients' cost data
or dictionaries. A/B retain their existing fits and GP-bounds recovery.

The dictionary communicates public action IDs; labels remain local inputs
to Gaussian information-message formation. Reference payload accounting
includes full dictionary publication/broadcast and dense Gaussian messages.
No compression, privacy guarantee or network-performance result is claimed
by this experiment. Communication choices remain a later task.

Before live tests, preflight compares this information-message posterior
against direct observation-space GP calculations. Synthetic histories use
0, 1, 4 and 24 eligible observations per client, including repeated inputs
across nine clients. Fixed public probes cover the actual pool, all contexts
and the extended K_y region. Means, variances and sharing weights must agree
within the fixed numerical tolerance. No vehicle cost labels enter this check.

## Reference reuse and fresh evaluation

Only after online decisions complete, reuse the two archived calibration
repeats for these 1,292 controllers: **46,512 observations**, with zero new
calibration rollouts. Source signatures, source hashes, archived simulator
parity and input fingerprints are checked. Select the eligible minimum
calibration-cost reference separately in each context.

The reference and union of all paid controllers from every arm/seed then
receive **four fresh common validation repeats**, with noise disjoint from
the archived calibration and validation banks. No validation-based reference
reselection or fallback. Evaluator labels and reference choices are excluded
from the explicit worker payload. Reusing archived labels does not warm-start
any learner. Missing/ineligible references remain explicit.

Primary success is a validation-eligible incumbent within 5% of the reference
J in an encountered context. Average contexts within each client, then clients
and seeds equally. Missing or ineligible recommendations remain failures;
unvisited contexts remain separate. Report D minus A, D minus B and D minus E,
per-seed signs, all failed requests, complete eligible policies, and paired
mean J on common complete eligible policies with its denominator.

The report separates success advantages and consistency from failure,
coverage and conditional-cost tradeoffs. A run completes whether these
descriptive criteria pass or fail. Three programme/noise seeds on one plant
are an exploratory matched follow-up, not independent-fleet confirmation.
The old 71-point online studies are not reranked or overwritten.

## Local commands

From the repository root in `paper_FBO_LQI`:

```bat
git fetch origin
git switch experiment/dense-gain-online
git pull --ff-only
python code/scripts/tracking_dense_gain.py run --workers 3
```

The command performs preflight, online trials, fresh validation, paid-decision
replay and reporting. Online checkpoints are atomic after each full round;
validation is checkpointed in candidate batches. Repeating the command reuses
completed work. BLAS runs with one thread per worker. Keep code/configuration
fixed during a run. No fresh GP optimization or simulation occurs during
decision replay.

Optional checks:

```bat
python code/scripts/tracking_dense_gain.py preflight
python code/scripts/tracking_dense_gain.py smoke --workers 3
```

Smoke uses a separate 169-point pool, three clients, eight requests, two
validation repeats and seed 8199. All 18 contexts remain in its programme.
Its report is explicitly software verification and cannot be collected as
the full pilot.

Full report: `code/results/tracking_dense_gain_v1/report.html`.

After the full run:

```bat
python code/scripts/tracking_dense_gain.py collect --workers 3
git commit -m "Add matched dense-gain online BO results"
git push
```

`collect` replays and verifies the run, regenerates its readout, and stages
only this experiment's output directory. It does not launch new simulations.
After analysing these results, consolidate whether the chosen scientific
claim is supported and practical tradeoffs are acceptable, then decide on
communication work.

## Software verification before publication

Five focused tests pass: the full-pool geometry and learner/evaluator boundary,
direct shared/local GP equivalence, local-model independence and unobserved
conditional variance, interrupted-round resume and logged-fit replay, and the
fresh-validation/smoke-collection guards. Full-grid preflight also passes the
synthetic representation check for all four declared history sizes.

The physical smoke workflow passes with spawned workers: 96 paid requests,
154 fresh validation rollouts, all 18 available references, and complete
paid-decision replay. A cached rerun with simulation prohibited preserves
the contents and timestamps of all 37 raw online/validation files. Previous
simulator, controller and result files remain unchanged. No full pilot
simulation was run before publication.
