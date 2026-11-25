# Matched six-speed D/E comparison without a cost uncertainty bonus

This experiment asks whether the shared/local contextual cost model improves
online controller tuning when D256 and its local-only E256 control both use
zero cost uncertainty bonus. The practical priorities remain beating A, then B.
The completed six-speed pilot and its results remain frozen.

The completed [read-only audit and analysis](analysis/six_speed_zero_bonus/report.md)
reproduce all 3,888 paid decisions and all eleven saved numerical tables.
D256 with zero bonus improves average success over A/B and its matched local
control, while retaining seed, complete-eligibility and frequency-specific
limitations. The report records the resulting decision before any new variant.

## Fixed design

| Arm | Cost model | Cost bonus multiplier |
|---|---|---:|
| A | Local adaptive Matérn GP | 2 |
| B | Local 65-coefficient RFF mean and adaptive residual GP | 2 |
| D256, reference | Shared/local spectral Gaussian model | 2 |
| E256, reference | Local spectral Gaussian model | 2 |
| D256, zero bonus | Same shared/local spectral Gaussian model | 0 |
| E256, zero bonus | Same local spectral Gaussian model | 0 |

The six arms run together on fresh programme/noise seeds **6100–6102**. The
reference arms are rerun on those seeds; old scores are not used as matched
controls. Nine homogeneous compact vehicles each request **24** paid tests per
arm, including one factory-gain test. This is **3,888 paid requests**, with no
early stopping. Identical physical requests within a trial share cached
simulation work but still count once against each requesting arm's budget.

Every trial assigns one speed from 20, 22, 24, 26, 28, 30 m/s, sampled
independently with replacement, and one maneuver frequency from 0.1, 0.3,
0.6 Hz in balanced shuffled blocks. The context is known before choosing gains.
All arms share the assignment and noise seed for each client/trial. Each arm
starts empty and keeps its own observations, peer catalogue and model.
All choices in a round precede that round's observations and communication.

The nonlinear bicycle plant, controller, trajectories, noise, feasibility
checks, candidate pool and local eligibility model are inherited without
changes from [the six-speed pilot](tracking_six_speed.md). There are 18 contexts
and 71 candidate feedback gains. The public rank-256 feature matrix is reused
exactly: 1,278 inputs and 257 coefficients, SHA-256
`5e9641525ecb1e32dd1723ebea751bf76166a2cba89fe04c51ed866d62682c4a`.
Its 0.521% Frobenius and 5.112% maximum diagonal errors are retained, including
the slight miss of the earlier 5% diagonal target. No old rollout labels,
trained parameters, histories or evaluator outcomes initialize the new arms.

## What changes in D/E

When no peer gain is available, minimize over locally untested gain/context
pairs in the current context:

\[
 a(x)=\mu_{\log J}(x)-\beta\sqrt{v_{\log J}(x)}
       -\lambda\log\max\{p_{\mathrm{eligible}}(x),p_{\min}\}.
\]

Only **beta changes from 2 to 0** in the two new arms. The frozen eligibility
penalty has lambda = 1 and probability floor = 1e-6. The local classifier,
Bayesian priors, observation noise, shared/local decomposition, evidence-based
sharing mixture, covariance and tie-breaking rule stay unchanged. The model
learns log J; the physical objective and reported costs remain J.

All methods retain the same exact-context, paid, peer-first verification rule:
try a locally untested gain published as an eligible peer incumbent in an
earlier round before proposing a BO gain. A peer suggestion is evaluated
locally and consumes one test. It becomes the incumbent only if locally
eligible and lower-cost than the existing incumbent in that context. There is
no evaluator-informed acceptance decision. Peer catalogues can diverge between
arms as their discoveries diverge.

Predictions and posterior variances are logged even at beta = 0. Numerical
models may be reused within a round only when their full fitting inputs are
identical. Query and communication charges remain per arm. Communication
accounting and messages are unchanged; this is not a communication experiment.

## Evaluation and interpretation

The primary outcome is final success over **encountered contexts**: average
within each vehicle, then equally over vehicles and seeds. Success requires a
locally verified incumbent whose held-out cost is within 5% of the held-out
cost of the independently calibration-selected reference. Ineligible or
missing recommendations in encountered contexts remain failures. Unvisited
contexts and full-grid coverage are reported separately.

The report presents these contrasts, using all three seeds:

1. Zero-bonus D256 versus A, the primary practical benchmark.
2. Zero-bonus D256 versus B, the secondary practical benchmark.
3. Zero-bonus D256 versus zero-bonus E256, isolating the sharing formulation.
4. Zero-bonus D256 versus reference D256, the shared-model bonus effect.
5. Zero-bonus E256 versus reference E256, the local-model bonus effect.
6. The beta = 2 D/E, D/A and D/B reference comparisons on the same new trials.

Complete eligible policies, failed BO/peer requests, first-context and
first-speed visits after the factory test, later visits, paired cost
denominators and all per-seed effects accompany success. Cost comparisons use
only common complete eligible policies; they cannot hide an eligibility loss.
Three programme/noise seeds on one physical plant remain an exploratory pilot.
There is no confidence interval, statistical confirmation flag, or claim
about heterogeneous vehicles.

Fresh evaluator-only dense calibration costs **2,556** rollouts (18 contexts ×
71 gains × 2 repeats), shared by the six arms. Independent validation uses two
repeats for the union of the calibration winner and every gain paid for by any
arm/seed in each context; its upper bound is also 2,556 rollouts. These labels
are absent from the explicitly whitelisted learner payload. All paid decisions
are replayed from earlier paid histories before loading labels for analysis.
A/B use their logged pre-query GP fits during replay, avoiding fresh
platform-dependent optimization. The existing GP bounds-error recovery is
retained for execution.

## Run locally

From the repository root in the existing `paper_FBO_LQI` environment:

```bat
git fetch origin
git switch experiment/six-speed-zero-bonus
git pull --ff-only
python code/scripts/tracking_six_speed_zero_bonus.py run --workers 3
```

This launches the online runs, shared offline references, held-out validation,
decision replay, CSV tables and a self-contained HTML report. Each seed is an
independent worker; BLAS uses one thread per process. Simulation requests are
batched across arms and identical fitted models reuse numerical work.
Repeat the same command after an interruption: checkpoints are atomic at the
round boundary and completed jobs/evaluator batches are reused. Do not change
configuration or source files midway through a run.

Optional checks:

```bat
python code/scripts/tracking_six_speed_zero_bonus.py preflight
python code/scripts/tracking_six_speed_zero_bonus.py smoke --workers 3
```

Smoke uses seed 6199, three clients and eight requests per client/arm (144
paid requests), in a separate `_smoke` directory. Smoke results cannot be
collected as pilot evidence. The full experiment report is:

`code/results/tracking_six_speed_zero_bonus_v1/report.html`

After the full run:

```bat
python code/scripts/tracking_six_speed_zero_bonus.py collect --workers 3
git commit -m "Add matched six-speed zero-bonus results"
git push
```

`collect` verifies/replays the complete experiment, regenerates reports and
stages only this experiment's result directory. Previous results are preserved.

Implementation: `code/scripts/tracking_six_speed_zero_bonus.py`,
`code/src/tracking_six_speed_zero_bonus.py`,
`code/src/tracking_six_speed_zero_bonus_report.py`, and
`code/config/runs/tracking_six_speed_zero_bonus.json`.

## Software verification

71 relevant tests passed, including exact matching of the four reference arms,
variance-invariant zero-bonus selection with the eligibility penalty retained,
independent histories, restart/replay, failure denominators and reporting.
The full 144-request smoke pipeline passed with multiprocessing, evaluator
batches and decision replay. A cached rerun forbidding any new simulator call
left all 37 raw online/evaluation batches unchanged. Smoke outputs are not
research results and are not committed. The full three-seed experiment is
prepared for the local run above.
