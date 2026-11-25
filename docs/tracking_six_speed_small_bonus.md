# One fixed small-bonus comparison before consolidation

This is the focused follow-up to the
[audited zero-bonus experiment](analysis/six_speed_zero_bonus/report.md).
Zero-bonus D had the strongest average success and fewer failed tests, but
neither zero-bonus method discovered a near-reference controller at 0.3 Hz.
The next question is whether a restrained cost uncertainty bonus can recover
useful discoveries while preserving the shared model's practical advantages.

**The only new policy parameter is beta = 0.5**, one quarter of the earlier
beta = 2 coefficient. It is fixed for every speed, frequency, client and
round. This is a single candidate, not a sweep or a frequency-specific rule.
Its ability to resolve the gap is a hypothesis to test.

## Matched experiment

| Arm | Model | Cost bonus beta |
|---|---|---:|
| A | Existing local adaptive GP | 2 |
| B | Existing local RFF mean and adaptive residual GP | 2 |
| D0 | Existing shared/local spectral Gaussian model | 0 |
| E0 | Existing local spectral Gaussian model | 0 |
| D-small | Same shared/local spectral Gaussian model | 0.5 |
| E-small | Same local spectral Gaussian model | 0.5 |

D0/E0 refer to the rank-256 spectral zero-bonus controls. A/B/D0/E0 are rerun
on the new trials; scores from prior seeds are not used as paired controls.
The beta = 2 D/E results remain frozen and provide historical context only.

- Fresh programme/noise seeds: **7100, 7101, 7102**.
- Nine homogeneous compact vehicles; **24 paid tests per vehicle and arm**.
- Six speeds: 20, 22, 24, 26, 28, 30 m/s, drawn independently with replacement.
- Three frequencies: 0.1, 0.3, 0.6 Hz, in balanced shuffled blocks.
- One externally assigned speed/frequency context per trial, known before gains.
- **3,888 paid requests**, including the charged factory test and failures.
- Fleet snapshots at 9, 18, 27, 54, 108 and 216 requests per arm.

The nonlinear plant, controller, trajectories, physical objective J, noise,
71 candidate gain vectors, feasibility and integration checks, and local
eligibility estimator are unchanged. The public rank-256 features are reused
exactly: 1,278 gain/context inputs and 257 coefficients. Their 0.521%
Frobenius and 5.112% maximum diagonal errors remain explicit, including the
slight miss of the earlier 5% diagonal target. No kernel accuracy target is
relaxed and no rank is changed.

All arms start with empty paid histories and maintain separate models and
peer catalogues. After the single factory test, each arm uses the same
prior-round exact-context peer-first rule. Peer gains must be tested locally,
consume one request and become incumbents only if locally eligible and
better than the existing incumbent. If no peer proposal is available, BO
chooses an untested gain/context pair. All actions in a round precede that
round's observations and communication.

The score remains

\[
 a(x)=\mu_{\log J}(x)-\beta\sqrt{v_{\log J}(x)}
      -\log\max\{p_{\mathrm{eligible}}(x),10^{-6}\}.
\]

Bayesian covariance, sharing evidence, priors, noise and Gaussian information
messages remain active at both bonuses. Communication accounting is retained;
this is not a communication study. The existing A/B GP bounds recovery and
logged-fit replay are reused.

## Reading the result, fixed before the run

The primary metric remains final near-reference success over encountered
contexts: average within a client, then equally over clients and seeds. A
success requires a locally eligible controller whose held-out J is within
5% of the held-out cost of the independent calibration-selected reference.
Missing or ineligible recommendations in encountered contexts remain failures;
unvisited contexts are reported separately.

Read these comparisons in order:

1. D-small versus **A**, then **B**, on the unchanged primary metric.
2. D-small versus E-small, measuring the added sharing contribution with the
   same acquisition and peer rules.
3. D-small versus D0: overall success, 0.3 Hz success, failed paid tests and
   complete eligible policies.
4. E-small versus E0, checking whether exploration helps the local model too.
5. D0 versus E0/A/B on the same new programmes.

The report exports final frequency-specific success with equal client/seed
weights within each frequency, plus pooled counts. It records the first
held-out near-reference discovery in each seed/frequency, including its
round, candidate, client and BO/peer origin. The targeted diagnostic asks
whether D-small improves 0.3 Hz success over its matched D0 control and
discovers a useful controller there in every seed. Other frequencies remain
visible; recovery at one frequency cannot silently replace the primary score.

A **clean improvement over D0** requires higher overall success, higher
0.3 Hz success, no more failed paid requests and no fewer complete eligible
client policies. Otherwise report the tradeoff, even if average cost or
success improves. The report also exposes the complete-policy gap to B.
Failed BO rates and total failed paid counts are both shown because the
BO/peer allocation can change between arms. Paired cost comparisons keep
their common-complete-policy denominators explicit.

These are descriptive engineering checks, not statistical noninferiority or
confirmation tests. Three seeds on one plant remain an exploratory pilot.
A valid experiment completes and can be collected whether the checks pass
or fail. No setting is automatically changed in response to the results.

## Efficient local run

Use the existing `paper_FBO_LQI` environment from the repository root:

```bat
git fetch origin
git switch experiment/six-speed-small-bonus
git pull --ff-only
python code/scripts/tracking_six_speed_small_bonus.py run --workers 3
```

The three seeds run in separate processes with one BLAS thread per process.
All six arms share cached identical physical simulations and batch distinct
requests within a trial. Exactly equal model-fitting inputs can reuse a
read-only numerical fit; each arm still pays its own query and communication
charges. Each completed round is checkpointed atomically. Repeat the same
run command after an interruption to resume.

One common fresh evaluator bank serves all six methods: **2,556 dense
calibration rollouts**, followed by two validation repeats for the union of
all paid gains and calibration winners (at most another 2,556 rollouts).
Old labels, histories and trained parameters are not reused. Evaluator data
are absent from the learner payload. Analysis replays all paid decisions
before loading evaluator labels, with new optimizer/simulator calls prohibited
during replay.

Optional checks:

```bat
python code/scripts/tracking_six_speed_small_bonus.py preflight
python code/scripts/tracking_six_speed_small_bonus.py smoke --workers 3
```

Smoke uses seed 7199, three clients and eight requests per arm/client (144
paid requests), in a separate `_smoke` directory. Smoke outputs cannot be
collected as research results. The main report is generated at:

`code/results/tracking_six_speed_small_bonus_v1/report.html`

After the full run, verify, stage and publish the results:

```bat
python code/scripts/tracking_six_speed_small_bonus.py collect --workers 3
git commit -m "Add matched six-speed small-bonus results"
git push
```

`collect` regenerates the report and stages only this experiment's directory.
It preserves the completed reference experiments and their source signatures.

## Consolidation is the next step

The runner generates `consolidation_inputs.md`, `pilot_readout.json`, all
per-seed comparisons, frequency outcomes/discoveries, paid-request logs,
allocation/eligibility tables and replay verification. These accompany the
raw simulations and pinned manifest when collected.

After this result, consolidate the original six-speed pilot, the zero-bonus
comparison and this fixed small-bonus comparison before adding another
variant. Keep within-study paired contrasts separate: programmes and held-out
noise differ between studies, so raw success percentages should not be
subtracted across them as matched effects.

The bigger-picture review should determine which controller-tuning problem
fleet information demonstrably solves, what the learned cost model adds
beyond gain transfer, which failure/coverage limits remain, and whether the
next scientific step is confirmation, communication or physical heterogeneity.
The current comparison does not predetermine that decision.

## Software verification

All 28 relevant tests passed, including matching A/B/D0/E0 against the frozen
implementation, beta = 0.5 selection in every context, independent histories,
restart/replay, failed-test denominators and the descriptive readout.
The complete 144-request smoke workflow passed with multiprocessing,
calibration, validation, replay and report/consolidation generation.
A cached rerun, with simulator and optimizer calls prohibited, left all
37 raw online/evaluator files unchanged. Smoke outputs are not committed.
The full three-seed run is prepared for local execution.
