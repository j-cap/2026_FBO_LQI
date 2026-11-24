# Matched online spectral BO: A first, then B

24 November 2026. This experiment follows the completed
[kernel representation assessment](analysis/kernel_representation_assessment/report.md).
That assessment identified spectral 256 as the smallest tested representation
meeting all declared numerical targets. Spectral 384 was closer to the exact
kernel. Neither result establishes online controller performance: changing a
decision changes later local histories and peer discoveries.

The experiment retains the [original D/E shared/local model](tracking_hierarchical.md)
and changes only its public feature representation. Completed experiments and
their configurations remain frozen.

## Question and declared comparison order

Can a more accurate kernel representation make D beat A within the same online
calibration budget? Beating B is the second objective. Compare homogeneous and
heterogeneous conditions separately, with the same metrics and denominators.

The **primary comparison is D256 minus A in near-reference context success at
216 fleet queries**. The secondary comparison is D256 minus B. D384/A and
D384/B are fixed accuracy controls; a favorable D384 result cannot replace the
declared primary result after seeing outcomes.

| Label | Runner method | Cost model | Role |
|---|---|---|---|
| A | `peer_gp_bo` | Original adaptive local GP | Primary benchmark, reused |
| B | `peer_local_cost_bo` | Original local fitted mean and residual GP | Secondary benchmark, reused |
| D0 | `peer_hierarchical_bayes_bo` | Original shared/local RFF64 model | Representation reference, reused |
| E0 | `peer_local_bayes_bo` | Original independent RFF64 model | Local representation reference, reused |
| D256 | `peer_spectral_shared_256_bo` | Shared/local spectral 256 plus intercept | Main candidate, simulated |
| E256 | `peer_spectral_local_256_bo` | Independent spectral 256 plus intercept | Matched sharing control, simulated |
| D384 | `peer_spectral_shared_384_bo` | Shared/local spectral 384 plus intercept | Accuracy control, simulated |
| E384 | `peer_spectral_local_384_bo` | Independent spectral 384 plus intercept | Matched sharing control, simulated |

D256/E256 and D384/E384 isolate cost-model sharing within their model class.
The E models still exchange peer gains. D256/D0 and D384/D0 assess the change
of representation in the complete online policy. D384/D256 assesses the
remaining effect of representation size. These comparisons include any later
changes to the arm's own peer catalogue caused by its discoveries.

## Public basis and unchanged inference

The public input grid contains 426 gain/context pairs: 71 gain vectors in six
contexts. Inputs remain three normalized log-gain coordinates, normalized speed
and normalized log frequency. No family label, vehicle parameter, historical
cost label or dense-reference outcome is a model input.

Construct the Matérn signal covariance on this public grid with the original
fixed length scales and amplitude. If its eigendecomposition is
`K_signal = U diag(lambda) U.T`, the prior-whitened factor at size r is

\[
 F_r=[\sqrt{0.25}\,\mathbf 1,\;U_{1:r}\operatorname{diag}(\sqrt{\lambda_{1:r}})].
\]

The intercept is exact. The signal approximation is `F_r F_r.T` after removing
the intercept contribution. With
`S_r = diag(0.25, 1, ..., 1)` and `phi_r = F_r S_r^(-1/2)`, retain

\[
 \log J_{ik}=\phi_r(x_{ik})^\top(w_g+u_i)+\epsilon_{ik},\quad
 w_g\sim\mathcal N(0,\rho S_r),\quad
 u_i\sim\mathcal N(0,(1-\rho)S_r).
\]

Observation noise remains 0.05 on log J, with the original numerical jitter.
The sharing grid `[0, .25, .5, .75, .95, 1]` retains its uniform prior.
Likelihoods, evidence weights, shared/local dependence, total mixture variance,
information messages and acquisition formulas call the original implementation.
There is no appended residual GP, additional shrinkage or hyperparameter search.
E fixes rho to zero with the same marginal prior as its D counterpart.

There are **257 coefficients at size 256 and 385 at size 384** for each shared
or local coefficient vector. These widths are not counts of tested controllers.
The allowed gains and local query budget are unchanged.

Before any new rollout, `public_bases.npz` pins both feature arrays, action
labels and public input coordinates. Every worker receives those same arrays.
The manifest hashes the arrays exactly; resume and analysis reload the saved
arrays rather than recomputing an eigendecomposition on another machine.
Public grid coordinates allow only `1e-14` absolute roundoff on reload, while
action labels and basis hashes remain exact. Keep this file with the results.

## Matched online protocol

Retain the nonlinear plants, desired trajectories, feedback controller,
objective, candidate gains, contexts, physical fleet and measurement noise from
the completed D/E experiment. Use nine clients, seeds 1100/1101/1102, and both
homogeneous and heterogeneous conditions. Snapshots remain at fleet budgets
9/18/27/54/108/216. Each vehicle has **24 charged rollout requests**; the
balanced programme visits each of six contexts four times.

Every arm starts with one factory-gain test. From the second query, it first
checks for an eligible peer incumbent in the scheduled context, published by a
different vehicle before the current round and not yet tried by this vehicle.
If available, it selects the best donor cost and tests that gain locally. A
donor's success never replaces recipient verification. If no such gain is
available, the arm uses the unchanged feasibility-penalized LCB. Eligibility
learning uses the target's own tests. Failed and unresolved attempts consume
budget, and recommendations must have been found eligible locally.

Each of the four new methods owns its own paid histories, posterior and peer
catalogue. No observations or gains move between experimental arms. All
decisions for a round are fixed before observations from that round arrive.
Identical requested simulator episodes may be computed once in a job's cache,
while every method is charged for every request.

The full campaign simulates four new methods across six fleet jobs:

- **5,184 newly charged requests:** 4 methods x 2 conditions x 3 seeds x 9
  clients x 24 requests.
- **5,184 existing charged requests:** A/B/D0/E0 traces reused only by the
  evaluator, giving 10,368 requests represented in the eight-arm comparison.

Completed baseline signatures, source hashes, fleet identities and factory
observations are checked. The simulation worker payload contains neither the
old result paths nor the dense evaluator tables. The new methods start with
zero historical training records. Reuse is an evaluation cost saving, not
online access to prior experiments.

## Evaluation and interpretation

Retain the 5% empirical-reference threshold, independent validation tables,
complete-policy eligibility and jointly eligible paired-J comparisons.
Missing or ineligible contexts remain unsuccessful in the full denominator.
Report per-seed results, all intermediate budgets, time to eligibility,
ineligible attempts, family/context breakdowns and peer/BO query modes.
Analysis replays all new Bayesian decisions from recorded observations before
opening the dense evaluator tables.

The report leads with D256/A, followed by D256/B and then D384 accuracy
controls. Separate flags report an improved context metric, preserved complete
eligibility and lower conditional paired cost; these are descriptive rather
than significance tests. Always retain eligible-pair counts. A lower mean cost
on a smaller eligible subset does not establish fleet-wide superiority.

The three seeds vary the existing context programmes and measurement noise on
fixed physical fleets. This is a matched development experiment informed by
earlier results, not a fresh independent-fleet confirmation. Read the two
physical conditions separately without choosing a favorable condition, seed,
budget or threshold after seeing results.

Peer publications and Gaussian model-message payloads remain separate. The
report adds a separate conservative one-off estimate for broadcasting the full
public basis to all nine clients, for both spectral D and E. It does not change
the original message accounting or count unused final-round training in the
final decision snapshot. Wider Gaussian messages grow quadratically; better
accuracy does not imply lower communication.

## Local execution and results

Run from the repository root in the same Python environment as the completed
D/E experiment. The earlier tracked reference results must be present.

```bat
git fetch origin
git switch experiment/spectral-online-bo
git pull --ff-only
python code/scripts/tracking_spectral.py preflight
python code/scripts/tracking_spectral.py run --workers 6
```

Six workers give one process per condition/seed job; each uses one BLAS thread.
Checkpoints are atomic after each round. Repeating the run command resumes
unfinished jobs and skips completed simulations. Do not run two processes
against the same result directory.

```bat
python code/scripts/tracking_spectral.py status
python code/scripts/tracking_spectral.py analyze
```

The offline report is `code/results/tracking_spectral_v1/report.html`.
`primary_comparison.csv`, `primary_per_seed.csv`, `secondary_comparison.csv`,
`accuracy_control_comparison.csv`, `benchmark_readout.csv` and the complete
paired tables preserve the declared comparison roles. Manifests, public bases,
round checkpoints, validation outputs and diagnostics stay in that directory.

After the full run, validate and stage only its result directory, then commit:

```bat
python code/scripts/tracking_spectral.py collect
git commit -m "Add matched spectral online BO results"
git push
```

`collect` replays and analyzes the complete campaign before staging; it rejects
missing results and smoke outputs. Completed frozen campaigns are untouched.

For a software check, use `python code/scripts/tracking_spectral.py smoke --workers 2`.
The smoke run has three clients, eight requests/client and one seed in both
conditions. It simulates all eight methods because reduced-fleet baselines
cannot reuse full-fleet traces: **384 charged requests**. Its separate
`tracking_spectral_v1_smoke/report.html` is software verification, not performance
evidence. Unit tests additionally check legacy-arm identity, own-arm isolation,
paid failures, posterior barriers, public-basis integrity and exact resume.
