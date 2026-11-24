# Joint shared/local cost learning: A first, then B

24 November 2026. This implements the statistical reference proposed in the
[heterogeneous modeling review](heterogeneous_cost_model_review.md). The
[condition-dependent findings](fedbo_consolidation.md) remain frozen.

Completed results and verification are in the
[joint-model analysis](analysis/joint_shared_local_cost_model/report.md).
The protocol below was fixed before those results.

## Question and comparison order

Can a jointly inferred shared/local cost surface improve the existing practical
gain-sharing method A within the same 24 tests per vehicle? Improvement over B
is the second objective. Beating the previous federated model C alone does not
meet the main objective.

The primary comparison is **D minus A in near-reference context success at the
final 216-query fleet budget**, separately for homogeneous and heterogeneous
conditions. The secondary comparison is D minus B. Retain the existing 5%
reference threshold, complete-policy counts, paired mean-J denominators,
per-seed results, intermediate budgets and family/context breakdowns.

| Method | Cost information beyond peer verification | Role |
|---|---|---|
| A: `peer_gp_bo` | Own cost observations, local GP | Primary practical benchmark |
| B: `peer_local_cost_bo` | Own fitted 65-coefficient mean and residual GP | Secondary benchmark |
| C: `peer_federated_cost_bo` | FedAvg mean and target-only residual GP | Previous architecture control |
| D: `peer_hierarchical_bayes_bo` | Joint shared/local Bayesian cost model | New method |
| E: `peer_local_bayes_bo` | Independent Bayesian cost model with the same marginal prior as D | Matched statistical sharing control |

Every method retains its own queries, catalogue and paid local verification.
D versus E isolates the additional cost-model dependence across vehicles in
this model class. D versus A/B/C changes the statistical model and inference,
so those comparisons are not pure sharing ablations. E still participates in
the common peer-gain channel; only its cost model is independent across clients.

The report leads with D/A, then D/B. `benchmark_readout.csv` reports separate
flags for an improved primary metric, preserved complete eligibility and lower
conditional paired cost, together with positive/negative/tied seed counts.
These flags are descriptive, not significance tests or a replacement composite
score. A lower cost on a reduced eligible subset does not establish fleet-wide
superiority. Do not select a favorable condition, seed, threshold or budget
after seeing the results.

## Statistical model and public prior

Inputs remain x = [three normalized log-gain coordinates, normalized speed,
normalized log frequency]. Use exactly the same public, data-independent
64 Matérn random Fourier features and intercept as B/C; no learned dense-bank
basis, family assignment, mass or tire parameter is an input.

For phi(x) in R^65, define the fixed marginal coefficient covariance

\[
 S=\operatorname{diag}(0.25,1,\ldots,1).
\]

These entries use the existing GP's initial constant variance and squared
amplitude. For sharing fraction rho, the model is

\[
 y_{ik}=\log J_{ik}=\phi(x_{ik})^\top(w_g+u_i)+\epsilon_{ik},
 \quad w_g\sim\mathcal N(0,\rho S),\quad
 u_i\sim\mathcal N(0,(1-\rho)S),
 \quad \epsilon_{ik}\sim\mathcal N(0,0.05^2).
\]

All priors are mutually independent before conditioning. The same vehicle's
marginal covariance is phi(x)^T S phi(x') for every rho; cross-vehicle
covariance is rho times that quantity. Local effects can depend on gains and
context. They are not scalar cost offsets. At rho=0 the vehicle cost models are
independent; at rho=1 they share one cost function.

D gives equal prior probability to the public grid
`[0, .25, .5, .75, .95, 1]`. After each paid round it computes the marginal
likelihood of all its eligible observations under each component, and updates
the component probabilities. It recomputes the posterior from the prior and
the complete accumulated likelihood: the previous posterior is not multiplied
by the same old observations again. Sharing strength is inferred online and
the same prior is used in both physical conditions.

Predictions integrate rho by their mixture mean and **total variance**,
including both within-component uncertainty and between-component mean
disagreement. Each component conditions each cost observation once. Shared and
local posterior dependence is included in the prediction.

E fixes rho=0, retaining the same marginal prior, feature map, observation
noise, prediction and decision formulas. Its target prediction uses only that
target's eligible observations. It makes no cost-model messages. With one
vehicle alone, D and E have identical predictions and that vehicle cannot
identify rho; this is checked analytically in the tests.

The feature length scales, amplitude, constant variance and observation noise
are fixed at the public settings inherited from the completed experiment.
Only rho's probabilities are learned in this first implementation. The old
ridge `.02` is not a second penalty on this Bayesian posterior. No old fitted
mean or post-hoc residual GP is appended to D/E. Their joint model replaces
that two-stage cost fit. A/B/C keep their original GP adaptation unchanged.
The finite feature approximation and fixed covariance settings can themselves
limit performance; D is not guaranteed to beat A.

## Federated inference and information contract

This first implementation uses **exact federated Gaussian information
aggregation, not FedAvg**. It tests the proposed statistical model without
introducing a gradient-convergence error. It does not establish that an
ordinary model-vector FedAvg implementation would have the same posterior or
policy. No gradients are needed for this conjugate reference.

Let F_i contain phi(x)^T S^(1/2) at client i's eligible queried inputs, and let
y_i contain its log costs. After integrating the local coefficients,

\[
 R_i=\sigma^2 I+(1-\rho)F_iF_i^\top,
 \qquad
 P_i=\rho F_i^\top R_i^{-1}F_i,
 \qquad
 h_i=\sqrt{\rho}F_i^\top R_i^{-1}y_i.
\]

Each client sends a packed symmetric P_i, h_i, and the three scalars
y_i^T R_i^(-1)y_i, log det R_i and its eligible record count. The server forms

\[
 \Sigma_g=(I+\textstyle\sum_iP_i)^{-1},\qquad
 \mu_g=\Sigma_g\textstyle\sum_i h_i.
\]

Here mu_g and Sigma_g refer to the whitened shared coefficient, with a standard
normal prior. The server broadcasts its posterior mean/covariance and each
component's log evidence. Clients form their own conditional local prediction
and combine component weights. Cholesky solves are used throughout; `1e-9`
diagonal jitter matches the existing numerical convention.

For clarity, at a test input with row F_* define

\[
 C_i=(1-\rho)F_*F_i^\top,\qquad
 H_i=\sqrt{\rho}F_*-C_iR_i^{-1}\sqrt{\rho}F_i.
\]

Then the latent cost prediction is

\[
 \mu_i=C_iR_i^{-1}y_i+H_i\mu_g,
 \qquad
 v_i=(1-\rho)F_*F_*^\top-C_iR_i^{-1}C_i^\top
       +H_i\Sigma_gH_i^\top.
\]

The last term retains the shared/local posterior dependence; simply adding
an independent shared-mean variance to the old residual GP would not give this
expression. Independent tests compare messages, evidence and predictions to
direct dense GP conditioning on input/client pairs, including rho=0 and rho=1.

The server aggregation function accepts information messages, not local
histories, raw cost arrays, family labels or plant parameters. The code runs
all client/server roles in one simulation process per fleet job; it is not a
network deployment. Information messages can disclose information about data,
especially in small samples. No formal privacy claim is made. The common peer
catalogue already publishes tested gains and donor costs in every method.

Payload accounting includes per-component packed matrices, vectors and evidence
scalars for D, and the original vector iterations for C. It separates the
model channel from peer publications. It does not assume that D communicates
less. End-of-final-round training is recorded but excluded from the final
decision snapshot, since no later query uses it.

## Protocol and efficient execution

Retain the frozen nonlinear plants, controller, objective, six contexts, 71
candidate gain vectors, nine clients and both physical conditions. All methods
start with one factory-gain test and use the same prior-round peer-first rule.
Only when no untried eligible peer incumbent is available does cost BO choose
the next gain. The existing feasibility-penalized LCB uses its unchanged
coefficients `(2, 1, 1e-6)`, and eligibility learning remains target-only.
The LCB uses the mixture's mean and standard deviation; it is not a certified
quantile or confidence bound for a Gaussian mixture.

Use seeds 1100/1101/1102, 24 tests/client, and snapshots at fleet budgets
9/18/27/54/108/216. Context programmes and client/query noise seeds are the same
counterfactual pairings as A/B/C. Failed and unresolved tests consume budget.
Recommendations must be gains actually found eligible locally. New discoveries
can change the arm's later peer catalogue and thereby its full-policy outcome.

The full run simulates only D and E: **2,592 newly charged requests** across
six fleet jobs. The completed A/B/C traces contribute 3,888 existing charged
requests to evaluation. Their completed source/result signatures, fleet and
factory observations are checked. Old results are not accessible in worker
payloads, cannot train D/E and do not provide a free prior or catalogue.
They are reused for comparison, not rerun or altered.

Simulator batching caches identical requested counterfactual episodes within
a job, while each arm pays for every requested rollout. Checkpoints are saved
atomically after each round. Re-running the same command resumes missing work.
Use up to six workers, one per condition/seed job; BLAS threads are limited to
one per worker. Do not run two processes against the same result directory.

The smoke run uses three clients and eight tests/client. It simulates all five
methods because the reduced fleet cannot reuse full-fleet baseline traces.
Its 240 requests verify simulation, multiprocessing, snapshots, report
generation and query replay. Smoke scores are not evidence of superiority.

All new-arm choices are replayed against their recorded observations during
analysis before evaluator tables are opened. Dense calibration and validation
tables are evaluator-only. The original empirical reference, held-out
validation, missing-context denominator and paired-J convention are reused.

## Local commands

From the repository root, in the existing `paper_FBO_LQI` environment:

```bat
git fetch origin
git switch experiment/joint-shared-local-bo
git pull --ff-only
python code/scripts/tracking_hierarchical.py preflight
python code/scripts/tracking_hierarchical.py smoke --workers 2
python code/scripts/tracking_hierarchical.py run --workers 6
```

The full command resumes automatically after interruption. Inspect progress:

```bat
python code/scripts/tracking_hierarchical.py status
```

The full report is `code/results/tracking_hierarchical_v1/report.html`.
Start with `primary_comparison.csv` and `primary_per_seed.csv` for A, followed
by `secondary_comparison.csv` for B and `benchmark_readout.csv`. Read
`sharing_strength.csv` and the D/E contrast after the primary outcomes.

After the full run, validate and stage only this campaign, then commit/push:

```bat
python code/scripts/tracking_hierarchical.py collect
git commit -m "Add joint shared-local BO comparison results"
git push
```

Collection rejects smoke or incomplete results. It recomputes all evaluation
tables and replays all new decisions before staging the result directory.
The frozen previous experiments and manuscript are unchanged.
