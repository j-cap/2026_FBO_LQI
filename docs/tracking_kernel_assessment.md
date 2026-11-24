# Kernel representation assessment with D's formulation retained

24 November 2026. This follows the
[low-frequency diagnostic](analysis/low_frequency_rankings/report.md).
The [assessment configuration](../code/config/runs/tracking_kernel_assessment.json)
fixes candidate families, dimensions and accuracy targets before conditioning
on the saved histories. [Results](analysis/kernel_representation_assessment/report.md)
record the outcome without changing that configuration.

## What is held fixed

Keep D's Gaussian shared/local effects, sharing-fraction grid and prior, log-J
observation noise, fixed kernel hyperparameters, common peer rule, eligibility
estimator, acquisition and observation masks. Only the base kernel
representation changes. No new simulator episode or full BO policy is run.

For a base kernel approximation K_r=F_r F_r^T, D still uses

\[
 k_{\rho,r}((x,i),(x',j))=
 [\rho+(1-\rho)\mathbf1_{i=j}]k_r(x,x').
\]

The exact reference uses the original Matérn-5/2 kernel plus constant .25.
Every representation uses the same likelihood and Bayesian mixture over rho;
it is not permitted to select a different noise level or tune length scales.

## Public feature construction

The declared domain has 426 gain/context inputs. Its normalized input
coordinates and kernel are known before any rollout. Construct every basis
using only that public information; no cost label, plant parameter, vehicle
family, optimal gain or historical response profile is an input.

Keep the constant feature exact. Approximate the Matérn signal covariance:

- Random Fourier maps with 64, 256 and 512 terms, using the existing seed. The
  64-term map is exactly the original D map. Different sizes use the existing
  generator and are not claimed to be nested samples.
- Public spectral factors with 64, 128, 256 and 384 leading signal-kernel
  eigenpairs. A full 426-term factor is included as an exact-recovery check.
- Greedy pivoted-Cholesky factors with 64, 128, 256 and 384 terms, selecting the
  largest remaining prior diagonal residual, with deterministic index ties.

With signal kernel K_s=U Lambda U^T, the spectral factor is

\[
 F_r=[\sqrt{.25}\,\mathbf1,\;U_r\Lambda_r^{1/2}].
\]

There are r+1 coefficients per shared or local vector. The factor is defined
on the current public grid; no unseen continuous-input result is claimed.
All clients need the same coordinates, so a deployment must pin one common
basis and verify its identity. Raw feature hashes are recorded; independent
eigensolver runs on different platforms should not be assumed identical.

Spectral truncation and kernel-column approximations are established reduced
rank constructions; see Rasmussen and Williams, *Gaussian Processes for Machine
Learning*, Chapter 8, [primary text](https://gaussianprocess.org/gpml/chapters/RW.pdf).
The greedy pivot rule follows the maximum residual diagonal construction
described in the [official pivoted-Cholesky documentation](https://www.tensorflow.org/probability/api_docs/python/tfp/math/pivoted_cholesky)
and Harbrecht et al., 2012,
[primary publication record](https://edoc.unibas.ch/entities/publication/d4eba30a-9b48-45ca-a4cb-915bfe3a83ec).
These sources justify the constructions, not their accuracy in this task.

## Saved-history evaluation

Use both saved A and D histories, both physical conditions, all three existing
seeds and all six contexts. At each actual source BO decision, condition D's
model on that source fleet's eligible earlier-round observations and evaluate
the untried gains in the scheduled context. The target eligibility probabilities
are identical for all compared representations on a source prefix.

Only the source's original recorded observation advances the next history.
Alternative rankings never generate or supply an outcome. No dense calibration
or validation cost table is looked up for accuracy measurement or selection.
The exact posterior on the same paid observations is the numerical reference.

Measure posterior mean/std error, exact-score differences, top-rank agreement,
rank correlation and sharing-fraction agreement. The exact-score gap is
s_exact(a_approx)-min_a s_exact(a), conservatively maximized over numerical
approximate-score ties. It is not physical controller regret or a realized
improvement in calibration. The known g012 case is reported separately and is
not the selection criterion.

For efficient assessment, every covariance is conditioned in observation space.
This produces the same Gaussian model as feature-message inference for F F^T.
Selected widths are separately checked using the unchanged client/server
message algebra, over every client/action prediction at a fixed late prefix.
The original 65-coefficient model must also reproduce all factual D choices
within numerical score tolerance and its recorded communication totals.

## Accuracy targets and selection rule

The configuration fixes the following engineering approximation targets:

| Quantity | Target |
|---|---:|
| Relative public-kernel Frobenius error | <= .02 |
| Maximum relative prior diagonal error | <= .05 |
| Worst context posterior-mean RMSE | <= .05 log-J units |
| Worst context posterior-std RMSE | <= .05 log-J units |
| Worst history/condition group mean exact-score gap | <= .05 |
| Worst history/condition group 95th percentile exact-score gap | <= .10 |

The posterior tolerance equals the existing log-cost noise SD; it is not a
uniform error guarantee or the research success threshold. Context RMSEs pool
decisions within each history-source/condition/context group; groups cover all
three seeds. Seed-level results are retained separately. Top-choice agreement
is descriptive because nearly tied exact scores can produce different gains.

Choose the smallest tested feature width satisfying every target. Break equal
widths by the worst context mean RMSE, then the representation name. If none
passes, report that outcome without changing the targets. This is an empirical
numerical-accuracy check on existing histories, not blind fleet validation or
an independent controller-performance test.

## Cost assessment

Estimate D's same packed float64 information-matrix/vector payload at the final
decision, excluding unused final-round training. Its model payload grows
quadratically with coefficient width. Report the one-off conservative cost of
broadcasting the full deterministic public basis to all nine clients separately.
No transport overhead, compression or optimized message factorization is assumed.
The wide random maps are assessed as constructed; finite-grid rank compression
could change their implementation cost without changing the represented kernel.

Benchmark the actual feature-message fit and all-client/all-action prediction
at the predeclared round-23 prefix of seed 1101 in each condition, at widths
65, 257, 385 and the full factor. These are single-machine timings with one
BLAS thread, not distributed latency measurements. The faster observation-space
assessment timings are not substituted for feature-deployment timings.

## Reproduction

```bat
python code/scripts/assess_tracking_kernels.py --workers 6
python code/scripts/plot_tracking_kernel_assessment.py
```

The assessment caches completed source/condition/seed jobs under a signature
containing code, configuration and input hashes. Re-running resumes those jobs.
Decision-level numerical diagnostics are generated in the ignored figure
workspace; aggregate and focused checks are saved with the report. This step
does not prepare or launch a new controller-calibration campaign.
