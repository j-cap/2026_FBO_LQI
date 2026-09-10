# Response to IEEE-style conference review — Round 1

**Manuscript:** *Fleet-Informed Bayesian Optimization for Data-Efficient Controller Calibration*  
**Revision date:** 2026-09-10

This document records how the Round-1 review was addressed in `main.tex`. All newly inserted or materially rewritten manuscript text is marked in red through `\rev{...}` for the next author pass.

## Major comments

### 1. Novelty relative to transfer Bayesian optimization
**Action:** Addressed.

- Added a direct statement that the contribution is not a new BO acquisition rule.
- Added a compact `Related work and positioning` subsection.
- Positioned the contribution around fleet commissioning, client-specific model-based controller synthesis, held-out incremental sample efficiency, and the distinction between dynamics heterogeneity and calibration-landscape heterogeneity.

### 2. Offline reference and definition of N_5%
**Action:** Substantially addressed.

- Added a dedicated `Offline reference and threshold metric` subsection.
- Defined the offline reference as a separate fixed Sobol scan over the calibration box, evaluated on the nonlinear plant and re-scored on a disjoint test bank.
- Clarified that the reference is reporting-only and cannot influence BO.
- Clarified that BO may outperform the finite offline reference.
- Defined the treatment of unsuccessful runs at the 12-experiment horizon by budget imputation rather than exclusion.

### 3. Bayesian-optimization reproducibility
**Action:** Partly addressed, with explicit verification items left in the manuscript.

- Added the common search box `[-3,3]^3`.
- Defined two-point Sobol initialization.
- Added expected-improvement citation and minimization context.
- Added the source-weight floor of 0.02.
- Added a dedicated BO implementation table.

The following values still require direct verification against the archived experiment code/configuration before submission:

- exact GP kernel,
- base GP noise `alpha_0`,
- EI maximization candidate/restart rule,
- normalized infeasibility penalty,
- fixed baseline calibration vector.

These are intentionally left as red verification items rather than reconstructed from memory.

### 4. Vehicle benchmark and identification
**Action:** Partly addressed, with exact generator/RLS configuration flagged for verification.

- Added nonlinear front/rear slip-angle equations.
- Added static axle-load equations.
- Added the nonlinear beta/yaw-rate state equations.
- Added an explicit identification subsection and clarified that the identification maneuver is common across systems.

Still to verify from the archived implementation:

- exact perturbed parameter list and sampling distribution for the 0.025 variability setting,
- clipping/rejection rule,
- exact identified parameter vector,
- identification data length/excitation,
- RLS initialization and forgetting factor,
- covariance convention.

### 5. Fairness and amortization
**Action:** Addressed.

- Introduced `N_i^{new}` for incremental experiments on the arriving system.
- Introduced `N_i^{hist}` for already-available source observations.
- Made the 725 historical observations explicit.
- Clarified that the 54% headline result is a reduction in incremental commissioning experiments, not total fleet-wide experimental effort.
- Added the nine-source coverage study as a robustness analysis.

### 6. Landscape-similarity methodology and claim restraint
**Action:** Mostly addressed.

- Clarified use of one common 32-point Sobol design per fleet realization.
- Clarified common episode-bank construction.
- Clarified treatment of infeasible points using the same penalty convention.
- Added an explicit limitation statement that the result applies to the investigated fixed-speed, three-parameter LQI benchmark.

Still to verify:

- exact number of fleet realizations, systems, and pairwise comparisons entering the final landscape figure.

### 7. Statistical treatment
**Action:** Mostly addressed.

- Stated that fleet realization is the replication unit.
- Stated that metrics are averaged over the six held-out systems before pairwise method differences are formed.
- Stated that the Wilcoxon signed-rank test is applied to the 20 paired realization-level mean differences.
- Described the bootstrap as percentile resampling of the paired realization-level differences.

Still to verify:

- exact number of bootstrap resamples used by the analysis script.

## Minor comments

1. **Figure numbering/file naming:** addressed by directly using the numbered PDF filenames currently present in `figures/`.
2. **Figure 1 placeholder:** still open and unchanged.
3. **Definition of C_i:** addressed with `C_i=[0 1]` for yaw-rate output.
4. **Calibration dimensions:** addressed. `xi_1`, `xi_2`, `xi_3` correspond to beta, yaw rate, and integral tracking error.
5. **Origin of state scales:** added as fixed values derived from representative nominal closed-loop trajectories. This statement should be checked once against the original setup documentation.
6. **Lane-change waveform:** still needs exact analytic/algorithmic definition and total episode duration. A red verification item is present in the manuscript.
7. **Yaw-rate RMSE:** explicit equation added. Episode duration remains to verify.
8. **Family labels:** clarified that dynamics weighting uses identified-model information only and never family labels.
9. **Broad color markup:** removed from the new revision. New review edits use local `\rev{...}` red markup only.
10. **IFAC bibliographic details:** still open in `references.bib` if final publication details are not yet available.
11. **Client terminology:** application-facing prose now uses `system` more often, with `source system` / `new system` terminology.
12. **Related Work section:** added.
13. **Meaning of data-efficient:** clarified throughout as new-system / incremental commissioning efficiency.
14. **Cautious conclusion:** retained and strengthened with benchmark-specific qualification.

## Remaining verification checklist before the next reviewer pass

The revision resolves the conceptual and presentation issues, but several implementation constants must still be copied from the archived experiment code/configuration rather than inferred:

1. GP kernel and hyperparameter treatment.
2. Base GP noise `alpha_0`.
3. EI maximization implementation.
4. Normalized infeasibility penalty.
5. Baseline calibration vector.
6. Fleet-generator perturbation distribution and perturbed parameter list.
7. Exact RLS/identification setup.
8. Lane-change waveform definition and episode duration.
9. Exact landscape-study comparison count.
10. Bootstrap resample count.

These unresolved items are explicitly marked in red in the manuscript so they are easy to locate during the author review.
