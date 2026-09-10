# IEEE-style conference review — Round 1

**Manuscript:** *Fleet-Informed Bayesian Optimization for Data-Efficient Controller Calibration*  
**Review date:** 2026-09-10  
**Review style:** CCTA / ECC / IEEE IV conference review  
**Overall recommendation:** **Weak Reject / Borderline**  
**Score:** **4/10**  
**Reviewer confidence:** **4/5**

## Summary

The paper studies whether controller-calibration data accumulated by previously commissioned vehicles can reduce the number of closed-loop experiments required for a newly arriving vehicle. The controller is an LQI yaw-rate controller whose three calibration parameters are optimized with Bayesian optimization. Historical scalar calibration observations from 29 source clients are pooled with two local initialization experiments from a held-out client. The resulting Gaussian-process surrogate is used to select subsequent controller evaluations. The paper reports a strong held-out sample-efficiency result: the mean number of new-client experiments required to reach within 5% of a client-specific offline reference decreases from 6.63 under independent BO to 3.07 under global fleet warm start. The paper also investigates a dynamics-weighted variant. This variant does not improve over unweighted pooling. A separate landscape study shows pairwise calibration-landscape rank correlations above 0.95 despite appreciable vehicle-dynamics heterogeneity.

The application question is relevant and the held-out evaluation is considerably stronger than a typical proof-of-concept BO study. The most interesting scientific observation is that plant heterogeneity does not translate into comparable heterogeneity of the higher-level LQI calibration landscape. However, in the present draft the methodological novelty is limited, and several details that are essential to reproduce and interpret the reported sample-efficiency result are still missing. I would currently place the paper just below the acceptance threshold. A focused revision could move it into weak-accept territory for a control-oriented venue such as CCTA or ECC.

## Strengths

1. **Practically meaningful problem formulation.** The paper asks an important commissioning question: if previous fleet members have already undergone controller tuning, how much of that knowledge can be reused for a new system?

2. **Strong held-out experimental result.** The reduction from 6.63 to 3.07 new-client experiments is large, easy to interpret, and supported by 20 held-out fleet realizations and 120 held-out client-calibration problems.

3. **Appropriate separation between transferred information and local controller synthesis.** The work transfers calibration observations in a common parameter space rather than directly transferring feedback gains. Each new client recomputes its own LQI controller from its local identified model.

4. **Useful negative result.** The paper does not hide the failure of dynamics-aware weighting to outperform simple pooling. The repeated null result is scientifically useful and is connected to the observed similarity of the calibration landscapes.

5. **Good attempt at mechanistic explanation.** The landscape-similarity study moves the paper beyond a purely empirical statement that warm starting works. The observation that dynamics heterogeneity and calibration-objective heterogeneity can be substantially different is potentially the most generalizable insight in the paper.

6. **Clearer control-oriented framing than a generic federated-learning paper.** The current terminology and the use of an offline reference are better aligned with the intended control audience.

## Major comments

### 1. The novelty relative to transfer Bayesian optimization is not yet sufficiently sharp

The proposed global fleet warm start is algorithmically simple: historical normalized observations are pooled with the new client's local observations and a standard GP with expected improvement is fitted. This is close in spirit to existing transfer-learning BO, warm-start BO, multi-task BO, and collaborative BO methods already cited in the manuscript. The paper currently states the related literature but does not clearly explain what is technically new relative to it.

The authors should explicitly separate **algorithmic novelty** from **control-domain and scientific novelty**. In my view, the most defensible contribution is not a new BO algorithm. It is the combination of:

- a fleet-commissioning formulation in which previous closed-loop controller-calibration experiments become reusable assets,
- client-specific model-based controller synthesis despite a shared BO calibration space,
- a carefully held-out sample-efficiency evaluation, and
- the finding that plant heterogeneity can coexist with highly invariant controller-calibration landscapes.

The introduction should state this distinction directly. A short related-work subsection would help. In particular, the paper should explain why simply applying an existing transfer-BO method does not already answer the investigated control question.

### 2. The offline reference and the definition of \(N_{5\%}\) need a complete specification

The primary metric depends on \(J_i^{\mathrm{ref}}\), but the manuscript currently describes it only as a "separate, dense evaluation procedure." This is not sufficient because essentially every headline result is measured relative to this quantity.

The paper should state exactly:

- how candidate calibration vectors for the offline reference are generated,
- how many candidate vectors are evaluated,
- whether a local refinement is used after the global design,
- which calibration and test episodes are used,
- whether the reference search uses the same stochastic realizations as BO,
- whether the reference is computed from the true nonlinear plant,
- and whether any BO trajectory can obtain a cost below \(J_i^{\mathrm{ref}}\).

The manuscript must also define what happens if the 5% criterion is **not reached within the 12-experiment budget**. This is important because the primary variable is right-censored by the experimental horizon. If unsuccessful clients are assigned a fixed value such as 13 evaluations, this must be stated explicitly and the interpretation of the resulting mean and Wilcoxon test must be discussed. If they are excluded, the current comparison would be biased. This point should be resolved before submission.

### 3. The Bayesian-optimization implementation is not reproducible from the manuscript

The BO section currently gives only the pooled dataset and the expected-improvement acquisition rule. Important implementation details are absent. At minimum, the paper should provide:

- the search bounds \(\Xi\) for all three components of \(\boldsymbol\xi\),
- the GP covariance kernel,
- the mean function,
- input and output standardization details,
- nominal observation-noise values,
- whether kernel hyperparameters are fixed or optimized,
- the hyperparameter optimizer and number of restarts if optimized,
- the precise expected-improvement definition for a minimization problem,
- how the EI maximization itself is performed,
- how many acquisition candidates or restarts are used,
- the numerical infeasibility penalty,
- the local initialization construction,
- the baseline calibration \(\boldsymbol\xi^{\mathrm{base}}\),
- and all numerical constants in the dynamics-weighted ablation, including \(\alpha_0\), \(\epsilon\), and \(\epsilon_s\).

Without these details, the reported sample-efficiency numbers cannot be reproduced. For an IEEE control conference, this is a significant weakness.

### 4. The vehicle benchmark and identification procedure remain underspecified

The addition of the linear bicycle-model equations and the parameter table is helpful, but the nonlinear evaluation plant is still not defined completely enough to reproduce. Equation (tire nonlinearity) alone does not specify the nonlinear vehicle model. The manuscript should also define the front and rear slip angles, the axle normal loads used in the saturation law, and the state equations into which the nonlinear tire forces enter.

Similarly, the statement that the generator uses a "variability setting of 0.025" is too vague. The reader needs to know which parameters are perturbed, the distribution used, whether perturbations are independent, and whether any rejection or clipping is applied.

The local system-identification step is also important because it produces \(\hat{\boldsymbol\theta}_i\) and \(\hat{\boldsymbol\Sigma}_i\), which are later used for the dynamics-weighted ablation. The manuscript should specify the parameter vector, excitation data, data length, RLS initialization/forgetting assumptions, and covariance construction. If the identification method is inherited directly from the authors' earlier IFAC paper, the present paper can remain concise, but the exact reused setup still needs to be stated.

### 5. The strong warm-start result needs a more explicit fairness and amortization discussion

The comparison is intentionally asymmetric. Independent BO begins with only two new-client measurements, while fleet-informed BO additionally receives 725 historical source observations. This is a valid operational comparison if these 725 measurements are interpreted as already-paid-for information from previous commissioning. The paper now explains this idea verbally, which is good, but the experimental section should make the accounting completely explicit.

I recommend introducing two quantities:

\[
N_i^{\mathrm{new}}
\]
for experiments newly executed on client \(i\), and
\[
N^{\mathrm{hist}}
\]
for source observations already available before that client arrives. The primary result should then be described as a reduction in **incremental commissioning experiments**, not as an overall reduction in total experimental effort.

The existing nine-peer coverage experiment is useful and should be presented as evidence that the effect is not solely tied to a very large 29-client source pool. If additional source-pool-size results already exist, they would strengthen the argument further. I do not consider a new experiment strictly necessary if the current scope is stated carefully.

### 6. The landscape-similarity claim needs a little more methodological detail and restraint

The finding \(\rho_{ij}^{J}\ge0.95\) for all evaluated pairs is striking and central to the paper's explanation of the null dynamics-weighting result. The current 32-point common Sobol design is plausible, but several details should be clarified:

- Are the costs evaluated on identical deterministic/stochastic episode banks for every client?
- How are infeasible points handled in the rank correlation?
- Are the 32 points sampled once globally or once per fleet realization?
- How many clients and pairwise comparisons enter the figure?
- Is the reported minimum correlation computed over all fleet realizations or over a selected subset?

The broader statement "plant heterogeneity need not imply optimization-task heterogeneity" is reasonable as an interpretation of this benchmark. It should not be presented as a general property of LQI calibration without qualification, since only one controller structure, three-dimensional search space, operating speed, and maneuver family are considered.

### 7. The statistical treatment of the primary metric should be explained more carefully

Using fleet realization as the replication unit is a sound choice and avoids treating all 120 clients as independent samples. This is a strength. However, the manuscript should specify the bootstrap procedure, including the number of bootstrap resamples and whether the reported interval is percentile, basic, or BCa. The use of a Wilcoxon signed-rank test on realization-level mean differences should also be stated explicitly in the methods rather than only implied.

If \(N_{5\%}\) is censored for clients that fail within the budget, the statistical interpretation becomes especially important. The current draft does not provide enough information to judge whether the reported mean difference is a true mean experiment count or a conservative budget-censored quantity.

## Minor comments

1. **Figure numbering and file naming are currently inconsistent.** The lane-change reference figure appears before the landscape figure in the LaTeX source, so it will receive the lower figure number even though its file is named `fig6_lane_change_reference.pdf`. The final source package should use filenames that match manuscript order or neutral descriptive filenames.

2. **Figure 1 is still a placeholder.** This must of course be replaced before submission. The proposed schematic is useful because the distinction between historical source data and new-client experiments is central to the paper.

3. The manuscript should define \(\hat{\mathbf C}_i\) before it appears in the LQI synthesis expression. Only \(\mathbf A\) and \(\mathbf B\) are introduced in the preceding identified-model equation.

4. The exact calibration dimensions should be named. The reader currently sees a three-dimensional vector \(\boldsymbol\xi\) but is not explicitly told which augmented-state penalties \(\xi_1,\xi_2,\xi_3\) correspond to.

5. The fixed state-scaling values are now given, but their origin should be stated. Are these nominal standard deviations, physical admissible ranges, manually selected scales, or values derived from baseline trajectories?

6. The lane-change reference plot is useful, but the reference waveform should also be defined analytically or algorithmically so that the benchmark is reproducible without reverse-engineering the figure.

7. The manuscript should define the exact yaw-rate RMSE expression and the episode duration over which it is computed.

8. The sentence stating that family labels are not provided to BO is useful. The paper should also make clear that the dynamics-weighted method uses no family labels, only identified-model information.

9. The current source still contains broad `\color{blue}` markup and relies on `latexmkrc` to render it black. This is convenient for internal revision, but the final IEEE source package should remove these global color commands rather than depend on a local latexmk configuration.

10. The bibliography entry for the authors' IFAC World Congress paper still contains placeholder publication details. These should be updated before submission if final bibliographic information is available.

11. Consider using "vehicle" or "system" more often than "client" in application-facing paragraphs. "Client" is natural in federated-learning language, but the proposed method is now framed as fleet-informed transfer rather than as a privacy-preserving federated algorithm.

12. The paper currently has no explicit Related Work section. A compact subsection would improve novelty positioning against transfer BO and controller-tuning BO without requiring much additional space.

13. The title is clear, but "Data-Efficient" is supported only with respect to **new-client** experiments, not total fleet-wide data generation. This is acceptable if the amortized-data interpretation is made explicit throughout.

14. The conclusion is appropriately cautious about the dynamics-weighting null result. I would retain this tone and avoid stronger claims such as dynamics information being generally unnecessary for controller calibration.

## Assessment by category

| Category | Score | Assessment |
|---|---:|---|
| Relevance to control conference | 4/5 | Strong fit for CCTA and good fit for ECC. Also relevant to IV through the vehicle application. |
| Technical correctness | 3/5 | No obvious conceptual error, but key implementation and metric details are currently missing. |
| Novelty | 2.5/5 | Limited algorithmic novelty. Stronger novelty lies in the fleet-calibration formulation and heterogeneity finding. |
| Experimental quality | 4/5 | Strong held-out evaluation design and useful negative ablation. |
| Reproducibility | 2/5 | Important BO, reference-generation, identification, and nonlinear-plant details are absent. |
| Clarity | 3.5/5 | Overall story is clear. Several quantities and experimental conventions still need exact definitions. |
| Significance | 3.5/5 | The 54% reduction is practically meaningful if the accounting and reference metric are fully specified. |

## Recommendation rationale

I currently recommend **Weak Reject / Borderline (4/10)**. The paper contains a strong and practically relevant empirical result, and the dynamics-versus-calibration-heterogeneity observation is interesting. The main reasons it falls just below my acceptance threshold are the limited methodological novelty and the current lack of sufficient detail to reproduce or fully interpret the primary metric.

I do **not** think the paper requires another broad algorithm-development phase. Most of the gap can be closed through careful manuscript work. The highest-priority items are to fully define the offline reference and censored \(N_{5\%}\), document the BO implementation, complete the nonlinear benchmark and identification description, and sharpen the novelty positioning relative to transfer BO. If these issues are addressed convincingly, I would expect the manuscript to become competitive for a control-focused IEEE conference.

## Highest-priority revision sequence

1. Fully specify \(J_i^{\mathrm{ref}}\), \(N_{5\%}\), and budget censoring.
2. Add all missing BO implementation details and the calibration search bounds.
3. Complete the vehicle-model, variability, task, and identification description.
4. Rewrite the novelty/related-work positioning around the fleet-commissioning formulation and the dynamics-versus-calibration heterogeneity insight.
5. Make the historical-data accounting explicit as amortized prior experimentation rather than free data.
6. Clarify the landscape-similarity protocol and soften any claims that extend beyond the investigated benchmark.
7. Finish Fig. 1, clean figure numbering/file naming, and remove draft-only color infrastructure before submission.
