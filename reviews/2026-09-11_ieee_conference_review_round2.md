# IEEE-style conference review — Round 2

**Manuscript:** *Fleet-Informed Bayesian Optimization for Data-Efficient Controller Calibration*  
**Review date:** 2026-09-11  
**Review style:** CCTA / ECC / IEEE IV conference review  
**Overall recommendation:** **Borderline / Weak Reject**  
**Score:** **5/10**  
**Reviewer confidence:** **4/5**

## Summary

The revised manuscript studies reuse of historical controller-calibration data from previously commissioned fleet members to reduce the incremental number of closed-loop experiments needed for a newly arriving heterogeneous vehicle. The controller is a three-parameter LQI yaw-rate controller optimized with Bayesian optimization (BO). Historical normalized calibration observations from source systems are pooled with a small number of local observations, while each candidate feedback gain is synthesized from the new system's own identified model and tested on its nonlinear plant.

The revised paper is materially stronger than the previous version. The novelty positioning is now much clearer, the distinction between historical and incremental experimental cost is explicit, the nonlinear vehicle model and statistical replication structure are better documented, and the new overview figure improves the control-oriented narrative. The headline experimental result remains compelling: global fleet warm start reduces the reported mean number of new-system experiments required to reach the 5% reference criterion from 6.63 to 3.07. The dynamics-weighted variant again shows no detectable improvement over global pooling, and the paper links this result to high cross-system calibration-landscape rank correlation.

I now view the manuscript as close to the acceptance boundary for a control-oriented conference. However, two methodological issues around the primary performance metric and landscape-similarity analysis require resolution before I would recommend acceptance. In addition, several implementation details are still explicitly marked for verification in the manuscript. These are no longer cosmetic omissions because they affect reproducibility and interpretation of the central claims.

## Progress since Round 1

The authors have addressed several major concerns from the first review:

1. **Novelty positioning is substantially improved.** The paper now explicitly states that the contribution is not a new acquisition rule and frames the novelty around fleet commissioning, cross-system reuse of calibration experience, client/system-local controller synthesis, and the relationship between plant and calibration heterogeneity.

2. **Historical-data accounting is clearer.** The distinction between incremental new-system experiments, $N_i^{\rm new}$, and pre-existing historical observations, $N_i^{\rm hist}$, is now explicit. This makes the 54% reduction much easier to interpret correctly.

3. **The vehicle benchmark is better specified.** The nonlinear tire model, slip angles, static axle loads, and nonlinear lateral state equations are now included.

4. **The statistical hierarchy is clearer.** Fleet realization is explicitly identified as the replication unit, and paired realization-level differences are used for confidence intervals and Wilcoxon tests.

5. **The scope of the heterogeneity claim is more cautious.** The discussion now qualifies the conclusion to the investigated fixed-speed, three-parameter LQI benchmark.

6. **The overview figure is now present.** The paper no longer begins with a placeholder and the commissioning workflow is easier to understand.

These changes move the paper from a clear weak reject toward the borderline region.

## Major comments

### 1. The handling of clients that do not reach the 5% criterion is currently not statistically defensible

The manuscript states that if the threshold is not reached within the 12-experiment budget, the paper assigns $N_{i,5\%}=12$ and calls this a conservative, right-censored measure. I do not think this interpretation is correct.

If a system has not reached the criterion by experiment 12, its true hitting time is **strictly greater than 12** or unknown. Assigning the value 12 therefore understates the hitting time. It also makes a failed system numerically indistinguishable from a system that succeeds exactly at the final experiment. This biases the reported mean downward and is not conservative in the usual sense.

This issue directly affects the headline metric and its paired statistical comparison. The authors should resolve it before submission. Reasonable options include:

- treat the observation as right-censored and use a survival/time-to-event analysis,
- define a finite-budget score with failure encoded as $B+1=13$ and state explicitly that it is a penalized finite-budget metric rather than the true hitting time,
- or make the fixed-budget success probabilities the primary metric and use $N_{5\%}$ only for successful trajectories.

The existing success probabilities at 3 and 5 experiments are already informative and robust to this issue. Whichever convention was actually used in the analysis scripts should be verified first. If the reported 6.63 and 3.07 values were computed with a different convention than the current manuscript states, the manuscript should be corrected rather than the analysis retrofitted to the draft.

### 2. The offline reference appears too weak for the role it plays in the primary metric

The revised text states that $J_i^{\rm ref}$ is obtained from a 32-point Sobol scan over the three-dimensional calibration box. This is a modest design for defining the denominator and target value used in the main 5% criterion. The manuscript also explicitly acknowledges that BO can outperform this reference.

This creates two concerns. First, reaching within 5% of a relatively coarse reference may be substantially easier than reaching within 5% of a high-quality client-specific optimum. Second, if the reference quality varies by system, the threshold difficulty may vary for reasons unrelated to the compared BO methods.

I recommend either strengthening the reference computation or demonstrating that the headline conclusion is insensitive to reference quality. A stronger reference could use a substantially denser common low-discrepancy grid, a dense grid followed by local refinement, or the best value from a large union of independent optimization runs that is never exposed to the compared methods. If the archived experiments already used such a stronger reference and the current 32-point description was introduced during revision, the manuscript should restore the actual procedure.

At minimum, report how often the compared BO trajectories beat $J_i^{\rm ref}$ and by how much. If this occurs frequently, the reference should not be described as a near-optimal benchmark without qualification.

### 3. The calibration-landscape correlation may be inflated by tied infeasibility penalties

The manuscript states that infeasible points are assigned the same fixed normalized penalty before Spearman rank correlations are computed. This can artificially increase pairwise rank correlation if the same or similar subsets of the 32 Sobol points are infeasible across systems. A block of tied worst-ranked points can make the landscapes appear more rank-invariant than the feasible objective surfaces actually are.

Because the statement $\rho_{ij}^{J}\ge0.95$ is central to the mechanistic explanation for why dynamics weighting is unnecessary, this deserves a dedicated robustness check. I suggest reporting:

- the fraction of feasible Sobol points per system,
- the fraction of jointly feasible points per system pair,
- Spearman correlation computed only on jointly feasible points when enough points are available,
- and, separately, similarity of the feasibility pattern itself.

If the high correlations persist on jointly feasible points, the mechanistic claim becomes much stronger. If not, the paper should distinguish similarity of feasibility regions from similarity of performance rankings.

### 4. Several reproducibility-critical values are still unresolved in the paper

The revised manuscript is much more explicit about what is missing, but the fact that the missing values are marked in red does not solve the reproducibility issue. Before submission, the following must be recovered from the archived experiment configuration or code:

- GP kernel and kernel parameterization,
- base observation noise $\alpha_0$,
- regularization $\epsilon$ used in the dynamics distance,
- EI maximization/candidate rule,
- normalized infeasibility penalty,
- baseline calibration vector $\boldsymbol\xi^{\rm base}$,
- exact set and distribution of the 0.025 vehicle perturbations,
- identification parameter vector, excitation, record length, RLS initialization and covariance convention,
- lane-change reference construction and total episode duration,
- exact landscape-study fleet/client/pair counts,
- and bootstrap resample count.

These items should not survive into the camera-ready draft. The paper is now sufficiently mature that unresolved numerical placeholders become the dominant reproducibility weakness.

### 5. The paper should avoid interpreting a non-significant difference as formal equivalence

The manuscript currently describes global pooling and dynamics-weighted warm start as "statistically indistinguishable" and concludes that dynamics weighting provides no measurable benefit. The latter phrasing is reasonable. The former can be read as an equivalence claim, but a non-significant Wilcoxon test is not an equivalence test.

The reported global-versus-weighted confidence interval is already very narrow, which is useful. I suggest wording the conclusion as "no detectable improvement" or "the observed difference is negligible at the resolution of this study." If the authors want to make a formal equivalence statement, they should define a practically meaningful equivalence margin a priori and perform an equivalence analysis.

### 6. Novelty is clearer, but the empirical baseline set remains somewhat narrow

The revised positioning correctly states that this is not a new BO algorithm. That is a good decision. However, once the paper is framed as a study of transferability in controller calibration, a reviewer may still ask why only independent BO, full pooling, and one dynamics-weighted heuristic are compared, while the paper cites transfer-GP and transfer-BO methods.

I do not consider a new broad experimental campaign mandatory for a control conference, especially because the central scientific result is that the task landscapes are already highly invariant. Still, the authors should make one of two choices explicit:

1. add one established transfer/multi-task BO baseline, or
2. state clearly that the paper is deliberately testing the minimal hypothesis "does reusable fleet history help, and does dynamics-based source weighting help beyond global pooling?" rather than benchmarking the transfer-BO literature.

The second option is acceptable if the contribution statement and title remain correspondingly modest.

### 7. The stochastic episode model is still unclear

The paper repeatedly refers to three stochastic episodes per lane-change maneuver, but it does not define what is stochastic. The generic formulation includes process and measurement disturbances, yet the experimental section does not state the distributions, amplitudes, initial-condition variation, sensor noise, or other episode-level randomization used in the nonlinear simulation.

This matters for both reproducibility and interpretation of the GP observation noise. The paper should state exactly what varies across the six episodes used for one controller evaluation and whether the same random seeds/episode banks are shared across methods for paired comparison.

## Minor comments

1. **Terminology is still inconsistent in the abstract and introduction.** The abstract uses "client" repeatedly, while the body has largely moved to "system." For a control-oriented paper, I recommend using "system" or "vehicle" throughout and reserving "client" only if needed when discussing federated-learning literature.

2. The phrase "LQI-calibration landscapes remain highly rank-correlated" is good, but the abstract should ideally state that this result is for the investigated benchmark rather than sounding universal.

3. The subsection title "Related work and positioning" is useful in the draft. In the final IEEE version, consider whether it should remain as an Introduction subsection or be compressed into the introduction if page pressure becomes severe.

4. The origin of the state-scale vector is still only qualitative ("representative nominal closed-loop trajectories"). A reproducible rule or exact derivation should be stated.

5. The steering-rate limit is given in rad per simulation step. Consider additionally reporting the equivalent rad/s or deg/s value for physical interpretation.

6. The table caption for BO implementation currently advertises unresolved red entries. This is appropriate for internal revision but obviously must be removed before submission.

7. The manuscript still contains `\rev{...}` revision markup. This is useful internally, but all color/revision commands should be removed from the submission source once the revision is accepted by the authors.

8. The author affiliation line is still a draft placeholder. This needs to be finalized.

9. The IFAC World Congress reference still contains "final bibliographic details to be inserted." Update this if the proceedings metadata are available by submission time.

10. The filename `fig6_lane_change_reference.pdf` no longer reflects manuscript figure order. This does not affect the paper, but neutral descriptive filenames would make the source package easier to maintain.

11. The manuscript could report the computational cost of fitting an exact GP to roughly 725 historical observations plus local points. Closed-loop experimentation likely dominates, but runtime/memory overhead would help assess practical commissioning use.

12. "The result is consistent across all three vehicle families" should ideally be supported by a compact family-wise table, confidence interval, or appendix result rather than asserted without numbers.

13. Consider reporting the number or fraction of targets that fail to reach the 5% criterion by the final budget for each method. This directly complements the censoring discussion.

## Assessment by category

| Category | Score | Assessment |
|---|---:|---|
| Relevance to control conference | 4.5/5 | Strong fit for CCTA and ECC, with a concrete commissioning and vehicle-control motivation. |
| Technical correctness | 3/5 | Main method is coherent, but the censoring convention and landscape-correlation treatment require correction/validation. |
| Novelty | 3/5 | Limited algorithmic novelty, but the fleet-calibration formulation and heterogeneity insight are now articulated much more convincingly. |
| Experimental quality | 4/5 | Strong held-out design and useful negative result. The primary metric/reference construction needs tightening. |
| Reproducibility | 2.5/5 | Much improved structure, but several numerical and identification details remain unresolved. |
| Clarity | 4/5 | Story and terminology are substantially clearer, and Fig. 1 improves accessibility. |
| Significance | 4/5 | A 54% reduction in incremental commissioning experiments is practically meaningful if the metric is corrected and reference quality is established. |

## Recommendation rationale

I recommend **Borderline / Weak Reject (5/10)** for the current draft. This is a one-point improvement over Round 1. The manuscript now has a coherent control-oriented contribution and a strong held-out empirical result. I would likely move to **Weak Accept (6/10)** if the authors resolve the censoring/hitting-time definition, verify that the offline reference is sufficiently strong, demonstrate that the landscape-correlation conclusion is not an artifact of common infeasibility penalties, and fill all remaining reproducibility placeholders.

Importantly, I do **not** think the paper needs another broad research phase. The remaining work is focused. One small analysis on feasible-only landscape correlation may be necessary, and the reference/hitting-time definitions must be reconciled with the actual archived analysis scripts. Beyond that, most remaining issues are specification, terminology, and submission cleanup.

## Highest-priority revision sequence

1. Verify the actual analysis convention for trajectories that do not reach the 5% threshold and correct the metric/statistics accordingly.
2. Verify the actual procedure used to compute $J_i^{\rm ref}$ and strengthen or qualify it if necessary.
3. Recompute/check landscape correlations on jointly feasible points to rule out inflation from tied infeasibility penalties.
4. Recover every red numerical/configuration placeholder from the experiment code and remove all "verify" notes.
5. Define the stochastic episode generation precisely.
6. Replace "statistically indistinguishable" with non-equivalence wording unless a formal equivalence margin is introduced.
7. Harmonize system/client terminology, finalize affiliations/bibliography, and remove revision markup.
8. Optionally add one established transfer-BO baseline, or explicitly state why the paper intentionally tests only the minimal pooling-versus-dynamics-weighting hypothesis.
