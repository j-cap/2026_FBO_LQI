# FedBO: frozen condition-dependent findings

24 November 2026. Latest completed experiment:
`fb9e7763b8863d5cdb52c03943650ce2299ac07c`.
Latest analysis: [common peer-verification comparison](analysis/peer_assisted_cost_model/report.md).

The evidence supports reusable controller-tuning experience and online fleet
cooperation. The matched peer-verification comparison now finds a homogeneous
benefit of federating the additional cost mean relative to fitting it locally,
and heterogeneous deterioration in every tested seed. A reliable calibration
advantage over simple gain sharing with a local GP remains unestablished.

The frozen conclusion is: **under common peer verification, federating this
cost-mean model improves on fitting it locally for homogeneous vehicles and
degrades the declared final-budget result for the tested heterogeneous fleet**.
Both directions occur in all three seeds. This is a statement about the current
model, protocol and fixed plants, not a general law about federated BO.

## Task being studied

The current experiments directly tune the three feedback gains Ky, Kpsi and Kr
of a fixed tracking-controller structure. They retain kinematic steering
feedforward and the CFL-FF nonlinear bicycle plant. The objective combines
normalized lateral-tracking RMSE and applied-steering-rate RMS. The candidate
pool has 71 gain vectors and six speed/frequency contexts. The current study
does not tune LQI Q/R weights despite the legacy repository name.

## What the experiment chain establishes

| Finding | Evidence | Consequence |
|---|---|---|
| There is useful tuning headroom, largely associated with reference context | A vehicle-fixed choice improves mean J by 2.77% over a common free gain vector; vehicle/context choices improve it by 34.34% | Context is a stronger organizing variable than family labels alone in this task |
| Historical experience can be compressed into a useful small catalogue | A transferred catalogue reaches 90.74--94.44% near-reference context success after six target tests in the historical study | Direct reuse is a necessary practical comparator |
| The historical cost landscape contains additional information | Gain-dependent priors improve search beyond context-specific centering under the frozen historical protocol | The learned landscape can help, although the extra mean-cost gain over a good catalogue is modest |
| Source-data availability matters | The historical archive is treated as already available; the online studies charge every newly generated rollout | Historical transfer results do not establish the sample efficiency of generating fleet knowledge online |
| Online cooperation works without pre-existing histories | Gain sharing establishes useful controllers from newly generated fleet observations and pays for target verification | A pre-existing archive is not necessary for useful fleet cooperation |
| The revised initialization is online from query 2 | Federated and independent BO choose different second gains in 25/27 homogeneous and 27/27 heterogeneous trials | The six-blind-test objection has been resolved |
| The earlier cost-only sharing strategy had condition-dependent benefits | Without mandatory peer verification in the model arms, heterogeneous federation improved conditional tracking quality but left incomplete policies | These results compare different information-use strategies and do not isolate model sharing under the same peer rule |
| The matched peer-first comparison isolates the added model-sharing channel | C beats B in homogeneous context success and mean J in every seed; C loses to B in heterogeneous context success in every seed | The sign depends on the tested physical condition; the result is now scientifically narrower and clearer |
| The practical gain-sharing reference remains strong | C's homogeneous aggregate mean J is below A's, but A has higher near-reference success and wins on both metrics in two seeds | Improving on B does not establish consistent superiority to A |
| FedAvg training is close to the exact same-data ridge fit | Mean-fit prediction differences are small in the completed model arms | A training-iteration sweep is not presently motivated as the main explanation; exact pooled fitting is not a tested full peer-assisted policy |

Detailed sources:

- [Historical tracking evidence](analysis/tracking_consolidation/report.md).
- [First online pilot](analysis/online_calibration_pilot/report.md).
- [One-test online pilot](analysis/online_calibration_lcb/report.md).
- [Matched peer-assisted cost-model comparison](analysis/peer_assisted_cost_model/report.md).

## Previous one-test result in context

At 216 fleet tests, homogeneous gain sharing reaches 72.22% near-reference
contexts versus 37.65% for federated-cost BO. Both eventually have 27/27 complete
eligible client policies, but gain sharing first covers all contexts after
8.30 local tests on average versus 13.41 for federation. Federated mean J is
1.39% below gain sharing, so the coverage result does not imply dominance on
every cost statistic. The scout has lower homogeneous mean J than either.

With heterogeneous vehicles, federation reaches 42.59% near-reference contexts,
versus 25.31% for the local cost model, 36.42% for independent BO and 37.04% for
gain sharing. Its improvement over local-cost BO occurs in every seed. Its
average advantage over gain sharing comes from one seed; it trails slightly in
the other two. On 22 jointly eligible gain-sharing/federated pairs, mean J is
28.76% lower for federation. However, federation has only 22/27 complete eligible
policies versus gain sharing's 27/27. Four of its five missing contexts occur on
SUV variants, and all five are at `s30_f0.1`.

The practical information channels have different strengths. Reusing a tested
gain is effective for finding eligible controllers. A learned cost mean can
improve which gains are explored and the quality obtained, but the current
cost-only channel does not reliably replace direct gain reuse.

The old and new online pilots change initialization and acquisition together.
Their numerical differences cannot identify either change's separate effect.
The one-test/LCB protocol was retained for the matched comparison because it
implements the intended online process, irrespective of the old pilot's larger
scores in some cases.

## Scope of the claims

These are finite-pool simulation studies on fixed physical configurations, with
limited independent noise/programme realizations. Near-reference success uses a
declared 5% research threshold. Conditional J comparisons retain their jointly
eligible denominator. Missing recommendations and failed experiments consume
budget and remain in success metrics. The cost-only one-test pilot's one
unresolved local-cost query is charged and excluded from eligible cost training.
All 3,888 queries in the matched peer-assisted study are numerically resolved.

The evidence does not establish a population-level fleet benefit, certified
safety, generalization to unseen trajectory shapes, a communication advantage,
formal privacy, or a need for clustering. Sharing a mean plus a local residual
GP is one cost-model architecture; its limitations are not a general result
about all federated BO methods.

## The matched comparison answers the incremental-value question

Every method now uses the same peer-first rule. Arm A uses a zero additional
mean with its local GP, B fits the added mean locally, and C fits it through
FedAvg. Every rollout, including verification and failure, is charged. All
three arms use 216 fleet tests, with one factory test and 24 total local tests.

| Condition | A context success | B context success | C context success | Complete eligible policies A/B/C |
|---|---:|---:|---:|---|
| Homogeneous | 72.22% | 39.51% | 61.11% | 27/27, 27/27, 27/27 |
| Heterogeneous | 37.04% | 46.91% | 30.25% | 27/27, 26/27, 24/27 |

For homogeneous plants, C improves over B by 21.60 points and lowers mean J by
24.97%; both effects have the same positive direction in every seed. C also
lowers aggregate mean J by 10.70% versus A, but has lower near-reference context
success and loses on both cost and context success in two of three seeds.
Improving on a locally trained mean does not establish a consistent advantage
over the stronger simple-sharing reference.

For heterogeneous plants, C loses to B on context success in every seed and all
three families. On 24 jointly eligible pairs, C's mean J is 30.45% higher than
B's. B improves context success over A in every heterogeneous seed. Global
cost-model sharing can therefore be detrimental even while direct controller
gain sharing remains useful in the same fleet.

Model fitting remains close to the exact same-data ridge optimum. A limited
fixed-kernel diagnostic on common saved histories does not find uniformly
worse one-step proposals from the shared mean; it does not reproduce complete
adaptive policies. Surrogate mismatch, GP adaptation and catalogue feedback
remain possible explanations rather than a uniquely identified cause.

## Claims to retain and claims to withhold

| Claim | Status and reason |
|---|---|
| The tested task has substantial context-dependent controller-tuning headroom | Supported by the dense tracking diagnostic |
| Eligible controller gains can be reused across vehicles with paid local verification | Supported by historical and online catalogue studies |
| A shared learned mean adds value over the same mean fitted locally on identical plants | Supported within the matched three-seed homogeneous condition |
| The current shared mean improves heterogeneous calibration over the matched local mean | Rejected by this experiment's declared final-budget comparison |
| C consistently improves on the stronger simple peer-sharing method A | Unestablished across metrics and seeds |
| The heterogeneous failure is caused by FedAvg convergence, a missing family label, or a particular uncertainty defect | Unidentified; these mechanisms have not been causally separated |
| More local correction is absent from C | Incorrect: C already has a target-only residual GP |
| A hierarchical, descriptor-conditioned, or clustered replacement will improve online calibration | Untested modeling hypotheses |

The machine-readable contrasts, paired-cost denominators and seed directions
are in [condition_dependent_claims.csv](analysis/heterogeneous_cost_model_review/condition_dependent_claims.csv).

## Freeze and modeling handoff

Keep one factory test, the common peer-first rule, the existing LCB,
24 queries/client, six contexts, the 71-gain pool, eligibility rules and
declared 5% reference threshold fixed. Preserve the completed source/result
signatures. Do not reinterpret a conditional cost improvement as complete-fleet
superiority, or substitute a favorable budget/seed for the declared primary
comparison.

The experiment and rule are documented in
[tracking_peer_assisted_cost_model.md](tracking_peer_assisted_cost_model.md).
The modeling handoff is [heterogeneous_cost_model_review.md](heterogeneous_cost_model_review.md).
It audits the shared/local assumptions, statistical uncertainty, actual input
coverage and vehicle-dependent cost shapes using existing data only. It
separates code facts, retrospective diagnostics and proposed model classes.
No additional physical campaign is prepared by this consolidation or review.
