# FedBO: consolidated findings and the remaining question

24 November 2026. Latest completed experiment:
`4c00e82d677f2d9669121bf4f00c8e9f9082f78c`.
Latest analysis: `3f42af8c8ca2ea99b2b211d77faedbdcc2c9418c`.

The evidence supports reusable controller-tuning experience and online fleet
cooperation. It has not yet established a reliable calibration advantage for
learning a federated cost model beyond directly sharing useful tested gains.
The remaining comparison should give every method the same peer-verification
procedure and vary the cost mean used by its BO fallback.

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
| Current shared-model benefits depend on the metric and physical condition | Homogeneous gain sharing is stronger on commissioning speed and near-reference coverage; heterogeneous federation improves conditional quality but leaves incomplete policies | No consistent end-to-end superiority of the federated cost model is established |
| FedAvg training is close to the exact same-data ridge fit | Prediction differences are small, and exact centralized fitting also struggles with homogeneous near-reference coverage | More FedAvg iterations alone are not a persuasive next experiment |

Detailed sources:

- [Historical tracking evidence](analysis/tracking_consolidation/report.md).
- [First online pilot](analysis/online_calibration_pilot/report.md).
- [One-test online pilot](analysis/online_calibration_lcb/report.md).

## Latest result in context

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
The one-test/LCB protocol remains the basis for the next comparison because it
implements the intended online process, irrespective of the old pilot's larger
scores in some cases.

## Scope of the claims

These are finite-pool simulation studies on fixed physical configurations, with
limited independent noise/programme realizations. Near-reference success uses a
declared 5% research threshold. Conditional J comparisons retain their jointly
eligible denominator. Missing recommendations and failed experiments consume
budget and remain in success metrics. The online pilot's one unresolved
local-cost query is charged and excluded from eligible cost training.

The evidence does not establish a population-level fleet benefit, certified
safety, generalization to unseen trajectory shapes, a communication advantage,
formal privacy, or a need for clustering. Sharing a mean plus a local residual
GP is one cost-model architecture; its limitations are not a general result
about all federated BO methods.

## The unanswered question

Does sharing the learned cost mean improve the remaining experiment choices
when every method already reuses eligible peer gains through the same rule?

The main comparison is between the same cost model trained locally and trained
federatively, with identical peer priority, local GP, eligibility surrogate,
acquisition, budgets and initial observations. The existing peer-sharing plus
zero-additional-mean GP method is retained as an architecture control.

This holds the direct-gain-reuse mechanism fixed. A different BO fallback can
still generate different later fleet catalogues; that is an intended downstream
effect of the model. Each method uses its own online fleet observations.

The proposed experiment and the precise meaning of peer verification are in
[tracking_peer_assisted_cost_model.md](tracking_peer_assisted_cost_model.md).
The three-arm implementation is prepared on `study/peer-assisted-cost-model`;
its full nonlinear campaign is intended for local execution. Completed pilot
results remain frozen.
