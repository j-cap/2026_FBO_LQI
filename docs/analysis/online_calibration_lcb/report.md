# One-test online calibration: result analysis

24 November 2026. Results commit: `4c00e82d677f2d9669121bf4f00c8e9f9082f78c`.
Protocol: [tracking_online_lcb.md](../../tracking_online_lcb.md).

Immediate online sharing is now active and affects second-query decisions.
The results establish a useful protocol correction, but do not establish a
reliable advantage for the federated cost model over simpler fleet cooperation.
Homogeneous gain sharing establishes eligible controllers much faster and
achieves more near-reference contexts. Heterogeneous cost-model sharing improves
tracking quality relative to local learning, but leaves incomplete client
policies and has a seed-dependent advantage over gain sharing. These are
different outcomes and should be reported together.

## Verification

All six jobs and **7,776 charged online queries** pass checkpoint validation.
The simulation cache contains 6,217 distinct counterfactual records. Every arm
spends 216 tests per job, every client spends 24, and the concurrent/scout arms
finish after 24/36 calibration rounds. All five concurrent arms share exactly
one factory test per client. There is no six-context initialization gate.

The audit reproduces all 1,944 client/budget rows, 11,664 context rows, query
and learning tables, summaries, and client/context eligibility-time tables from
the saved checkpoints and independent validation bank. Query/context identities,
client-specific paired noise, prior-round information barriers, scout timing,
peer proposals, observed incumbents and simulator labels are verified. Federated
regression replay reproduces its logged optimization gaps and final weights.

One of the 7,776 queries is numerically unresolved at the maximum refinement:
heterogeneous seed 1102, local-cost BO, sport/payload client, local query 8,
context `s30_f0.1`, candidate `g057`. It also fails physical feasibility in the
recorded rollout. It remains charged, is marked ineligible, and does not enter
cost training or recommendations. This is recorded in
[numerically_unresolved_queries.csv](numerically_unresolved_queries.csv).

Reproduce the audit and derived tables without new physical trials:

```bat
python code/scripts/analyze_tracking_online_lcb.py
```

The experiment sources and completed result archive are unchanged. Fingerprints
and verification details are in [verification.json](verification.json).

## 1. Sharing affects decisions immediately

At local query 1, all concurrent methods have identical actions and outcomes.
At local query 2, federated and independent BO select different gains in
25/27 homogeneous and 27/27 heterogeneous client trials. Federated and exact
centralized BO select the same second gain in every trial.

Thus the learned mean is being used from round 2, even without a local eligible
incumbent. Missing early benefits cannot be attributed to the old blind
initialization gate. These action differences demonstrate algorithmic influence,
not necessarily better experiment choices.

## 2. Final budget: quality and eligibility

Each condition contains three runs of nine clients: 27 policies and 162
client/context outcomes. Missing or ineligible recommendations count as
unsuccessful in the near-reference metric. These observations depend on the
same fixed plants and are not independent fleet draws.

| Method | Homogeneous contexts within 5% | Homogeneous complete eligible policies | Heterogeneous contexts within 5% | Heterogeneous complete eligible policies |
|---|---:|---:|---:|---:|
| Independent GP BO | 40.74% | 22/27 | 36.42% | 25/27 |
| Local cost model + BO | 35.80% | 27/27 | 25.31% | 23/27 |
| Online gain sharing | **72.22%** | **27/27** | 37.04% | **27/27** |
| Single scout + verification | 62.35% | 27/27 | **44.44%** | 24/27 |
| Federated cost model + BO | 37.65% | 27/27 | 42.59% | 22/27 |
| Centralized cost model + BO | 37.65% | 27/27 | 38.89% | 24/27 |

Homogeneous federation improves near-reference context success by only 1.85
percentage points over the same model trained locally and is 3.09 points below
independent BO. Gain sharing exceeds it by 34.57 points. No homogeneous
federated client reaches the 5% threshold in all six contexts, compared with
7/27 gain-sharing clients and 4/27 scout clients.

Heterogeneous federation improves context success by 17.28 points over local
cost BO and 6.17 points over independent BO. Relative to gain sharing it gains
5.56 points, while five clients lack complete eligible policies. None of the
methods achieves an all-six-context near-reference policy in this condition.

### Cost is a separate outcome

For homogeneous vehicles, the complete eligible set is the same 27 policies
for local, gain-sharing, scout, federated and centralized methods. Their mean
validation costs are 0.979046, 0.936566, 0.874604, 0.923568 and 0.914141.
Federation reduces mean J by 5.67% relative to local cost BO and 1.39% relative
to gain sharing, while scout mean J is 5.30% below federated mean J.

The small federated mean-cost advantage over homogeneous gain sharing does not
contradict the latter's larger near-reference success rate: the distributions
differ. Gain sharing has median J=0.696806 versus 0.884308 for federation, but
one poorer seed raises its overall mean. Neither metric should be used to claim
universal dominance. Independent BO's mean uses only 22 eligible policies and
is not an all-client mean.

For heterogeneous vehicles, conditional means use different policy subsets.
On the **22 jointly eligible** gain-sharing/federated pairs, mean J is 1.113698
versus 0.793406: federation reduces that paired-set mean by **28.76%**. The five
incomplete federated policies remain outside this cost comparison and inside
all success/eligibility denominators. Against local-cost BO the corresponding
reduction is 20.46% on 20 jointly eligible pairs. This is a conditional quality
benefit, not a complete-fleet commissioning result.

See [paired_final.csv](paired_final.csv) for the denominator of each comparison.
The ratio of paired-set means and the mean of individual paired ratios are
different estimands and are both retained with explicit column names.

## 3. Early calibration and completion time

| Condition | Method | Contexts within 5% at 54 tests | At 108 tests | At 216 tests |
|---|---|---:|---:|---:|
| Homogeneous | Independent BO | 9.26% | 19.75% | 40.74% |
| Homogeneous | Local cost BO | 11.11% | 20.37% | 35.80% |
| Homogeneous | Gain sharing | 21.60% | 27.78% | 72.22% |
| Homogeneous | Federated cost BO | 5.56% | 14.81% | 37.65% |
| Heterogeneous | Independent BO | 4.94% | 14.20% | 36.42% |
| Heterogeneous | Local cost BO | 7.41% | 11.73% | 25.31% |
| Heterogeneous | Gain sharing | 11.11% | 17.90% | 37.04% |
| Heterogeneous | Federated cost BO | 16.05% | 24.69% | 42.59% |

No clear homogeneous early calibration benefit appears for the shared model.
At 108 tests, gain sharing has an eligible gain in 99.38% of contexts versus
89.51% for federation. All 27 homogeneous gain-sharing clients eventually cover
all six contexts, requiring a mean of 8.30 local queries; federation also covers
all 27, but requires 13.41 queries on average.

With heterogeneous vehicles, final gain-sharing coverage is 27/27 complete
clients versus 22/27 for federation. Their conditional mean query counts to full
coverage are 13.59 and 13.73, but the latter omits five censored clients. The
mean of time-to-coverage truncated at the common 24-query cap is 13.59 versus
15.63. This metric assigns missing events their censored cap, without treating
them as successful coverage at that cap. The scout's local-query clock also
excludes its initial waiting; elapsed calibration rounds must be reported.

All five missing heterogeneous federated contexts are `s30_f0.1`. Four belong
to SUV/payload or SUV/front-tire configurations; one is compact/payload. In
contrast, the sport family has 9/9 complete federated policies and 55.56%
near-reference contexts, versus gain sharing's 37.04%. The heterogeneous benefit
therefore varies across vehicle families and operating conditions.

There is one selected gain that fails independent validation, in homogeneous
independent BO. All other final eligibility failures are missing locally
eligible gains. See the missing-context and validation-failure tables.

## 4. Seed dependence limits the aggregate heterogeneous advantage

| Heterogeneous seed | Federated context success | Gain-sharing context success | Difference |
|---|---:|---:|---:|
| 1100 | 42.59% | 22.22% | +20.37 points |
| 1101 | 29.63% | 31.48% | -1.85 points |
| 1102 | 55.56% | 57.41% | -1.85 points |

The average +5.56-point federated advantage over gain sharing comes from seed
1100. Federation exceeds local-cost BO on this metric in every heterogeneous
seed, but is not a consistent winner over direct gain sharing. No population
significance claim follows from three seeds on fixed physical configurations.

## 5. FedAvg is close to its same-data optimum

Across regression rounds 1--23, which can inform later gain queries:

| Condition | Median objective gap | Maximum gap | Median RMS prediction difference in log J | Maximum RMS difference |
|---|---:|---:|---:|---:|
| Homogeneous | 0.000008 | 0.000117 | 0.002603 | 0.012415 |
| Heterogeneous | 0.000018 | 0.000232 | 0.003524 | 0.011322 |

Exact centralized cost BO also reaches only 37.65% homogeneous context success.
The federated solver is therefore not the principal explanation for the large
homogeneous gap to gain sharing. Small numerical fitting differences can still
change adaptive choices and later datasets. The heterogeneous federated score
exceeding the centralized score does not establish that federation is
intrinsically statistically superior to pooled learning.

The comparator fits the same mean model exactly; it is not a globally
conditioned full-GP optimizer. These findings concern the current shared-mean
plus local-residual-GP architecture.

## 6. Useful peer gains remain underused by the cost-only channel

A retrospective diagnostic considers only prior-round observations in the
federated arm's own trace. It does not borrow the gain-sharing arm's data.
When a scheduled context has no local eligible incumbent, an untried peer
incumbent is available at 168 homogeneous and 171 heterogeneous decisions.
That candidate is validation-eligible in 168/168 and 151/171 cases respectively.
After a local incumbent exists, a peer candidate is validation-better than the
chosen action in 255/401 homogeneous and 248/412 heterogeneous opportunities.

These are post hoc one-step comparisons on the fixed validation bank, not new
paired-noise physical rollouts or estimates of a different complete policy.
They identify information already available in the fleet that is not reliably
converted into useful finite-budget decisions by sharing a mean alone. They
do not prove that exploratory actions were unnecessary.

Actual peer-verification eligibility is 379/382 (99.2%) for homogeneous and
348/379 (91.8%) for heterogeneous gain sharing. This confirms that peer reuse
is useful but is not a guarantee of target eligibility. Its BO fallback
accounts for 239 and 242 adaptive queries respectively, leaving a concrete
set of decisions where an additional learned model could potentially help.

## 7. Previous protocol and communication

Relative to the frozen six-test/EI pilot, homogeneous federated context success
falls from 68.52% to 37.65%; gain sharing falls from 88.89% to 72.22%.
Heterogeneous federation changes from 46.30% to 42.59%, and gain sharing from
62.96% to 37.04%. Independent BO improves modestly in both conditions.

Both initialization and acquisition changed, so these differences cannot
identify either change's individual causal effect. The new protocol better
matches online learning from the second experiment; that does not imply
better finite-budget optimization performance. Do not silently choose the old
protocol for its larger federated score or the new protocol for a more favorable
heterogeneous comparison.

Federated payload before the final query averages about 42.0/42.4 MB for the
homogeneous/heterogeneous conditions, versus 0.127/0.120 MB for gain sharing.
The present 200-server-iteration protocol provides no communication advantage.
It exchanges model vectors, but no formal privacy or application-specific
restriction on sharing incumbents has been established.

## Consolidated scientific decision

The empty-history protocol is now resolved: every observation is generated
online and sharing affects query 2. Further initialization changes are not the
next scientific question. Training convergence is also insufficient to explain
the remaining performance gap.

The evidence supports online fleet cooperation and a conditional heterogeneous
tracking-quality benefit from sharing a cost model. It does not yet support a
reliable end-to-end calibration advantage of federated cost learning over
simple gain reuse. Homogeneous commissioning speed, heterogeneous policy
completion and seed dependence remain limiting evidence.

Freeze this protocol and retain gain sharing as the practical comparator. If
the learned-model route is continued, the focused unanswered question is its
**incremental value when both methods use the same peer-verification rule**.
Comparing local-model and federated-model BO fallbacks under that common rule
would separate direct reuse of tested gains from learning a transferable
landscape. This is a proposed decision, not an experiment implemented here.
