# Spectral online BO: D256 beats A and B in aggregate, with seed sensitivity

24 November 2026. Results commit:
`3aacf6702854d84468760ba956a0efcccd8a9f33`.
Protocol: [tracking_spectral.md](../../tracking_spectral.md).
Numerical motivation: [kernel assessment](../kernel_representation_assessment/report.md).

## Decision against the declared goal

**D256 meets the aggregate final-budget objective against A and B in both
tested physical conditions.** It increases near-reference context success,
preserves complete eligibility, and lowers mean cost on common eligible
policies. This is the first completed variant in this experimental chain to
meet that combination against both practical comparators in both conditions.

The result supports retaining D256 as the candidate. It does not establish
consistent superiority across seeds or independent fleets. Seed 1102 reverses
the D256/A result in both conditions and the D256/B result in the heterogeneous
condition. The matched local controls also show that sharing's contribution
depends on the metric and representation.

## Audit and information contract

The audit validates all six completed jobs, reloads and hashes the pinned
public bases, and replays all new Bayesian decisions using only their own
recorded paid earlier observations. It independently recomputes all **21 saved
numerical tables** and checks them against the committed results. It writes
only this analysis directory; source campaign results are unchanged.

Checks cover signatures, physical fleets, factory observations, simulator
records, context programmes, noise pairing, peer-first decisions, local caps,
posteriors, information-message payloads and pre-training budget snapshots.
The four new methods make 5,184 charged requests. A/B/D0/E0 contribute 5,184
existing requests for evaluation only: 10,368 total compared requests. The
new-arm cache contains 4,115 unique requested simulator episodes; cache reuse
does not reduce any method's charged count. Every client uses exactly 24
requests, including failures and peer verification.

All requests are numerically resolved. Every selected final gain passes the
archived independent validation; missing contexts in the old baselines remain
unsuccessful. No new physical rollout, model variant or counterfactual policy
is run by this analysis. Full audit details are in [verification.json](verification.json).

Each condition has 162 client/context outcomes and 27 client policies across
three programme/noise seeds. These are fixed physical plants, not independent
fleet draws. The homogeneous clients share a physical validation bank. Treat
the seed results as matched development evidence rather than 162 independent
replications or a confirmatory significance test.

## Final outcomes at 216 fleet requests

Context success uses the unchanged 5% empirical-reference tolerance. Mean J is
over all 27 policies when they are all eligible. Asterisks mark means over the
eligible subset; cost contrasts below use their own common eligible pairs.

| Condition | Method | Context successes / 162 | Success rate | Eligible policies / 27 | Mean J |
|---|---|---:|---:|---:|---:|
| Homogeneous | A | 117 | 72.22% | 27 | .9366 |
| Homogeneous | B | 64 | 39.51% | 27 | 1.1147 |
| Homogeneous | D0: original shared/local RFF64 | 86 | 53.09% | 27 | .9928 |
| Homogeneous | E0: original local RFF64 | 70 | 43.21% | 27 | 1.2389 |
| Homogeneous | D256: shared/local spectral | 135 | 83.33% | 27 | .8132 |
| Homogeneous | E256: local spectral | 120 | 74.07% | 27 | .7406 |
| Homogeneous | D384: shared/local accuracy control | 118 | 72.84% | 27 | .9140 |
| Homogeneous | E384: local accuracy control | 76 | 46.91% | 27 | 1.2246 |
| Heterogeneous | A | 60 | 37.04% | 27 | 1.1457 |
| Heterogeneous | B | 76 | 46.91% | 26 | 1.0670* |
| Heterogeneous | D0: original shared/local RFF64 | 73 | 45.06% | 27 | .9302 |
| Heterogeneous | E0: original local RFF64 | 52 | 32.10% | 25 | 1.0790* |
| Heterogeneous | D256: shared/local spectral | 82 | 50.62% | 27 | .9314 |
| Heterogeneous | E256: local spectral | 66 | 40.74% | 27 | 1.0050 |
| Heterogeneous | D384: shared/local accuracy control | 72 | 44.44% | 27 | 1.0971 |
| Heterogeneous | E384: local accuracy control | 76 | 46.91% | 27 | .9459 |

### Primary: D256 versus A

| Condition | Context-success change | Relative paired mean-J change | Eligible policies D256/A | Positive / tied / negative success seeds |
|---|---:|---:|---|---|
| Homogeneous | +11.11 pp | -13.17% | 27 / 27 | 2 / 0 / 1 |
| Heterogeneous | +13.58 pp | -18.70% | 27 / 27 | 2 / 0 / 1 |

All cost pairs are complete in this primary comparison. The heterogeneous
context improvement is present in every family in aggregate: +22.22 pp in
compact and +9.26 pp in both sport and SUV. D256 has lower aggregate family
mean J than A in all three families.

### Secondary: D256 versus B

| Condition | Context-success change | Relative paired mean-J change | Eligible policies D256/B | Common eligible policies | Positive / tied / negative success seeds |
|---|---:|---:|---|---:|---|
| Homogeneous | +43.83 pp | -27.05% | 27 / 27 | 27 | 3 / 0 / 0 |
| Heterogeneous | +3.70 pp | -14.46% | 27 / 26 | 26 | 2 / 0 / 1 |

The heterogeneous secondary margin is only six additional successful contexts,
so it needs particular care in confirmation. Compact and sport improve by
14.81 pp and 1.85 pp respectively; SUV loses 5.56 pp to B. The aggregate win
must not be described as superiority in every family or context.

## Seed sensitivity is the main qualification

Each seed has 54 client/context outcomes. The table retains the sign of the
mean-J change: positive means D256 has higher cost.

| Condition | Seed | D256 minus A context success | D256/A mean-J change | D256 minus B context success | D256/B paired mean-J change |
|---|---:|---:|---:|---:|---:|
| Homogeneous | 1100 | +37.04 pp | -31.94% | +37.04 pp | -22.62% |
| Homogeneous | 1101 | +3.70 pp | -11.31% | +66.67 pp | -33.62% |
| Homogeneous | 1102 | -7.41 pp | +20.56% | +27.78 pp | -25.64% |
| Heterogeneous | 1100 | +35.19 pp | -21.37% | +12.96 pp | -19.31% |
| Heterogeneous | 1101 | +35.19 pp | -49.88% | +31.48 pp | -33.55% |
| Heterogeneous | 1102 | -29.63 pp | +46.10% | -33.33 pp | +12.01% |

Seed 1102's heterogeneous loss to A is concentrated in low-frequency
discoveries. At 20 m/s, .1 Hz and 30 m/s, .1 Hz, A finds near-reference gains
for all nine clients, while D256 has zero successes in each context. Those
two contexts account for 18 lost successes; other contexts bring the net
loss to 16. Their mean-cost differences account for about 99% of the total
D256/A policy-cost increase in that seed.

In the homogeneous seed-1102 loss, the 20 m/s, .1 Hz context has zero D256
successes versus nine for A. Other contexts recover five, leaving a net loss
of four. This is a recurring sensitivity in the low-frequency search, even
though the previously identified 30 m/s, .1 Hz failure is repaired overall.

These are outcomes of different adaptive histories. They locate the remaining
failure but do not prove that a particular individual proposal, inferred rho,
or peer transfer caused it. The checked histories are available for a focused
discovery-chain inspection without another controller experiment.

## What cost-model federation adds

D/E at each width retains the same marginal prior, basis, observation noise,
decision formulas and peer-verification rule. Only the cost-sharing channel
changes. Their later paid histories and own peer catalogues can consequently
diverge; the contrast measures the effect on the complete online policy.

| Condition | Contrast | Context-success change | Relative paired mean-J change | Common eligible policies |
|---|---|---:|---:|---:|
| Homogeneous | D256 minus E256 | +9.26 pp | +9.81% | 27 |
| Heterogeneous | D256 minus E256 | +9.88 pp | -7.32% | 27 |
| Homogeneous | D384 minus E384 | +25.93 pp | -25.36% | 27 |
| Heterogeneous | D384 minus E384 | -2.47 pp | +15.99% | 27 |

The strongest aggregate evidence for an added cost-sharing channel is D256/E256
in the heterogeneous condition: it improves both the declared success metric
and mean cost. However, success improves in two seeds and mean J improves in
only seed 1101. Sharing hurts both metrics in seed 1102. This is a useful
aggregate effect with material variability, not universal positive transfer.

Homogeneous D256/E256 is a metric tradeoff: 15 more contexts are near their
own reference, but mean J is higher. The context-specific 5% thresholds and
the average raw cost weight the outcomes differently. E256's lower mean cost
does not invalidate D256's context-success improvement, and that improvement
does not establish an overall mean-cost advantage from sharing.

Before the last query, D256's expected rho is 1.000 for all homogeneous seeds
and approximately .750/.750/.500 for heterogeneous seeds 1100/1101/1102.
The inference retains meaningful sharing in both conditions. Rho is a
covariance-model parameter, not a physical similarity percentage or a guarantee
of useful transfer. D384 also infers sharing but loses to its local control in
the heterogeneous aggregate, so a learned shared component alone is not
evidence of a better controller-search policy.

## Representation improves the candidate, but accuracy is not monotonic

Relative to D0, D256 gains 30.25 pp homogeneous context success and lowers
mean J by 18.09%. In the formerly problematic homogeneous 30 m/s, .1 Hz
context, success rises from 0/27 to 27/27. The new representation repairs that
specific observed failure. At 20 m/s, .1 Hz it still has only 9/27 successes,
so the low-frequency difficulty is not eliminated across the grid.

The heterogeneous D256/D0 comparison is a tradeoff: +5.56 pp success with
mean J higher by .13%. Additional heterogeneous contexts near their own
reference do not translate into a lower pooled cost than the original D0.

The more exact D384 control underperforms D256: -10.49 pp success and +12.40%
mean J in the homogeneous condition, and -6.17 pp and +17.79% in the
heterogeneous condition. D384's heterogeneous success is below B's and its
paired mean J is 1.79% higher than B's. It is retained as the declared accuracy
control, not promoted or discarded after seeing results.

Thus the matched experiment shows that the representation changes practical
online performance. It does **not** show that monotonically reducing kernel
approximation error improves BO, or that the nearly exact original kernel is
the best controller-search prior. A regularization effect of truncation is a
possible explanation, but these four adaptive arms do not identify it causally.

## Budgets, failures and communication

The complete [budget curves](../../../code/results/tracking_spectral_v1/spectral_curves.png)
and [budget contrasts](budget_contrasts.csv) remain part of the result.
At 108 fleet queries, D256/A success is 51.23%/27.78% homogeneous and
23.46%/17.90% heterogeneous. Heterogeneous E256 is at 26.54%, ahead of D256
at that intermediate budget. At 54 heterogeneous queries D256 also trails A.
There is no claim of dominance at every budget or faster learning at every
stage.

Mean ineligible tests per fleet at the final budget are 28.33 for homogeneous
D256 versus 19.33 for A, and 26.00 for heterogeneous D256 versus 36.33 for A.
D256's homogeneous quality improvement therefore requires more failed tests
within the same cap. Its heterogeneous aggregate has both better final quality
and fewer failures than A.

D256's final-decision Gaussian model payload estimate is about 658--659 MB
per fleet, compared with 43.5--43.6 MB for D0. D384 is about 1.47 GB. The
conservative one-off basis broadcasts add 7.88 MB and 11.81 MB respectively;
peer publications are separately recorded. These are estimates for the exact
Gaussian information-message reference, without transport overhead or an
optimized factorization. The result establishes a statistical candidate, not
a communication-efficient deployment.

## Retain, consolidate, then confirm

Keep D256's model, representation, acquisition, budget and peer rule fixed.
Record the aggregate A/B result as positive and retain the unfavorable seeds,
homogeneous D/E cost tradeoff, heterogeneous SUV/B gap and D384 control.
Do not add another representation search to improve this table.

The immediate diagnostic should inspect the already-paid low-frequency
discovery/peer chains in seed 1102, without changing the policy. Then prepare
a separate confirmation campaign with A, B, D256 and E256, fresh programme/noise
seeds and independent heterogeneous fleet draws. Fix fleet/seed counts and
failure interpretation before running it. The homogeneous replication should
use fresh programme/noise seeds on its declared common plant. This separates
the practical A/B objective from whether cost sharing adds value beyond the
strong matched local model. No broader fleet claim is justified by the three
current programme/noise seeds alone.

Reproduce the audit and numerical analysis from the repository root:

```bat
python code/scripts/analyze_tracking_spectral.py
```
