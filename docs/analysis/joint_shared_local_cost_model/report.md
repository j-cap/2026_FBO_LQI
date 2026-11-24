# Joint shared/local cost BO: a heterogeneous gain over A, with remaining limits

24 November 2026. Results commit:
`47e8ee396eddf7379ddb463c5c318c7ae9d2f8e2`.
Protocol: [tracking_hierarchical.md](../../tracking_hierarchical.md).
Previous frozen result: [matched C/B analysis](../peer_assisted_cost_model/report.md).

## Decision against the stated goal

**D improves on A in the tested heterogeneous condition at the declared final
budget, with complete eligibility preserved. It does not improve on A in the
homogeneous condition, and it does not beat B on heterogeneous near-reference
context success.** The goal of consistently beating A and then B is partly met.

The heterogeneous D/A result is the useful positive finding: context success
increases by 8.02 percentage points, mean J falls by 18.81%, and all 27 client
policies are eligible for both methods. Success improves in two seeds and ties
in one; cost is lower in all three. All three families improve in aggregate.

## Audit and unchanged information contract

The audit recomputes all 20 saved numerical analysis tables, validates six
completed jobs, and replays every D/E choice using only its own paid earlier
observations. It checks source/configuration/result fingerprints, simulator
records, the common peer rule, context programmes, noise pairing, local caps,
incumbents, information-message payloads and pre-training budget snapshots.

The comparison contains 6,480 charged requests: 2,592 from the new D/E campaign
and 3,888 from the frozen A/B/C traces reused only for evaluation. Every method
has 24 tests per client, including failures and peer verification. All requests
are numerically resolved. All selected final gains pass archived validation;
missing contexts remain unsuccessful and are never dropped from success rates.

There are 162 client/context outcomes and 27 client policies per physical
condition across the three programme/noise seeds. These are fixed physical
plants, not independent fleet draws. Homogeneous clients share the validation
bank. The 162 outcomes must not be treated as independent fleet replications.

A uses peer gains and a local GP. B adds a locally fitted mean; C federates that
mean through FedAvg. D uses joint shared/local Bayesian inference and E the
same marginal Bayesian cost prior with cost-model sharing disabled. Every arm
retains peer-gain verification and owns its own adaptive catalogue.

## Final outcomes

Near-reference success retains the existing 5% threshold at 216 fleet queries.
Mean J is across all policies only when all 27 are eligible. Asterisks mark
conditional means over the eligible subset; the paired contrasts below retain
their own common eligible denominators.

| Condition | Method | Context successes / 162 | Success rate | Eligible policies / 27 | Mean J |
|---|---|---:|---:|---:|---:|
| Homogeneous | A: peer gains + GP | 117 | 72.22% | 27 | .9366 |
| Homogeneous | B: local mean | 64 | 39.51% | 27 | 1.1147 |
| Homogeneous | C: federated mean | 99 | 61.11% | 27 | .8364 |
| Homogeneous | D: joint shared/local | 86 | 53.09% | 27 | .9928 |
| Homogeneous | E: local Bayesian control | 70 | 43.21% | 27 | 1.2389 |
| Heterogeneous | A: peer gains + GP | 60 | 37.04% | 27 | 1.1457 |
| Heterogeneous | B: local mean | 76 | 46.91% | 26 | 1.0670* |
| Heterogeneous | C: federated mean | 49 | 30.25% | 24 | 1.3414* |
| Heterogeneous | D: joint shared/local | 73 | 45.06% | 27 | .9302 |
| Heterogeneous | E: local Bayesian control | 52 | 32.10% | 25 | 1.0790* |

### Primary: D versus A

| Condition | Success change | Relative mean-J change | Joint eligible policies | Positive / tied / negative success seeds |
|---|---:|---:|---:|---|
| Homogeneous | -19.14 pp | +6.00% | 27 | 0 / 1 / 2 |
| Heterogeneous | +8.02 pp | -18.81% | 27 | 2 / 1 / 0 |

Heterogeneous D/A by seed:

| Seed | Success change | Relative mean-J change | Eligible policies D/A |
|---|---:|---:|---|
| 1100 | +9.26 pp | -9.99% | 9 / 9 |
| 1101 | +14.81 pp | -35.47% | 9 / 9 |
| 1102 | tie | -1.49% | 9 / 9 |

Heterogeneous success also increases by 7.41 pp for compact, 7.41 pp for sport
and 9.26 pp for SUV clients. D's family mean J is lower than A's for every
family, using all nine policies in each family. The strongest cost reduction
is in sport; this is not a single-family success-rate reversal.

Homogeneous D ties A's success in seed 1100 but loses in 1101/1102. Its cost is
lower only in seed 1100. Improvements over E or B do not overturn this failure
against the primary practical comparator.

### Secondary: D versus B

| Condition | Success change | Relative paired mean-J change | Eligible policies D/B | Joint eligible policies |
|---|---:|---:|---|---:|
| Homogeneous | +13.58 pp | -10.94% | 27 / 27 | 27 |
| Heterogeneous | -1.85 pp | -15.92% | 27 / 26 | 26 |

The heterogeneous result is a metric tradeoff. B has three more near-reference
contexts, while D has a complete eligible policy for every client and lower J
on the 26 common eligible policies. D's paired cost is lower in each seed, but
its context success is lower in two of three. Do not describe this as a clean
win over B under the declared primary metric.

The aggregate context counts locate this tradeoff. At 20 m/s, .1 Hz D has
19 successes versus B's 7; at 30 m/s, .6 Hz it has 12 versus B's 24. D ties B's
family-level success in compact and sport and trails in SUV. Mean raw J and
the number of contexts within 5% of their own reference need not rank these
policies identically.

## What federation contributes within the Bayesian model

D/E keeps the marginal cost prior, features, noise and decision rule matched,
while introducing dependence between vehicles' cost surfaces in D.

| Condition | D minus E success | Relative paired mean-J change | Eligible policies D/E | Joint eligible policies |
|---|---:|---:|---|---:|
| Homogeneous | +9.88 pp | -19.87% | 27 / 27 | 27 |
| Heterogeneous | +12.96 pp | -13.94% | 27 / 25 | 25 |

Homogeneous success improves in all three seeds. Heterogeneous success and
paired cost improve in two seeds and deteriorate in seed 1100. Thus cost-model
sharing adds aggregate value within this model class in both conditions, with
limited seed consistency in the heterogeneous case.

This contrast is the evidence for the additional cost-sharing channel. D/A is
the practical benchmark. D/C changes the model, fitting and uncertainty together;
it does not isolate any one repair to C. D uses exact Gaussian information
aggregation, so its result is not a claim that FedAvg optimization was repaired.

## Sharing strength responds to the physical condition

Before the last query, D's posterior expected sharing fraction is:

| Condition | Seed 1100 | Seed 1101 | Seed 1102 |
|---|---:|---:|---:|
| Homogeneous | 1.000 | 1.000 | 1.000 |
| Heterogeneous | .750 | .520 | .498 |

Homogeneous evidence concentrates on essentially full sharing by round 2--3.
Heterogeneous learning retains a shared component and permits substantial local
variation. It does not obtain its gain by simply switching all cost sharing off.
Rho is a covariance parameter conditional on the fixed feature/noise model,
not a physical percentage of vehicle similarity or a fraction of shared labels.

Correctly distinguishing full versus partial sharing does not ensure good BO.
In the homogeneous case rho is consistent with identical plants, yet A remains
stronger. The finite-feature cost prior, fixed covariance settings, local
eligibility model and their adaptive interaction remain relevant differences;
these results do not causally separate them.

## Localizing the remaining failure without another variant

For homogeneous D/A, 25 of the net 31 lost near-reference contexts occur at
.1 Hz. The 30 m/s, .1 Hz context has zero D successes versus 16 for A. Its
mean-cost difference accounts for about 96.8% of D's net aggregate J increase.

In that context D spends 56 BO fallback queries, of which 13 are eligible; A
spends 44, of which 20 are eligible. D has 47 peer verifications versus A's 59.
These are observed adaptive histories, not matched causal proposal comparisons.
They locate a costly search failure in the low-frequency region rather than
establishing which model assumption caused it. E also has zero final successes
there, so sharing alone is not sufficient to explain the weakness.

The analysis additionally reconstructs the actual prior-round D/E forecasts
at their recorded queries. For eligible next labels it reports log errors and
bands of two predictive observation standard deviations, including the fixed
noise variance. These are descriptive on-policy moment bands, not 95% Gaussian
mixture intervals or an independent calibration test. In the homogeneous
30 m/s, .1 Hz case all 13 eligible D fallback costs fall inside those bands;
the other 43 queries provide no eligible cost label. This does not support a
simple explanation based solely on narrow cost bands. It also says nothing
about whether the eligibility predictions are reliable on untested gains.

Across all homogeneous contexts, D has 33.67 ineligible tests/fleet on average
versus A's 19.33. Heterogeneous D has 37.33 versus A's 36.33. The demonstrated
heterogeneous advantage is in final controller quality and complete coverage,
not fewer failed experiments.

Nor is there uniform dominance over budgets. At 108 heterogeneous fleet queries
D has 19 complete policies versus A's 13 and higher context success, but its
mean J on the 12 common eligible policies is 81.56% higher. The declared final
budget remains the comparison that supports the positive D/A conclusion.

## Scope and next decision

Retain D as a heterogeneous candidate with a positive result against A on the
tested fixed fleet, and retain A as the stronger homogeneous reference. Keep B's
heterogeneous context-success advantage visible. The old negative C/B result
does not generalize to D, and the new D/A result does not establish general
superiority of shared/local federated BO.

The next useful diagnostic concerns how the fixed feature prior ranks and
explores the remaining low-frequency gains compared with A's adapted local GP,
and why D misses B's high-frequency near-reference gains. It should use these
saved histories before choosing another experimental variant. No change to the
plants, budget, peer rule, threshold or acquisition coefficients is justified
merely to reverse the ranking. An independent fleet confirmation would be
needed for a broader heterogeneous-fleet claim.

Model payload at the final decision is about 43.6 MB/fleet for heterogeneous D,
versus 42.4 MB for C and zero cost-model payload for A/B/E; peer publications are
reported separately. No communication, formal privacy or stability advantage
is established.

Reproduce the read-only audit and derived tables with:

```bat
python code/scripts/analyze_tracking_hierarchical.py
```

See [verification.json](verification.json), [final_contrasts.csv](final_contrasts.csv),
[context_contrasts.csv](context_contrasts.csv), and
[recorded_forecast_summary.csv](recorded_forecast_summary.csv).
The analysis performs no new physical rollouts or counterfactual proposals.
