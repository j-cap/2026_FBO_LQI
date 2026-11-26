# FedBO: consolidation after the lossless-catalogue confirmation

26 November 2026. Latest evidence: the
[ten-programme lossless confirmation analysis](analysis/lossless_catalogue/report.md).
The [preceding communication consolidation](fedbo_consolidation_communication_2026_11_26.md)
and [dense-gain consolidation](fedbo_consolidation_dense_gain_2026_11_26.md)
remain archived.

**Timely online fleet cost information improves near-reference controller
tuning beyond matched local modeling in the tested homogeneous task.**
Every-round D beats A, B and matched E in all ten fresh programmes. The
predeclared two-round A/B comparisons are favorable, while its D/E effect
remains uncertain. The larger study favors every-round lossless exchange;
the earlier three-programme indication of little performance loss at k=2
does not replicate.

## Frozen task and information model

Nine homogeneous clients tune direct K_y/K_psi/K_r feedback gains for
trajectory tracking on the nonlinear bicycle plant inherited from CFL-FF.
The public pool contains 1,292 gains, with bounds [0.0005, 0.01, 0.01] to
[0.15, 1, 1]. Six speeds and three trajectory frequencies define 18 contexts.
Each client has 24 paid tests, including factory and failed trials, with
empty initial histories. Cost is lateral RMSE / 0.25 m plus steering-rate
RMS / 10 deg/s. No archived objective labels initialize a learner.

All arms use the same locally paid, exact-context peer-first verification
rule. A has an adaptive local GP; B adds its original local cost mean; D has
the exact shared/local contextual GP; E has the matched independent marginal
prior. A/B beta=2 and D/E beta=0.5 stay fixed. Eligibility labels remain
local. The active-input representation retains the exact conditional GP
remainder. Independent observation-space checks verify recorded predictions.

Scalar eligible-cost records with gain/context and sender/trial IDs are
permitted. D uses those records to reconstruct its posterior, with no raw
trajectory sharing. This is not a parameter-only or stronger privacy claim.
All arms now send only changed peer entries to persistent recipient caches.
The old dense information matrices and unchanged full-catalogue repetitions
are unnecessary under this permitted information model.

## Fixed ten-programme results

The primary operating point was k=2, with D/A first, then D/B and matched
D/E; k=1 was a predeclared reference. The ten seeds are 10100–10109 on one
physical plant. Descriptive 95% bootstrap intervals resample programme
means, not the 90 dependent client runs, without multiplicity adjustment.

| Contact interval | A success | B success | D success | E success | D kB used/fleet |
|---|---:|---:|---:|---:|---:|
| Every round | 13.79% | 17.90% | 40.11% | 21.32% | 87.44 |
| Every 2 rounds | 13.78% | 18.35% | 27.79% | 22.02% | 76.34 |

Success is within 5% of the empirical finite-pool reference, averaged over
encountered contexts within clients, then clients/programmes equally. Missing
or ineligible recommendations remain failures. Used traffic includes actual
application headers and all recipients before the last paid decision.

At k=2, D/A and D/B effects are +14.01 and +9.44 pp, with descriptive
intervals [+8.81,+19.07] and [+3.63,+15.35]. D/E averages +5.78 pp with interval
[-0.85,+13.50] and five positive/five negative programmes.

At k=1, D beats A/B/E in all ten programmes, averaging +26.32/+22.21/+18.79
pp. Their descriptive intervals are [+20.94,+31.36], [+13.46,+30.71] and
[+12.91,+24.90]. Matched D/E supplies direct evidence for cost information
beyond the common peer-controller transfer rule. D/A alone cannot isolate
that mechanism because A also differs in its local GP/exploration settings.

## Communication sensitivity and practical tradeoffs

D2 minus D1 is -12.32 pp, with descriptive interval [-18.18,-6.31], and is
negative in nine of ten programmes. A/B/E change by only -0.01/+0.45/+0.69 pp,
with intervals spanning zero. Both remote channels were delayed together;
the experiment does not isolate model staleness from delayed peer-controller
availability. Own local observations were always current.

After lossless catalogue updates, k=2 saves only 12.69% used bytes, or 8.39%
including the final flush. It halves full-programme packet deliveries. The
extra byte savings mainly come from fewer headers. The old naive full-
catalogue savings are not the efficient communication comparison. D's
cost-sharing channel has a real premium over E: about 42.87 kB and twice
the used packet deliveries at every-round contact.

The primary success advantage is not uniform superiority. Every-round D
has 47/90 complete eligible policies and 264/2,160 failed requests, compared
with B's 57/90 and 224/2,160. Its paired mean physical cost is 13.06% higher
than B on 39 common complete policies, while 10.69% below E on 31. Two-round
D has 29 complete policies and 302 failures. Low-frequency discovery remains
weak: every-round D has zero successes among 405 encountered 0.1 Hz cases.
No arm reaches the near-reference target in every encountered context of
any client run. Keep these limitations distinct from the useful success-rate
improvement.

## Decision

Retain **every-round lossless D** as the statistical reference and consolidate
this method/evidence. The supported story is useful timely online fleet
cost information under a local trial cap, beyond the same peer-verification
mechanism, using an exact compact protocol. The new 40.11% success rate is
not caused by compression; the archived replay proves that compression
preserves decisions, and the confirmation uses new programmes/validation.

If communication work continues, first isolate the two contact schedules
with the lossless protocol fixed. Do not infer that stale model information
alone caused the coupled-schedule loss, add lossy compression before that
diagnostic, or reopen kernel/bonus/gain tuning on these outcomes. The current
study is about programme/noise variation on one homogeneous plant; it does
not establish heterogeneous-fleet generalization or a need for personalization.
The earlier development chain remains separate from this fixed confirmation.

## Reproducibility

The read-only audit reproduces all 17,280 choices and 22 tables, independently
checks 3,056 D/E predictions, and verifies 12,316 validation rollouts. All
18 references are available; no request is unresolved. Source files retain
their hashes, and no fresh simulation or hyperparameter optimization is used.

```powershell
python code/scripts/analyze_tracking_delta_catalogue.py --workers 3
```

See the [analysis](analysis/lossless_catalogue/report.md) and
[verification record](analysis/lossless_catalogue/verification.json) for
conditional denominators, paired uncertainty, frequency effects and provenance.
