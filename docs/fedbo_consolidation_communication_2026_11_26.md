# FedBO: consolidation after the communication pilot

26 November 2026. Latest evidence: the
[read-only communication analysis](analysis/communication/report.md).
The [preceding dense-gain consolidation](fedbo_consolidation_dense_gain_2026_11_26.md)
remains archived.

**Exact compact summaries preserve the shared/local statistical method.
Two-round contact is promising under the implemented protocol, while the
added cost-sharing benefit remains programme dependent.** Remove repeated
unchanged peer catalogue entries before introducing information loss or
further BO tuning.

## Fixed task and methods

Nine homogeneous clients tune three feedback gains on the nonlinear bicycle
tracking task inherited from CFL-FF. The pool has 1,292 controllers; physical
bounds are [0.0005, 0.01, 0.01] to [0.15, 1, 1]. Six speeds and three
frequencies define 18 contexts. Clients receive 24 paid requests, including
factory and failed tests, with empty initial histories. Cost is lateral
RMSE / 0.25 m plus steering-rate RMS / 10 deg/s.

All methods use the same paid, exact-context peer-gain verification rule.
A uses an adaptive local GP, B adds its locally learned mean, D uses the
exact shared/local contextual GP, and E has the matched independent marginal
prior. A/B beta=2 and D/E beta=0.5 remain frozen. Eligibility labels remain
local. The active-input representation retains the exact conditional GP
remainder; independent observation-space checks verify recorded predictions.

Scalar cost summaries with public gain/context IDs and observation provenance
are permitted. Under that assumption, D can exchange 20-byte records and
reconstruct the same posterior instead of communicating dense Gaussian
matrices. Raw trajectories are not exchanged. This establishes no stronger
privacy property.

## Latest evidence

The compact representation reproduces all 2,592 old choices and 437 posterior
checks. The fresh contact experiment has 7,776 paid requests, three new
programmes and 6,028 validation rollouts, with no unresolved requests.
All 19 tables and paid decision/packet histories reproduce; 1,373 independent
causal GP checks agree to numerical precision.

| Contact interval | A success | B success | D success | E success | D kB used per fleet |
|---|---:|---:|---:|---:|---:|
| Every round | 17.90% | 17.64% | 24.98% | 24.25% | 445.73 |
| Every 2 rounds | 14.78% | 19.17% | 25.24% | 19.84% | 232.39 |
| Every 4 rounds | 22.06% | 11.90% | 22.38% | 18.37% | 120.72 |

Success is within 5% of the finite-pool reference, averaged within clients,
then equally over clients/programmes. Used bytes are application packets
arriving before the final paid decision, including headers and every recipient.

Two-round D retains average success (+0.26 pp), reduces failed requests from
77 to 63, increases complete eligible policies from 14 to 19 of 27, and cuts
used traffic by 47.86%. Including the final flush, the byte reduction is
44.44%. Used packet deliveries fall from 828 to 396. Local data remain fresh;
both remote channels synchronize less often.

D beats two-round A in all three programmes (+10.46 pp on average) and B
by 6.07 pp in aggregate. Compared with B it nevertheless has nine more
failures, two fewer complete policies and a 6.71% higher paired mean cost
on 19 common complete policies. All 63 D failures at this interval occur
at 0.1 Hz; difficult-context discovery remains unresolved.

Two-round D/E effects are -4.45/-5.72/+26.35 pp across the programmes.
Every-round D/E averages only +0.73 pp in these fresh programmes, compared
with +8.16 pp in the earlier development study. The practical average A/B
advantage is better supported than a uniform added-cost-sharing advantage.
Four-round D almost ties A (+0.31 pp), despite beating E in all three seeds.

## Communication interpretation and next step

The peer catalogue is 90.35% of every-round D traffic. Reduced contact
frequency saves mostly repeated catalogue publication, rather than new cost
observations. Dense Gaussian messages are unnecessary under the permitted
summary exchange and should not be the principal communication baseline.

A posthoc changed-record diagnostic reconstructs exactly the same catalogue
at all 504 logged contacts. Keeping current headers, recipients, cost
records and contact times gives hypothetical D totals of **88.93/80.55/69.20
kB** at k=1/2/4. These are serialized-byte estimates with cache-equivalence
checks, not a new live protocol result. Removing unchanged entries reduces
the k=2 versus k=1 used-byte benefit to about 9.4%; packet/contact savings
remain separate.

The [lossless changed-catalogue implementation](tracking_delta_catalogue.md)
now reproduces all 7,776 archived choices and every recipient catalogue at
504 contacts. Its actual serialized totals match the preceding diagnostic.
The next scientific step is the prepared ten-programme fixed confirmation
at k=1/2, retaining A/B and especially matched E; those fresh results are
still pending.
Keep k=2 as a candidate, not a declared optimum or demonstrated noninferior
protocol. Do not launch further kernel, bonus or gain-domain tuning on these
programme outcomes.

The developing story is useful online fleet cost information under a local
trial cap, exact compact exchange, and the effect of limited contact
opportunities after removing redundant traffic. Current evidence does not
establish heterogeneous-fleet generalization, uniform superiority, lossy
compression gains or real-network robustness. Earlier studies form an
adaptive development chain, not one pooled confirmation experiment.

## Reproduce

```powershell
python code/scripts/analyze_tracking_communication.py --workers 3
```

The analysis prohibits new simulation/optimization and verifies unchanged
source files. Denominators, paired cost comparisons, frequency diagnostics,
seed effects and communication accounting are in the
[report](analysis/communication/report.md) and
[verification record](analysis/communication/verification.json).
