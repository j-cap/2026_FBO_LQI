# Seed 1102: useful gains are missed before peer verification can help

24 November 2026. Source campaign:
`3aacf6702854d84468760ba956a0efcccd8a9f33`.
This follows the [matched spectral analysis](../spectral_online_cost_model/report.md).

## Question and scope

Does the seed-1102 low-frequency loss arise because D256 discovers useful
gains but does not transfer them, because recipients fail verification, or
because the arm never requests those gains? This diagnostic reads the actual
paid histories of A, B, D256 and E256 in both physical conditions. It replays
all four methods' decisions and follows strict prior-round peer provenance
back to its first locally tested origin. No new gain is proposed or simulated.

The 1,728 inspected requests remain part of the completed experiment. Archived
independent validation labels are opened only after decision replay and used
for post-hoc near-reference annotations. They do not enter any online choice.

## Observed discovery chains

All rows concern seed 1102. Each context has 36 paid requests per method: nine
vehicles with four visits. A dash means that no eligible queried controller is
near the target's independent reference. The last column counts final
near-reference recipients out of nine.

| Condition | Context | Method | First useful paid discovery | Later paid peer chain | Final successes |
|---|---|---|---|---|---:|
| Homogeneous | 20 m/s, .1 Hz | A | `g012`, client 06, round 18 | Eight other recipients verify it | 9 |
| Homogeneous | 20 m/s, .1 Hz | D256 | -- | No near-reference peer incumbent available | 0 |
| Homogeneous | 20 m/s, .1 Hz | E256 | `g012`, client 04, round 19 | Eight other recipients verify it | 9 |
| Homogeneous | 30 m/s, .1 Hz | A | `g012`, client 08, round 15 | Eight other recipients verify it | 9 |
| Homogeneous | 30 m/s, .1 Hz | D256 | `g012`, client 08, round 15 | Eight other recipients verify it | 9 |
| Homogeneous | 30 m/s, .1 Hz | E256 | `g012`, client 02, round 19 | Seven other recipients verify it before the cap | 8 |
| Heterogeneous | 20 m/s, .1 Hz | A | `g028`, round 2; broader `g012` discovery at round 12 | Final `g012` gains trace to clients 04/05 at round 12 | 9 |
| Heterogeneous | 20 m/s, .1 Hz | D256 | -- | No near-reference peer incumbent available | 0 |
| Heterogeneous | 20 m/s, .1 Hz | E256 | `g028`, client 03, round 2 | One other recipient verifies a near-reference gain | 2 |
| Heterogeneous | 30 m/s, .1 Hz | A | `g012`, client 03, round 15 | Eight other recipients verify it | 9 |
| Heterogeneous | 30 m/s, .1 Hz | D256 | -- | No near-reference peer incumbent available | 0 |
| Heterogeneous | 30 m/s, .1 Hz | E256 | `g012`, client 03, round 15 | Eight other recipients verify it | 9 |

Useful is target-specific here: `g028` is near-reference for a subset of the
heterogeneous recipients, while the final `g012` controllers succeed across
all nine in A's 20 m/s context. Donor costs are still the online selection
criterion; validation costs are retrospective annotations only.

## Discovery, rather than a blocked good transfer

In all three D256 failure rows, **none of its requested gain vectors is
near-reference on the recipient under independent validation**, even before
requiring the online test to be eligible. Thus these cases are not explained
by having selected the right gain and then merely failing its noisy local test.

There is also no decision in those focused low-frequency contexts where a
previously published, untried, near-reference peer incumbent is available but
another gain is chosen. No useful gain is waiting in the same arm's catalogue.
Across all four methods and all six contexts, nine other decisions do have
such an available alternative but follow the fixed donor-cost ordering; they
are outside these low-frequency cases. The diagnostic does not claim that the
peer rule can never matter elsewhere.

The contrast in search allocation is informative. In heterogeneous 20 m/s,
.1 Hz, A makes 17 BO requests and 17 peer verifications; D256 makes nine BO
requests and 25 peer verifications. D256's paid peer channel propagates gains,
but none is near-reference. In heterogeneous 30 m/s, .1 Hz, D256 has 17 BO
requests and 16 peer verifications, versus A's 14 and 19, and still misses a
near-reference gain. Simply counting more or fewer BO requests is therefore
not sufficient to explain both failures.

The homogeneous 30 m/s chain is a useful control: A and D256 independently
select `g012` at the same client/round, and both propagate it to all nine
vehicles. The peer machinery is capable of completing the same chain when
the discovery occurs. Late discovery also has a visible budget consequence:
E256's round-19 discovery in that context reaches only eight of nine vehicles
within their remaining scheduled visits.

These observations locate the failure in the arm's discovery history and its
subsequent catalogue. They do not isolate a causal defect in an individual
posterior, sharing fraction or acquisition score: each arm has accumulated
different costs and eligibility labels. A cannot donate its discovered gain
to D256 in this comparison because the experimental arms own separate
catalogues. A counterfactual forced discovery would be a new policy variant,
which this diagnostic does not run.

## Decision

Retain D256, the common peer rule and the 24-test cap. The audit supplies no
reason to change communication or peer verification before checking whether
the aggregate result replicates. Keep seed 1102 as an observed discovery-chain
failure, without dropping it, extending its budget or tuning to its gains.
The [confirmation protocol](../../tracking_confirmation.md) keeps the method
fixed and tests fresh programmes/noise and independent physical fleet draws.

Reproduction:

```bat
python code/scripts/diagnose_tracking_discovery.py
```

The event annotations, final-controller origins, context summary and
[verification record](verification.json) are saved beside this report.
