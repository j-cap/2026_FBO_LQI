# Compact exchange and periodic synchronization

This is the communication follow-up to the frozen
[dense-gain comparison](tracking_dense_gain.md). It has two separate stages:
first prove that the exchanged representation can be reduced without changing
any archived decision, then measure what changes when clients contact the
coordinator less often. No prior result or runner is edited.

## 1. Exact compact-exchange audit

The completed programmes at seeds 8100–8102 are replayed on the **full frozen
1,292-controller pool**. No simulator or hyperparameter optimizer may run.
Before returning each archived outcome, the replay checks that the requested
controller, peer provenance and diagnostics match the archived request.
It compares all locally verified recommendations and all allowed D/E
mean/variance predictions at BO decisions, including sharing-mixture weights
and shared coefficient covariances.

D previously exchanged dense Gaussian information matrices and a coefficient
posterior. Here the coordinator relays each new eligible trial summary and
each recipient reconstructs the unchanged exact shared/local posterior.
The public input index identifies **both the gain and context**. The fixed
manifest and pinned input array define its mapping. No dense matrices are
sent in either direction.

| Application item | Exact little-endian format | Bytes |
|---|---|---:|
| Packet header | uint32 version, channel, publication round, count | 16 |
| Eligible-cost record, D only | uint32 client, trial, public input; float64 J | 20 |
| Peer-catalogue record, all arms | uint32 client, trial, public input, source-mode code; float64 J | 24 |

Client ordinals refer to the pinned client ordering; trial numbers start at 1.
The source-mode code preserves factory/peer/BO provenance. Every channel at
each contact has one uplink packet per client and one coordinator batch
delivered to every client. Empty packets retain their headers. Bytes are
computed from actual encoded packets; hashes and per-packet lengths are
saved. Downlink counts **every recipient**, including delivery back to the
sender. The catalogue is a full replacement containing the best eligible
controller per sender/context; the cost stream contains only new eligible
trials. Duplicate input locations from different clients remain separate
observations. Sender/trial IDs prevent duplicate conditioning. Conflicting
records with the same ID are rejected. Failed-test labels remain private.

The former communication numbers were an **analytical payload reference**,
including 12-byte peer entries without this explicit provenance. Report
those separately from the new application packets. The audit also compares
the legacy model estimate plus the **same new peer application traffic**
against the compact protocol, isolating the cost-channel reduction. Neither
comparison measures a real transport stack: IP/TLS, routing, acknowledgements,
retries, latency and public-design provisioning are excluded.

Audit output: `code/results/tracking_communication_v1_equivalence/`.
The online pilot is gated on a successful audit. `run` performs it
automatically, and a valid completed audit is reused.

## 2. Fresh matched online experiment

| Quantity | Fixed choice |
|---|---|
| Contact interval k | 1, 2, 4 rounds; both channels, every arm |
| Fresh programme/noise seeds | 9100, 9101, 9102 |
| Clients | Nine homogeneous compact vehicles |
| Requests per client/arm/interval | 24, including factory and failed tests |
| Paid total | 9 jobs × 9 clients × 24 × 4 arms = **7,776** |
| Controller pool | The frozen 1,292 gains; K_y ≤ 0.15, K_psi/K_r ≤ 1 |
| Contexts | Six speeds 20/22/24/26/28/30 m/s × three frequencies 0.1/0.3/0.6 Hz |
| Initial histories | Empty; one charged factory test |
| Reference selection | Reuse the archived calibration bank, evaluator only |
| Fresh validation | Four common repeats of the union of all paid controllers and references across all intervals/arms/seeds |

The plant, noise, trajectories, 25-second rollout, cost, actuator limits,
eligibility checks, exact GP prior and acquisition settings remain frozen.
The objective is lateral-error RMSE / 0.25 m plus steering-rate RMS /
(10 degrees/s). The common peer-first rule tests an untried cached donor
gain in the assigned exact context before making a BO proposal. A peer
test still consumes one request and can fail; a controller must be tested
locally before becoming a local recommendation.

| Arm | Cost information available to its BO model | Beta |
|---|---|---:|
| A | Own observations; adaptive local GP | 2 |
| B | Own observations; original 65-coefficient local cost mean plus adaptive GP | 2 |
| D | Delivered peer eligible costs plus all own latest eligible observations; exact shared/local GP | 0.5 |
| E | Own observations; exact independent GP with D's marginal prior | 0.5 |

All arms use a local eligibility penalty and the same contact interval for
their own independent peer catalogues. They never borrow another arm's
discoveries. **Local data remain current between contacts.** In round r,
remote records cover rounds through `k * floor((r - 1) / k)`; new own
observations are available after every round. D reconstructs its posterior
from that causal view. It never reads an unpublished peer history. A/B/E
receive peer gains but no peer cost-training stream. At k=1 the compact
protocol matches the audited reference semantics.

Each interval is run afresh. Replaying a fixed old decision sequence would
not measure the effect of stale remote data on future decisions. Schedules,
noise and public priors are paired across intervals and methods. Physical
duplicate simulations within a job can be cached, while every arm's request
remains charged. One worker handles a whole seed/interval job; process pools
parallelize independent jobs and validation batches. BLAS uses one thread
per worker. With three workers, the pilot runs nine jobs in waves.

## Readout

The primary question remains **D versus A**, then **D versus B**, with **D
versus E** isolating added shared cost information under the same peer rule.
Report these at each k, and paired changes from k=1 for every method.

The main success metric averages near-reference success within each client's
encountered contexts, then equally across clients and seeds. Missing and
ineligible recommendations in encountered contexts remain failures.
Unvisited contexts, failed paid requests, complete-policy coverage and
per-seed effects remain visible. Conditional cost comparisons state their
common complete eligible-policy denominator. No noninferiority threshold
or statistical confirmation is asserted for this three-seed pilot.

The report compares success with application bytes and packet deliveries.
It separates traffic available **before the final paid decision** from all
traffic including the final unused contact. Intervals divide the 24-round
cap, so pending eligible summaries are delivered in that last flush.
Batching reduces repeated catalogues and header/contact overhead; it need
not reduce the number of distinct new eligible-cost records. Changed
decisions can also change eligibility counts. Coordinator broadcasts count
generated batches, whereas packet deliveries count each recipient.

Proposal and serialization/cache-update times are recorded separately.
There is no simulated network waiting time. Recipient reconstruction moves
computation to clients, which the timings make explicit. This stage tests
periodic synchronization, with no asynchronous, loss or latency experiment.

## Run locally

From the repository root with the existing NumPy/SciPy/Pandas/Matplotlib
environment; no additional dependency is needed:

```powershell
git fetch origin
git switch experiment/compact-communication
git pull --ff-only
python code/scripts/tracking_communication.py run --workers 3
```

Optional commands:

```powershell
# Archive-only audit; zero new rollouts
python code/scripts/tracking_communication.py audit --workers 3
# Design/audit gate only; no new physical simulation
python code/scripts/tracking_communication.py preflight --workers 3
# Separate software smoke: 3 clients, 8 requests, 169 gains, all three intervals
python code/scripts/tracking_communication.py smoke --workers 3
```

The smoke contains 288 paid requests and two fresh validation repeats;
it is never collected as pilot evidence. The equivalence audit always uses
the full frozen pool, even when invoked by smoke.

Rerun the same `run` command after interruption. Atomic round checkpoints
preserve observation caches, contact queues, packet journals and logged GP
fits. Validation batches are independently resumable. Code, design or
configuration changes invalidate an existing result signature; do not
silently mix outputs from different implementations. Analysis reconstructs
paid choices and packet contents without fresh simulation or optimization.

Main output: `code/results/tracking_communication_v1/report.html`, alongside
CSV tables, source fingerprints, pinned inputs, checkpoints, raw paid
observations, validation batches and replay verification. The validation
count is known after the online runs, since the union depends on the actual
paid decisions. No new dense calibration bank is generated.

## Commit results

`collect` reruns the read-only checks, regenerates the report and stages only
the pilot and its equivalence-audit directory. It does not run new rollouts.

```powershell
python code/scripts/tracking_communication.py collect --workers 3
git diff --cached --stat
git commit -m "results: compact communication and matched synchronization pilot"
git push origin HEAD
```

## Implementation checks

```powershell
python -m unittest discover -s code/tests -p "test_tracking_communication.py" -v
```

Checks cover bit-preserving serialization, malformed/provenance-invalid
packets, repeated-input observation identities, unpublished-peer isolation,
fresh own data, matched contact schedules, incremental eligible-only exchange,
JSON checkpoint resume, packet/timestamp corruption detection, read-only
decision replay, fresh common validation and evaluator isolation.

Verified before release: six new checks and five existing dense-gain checks
pass. The archive audit reproduces all **2,592 paid decisions** and **437
posterior queries covering 564,156 candidate predictions**, with zero observed
mean, variance or mixture-weight differences. D's mean cost-channel traffic
falls from the legacy 12,309,376-byte payload estimate to **44,024 actual
application bytes**; total compact application traffic averages 449,368 bytes,
of which 405,344 are the peer catalogue. These exclude the unused final flush.
The separate 288-request smoke completes with 186 fresh validation rollouts
and no unresolved requests. A cached rerun succeeds with the simulator and
hyperparameter optimizers disabled. These are software/equivalence checks;
the fresh 7,776-request pilot remains to be run locally.
