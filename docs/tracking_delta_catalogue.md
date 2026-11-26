# Lossless catalogue updates and fixed programme confirmation

This follows the [communication pilot analysis](analysis/communication/report.md).
The immediate implementation question is whether changed catalogue entries
can preserve every controller decision. The fresh statistical question is
whether D's advantage over A, then B, and matched E repeats on new programmes
once every arm uses that efficient communication representation.

## Lossless catalogue protocol

Each sender keeps the last catalogue it delivered. At a contact it computes
its current best eligible controller for every encountered context and sends
only entries that differ. A new context or an improved local incumbent
creates an entry; an unchanged incumbent creates no record. The coordinator
and every recipient apply those entries to persistent caches. The recipient
catalogue is therefore the same complete catalogue that full publication
would have supplied at that contact.

| Channel | Record | Header | Publication |
|---|---|---:|---|
| Peer catalogue, every arm | uint32 client, trial, public gain/context input, source-mode code; float64 J (24 bytes) | 16 bytes, version 2 | Changed entries only |
| Eligible costs, D only | uint32 client, trial, public input; float64 J (20 bytes) | 16 bytes, original version 1 | Every new eligible trial once |

Version 2 distinguishes the incremental catalogue from the original full
publication, without increasing the packet size. Source-mode and donor/trial
provenance are retained. Float64 scores are not quantized. Repeated public
inputs from different clients remain different cost observations. Failed-test
labels remain private and consume the local paid budget.

The fixed coordinator topology and reliable scheduled delivery are retained.
Every contact still charges one uplink packet per client and one coordinator
packet delivered to every client, per channel. Empty packets retain their
headers. Sending fewer entries saves **bytes**, while packet and contact
counts stay fixed. Header and recipient counts therefore match the earlier
changed-entry diagnostic. IP/TLS and other transport overhead are outside
this application-level experiment.

The fixed fleet has no catalogue deletions or cache resets. Inconsistent
sender/coordinator/recipient caches, repeated contact times and unsupported
removals are rejected. All cache states are in atomic checkpoints, so ordinary
process interruption/resume preserves them. Rejoining clients, packet loss
and reset/recovery protocols would require a separate design.

The statistical planner is reused directly from the frozen communication
implementation. The new runner and protocol modules leave the old hashed
sources and results intact. Own data remain current after every round;
remote information is held between contacts exactly as before.

## Gate 1: replay the archived controllers

Before any fresh online run, replay all nine archived seed/interval jobs:
seeds 9100–9102, intervals 1/2/4, and all four methods. The gate checks:

- All **7,776** controller requests, peer provenance, GP diagnostics, local
  histories and locally verified recommendations agree with the archived run.
- At each contact, a shadow full publication reproduces the archived packet
  hashes. Every new recipient cache matches that complete catalogue.
- D's eligible-cost packets and the cost-training cache stay unchanged.
- Packet/contact counts stay unchanged; actual encoded byte savings are saved.

The simulator and hyperparameter optimizers are disabled during replay.
Archived outcomes are returned only after the new request matches the old
one. Audit inputs are in separate workers; fresh online workers receive no
old histories, evaluator labels or reference winners.

Output: `code/results/tracking_delta_catalogue_v1_equivalence/`. The gate
always uses the full 1,292-controller domain, even when invoked by smoke.
`run` checks it automatically and reuses a valid completed audit.

## Gate 2: fresh fixed confirmation

| Quantity | Choice fixed before these outcomes |
|---|---|
| Programme/noise seeds | 10100–10109, ten fresh programmes |
| Contact intervals | 1 and 2 rounds, both channels, all arms |
| Primary operating point | Two-round contact |
| Clients | Nine homogeneous compact vehicles per programme |
| Paid requests | 24 per client/arm/interval, factory and failures included |
| Total | 20 jobs × 9 clients × 24 × 4 arms = **17,280** |
| Public domain | Frozen 1,292 gains and 18 speed/frequency contexts |
| Validation | Four fresh common repeats of the union over all programmes/arms/intervals |
| Reference selection | Reuse archived calibration, evaluator only; no new dense calibration bank |

There is no kernel, gain bound, acquisition-bonus, context, plant, trajectory,
noise, cost or eligibility change. A/B beta=2, D/E beta=0.5 remain fixed.
Histories start empty. A is the adaptive local GP; B adds the original local
mean; D is the exact shared/local contextual GP; E has the matched independent
marginal prior. Every arm has the same paid peer-first verification rule,
using only its own arm's discoveries and the locally cached exact context.

Declare comparisons in order: **D2 versus A2**, then **D2 versus B2**, then
**D2 versus E2** for added shared cost information. One-round versions are
references; paired D2/D1 comparisons describe the effect of less frequent
contact. Do not add k=4 or tune settings after these outcomes.

Primary success remains within 5% of the finite-pool reference, averaged
over encountered contexts within each client and then equally over clients
and programmes. Missing/ineligible encountered-context recommendations stay
failures. Unvisited contexts, failed requests, complete eligible policies,
conditional cost denominators, frequency effects and per-programme effects
remain visible. D's success lead is not automatically a feasibility or
mean-cost advantage.

The report provides paired 95% percentile bootstrap intervals by resampling
the ten **programme-level effects**, not the 90 dependent client observations.
The seed and 20,000 bootstrap repetitions are fixed in the configuration.
These intervals are descriptive; no multiplicity-adjusted significance,
noninferiority or heterogeneous-vehicle claim is declared. Smoke has one
programme and therefore reports uncertainty as unavailable. The output
reports the evidence without forcing a positive D outcome.

## Run locally

Use the existing experiment environment; no new dependency is required.

```powershell
git fetch origin
git switch experiment/lossless-catalogue-confirmation
git pull --ff-only
python code/scripts/tracking_delta_catalogue.py run --workers 3
```

Independent seed/interval jobs run in a process pool; BLAS uses one thread
per worker. Use `--workers 6` if six CPU workers fit your machine. The 20
jobs and validation batches are resumable. Repeating the same command skips
completed jobs and valid evaluator batches. All arms' requests stay charged
when physical duplicate simulations within a job are cached.

Optional first checks:

```powershell
python code/scripts/tracking_delta_catalogue.py audit --workers 3
python code/scripts/tracking_delta_catalogue.py preflight --workers 3
python code/scripts/tracking_delta_catalogue.py smoke --workers 3
```

`audit` and `preflight` spend no fresh control rollouts. The separate smoke
uses three clients, eight requests, 169 gains, two contact intervals and one
fresh programme: **192 paid requests**, plus two fresh validation repeats.
The smoke cannot be collected as confirmation evidence.

Main output: `code/results/tracking_delta_catalogue_v1/report.html`.
`confirmation_readout.json` summarizes D2/A2, D2/B2 and D2/E2 effects with
programme uncertainty. `primary_programme_bootstrap.csv` holds those
intervals; `paired_comparisons.csv` retains conditional cost and coverage
denominators. Communication tables separate traffic before the last paid
decision from all traffic including the unused final contact. The validation
count is determined by the observed candidate union.

## Verify and upload results

```powershell
python code/scripts/tracking_delta_catalogue.py collect --workers 3
git diff --cached --stat
git commit -m "results: lossless catalogue and fixed programme confirmation"
git push origin HEAD
```

`collect` replays the paid choices, regenerates the report without fresh
simulation, and stages only the new confirmation and equivalence-audit
directories. It leaves smoke outputs and previous studies untouched.

## Implementation checks

```powershell
python -m unittest discover -s code/tests -p "test_tracking_delta_catalogue.py" -v
```

Checks cover packet version/size, changed-entry delivery, every receiver's
cache, empty header charges, cache desynchronization and removal rejection,
JSON checkpoint resume, read-only decision replay, packet/timestamp/cache
corruption, fresh validation and learner isolation, and programme-level
uncertainty without client pseudoreplication.

Before release, the six new tests and six existing communication tests pass.
The full archived replay reproduces all 7,776 paid choices and all 504 contact
states with every recipient cache intact. All 36 method/seed/interval byte
totals match the earlier posthoc diagnostic exactly. The 192-request smoke
completes validation and decision/packet replay. A cached rerun also succeeds
with physical simulation and hyperparameter optimization disabled. The
17,280-request fresh confirmation remains for local execution.
