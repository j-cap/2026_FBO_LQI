# Online controller calibration from one factory test

Prepared 23 November 2026. Branch: `study/online-calibration-lcb`.
Configuration: `code/config/runs/tracking_online_lcb.json`.
Results: `code/results/tracking_online_lcb_v1/`.

## Question

Can a federated contextual cost model reduce controller-calibration effort when
all observations are generated online and collaboration can influence the
second vehicle experiment?

The completed six-test pilot remains frozen. This experiment changes its
initialization and acquisition together. Comparisons among methods within this
experiment share the new protocol; a difference from the old pilot cannot
attribute an effect to either protocol change alone. No existing historical
observations, catalogue, or learned weights initialize any method.

## Changes and fixed settings

| Item | New protocol |
|---|---|
| Concurrent initialization | One factory-gain rollout per client, charged in full |
| First adaptive query | Local query 2, in calibration round 2 |
| BO acquisition | Cost LCB plus negative-log local eligibility penalty, including when no eligible incumbent exists |
| Early fleet snapshots | 9, 18, 27, 54, 108, 216 tests |
| Final local cap | 24 tests on every client, including scout followers |
| Eligibility reporting | Context coverage, first eligible observation, first full context coverage, explicit censoring |

Reuse the physical model, references, gain candidates, cost/eligibility rules,
seeds, client noise, six methods and cost-model settings of
[tracking_online.md](tracking_online.md). The underlying nonlinear bicycle
model and numerical-resolution checks are described in
[tracking_diagnostic.md](tracking_diagnostic.md).

Specifically, the study retains nine clients, six contexts (20/30 m/s crossed
with .1/.3/.6 Hz at 2.5 m/s² reference lateral demand), a 71-point common pool of
three feedback gains, and 25-second episodes. One condition has nine identical
compact/nominal plants; the other has the existing three vehicle families and
their nominal/payload/front-tire configurations. Seeds are 1100/1101/1102.
The full campaign is six jobs and **7,776 charged online rollout requests**.

## Round protocol and acquisition

Each client's predetermined sequence of randomly ordered six-context cycles
matches the frozen pilot exactly. Its first test uses the factory gains in the
first available context. A failed factory test still consumes a query and
updates local eligibility learning; no feasible initial controller is assumed.
All five concurrent methods receive the same initial observation for each
client. From the second query, gains are selected adaptively in the currently
scheduled context. There is no requirement to observe all contexts or collect
a successful trial first.

All actions in a round are frozen before any outcome from that round is
exposed. After round 1 the federated and centralized arms train their respective
cost means on their new eligible observations. These means are available for
round 2. Context coverage can remain sparse at that point. Gains stay fixed
inside each episode; the online update occurs between episodes.

At every BO query, minimize over untried gain/context pairs in the scheduled
context:

\[
 a(\theta,c)=\mu(\theta,c)-2\sigma(\theta,c)
 -\log\!\left(\max\{10^{-6},\widehat p_{\mathrm{eligible}}(\theta,c)\}\right).
\]

The GP mean and standard deviation refer to **log J**. The prior mean is zero,
locally learned, federated, or centrally learned according to the method. The
residual GP uses actual eligible local observations only. The eligibility
surrogate uses all local outcome labels and the same neutral .5 prior in every
method. Its negative-log term penalizes unlikely eligibility, and the LCB term
uses the learned cost mean even without a local incumbent. The exploration
coefficient 2, penalty coefficient 1 and probability floor 1e-6 are fixed for
this pilot. They are not separately tuned for any method. Seeded ties are
deterministic. An already tested local gain/context pair cannot be repeated.

The fixed-feature regressor, ridge .02, 64 public random Fourier features plus
intercept, conventional sample-weighted FedAvg (200 server iterations, E=1,
learning rate .2), and same-data optimization-gap diagnostic are unchanged.
Failed/ineligible costs do not train the cost regressor. The server shares
model vectors rather than raw rows in the federated algorithm. This is a
co-located simulation, not a network deployment or privacy guarantee.

## Methods and comparator timing

| Method | Behavior after initialization |
|---|---|
| `independent_bo` | Local GP with zero prior mean and the new acquisition |
| `local_cost_bo` | Same acquisition with a locally fitted cost mean |
| `online_gain_sharing` | Test an untried peer incumbent if available in the scheduled context; otherwise use independent LCB BO |
| `federated_cost_bo` | Same acquisition with the updated federated cost mean |
| `centralized_cost_bo` | Same acquisition with an exact pooled fit of the same regression objective |
| `single_scout` | One scout learns first, then followers verify its frozen catalogue and adapt |

Gain sharing can propose peer gains from round 2. It publishes only the current
best eligible gain and observed cost per peer/context. The client still pays
for its own verification, and only actual locally eligible observations can
become recommendations. A peer candidate is not an extra observation in the GP.
It is the same separate gain-sharing comparator as before.

Single scout intentionally retains its sequential structure: the prespecified
scout runs 12 queries while followers wait. Its first query is the factory test;
the next 11 use LCB BO. The catalogue is then frozen. Each follower's first six
queries verify these context-specific gains, using factory gains when a context
has no scout incumbent; remaining queries use LCB BO. These six verification
tests are part of that comparator, not shared blind initialization. All clients
eventually spend 24 queries, so this method needs 36 calibration rounds rather
than the concurrent methods' 24. The scout's selected client index cycles
through compact/sport/SUV nominal across the three seeds, as before.

## Evaluation and accounting

Freeze recommendations at total fleet budgets 9/18/27/54/108/216. The five
concurrent methods have 1/2/3/6/12/24 queries per client at these snapshots.
Scout counts can be unequal, and its snapshots use the same exact fleet
budgets, including a partial parallel round where needed. Every rollout is
charged, including failures and verification. Counterfactual simulation caching
only saves computation. Noise remains paired across methods for the same
client/local-query index and independent across clients/queries.

Report all six contexts for every client at every snapshot. Missing
recommendations count as zero coverage and zero near-reference success; they
are not dropped. Distinguish:

- **Online eligible-context coverage:** fraction of contexts with at least one
  actually observed eligible local controller.
- **Validation eligible-context coverage:** fraction with a recommendation that
  also satisfies the frozen independent validation criteria.
- **Near-reference success:** validation cost within 5% of the finite-pool
  empirical reference, with missing/ineligible recommendations unsuccessful.

Also report all-context/fleet success, cost only with its eligible denominator,
failed/unresolved/ineligible tests, actual local counts, calibration rounds and
declared communication payloads. Cost deltas use jointly eligible pairs. Model
updates after a snapshot's last query are logged separately and are excluded
from that snapshot's payload count because they did not inform its queries.

First-eligibility metrics use online observations rather than validation labels.
`eligibility_times.csv` gives each client's first eligible local query/round and
first query/round at which all contexts have an eligible local gain.
`context_eligibility_times.csv` gives the same first-event times per context.
Missing events are explicitly censored at the query cap, with final round
available for the elapsed-round clock. The timing summary displays observed
and censored counts alongside conditional means. A method with more censoring
must not appear faster merely because only its successful clients are averaged.

The evaluator opens the dense archive only after all query traces are complete;
workers receive no historical outcomes or evaluation archive path. The two
frozen validation-noise repeats and calibration-selected empirical reference
are unchanged. Three seeds on fixed plants are a development pilot, not
independent-fleet confirmation. Read homogeneous results first, then examine
heterogeneous transfer and the simple-sharing comparators. The acquisition and
eligibility surrogates do not establish certified safety or calibrated
uncertainty.

## Local execution, resume and collection

From the repository root in the existing Windows environment:

```bat
git fetch origin
git switch study/online-calibration-lcb
git pull --ff-only origin study/online-calibration-lcb
conda activate paper_FBO_LQI
python code/scripts/tracking_online_lcb.py smoke --workers 2
python code/scripts/tracking_online_lcb.py run --workers 4
start code\results\tracking_online_lcb_v1\report.html
```

Optional regression check:

```bat
python -m unittest discover -s code/tests -p "test_tracking*.py" -v
```

Smoke retains the full physics and gain pool but uses three nominal clients,
one seed, eight queries/client, both conditions, and snapshots 3/6/9/18/24.
Its scout waits six queries, then followers verify. There are 288 charged
rollout requests. These results check software, not scientific effects.

The runner uses process-pool spawn on every platform. Round checkpoints are
atomic and include observed outcomes, models, GP parameters and the simulation
cache. Repeating `run` resumes completed rounds without recharging old queries.
Source/configuration/archive signatures prevent mixed runs; CRLF-normalized
source hashes support Windows checkouts. Report provenance is stored separately.

```bat
python code/scripts/tracking_online_lcb.py status
python code/scripts/tracking_online_lcb.py collect
git status --short
git commit -m "Add one-test online federated BO results"
git push origin study/online-calibration-lcb
```

`collect` requires the complete full campaign and validates timing, query
counts, observed recommendations, early snapshots, scout verification,
prior-round peer proposals and simulator records before staging only the new
result directory. Smoke/incomplete collection is rejected.

## Preparation checks

On 23 November 2026, all **76 tracking tests** passed, including 13 checks for
the new protocol. Both nonlinear smoke jobs completed using two spawned
workers: 288 charged queries, 187 unique counterfactual simulations and zero
unresolved integrations. Repeating the smoke command found both jobs complete
and did not generate further trials. The report and coverage plots were checked.

A synthetic-oracle canary using the full nine-client, 24-query configuration
verified every 9/18/27/54/108/216 snapshot, all six methods' 216-query totals,
and the concurrent/scout 24/36-round completion times. CLI collection guards
rejected smoke and incomplete campaigns without staging files. Full preflight
passed with all six physical jobs still unexecuted. These are software and
protocol checks; scientific conclusions await the full local campaign.
