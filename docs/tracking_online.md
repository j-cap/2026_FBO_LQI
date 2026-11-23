# Online federated contextual controller calibration

Prepared 23 November 2026. Branch: `study/online-contextual-calibration`.
Configuration: `code/config/runs/tracking_online.json`.

## Question and scope

Can a fleet calibrate contextual tracking controllers more efficiently by sharing
a cost model while generating its own calibration observations?

Every method starts with empty calibration histories, no trained cost model and
no learned catalogue. The permitted common knowledge consists of the fixed
controller structure, factory gains, search bounds, reference programme and
declared learner settings. All new observations come from requested nonlinear
rollouts. The previous cost arrays, selected gains and source GP means are never
provided to an optimizer.

This is a focused development pilot, with two matched physical conditions and
three experiment seeds. It tests online information sharing, not centralized
selection of which vehicle should explore which region. The old experiments and
their result signatures remain frozen.

## Physical task and paired fleets

Reuse the CFL-FF six-state nonlinear bicycle implementation, Pacejka tire forces,
friction-circle coupling, longitudinal control, controller clipping/rate limits,
25-second reference duration and numerical-resolution checks described in
[tracking_diagnostic.md](tracking_diagnostic.md).

The controller tunes Ky, Kpsi and Kr directly. The common 71-candidate pool spans
the existing log-multiplier bounds. The six reference contexts cross 20/30 m/s
and .1/.3/.6 Hz at peak reference lateral demand 2.5 m/s². The objective remains
true lateral RMSE divided by .25 m plus applied-steering-rate RMS divided by
10 deg/s. Existing completion, state-bound and numerical-resolution criteria
determine eligibility. Factory gains are an initial test point, not a presumed
eligible controller across all contexts.

| Condition | Physical clients | Purpose |
|---|---|---|
| Homogeneous | Nine copies of compact/nominal | Check online information sharing without physical mismatch |
| Heterogeneous | Existing compact/sport/SUV × nominal/payload/front-tire configurations | Check sharing with vehicle differences |

Client IDs, seed assignments, query caps, reference programmes and initialization
rules match between conditions. Different homogeneous clients have independent
measurement-noise streams; they do not repeat identical noisy observations.

## Context programme and initialization

Each client has a seed-dependent sequence of four six-context cycles. Within
each cycle every reference appears once, in a predetermined random order. The
sequence advances when that client executes a query. A waiting scout follower
starts the same programme later; it does not receive additional contexts or
tests. Context availability is fixed before any outcomes are inspected.

For each client, the first six blind gain choices are the factory gain followed
by five seeded choices without replacement from the other 70 candidates. These
actions and their observations are identical across the five non-scout methods.
This initialization provides gain diversity without any historical outcome
selection. It is charged in full. Subsequent gain queries are adaptive, and an
already tested gain/context pair cannot be repeated on the same client/method.

Measurement seeds are deterministic hashes of experiment seed, client ID and
local query number. Different clients and queries have independent noise. A
method requesting the same counterfactual experiment receives the same noise as
other methods. The simulator's internal stable reference-ID hash also applies.

At the start of a calibration round all client actions are chosen using only
completed earlier rounds. The new outcomes become shareable after the whole
round is complete. Gains stay fixed throughout each 25-second episode; there is
no within-trajectory gain adaptation.

## Methods and what they isolate

| Method | Cost prior mean | Additional information/behavior |
|---|---|---|
| `independent_bo` | Zero | Local contextual GP and local feasibility updates |
| `local_cost_bo` | Same fixed-feature model fitted only to own records | Controls for adding a learned mean without collaboration |
| `online_gain_sharing` | Zero | Each peer's current best eligible gain/cost per context |
| `single_scout` | Zero | One prespecified scout calibrates first, followers verify and adapt |
| `federated_cost_bo` | Online federated fixed-feature regression | Model updates from current local histories |
| `centralized_cost_bo` | Exact pooled fit of the same regression objective | Information/optimization reference |

All methods retain a local GP residual, local feasibility learning and the same
context-restricted acquisition rule. The residual GP sees each actual local
observation once. Shared predictions are prior means, not extra observed trials.
Recommendations must always be best observed eligible target gains. Peer/scout
gains enter a recommendation only after a target verification experiment.

The shared regressor has 64 public random Fourier features for a Matérn-5/2
kernel, plus a trainable intercept: 65 trainable coefficients. Features depend
only on the public five-dimensional gain/context coordinates and a fixed seed.
For eligible observations, labels are log of individual episode cost. The
ridge parameter is .02. Centralized and local regression solve exactly the same
regularized objective on their available records.

Federated training uses full participation of clients with eligible observations,
sample-count-weighted FedAvg, 200 server rounds after each calibration round,
one full-batch local gradient step, learning rate .2, and the previous model as
initialization. There are no new physical tests during these training rounds.
Client updates contain model vectors and counts; raw calibration rows are not
passed into server averaging. This is a co-located simulation of the federated
algorithm, not a deployed network or formal privacy mechanism.

An offline diagnostic records the nonnegative quadratic objective gap from the
federated weights to the exact pooled ridge optimum on the federated arm's own
current records. That diagnostic cannot influence querying. It separates a
possible optimization failure from a failure of information sharing. The
centralized arm can collect different later data because acquisition is adaptive.

The local GP uses the previous frozen Matérn configuration and target-only
hyperparameter refitting policy. Acquisition maximizes local feasibility-weighted
expected improvement among unqueried gains in the scheduled context; if that
context lacks an eligible incumbent, it uses the existing feasibility-search
heuristic. The historical feasibility prior stays neutral at .5 in every arm.

The learned mean is empirical Bayes: shared fitting can include the client's
own observations. Residual covariance and feasibility weights are working
surrogates, not calibrated posterior or safety guarantees.

### Simple sharing baselines

Online gain sharing publishes only each peer's lowest-cost eligible incumbent in
each context. After blind initialization, a client tests the lowest-cost published
incumbent it has not already tried in its scheduled context. If none exists, it
uses independent BO. Missing or infeasible source candidates get no free credit.

For single-scout calibration, the scout runs 12 independent-BO tests while the
other clients wait. Its context catalogue is then frozen. Each follower spends
six target tests verifying these gains, with the factory gain as fallback when
the scout has no eligible recommendation. Remaining tests use local independent
BO. The scout continues local BO to its own cap. Scouts are compact/nominal,
sport/nominal and SUV/nominal for seeds 1100/1101/1102 respectively; the same
client indices play the scout role among homogeneous clones. No donor is chosen
using target validation performance.

## Budgets, time and communication

Each client has a final cap of 24 rollout queries. Metrics are frozen at fleet
budgets 54, 108 and 216. These correspond to six, twelve and twenty-four tests
per client for the concurrent methods. The scout baseline can have unequal
client counts at intermediate fleet budgets; its snapshots are taken at the
exact same total query counts, including partial parallel rounds.

All methods eventually spend 216 fleet tests. The concurrent methods need 24
parallel calibration rounds; single scout needs 36, because followers wait for
12 rounds before beginning their 24 tests. This distinction prevents equating
parallel speedup with a reduction in experimental effort.

The full pilot contains six jobs: two physical conditions × three seeds. Each
job contains six methods × 216 charged tests: **7,776 requested online rollouts**.
Paired counterfactual simulations can be batched or cached for computation, but
each method still pays for every requested test. Numerical RK4 refinements are
computation, not additional physical calibration observations.

Communication is recorded as an explicit payload estimate. FedAvg includes
float64 upload and broadcast vectors for each participating client/server round.
Centralized sends each new five-input/cost/eligibility record once. Gain sharing
uses candidate/cost pairs, and scout communication broadcasts its frozen catalogue.
Estimates exclude transport overhead. Budget snapshots record communication
completed before their last query; learning after that query is reported in the
separate learning log. No communication-efficiency advantage is presumed.

## Evaluation and interpretation

Only after complete query traces are frozen, the evaluator opens the old dense
tracking archive. Its calibration-selected finite-pool reference and the two
independent validation-noise repeats score each selected candidate. These tables
are excluded from worker payloads. No historical cost data initialize a learner.

Report contexts within the declared 5% empirical-reference threshold, all-context
vehicle success, whole-fleet success, final-policy eligibility, original-unit J,
failed/unresolved tests, actual client query counts, parallel rounds and payload.
Missing or ineligible recommendations remain unsuccessful. Costs conditioned on
feasibility retain their denominator, and paired cost effects use jointly eligible
pairs only. Early scout followers without tested recommendations remain in the
denominator.

Three seeds provide independent online noise/programme realizations on fixed
plants, not independent fleet draws. Homogeneous clones share the same archived
validation bank, so their scores are dependent. The finite pool, boundary optima,
simulator-truth score/constraints, research budget and task-specific eligibility
limits remain. This pilot does not establish real-world commissioning standards,
certified safety, new-trajectory generalization, or population-level significance.

Read the homogeneous condition first. A centralized benefit without a federated
benefit calls for inspecting the regression optimization gap. A benefit of the
shared model without an advantage over online gain sharing or a scout supports
the simpler strategy. Compare heterogeneous results next to identify mismatch
and negative transfer. The local-cost arm keeps model architecture from being
confounded with collaboration.

## Run, resume and collect

From the repository root in the existing Windows environment:

```bat
git fetch origin
git switch study/online-contextual-calibration
git pull --ff-only origin study/online-contextual-calibration
conda activate paper_FBO_LQI
python -m unittest discover -s code/tests -p "test_tracking*.py" -v
python code/scripts/tracking_online.py smoke --workers 2
python code/scripts/tracking_online.py run --workers 4
start code\results\tracking_online_v1\report.html
```

Smoke uses three clients (one per family in the heterogeneous case), both
conditions, one seed, eight queries per client and fleet budgets 18/24. Its
single-scout warm-up is six queries. It retains the full reference duration,
71 candidates and all numerical-resolution checks: 288 charged rollout requests.
It verifies software and cannot establish the scientific findings.

The process pool uses Windows-compatible spawn on all platforms. Checkpoints are
atomic after each completed calibration round and include observations, model
weights, GP parameters and the counterfactual simulation cache. Repeating `run`
validates and resumes them. Code/configuration/input fingerprints reject mixed
runs; presentation code is recorded separately.

```bat
python code/scripts/tracking_online.py status
python code/scripts/tracking_online.py collect
git status --short
git commit -m "Add online contextual fleet calibration pilot results"
git push origin study/online-contextual-calibration
```

Collection rejects incomplete and smoke campaigns, verifies observation records,
context programmes, local/fleet counts, timing, incumbents and common initial
observations, and stages only the new full result directory. The earlier
diagnostic and historical-transfer campaigns are unchanged.
