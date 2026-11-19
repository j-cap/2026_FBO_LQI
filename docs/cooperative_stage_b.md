# Stage B: from-scratch discovery comparison

Development study on the existing six vehicles and five actuator bandwidths.
The provisional requirement remains expected measured episode RMSE <= 0.92 deg/s
separately on both maneuvers, with existing constraints and numerical checks.
Passing the earlier frozen-common-weight validation established feasibility for
these cases; it did not establish how cheaply acceptable weights can be discovered.

## Pilot first

The pilot uses optimizer seed 701 and 14 independent algorithm trials:
- Transfer: calibrate each of the six possible source vehicles separately at tau=0.2.
- Repair: repeat those six choices, allowing calibration of a failing condition.
- Coordinated Sobol sampling: one trial.
- Selective GP-based allocation: one trial.

The optional full development scope adds seeds 702 and 703 (42 trials total), reusing
completed seed-701 trials. These are optimizer replications on the SAME systems,
not independent fleet draws. Analyze the pilot before starting the full scope.
No publication-level superiority inference should be drawn from this pilot.

## Common search protocol

Each seed supplies a fresh scrambled Sobol pool of 256 candidate weight vectors
in the original 3D log-weight box. Every method gets the same pool. Neither the
retrospective common weights nor references/historical observations are inserted.
This is a finite-candidate comparison; failure does not prove no continuous feasible
weights exist or that other optimizers cannot find them.

A query tests one (vehicle, bandwidth, weights) combination with 20 noise episodes
per maneuver (40 rollouts). The objective returned to the policy is the larger of
the two sample-mean RMSEs divided by the threshold. Each query also returns a variance
estimate and validity status. Invalid queries receive a fixed surrogate penalty 10.
Fixed sample means are only screening evidence, not statistical acceptance.

Every method has a hard cap of 150 distinct queries, including fleet screening
queries. Cached repeated queries within a trial are charged once; no observations
or simulation-cache savings are shared across methods or donor trials. Initialization
is charged. Each simulated episode is counted once as an experimental rollout; its
5/10-substep numerical pair is reported separately as two numerical rollouts. Failed
and truncated episodes still count. A gain-design failure consumes a query but zero
rollouts, and is logged as invalid. Local model identification is common fixed input
for all methods and is outside this marginal calibration-cost comparison.

Search noises are paired across candidates and methods for each optimizer seed and
condition. Fresh audit noises are disjoint from search and earlier validation noise.
Search uses a fixed noise bank, so overfitting it is possible; the sealed audit is
specifically needed to assess the selected output. Simulated queries are not guaranteed
safe: no real-vehicle deployment claim is made by this pilot.

## Policies

1. Transfer: 25 local BO queries at source tau=0.2, select the lowest-scoring local
   candidate, then screen it across all 30 conditions. Stop screening after a failure.
   Its one final candidate is audited even if screening failed.
2. Repair: same initial calibration. On screening failure, spend up to 10 additional
   local BO queries on the failing condition and select that condition's best observed
   candidate. Repeat until full screening passes or the total cap is reached.
   Prior observations are retained. No fresh audit data can trigger a repair.
3. Coordinated: traverse the fresh Sobol pool and screen each candidate. Reject on
   the first observed invalid result or mean score >1. Stop at a fully screened pass
   or the query cap.
4. Selective: fit an independent fixed-kernel GP per condition. Choose weights that
   minimize the largest GP lower confidence bound over conditions; then query the
   unobserved condition with the largest upper confidence bound at those weights.
   Observed screening failures exclude a candidate. Fully observed passes can stop
   discovery. This is a transparent GP-LCB minimax HEURISTIC, not a reproduction of a
   published SOTA acquisition or a claim of new theory. Cross-condition correlations
   are not learned; the first experiment isolates allocation with scalar sharing.

Local BO uses two initial pool points, then expected improvement. All GPs use a fixed
Matern-5/2 kernel (normalized length scale 0.3, prior SD 0.5 in normalized score units),
prior mean 1, empirical observation variance with floor 1e-5, and no hyperparameter
optimization. The selective confidence multiplier is 2 and has no certification role.
This avoids expensive small-sample hyperparameter fitting and convergence warnings.

Transfer, repair and coordinated sampling use the same deterministic rotating condition
order per candidate. Selective allocation chooses the order by its stated rule.
If coordinated/selective discovery exhausts its cap, its output maximizes the number
of observed passing conditions, with worst observed score and pool index as tie-breakers.
This fallback is audited and is not automatically a failure or a success.

The source tau=0.2 follows the original transfer baseline. A deliberately chosen slow-
actuator source is NOT included in this bounded pilot; any favorable allocation result
must subsequently compete with that informed baseline. All six nominal-bandwidth
source identities are reported, including the strongest, not just a favorable average.

## Sealed final audit and cost accounting

Freeze and write selected.json before opening any audit observations. Exactly one
candidate per trial is audited; no reselection, repair or adaptive sample extension.
Use 256 fresh episodes per maneuver per condition. This count was fixed using the
previous development validation's noise scale and margin, before this pilot's results.
It reduces the common audit cost relative to the earlier 1,000-episode feasibility
study. It does not guarantee that marginal candidates will be classified.

Use two-sided Student-t intervals with Bonferroni alpha=0.05 across 60 means per trial,
widened by the maximum coarse/fine score discrepancy. As before, this is approximate
large-sample inference, not distribution-free certification. Correction is per trial,
not simultaneous across all 14/42 algorithm trials; no winner-selected confidence claim.

Supported requires every upper endpoint <= threshold and all numerical/physical checks
valid. A lower endpoint > threshold gives exceeds. Otherwise retain unresolved.
Invalid data block acceptance. Once a complete fixed-size audited condition rules out
the candidate (exceeds or invalid), the remaining conditions need not be simulated.
This shared early-rejection rule saves cost but does not make a failed trial efficient
in the success comparison. Do not report successful-trial costs alone.

Maximum per trial: 6,000 search rollouts + 15,360 audit rollouts = 21,360 experimental
rollouts (42,720 numerical rollouts). For all 14 pilot trials: at most 299,040 experimental
rollouts. No epoch/time guarantee: runtime depends on simulation hardware and early
rejection. Report search-only cost AND total search-plus-audit cost, success rates,
failures and unresolved trials. A small search saving can be diluted by validation.
The earlier 20% total-cost ambition remains an ambition, not a reason to omit audit cost.

No intermediate validation-backed anytime success curve can be inferred from these
single-output audits. Search histories support descriptive screening-progress plots only.
Models and trajectories stay within the evaluator; the policy consumes only candidate
coordinates, condition indices and scalar observations. This is a simulator separation
of responsibilities, not a deployed federated communication stack or a privacy guarantee.

## Run locally from repository root

```bat
git fetch origin
git switch study/cooperative-calibration
git pull --ff-only
python code/tests/test_cooperative_stage_b.py
python code/scripts/cooperative_stage_b.py preflight
python code/scripts/cooperative_stage_b.py smoke --n-jobs 2
python code/scripts/cooperative_stage_b.py run --scope pilot --n-jobs 4
```

Stop on errors. Each worker runs one independent trial. Reduce workers if sharing the
machine. Progress prints for each query/audited condition and approximately every 30s
inside long evaluations. Atomic checkpoints every 25 episodes allow interrupted runs
to resume by repeating the same run command. Do not run duplicate processes against
the same trial directory. Changed code/inputs/configuration cause a protocol mismatch.
Smoke uses a separate directory and seed 1701 (distinct from pilot/full), one vehicle,
two bandwidths and tiny counts; it is
never scientific evidence. No full pilot was run during implementation.

Results: code/results/cooperative_stage_b_v1.
The pilot result table is processed/pilot_trials.csv. Each trial saves selected.json,
queries/, audit/, raw evaluation blocks and result.json. Cost columns separate search,
audit, total experimental rollouts and numerical rollouts. Keep rejected/unresolved
rows when comparing methods. Existing inputs and model files are available on this branch.

```bat
python code/scripts/cooperative_stage_b.py collect --scope pilot
git commit -m "Add Stage B cooperative discovery pilot results"
git push -u origin study/cooperative-calibration
```

Collection checks the episode/seed ledger and costs and force-adds the ignored results
files. It excludes smoke and unrequested scopes. For later full development runs,
replace --scope pilot with --scope full in run/collect; pilot trials are reused.
Do not start full scope until the pilot is analyzed. The sealed audit from this pilot
must not become free training data for the next method comparison.
