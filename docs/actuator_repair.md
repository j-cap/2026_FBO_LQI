# Step 2 correction: optimizer audit, episode replay, broader benchmarks

Branch: `experiment/actuator-diagnostic-repair`. Prior result commit: `b4c299d`.
This study leaves all v1/v2 code and results unchanged. It uses the same six
physical vehicles, known actuator parameters, bounds, and constraints. It does
not run a fleet-informed target BO comparison.

## A. Optimizer audit

Use the expanded v2 calibration bank (20 episodes), fixed tau=0.20 s, optimizer
seeds 101/202/303, and checkpoints at 25/50/75 proposed evaluations. Compare:

| Method | Repeated queries | Infeasible training value | EI margin |
|---|---|---:|---:|
| bo_original (read existing v2 history) | allowed | 10 | 0.01 |
| sobol | excluded | not used for selection | not applicable |
| bo_unique | excluded | 10 | 0.01 |
| bo_zero_margin | excluded | 10 | 0 |
| bo_lower_penalty | excluded | 1 | 0.01 |
| bo_both | excluded | 1 | 0 |

All methods share the original first two Sobol points and fixed 256-point pool.
Sobol then visits that pool in sequence; the BO variants select among its unused
points. GP settings and objective scaling (cost/r_max) remain unchanged. This
isolates repeat suppression and crosses two numerical choices; it is not a new
BO method or a replacement for explicit constrained optimization. The penalty
values are diagnostic choices, not physical costs. Infeasible controllers can
never become recommendations, regardless of the penalized GP prediction.

There are 90 new trajectories, at most 6750 proposed evaluations. Exact score
cache hits avoid duplicate simulation across methods; each nominal evaluation
still counts toward its method's budget. Every completed query is checkpointed,
so interruption resumes from the last completed optimization query. The original
method is assessed using its saved history, without rerunning the optimizer.
No best-of-three selection: retain each optimizer seed as a separate repeat.

## B. Episode-level replay (development data)

Replay every previously selected reference at every tau, and the 75-evaluation
expanded-bank source recommendation for each optimizer seed. Use the OLD v2 test
bank explicitly as development data, not new validation evidence. For every
episode, report cost, sideslip/steering/rate peaks, saturation fraction, failure
reason, and 5-versus-10-substep consistency. Unlike the bank-level objective,
replay continues to subsequent episodes after a failed episode.

Save full trajectories at both resolutions for the first failed episode and two
highest-cost feasible episodes. Time series include beta, yaw rate, actual and
commanded steering, raw steering command, reference, noisy tracking error, and
actuator rate. They terminate at the original sideslip/nonfinite abort boundary.
Selection of example trajectories is explanatory only, never used for tuning.

## C. Broader reference candidates

For each of the 30 vehicle/tau conditions, evaluate the union of:
- 256 new Sobol candidates (seed fleet_seed+1900);
- the original v2 reference shortlist;
- calibration-selected incumbents from ALL original and new optimizer methods,
  seeds and budgets (not test-selected winners), recomputed at the current tau;
- five fixed anchors [0,0,q_int] with q_int=-3,-2,-1,0,1 in log-weight coordinates.

Screen on the same 20 calibration episodes. Select a fixed shortlist of up to
32 feasible candidates: half lowest calibration cost, half farthest-point
coverage in normalized weight space. The latter includes candidates outside the
narrow near-optimal region. Lower integral weights are only conservative trial
candidates, not assumed safe controllers.

Validate on a NEW bank of 80 episodes, seed 1500000. Require feasibility and
5/10-substep relative cost agreement <=0.1%. Select lowest validation cost among
eligible candidates; if none qualify, keep the condition unresolved. Every
candidate and origin is recorded. This deliberately finite search produces a
validated BENCHMARK, not a global-optimality certificate. It does not inherit
the inadequate v2 grid-based `reference_usable` optimality interpretation.

## D. Fresh final assessment

Only the separate `test` stage may access the NEW test bank (80 episodes, seed
1600000). Assess the chosen benchmarks and every optimizer method/seed/budget
recommendation. Recommendations depend on calibration only; final test data never
select a replacement controller. Verify all bank seed sets are disjoint for each
vehicle. Banks change noise realizations; the two maneuver shapes stay unchanged.

Export absolute costs and feasibility alongside 5% gaps to the matched benchmark.
Retain negative gaps and distinguish unresolved reference/integration outcomes.
A substantially negative gap triggers a benchmark review. Do not replace it using
test winners. New benchmark episode summaries and example trajectories are saved
as well. Two fleets support descriptive diagnostics, not population-level claims.

## Run from the repository root (Windows CMD or PowerShell)

```bat
git fetch origin
git switch --track origin/experiment/actuator-diagnostic-repair
conda activate paper_FBO_LQI
python code/scripts/actuator_repair.py preflight
python code/scripts/actuator_repair.py smoke --n-jobs 1
python code/scripts/actuator_repair.py audit --n-jobs 2
python code/scripts/actuator_repair.py replay --n-jobs 2
python code/scripts/actuator_repair.py references --n-jobs 2
python code/scripts/actuator_repair.py test --n-jobs 2
```

Run commands sequentially. If the branch already exists locally, use
`git switch experiment/actuator-diagnostic-repair` and `git pull --ff-only`.
The main run uses `code/results/actuator_repair_v3`; smoke has its own namespace.
This remains a substantial simulation workload. It prints stage/vehicle starts,
optimizer checkpoints, candidate counts, validation progress, elapsed time and
cache hits/new evaluations. Heartbeats occur after a score/episode finishes,
nominally every 30 seconds; one long evaluation can delay them. Six vehicle jobs
can be parallelized; increasing jobs trades memory/CPU usage against elapsed time.

Restart an interrupted stage with the SAME command. Keep cache/partial files
locally. Do not edit code/config during a run: hashes protect resumability. Old
v2 files are fingerprinted; the runner verifies the original code hashes before
using its histories. Re-running test resumes assessment; it never chooses a new
benchmark from test outcomes. `test` automatically generates processed CSVs.

## Commit and push (ignore-rule and Windows-safe)

```bat
python code/scripts/actuator_repair.py collect
git diff --cached --stat
git commit -m "Add actuator optimizer audit and reference repair results"
git push -u origin experiment/actuator-diagnostic-repair
```

`collect` force-adds only this run's protocol, completed audit/reference/assessment
JSON, episode summaries/trajectories, and processed CSVs. It excludes disposable
cache/partial directories, smoke files and unrelated workspace files. It does not
commit or push; inspect the staged list first. To regenerate tables, run:

```bat
python code/scripts/actuator_repair.py analyze
python code/scripts/actuator_repair.py collect
git diff --cached --stat
git commit -m "Refresh actuator repair analysis tables"
git push
```

## Analysis decisions

1. Compare paired calibration and fresh-test costs at equal budgets: does Sobol
   beat the existing BO; what changes after repeat suppression; are penalty and
   margin effects consistent across both fleets?
2. Explain the sideslip failures and anomalous payload cost from episode traces.
   Check whether integration sensitivity, saturation, or specific noise/maneuver
   realizations accompany them; do not infer causality from correlation alone.
3. Check whether broader candidates validate and survive fresh testing. If they
   do not, investigate the controller/task/constraint formulation before expanding
   the transfer campaign. Do not keep increasing search budgets without diagnosis.
