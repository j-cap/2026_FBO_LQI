# Corrected direct-transfer pilot (v6)

Question: can existing source-calibrated LQI weights meet the target performance
criterion without target weight optimization? Identification and model-based gain
recomputation still occur on every target; this is not transfer of fixed gains.

## Frozen protocol

Six existing v5 vehicles serve as targets. For each target, all five other vehicles
are donors, identified by (fleet seed, client ID); exclude the target's own histories
at every bandwidth. This includes same-family donors in the other fleet realization.
All donors have source actuator tau=.2 seconds.

For each donor, retain Sobol and lower-penalty BO histories, seeds 101/202/303,
and budgets 25/75. Choose each incumbent using ONLY that donor's calibration costs.
Do not choose a winning donor or optimizer repeat using target results. Evaluate all
60 source configurations per target at tau=.1,.15,.2,.3,.4 seconds: 1800 transfer
rows. The primary arm is Sobol at budget 25, with all donors/repeats retained.
BO and budget 75 are secondary. Source calibration costs are already paid historical
costs; this experiment makes no claim of reducing total lifetime calibration cost.

Transfer xi (log design weights), then recompute LQI gains from the identified target
model augmented with the target's known actuator tau. Use corrected tracking
back-calculation with T_aw=.2 seconds. No target weight search or adjustment occurs.

Re-evaluate the 30 frozen v5 reference controllers on the same new target episode
banks used for transfer. References are finite candidate benchmarks, not global
optima. The retrospectively identified common vector is a SEPARATE exploratory
baseline; it has been informed by these vehicles and cannot establish independent
transfer success.

New episode bank: seed 1900000, 40 episodes per maneuver, 80 per target/tau/controller.
Check disjointness against prior calibration/validation/test banks. Evaluate every
episode with 5 and 10 RK4 substeps. A successful result requires all episodes feasible,
all integration checks passing (.1% relative cost tolerance), and mean episode RMSE
within 5% (primary) or 1% (secondary) of the reference evaluated on the same episodes.
Keep negative gaps. Unresolved references never count as successes.

The manifest freezes source-only assignments and code/input hashes before simulation.
Each unique target/tau/weight combination is cached, with per-episode checkpoints;
identical weights from multiple donors are simulated once but provenance is retained.
Every episode checkpoint is collected for inspection. Source optimizer repeats,
multiple donors, and shared weights are NOT independent physical systems. This is a
development pilot on six familiar vehicles, not a population-level guarantee.
Do not claim statistical significance from the 1800 correlated rows. Inspect results
by target, family, donor and bandwidth, including failures and worst gaps.

## Run from repository root (Windows CMD, activated paper environment)

```bat
git fetch origin
git switch --track origin/experiment/actuator-direct-transfer
python code/scripts/actuator_direct_transfer.py preflight
python code/scripts/actuator_direct_transfer.py smoke --n-jobs 2
python code/scripts/actuator_direct_transfer.py run --n-jobs 2
```

If the branch already exists, use `git switch experiment/actuator-direct-transfer`
and `git pull --ff-only`. Stop on errors. Smoke uses one target, tau=.2, two donor
configurations and two episodes, stored separately. The full run performs evaluations
only, no GP fitting; runtime depends on the number of distinct transferred weights.
Progress is printed at bandwidth boundaries and approximately every 30 seconds between
episodes. Repeating `run` resumes; completed episode checkpoints are reused.

Full results: `code/results/actuator_transfer_v6`. Analysis runs automatically and
can be regenerated with `python code/scripts/actuator_direct_transfer.py analyze`.
Processed outputs: transfers.csv, references.csv, exploratory_common.csv, summary.csv.

```bat
python code/scripts/actuator_direct_transfer.py collect
git commit -m "Add corrected direct weight transfer pilot results"
git push -u origin experiment/actuator-direct-transfer
```

Collect checks expected assessments and episode counts, regenerates summaries and
force-adds only this run's manifest, assessments, episode records and summaries.
It excludes smoke output and never changes older runs.

## Interpretation

If source-only transfer consistently meets 5%, this setup does not demonstrate a
practical need for target weight optimization at that tolerance. If some cases fail,
inspect whether this reflects poor source calibration, a donor mismatch, or a real
bandwidth-dependent loss of transferability before introducing target BO. Performance
at 1% is supplementary; do not change the primary criterion after seeing results.
