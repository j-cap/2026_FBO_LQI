# Fresh-noise validation at 0.92 deg/s

This is a fixed-sample validation of frozen candidates, not an optimization campaign.
Run locally from the repository root in the existing `paper_FBO_LQI` environment.

## Fixed scope and statistical rule

- Six existing vehicles, five actuator bandwidths, two maneuvers, two frozen controllers.
- Common weights and target-specific references use the exact gains saved by v7.
- 1,000 new noise episodes per maneuver/vehicle/bandwidth/controller: 120,000 episode
  pairs. Each pair uses both 5 and 10 RK4 substeps, giving 240,000 numerical rollouts.
- No new physical vehicles or maneuvers: this validates noise expectations on the
  existing simulated cases, not generalization to unseen fleets.
- Same noise seeds pair controllers and bandwidths within each vehicle/maneuver.
  Other vehicle realizations and maneuvers have distinct banks.
- Seeds are fixed in the upper uint32 half, disjoint from the legacy episode builder's
  lower-half draws. Smoke has its own bank and never contributes to validation.
- Mean episode MEASURED yaw-rate RMSE is checked separately for all 120 cells.
  Threshold is 0.92 deg/s = approximately 0.016057 rad/s. True RMSE is diagnostic only.
- Two-sided Student-t intervals use Bonferroni alpha=0.05 across all 120 cells:
  mean +/- t_(1-0.05/(2*120),999) * SD/sqrt(1000).
  Endpoints are additionally widened by the largest observed coarse/fine RMSE difference
  in that cell; this is a numerical sensitivity allowance, not an integration-error proof.
- This is approximate large-sample inference. The RMSE distribution is not assumed
  exactly Gaussian; these intervals are not distribution-free finite-sample guarantees.
  Bonferroni handles dependence between cells, provided the marginal intervals are valid.
- Planning used development SD about 0.062 deg/s, yielding half-width about 0.007 deg/s.
  This does not guarantee classification at the chosen threshold.

Per cell: upper <= threshold => supported; lower > threshold => exceeds; otherwise
unresolved. Physical failures or unresolved integration prevent acceptance. Failure
at both resolutions is recorded as a physical failure; resolution disagreement is
numerically unresolved. Truncated/nonfinite errors are never accepted or averaged away.

A controller is supported only when every one of its 60 cells is supported, with no
physical failures or numerical issues. This means support for the simulated mean
criterion under the stated inference assumptions. Zero observed constraint violations
is not a proof of safety or zero failure probability. The interval family covers both
controllers; no controller is selected for a favorable correction after results are seen.

There is one fixed final analysis. Do not extend the sample count after inspecting an
unresolved result. Any new follow-up requires a separately documented protocol. The
references were optimized for combined error, so their failure cannot establish that
all possible controllers are infeasible under the per-maneuver requirement.

## Run

```bat
git fetch origin
git switch study/cooperative-calibration
git pull --ff-only
python code/tests/test_cooperative_validation.py
python code/scripts/cooperative_validation.py preflight
python code/scripts/cooperative_validation.py smoke --n-jobs 2
python code/scripts/cooperative_validation.py run --n-jobs 6
```

Stop if a command fails. Six is the maximum useful number of parallel workers, one
per vehicle; use 2 or 4 if CPU or memory is shared. The runner prints progress at most
30 seconds apart while computing and checkpoints every 25 episode pairs. Repeating
an interrupted run command resumes completed blocks. At most one incomplete block
per worker must be repeated. Do not run two copies against the same output directory.
The manifest rejects changed code, inputs, gains, threshold or seeds on resume.

Smoke checks the actual simulator, fine/coarse comparison, checkpoint format and
analysis using 16 pairs. Its decisions are always labelled smoke_only. Smoke progress
is not a reliable full-runtime estimate because vehicle reconstruction adds overhead.
No full 120,000-pair campaign was run during implementation.

## Outputs and collection

Results: `code/results/cooperative_validation_v1`.
`processed/conditions.csv` contains mean, SE, interval endpoints and status per cell;
RMSE columns are in rad/s. `processed/episodes.csv` retains per-episode paired-resolution
scores. `processed/summary.json` gives the overall decision for each frozen controller.

```bat
python code/scripts/cooperative_validation.py analyze
python code/scripts/cooperative_validation.py collect
git commit -m "Add fresh-noise validation at 0.92 deg per second"
git push -u origin study/cooperative-calibration
```

Collect requires a complete run and stages only its manifest, processed summaries and
checkpoint blocks, using force-add for the ignored results directory. It excludes smoke.
After interruption, finish `run` before `analyze` or `collect`. No manual wildcard adds
are needed. Return the completed results for analysis before starting Stage B.
