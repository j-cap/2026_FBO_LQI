# Stage 3 diagnostic and symmetric robustness-margin replay

No new plant simulations. Reuse committed Stage2 source records and Stage3 target
records. This is a retrospective developmental follow-up to observed Stage3 failures,
not independent confirmation. Original outputs, budgets, requirement grid and quality
gate remain unchanged. Run inputs are hashed, including all raw records.

## Fixed margin rule

For each target, learner(local/fleet), and budget0/4/8/16:
1. Select ONLY the original nested calibration prefix.
2. Leave out one candidate weight and ALL its maneuver observations together.
3. Fit the original two-feature ridge correction on the remaining calibration records,
   retaining the original local/unit or frozen fleet prior and lambda=1.
4. Predict each held-out calibration observation. Compute
   (actual E + measured integration difference in E) / predicted E.
5. Margin=max(1, all valid held-out ratios); no hyperparameter tuning or clipping.
6. Refit on ALL budgeted calibration records and multiply tracking predictions by
   the margin for requirement-based selection. Steering predictions stay unchanged.

No evaluation outcome enters fitting, fold construction or margin selection. Fleet
prior still excludes the target vehicle. Fits use only numerically resolved physically
feasible observations; all attempted episodes retain their budget cost. At budget0
there is no calibration evidence: margin=1 and empirically_calibrated=false. Tiny
budgets mean tiny folds (budget4 has two weight groups). Cross-fit residual envelopes
are heuristic; they are NOT a calibrated confidence bound or a conformal guarantee,
especially after refitting and adaptive controller selection. Raw fold records export
training candidate IDs, held-out IDs, predictions and ratios for audit.

Compare original local/fleet-adjusted, local/fleet with margins, and fixed25% margin.
All models/weight pools and target episodes are unchanged. Frozen fixed-margin rows
consume zero target episodes even when repeated along the budget axis. Historical
fleet costs24 source observations per prior remain distinct from target budget.

## Diagnostic only

weight_dependence.csv examines the32 held-out weights at fixed vehicle and demand:
range of actual/linear-model tracking ratio, range and median of actual/corrected
prediction ratio, calibration-derived margin, count exceeding it, and integration
error. Spread of ratios at a fixed demand measures failure of an exact demand-only
multiplicative model. It does not prove which physical mechanism causes the spread
or that a more complex learner is warranted. These evaluation diagnostics must not
be fed back into the margin rule or used to select a winning margin after inspection.

## Evaluation

Keep original gate: zero violations/other failures; >=95% coverage of sampled-feasible
requests; mean steering excess<=2%; worst<=10%. Compare curves and tail costs, not only
first passing budget. If a simple local margin eliminates the apparent budget advantage,
report that. If both methods abstain or become conservative, that is not success.
No additional seeds, larger fleet, optimization, or numerical re-evaluation is added.

## Windows commands, repository root

```bat
conda activate paper_FBO_LQI
git switch study/cooperative-calibration
git pull --ff-only origin study/cooperative-calibration
python code/scripts/coverage_stage3_margin.py smoke
python code/scripts/coverage_stage3_margin.py run
start code\results\coverage_stage3_margin_v1\report.html
python code/scripts/coverage_stage3_margin.py collect
git commit -m "Add Stage 3 calibration-only margin replay results"
git push origin study/cooperative-calibration
```

Requires full committed Stage2 and Stage3 raw case directories. Smoke replays nominal
target budgets0/4 in a separate directory. Runtime should be seconds, not minutes.
Collect recomputes replay, checks frozen input signature, and force-stages only the new
result directory. It neither alters original results nor stages smoke outputs.
