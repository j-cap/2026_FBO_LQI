# Stage 2 — uneven overlapping coverage and target rotations

Developmental test of whether complementary source information remains useful when
coverage overlaps and each of three vehicles is held out in turn. Same nominal,
payload(m x1.20,Iz x1.25),front-tire-change(Cf x0.75) vehicles and noise-free plant/model
evaluator as the preceding pilot. No large fleets or random repetitions.

## Fixed patterns and budgets

Calibration demands:1.5,4,7 m/s². Entries below are source experiment counts at these
three demands. Each source contributes12 observations; pooled budget24.

| Pattern | Source1 counts | Source2 counts |
|---|---|---|
| Complementary overlap |8,4,0|0,4,8|
| Redundant narrow |8,4,0|8,4,0|
| One full source |4,4,4|8,4,0|

For each target, remaining sources are assigned in cyclic order nominal→payload→tire.
No assignment selected from outcomes. Each single-source restricted comparator gets
double its contribution (24 experiments); full-coverage single-source controls get
8 per demand (24). Pooled direct weight-library transfer uses the same24 observations
as pooled correction. Model-only and fixed25% margin need none. Each vehicle also
requires20 seconds of local identification; source identification costs are separate
from calibration counts. Source availability is an explicit scenario assumption,
not an empirically established property of real driving.

Fresh16 calibration weights(seed4901),32 distinct evaluation weights(seed5901).
Evaluate all48 on five demands1.5,2.75,4,5.5,7 for all3 vehicles:720 paired cases.
The2.75 and5.5 demands are absent from ALL calibration records. All48 weights are
available to target-model selection; prediction error is evaluated on the32 excluded
from fitting. A 48-point hindsight reference is only a finite-grid diagnostic.
Larger raw tables are offline scoring data; no method gets free calibration access.
Method allocations are exported in calibration_ledger.json. All comparisons reuse
these simulations, no repeated physical/noise trials.

## Simple fixed predictor and controls

At each observed demand, each source computes counts and sums of log(actual/predicted)
tracking and steering metrics using resolved, physically feasible observations.
Counts and sums are merged; geometric-mean correction factors are interpolated in
squared demand between all available knots (INCLUDING the middle knot). Outside a
source's observed interval, hold the nearest correction constant. At least3 valid
observations are needed for a knot; if none are available abstain. No new extrapolation
model, hyperparameter tuning or target calibration is introduced.

Centralized fitting uses the exact same selected observations. Distributed versus
centralized correction factors are checked to1e-10 at every evaluated context.
Equality is expected algebraically, not evidence for a new FL optimization algorithm.
Summary exchange does not imply formal privacy. All clients run in one simulation
process. Direct-library comparison interpolates per-weight log(E/d),log(S/d), averaged
across source records when shared; it uses nearest extension outside its measured
range and recomputes controller gains on the target. It is not an optimized BO baseline.

Score the unchanged1%..30% tracking limit grid:150 correlated requests per target,
pattern and method. Eight methods, three patterns, three rotations:10800 decisions,
not10800 independent experiments. Separate all-demand and unseen-demand tables.
Check violations and abstentions alongside mean/median/max steering excess conditional
on passing. Numerical unknowns are not bad objective penalties. Existing physical
limits, antiwindup and convergence checks are unchanged. No safety probability claim.

## Decision rule for this developmental stage

Complementary pooling should help more than redundant pooling, across target roles,
without a large loss versus full-coverage controls. A full-coverage source removing
the benefit of pooling is an informative boundary, not a failure. Improvement only
for one target or only via additional abstention is weak evidence. Do not use aggregate
counts to hide target-specific failures or tails. Equal calibration budget does not
mean equal total identification cost, and the32 evaluation weights remain sparse.

## Run locally, repository root

```bat
conda activate paper_FBO_LQI
git switch study/cooperative-calibration
git pull --ff-only origin study/cooperative-calibration
python code/scripts/coverage_stage2.py smoke
python code/scripts/coverage_stage2.py run
start code\results\coverage_stage2_v1\report.html
python code/scripts/coverage_stage2.py collect
git commit -m "Add Stage 2 overlapping coverage rotation results"
git push origin study/cooperative-calibration
```

Smoke preserves calibration allocations, reduces evaluation weights to2:270 cases
in a separate output directory. Full run usually takes a few minutes; progress after
each weight, atomic resume, normalized input signature. Changed inputs need a new tag.
Collect validates case completeness and signature, regenerates reports and force-stages
only this output directory despite ignore rules; smoke results are excluded.
