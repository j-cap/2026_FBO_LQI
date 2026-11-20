# Stage 3 — additional target calibration budgets

Question: does frozen historical fleet knowledge save target calibration relative to
collecting a few local records? Small developmental diagnostic, no new source campaign.
Uses committed coverage_stage2_v1 raw results as frozen source inputs. Each target's
prior uses the OTHER two vehicles, complementary [8,4,0]+[0,4,8] allocation:24 source
episodes. Counts/sums and demand interpolation are unchanged from Stage2. Source hashes
are frozen in the new manifest. All three vehicles rotate as targets.

## Target protocol

Fresh6 calibration weights(Sobol seed6901, first6 of8),32 separate evaluation weights
(seed7901). Evaluate all38 weights at1.5,2.75,4,5.5,7 m/s² on each vehicle:570 paired
model/plant cases. No added noise or replication. Local identification remains the
same20-second gentle experiment. Calibration budgets0,4,8,16 are nested prefixes of
j=0..15: weight=floor(j/3), demand=[1.5,4,7][j mod3]. This yields fixed balanced context
coverage without using outcomes to choose episodes. Every attempted episode counts,
including unusable observations; the evaluator does not provide free replacements.
Two interior demands are absent from calibration. All38 weights are eligible for
model-based selection; prediction error uses only the32 excluded from calibration.
The full plant table is for offline scoring, not available to policies.

## Equal learner classes

Both local-only and fleet-assisted methods fit the SAME two-feature ridge adjustment
with phi=[1,(d/7)^2], fixed lambda=1, no tuning:

  log(actual/predicted) = log(prior_factor(d)) + phi(d)^T beta.

Local prior_factor=1; fleet prior_factor=frozen Stage2 correction. Fit E and S separately
using only budgeted resolved feasible target records. Minimize sum of squared log
residuals +lambda*||beta||². Zero-budget beta=0. Same regularization and feature class
for both; advantage is prior knowledge, not more parameters. This is a specific local
learning baseline, not the best possible calibration method. Demand-only adjustment
cannot address arbitrary weight-dependent discrepancy.

Compare local correction, fleet-adjusted correction, frozen fleet, model-only, fixed25%
margin, and direct local weight-library interpolation using exactly the same target
records. The latter selects only weights actually tested locally, interpolating log
E/d,S/d with nearest extension; no BO or adaptive experiment selection is claimed.
Frozen baselines are repeated along budget axes but consume ZERO target calibration.
Their rows must not be counted as independent repeats. Historical fleet acquisition
cost24 source episodes per target prior and source identification remain separate from
additional target costs. All methods need local target identification.

## Scoring and predeclared developmental quality criterion

Unchanged1%..30% tracking-limit grid at five demands. Report per target and budget:
violations, abstentions, fraction of sampled-feasible requests passed, mean/worst
steering excess among passes. Numerically unresolved/boundary cases remain separate.
Reference is the best feasible candidate in the finite38-point target table, not a
global optimum. Correlated requirements are not statistical trials.

A budget passes the DEVELOPMENTAL gate if: zero tracking violations and other failures;
>=95% coverage of sampled-feasible requests; mean steering excess<=2%; worst<=10%.
These fixed useful-quality thresholds were chosen before the new run, are not externally
validated requirements, and must not replace inspection of the full curves. Missing
first-passing-budget entries mean no tested budget passes. Fixed baselines' first
passing budget is0 if they pass. No guarantee or statistical sample-complexity claim.

If local-only catches up with4 episodes, the fleet benefit is small. If the fleet reaches
useful quality substantially earlier on several targets, proceed to a stronger local
identification comparison. If neither passes, inspect correction limitations and tails;
do not silently relax the gate. Broader-excitation identification, optimized local BO,
noise, and unmodeled-law heterogeneity remain separate outstanding experiments.

## Run locally (repository root)

```bat
conda activate paper_FBO_LQI
git switch study/cooperative-calibration
git pull --ff-only origin study/cooperative-calibration
python code/scripts/coverage_stage3.py smoke
python code/scripts/coverage_stage3.py run
start code\results\coverage_stage3_v1\report.html
python code/scripts/coverage_stage3.py collect
git commit -m "Add Stage 3 target calibration budget results"
git push origin study/cooperative-calibration
```

Requires the committed720 Stage2 case files. Smoke retains all budgets and6 calibration
weights, uses2 evaluation weights:120 paired cases, separate output directory. Full run
usually a few minutes; progress and atomic resume. Inputs and source data are hashed;
changed inputs require a new tag. Collect checks completion/signature, regenerates
reports and stages only this run despite results ignore rules. No smoke data staged.
