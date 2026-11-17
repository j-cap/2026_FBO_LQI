# Direct-transfer baseline audit

Purpose: test whether target weight optimization adds value beyond transferring source LQI weights and recomputing gains with the target identified model. This is an exploratory response to AK's feedback, not a new confirmatory study. No manuscript edits or heterogeneity changes.

## Existing evidence and default campaign

The default uses `fleet_maturity_sweep_c1cds02`, M=25: 20 existing fleet realizations, six targets/fleet, 30 vehicles/fleet. Its nested histories started with two source initial evaluations. The earlier `phase2_warmstart_confirmatory_c1cds02` histories started with eight. These are different histories: never mix their direct-transfer and BO results.

The default campaign and numerical references are tracked in Git. The earlier campaign is also supported when its original `mature_fleet_records.parquet` and `target_convergence_traces.parquet` are available locally. Pass `--config config/runs/phase2_warmstart_confirmatory_c1cds02.yaml` to every command for that campaign. Missing files cause an explicit error; the script never silently recreates source histories.

Both campaigns use leave-one-target-out evaluation: exclude the target's entire mature history; the other 29 clients remain sources, including vehicles evaluated as targets in separate leave-one-out problems. A pooled recommendation is therefore common across the source pool for each leave-one-out problem, not one universal vector across all six problems.

## Selection rules

* Uniform source transfer: select the best feasible observed calibration weight per source, then report the expected outcome under uniform source selection. Evaluate all 29 choices; never select the best using target scores. Stable history order resolves calibration ties.
* Dynamics-selected transfer: pick the source with smallest existing uncertainty-aware dynamics distance, with client ID breaking ties. Transfer its best feasible calibration weight.
* Source-only pooled recommendation: fit the existing pooled GP to all 29 source histories (including existing infeasibility penalties), then minimize posterior mean over the same deterministic 256-point Sobol grid as target BO. No target observations or expected improvement. This is a historical-data recommendation, not an exact fleet optimum or additional source optimization.
* BO comparators: rescore the saved calibration-selected recommendation at every budget. Do not take a running minimum over test scores. Default: local and global BO at M=25. The original campaign additionally includes similarity BO.

All choices are fixed before test evaluation. Source baseline normalization remains unchanged. Test performance and reference costs never influence selection. If a source has no feasible record, stop rather than remove it from the random-source denominator.

## Evaluation and accounting

Reconstruct the existing fleet and identified models using the original seeds/configuration. Do not rerun source BO or numerical-reference optimization. Synthesize each controller using target identified dynamics, evaluate on the original nonlinear plant and the same separate test bank (20 episodes per maneuver, base seed 900000), and score against saved reference costs.

A direct policy uses zero target optimization evaluations plus one target validation evaluation. BO uses its recorded calibration budget plus one final validation evaluation. The test bank used here is an offline performance-assessment bank, not one physical rollout; episode counts are unchanged and recorded in the reference manifest. Source commissioning and identification costs are shared prerequisites, excluded consistently from the incremental comparison.

Infeasible controllers count as failures and have infinite regret. Summary quantiles include failures. Cost differences involving two failures can be undefined; use success/feasibility differences for those cases. Negative regret is retained because the numerical reference is not a proven optimum. Random-source success is an expectation of per-source success indicators, not a threshold applied to an averaged controller cost.

The target is the paired comparison unit; fleets are the resampling unit. The 29 source choices are not independent fleets. Report success within 5%, feasibility, median and 90th-percentile regret, and paired differences against BO. Bootstrap intervals in `summary.csv` resample fleet-level success means. Smoke intervals from one fleet have no inferential meaning.

## Run from repository root (existing paper_FBO_LQI environment)

```bash
python code/scripts/direct_transfer_baselines.py preflight
python code/scripts/direct_transfer_baselines.py smoke --n-jobs 1
python code/scripts/direct_transfer_baselines.py run --n-jobs 4
```

`preflight` checks existing histories, reference configuration, source feasibility, target coverage, methods, and trace budgets without simulations. `smoke` evaluates one target of one existing fleet using the full reference-test bank. `run` evaluates all 120 targets, with process parallelism across fleets. Choose workers to match available RAM and cores; each worker fits one pooled GP at a time.

Reissue the same run command after interruption. Completed fleets are resumed atomically; partial fleets reuse cached controller evaluations. Input/config/code fingerprints reject stale checkpoints. Use a new `--output-tag` after an intentional protocol/code change. Do not run two jobs concurrently into the same output tag.

```bash
python code/scripts/direct_transfer_baselines.py analyze
```

This regenerates summaries/figures from completed full-run checkpoints. It requires every fleet, preventing accidental interpretation of a partial campaign as complete.

## Outputs

Default full results: `code/results/fleet_maturity_sweep_c1cds02_direct_transfer/` (smoke has `_smoke` suffix).

* `protocol.json`: resolved settings and source-file/code fingerprints.
* `seeds/*.json`: resumable fleet records.
* `processed/controller_scores.csv`: source-target matrix entries, selected weights, test feasibility, failures, and actual BO recommendations.
* `processed/summary.csv`: success, fleet bootstrap interval, feasibility, median/p90 regret by method and budget.
* `processed/target_metrics.csv`, `fleet_metrics.csv`: aggregation audit.
* `processed/paired_differences.csv`, `paired_fleet_success.csv`: paired BO-minus-transfer differences; positive success difference favors BO.
* `processed/transfer_matrix_<seed>.csv`: full-precision source-target regret matrices.
* `figures/success_vs_budget.pdf` and PNG; per-fleet matrix PNGs. Matrix colors clip regret to [0, .25] for readability; CSV values remain exact. Grey is excluded self-transfer, crimson crosses indicate failures.

Commit only results needed for review (never caches):

```bash
git add -f code/results/fleet_maturity_sweep_c1cds02_direct_transfer/protocol.json code/results/fleet_maturity_sweep_c1cds02_direct_transfer/processed/ code/results/fleet_maturity_sweep_c1cds02_direct_transfer/figures/ code/results/fleet_maturity_sweep_c1cds02_direct_transfer/seeds/
git commit -m "Add direct-transfer baseline audit results"
git push
```

Interpretation: first determine whether direct transfer already meets the requirement, then whether model-based selection or a pooled fixed recommendation helps. Only then decide whether further target adaptation or a broader heterogeneity study is warranted. Fleet admission used a feasible hand-tuned baseline in the original campaign; this audit preserves that selection and cannot establish performance on arbitrary vehicles.
