# Actuator-bandwidth experiment sequence

This is a new exploratory study following the direct-transfer audit. It asks whether varying a known steering-actuator time constant changes the useful LQI weights despite target-specific gain synthesis. No outcome is required to favor BO. Historical ideal-actuator calibration observations and numerical references are not reused.

## Model and assumptions

The simulated plant has states [beta, yaw rate, actual road-wheel steering]. The LQI command u is desired road-wheel steering. The added dynamics are d(delta)/dt=(u-delta)/tau, with unity steady-state gain and no transport delay. Nonlinear tire forces use actual delta. Coupled RK4 integration uses five substeps per control interval. The actuator parameter supplied to controller synthesis equals the simulated parameter: actuator identification uncertainty is deliberately excluded initially.

Bare-vehicle models are the original locally identified (Ad_hat,Bd_hat), obtained by the existing ideal-steering identification procedure. We compute a principal real continuous ZOH embedding using log([[Ad,Bd],[0,1]])/Ts, verify reconstruction, attach the actuator, and exactly discretize the entire augmented linear model. This preserves the existing vehicle identification experiment. It is modular model composition, NOT a newly identified actuator-in-loop model or a claim of recovering physical tire parameters. A nonexistent/nonreal embedding stops the run. Actuator-in-loop system identification is a separate future study.

The control state is [beta,r,delta,eta], where eta=Ts*z_legacy and eta accumulates measured r-r_ref. Thus its Q entry is 10^xi_3/(Ts*s_z)^2; the other two existing weights/scales are unchanged. The additional delta penalty is zero and R=1 penalizes commanded steering. The existing static prefilter, leaky integral factor 0.998, and antiwindup coefficient 0.2 are retained; the leak is now included consistently in the Riccati model. This differs from the legacy synthesis, which ignored its rollout leak. Gains use true state feedback as in the existing simulator; yaw measurement noise affects integral feedback and scored RMSE. Actual steering is assumed available exactly. We do not claim a noisy full-state estimation experiment.

Command angle saturation is retained. Steering bandwidth and physical rate constraints are distinct: there is NO added hard rate limiter. Peak actual steering rate |u-delta|/tau is checked against the existing rate limit converted from rad/sample to rad/s. A violation is infeasible. Sideslip is checked at integration substeps. The nominal stability test requires rho<1-1e-8, not the old empirical 0.99 margin, which would preferentially reject slow actuators. These design conventions are common to all new sources and targets; do not splice these results into legacy curves.

All methods use calibration RMSE divided by the fixed reference amplitude r_max as surrogate cost, with dimensionless infeasibility penalty 10. This avoids purchasing a target baseline evaluation outside the budget or discarding targets for an infeasible baseline. It deliberately differs from legacy per-client baseline-cost normalization. Physical tracking RMSE and the 5% reference-relative criterion retain their meaning. The original fleet admission selection is preserved; there is no new actuator-feasibility rejection/resampling.

## Frozen initial range and stages

Source tau=0.20 s. Target taus=[0.10,0.15,0.20,0.30,0.40] s. This is an illustrative factor-of-two sensitivity range, not a claimed measured production-fleet distribution. Keep the range unchanged after seeing outcomes. Vehicle realization, tasks, noise banks, constraints, scales and source histories stay fixed across tau. The matched tau is mandatory.

1. **Preflight:** rebuild the first pilot fleet and check all 30 model embeddings.
2. **Smoke:** reduced six-vehicle fleet, three source evaluations, eight landscape points, one matched tau. Checks plumbing only; never cite as evidence.
3. **Sources:** independently calibrate all 30 source vehicles at nominal tau, 25 evaluations each (two initial Sobol points, then EI). Save all observations. Sources with no feasible recommendation remain explicit failures, not silently removed.
4. **Diagnostic:** leave-one-target-out transfer using all 29 source choices and the source selected by existing bare-vehicle dynamics distance. Recompute target gains at each tau. Evaluate a shared nested Sobol grid (128/256/512 points) and four bounded Powell searches (200 evaluations/start) on calibration episodes to construct a numerical reference. Test selected weights separately. Compare each target's nominal-tau landscape with its shifted-tau landscape on the same grid, using joint-feasible Spearman correlation and Jaccard overlap of grid near-optimal sets. These are within-vehicle bandwidth diagnostics, not cross-source causal attribution.
5. **BO (only after diagnostic review):** cold local BO, transferred-weight-initialized local BO, and transferred-weight-initialized fleet-informed BO. The latter two use the same first transferred query and same second Sobol query; only historical source observations differ. Each target evaluation counts, including evaluation of transferred weights. Source histories and test data never change during target BO. Record the calibration-selected recommendation at every budget; never choose by test scores.

Initial pilot: first two existing seeds, one target per family, five taus = 30 target conditions. Full scope: all 20 existing seeds, two targets per family = 600 conditions. Full is explicit, not launched by the pilot commands.

## Reference and numerical checks

The numerical reference is selected using calibration data, then scored on 20 separate test episodes per maneuver (seed base 900000). BO uses the six calibration episodes only. Reference search is outside the commissioning budget and is never passed to acquisition or normalization.

`reference_check_pass` compares the best cost from the penultimate grid size with the refined reference (relative change <=1%). This is a conservative adequacy flag, NOT a proof of convergence or global optimality. If it fails, inspect nested-grid and refinement outcomes and strengthen reference search before interpreting the 5% result. Do not drop failing conditions. Negative test regrets are retained.

`integration_check_pass` compares reference-controller test costs at the configured RK4 substeps and twice that number (relative difference <=0.1%, both feasible). This checks selected reference controllers, not every possible candidate. A stronger integrator check is warranted if failures concentrate near a boundary.

The BO stage refuses conditions with failed reference or integration checks. If a numerical reference is infeasible, output marks reference_valid=false and success=false; such rows are unresolved, not evidence of negative transfer. A physically unachievable task and inadequate weight search must be distinguished before claiming BO is needed.

## Commands from repository root

Use the existing paper_FBO_LQI environment. No new packages are required.

```bash
python code/scripts/actuator_bandwidth_experiment.py preflight
python code/scripts/actuator_bandwidth_experiment.py smoke --n-jobs 1
python code/scripts/actuator_bandwidth_experiment.py sources --scope pilot --n-jobs 2
python code/scripts/actuator_bandwidth_experiment.py diagnostic --scope pilot --n-jobs 2
```

Stop here and inspect `processed/diagnostic_summary.csv`, `reference_checks.csv`, `grid_convergence.csv`, and `landscape_metrics.csv`. First verify source and matched-condition feasibility; then assess transfer at shifted taus. The diagnostic stage is substantially more expensive than the previous saved-result audit: it builds new source histories and numerical references.

After review, select BO conditions explicitly, always including matched tau. Example syntax (not a decision to run these conditions):

```bash
python code/scripts/actuator_bandwidth_experiment.py bo --scope pilot --taus 0.2 0.4 --n-jobs 2
```

Expand to the full campaign only after pilot review:

```bash
python code/scripts/actuator_bandwidth_experiment.py sources --scope full --n-jobs 4
python code/scripts/actuator_bandwidth_experiment.py diagnostic --scope full --n-jobs 4
```

Runs resume by source client, target/tau condition, and cached candidate score. Reissue the same command. Fingerprinted protocols reject changed configurations/code; use `--run-tag actuator_bandwidth_v2` for an intentional revision. Do not run concurrent commands into the same output namespace. Pilot and full scopes are separate; they do not silently copy checkpoints.

Regenerate pilot summaries without simulations:

```bash
python code/scripts/actuator_bandwidth_experiment.py analyze --scope pilot
```

For BO summaries, also pass `--analyze-stage bo --taus ...` with the same conditions. All requested fleet checkpoints must exist.

## Review outputs and result upload

Default pilot directory: `code/results/actuator_bandwidth_v1_pilot/`.

* `protocol.json`: resolved configuration and normalized code fingerprints.
* `seeds/<seed>/sources.json`: new source histories and continuous vehicle-model embeddings.
* `seeds/<seed>/diagnostic.json`: transfer scores, full landscapes, references and convergence checks.
* `seeds/<seed>/bo_*.json`: budget-by-budget BO recommendation scores (only if run).
* `processed/`: compact CSV tables. Uniform-source averages are computed within each target before aggregation. Source pairs are not independent replicates. These exploratory summaries do not provide inferential confidence intervals.
* `figures/`: success-versus-tau plots; check reference validity before interpreting them.

To upload the pilot, add aggregate checkpoints explicitly (never the cache directories):

```bash
git add -f code/results/actuator_bandwidth_v1_pilot/protocol.json code/results/actuator_bandwidth_v1_pilot/processed/ code/results/actuator_bandwidth_v1_pilot/figures/
git add -f code/results/actuator_bandwidth_v1_pilot/seeds/*/sources.json code/results/actuator_bandwidth_v1_pilot/seeds/*/diagnostic.json
git commit -m "Add actuator-bandwidth pilot diagnostic results"
git push
```
