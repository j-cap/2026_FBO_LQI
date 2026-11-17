# Step 2 — source calibration and reference validity

This is a diagnostic study, not a transfer-BO comparison. It leaves the original
actuator pilot and its code unchanged. No new pooled or target BO runs are made.

## Fixed study design

- Reconstruct the original two fleets (355238, 987946), including the original
  vehicle admission and identification procedure. Audit nominal_00, payload_00,
  and tire_degraded_00 from each. The remaining 24 vehicles per fleet are not
  recalibrated in this step. These are six fixed vehicles, not six independent fleets.
- Source actuator tau = 0.20 s. Three independent optimization seeds (101, 202,
  303), crossed with both fleets and both calibration banks. Same candidate sets
  across vehicles for a given optimizer seed; this supports paired comparisons.
- Two source arms: original 6 episodes (legacy, seed 5000), and expanded 20
  episodes (calibration, seed 1200000). Same existing GP/EI implementation,
  normalization and infeasibility penalty. Budget snapshots 25, 50, 75 come from
  ONE trajectory, not best-of-several restarts. Recommendations use only their
  own arm's calibration scores. Cross-bank scores are diagnostic only.
- 36 source optimization trajectories, 2700 proposed evaluations. Cache hits and
  repeated proposals are retained in the nominal budget; unique query counts are
  exported so repeated proposals cannot masquerade as new information.
- Reference search: each of the six vehicles at tau = 0.10, 0.15, 0.20, 0.30,
  0.40 s, using the expanded calibration bank. Nested Sobol grids 512/1024;
  Powell refinement from six starts, maximum 300 evaluations each. Preserve the
  best intermediate evaluation even if an optimizer hits its budget.
- Validation: 40 episodes (seed 1300000); a predeclared shortlist combines the
  eight best calibration candidates and eight best grid candidates, deduplicated.
  Among calibration- and validation-feasible candidates, select lowest validation
  cost. If none pass, mark unresolved; never rescue the reference using test data.
- Fresh test: 80 episodes (seed 1400000), only in the separate `test` stage.
  Different banks vary noise realizations, not maneuver families: all retain the
  same two reference maneuvers. Assert bank seed separation for each vehicle.
- Assess source recommendations at matched tau only; assess reference controllers
  at all five taus. Every assessed controller is repeated with 10 RK4 substeps
  versus 5, including failed trajectories. Failure agreement is not a cost
  convergence certificate. Relative cost tolerance is 0.1% for feasible runs.

The test bank is for assessment, never recommendation selection. Repeated stages
resume completed files; configuration/code hashes reject stale checkpoints.
The smoke namespace is separate and retains the original fleet-generation context.
Do not splice new-bank results into the old pilot's performance curves.

## Reference limitations and reporting

`reference_usable` requires a validation-selected controller, feasible test score,
passing integration check, and <=1% improvement from the 512 to 1024 Sobol grid.
The grid criterion is a conservative resolution screen, NOT global optimality
or a proof of refinement convergence. Inspect optimizer termination and shortlisted
validation scores too. A reference can still be beaten: negative regrets are
retained and should trigger a quality review, not clipping or test-driven replacement.

Statuses distinguish `success`, `inaccurate`, `infeasible`, `unresolved_reference`,
and `unresolved_integration`. Do not interpret unresolved cases as zero success
or silently drop them. Raw costs, feasibility, reasons, reference coverage and
status counts are exported. Reports are descriptive; assess each fleet separately.

## Run locally (from repository root)

```bash
conda activate paper_FBO_LQI
python code/scripts/actuator_diagnostics.py preflight
python code/scripts/actuator_diagnostics.py smoke --n-jobs 1
python code/scripts/actuator_diagnostics.py sources --n-jobs 2
python code/scripts/actuator_diagnostics.py references --n-jobs 2
python code/scripts/actuator_diagnostics.py test --n-jobs 2
```

Each command must finish successfully before the next. A smoke result can be
scientifically unresolved; it checks execution, not source/reference quality.
`test` automatically writes the analysis. To regenerate it:

```bash
python code/scripts/actuator_diagnostics.py analyze
```

Main output: `code/results/actuator_diagnostics_v2/`. Read
`processed/source_convergence.csv` (budget, unique proposals, infeasibility,
expanded-bank cost gap), `source_test.csv` (held-out source quality),
`reference_search.csv` (validation/search diagnostics), and `references.csv`
(held-out reference usability). JSON retains every source history and shortlist.

For returned results, commit the protocol, processed tables, and source/reference/
test JSON, excluding disposable `cache` directories:

```bash
git add code/results/actuator_diagnostics_v2/protocol.json code/results/actuator_diagnostics_v2/processed
git add code/results/actuator_diagnostics_v2/seeds/*/sources code/results/actuator_diagnostics_v2/seeds/*/references code/results/actuator_diagnostics_v2/seeds/*/tests
git commit -m "Add Step 2 actuator calibration diagnostics"
git push
```

## Decision after analysis

First determine whether source quality is dominated by optimizer seed, search
budget, or episode-bank choice. Then review validation/test failures and reference
resolution. Only after these checks define the next transfer experiment. Poor
sources, missing references, or failed integration checks do not establish a
bandwidth-induced need for BO. This step does not yet implement the source-only
common-weight baseline or the source-target landscape study from Step 3.
