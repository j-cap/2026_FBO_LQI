# Corrected calibration and references (v5)

This run rebuilds source calibrations and finite-candidate reference controllers
with tracking back-calculation, T_aw=.2 seconds. It uses the same six physical
vehicles, identification and LQI synthesis as v2/v3. The old fleet construction
code stays frozen so changing controller implementation does not change admission.
The new evaluator is `src.actuator_antiwindup.evaluate`; no legacy score is reused.

## Fixed scope

- Six vehicles: two fleet seeds, three families, slot 00.
- Source actuator tau=.2 seconds; source methods Sobol and BO with infeasibility
  penalty 1, EI margin .01, no repeated candidate evaluations.
- Three optimizer seeds; snapshots at 25, 50 and 75 evaluations: 36 new histories.
- Reference tau=.1,.15,.2,.3,.4 seconds: 30 references.
- Original calibration bank: 10 episodes per maneuver (20 total).
- New validation: 40 per maneuver, seed1700000; new evaluation: 40 per maneuver,
  seed1800000. All bank seed sets are checked for overlap within a vehicle.
- Reference candidates: 256 Sobol points, v2/v3 shortlisted weights, corrected
  source incumbents at all budgets/methods/seeds and five integral-weight anchors.
  ALL candidates are rescored under the corrected controller. A fixed shortlist
  combines low calibration costs and weight-space coverage; select on validation.
- Validate/evaluate at 5 and 10 RK4 substeps. Reference usability also requires
  every final episode to pass feasibility and the .1% integration check.

The new banks have not been used by previous runs, but this is still a development
pilot on six familiar vehicles, not an independent fleet-level confirmation.
Reference controllers are validated candidates, not global optima. Retain negative
gaps; they indicate that the finite reference should be investigated.
Do not interpret source optimizer repeats as independent fleet realizations.

## Run (Windows CMD, repository root)

```bat
git fetch origin
git switch --track origin/experiment/actuator-corrected
python code/scripts/actuator_corrected.py preflight
python code/scripts/actuator_corrected.py smoke --n-jobs 2
python code/scripts/actuator_corrected.py audit --n-jobs 2
python code/scripts/actuator_corrected.py references --n-jobs 2
python code/scripts/actuator_corrected.py test --n-jobs 2
```

Use `git switch experiment/actuator-corrected` and `git pull --ff-only` if the
branch already exists. Run stages in order and stop if any command fails.
`test` automatically writes summaries to `code/results/actuator_corrected_v5/processed`.
The smoke uses a separate namespace, one vehicle and shortened budgets/banks.

Optimizer histories checkpoint after each evaluation. Reference scoring is cached
per weight/bank/tau/integration resolution. Completed reference/test artifacts are
skipped on resume. Repeat the interrupted command; cached work is reused. Individual
in-progress evaluations or episode reports may repeat. Progress prints approximately
every 30 seconds between evaluations and at stage boundaries. Reference validation
will be the expensive part. No full optimization is run by `preflight`.

```bat
python code/scripts/actuator_corrected.py collect
git diff --cached --stat
git commit -m "Add corrected actuator calibration and reference results"
git push -u origin experiment/actuator-corrected
```

`collect` regenerates summaries (requiring all expected assessments), then force-adds
only the manifest, histories, references, assessments, episode reports and CSVs.
Caches, partial histories and smoke files are excluded. Original results are preserved.

## Decision after results

Check reference feasibility and episode-level convergence, then compare source
optimizer reliability at each budget. Choose a source-calibration procedure from
this pilot, rebuild the full source fleet under the corrected evaluator, and assess
direct weight transfer on targets before testing fleet-informed BO. This run alone
cannot establish the value of transfer or the need for target optimization.
