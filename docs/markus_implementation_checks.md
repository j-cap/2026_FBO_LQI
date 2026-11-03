# Markus review: implementation checks

Run these checks before the November text and figure revisions. The audit writes
new reports only. It does not change production code, historical results, the
manuscript, figures, or Overleaf comments. No BO or plant simulation is rerun.

## Run locally (Windows, repository root)

```bat
git pull --ff-only
conda activate paper_FBO_LQI
python code/scripts/audit_review_implementation.py
```

If that environment is not installed:

```bat
conda env create -f code/env_paper_FBO_LQI.yaml
conda activate paper_FBO_LQI
```

The default output is `reviews/implementation_audit/`, deliberately outside the
ignored `code/results/` tree so ordinary `git add` works. Existing output directories
are never overwritten. For a repeat run, use a new directory:

```bat
python code/scripts/audit_review_implementation.py --output reviews/implementation_audit_r2
```

The historical pair archive is expected at
`code/results/calibration_landscape_similarity/processed/landscape_similarity_pairs.csv`.
It is ignored by Git and may exist only on the experiment machine. If it is elsewhere:

```bat
python code/scripts/audit_review_implementation.py --pairs "H:\path\landscape_similarity_pairs.csv" --output reviews/implementation_audit_with_archive
```

Explicit paths resolve from the current directory; defaults resolve from the script's
location. Do not rerun `calibration_landscape_similarity.py` merely to fill in a
missing archive: that would regenerate the frozen experiment.

## What is checked

1. **Steering increment:** invokes the real `candidate_eval.evaluate` with controlled
   rollout arrays and stubbed synthesis/simulation. Checks below/equal/above threshold,
   negative increments, constant nonzero input, single sample, and a violation in a
   later episode. Repeats with doubled time spacing to distinguish radians per step
   from radians per second. Reads Ts and the threshold from the resolved confirmatory
   configuration. At 0.05 s, 0.2618 rad/step is 300.000701530 deg/s: retain the original
   numeric threshold and use approximately 300 deg/s in prose.
2. **Near-best sets:** parses and executes the actual `top_n` and `top_sets` statements
   from the landscape driver, without executing its simulation. Checks empty, sparse,
   and fully feasible grids with noncontiguous feasible coordinates. It distinguishes
   `min(13, n_feasible)` from `ceil(0.1*n_feasible)`. A changed source structure fails
   explicitly instead of silently switching to a duplicate implementation.
3. **Frozen archive, if available:** checks the first five development seeds, 30 clients
   and 435 unique unordered pairs per fleet, absence of self/duplicate pairs, metric
   validity, and aggregate correlation/overlap summaries. It records the file SHA-256.

The current code measures differences between applied steering samples. The production
plant clips steering amplitude, and the adapter trims its unused final array element.
The evaluator's `np.diff` does not include a jump from an assumed pre-rollout input.
These checks establish current code semantics, not pre-execution safety.

## Outputs and interpretation

- `summary.txt`: concise findings and coverage.
- `audit.json`: resolved settings, source hashes, Git commit/status, dependency versions,
  conversion, archive provenance, limitations, and overall status.
- `steering_cases.csv`: 14 actual-evaluator checks.
- `selection_cases.csv`: seven actual-selection checks, including selected indices.
- `selection_source.txt`: exact production statements exercised.
- `evaluator_source.txt`: candidate evaluator used by the constraint probes.
- `archive_fleets.csv`: per-fleet archive checks, when an archive exists.

`PASS` means current implementation checks and available archive checks passed.
`PARTIAL` means the implementation passed but the historical archive was unavailable.
`FAIL`/`ERROR` return a nonzero exit code and must be inspected before changing prose.
Missing archive is reported explicitly but does not fail the implementation checks.

**Evidence limit:** a pair-level CSV cannot reconstruct each system's selected grid
points or establish the historical generating code revision. The audit never claims
that its synthetic cases reproduce historical memberships. If those become necessary,
we must inspect the original per-point records separately. No pickle cache is loaded.

## Commit the reports

```bat
git add reviews/implementation_audit
git commit -m "Record steering and near-best selection implementation audit"
git push
```

Use the chosen output directory if you supplied `--output`. Commit the report even
when it says PARTIAL or FAIL, so the missing evidence or mismatch can be reviewed.
The test fixtures are synthetic and must not be treated as paper results.

## Developer validation

```bat
python -m unittest discover -s code/tests -p test_review_implementation_audit.py -v
```
