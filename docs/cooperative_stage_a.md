# Stage A: corrected-evidence opportunity audit

No plant simulation or optimization. Existing v5/v6/v7 files are read only.
The analysis runs in approximately a minute on the development machine; local disk
speed can change this. Progress prints while reading the unique evaluations.

From the repository root in the existing paper_FBO_LQI environment:

```bat
git fetch origin
git switch study/cooperative-calibration
git pull --ff-only
python code/tests/test_cooperative_stage_a.py
python code/scripts/cooperative_stage_a.py run
```

If the local branch does not exist yet, replace the switch command with:

```bat
git switch --track origin/study/cooperative-calibration
```

Open `docs/analysis/cooperative_stage_a/report.html` in a browser. The HTML embeds
its plot. Full CSV outputs retain exact threshold breakpoints; the plot crops the
screening threshold axis for readability. No requirement is selected automatically.

To commit and push a local regeneration:

```bat
python code/scripts/cooperative_stage_a.py collect
git commit -m "Regenerate Stage A cooperative calibration opportunity audit"
git push -u origin study/cooperative-calibration
```

If nothing changed, Git may report nothing to commit; that is expected.
The repository contains the script, tests and compact review outputs. Running the
script also reconstructs the larger aligned tables, episode table and input-hash
manifest. Collect verifies the output hashes before staging all generated outputs.

Interpretation safeguards:
- Exact weights, not row numbers, determine grid alignment. No matched points exist
  across the two fleet realizations, so there is no combined 30-condition grid claim.
- Grid costs combine maneuvers and mostly lack saved numerical validation checks.
  These are screening curves, not certified acceptable sets.
- Transfer scores are recomputed per maneuver from 80 saved episodes per evaluation.
  Repeated donor labels do not multiply the unique episode count.
- Each source run is checked on five OTHER vehicles and five bandwidths. The donor
  at every bandwidth is missing; this is not complete donor-inclusive validation.
- True-state scores exist for the v7 common/reference probes, not for the full grid
  or all optimized donor candidates.
- A candidate's largest observed mean RMSE is descriptive, not a confidence bound.
- No threshold or method is selected using these results for a confirmation claim.

Stage A does not demonstrate fewer calibration experiments. The engineering RMSE
requirement remains unresolved. The demanding-condition baseline suggested by the
screening data is exploratory and must be checked on fresh data if pursued.
