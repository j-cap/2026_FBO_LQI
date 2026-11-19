# Requirement sensitivity follow-up

This read-only follow-up compares common/reference controllers per maneuver and
recomputes donor/common comparisons on identical target coverage. It also quantifies
noise uncertainty in the observed worst common-weight condition. It does not choose
an engineering acceptance threshold or launch new experiments.

Already executed and saved in `docs/analysis/cooperative_requirements/report.md`.
To reproduce from the repository root:

```bat
git fetch origin
git switch study/cooperative-calibration
git pull --ff-only
python code/scripts/cooperative_requirements.py
```

The required Stage A transfer table and v7 episode table are tracked on this branch.
The script checks completeness, validity and paired noise-seed alignment. Bootstrap
seed and replication count are fixed. Input hashes normalize CRLF/LF line endings.

After regenerating:

```bat
git add docs/analysis/cooperative_requirements
git commit -m "Regenerate controller requirement sensitivity analysis"
git push -u origin study/cooperative-calibration
```

If Git reports nothing to commit, the checked-in results were reproduced unchanged.
