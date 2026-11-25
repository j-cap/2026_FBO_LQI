# Offline gain-domain and sampling-resolution audit

Completed results and the [read-only analysis](analysis/gain_domain/report.md)
show that original-box refinement reduces mean held-out J by 5.85%, with a
further 2.32% from genuine extension. All selected exterior winners exceed
only the old K_y upper bound. The result remains an offline finite-pool
diagnosis; the previous online comparisons remain frozen.

This audit checks the adequacy of the search domain underlying the completed
[six-speed studies](fedbo_consolidation.md). It separates more sampling inside
the original physical gain box from evaluations genuinely outside that box.
It does not run another online BO method or change the previous results.

## Fixed nominal gains and physical bounds

| Gain | Nominal | Original range | Expanded range |
|---|---:|---:|---:|
| K_y | 0.005 | 0.0005–0.05 | 0.0005–0.15 |
| K_psi | 0.1 | 0.01–1 | 0.01–3 |
| K_r | 0.1 | 0.01–1 | 0.01–3 |

The nominal values stay fixed. The original multiplier box is `[0.1,10]^3`;
the expanded box is `[0.1,30]^3`. Upper limits are extended for all three
gains; lower limits remain unchanged. Expanded candidates therefore include
actual heading and yaw-rate gains above 1. This is a physical-domain change,
not a relabeling or a change of nominal controller.

The plant, 25-second trajectories, six speeds, three frequencies, noise,
objective J, actuator clipping, feasibility and integration checks are
inherited unchanged from `tracking_six_speed_small_bonus_v1`. The audit uses
the one common compact plant of that homogeneous study.

## Three nested candidate sets

| Stage | Candidates | Purpose |
|---|---:|---|
| Original | 71 | Preserve the previous finite-pool reference |
| Dense original box | 1,033 | Increase coverage within the original bounds |
| Expanded box | 2,011 | Add 978 genuine outside points to the dense-original set |

The dense set contains 1,024 nested Sobol draws, the seven original common
scaling anchors, and two explicit points on the old upper faces. The original
71 records, IDs and physical gains are retained exactly. New points are not
screened for feasibility before simulation.

For the expanded set, 2,048 independent Sobol draws are generated in the
larger log-domain. The 969 that lie outside the old box are retained; the
interior draws are discarded. This gives roughly comparable random-point
density per unit log-volume in the added shell and original box. It also
prevents extra interior sampling from being credited as a range-extension
benefit. Six outside profile probes and common scaling factors 12/20/30
complete the 978 added points.

The targeted profiles come from the earlier boundary observations:

- Vary only K_y of source candidate `g012` at 0.1 Hz.
- Vary only K_psi of source candidate `g030` at 0.3 Hz.
- Keep the other two physical gains exactly fixed.
- Test coordinate multipliers 10, 12, 20 and 30, in addition to the source point.

The multiplier-10 probes belong to the dense-original stage. Only multipliers
12/20/30 extend beyond the old upper face. These are retrospective diagnostic
profiles based on the earlier study, not a new data-independent online design.
The wider Sobol shell also explores changes in the other coordinates.

## Calibration, reuse and independent validation

The original study's **2,556 calibration observations** are reused: 71 gains ×
18 contexts × two noise repeats. Source signatures, simulator/configuration
hashes, candidate IDs, context metadata and noise identities are verified.
Before extending the bank, the runner checks three fixed old candidates
(`g000/g012/g030`) in every context and repeat: **108 source-parity rollouts**.
Eligibility/classification and selected numerical outputs must match within
the declared tolerances; otherwise reuse is refused.

New candidates use exactly the same two calibration noise seeds as the
original candidates. This requires **69,840 additional calibration rollouts**.
Work is vectorized in batches of up to 128 candidates and distributed across
the requested worker processes. Every candidate is evaluated, including those
that fail. All raw failures and unresolved attempts remain in the archives
and availability counts.

For each context, select the eligible minimum calibration J separately in
the original, dense-original and expanded sets. Then evaluate the union of
those winners on **four fresh common validation repeats**. Also validate the
fixed profile points in their designated frequency contexts. At most 456
validation rollouts are needed; duplicates are evaluated once.

Validation noise is disjoint from the original calibration and validation
seeds. A selected gain that fails validation remains ineligible: there is no
fallback or reselection using held-out outcomes. Comparisons retain common
eligible denominators and per-repeat signs.

## What the result answers

Read the two effects separately:

1. **Resolution inside the original box:** dense-original versus original.
2. **Genuine upper-domain extension:** expanded versus dense-original.

Positive improvement means a lower held-out mean J. The report also flags a
predeclared material improvement of at least 5%. This is a reduction in J,
not a recomputation of the earlier online near-reference success rates.
Every selected controller's actual gains, origin, eligibility and proximity
to its stage's boundaries are exported.

The profiles show whether cost continues improving beyond the old upper
limits and where feasibility is lost. A best-found point near the expanded
upper or unchanged lower boundary leaves that direction unresolved. Even
2,011 candidates establish a finite-set diagnostic, not a continuous-domain
global optimum or statistical evidence across a vehicle population.

This audit does not rerank recorded online policies against a new reference,
train a cost model, construct a public spectral basis, or generate new paid
online requests. The existing 24-test online cap and all recorded BO studies
remain unchanged. Original normalized gain coordinates `(log10 multiplier +
1)/2` are retained in metadata, including coordinates above 1. A future
larger-domain BO experiment would need a scalable feature construction and a
separately fixed protocol.

## Run locally

From the repository root in the existing `paper_FBO_LQI` environment:

```bat
git fetch origin
git switch experiment/gain-domain-audit
git pull --ff-only
python code/scripts/tracking_gain_domain.py run --workers 3
```

The command runs the parity checks, new calibration batches, fresh validation
and report generation. Each completed batch is saved atomically. Repeating
the command resumes the audit, validates cached batch identities and reuses
completed work. Keep source/configuration files fixed during a run.

Optional checks:

```bat
python code/scripts/tracking_gain_domain.py preflight
python code/scripts/tracking_gain_domain.py smoke --workers 3
```

Smoke uses three 20 m/s contexts, 128 inside Sobol draws, 64 expanded Sobol
draws and two fresh validation repeats. Its output is separate and cannot
be collected as the full audit. Neither command launches the full calibration
unless `run` is explicitly requested.

The full report is generated at:

`code/results/tracking_gain_domain_v1/report.html`

After completion, verify, stage and publish:

```bat
python code/scripts/tracking_gain_domain.py collect --workers 3
git commit -m "Add gain-domain and boundary-extension audit results"
git push
```

`collect` verifies source parity and all required batch identities, regenerates
the report and stages only this audit's output directory. It performs no
new simulations. The output includes the candidate pool, calibration
availability, selected controllers, paired held-out improvements, boundary
profiles, raw batches, provenance and a machine-readable readout.

## Software verification

Before publishing this runner, 19 focused and existing diagnostic tests passed.
The complete smoke workflow passed with three worker processes, including
source parity, calibration, fresh validation and HTML/figure generation. A
cached rerun with both simulation and spectral-basis construction prohibited
reused all 18 raw batches without changing their contents or timestamps.
The full 18-context audit was preflighted but has not been run here.
