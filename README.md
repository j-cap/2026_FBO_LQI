# 2026_FBO_LQI

Research repository for online fleet-assisted calibration of trajectory-tracking
controller gains. Earlier LQI experiments remain archived.

Completed validity check: [offline gain-domain and boundary-extension audit](docs/analysis/gain_domain/report.md).
Original-box refinement from 71 to 1,033 points reduces mean held-out J by
5.85%; the 2,011-point expanded set adds 2.32%. All seven selected exterior
winners exceed only the old K_y bound. All six saved tables and 72,396
calibration observations are verified. Reproduce without new simulation:
`python code/scripts/analyze_tracking_gain_domain.py`.
This is an offline search-domain check; the recorded online methods remain frozen.

Current [consolidation and bigger-picture assessment](docs/fedbo_consolidation.md)
freeze the exploration ablation and distinguish near-reference success from
failed-test burden, mean cost and complete-policy eligibility.

Latest completed experiment: [fixed small-bonus comparison](docs/tracking_six_speed_small_bonus.md).
The [three-study audit](docs/analysis/six_speed_consolidation/report.md) replays
all 10,368 paid decisions and reconstructs 36 tables. Beta=0.5 D improves
success over A/B, matched E and zero-bonus D in all three new seeds, while
recovering 0.3 Hz discoveries. It increases failed tests and loses complete
policies against zero-bonus D, so its clean-improvement criterion fails.
Reproduce with `python code/scripts/consolidate_tracking_six_speed.py --workers 3`.

Previous completed experiment: [matched six-speed zero-bonus D/E comparison](docs/tracking_six_speed_zero_bonus.md).
The [audited results](docs/analysis/six_speed_zero_bonus/report.md) reproduce
all 3,888 paid decisions and 11 tables. Zero-bonus D256 has the highest average
success and a better failure tradeoff than beta=2, with an added sharing
benefit against matched E256. It still loses to A in one seed, has fewer
complete eligible policies than B, and misses every 0.3 Hz near-reference
target. Reproduce with
`python code/scripts/analyze_tracking_six_speed_zero_bonus.py --workers 3`.

Previous completed experiment: [six-speed contextual BO pilot](docs/tracking_six_speed.md).
The [audited analysis](docs/analysis/six_speed_contextual_pilot/report.md) reproduces
all 2,592 paid decisions and 11 tables. D256 improves encountered-context success
over A/E256 but loses complete-policy eligibility; B has the highest success.
The fixed-history uncertainty diagnostic motivates examining BO exploration
before further task or communication changes. This remains a three-seed
homogeneous pilot, with the completed reference experiments preserved.

Previous completed experiment: [frozen D256 confirmation](docs/tracking_confirmation.md).
The [confirmation analysis](docs/analysis/spectral_confirmation/report.md) reproduces
all score tables: the declared A criterion passes in both conditions, while
added cost sharing is supported in the homogeneous condition and remains
unconfirmed against E256 in the heterogeneous condition. The
[GP replay follow-up](docs/analysis/gp_replay_closure/report.md) validates the
original A choices and recovers all 594 affected BO decisions from logged fit
states on Linux. Fresh GP refits remain platform dependent. Reproduce with
`python code/scripts/analyze_tracking_confirmation.py --workers 6`.
The follow-up uses `python code/scripts/compare_tracking_gp_traces.py`.
The [seed-1102 discovery inspection](docs/analysis/spectral_discovery_chains/report.md)
locates missed useful gains before peer transfer, retaining the existing method.

Previous completed experiment: [matched online spectral BO](docs/tracking_spectral.md).
The [audited spectral results](docs/analysis/spectral_online_cost_model/report.md)
show aggregate D256 improvements over A and B in both tested physical conditions,
with a loss to A in seed 1102 in each condition. Matched local and D384 controls
limit the claim of sharing benefit and monotonic representation accuracy.
Reproduce the audit with `python code/scripts/analyze_tracking_spectral.py`.

Previous completed online calibration experiment: [joint shared/local Bayesian cost
learning](docs/tracking_hierarchical.md), with A as the primary benchmark and B
as the secondary benchmark. It reuses the
completed [common peer-verification comparison](docs/tracking_peer_assisted_cost_model.md)
for evaluation. The previous experiments remain frozen.

The [audited joint-model results](docs/analysis/joint_shared_local_cost_model/report.md)
show a heterogeneous final-budget improvement over A, a homogeneous loss to A,
and a heterogeneous context-success tradeoff against B.

The [low-frequency ranking diagnostic](docs/analysis/low_frequency_rankings/report.md)
uses saved histories to locate a contribution from the finite kernel
approximation, with an exact-kernel analytical reference and no new rollouts.

The [kernel representation assessment](docs/analysis/kernel_representation_assessment/report.md)
retains D's model and identifies public spectral features as a more accurate
representation. Reproduce it with `python code/scripts/assess_tracking_kernels.py --workers 6`;
it does not launch a controller experiment.

The [consolidated findings](docs/fedbo_consolidation.md) freeze the completed
A/B/C comparison and record the joint-model follow-up.
The [heterogeneous cost-model review](docs/heterogeneous_cost_model_review.md)
motivates joint shared/local inference with the same tracking task and budget.

## Repository structure

- `code/` — simulation, optimization, experiment drivers, configurations, tests, and generated analysis outputs.
- `latex/` — L4DC/PMLR manuscript, bibliography, review notes, and publication-ready figures.

The manuscript is self-contained inside `latex/` and can be used as the Overleaf project root. Publication figures generated by the experiment code should be copied or written to `latex/figures/`.

For the L4DC 2027 submission, `main.tex` uses the `l4dc2026.cls` class currently linked by the official L4DC 2027 call-for-papers page. The class is kept unmodified so that it can be replaced directly if the organizers publish a 2027-specific revision. Do not manually override the PMLR volume/year header; replace the class with the organizer-provided 2027 revision when available. The current public L4DC 2027 instructions do not specify double-blind review, so the manuscript remains in the class's default named-author mode; switch to the optional `anon` class mode only if the OpenReview submission instructions explicitly require it.

## Workflow

1. Develop and run experiments under `code/`.
2. Keep raw/intermediate outputs under `code/results/`.
3. Export only final paper figures to `latex/figures/`.
4. Regenerate figures with `python code/scripts/make_paper_figures.py` and `python code/scripts/make_maturity_figures.py` when figure code changes. The paper-figure script skips historical figures whose ignored `code/results/` archives are not present locally and preserves their tracked PDFs; use `--strict` only on a machine containing the complete archived result tree.
5. Build the L4DC submission from `latex/main.tex`.
6. Build optional external supplementary material from `latex/supplementary.tex`; it is deliberately not included by `main.tex`.
## RGPE transfer-BO comparison

The opt-in RGPE literature baseline is isolated from the frozen three-method confirmatory campaign. See `docs/rgpe_comparison_plan.md` for the frozen protocol, input checks, tests, smoke/full commands, dense-reference rescoring, and paired analysis workflow.

