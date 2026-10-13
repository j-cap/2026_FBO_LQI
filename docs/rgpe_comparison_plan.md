# RGPE comparison campaign

This experiment adds one literature transfer-BO baseline to the frozen L4DC study:
the Ranking-Weighted Gaussian Process Ensemble (RGPE; Feurer, Letham, Bakshy,
2018).  Existing independent/global/dynamics-weighted results are never rerun.

## Frozen comparison

The RGPE campaign reuses:

- the same 20 confirmatory fleet seeds;
- the exact mature independent-BO histories from
  `phase2_warmstart_confirmatory_c1cds02`;
- the same six held-out targets per fleet;
- the same two target Sobol initialization evaluations;
- the same 12-evaluation target budget;
- the same 256 Sobol acquisition candidates;
- the same nonlinear plant, calibration/test episode banks and constraints;
- the same dense offline reference and post-hoc independent rescoring.

Only the transfer surrogate differs.

## RGPE implementation

For each source system, one GP is fit to its completed 25-point independent-BO
history.  Source GPs are fixed.  The target GP is refit after every target
evaluation.  Model weights are the fraction of posterior samples for which a
model achieves the minimum pairwise ranking loss on observed target points.
Target-model ranking loss uses leave-one-out predictions with the fitted target
hyperparameters.

Because there are 29 source tasks, the original RGPE weight-dilution protection
is enabled: a source model is discarded when its median sampled ranking loss is
above the 95th percentile of target-model ranking loss.  Minimum-loss ties are
awarded to the target model when it is involved; source-only ties are broken by
the deterministic method RNG.

The RGPE ensemble uses

```
mu(x)    = sum_i w_i mu_i(x)
var(x)   = sum_i w_i^2 var_i(x)
```

and the existing closed-form minimization EI over the same fixed candidate set.

Full-run frozen settings:

- posterior ranking samples: 256
- dilution cutoff: target loss 95th percentile
- source histories: 25 evaluations/system
- target initialization: 2
- target BO iterations: 10
- candidate set: 256

## Local workflow

Run all commands from `code/`.

### 1. Validate frozen inputs

```bat
python scripts\rgpe_warmstart_campaign.py --config config\runs\phase2_rgpe_smoke_c1cds02.yaml --check-inputs
```

This expects:

```
results\phase2_warmstart_confirmatory_c1cds02\processed\mature_fleet_records.parquet
```

with 30 systems x 25 source-history records for every configured seed.

### 2. Tests

```bat
pytest tests\test_rgpe.py tests\test_fleet_bo.py -q
```

For the complete regression suite:

```bat
pytest tests -q
```

### 3. Smoke campaign

```bat
python scripts\rgpe_warmstart_campaign.py --config config\runs\phase2_rgpe_smoke_c1cds02.yaml
```

Smoke output:

```
results\phase2_rgpe_smoke_c1cds02\processed\
```

It uses one fleet seed, one target per family, two BO iterations, 64 candidates,
and 32 RGPE posterior samples.  It is code validation only, not paper evidence.

### 4. Full frozen campaign

```bat
python scripts\rgpe_warmstart_campaign.py --config config\runs\phase2_rgpe_confirmatory_c1cds02.yaml --n-jobs 8
```

Adjust `--n-jobs` to local CPU/RAM.  Scientific settings are unaffected.

Full output:

```
results\phase2_rgpe_confirmatory_c1cds02\processed\
    target_convergence_traces.parquet
    target_summary.parquet
    rgpe_weight_diagnostics.csv
```

### 5. Dense-reference rescore

```bat
python scripts\rescore_bo_trajectories.py ^
  --config config\runs\phase2_rgpe_confirmatory_c1cds02.yaml ^
  --campaign-tag phase2_rgpe_confirmatory_c1cds02 ^
  --output-tag reference_robustness_rgpe ^
  --n-jobs 8
```

This produces:

```
results\reference_robustness_rgpe\processed\
    candidate_test_scores.csv
    n5pct_test.csv
```

### 6. Compare with frozen methods

```bat
python scripts\analyze_rgpe_comparison.py
```

This combines RGPE with the existing
`results\reference_robustness\processed\` results and writes:

```
results\rgpe_comparison\processed\
    combined_n5pct_test.csv
    combined_candidate_test_scores.csv
    method_summary.csv
    paired_deltas.csv
```

The primary comparison is the fleet-realization paired difference in restricted
mean N_5% for RGPE versus global pooling.  RGPE versus independent BO and versus
dynamics-weighted pooling are secondary comparisons.

## Interpretation

Three outcomes are informative:

1. RGPE approximately equals global pooling: supports the landscape-universality
   story; structured transfer cannot extract much more because source tasks are
   already strongly aligned.
2. RGPE modestly improves over pooling: residual task differences are exploitable
   by target-ranking adaptation even though physical dynamics weighting is not.
3. RGPE materially improves: reposition global pooling as the diagnostic baseline
   and RGPE as the stronger operational transfer method.

Do not edit the existing confirmatory result directories.  The RGPE campaign and
its rescored outputs use separate run tags by design.
