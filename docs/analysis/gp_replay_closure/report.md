# A replay follow-up: original decisions validated, refitting is platform dependent

25 November 2026. Original trace commit:
`4b14a428de677d81634050edeb8186a533194335`.
Confirmation results remain at
`ef7192086b4318e93cf2f960490f39a34032a4fa`.

## Decision

**The validity concern about A's recorded decisions is resolved.** The original
Windows environment reproduces all 594 BO decisions in the eight previously
affected jobs. Their input histories and refit counters match the Linux traces
exactly. Reusing the original fitted GP parameters on Linux independently
recovers all 594 recorded controller choices with the frozen acquisition code.

The remaining limitation is narrower: fresh GP hyperparameter optimization
from the same histories is not identical across these numerical environments.
This follow-up does not claim that fresh Linux refits now reproduce the Windows
trajectory. It validates the original decisions and their predictions from the
logged fit states, without changing any experimental result.

## What the new traces establish

The original and Linux traces use the same tracing source, confirmation
signature, eight jobs and 594 uniquely paired BO events. The follow-up checks
that those are all BO events for A in the selected jobs, rather than a selected
subset. It reconstructs every prior training prefix from the frozen paid
histories and verifies the training fingerprints, eligible observation counts,
scheduled refits and non-refit parameter carryover.

Both traces report Python 3.11.16, NumPy 2.3.2, SciPy 1.16.1 and pandas 2.3.1.
The original platform is Windows and the previous audit platform is Linux.
The nominal OpenBLAS versions are also the same (NumPy 0.3.30, SciPy 0.3.28),
while their platform/compiler builds differ. Matching package version numbers
does not make those numerical implementations identical.

| Check | Result |
|---|---:|
| Original-environment choices matching the campaign | 594 / 594 |
| Identical paired prior-history fingerprints and fit counters | 594 / 594 |
| Original logged GP states reproducing campaign choices on Linux | 594 / 594 |
| Linux logged GP states reproducing the earlier Linux diagnostic choices | 594 / 594 |
| Fresh Linux-trace choices differing from the campaign | 15 / 594 |
| Differing choices associated with changed fitted parameters | 15 / 15 |

The comparison spans these eight jobs:

| Seed | BO decisions | Original-environment mismatches | Linux-refit differences | Recovered with original fitted state |
|---:|---:|---:|---:|---:|
| 2103 | 80 | 0 | 1 | 80 |
| 2106 | 77 | 0 | 2 | 77 |
| 4104 | 87 | 0 | 4 | 87 |
| 4111 | 73 | 0 | 1 | 73 |
| 4112 | 65 | 0 | 3 | 65 |
| 4117 | 69 | 0 | 1 | 69 |
| 4123 | 73 | 0 | 2 | 73 |
| 4129 | 70 | 0 | 1 | 70 |

There are 15 differing Linux diagnostic choices across eight jobs. The initial
strict fresh-replay audit stopped at the first difference in each job; this is
why its earlier table had eight first differences rather than fifteen rows.

## Isolating refitting from prediction and acquisition

The follow-up restores each recorded hyperparameter vector and disables any
new optimization. It then calls the frozen acquisition routine with the same
prior observations, public inputs, eligibility penalty, random seed and
unqueried-candidate restriction. A guard raises an error if the GP optimizer
is invoked during this logged-state replay. No score tie threshold is relaxed.

With the original parameters, the largest absolute differences from the
original trace are approximately:

| Quantity | Maximum absolute error |
|---|---:|
| Recorded-action acquisition score | 1.51e-14 |
| Predicted log-cost mean | 1.60e-14 |
| Latent variance | 1.78e-15 |
| Eligibility probability | 2.22e-16 |
| GP log likelihood | 4.05e-13 |

Thus the original fitted states are sufficient to recover the original
decisions on Linux. Restoring the Linux fitted states separately recovers the
Linux diagnostic choices. This locates the choice differences in the fitted
GP state, rather than a different training history, peer rule or acquisition
definition. The individual contribution of compiler, math library or BLAS
implementation has not been isolated, and no specific library bug is claimed.

Fitted log-parameter vectors differ by more than 1e-8 on 547 BO rows; many rows
reuse a previous fit, so that is not a count of independent optimization calls.
The largest coordinate difference is about 3.219. These fit changes can affect
rankings, rather than merely selecting another member of the declared 1e-12
tie band. That earlier warning remains valid for fresh-refit portability.

## What remains frozen

No plant, desired trajectory, objective, gain pool, query cap, prior,
hyperparameter bound, acquisition coefficient or communication rule is changed.
No controller is newly simulated and no alternative policy campaign is run.
Every restoration uses a parameter state logged after fitting the paid prior
history and before receiving the current test outcome.

The previous 13-table confirmation audit and its raw source inputs remain
unchanged. Its historical `all_four_paid_policies_exactly_replayed=false` field
still describes fresh replay on that Linux environment. It is not rewritten
to true: logged-state reproduction and identical fresh optimization are
different checks. This follow-up has its own [verification.json](verification.json).

## Consolidated scientific conclusion

The [confirmation conclusions](../spectral_confirmation/report.md) remain:

- D256 meets the declared primary A criteria on the committed campaign in both
  tested conditions. The original A decisions are now validated by the new
  original-environment and logged-state evidence.
- The homogeneous B comparison passes. Heterogeneous D256/B improves context
  success and paired mean cost but fails the complete-eligibility requirement.
- Added cost-model sharing over E256 is supported in the homogeneous condition.
  Its heterogeneous benefit is not confirmed: the success interval includes
  zero, paired mean cost is worse, and complete-policy eligibility is lower.

We can retain these findings and move to communication work. The useful next
target is an algebraically exact reduction in messages that preserves D256's
posterior predictions and controller queries. The homogeneous D/E result is
the demonstrated additional-sharing benefit for that comparison. Reducing
messages would not establish the presently unconfirmed heterogeneous benefit.

For future reproducibility, retain fitted GP parameter states and refit
counters together with numerical-backend information. This is an artifact and
logging requirement; it does not require changing A's frozen algorithm or
rerunning the completed confirmation solely to resolve this audit concern.

Reproduce this read-only follow-up from the repository root:

```bat
python code/scripts/compare_tracking_gp_traces.py
```

The [per-decision comparison](trace_comparison.csv),
[changed-choice subset](changed_decisions.csv) and [job summary](job_summary.csv)
are saved beside this report.
