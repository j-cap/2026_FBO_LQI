# Code

This directory is the root of the experimental framework.

The communication follow-up to the completed dense-gain BO comparison is
documented in [docs/tracking_communication.md](../docs/tracking_communication.md).
Run `python code/scripts/tracking_communication.py run --workers 3` from the
repository root. It first audits exact compact exchange without new rollouts,
then runs the fresh matched A/B/D/E comparison at 1/2/4-round contact intervals.

The trajectory-tracking diagnostic is documented in
[docs/tracking_diagnostic.md](../docs/tracking_diagnostic.md). Its isolated runner
is `python code/scripts/tracking_diagnostic.py smoke --workers 4` from the repository
root; the full experiment uses `run`. It reuses the CFL-FF nonlinear plant and
feedback structure and tests tuning headroom, context dependence, and donor
selection before adding a new BO algorithm.

Recommended structure once the experiment code is added:

```text
code/
├── src/
├── scripts/
├── config/
│   └── runs/
├── tests/
├── results/
├── requirements.txt
└── pyproject.toml
```

Raw and intermediate experiment outputs should remain under `results/`. Publication-ready figures should be exported to `../latex/figures/`.

The upcoming fleet-maturity study should live in this framework rather than in a parallel code path.
