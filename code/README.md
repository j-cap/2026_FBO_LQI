# Code

This directory is the root of the experimental framework.

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
