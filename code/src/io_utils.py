"""
Shared run-config / manifest / seeding plumbing (doc §26 reproducibility rules),
following the `paper_CFL_FF/src/runconfig.py` convention: one YAML file = one
fully-reproducible run, a required `kind:` dispatch key, resolved config stamped next
to results. No git-commit-hash stamping yet - git support for paper_FBO_LQI is
deferred for now.
"""
from __future__ import annotations

import datetime
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `overrides` into a COPY of `base`; returns the copy."""
    out = dict(base)
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_run_config(path: str) -> Dict[str, Any]:
    """Only checks for the required 'kind' key; each script validates the fields it
    specifically needs beyond that (paper_CFL_FF's convention)."""
    cfg = load_yaml(path)
    if "kind" not in cfg:
        raise ValueError(f"Run config {path!r} is missing the required 'kind' key.")
    cfg["_config_path"] = str(path)
    return cfg


def require_keys(cfg: Dict[str, Any], keys: List[str]) -> None:
    missing = [k for k in keys if k not in cfg]
    if missing:
        raise ValueError(
            f"Run config {cfg.get('_config_path', '<unknown>')!r} "
            f"(kind={cfg.get('kind')!r}) is missing required keys: {missing}"
        )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def write_run_manifest(output_dir: str, resolved_config: Dict[str, Any]) -> Path:
    """Stamp the resolved config + timestamp next to a run's outputs."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "resolved_config": resolved_config,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    manifest_path = out_dir / "run_manifest.yaml"
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    return manifest_path
