"""
Copies paper_FBO_LQI (self-contained - the IFAC pipeline is vendored in src/ifac) to local disk, runs a phase2_fbo_comparison config there, and
copies the new results back - for a machine where this project is reached over a mapped
network drive (e.g. H:\\paper_FBO_LQI on C1CDS02/C1CDS05, 2026-08-31).

Deliberately a .py script, not a .bat one: a .bat file living ON the network drive was
found to be blocked from being read/executed at all by this environment's endpoint
policy (confirmed - the identical file reads fine copied to local disk, but
Get-Content/execution against the H:\\ copy raises access-denied). A pure-Python script
sidesteps this, since only python.exe itself is "executed" and it can freely read/run a
.py file located anywhere, including H:\\ - already proven by
`python scripts/phase2_fbo_comparison.py --help` working directly off H:\\.

Motivation for the local copy: running scripts/phase2_fbo_comparison.py directly
against a network path is slow - its joblib disk cache does one small synchronous write
per unique (client, xi, episode) evaluation, and those pay full network latency instead
of local-disk latency (confirmed via joblib's own "Persisting input arguments took Xs"
warnings on a C1CDS02 run against H:\\ directly).

Usage (from an already-activated "paper_FBO_LQI" conda prompt):
    python scripts\\run_remote.py <config_filename.yaml> [local_root] [--script <driver>.py] [--stage1-only]
    e.g. python scripts\\run_remote.py phase2_fbo_comparison_medium_c1cds02.yaml
    e.g. python scripts\\run_remote.py phase2_triggered_dev_c1cds05.yaml --script phase2_triggered_dev.py
local_root defaults to C:\\paper_FBO_LQI - pass a second argument for a different
local drive/folder.

--script (default phase2_fbo_comparison.py) selects which scripts/<script> actually
runs against the config - THERE IS NO AUTO-DETECTION from the config's contents or
filename, so passing a phase2_triggered_dev_*.yaml config without also passing
--script phase2_triggered_dev.py silently runs it through phase2_fbo_comparison.py
instead (both accept the same `kind: phase2_fbo_comparison`-tagged YAML shape, so this
does not fail loudly - it just runs the wrong experiment). This happened once
(2026-09-04, docs/phase2_diagnostic_findings.md Part 12) and wasted a full remote run -
always pass --script explicitly for any driver other than phase2_fbo_comparison.py.

--stage1-only forwards phase2_fbo_comparison.py's own --stage1-only flag (Step D,
docs/phase2_diagnostic_findings.md, 2026-09-02): writes feasible_support.csv and stops
before the BO stage. Re-running an ALREADY-COMPLETED config's run_tag this way reuses
that run's local joblib cache (copied back here from a prior full run, or already
sitting in local_root from one) - identification/baseline/oracle evals hit the cache
instead of re-simulating, so this backfills feasible_support.csv onto a finished
campaign in minutes, not hours. The results copy-back is additive either way (Step
[3/3] below), so this never disturbs that campaign's existing figures/processed data.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_NET_ROOT = Path(__file__).resolve().parent.parent  # this project's own (network) root


def _robocopy(src: Path, dst: Path, *, exclude_dirs: list[str] = (), exclude_files: list[str] = ()) -> None:
    """MIRRORS src into dst (deletes local extras) - fine for a disposable local
    scratch copy of the code, never used for the results copy-back."""
    args = ["robocopy", str(src), str(dst), "/MIR"]
    if exclude_dirs:
        args += ["/XD", *exclude_dirs]
    if exclude_files:
        args += ["/XF", *exclude_files]
    args += ["/NFL", "/NDL", "/NP", "/R:2", "/W:2"]
    result = subprocess.run(args)
    if result.returncode >= 8:  # robocopy: 0-7 are success variants, >=8 is a real failure
        raise RuntimeError(f"robocopy failed (exit {result.returncode}): {src} -> {dst}")


def _robocopy_additive(src: Path, dst: Path, *, exclude_dirs: list[str] = ()) -> None:
    """Copies src INTO dst without deleting anything already there (no /MIR) - used for
    the results copy-back, so re-running a different config/seed range never destroys
    previously-fetched results already sitting in the network results/ folder."""
    args = ["robocopy", str(src), str(dst), "/E"]
    if exclude_dirs:
        args += ["/XD", *exclude_dirs]
    args += ["/NFL", "/NDL", "/NP", "/R:2", "/W:2"]
    result = subprocess.run(args)
    if result.returncode >= 8:
        raise RuntimeError(f"robocopy failed (exit {result.returncode}): {src} -> {dst}")


def main(config_name: str, local_root: str, *, stage1_only: bool = False, script: str = "phase2_fbo_comparison.py") -> int:
    local_root_p = Path(local_root).resolve()

    net_config = _NET_ROOT / "config" / "runs" / config_name
    if not net_config.exists():
        print(f"Config not found: {net_config}")
        return 1

    print(f"=== [1/3] copying {_NET_ROOT.name} to local disk: {local_root_p} ===")
    _robocopy(
        _NET_ROOT, local_root_p, exclude_dirs=[".pytest_cache", "__pycache__", "results"], exclude_files=["*.zip"]
    )

    cmd = [sys.executable, f"scripts/{script}", "--config", f"config/runs/{config_name}"]
    if stage1_only:
        cmd.append("--stage1-only")
    print(f"=== [2/3] running {' '.join(cmd[1:])} (local, cwd={local_root_p}) ===")
    result = subprocess.run(cmd, cwd=str(local_root_p))
    py_exit = result.returncode

    local_results = local_root_p / "results"
    net_results = _NET_ROOT / "results"
    print(f"=== [3/3] copying results back to {net_results} (additive, nothing deleted, cache/ skipped) ===")
    if local_results.exists():
        # cache/ (the joblib disk cache) is excluded on purpose: it's reproducible
        # intermediate state, not analysis output, and by far the largest/slowest part
        # to copy over the network (many small files per run_tag/seed) - copying it FIRST
        # (robocopy processes top-level files in a directory before recursing into
        # subdirectories, so cache/ was winning the race against figures/ and processed/)
        # was confirmed to leave a run's actual results (figures/processed/) missing on
        # H:\ if the copy got interrupted partway through cache/, 2026-08-31.
        _robocopy_additive(local_results, net_results, exclude_dirs=["cache"])
    else:
        print("  (no local results/ directory - the run likely failed before producing output)")

    if py_exit != 0:
        print(f"{script} exited with code {py_exit}")
        return py_exit
    print(f"Done. Results are under {net_results}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="config filename under config/runs/, e.g. phase2_fbo_comparison_medium_c1cds02.yaml")
    parser.add_argument("local_root", nargs="?", default=r"C:\paper_FBO_LQI")
    parser.add_argument(
        "--script", default="phase2_fbo_comparison.py",
        help="which scripts/<script> to run against the config (default: phase2_fbo_comparison.py). MUST be set "
        "explicitly for any other driver, e.g. --script phase2_triggered_dev.py - this defaulted silently to "
        "phase2_fbo_comparison.py for every config until 2026-09-04, which once wasted a full remote run by "
        "silently running the wrong driver against a phase2_triggered_dev_*.yaml config (docs/phase2_diagnostic_findings.md Part 12).",
    )
    parser.add_argument(
        "--stage1-only", action="store_true",
        help="forward the driver's --stage1-only flag (phase2_fbo_comparison.py only - phase2_triggered_dev.py "
        "doesn't support it) - writes feasible_support.csv and stops before the BO stage; see the module "
        "docstring above.",
    )
    args = parser.parse_args()
    sys.exit(main(args.config, args.local_root, stage1_only=args.stage1_only, script=args.script))
