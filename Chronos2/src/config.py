"""Central path configuration for the Chronos-2 zero-shot study.

Mirrors the path-resolution scheme of the sibling Chronos-T5 benchmark
(`Chronos_benchmark/src/config.py`) so the two projects are laid out identically
and remain portable: clone anywhere and the scripts resolve their own results /
reference dirs with no edits. Each path can be overridden via an environment
variable.

Resolution order for each path:
    1. the matching environment variable (if set);
    2. otherwise a default derived from PROJECT_ROOT.

Environment variables
----------------------
    CHRONOS2_ROOT       project root      (default: parent of this src/ dir)
    CHRONOS2_RESULTS    result CSV/MD     (default: <root>/results)
    CHRONOS2_REFERENCE  paper refs        (default: <root>/reference)
    CHRONOS2_MODELS     model checkpoints (default: <root>/models)
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_path(var: str, default: Path) -> Path:
    """Return env var `var` as a Path if set & non-empty, else `default`."""
    val = os.environ.get(var)
    return Path(val).expanduser() if val else default


# Project root = parent of src/ (this file lives in <root>/src/config.py).
PROJECT_ROOT: Path = _env_path("CHRONOS2_ROOT", Path(__file__).resolve().parent.parent)
RESULTS_DIR: Path = _env_path("CHRONOS2_RESULTS", PROJECT_ROOT / "results")
REFERENCE_DIR: Path = _env_path("CHRONOS2_REFERENCE", PROJECT_ROOT / "reference")
MODELS_DIR: Path = _env_path("CHRONOS2_MODELS", PROJECT_ROOT / "models")


if __name__ == "__main__":
    print("Resolved Chronos-2 study paths:")
    for name in ("PROJECT_ROOT", "RESULTS_DIR", "REFERENCE_DIR", "MODELS_DIR"):
        p = globals()[name]
        exists = "ok" if Path(p).exists() else "MISSING"
        print(f"  {name:14s} = {p}   [{exists}]")