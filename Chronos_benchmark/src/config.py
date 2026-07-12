"""Central path configuration for the Chronos-T5 Benchmark II reproduction.

All paths derive from the project root so the benchmark is portable: clone the
repo anywhere and the scripts resolve their own results / models dirs with no
edits. Each path can be overridden via an environment variable (e.g. results on a
shared mount). Datasets are not stored here -- they live in the HuggingFace cache.

Resolution order for each path:
    1. the matching environment variable (if set);
    2. otherwise a default derived from PROJECT_ROOT.

Environment variables
----------------------
    CHRONOS_BENCH_ROOT     project root      (default: parent of this src/ dir)
    CHRONOS_BENCH_RESULTS  result CSV / MD   (default: <root>/results)
    CHRONOS_BENCH_MODELS   model checkpoints (default: <root>/models)
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_path(var: str, default: Path) -> Path:
    """Return env var `var` as a Path if set & non-empty, else `default`."""
    val = os.environ.get(var)
    return Path(val).expanduser() if val else default


# Project root = parent of src/ (this file lives in <root>/src/config.py).
PROJECT_ROOT: Path = _env_path("CHRONOS_BENCH_ROOT", Path(__file__).resolve().parent.parent)
RESULTS_DIR: Path = _env_path("CHRONOS_BENCH_RESULTS", PROJECT_ROOT / "results")
MODELS_DIR: Path = _env_path("CHRONOS_BENCH_MODELS", PROJECT_ROOT / "models")


if __name__ == "__main__":
    # `python config.py` prints the resolved layout -- handy on a new machine to
    # confirm where everything will read/write before launching a long run.
    print("Resolved benchmark paths:")
    for name in ("PROJECT_ROOT", "RESULTS_DIR", "MODELS_DIR"):
        p = globals()[name]
        exists = "ok" if Path(p).exists() else "MISSING"
        print(f"  {name:14s} = {p}   [{exists}]")
