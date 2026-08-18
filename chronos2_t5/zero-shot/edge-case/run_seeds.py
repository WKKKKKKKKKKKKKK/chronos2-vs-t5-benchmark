"""E3 -- Multi-seed replication of the spike sweeps.

WHY
---
`run_edge_cases.py` runs one seed (SEED = 0), and `_rng` puts `severity` into the seed,
so EVERY point on a severity curve draws different spike positions and signs. The
Chronos-T5 intensity curve is non-monotonic (peaks mid-range, then recovers) and that
shape is currently the paper's central observation -- but with a single draw per cell it
could be position luck. Chronos-2's curve being monotone on most datasets argues against
pure noise, yet "argues against" is not a measurement.

This script re-runs the two spike families under additional independent seeds so the
curves get error bars and the shape can be tested rather than eyeballed. Both randomness
sources move together with the seed, which is what a run-to-run error bar should cover:

  * the corruption draw   (`_rng` -> spike positions and signs)
  * Chronos-T5's decode   (`torch.manual_seed` -> its 20-sample stochastic decode;
                           Chronos-2's quantile head is deterministic)

Everything else is the sweep's own code, so the numbers stay comparable to
`edge_case_results.csv` (which is seed 0 and is never touched).

Resumable: results are appended per (seed, dataset), and a re-run skips whatever is
already in the CSV -- safe to kill and restart.

Outputs (results/):
  edge_case_seeds.csv   same schema as edge_case_results.csv plus a `seed` column

Usage:
    python run_seeds.py --seeds 1,2,3
    python run_seeds.py --seeds 1,2,3 --families spikes_intensity
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_edge_cases as RE  # noqa: E402

OUT = HERE / "results"
DEFAULT_CSV = "edge_case_seeds.csv"
DEFAULT_FAMILIES = ["spikes_intensity", "spikes_density"]

# The resume key is (seed, dataset) and does NOT include the family. Running a NEW family
# into an existing CSV would therefore skip every cell as "already done" and silently
# produce nothing. Each family group gets its own file via --out; the analysis concatenates.


def _arg(flag: str, default: str) -> str:
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def add_degradation_per_seed(df: pd.DataFrame) -> pd.DataFrame:
    """`RE._add_degradation` joins on (model, dataset); with several seeds stacked that
    would mix a cell with another seed's clean baseline. Apply it within each seed."""
    return pd.concat([RE._add_degradation(g) for _, g in df.groupby("seed")],
                     ignore_index=True)


def run(seeds: list[int], families: list[str], CSV: Path,
        severities: list[float] | None = None):
    OUT.mkdir(parents=True, exist_ok=True)
    # `--severities` overrides the registry grid for every family in this run. Used to
    # extend a family downward without disturbing the grid the earlier runs used; since
    # `_rng` hashes int(round(severity*1000)), new levels get their own corruption draws.
    grid = {f: (severities if severities else RE.SEVERITIES[f]) for f in families}
    conditions = [("clean", 0.0)] + [(f, s) for f in families for s in grid[f]]
    print(f"-> {CSV.name}: families={families} seeds={seeds}")
    if severities:
        print(f"   severity override: {severities}")

    done: set[tuple[int, str]] = set()
    if CSV.exists():
        prev = pd.read_csv(CSV)
        done = set(zip(prev.seed.astype(int), prev.dataset))
        print(f"resuming: {len(done)} (seed, dataset) cells already done")

    pipes = RE._load_pipes()
    cuda = torch.cuda.is_available()
    todo = [(s, d, h) for s in seeds for (d, h) in RE.EDGE_DATASETS if (s, d) not in done]
    print(f"{len(todo)} (seed, dataset) cells to run x {len(conditions)} conditions "
          f"x {len(pipes)} models", flush=True)

    t_all = time.perf_counter()
    for n, (seed, config, horizon) in enumerate(todo, 1):
        RE.SEED = seed                    # drives BOTH _rng and the T5 decode seed
        t0 = time.perf_counter()
        test_data, contexts, starts = RE.build_dataset(config, horizon)

        rows = []
        for fam, sev in conditions:
            pc = RE.perturb_contexts(config, contexts, fam, sev)
            for label, (pipe, kind) in pipes.items():
                fcs = RE.forecast(pipe, pc, starts, horizon, kind)
                if cuda:
                    torch.cuda.synchronize()
                mase, wql = RE.R2.evaluate(fcs, test_data)   # shared harness, same as the sweep
                rows.append({"seed": seed, "dataset": config, "model": label,
                             "family": fam, "severity": sev,
                             "MASE": mase, "WQL": wql, "n_series": len(contexts)})

        pd.DataFrame(rows).to_csv(CSV, mode="a", header=not CSV.exists(),
                                  index=False, lineterminator="\n")
        del contexts, test_data, starts
        if cuda:
            torch.cuda.empty_cache()
        dt = time.perf_counter() - t0
        eta = (time.perf_counter() - t_all) / n * (len(todo) - n)
        print(f"[{n}/{len(todo)}] seed={seed} {config} ({dt:.0f}s)  ETA {eta/60:.0f}m",
              flush=True)

    print(f"\ndone in {(time.perf_counter() - t_all)/60:.1f}m -> {CSV}")


if __name__ == "__main__":
    seeds = [int(s) for s in _arg("--seeds", "1,2,3").split(",")]
    families = _arg("--families", ",".join(DEFAULT_FAMILIES)).split(",")
    sev = _arg("--severities", "")
    run(seeds, families, OUT / _arg("--out", DEFAULT_CSV),
        [float(s) for s in sev.split(",")] if sev else None)
