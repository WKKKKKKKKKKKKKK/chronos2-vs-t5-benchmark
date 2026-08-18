"""E2a -- is Chronos-T5's flat severity slope an artefact of its stochastic decode?

THE QUESTION
------------
Chronos-T5's error does not track spike magnitude (rho ~ -0.08) while Chronos-2's does.
Chronos-T5 differs from Chronos-2 in three ways at once, and one of them is that it
forecasts by drawing 20 samples autoregressively while Chronos-2 emits quantiles in a
single deterministic pass. Sampling noise is therefore a live candidate: if the decode adds
enough variance to the measured degradation, a real but shallow slope could be buried in it
and read as zero.

E1 already excluded the two representation-level candidates (bounded-grid clamping,
mean-scale inflation) by direct measurement. Decode is the remaining cheap one, and unlike
the Chronos-Bolt comparison it is a genuine SINGLE-VARIABLE manipulation: the corrupted
contexts are byte-identical across runs and only `torch.manual_seed` moves.

THE DESIGN
----------
`run_edge_cases.SEED` normally drives both the corruption draw and the decode, which is
right for a run-to-run error bar and useless here. `DECODE_SEED` decouples them. We hold
the corruption fixed at SEED = 0 and sweep DECODE_SEED, on the spike-magnitude family only
-- that is where the flat slope lives.

Chronos-2 is not re-run: its quantile head is deterministic, so a decode seed does nothing
to it. Its numbers come from the existing sweep.

WHAT THE OUTCOMES MEAN
----------------------
Averaging over D decode seeds cuts the decode-induced noise in the measured degradation by
roughly sqrt(D). So:

  slope becomes positive once decode noise is averaged down
      -> the flat reading was sampling noise, not undersensitivity. C1 would need
         restating, and the paper gains a mechanism.
  slope stays flat
      -> decode is not the explanation either. Combined with E1 that leaves the
         phenomenon characterised and all three cheap mechanisms excluded, which is
         a weaker but honest position -- and a more useful one than an unfalsified guess.

Either way we also get something the study currently lacks: a direct measurement of how
much of the curve-to-curve variation in Chronos-T5's degradation is decode noise rather
than corruption placement.

Resumable per (decode_seed, dataset). Output: results/decode_ablation.csv

Usage:
    python run_decode_ablation.py                      # decode seeds 0..7
    python run_decode_ablation.py --decode-seeds 0,1   # smoke test
"""
from __future__ import annotations

import hashlib
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
CSV = OUT / "decode_ablation.csv"
FAMILY = "spikes_intensity"
CORRUPTION_SEED = 0          # held FIXED; this is the whole point of the experiment


def _arg(flag: str, default: str) -> str:
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def ctx_digest(contexts) -> str:
    """Fingerprint of the corrupted contexts, so 'identical inputs' is checked, not assumed.

    If this differs between two decode seeds, the experiment is not a single-variable
    manipulation and its result means nothing -- so we assert on it rather than trust it.
    """
    h = hashlib.sha256()
    for c in contexts:
        a = np.ascontiguousarray(np.asarray(c, dtype=np.float32))
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def run(decode_seeds: list[int]):
    OUT.mkdir(parents=True, exist_ok=True)
    conditions = [("clean", 0.0)] + [(FAMILY, s) for s in RE.SEVERITIES[FAMILY]]

    done: set[tuple[int, str]] = set()
    if CSV.exists():
        prev = pd.read_csv(CSV)
        done = set(zip(prev.decode_seed.astype(int), prev.dataset))
        print(f"resuming: {len(done)} (decode_seed, dataset) cells already done")

    RE.SEED = CORRUPTION_SEED                 # corruption held fixed for every run
    pipes = {k: v for k, v in RE._load_pipes().items() if v[1] == "t5"}
    if not pipes:
        sys.exit("no Chronos-T5 pipeline found -- check RE.MODELS")
    print(f"models: {list(pipes)}  (Chronos-2 omitted: its quantile head is deterministic)")

    cuda = torch.cuda.is_available()
    todo = [(ds_, d, h) for ds_ in decode_seeds
            for (d, h) in RE.EDGE_DATASETS if (ds_, d) not in done]
    print(f"{len(todo)} (decode_seed, dataset) cells x {len(conditions)} conditions",
          flush=True)

    digests: dict[tuple[str, float], str] = {}
    t_all = time.perf_counter()
    for n, (dseed, config, horizon) in enumerate(todo, 1):
        RE.DECODE_SEED = dseed
        t0 = time.perf_counter()
        test_data, contexts, starts = RE.build_dataset(config, horizon)

        rows = []
        for fam, sev in conditions:
            pc = RE.perturb_contexts(config, contexts, fam, sev)

            # The corruption must not move with the decode seed. Check it.
            key = (config, sev if fam != "clean" else -1.0)
            dg = ctx_digest(pc)
            if key in digests and digests[key] != dg:
                sys.exit(f"FATAL: corrupted contexts changed with decode seed at {key}. "
                         "The experiment is not single-variable; aborting rather than "
                         "writing meaningless rows.")
            digests[key] = dg

            for label, (pipe, kind) in pipes.items():
                fcs = RE.forecast(pipe, pc, starts, horizon, kind)
                if cuda:
                    torch.cuda.synchronize()
                mase, wql = RE.R2.evaluate(fcs, test_data)
                rows.append({"decode_seed": dseed, "dataset": config, "model": label,
                             "family": fam, "severity": sev, "MASE": mase, "WQL": wql,
                             "n_series": len(contexts), "ctx_digest": dg})

        pd.DataFrame(rows).to_csv(CSV, mode="a", header=not CSV.exists(),
                                  index=False, lineterminator="\n")
        del contexts, test_data, starts
        if cuda:
            torch.cuda.empty_cache()
        dt = time.perf_counter() - t0
        eta = (time.perf_counter() - t_all) / n * (len(todo) - n)
        print(f"[{n}/{len(todo)}] decode_seed={dseed} {config} ({dt:.0f}s)  "
              f"ETA {eta/60:.0f}m", flush=True)

    RE.DECODE_SEED = None                     # leave the module as we found it
    print(f"\ndone in {(time.perf_counter() - t_all)/60:.1f}m -> {CSV}")


if __name__ == "__main__":
    run([int(s) for s in _arg("--decode-seeds", "0,1,2,3,4,5,6,7").split(",")])
