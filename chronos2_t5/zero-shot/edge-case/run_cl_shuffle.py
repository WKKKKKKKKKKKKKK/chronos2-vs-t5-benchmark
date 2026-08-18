"""E9 -- two-arm sibling-shuffle control for Chronos-2 cross-learning.

WHY
---
Chronos-2's group attention improves clean-data accuracy on collections of *univariate*
series by ~4% WQL (19/25 datasets, sign test p~0.007), yet confers no measurable benefit
under corruption. Those two measurements together are consistent only with the pathway
transmitting POOLED STATISTICS -- level, scale, noise amplitude, seasonal shape -- rather
than the per-series structure that would be needed to repair a damaged sibling. But that
is an inference from two observations, not a test.

This is the test. We keep the target series fixed and swap out who its 99 group-mates
are, in two arms that separate the candidate mechanisms:

  native            siblings are the target's own dataset          (the reported +4%)
  foreign_samefreq  siblings from OTHER datasets, SAME frequency   (arm A)
  foreign_difffreq  siblings from OTHER datasets, DIFFERENT freq   (arm B)

The outcome table is fully diagnostic:

  gain survives A and B  ->  generic pooling of level/scale; sibling identity irrelevant
  survives A, dies in B  ->  SHARED SEASONALITY is the active ingredient
  dies in A and B        ->  not pooling at all, but a dataset-level prior

PROTOCOL MATCHING. Under `native` the whole 100-series dataset is one cross-learning
group, so each target already sits with 99 same-dataset siblings. The foreign arms
reproduce exactly that shape -- one target plus 99 siblings, group size 100 -- so the
only thing that differs across arms is WHO the siblings are.

Chronos-2's quantile head is deterministic (no sampling), so no seed is needed for the
forecast; the sibling draw is seeded and therefore reproducible.

Resumable: appends per (dataset, arm) and skips whatever is already in the CSV.

Outputs (results/):
  crosslearning_shuffle.csv   per (dataset, arm): MASE / WQL over that dataset's targets

Usage:
    python run_cl_shuffle.py                 # all 25 datasets, all arms
    python run_cl_shuffle.py --smoke         # 4 datasets
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

import run_edge_cases as RE  # noqa: E402  (pulls in the shared harness as RE.R2)

R2 = RE.R2
OUT = HERE / "results"
CSV = OUT / "crosslearning_shuffle.csv"
SEED = 0
ARMS = ["univariate", "native", "foreign_samefreq", "foreign_difffreq"]


def coarse_freq(freqstr: str) -> str:
    """Bucket a pandas period freqstr ('30T', 'H', 'W-SUN', 'Q-DEC', ...) into the
    seasonal class that matters here. Strip any multiplier and anchor first, then match
    longest-prefix-first so MIN beats M and MS beats M."""
    head = freqstr.upper().split("-")[0].lstrip("0123456789")
    for prefix, label in (("MIN", "subhourly"), ("T", "subhourly"), ("S", "subhourly"),
                          ("H", "hourly"), ("D", "daily"), ("B", "daily"),
                          ("W", "weekly"), ("MS", "monthly"), ("M", "monthly"),
                          ("Q", "quarterly"), ("A", "yearly"), ("Y", "yearly")):
        if head.startswith(prefix):
            return label
    return f"other:{head}"


def load_all(datasets):
    """Load every dataset once: targets (for scoring) and a clean sibling pool."""
    import datasets as hfds

    store = {}
    for i, (config, horizon) in enumerate(datasets, 1):
        ds = hfds.load_dataset(RE.HF_REPO, config, split="train")
        ds.set_format("numpy")
        freq = coarse_freq(pd.DatetimeIndex(ds[0]["timestamp"]).to_period().freqstr)
        gts = R2.to_gluonts_univariate(ds, RE.EDGE_MAX_SERIES)
        from gluonts.dataset.split import split
        _, tt = split(gts, offset=-horizon)
        test_data = tt.generate_instances(horizon, windows=1)
        test_input = list(test_data.input)
        store[config] = {
            "horizon": horizon, "freq": freq, "test_data": test_data,
            "test_input": test_input,
            "contexts": [np.asarray(e["target"], np.float32) for e in test_input],
        }
        print(f"  [{i}/{len(datasets)}] {config:<28} freq={freq:<10} n={len(test_input)}",
              flush=True)
    return store


def build_pool(store, target_ds: str, same_freq: bool):
    """All candidate sibling contexts from OTHER datasets, matched/mismatched on frequency.
    Built once per (dataset, arm) -- the per-target draw then just indexes into it."""
    tf = store[target_ds]["freq"]
    return [c for d, s in store.items()
            if d != target_ds and ((s["freq"] == tf) == same_freq)
            for c in s["contexts"]]


def forecast_one_in_group(pipe, target_ctx, siblings, start, horizon):
    """Cross-learning forecast of a single target embedded in a given sibling group."""
    grp = [np.asarray(target_ctx, np.float32)] + [np.asarray(c, np.float32) for c in siblings]
    q, _ = pipe.predict_quantiles(
        [torch.tensor(c) for c in grp], prediction_length=horizon,
        quantile_levels=R2.QUANTILES, cross_learning=True,
        batch_size=len(grp), limit_prediction_length=False)
    a = q[0]
    a = a.cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
    if a.ndim == 3:
        a = a[0]
    return R2._quantile_forecast(np.asarray(a, np.float32), start)


def run(smoke: bool = False):
    OUT.mkdir(parents=True, exist_ok=True)
    datasets = RE.EDGE_DATASETS[:4] if smoke else RE.EDGE_DATASETS

    done: set[tuple[str, str]] = set()
    if CSV.exists():
        prev = pd.read_csv(CSV)
        done = set(zip(prev.dataset, prev.arm))
        print(f"resuming: {len(done)} (dataset, arm) cells already done")

    print("loading datasets and building the sibling pool...", flush=True)
    store = load_all(datasets)
    freqs = pd.Series({d: s["freq"] for d, s in store.items()})
    print("\nfrequency buckets:")
    print(freqs.value_counts().to_string(), "\n")

    pipe = RE.BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2", device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=R2.DTYPE)

    B = R2.CROSS_LEARNING_BATCH
    t_all = time.perf_counter()
    for di, (config, _) in enumerate(datasets, 1):
        s = store[config]
        H, ti, td = s["horizon"], s["test_input"], s["test_data"]
        for arm in ARMS:
            if (config, arm) in done:
                continue
            t0 = time.perf_counter()

            if arm == "univariate":
                fcs = R2.forecast_univariate(pipe, ti, H)
            elif arm == "native":
                fcs, _, _ = R2.forecast_cross_learning(pipe, ti, H, batch=B)
            else:
                # Stable seed: dataset INDEX, not hash() -- Python randomises string
                # hashing per process, which would make the sibling draw irreproducible.
                rng = np.random.default_rng(
                    np.random.SeedSequence([SEED, di, ARMS.index(arm)]))
                pool = build_pool(store, config, arm == "foreign_samefreq")
                if len(pool) < B - 1:
                    print(f"  {config} / {arm}: sibling pool has {len(pool)} < {B-1} "
                          f"series -- skipped", flush=True)
                    continue
                fcs = []
                for e, ctx in zip(ti, s["contexts"]):
                    idx = rng.choice(len(pool), size=B - 1, replace=False)
                    fcs.append(forecast_one_in_group(
                        pipe, ctx, [pool[j] for j in idx],
                        e["start"] + len(e["target"]), H))

            mase, wql = R2.evaluate(fcs, td)
            pd.DataFrame([{"dataset": config, "freq": s["freq"], "arm": arm,
                           "MASE": mase, "WQL": wql, "n_series": len(ti)}]) \
              .to_csv(CSV, mode="a", header=not CSV.exists(), index=False,
                      lineterminator="\n")
            print(f"[{di}/{len(datasets)}] {config:<28} {arm:<17} "
                  f"MASE={mase:.4f} WQL={wql:.4f}  ({time.perf_counter()-t0:.0f}s)", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\ndone in {(time.perf_counter()-t_all)/60:.1f}m -> {CSV}")


if __name__ == "__main__":
    run(smoke="--smoke" in sys.argv)
