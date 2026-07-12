"""Quick correctness check for the Chronos-2 zero-shot pipeline.

Runs both modes (univariate + full cross-learning) on a couple of small Benchmark II
datasets and prints shapes + metrics, so you can confirm the pipeline is wired
correctly before launching the full 25-dataset run. Downloads the Chronos-2
weights from HuggingFace on first use (~all other inputs are cached locally).

Usage:
    python src/smoke_test.py
"""
import sys
import time
from pathlib import Path

import datasets as hfds
import numpy as np
import torch
from gluonts.dataset.split import split

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from datasets_lib import HF_REPO, MAX_SERIES  # noqa: E402
import run_zeroshot_chronos2 as r  # noqa: E402

# (config, horizon): exchange_rate (8 series) -> one cross-learning group;
# monash_m1_yearly (181 series) -> 2 cross-learning groups at batch=100.
CASES = [("exchange_rate", 30), ("monash_m1_yearly", 6)]


def main():
    from chronos import BaseChronosPipeline

    print(f"Loading {r.MODEL_ID} ...", flush=True)
    t0 = time.perf_counter()
    pipe = BaseChronosPipeline.from_pretrained(r.MODEL_ID, device_map="cuda", torch_dtype=r.DTYPE)
    print(f"  loaded in {time.perf_counter()-t0:.1f}s  ({type(pipe).__name__})", flush=True)

    for config, horizon in CASES:
        ds = hfds.load_dataset(HF_REPO, config, split="train")
        ds.set_format("numpy")
        gts = r.to_gluonts_univariate(ds, MAX_SERIES)
        _, tt = split(gts, offset=-horizon)
        test_data = tt.generate_instances(horizon, windows=1)
        test_input = list(test_data.input)

        print(f"\n=== {config} | n_series={len(gts)} | horizon={horizon} ===")

        fc_u = r.forecast_univariate(pipe, test_input, horizon)
        arr = fc_u[0].forecast_array
        assert arr.shape == (len(r.QUANTILES), horizon), arr.shape
        mase_u, wql_u = r.evaluate(fc_u, test_data)
        print(f"  univariate     : forecast_array{arr.shape} MASE={mase_u:.4f} WQL={wql_u:.4f}")

        fc_c, n_series, n_groups = r.forecast_cross_learning(pipe, test_input, horizon)
        assert len(fc_c) == len(test_input) and all(f is not None for f in fc_c), \
            "missing forecast in cross-learning output"
        exp_groups = (len(test_input) + r.CROSS_LEARNING_BATCH - 1) // r.CROSS_LEARNING_BATCH
        assert n_groups == exp_groups, (n_groups, exp_groups)
        mase_c, wql_c = r.evaluate(fc_c, test_data)
        print(f"  cross_learning : n_series={n_series} n_groups={n_groups} "
              f"(batch={r.CROSS_LEARNING_BATCH}) MASE={mase_c:.4f} WQL={wql_c:.4f}")

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()