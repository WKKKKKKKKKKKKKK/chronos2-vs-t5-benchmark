"""Strict, paper-faithful zero-shot evaluation using the OFFICIAL Chronos method.

This mirrors the official
chronos-forecasting `scripts/evaluation/evaluate.py`:
  - gluonts `split` / `generate_instances` for the backtest windows (no
    series-length filter, no min-context cut),
  - gluonts `MASE` and `MeanWeightedSumQuantileLoss` metrics (seasonality inferred
    from the data frequency by gluonts; constant series handled gluonts' way),
so per-dataset numbers are directly comparable to the paper's reference CSVs.

The only deviation from the paper is a deterministic per-dataset series cap
(MAX_SERIES) for laptop tractability; metric *definitions* are identical.

Outputs:
  results/zeroshot_official_results.csv   -- our gluonts MASE/WQL per dataset
  results/OFFICIAL_REPORT.md              -- ours vs paper + aggregated relative score
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gluonts.dataset.split import split                                  # Split train/test by the "last H" observations.
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
from gluonts.itertools import batcher                                    # Batch a list of dicts (gluonts dataset) into smaller lists for memory efficiency.
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.model.forecast import SampleForecast                        # GluonTS "forecast results" container.
from scipy.stats import gmean

SRC = Path(__file__).resolve().parent.parent   # src root (config.py, datasets_lib.py live here)
sys.path.insert(0, str(SRC))
from config import PROJECT_ROOT, RESULTS_DIR as RESULTS  # noqa: E402
from datasets_lib import BENCHMARK_II, HF_REPO, MAX_SERIES  # noqa: E402

from chronos import BaseChronosPipeline  # noqa: E402      Chronos model

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MODEL_ID = "amazon/chronos-t5-small" # HuggingFace model ID for the official Chronos-T5 Small zero-shot model (paper's reference).
NUM_SAMPLES = 20    # Generate 20 samples per series for the stochastic forecast, as in the paper.
BATCH_SIZE = 32
SEED = 0   # fixed so the stochastic 20-sample forecast is reproducible
# Paper's official reference numbers (Seasonal-Naive + Chronos-T5 Small zero-shot),
# bundled in the repo under reference/ so the comparison is self-contained and
# portable (originally from chronos-forecasting/scripts/evaluation/results).
REFERENCE_DIR = PROJECT_ROOT / "reference"


def to_gluonts_univariate(ds, cap: int):
    """HF dataset -> list of gluonts {start, target}, capped to `cap` series.
    Input:
    ds: HuggingFace dataset with a timestamp column and one or more sequence columns (gluonts "target" is a sequence, so we need to find it).
        The timestamp column is used to infer the frequency for the gluonts Period start.
        The sequence columns are converted to numpy arrays and stored in the "target" field of the gluonts dict.
        The function returns a list of dicts, each with a "start" (Period) and "target" (numpy array) key, suitable for gluonts evaluation.
    cap: maximum number of series to keep (deterministic evenly-spaced subsample).
        If the dataset has more than `cap` series, it will be subsampled to `cap` series.
        If the dataset has fewer than `cap` series, all series will be kept.
        The function returns a list of dicts, each with a "start" (Period) and "target" (numpy array) key, suitable for gluonts evaluation.
    return:
    e.g. [
    {"start": Period("2020-01-01", freq="D"), "target": array([1., 2., 3.])},
    {"start": Period("2020-01-01", freq="D"), "target": array([4., 5., 6.])},
    ...
    ]
    """
    import datasets as hfds

    seq_fields = [c for c in ds.features if isinstance(ds.features[c], hfds.Sequence)] # Find the sequence fields (gluonts "target" is a sequence, so we need to find it).
    if "timestamp" in seq_fields:
        seq_fields.remove("timestamp")
    freq = pd.DatetimeIndex(ds[0]["timestamp"]).to_period().freqstr                    # Infer the frequency from the first series' timestamp (gluonts needs a freq for the Period start).
    n = len(ds)
    keep = range(n) if n <= cap else set(np.linspace(0, n - 1, cap).astype(int).tolist())
    out = []
    for i, row in enumerate(ds):
        if i not in keep:
            continue
        for f in seq_fields:
            out.append({"start": pd.Period(row["timestamp"][0], freq=freq),
                        "target": np.asarray(row[f], dtype=np.float32)})
    return out


def chronos_forecasts(pipeline, test_input, horizon):
    """Replicates the official generate_forecasts (SampleForecast from quantiles).

    Seeds torch first so the 20-sample stochastic forecast is reproducible. This
    is the single sampling point shared by zero-shot eval, one-shot eval, and the
    notebook demo, so seeding here makes the whole experiment deterministic and
    order-independent (each dataset starts from the same RNG state).

    input:
    pipeline: Chronos pipeline (BaseChronosPipeline) loaded with the official zero-shot model (chronos-t5-small).
    test_input = list of gluonts {start, target} dicts (from to_gluonts_univariate)
    horizon: prediction length (gluonts split offset)
    return: list of gluonts SampleForecast objects, one per series in test_input
    """
    torch.manual_seed(SEED)
    outputs = []
    for batch in batcher(test_input, batch_size=BATCH_SIZE):                       # Batch the test_input (list of dicts) into smaller lists for memory efficiency.
        context = [torch.tensor(e["target"]) for e in batch]
        q, _ = pipeline.predict_quantiles(                                         # Predict quantiles with the Chronos pipeline, which replicates the official method for zero-shot eval (gluonts SampleForecast from quantiles). The pipeline handles the autoregressive sampling internally, so we just need to pass the context and prediction_length (horizon) and it will return the quantiles for the forecast horizon.
            context, prediction_length=horizon, quantile_levels=QUANTILES,
            num_samples=NUM_SAMPLES)
        if isinstance(q, list):                                                    # Some Chronos pipelines return a list of quantiles, one per quantile level; others return a single tensor with a quantiles dimension. Handle both cases.
            q = np.stack(q).squeeze(axis=1)
        q = np.asarray(q.cpu() if torch.is_tensor(q) else q)
        outputs.append(q.swapaxes(-1, -2))  # [B, Q, H]
    outputs = np.concatenate(outputs)
    forecasts = []
    for item, ts in zip(outputs, test_input):                                      # Forecast sample and its corresponding start date.
        start = ts["start"] + len(ts["target"])
        forecasts.append(SampleForecast(samples=item, start_date=start))
    return forecasts


def main():
    import datasets as hfds

    pipe = BaseChronosPipeline.from_pretrained(MODEL_ID, device_map="cuda",        # Load the official zero-shot model (chronos-t5-small) from HuggingFace with the BaseChronosPipeline.
                                               torch_dtype=torch.bfloat16)
    rows = []
    for config, horizon in BENCHMARK_II:
        ds = hfds.load_dataset(HF_REPO, config, split="train")
        ds.set_format("numpy")
        gts = to_gluonts_univariate(ds, MAX_SERIES)
        _, test_template = split(gts, offset=-horizon)                             # Zero-shot eval: one held-out last-H window per series, exactly as the paper (gluonts split with offset=-horizon, num_windows=1).
        test_data = test_template.generate_instances(horizon, windows=1)

        # ---- time the forecast + record peak GPU memory (inference efficiency) ----
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()                                   # peak resets per dataset; baseline includes the resident model weights
        t0 = time.perf_counter()
        forecasts = chronos_forecasts(pipe, test_data.input, horizon)
        if torch.cuda.is_available():
            torch.cuda.synchronize()                                               # finish all GPU work before stopping the clock
        latency = time.perf_counter() - t0
        peak_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else float("nan")

        m = (evaluate_forecasts(                                                   # Evaluate_forecasts with gluonts MASE and MeanWeightedSumQuantileLoss. Seasonality is inferred from the data frequency by gluonts.
                forecasts, test_data=test_data,
                metrics=[MASE(), MeanWeightedSumQuantileLoss(QUANTILES)],
                batch_size=5000)
             .reset_index(drop=True).to_dict("records")[0])
        mase, wql = m["MASE[0.5]"], m["mean_weighted_sum_quantile_loss"]
        rows.append({"dataset": config, "model": MODEL_ID, "MASE": mase, "WQL": wql,
                     "n_series": len(gts), "latency_s": round(latency, 3),
                     "ms_per_series": round(latency / len(gts) * 1000, 2),
                     "peak_mem_mb": round(peak_mb, 1)})
        print(f"  {config:30s} MASE={mase:.4f} WQL={wql:.4f} n={len(gts)} "
              f"{latency:6.2f}s {peak_mb:6.0f}MB", flush=True)

    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "zeroshot_official_results.csv", index=False)              # Save our per-dataset results to CSV for record-keeping and comparison to the paper's reference numbers.

    # ---- compare to paper + aggregated relative score (gmean of model/seasonal-naive) ----
    ours = df.set_index("dataset")[["MASE", "WQL"]]
    base = pd.read_csv(REFERENCE_DIR / "seasonal-naive-zero-shot.csv").set_index("dataset")[["MASE", "WQL"]]
    paper = pd.read_csv(REFERENCE_DIR / "chronos-t5-small-zero-shot.csv").set_index("dataset")[["MASE", "WQL"]]
    common = ours.index.intersection(base.index).intersection(paper.index)

    agg_ours = (ours.loc[common] / base.loc[common]).apply(gmean)
    agg_paper = (paper.loc[common] / base.loc[common]).apply(gmean)

    cmp = ours.loc[common].join(paper.loc[common], lsuffix="_ours", rsuffix="_paper")
    cmp = cmp[["WQL_ours", "WQL_paper", "MASE_ours", "MASE_paper"]].reset_index()

    md = ["# Strict (official-method) Zero-shot Reproduction\n",
          f"chronos-t5-small via the official gluonts metric pipeline "
          f"(MASE + MeanWeightedSumQuantileLoss, gluonts split), cap={MAX_SERIES}/dataset, "
          f"on {len(common)} Benchmark II datasets.\n",
          "## Aggregated relative score (gmean of model / Seasonal-Naive, paper's method)\n",
          f"| metric | ours | paper (Chronos-T5 Small) |",
          "| --- | --- | --- |",
          f"| WQL  | {agg_ours['WQL']:.3f} | {agg_paper['WQL']:.3f} |",
          f"| MASE | {agg_ours['MASE']:.3f} | {agg_paper['MASE']:.3f} |",
          "\n## Per-dataset: ours vs paper (official Chronos-T5 Small)\n",
          "| dataset | WQL_ours | WQL_paper | MASE_ours | MASE_paper |",
          "| --- | --- | --- | --- | --- |"]
    for _, r in cmp.iterrows():
        md.append(f"| {r['dataset']} | {r['WQL_ours']:.4f} | {r['WQL_paper']:.4f} | "
                  f"{r['MASE_ours']:.4f} | {r['MASE_paper']:.4f} |")

    # ---- inference efficiency (latency + peak GPU memory) ----
    dev = "GPU (bf16)" if torch.cuda.is_available() else "CPU"
    md += [f"\n## Inference efficiency (ours, chronos-t5-small, {dev})\n",
           f"Total forecast wall-time {df['latency_s'].sum():.1f}s over {len(df)} datasets; "
           f"peak GPU memory {df['peak_mem_mb'].max():.0f} MB; "
           f"mean {df['ms_per_series'].mean():.1f} ms/series.\n",
           "| dataset | n_series | latency_s | ms/series | peak_mem_MB |",
           "| --- | --- | --- | --- | --- |"]
    for _, r in df.iterrows():
        md.append(f"| {r['dataset']} | {int(r['n_series'])} | {r['latency_s']:.2f} | "
                  f"{r['ms_per_series']:.1f} | {r['peak_mem_mb']:.0f} |")
    (RESULTS / "OFFICIAL_REPORT.md").write_text("\n".join(md), encoding="utf-8")

    print("\n=== Aggregated relative score (gmean model/seasonal-naive) ===")
    print(f"  WQL : ours={agg_ours['WQL']:.3f}  paper={agg_paper['WQL']:.3f}")
    print(f"  MASE: ours={agg_ours['MASE']:.3f}  paper={agg_paper['MASE']:.3f}")
    print(f"\nSaved -> {RESULTS / 'zeroshot_official_results.csv'} and OFFICIAL_REPORT.md")


if __name__ == "__main__":
    main()
