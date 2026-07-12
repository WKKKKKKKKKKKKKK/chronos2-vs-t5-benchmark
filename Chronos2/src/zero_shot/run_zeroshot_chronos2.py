"""Chronos-2 zero-shot evaluation on Chronos Benchmark II — paper-faithful.

This is the Chronos-2 counterpart of the sibling Chronos-T5 reproduction
(`Chronos_benchmark/src/run_zeroshot_official.py`). It deliberately reuses the
*identical* dataset registry, series cap, held-out windows and gluonts metric
pipeline (MASE + MeanWeightedSumQuantileLoss, seasonality inferred from each
series' frequency), so the numbers drop straight next to the Chronos-T5 baseline.

How Chronos-2 handles Benchmark II — exactly as in the technical report
(arXiv:2510.15821, §5.1). Benchmark II is a *univariate* benchmark (no multivariate
entities), so the report does NOT do shared-dynamics multivariate here; instead it
reports two settings, and we mirror both:

  * "univariate"     — every series is forecast independently. Each series is its
                       own group, so Chronos-2's group-attention layers are inert.
                       The apples-to-apples counterpart of Chronos-T5.

  * "cross_learning" — Chronos-2's in-context learning in *full cross-learning mode*
                       (the report's headline Benchmark II setting): every item in a
                       batch is assigned the SAME group id, so group attention shares
                       information across all series in the batch. Inputs stay 1-d
                       (univariate); only the grouping changes. Per the report the
                       group/batch size is ~100, which we use here
                       (CROSS_LEARNING_BATCH). Because the group is the batch, results
                       depend on batch size — this is intended and matches the report.

Both modes score every series against the same held-out window, so they are directly
comparable to each other and to Chronos-T5.

Key differences from the Chronos-T5 script (and *why*):
  * Chronos-2 is a QUANTILE forecaster, not a sampler — there is no 20-sample
    Monte-Carlo step and no `torch.manual_seed`. We build a gluonts
    `QuantileForecast` (not `SampleForecast`) directly from the predicted
    quantiles. Inference is therefore exactly reproducible by construction.
  * We request the SAME 9 quantile levels Chronos-T5 used (0.1 … 0.9) so WQL is
    computed over an identical quantile grid (Chronos-2 natively predicts 21).

Note: genuine shared-dynamics *multivariate* (stacking aligned variates into a
(n_variates, length) item) is how the report evaluates fev-bench / GIFT-Eval, NOT
Benchmark II, so it is intentionally not used here.

Outputs:
  results/zeroshot_chronos2_results.csv   -- per (dataset, mode): MASE/WQL + timing
  results/CHRONOS2_REPORT.md              -- univariate vs cross-learning vs T5(paper)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gluonts.dataset.split import split
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.model.forecast import QuantileForecast
from scipy.stats import gmean

SRC = Path(__file__).resolve().parent.parent   # src root (config.py, datasets_lib.py live here)
sys.path.insert(0, str(SRC))
from config import REFERENCE_DIR, RESULTS_DIR as RESULTS  # noqa: E402
from datasets_lib import BENCHMARK_II, HF_REPO, MAX_SERIES  # noqa: E402

from chronos import BaseChronosPipeline  # noqa: E402

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QKEYS = [str(q) for q in QUANTILES]
MODEL_ID = "amazon/chronos-2"
# Inference dtype. We use bfloat16 to MATCH the Chronos-T5 benchmark (which runs bf16),
# so the head-to-head accuracy AND efficiency comparison is same-precision. bf16 changes
# Chronos-2's metrics only marginally vs fp32 (<=~2% per dataset, often negligible) and
# stays deterministic run-to-run on a fixed GPU. Set to torch.float32 for full precision.
DTYPE = torch.bfloat16
# Univariate batch size: counts every series fed to the model in one forward pass.
BATCH_SIZE = 256
# Full cross-learning group/batch size. The Chronos-2 technical report uses ~100
# (each batch is one cross-learning group); larger batches deviate from the group
# sizes seen during pretraining. This is the single knob the report calls out.
CROSS_LEARNING_BATCH = 100
MODES = ("univariate", "cross_learning")


def to_gluonts_univariate(ds, cap: int):
    """HF dataset -> list of gluonts {start, target}, capped to `cap` series.

    Verbatim port of `run_zeroshot_official.to_gluonts_univariate` from the
    Chronos-T5 benchmark, so the series set / order / cap are byte-identical.
    """
    import datasets as hfds

    seq_fields = [c for c in ds.features if isinstance(ds.features[c], hfds.Sequence)]     # Select the sequence fields (the time series columns)
    if "timestamp" in seq_fields:
        seq_fields.remove("timestamp")                                                     # Remove the timestamp field from the sequence fields, as it is not a target series
    freq = pd.DatetimeIndex(ds[0]["timestamp"]).to_period().freqstr                        # Infer the frequency of the time series from the first row's timestamp
    n = len(ds)
    keep = range(n) if n <= cap else set(np.linspace(0, n - 1, cap).astype(int).tolist())  # Keep all series if n <= cap, otherwise select `cap` evenly spaced indices from the dataset
    out = []
    for i, row in enumerate(ds):
        if i not in keep:
            continue
        for f in seq_fields:
            out.append({"start": pd.Period(row["timestamp"][0], freq=freq), # The start period of the time series, using the first timestamp and the inferred frequency
                        "target": np.asarray(row[f], dtype=np.float32)})    # The target series as a numpy array of float32
    return out


def _quantile_forecast(q_hwq: np.ndarray, start) -> QuantileForecast:
    """Build a gluonts QuantileForecast from a (horizon, n_quantiles) array.

    `start` is the Period of the first predicted step. We transpose to the
    gluonts (n_quantiles, horizon) layout and key each row by its quantile level.
    """
    return QuantileForecast(forecast_arrays=np.ascontiguousarray(q_hwq.T),  # (Q, H) (n_quantiles, horizon)
                            start_date=start, forecast_keys=QKEYS)          # start: The first forecasted timestamp
                                                                            # forecast_keys: The quantile levels corresponding to the rows of forecast_arrays
                                                                            

def forecast_univariate(pipeline, test_input, horizon):
    """One independent forecast per series (each series is its own group)."""
    contexts = [np.asarray(e["target"], dtype=np.float32) for e in test_input]
    quantiles, _ = pipeline.predict_quantiles(                              # quantiles: (n_series, n_variates=1, horizon, n_quantiles)
        contexts, prediction_length=horizon, quantile_levels=QUANTILES,
        batch_size=BATCH_SIZE, limit_prediction_length=False)
    forecasts = []
    for q, e in zip(quantiles, test_input):
        # q: (n_variates=1, horizon, n_quantiles)
        arr = np.asarray(q[0].cpu() if torch.is_tensor(q) else q[0])
        start = e["start"] + len(e["target"])                               # The start period of the forecast is the end of the input series
        forecasts.append(_quantile_forecast(arr, start))                    # map time to quantiles
    return forecasts


def forecast_cross_learning(pipeline, test_input, horizon, batch=CROSS_LEARNING_BATCH):
    """Chronos-2 full cross-learning (the report's Benchmark II setting).

    Inputs stay 1-d (univariate), exactly as in the `univariate` mode; the only
    change is `cross_learning=True`, which makes every item in a forward batch share
    one group id so group attention shares information across the whole batch. With
    `batch_size=batch`, each batch is one cross-learning group of up to `batch`
    series (the report uses ~100). Series in different batches do not mix, so the
    grouping is, by design, the batch — hence batch-size dependent.

    Returns (forecasts_in_original_order, n_series, n_groups), where n_groups is the
    number of cross-learning groups (= number of batches).
    """
    contexts = [np.asarray(e["target"], dtype=np.float32) for e in test_input]
    quantiles, _ = pipeline.predict_quantiles(                                 # (horizon, n_quantiles)
        contexts, prediction_length=horizon, quantile_levels=QUANTILES,
        batch_size=batch, cross_learning=True, limit_prediction_length=False)  # cross_learning=True: every item in a batch shares one group id, so group attention shares information across the whole batch
    forecasts = []
    for q, e in zip(quantiles, test_input):
        arr = np.asarray(q[0].cpu() if torch.is_tensor(q) else q[0])  # (horizon, n_quantiles)
        forecasts.append(_quantile_forecast(arr, e["start"] + len(e["target"])))
    n_groups = (len(test_input) + batch - 1) // batch  # ceil: 1-d items, sequential batches
    return forecasts, len(test_input), n_groups


def evaluate(forecasts, test_data):
    m = (evaluate_forecasts(
            forecasts, test_data=test_data,
            metrics=[MASE(), MeanWeightedSumQuantileLoss(QUANTILES)],
            batch_size=5000)
         .reset_index(drop=True).to_dict("records")[0])
    return m["MASE[0.5]"], m["mean_weighted_sum_quantile_loss"]


def main():
    import datasets as hfds                                                                     # HuggingFace datasets

    pipe = BaseChronosPipeline.from_pretrained(MODEL_ID, device_map="cuda", torch_dtype=DTYPE)  # Chronos-2 pipeline
    cuda = torch.cuda.is_available()
    rows = []
    for config, horizon in BENCHMARK_II:
        ds = hfds.load_dataset(HF_REPO, config, split="train")                                  
        ds.set_format("numpy")
        gts = to_gluonts_univariate(ds, MAX_SERIES)                                             # Convert to GluonTS format
        _, test_template = split(gts, offset=-horizon)
        test_data = test_template.generate_instances(horizon, windows=1)
        test_input = list(test_data.input)

        for mode in MODES:
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            if mode == "univariate":
                forecasts = forecast_univariate(pipe, test_input, horizon)
                n_grouped, n_groups = 0, len(test_input)
            else:
                forecasts, n_grouped, n_groups = forecast_cross_learning(pipe, test_input, horizon)
            if cuda:
                torch.cuda.synchronize()
            latency = time.perf_counter() - t0
            peak_mb = torch.cuda.max_memory_allocated() / 1e6 if cuda else float("nan")

            mase, wql = evaluate(forecasts, test_data)
            rows.append({"dataset": config, "model": MODEL_ID, "mode": mode,
                         "MASE": mase, "WQL": wql, "n_series": len(gts),
                         "n_grouped": n_grouped, "n_groups": n_groups,
                         "latency_s": round(latency, 3),
                         "ms_per_series": round(latency / len(gts) * 1000, 2),
                         "peak_mem_mb": round(peak_mb, 1)})
            print(f"  {config:30s} {mode:12s} MASE={mase:.4f} WQL={wql:.4f} "
                  f"n={len(gts)} grouped={n_grouped} {latency:6.2f}s {peak_mb:6.0f}MB",
                  flush=True)

    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "zeroshot_chronos2_results.csv", index=False)
    _write_report(df, cuda)
    print(f"\nSaved -> {RESULTS / 'zeroshot_chronos2_results.csv'} and CHRONOS2_REPORT.md")


def _agg_rel_score(per_ds: pd.DataFrame, base: pd.DataFrame):
    """Aggregated relative score = gmean(model / Seasonal-Naive), paper's method.
       Combines all datasets with a geometric mean, ignoring datasets missing from either side."""
    common = per_ds.index.intersection(base.index)
    return (per_ds.loc[common] / base.loc[common]).apply(gmean), common


def _write_report(df: pd.DataFrame, cuda: bool):
    base = pd.read_csv(REFERENCE_DIR / "seasonal-naive-zero-shot.csv").set_index("dataset")[["MASE", "WQL"]]

    uni = df[df["mode"] == "univariate"].set_index("dataset")[["MASE", "WQL"]]
    cl = df[df["mode"] == "cross_learning"].set_index("dataset")[["MASE", "WQL"]]

    agg_uni, _ = _agg_rel_score(uni, base)
    agg_cl, _ = _agg_rel_score(cl, base)
    # Chronos-2 paper Benchmark II (arXiv:2510.15821, Table 5 skill scores) -> G = 1 - skill/100
    c2_paper = {"WQL": 1 - 0.466, "MASE": 1 - 0.265}

    md = [
        "# Chronos-2 zero-shot on Chronos Benchmark II — univariate vs cross-learning\n",
        f"`{MODEL_ID}` via the official gluonts metric pipeline (MASE + "
        f"MeanWeightedSumQuantileLoss, gluonts split), cap={MAX_SERIES}/dataset, "
        f"quantile grid {QUANTILES}, on {len(uni)} Benchmark II datasets. Cross-learning "
        f"is Chronos-2's full cross-learning mode (technical report §5.1): 1-d inputs, "
        f"every item in a batch shares one group id, group/batch size "
        f"{CROSS_LEARNING_BATCH}.\n",
        "## Aggregated relative score (gmean of model / Seasonal-Naive) — vs the Chronos-2 paper\n",
        "Lower is better. `Chronos-2 (paper)` = the report's Benchmark II skill scores "
        "(arXiv:2510.15821 Table 5) converted via G = 1 - skill/100; it aggregates all 27 "
        "datasets + full data (ours: 25 + cap=1000 + bf16). The Chronos-T5 head-to-head is "
        "in the sibling `chronos2_t5/zero-shot/` project.\n",
        "| metric | C2 univariate | C2 cross-learning | Chronos-2 (paper) |",
        "| --- | --- | --- | --- |",
        f"| WQL  | {agg_uni['WQL']:.3f} | {agg_cl['WQL']:.3f} | {c2_paper['WQL']:.3f} |",
        f"| MASE | {agg_uni['MASE']:.3f} | {agg_cl['MASE']:.3f} | {c2_paper['MASE']:.3f} |",
        "\n## Per-dataset MASE / WQL (univariate vs cross-learning)\n",
        "| dataset | MASE uni | MASE xl | WQL uni | WQL xl |",
        "| --- | --- | --- | --- | --- |",
    ]
    for d in uni.index:
        def cell(frame, col):
            return f"{frame.loc[d, col]:.4f}" if d in frame.index else "—"
        md.append(f"| {d} | {cell(uni,'MASE')} | {cell(cl,'MASE')} "
                  f"| {cell(uni,'WQL')} | {cell(cl,'WQL')} |")

    # cross-learning grouping = batches of CROSS_LEARNING_BATCH series
    clinfo = df[df["mode"] == "cross_learning"].set_index("dataset")[["n_series", "n_groups"]]
    md += [f"\n## Cross-learning grouping (each batch of ~{CROSS_LEARNING_BATCH} series = one group)\n",
           "All series participate; `n_groups` = number of cross-learning groups (= batches).\n",
           "| dataset | n_series | n_groups |",
           "| --- | --- | --- |"]
    for d in clinfo.index:
        r = clinfo.loc[d]
        md.append(f"| {d} | {int(r['n_series'])} | {int(r['n_groups'])} |")

    dev = f"GPU ({str(DTYPE).replace('torch.', '')})" if cuda else "CPU"
    md += [f"\n## Inference efficiency ({dev})\n",
           "| mode | total latency_s | mean ms/series | peak_mem_MB |",
           "| --- | --- | --- | --- |"]
    for mode in MODES:
        sub = df[df["mode"] == mode]
        md.append(f"| {mode} | {sub['latency_s'].sum():.1f} | "
                  f"{sub['ms_per_series'].mean():.1f} | {sub['peak_mem_mb'].max():.0f} |")

    (RESULTS / "CHRONOS2_REPORT.md").write_text("\n".join(md), encoding="utf-8")

    print("\n=== Aggregated relative score (gmean model/seasonal-naive) ===")
    print(f"  WQL : uni={agg_uni['WQL']:.3f}  xl={agg_cl['WQL']:.3f}  C2-paper={c2_paper['WQL']:.3f}")
    print(f"  MASE: uni={agg_uni['MASE']:.3f}  xl={agg_cl['MASE']:.3f}  C2-paper={c2_paper['MASE']:.3f}")


if __name__ == "__main__":
    main()