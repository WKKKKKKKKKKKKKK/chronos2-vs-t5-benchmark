"""One-shot (LoRA fine-tuned) Chronos-2 evaluation -- mirrors run_zeroshot_chronos2.

The Chronos-2 counterpart of `Chronos_benchmark/src/run_oneshot_official.py`. Loads
each per-dataset LoRA adapter produced by `finetune_oneshot_chronos2.py`, merges it
into a fresh `amazon/chronos-2`, and evaluates on the SAME Benchmark II datasets with
the SAME official gluonts pipeline as the zero-shot study -- by importing
`run_zeroshot_chronos2` and reusing its loader / forecaster / metric code verbatim
(`to_gluonts_univariate`, `forecast_univariate`, `evaluate`). So the one-shot numbers
drop straight next to C2 zero-shot AND the Chronos-T5 one-shot.

Evaluated in UNIVARIATE mode (each series forecast independently) -- the apples-to-apples
counterpart of the T5 one-shot and of C2 zero-shot `univariate`. The adapter is merged
into the base weights (`merge_and_unload`), so inference is a plain Chronos2Model and
the zero-shot forecaster is reused unchanged.

The Chronos-2 paper reports no one-shot Benchmark II number; the one-shot reference we
compare against is the Chronos-T5 paper's one-shot aggregate (Figure 6): WQL 0.597,
MASE 0.760 (the head-to-head C2-vs-T5 one-shot itself lives in chronos2_t5/one-shot/).

Outputs:
  results/oneshot_chronos2_results.csv
  results/CHRONOS2_ONESHOT_REPORT.md
"""
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from gluonts.dataset.split import split
from scipy.stats import gmean

SRC = Path(__file__).resolve().parent.parent   # src root (config.py, datasets_lib.py)
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "zero_shot"))     # run_zeroshot_chronos2.py (shared harness)
from config import MODELS_DIR, REFERENCE_DIR, RESULTS_DIR as RESULTS  # noqa: E402
from datasets_lib import BENCHMARK_II, HF_REPO, MAX_SERIES  # noqa: E402
import run_zeroshot_chronos2 as R  # reuse the zero-shot harness (identical pipeline)  # noqa: E402

from chronos import BaseChronosPipeline  # noqa: E402
from peft import PeftModel  # noqa: E402

FT_ROOT = MODELS_DIR / "ft_oneshot"
# Chronos-T5 paper one-shot aggregate, Benchmark II (Fig. 6) -- the only one-shot reference.
PAPER_T5_ONESHOT_AGG = {"WQL": 0.597, "MASE": 0.760}


def load_oneshot_pipeline(adapter_dir: Path):
    """Fresh amazon/chronos-2 with the per-dataset LoRA adapter merged into its weights for the following forecast"""
    pipe = BaseChronosPipeline.from_pretrained(R.MODEL_ID, device_map="cuda", torch_dtype=R.DTYPE)
    peft_model = PeftModel.from_pretrained(pipe.model, str(adapter_dir))     # load LoRA adapter on top of base model (LoRA does not change the original weights directly. It adds small low-rank update weights to some layers.)
    pipe.model = peft_model.merge_and_unload()     # fold LoRA into base -> plain Chronos2Model
    return pipe


def main():
    import datasets as hfds

    rows = []
    for config, horizon in BENCHMARK_II:
        adapter = FT_ROOT / config
        if not (adapter / "adapter_config.json").exists():
            print(f"[skip] {config}: no LoRA adapter at {adapter}", flush=True)
            continue

        pipe = load_oneshot_pipeline(adapter)
        ds = hfds.load_dataset(HF_REPO, config, split="train")
        ds.set_format("numpy")
        gts = R.to_gluonts_univariate(ds, MAX_SERIES)
        _, test_template = split(gts, offset=-horizon)
        test_data = test_template.generate_instances(horizon, windows=1)
        test_input = list(test_data.input)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()   # after model load -> peak reflects inference
        t0 = time.perf_counter()
        forecasts = R.forecast_univariate(pipe, test_input, horizon)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.perf_counter() - t0
        peak_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else float("nan")

        mase, wql = R.evaluate(forecasts, test_data)
        rows.append({"dataset": config, "model": "chronos-2-oneshot-lora", "mode": "univariate",
                     "MASE": mase, "WQL": wql, "n_series": len(gts),
                     "latency_s": round(latency, 3),
                     "ms_per_series": round(latency / len(gts) * 1000, 2),
                     "peak_mem_mb": round(peak_mb, 1)})
        print(f"  {config:30s} MASE={mase:.4f} WQL={wql:.4f} n={len(gts)} "
              f"{latency:6.2f}s {peak_mb:6.0f}MB", flush=True)
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        sys.exit("No LoRA adapters found. Run finetune_oneshot_chronos2.py first.")

    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "oneshot_chronos2_results.csv", index=False)
    _write_report(df)
    print(f"\nSaved -> {RESULTS / 'oneshot_chronos2_results.csv'} and CHRONOS2_ONESHOT_REPORT.md")


def _write_report(df: pd.DataFrame):
    base = pd.read_csv(REFERENCE_DIR / "seasonal-naive-zero-shot.csv").set_index("dataset")[["MASE", "WQL"]]
    one = df.set_index("dataset")[["MASE", "WQL"]]
    agg_one, common = R._agg_rel_score(one, base)

    # C2 zero-shot (univariate) on the same datasets, if the zero-shot run exists.
    zs_lines = []
    zs_path = RESULTS / "zeroshot_chronos2_results.csv"
    if zs_path.exists():
        zs = pd.read_csv(zs_path)
        for mode in ("univariate", "cross_learning"):
            sub = zs[zs["mode"] == mode].set_index("dataset")[["MASE", "WQL"]]
            if sub.empty:
                continue
            agg_zs, _ = R._agg_rel_score(sub, base)
            zs_lines.append(f"| zero-shot {mode} (ours) | {agg_zs['WQL']:.3f} | {agg_zs['MASE']:.3f} |")

    md = [
        "# Chronos-2 one-shot (LoRA fine-tuned) on Chronos Benchmark II\n",
        f"`{R.MODEL_ID}` LoRA-fine-tuned per dataset (explicit PyTorch loop, lr 1e-3 -> 0 "
        f"over 1000 steps, LoRA r=8/alpha=16 on q/k/v/o + output layer), evaluated in "
        f"univariate mode via the official gluonts pipeline (MASE + "
        f"MeanWeightedSumQuantileLoss, gluonts split), cap={MAX_SERIES}/dataset, "
        f"quantile grid {R.QUANTILES}, on {len(common)} Benchmark II datasets.\n",
        "## Aggregated relative score (gmean of model / Seasonal-Naive)\n",
        "Lower is better. The one-shot reference is the Chronos-**T5** paper's one-shot "
        "aggregate (Fig. 6); the C2-vs-T5 one-shot head-to-head lives in "
        "`../../chronos2_t5/one-shot/`.\n",
        "| scenario | WQL | MASE |",
        "| --- | --- | --- |",
        f"| one-shot LoRA (ours) | {agg_one['WQL']:.3f} | {agg_one['MASE']:.3f} |",
        *zs_lines,
        f"| one-shot (Chronos-T5 paper, Fig. 6) | {PAPER_T5_ONESHOT_AGG['WQL']:.3f} | {PAPER_T5_ONESHOT_AGG['MASE']:.3f} |",
        "\n## Per-dataset (one-shot LoRA, ours)\n",
        "| dataset | MASE | WQL | n_series |",
        "| --- | --- | --- | --- |",
    ]
    for _, r in df.iterrows():
        md.append(f"| {r['dataset']} | {r['MASE']:.4f} | {r['WQL']:.4f} | {int(r['n_series'])} |")

    dev = f"GPU ({str(R.DTYPE).replace('torch.', '')})" if torch.cuda.is_available() else "CPU"
    md += [f"\n## Inference efficiency ({dev})\n",
           f"Total forecast wall-time {df['latency_s'].sum():.1f}s over {len(df)} datasets; "
           f"peak GPU memory {df['peak_mem_mb'].max():.0f} MB; "
           f"mean {df['ms_per_series'].mean():.1f} ms/series.\n",
           "| dataset | n_series | latency_s | ms/series | peak_mem_MB |",
           "| --- | --- | --- | --- | --- |"]
    for _, r in df.iterrows():
        md.append(f"| {r['dataset']} | {int(r['n_series'])} | {r['latency_s']:.2f} | "
                  f"{r['ms_per_series']:.1f} | {r['peak_mem_mb']:.0f} |")
    (RESULTS / "CHRONOS2_ONESHOT_REPORT.md").write_text("\n".join(md), encoding="utf-8")

    print("\n=== Aggregated relative score (one-shot LoRA, gmean model/seasonal-naive) ===")
    print(f"  WQL : ours={agg_one['WQL']:.3f}  T5-paper(Fig.6)={PAPER_T5_ONESHOT_AGG['WQL']:.3f}")
    print(f"  MASE: ours={agg_one['MASE']:.3f}  T5-paper(Fig.6)={PAPER_T5_ONESHOT_AGG['MASE']:.3f}")


if __name__ == "__main__":
    main()
