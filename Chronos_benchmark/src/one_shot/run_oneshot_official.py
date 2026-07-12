"""One-shot evaluation via the official gluonts method (mirrors run_zeroshot_official).

Loads each per-dataset fine-tuned checkpoint produced by finetune_oneshot.py and
evaluates it on the SAME Benchmark II datasets with the SAME official gluonts
pipeline (split + MASE + MeanWeightedSumQuantileLoss, cap=MAX_SERIES), so the
one-shot numbers are directly comparable to the zero-shot ones and to the paper.

The paper's one-shot reference is only an aggregate (Figure 6, Benchmark II):
fine-tuned Chronos-T5 Small reaches WQL 0.597 and MASE 0.760 (down from zero-shot
0.667 / 0.841). There is no per-dataset one-shot reference CSV in the repo, so we
report the aggregated relative score and compare to those two numbers.

Outputs:
  results/oneshot_official_results.csv
  results/OFFICIAL_ONESHOT_REPORT.md
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from gluonts.dataset.split import split
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
from gluonts.model.evaluation import evaluate_forecasts
from scipy.stats import gmean

SRC = Path(__file__).resolve().parent.parent   # src root (config.py, datasets_lib.py)
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "zero_shot"))     # run_zeroshot_official.py (shared harness)
from config import MODELS_DIR, RESULTS_DIR as RESULTS  # noqa: E402
from datasets_lib import BENCHMARK_II, HF_REPO, MAX_SERIES  # noqa: E402
import run_zeroshot_official as R  # reuse to_gluonts_univariate / chronos_forecasts  # noqa: E402

from chronos import BaseChronosPipeline  # noqa: E402
from transformers import AutoModelForSeq2SeqLM  # noqa: E402
from peft import PeftModel  # noqa: E402

MODEL_ID = "amazon/chronos-t5-small"
FT_FULL = MODELS_DIR / "ft_oneshot"          # full fine-tuning (paper) -- preserved
FT_LORA = MODELS_DIR / "ft_oneshot_lora"     # LoRA one-shot (head-to-head)
MERGED_TMP = MODELS_DIR / "_merged_tmp"      # transient: LoRA merged into base for eval
PAPER_ONESHOT_AGG = {"WQL": 0.597, "MASE": 0.760}   # paper Figure 6, Benchmark II


def load_pipe(ckpt_dir, use_lora):
    """Fine-tuned Chronos-T5 pipeline. Full: load the checkpoint dir directly. LoRA: merge the
    adapter into a fresh base, save to a temp dir, load that (reuses the full-checkpoint path;
    the base config carries chronos_config, so merged inference == a full checkpoint)."""
    if not use_lora:
        return BaseChronosPipeline.from_pretrained(str(ckpt_dir), device_map="cuda", torch_dtype=torch.bfloat16)
    base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)                      # fresh base model (no LoRA)
    merged = PeftModel.from_pretrained(base, str(ckpt_dir)).merge_and_unload()  # fold LoRA into base -> plain ChronosT5Model
    shutil.rmtree(MERGED_TMP, ignore_errors=True)
    merged.save_pretrained(MERGED_TMP)                                          # save the merged model to a temp dir (reuses the full-checkpoint path; the base config carries chronos_config, so merged inference == a full checkpoint)
    return BaseChronosPipeline.from_pretrained(str(MERGED_TMP), device_map="cuda", torch_dtype=torch.bfloat16)


def main():
    import datasets as hfds
    ap = argparse.ArgumentParser(description="One-shot eval of chronos-t5-small (full FT or LoRA).")
    ap.add_argument("--lora", action="store_true", help="eval LoRA adapters (ft_oneshot_lora/); else full FT (ft_oneshot/)")
    ap.add_argument("--only", default=None, help="only datasets whose name contains this substring")
    args = ap.parse_args()

    ft_root = FT_LORA if args.lora else FT_FULL
    marker = "adapter_config.json" if args.lora else "model.safetensors"
    label = "chronos-t5-small-oneshot-lora" if args.lora else "chronos-t5-small-oneshot"
    todo = [(c, h) for c, h in BENCHMARK_II if args.only is None or args.only in c]

    rows = []
    for config, horizon in todo:                                           # loop over the Benchmark II datasets
        ckpt = ft_root / config                                            # checkpoint dir for this dataset (full FT or LoRA)
        if not (ckpt / marker).exists():                                   # skip if the expected checkpoint file does not exist
            print(f"[skip] {config}: no checkpoint at {ckpt}", flush=True)
            continue
        pipe = load_pipe(ckpt, args.lora)                                  # load the fine-tuned pipeline (full FT or LoRA) 
        ds = hfds.load_dataset(HF_REPO, config, split="train")
        ds.set_format("numpy")
        gts = R.to_gluonts_univariate(ds, MAX_SERIES)                      # reuse zero-shot template                                    # Same as run_zeroshot_official.py
        _, test_template = split(gts, offset=-horizon)                     # official gluonts split: last `horizon` points for test, rest for train
        test_data = test_template.generate_instances(horizon, windows=1)

        # ---- time the forecast + record peak GPU memory (inference efficiency) ----
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()                                   # reset after the model is loaded -> peak reflects inference
        t0 = time.perf_counter()
        forecasts = R.chronos_forecasts(pipe, test_data.input, horizon)
        if torch.cuda.is_available():
            torch.cuda.synchronize()                                               # wait for all kernels to finish before measuring time
        latency = time.perf_counter() - t0
        peak_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else float("nan")

        m = (evaluate_forecasts(
                forecasts, test_data=test_data,
                metrics=[MASE(), MeanWeightedSumQuantileLoss(R.QUANTILES)],
                batch_size=5000)
             .reset_index(drop=True).to_dict("records")[0])
        mase, wql = m["MASE[0.5]"], m["mean_weighted_sum_quantile_loss"]
        rows.append({"dataset": config, "model": label,
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
        sys.exit("No checkpoints found. Run finetune_oneshot.py first.")

    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    csv_name = "oneshot_lora_results.csv" if args.lora else "oneshot_official_results.csv"
    report_name = "ONESHOT_LORA_REPORT.md" if args.lora else "OFFICIAL_ONESHOT_REPORT.md"
    df.to_csv(RESULTS / csv_name, index=False)

    # ---- aggregated relative score vs Seasonal-Naive, and vs zero-shot + paper ----
    ours = df.set_index("dataset")[["MASE", "WQL"]]
    base = pd.read_csv(R.REFERENCE_DIR / "seasonal-naive-zero-shot.csv").set_index("dataset")[["MASE", "WQL"]]
    common = ours.index.intersection(base.index)
    agg_one = (ours.loc[common] / base.loc[common]).apply(gmean)

    # zero-shot agg on the same datasets, if available
    zs_line = ""
    zs_path = RESULTS / "zeroshot_official_results.csv"
    if zs_path.exists():
        zs = pd.read_csv(zs_path).set_index("dataset")[["MASE", "WQL"]]
        c2 = zs.index.intersection(base.index)
        agg_zs = (zs.loc[c2] / base.loc[c2]).apply(gmean)
        zs_line = f"| zero-shot (ours) | {agg_zs['WQL']:.3f} | {agg_zs['MASE']:.3f} |\n"

    md = [
        "# One-shot (fine-tuned) Zero-shot-benchmark Reproduction\n",
        f"chronos-t5-small fine-tuned per dataset (lr 1e-3 -> 0 over 1000 steps, "
        f"explicit PyTorch loop), evaluated via the official gluonts pipeline, "
        f"cap={MAX_SERIES}, on {len(common)} Benchmark II datasets.\n",
        "## Aggregated relative score (gmean of model / Seasonal-Naive)\n",
        "| scenario | WQL | MASE |",
        "| --- | --- | --- |",
        f"| one-shot (ours) | {agg_one['WQL']:.3f} | {agg_one['MASE']:.3f} |",
        zs_line +
        f"| one-shot (paper, Fig. 6) | {PAPER_ONESHOT_AGG['WQL']:.3f} | {PAPER_ONESHOT_AGG['MASE']:.3f} |",
        "\n## Per-dataset (one-shot, ours)\n",
        "| dataset | MASE | WQL | n_series |",
        "| --- | --- | --- | --- |",
    ]
    for _, r in df.iterrows():
        md.append(f"| {r['dataset']} | {r['MASE']:.4f} | {r['WQL']:.4f} | {int(r['n_series'])} |")

    # ---- inference efficiency (latency + peak GPU memory) ----
    dev = "GPU (bf16)" if torch.cuda.is_available() else "CPU"
    md += [f"\n## Inference efficiency (ours, fine-tuned chronos-t5-small, {dev})\n",
           f"Total forecast wall-time {df['latency_s'].sum():.1f}s over {len(df)} datasets; "
           f"peak GPU memory {df['peak_mem_mb'].max():.0f} MB; "
           f"mean {df['ms_per_series'].mean():.1f} ms/series.\n",
           "| dataset | n_series | latency_s | ms/series | peak_mem_MB |",
           "| --- | --- | --- | --- | --- |"]
    for _, r in df.iterrows():
        md.append(f"| {r['dataset']} | {int(r['n_series'])} | {r['latency_s']:.2f} | "
                  f"{r['ms_per_series']:.1f} | {r['peak_mem_mb']:.0f} |")
    (RESULTS / report_name).write_text("\n".join(md), encoding="utf-8")
    shutil.rmtree(MERGED_TMP, ignore_errors=True)   # drop the transient merged-LoRA checkpoint

    print("\n=== Aggregated relative score (one-shot, gmean model/seasonal-naive) ===")
    print(f"  WQL : ours={agg_one['WQL']:.3f}  paper(Fig.6)={PAPER_ONESHOT_AGG['WQL']:.3f}")
    print(f"  MASE: ours={agg_one['MASE']:.3f}  paper(Fig.6)={PAPER_ONESHOT_AGG['MASE']:.3f}")
    print(f"\nSaved -> {RESULTS / csv_name} and {report_name}")


if __name__ == "__main__":
    main()
