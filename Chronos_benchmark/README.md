# Chronos-T5 Benchmark II Reproduction (zero-shot + one-shot)

A faithful reproduction of the **Chronos paper's Benchmark II** (Ansari et al.,
2024, *Chronos: Learning the Language of Time Series*, TMLR), evaluating
`amazon/chronos-t5-small` on the paper's Benchmark II datasets under two scenarios
and comparing to the paper's official numbers:
- **zero-shot** — pretrained weights used as-is;
- **one-shot** — pretrained weights fine-tuned per dataset (lr 1e-3 → 0 over 1000
  steps), as in the paper's Section 5.5.2 / Figure 6. The fine-tuning is written
  out as an explicit PyTorch loop in `src/finetune_oneshot.py` (no `train.py`, no
  HF `Trainer`).

The evaluation deliberately reuses the **official Chronos evaluation method** so the
numbers are directly comparable to the paper:
- backtest windows via `gluonts.dataset.split` (no series-length filtering, no
  min-context cut),
- metrics via gluonts `MASE` and `MeanWeightedSumQuantileLoss` (seasonality inferred
  from each dataset's frequency by gluonts),
- the **aggregated relative score** (geometric mean of `model / Seasonal-Naive`),
  exactly as in the paper.

The only deviation from the paper is a deterministic per-dataset **series cap**
(`MAX_SERIES = 1000`) for laptop tractability — metric *definitions* are identical.
The stochastic 20-sample forecast is **seeded** (`SEED = 0`), so every re-run is
bit-identical.

## Datasets

25 of the paper's 27 Benchmark II datasets, loaded from
[`autogluon/chronos_datasets`](https://huggingface.co/datasets/autogluon/chronos_datasets)
(the two ETT datasets are absent from that repo). Per-dataset horizon = the paper's
Table 3 prediction length; one held-out last-H window (`num_rolls = 1`). The full
list lives in `BENCHMARK_II` in [`src/datasets_lib.py`](src/datasets_lib.py).

## Headline result

Aggregated relative score (gmean of `chronos-t5-small / Seasonal-Naive`), ours vs.
the paper, on the 25 Benchmark II datasets:

| scenario | WQL (ours) | WQL (paper) | MASE (ours) | MASE (paper) |
|---|---|---|---|---|
| zero-shot | 0.687 | 0.675 | 0.852 | 0.839 |
| one-shot  | 0.619 | 0.597 | 0.771 | 0.760 |

Fine-tuning improves the aggregate exactly as the paper reports (ΔMASE ≈ −0.08,
ΔWQL ≈ −0.07). Per-dataset tables: [`results/OFFICIAL_REPORT.md`](results/OFFICIAL_REPORT.md)
(zero-shot) and [`results/OFFICIAL_ONESHOT_REPORT.md`](results/OFFICIAL_ONESHOT_REPORT.md)
(one-shot). The residual gap comes from the 1000-series cap (the paper uses full
datasets) and the fixed seed. (Paper one-shot reference: Figure 6 aggregate only.)

## Layout

```
Chronos_benchmark/
├── README.md
├── environment.yml            # conda env (curated; torch installed separately)
├── requirements.txt           # pip deps (curated; torch installed separately)
├── src/
│   ├── config.py                 # central, portable path config (env-var overrides)
│   ├── datasets_lib.py           # Benchmark II dataset list + loading
│   ├── run_zeroshot_official.py  # zero-shot eval + aggregated relative score
│   ├── finetune_oneshot.py       # explicit one-shot fine-tuning (plain PyTorch loop)
│   ├── run_oneshot_official.py   # one-shot eval + aggregated relative score
│   └── make_notebook.py          # builds the deliverable notebook
├── results/
│   ├── zeroshot_official_results.csv · OFFICIAL_REPORT.md          # zero-shot
│   └── oneshot_official_results.csv  · OFFICIAL_ONESHOT_REPORT.md  # one-shot
├── notebooks/
│   └── Chronos_BenchmarkII_Reproduction.ipynb  # workflow + results + charts
├── reference/                 # paper's official numbers (for the comparison)
└── models/ft_oneshot/         # 25 per-dataset fine-tuned checkpoints (+ manifest)
```

## Setup on a new machine

Datasets download automatically on first run; you only need a Python environment.

**1. Create the environment** (Python 3.11):

```powershell
conda env create -f environment.yml
conda activate chronos_bench
# or: python -m venv .venv ; .\.venv\Scripts\Activate.ps1 ; pip install -r requirements.txt
```

**2. Install PyTorch for your GPU** (kept out of the env files so the CUDA build is
correct):

```powershell
# GPU, CUDA 12.8 (what this was measured on):
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
# CPU-only: pip install torch==2.9.1
```

## Reproduce

```powershell
conda activate chronos_bench
python src/config.py                  # confirm paths resolve on this machine

# zero-shot (~20-30 min on a 12 GB GPU)
python src/run_zeroshot_official.py

# one-shot: fine-tune 25 checkpoints then evaluate (~45-60 min total)
python src/finetune_oneshot.py        # writes models/ft_oneshot/ (resumable)
python src/run_oneshot_official.py
```

Zero-shot writes `results/zeroshot_official_results.csv` + `OFFICIAL_REPORT.md`;
one-shot writes `results/oneshot_official_results.csv` + `OFFICIAL_ONESHOT_REPORT.md`.

> If HuggingFace hub pings time out (datasets are already cached), set
> `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` before running.

> The ours-vs-paper comparison uses the paper's official reference numbers bundled
> in [`reference/`](reference/) (Seasonal-Naive + Chronos-T5 Small) — self-contained,
> no external repo or local paths needed.

## Portability / configuration

All paths resolve at runtime from the project root via `src/config.py`, overridable
by env vars: `CHRONOS_BENCH_ROOT`, `CHRONOS_BENCH_RESULTS`, `CHRONOS_BENCH_MODELS`.
Run `python src/config.py` to print the resolved layout.

## Benchmarking Chronos-2

Point `run_zeroshot_official.py` at `amazon/chronos-2` (via `BaseChronosPipeline`)
and run the same datasets through the identical gluonts pipeline; compare against the
per-dataset numbers and aggregated relative score in `OFFICIAL_REPORT.md`.
