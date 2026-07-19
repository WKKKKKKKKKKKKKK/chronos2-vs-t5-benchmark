# Chronos-2 vs Chronos-T5 — A Rigorous Zero-shot & One-shot Benchmark

An **apples-to-apples** evaluation of two time-series foundation models — the incumbent
**Chronos-T5** and the newer **Chronos-2** — on the **Chronos Benchmark II** suite
(25 held-out datasets). Every number is produced through one shared GluonTS pipeline, so
the measured gap reflects real model quality, not a benchmarking artifact.

> **TL;DR** — Zero-shot, Chronos-2 is **~18 % lower WQL** than Chronos-T5, wins **23/25**
> datasets, and is **~7–12× faster**. Once *both* models are LoRA fine-tuned, they **tie**.
> So Chronos-2's edge is out-of-the-box (zero-shot), and its value is *not needing* fine-tuning.

---

## Why this project

The practical question: *should we replace Chronos-T5 with Chronos-2 as the default
forecaster — and by how much?* To answer it defensibly, the study fixes every variable
(same 25 datasets, same metrics, same Seasonal-Naive baseline, same split) and tests **both
regimes** — zero-shot and one-shot (fine-tuned) — for **both models**.

## Repository structure

| Folder | Role | Runs models? |
|---|---|---|
| [`Chronos_benchmark/`](Chronos_benchmark/) | **Chronos-T5** side — reproduce & measure the incumbent: zero-shot + one-shot (LoRA) | ✅ train + infer |
| [`Chronos2/`](Chronos2/) | **Chronos-2** side — zero-shot (univariate + cross-learning) + one-shot (LoRA) | ✅ train + infer |
| [`chronos2_t5/`](chronos2_t5/) | **Comparison & analysis layer** — fair head-to-head, leaderboard, pairwise dominance, cross-learning, robustness. Mostly *aggregates* the two engines' CSVs (the `edge-case/` sub-study is the one exception that runs inference itself) | mostly aggregates |

Each folder has its own detailed `README.md`. The upstream library
([`amazon-science/chronos-forecasting`](https://github.com/amazon-science/chronos-forecasting))
is **not vendored here** — install it via pip (see Setup).

### Three design principles (read these first — they explain everything)

1. **Single source of truth.** Data loading + metric computation live in exactly one place —
   each project's `run_zeroshot_*.py`. Zero-shot, one-shot and cross-learning all call it, so
   the numbers are directly comparable.
2. **Byte-identical datasets.** Both projects share the same `datasets_lib.py` registry
   (same 25 `(dataset, horizon)` pairs, `MAX_SERIES = 1000`), so C2 and T5 see the exact same
   series and windows.
3. **One fixed denominator.** All relative scores use the paper's official
   `Seasonal-Naive` numbers (`Chronos_benchmark/reference/seasonal-naive-zero-shot.csv`) as
   the denominator, keeping every score comparable to the published leaderboard.

---

## Key results (honest, by regime)

**Zero-shot (clean data) — Chronos-2 wins clearly**
- WQL **−18 %** vs T5 (C2 cross-learning) / −14 % (C2 univariate); wins **23/25** datasets.
- **~7–12× faster** inference; cross-learning mode is also the most memory-light (~2.2 GB).
- Cross-learning adds **~4 % WQL** over univariate C2 (helps 19/25 datasets) — an in-context
  lever T5 lacks (its real strength is multivariate/covariate tasks, out of scope here).

**One-shot (LoRA fine-tuned) — a statistical tie**
- C2 and T5 are within noise (Wilcoxon p > 0.1). Fine-tuning does not change the winner —
  so there is no need to fine-tune C2.

**Robustness to corrupted input (zero-shot)**
- Under injected sensor faults, C2 stays **more accurate on 4/6 fault types** (biggest on
  missing data), but the aggregate edge is marginal. It is **not "more robust"** in the
  relative-degradation sense — T5 degrades less only because its tokenizer clamps outliers
  (an artifact). Cross-learning gives **no** robustness benefit; use plain C2-univariate for
  noisy/faulty sensors.

---

## Setup

Python 3.11. Install PyTorch for your CUDA version first, then the pinned deps.

```bash
conda create -n chronos_bench python=3.11 -y && conda activate chronos_bench
# PyTorch (match your CUDA), e.g. CUDA 12.8:
pip install torch --index-url https://download.pytorch.org/whl/cu128
# core deps (each sub-project also ships requirements.txt / environment.yml)
pip install chronos-forecasting gluonts datasets peft scipy pandas matplotlib
```

- `Chronos_benchmark/` needs `chronos-forecasting` (Chronos-T5).
- `Chronos2/` needs `chronos-forecasting >= 2.x` (provides `amazon/chronos-2`).

## Reproduce

Run from each sub-project root. Heavy artifacts (`models/`, `runs/`) are **not** committed —
re-generate them, or just read the committed `results/*.csv` and `*_REPORT.md`.

```bash
# 1) Chronos-T5 engine (incumbent) — see Chronos_benchmark/README.md
python Chronos_benchmark/src/zero_shot/run_zeroshot_official.py       # zero-shot
python Chronos_benchmark/src/one_shot/finetune_oneshot.py            # one-shot (paper reproduction, full FT)
python Chronos_benchmark/src/one_shot/run_oneshot_official.py

# 2) Chronos-2 engine (new) — see Chronos2/README.md
python Chronos2/src/zero_shot/run_zeroshot_chronos2.py               # zero-shot (univariate + cross-learning)
python Chronos2/src/one_shot/finetune_oneshot_chronos2.py           # one-shot (LoRA)
python Chronos2/src/one_shot/run_oneshot_chronos2.py

# 3) Fair comparison — aggregates the engines' CSVs (run from each folder; see chronos2_t5/README.md)
#    3a. one-shot: LoRA-tune BOTH models via the shared HPO pipeline, then the head-to-head.
#        (produces the oneshot_hpo_*.csv that the leaderboard + head-to-head + summary all read)
cd chronos2_t5/one-shot
python scripts/hpo.py --model c2
python scripts/hpo.py --model t5
python scripts/final_run.py --model c2
python scripts/final_run.py --model t5
python scripts/phase6_cltrain.py         # C2 train-time cross-learning (leaderboard/summary input)
python scripts/head_to_head.py           # -> reports/HEAD_TO_HEAD_REPORT.md
python scripts/summary_matrix.py
#    3b. zero-shot: dashboard + leaderboard (no GPU; reads the CSVs from steps 1-3a)
cd ../zero-shot
python scripts/compare_zeroshot.py       # dashboard + head-to-head report
python scripts/benchmark_table.py        # leaderboard (win rate, skill, runtime)

# 4) The deliverable notebook — the whole comparison, CPU-only, from the committed CSVs
cd ..
python make_comparison_notebook.py --execute
```

---

## Learning path (roadmap for newcomers)

Fastest way to understand the whole thing:

1. **Concepts** → [`Chronos2/docs/CHRONOS2_VS_T5.md`](Chronos2/docs/CHRONOS2_VS_T5.md) — the two
   architectures (T5 quantize→tokens→sample vs C2 continuous patches→direct quantile head) and
   the group-attention / cross-learning mechanism.
2. **The core harness** → [`Chronos2/src/zero_shot/run_zeroshot_chronos2.py`](Chronos2/src/zero_shot/run_zeroshot_chronos2.py) —
   read `to_gluonts_univariate`, `forecast_univariate` vs `forecast_cross_learning`, `evaluate`,
   `_agg_rel_score`. Contrast with the T5 harness `Chronos_benchmark/src/zero_shot/run_zeroshot_official.py`
   (T5 needs 20-sample decode; C2 does not). Run `Chronos2/src/smoke_test.py`.
3. **One-shot (LoRA)** → `*/one_shot/finetune_oneshot*.py` then `run_oneshot*.py` (they reuse the
   harness above so the numbers stay comparable).
4. **Comparison & analysis** → the one-notebook overview
   [`chronos2_t5/Chronos2_vs_ChronosT5_HeadToHead.ipynb`](chronos2_t5/Chronos2_vs_ChronosT5_HeadToHead.ipynb)
   (runs on CPU from the committed CSVs), then the `chronos2_t5/` scripts + `*_REPORT.md` files.
5. **Robustness** → `chronos2_t5/zero-shot/edge-case/` (the one self-running sub-study).

## Metrics (quick reference)

- **WQL** (primary, probabilistic): weighted quantile / pinball loss over the 9-quantile grid
  0.1–0.9, normalized by `Σ|y|`. Lower is better.
- **MASE** (point): mean abs error ÷ in-sample Seasonal-Naive error. Scale-free; `<1` beats
  Seasonal-Naive.
- **Skill** = `100·(1 − G)`, `G = geomean(model / Seasonal-Naive)`. Higher is better;
  Seasonal-Naive = 0.
- **Win rate** = share of pairwise (dataset, opponent) comparisons a model wins.

## Dataset — Chronos Benchmark II

25 held-out (unseen-in-pretraining) univariate datasets, ~13k series (capped 1000/set),
horizons 4–56, 6 frequencies, across 8 sectors (energy, retail, finance, transport, weather,
tourism, health, banking). Loaded from `autogluon/chronos_datasets`.

## Notes & caveats

- `models/` (LoRA checkpoints, up to several GB) and `runs/` (TensorBoard) are git-ignored;
  the committed `results/*.csv` + reports are enough to inspect every finding.
- Only deviation from the paper protocol: `MAX_SERIES = 1000` cap + bf16 precision (explained
  in each project's reproduction-check).
- Validated against **both** original papers before drawing conclusions — see the
  reproduction-check reports.

## References

- Ansari et al., *Chronos: Learning the Language of Time Series*, TMLR 2024.
- Ansari et al., *Chronos-2* technical report, 2025 (arXiv:2510.15821).
- Upstream: https://github.com/amazon-science/chronos-forecasting

## License

Released under the **Apache License 2.0** — see [`LICENSE`](LICENSE). This matches the
upstream `chronos-forecasting` license.

---

*Author: Weikang Kong (KAUST). Research benchmark for model-selection guidance.*