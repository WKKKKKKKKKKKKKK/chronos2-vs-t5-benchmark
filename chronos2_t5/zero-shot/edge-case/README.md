# edge-case — corruption-robustness sub-study (Chronos-2 vs Chronos-T5)

Injects controlled sensor faults (spikes, drift, level-shift, missing chunks, persistent
trend change) into the **forecast context** across the 25 Benchmark-II datasets and
measures error vs the clean held-out window. Headline metric = **relative degradation** =
metric(corrupted) / metric(clean). Run every script from this directory with the
`chronos_bench` conda env (`chronos2_t5/environment.yml`).

> **Start at [`REPRODUCING.md`](REPRODUCING.md).** It maps every claim to the command that
> produces it and the file it lands in, gives the full run order with timings, and lists
> the six ways this pipeline can silently return a wrong answer. This README describes the
> layout; that one tells you what to run and what to trust.

> **Naming convention.** Files **without** an underscore prefix are the core reproducible
> pipeline — the `run_*` / `analyse_*` pair per experiment, plus `perturbations.py` and
> `statistics.py`. `_`-prefixed files are **supplementary one-off analyses layered on top**
> of the original sweep (extra model configs, side probes, and the report-figure
> generators). They are not scratch — each is documented in the second table below, and
> some feed the report figures in `plots/`.

## Layout

### Paper pipeline

These produce every number in the write-up. Run order and timings are in
[`REPRODUCING.md`](REPRODUCING.md) §4.

| path | what it does | GPU | writes |
| --- | --- | --- | --- |
| `perturbations.py` | the seven corruption families; also the canonical naming table | — | — |
| `run_edge_cases.py` | main sweep, seed 0 — perturb, forecast C2 & T5, score MASE/WQL | yes | `results/edge_case_results.csv` |
| `run_seeds.py` | re-runs chosen families under further seeds; `--out` and `--severities` let a new family or grid go to its own file | yes | `results/edge_case_seeds.csv`, `results/edge_case_regime*.csv` |
| `measure_clamping.py` | instruments the Chronos-T5 tokeniser directly — what it admits, clamps, rescales | no | `results/clamping_measurements.csv` |
| `run_cl_shuffle.py` | cross-learning sibling shuffle: native / foreign-same-freq / foreign-diff-freq | yes | `results/crosslearning_shuffle.csv` |
| `run_decode_ablation.py` | holds the corruption fixed and sweeps Chronos-T5's decode seed only — is the flat slope sampling noise? | yes | `results/decode_ablation.csv` |
| `statistics.py` | **the single source of truth for every p-value.** Fixes analysis unit, pairing and multiplicity; reports both metrics | no | `results/STATISTICS.md` |
| `analyse_seeds.py` | C1 — per-curve monotonicity, held-out recovery test | no | `SEED_ANALYSIS_*.md`, `fig_seed_curves_*.png` |
| `analyse_clamping.py` | C2 — do the candidate mechanisms predict recovery? (they do not) | no | `CLAMPING_ANALYSIS_*.md`, `fig_clamping_mechanism_*.png` |
| `analyse_cl_shuffle.py` | C4 — gain by sibling condition, paired arm contrasts | no | `CL_SHUFFLE_ANALYSIS_*.md`, `fig_cl_shuffle_*.png` |
| `analyse_cl_sensitivity.py` | C4 — does that gain survive without the four datasets holding fewer than 100 series? Imports its statistics from `analyse_cl_shuffle.py` so the two cannot drift | no | `CL_SENSITIVITY_ANALYSIS.md` |
| `analyse_decode_ablation.py` | C2b — slope per decode seed, and after averaging the sampler down | no | `DECODE_ABLATION_*.md`, `fig_decode_ablation_*.png` |
| `measure_homogeneity.py` | mean within-dataset series correlation. Written to test one explanation offered for C4 — and it **refuted** it, so the claim is not in the paper; kept so the removal leaves a trace | no | `HOMOGENEITY.md` |
| `mk_fig_suite.py` | one-figure overview of the corruption families, PNG and vector PDF | no | `fig_corruption_suite.{png,pdf}` |
| `analyse_regime.py` | C5 — slope on the new family, matched-pair vs `drift`, effect-size-matched control | no | `REGIME_ANALYSIS_*.md`, `fig_regime_*.png` |

### Original sweep and report figures

| path | what it holds | produced by |
| --- | --- | --- |
| `_run_c2cl_full.py` | adds Chronos-2 **cross-learning** to the sweep (same protocol) → `_c2cl_full.csv` | — |
| `_mk_three_config.py` | **3-config win-rate heatmap** figure (C2-CL / C2-uni / T5) | → `plots/` |
| `_mk_examples.py` | **illustrative-series** figure (2 panels: missing@boundary, spikes) | → `plots/` |
| `_cl_rescue_quick.py` | probe: does a clean related-series group rescue a corrupted target | → `results/_cl_rescue_full.csv` |
| `_measure_width.py` | 80% prediction-interval width, C2-uni vs T5 | → `results/_interval_width.csv` |
| `make_edgecase_notebook.py` | rebuilds `Chronos2_EdgeCase_Robustness.ipynb` | — |
| `results/` | all CSV outputs + `EDGE_CASE_REPORT.md` + `fig_degradation_curves.png` (relative) + `fig_absolute_curves.png` (absolute, 1:1 counterpart) + per-dataset `examples/` | `run_edge_cases.py` |
| `plots/` | **report figures** — see below | `_mk_three_config.py`, `_mk_examples.py` |

## Report figures (`plots/`)

These are the two figures used on the "Result 4 — Robustness to Corrupted Input" slide of
`weekly_report/Chronos2_vs_T5_Final.pptx`. Each script reads only the CSVs in `results/`.

| figure | script | needs GPU | shows |
| --- | --- | --- | --- |
| `plots/robust_three_config.png` | `_mk_three_config.py` | no | pairwise win rate (95% CI, 2000-resample bootstrap over the 25 datasets) on absolute WQL and on relative degradation, for C2-CL / C2-uni / T5 |
| `plots/robust_examples.png` | `_mk_examples.py` | yes | two illustrative series (traffic missing-chunk @ boundary; australian-electricity noisy spikes) — C2 tracks the actual, T5 breaks |

## Dependencies & run order

For the **paper pipeline**, see [`REPRODUCING.md`](REPRODUCING.md) §4 — it has the full
sequence, timings, and the traps.

For the **original report figures** (conda env `chronos_bench`):

```bash
# 1. main sweep (GPU) -> results/edge_case_results.csv, EDGE_CASE_REPORT.md, examples/
python run_edge_cases.py

# 2. add the cross-learning config (GPU) -> results/_c2cl_full.csv
python _run_c2cl_full.py

# 3. figures (read CSVs from results/)
python _mk_three_config.py     # CPU-only  -> plots/robust_three_config.png
python _mk_examples.py         # GPU       -> plots/robust_examples.png
```

`Chronos2_EdgeCase_Robustness.ipynb` and `make_edgecase_notebook.py` predate the paper work
(June 2026) and are kept as a historical record of the internship deliverable. They do not
reflect the corrected statistics — for anything quantitative use `results/STATISTICS.md`.

Data inputs for the two report figures:

- `_mk_three_config.py` → `results/edge_case_results.csv` + `results/_c2cl_full.csv`
- `_mk_examples.py`     → same two CSVs (for the annotated WQL) **plus** `run_edge_cases.py`
  (imported as `E` to rebuild the datasets and re-run the pipelines for the plotted series)

`_mk_three_config.py` is deterministic (bootstrap seed 0) — it reproduces the slide figure
byte-for-byte. `_mk_examples.py` re-runs the C2 and T5 pipelines, so it needs a GPU.
