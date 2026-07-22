# edge-case — corruption-robustness sub-study (Chronos-2 vs Chronos-T5)

Injects controlled sensor faults (spikes, drift, level-shift, missing chunks) into the
**forecast context** across the 25 Benchmark-II datasets and measures error vs the clean
held-out window. Headline metric = **relative degradation** = metric(corrupted) / metric(clean).
Run every script from this directory with the `chronos_bench` conda env.

> **Naming convention.** Files **without** an underscore prefix are the core reproducible
> pipeline (`run_edge_cases.py`, `perturbations.py`, `make_edgecase_notebook.py`).
> `_`-prefixed files are **supplementary one-off analyses layered on top** of that sweep
> (extra model configs, side probes, and the report-figure generators). They are not scratch
> —each is documented in the table below, and some feed the report figures in `plots/`.

## Layout

| path | what it holds | produced by |
| --- | --- | --- |
| `run_edge_cases.py` | main sweep — perturb context, forecast C2 & T5, score MASE/WQL + degradation | — |
| `perturbations.py` | the controlled corruption families (spikes / drift / level-shift / missing) | — |
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

Everything is driven by the CSVs in `results/`. To rebuild from scratch (conda env `chronos_bench`):

```bash
# 1. main sweep (GPU) -> results/edge_case_results.csv, EDGE_CASE_REPORT.md, examples/
python run_edge_cases.py

# 2. add the cross-learning config (GPU) -> results/_c2cl_full.csv
python _run_c2cl_full.py

# 3. figures (read CSVs from results/)
python _mk_three_config.py     # CPU-only  -> plots/robust_three_config.png
python _mk_examples.py         # GPU       -> plots/robust_examples.png
```

Data inputs for the two report figures:

- `_mk_three_config.py` → `results/edge_case_results.csv` + `results/_c2cl_full.csv`
- `_mk_examples.py`     → same two CSVs (for the annotated WQL) **plus** `run_edge_cases.py`
  (imported as `E` to rebuild the datasets and re-run the pipelines for the plotted series)

`_mk_three_config.py` is deterministic (bootstrap seed 0) — it reproduces the slide figure
byte-for-byte. `_mk_examples.py` re-runs the C2 and T5 pipelines, so it needs a GPU.
