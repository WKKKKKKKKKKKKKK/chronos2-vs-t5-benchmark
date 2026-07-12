# zero-shot — Chronos-2 vs Chronos-T5 zero-shot study (Benchmark II)

All analysis **code** lives in `scripts/`; each analysis writes into its own **topic folder**.
Run every script from this directory as `python scripts/<name>.py` (conda env `chronos_bench`).
Scripts only READ the per-model result CSVs produced by the sibling projects
(`Chronos2/results/`, `Chronos_benchmark/results/`); none retrain a model.

## Layout

| folder | what it holds | produced by |
| --- | --- | --- |
| `scripts/` | all analysis code (6 scripts) | — |
| `forecasts/` | per-dataset forecast plots: `chronos2/{univariate,cross_learning}/`, `chronos_t5/` + overview grids + `representative_series_scores.csv` | `make_forecast_plots.py`, `count_plot_scores.py` |
| `benchmark/` | leaderboard (win rate, skill, runtime, leakage, failures) + win-rate bars + pairwise win-rate matrices | `benchmark_table.py` |
| `headtohead/` | C2-vs-T5 aggregate dashboard + report | `compare_zeroshot.py` |
| `cl_length/` | cross-learning benefit vs series length (scatter + trend) + per-series CSV | `cl_benefit_vs_length.py` |
| `crosslearning_io/` | cross-learning input-group → target-output mechanism figure (per dataset) | `plot_crosslearning_io.py` |
| `edge-case/` | corruption-robustness sub-study (own scripts + `results/`) | `edge-case/run_edge_cases.py` |

## Scripts

| script | needs GPU | writes to |
| --- | --- | --- |
| `scripts/compare_zeroshot.py` | no | `headtohead/` |
| `scripts/benchmark_table.py` | no | `benchmark/` |
| `scripts/make_forecast_plots.py` | yes | `forecasts/` |
| `scripts/count_plot_scores.py` | yes | `forecasts/chronos2/` (imports make_forecast_plots) |
| `scripts/cl_benefit_vs_length.py` | yes | `cl_length/` (imports make_forecast_plots) |
| `scripts/plot_crosslearning_io.py` | yes | `crosslearning_io/` |

Path convention (all scripts): `HERE` = `scripts/`, `ZS` = this `zero-shot/` dir (output root),
`ROOT` = repo root (`SAUDI_ARAMCO`).

## Headline (aggregate, rel. Seasonal-Naive skill, 25 datasets)

C2 zero-shot + cross-learning is #1 (win rate WQL 75.4, skill 43.7), ahead of C2 univariate
(41.2) and well ahead of Chronos-T5 zero-shot (31.3). See `benchmark/` and `headtohead/`.