# Reference numbers (paper's official results)

The two CSVs here are the **Chronos paper's official Benchmark II (zero-shot)
numbers**, used by `run_zeroshot_official.py` / `run_oneshot_official.py` to
compute the *ours-vs-paper* comparison and the aggregated relative score
(`gmean(model / Seasonal-Naive)`).

| file | what |
|---|---|
| `seasonal-naive-zero-shot.csv` | Seasonal-Naive baseline (the denominator of the relative score) |
| `chronos-t5-small-zero-shot.csv` | Chronos-T5 Small zero-shot (the paper's numbers we reproduce) |

Source: the official [chronos-forecasting](https://github.com/amazon-science/chronos-forecasting)
repo, `scripts/evaluation/results/`. Bundled here (a few KB each) so the
reproduction is self-contained and needs no external repo clone or local paths.
Each file has columns `dataset, model, MASE, WQL`.
