# chronos2_t5 — Chronos-2 vs Chronos-T5 comparison

The **comparison** project, parallel to the two model projects:

```
d:\KAUST\SAUDI_ARAMCO\
├── Chronos2\          # Chronos-2 zero-shot (its own results + vs the Chronos-2 paper)
├── Chronos_benchmark\ # Chronos-T5 zero-shot + one-shot reproduction
└── chronos2_t5\       # THIS: head-to-head Chronos-2 vs Chronos-T5
    ├── zero-shot\
    │   └── edge-case\ # corrupted-sensor robustness study (runs both models)
    └── one-shot\
```

The head-to-head artifacts **do not** run/own a model — they read the per-dataset results the
two sibling projects already produced. (The one exception is `zero-shot/edge-case/`, which
*runs* both models itself, reusing Chronos2's shared harness.) All use the identical gluonts
pipeline / 25 Benchmark II datasets / cap=1000 / bf16, so the comparison is apples-to-apples.

## zero-shot/

| file | what |
| --- | --- |
| `compare_zeroshot.py` | reads the two projects' result CSVs → writes the dashboard + head-to-head MD (no GPU) |
| `make_forecast_plots.py` | per-dataset forecast plots for both models at native horizons (needs GPU) |
| `c2_vs_t5_dashboard.png` | accuracy scatter + aggregated relative score + forecast time + peak GPU memory |
| `CHRONOS2_VS_T5_HEADTOHEAD.md` | aggregated relative score, win rate, efficiency, per-dataset MASE/WQL |
| `forecasts/chronos2/`, `forecasts/chronos_t5/` | one forecast PNG per dataset, per model + 5×5 overview |
| `edge-case/` | corrupted-sensor robustness study (has its own `README.md`; see below) |

Reproduce (after the two sibling projects have produced their results):

```powershell
cd zero-shot
python compare_zeroshot.py        # dashboard + head-to-head (fast, from CSVs)
python make_forecast_plots.py     # per-dataset forecast plots (GPU)
```

Headline (25 datasets, gmean of model / Seasonal-Naive, lower=better):
Chronos-2 cross-learning **WQL 0.563 / MASE 0.751** vs Chronos-T5 **0.687 / 0.852** —
~18% lower WQL, ~12% lower MASE, and ~7× faster / ~2.6× less peak GPU memory.

### zero-shot/edge-case/ — corrupted-sensor robustness stress test

A complementary **Chronos-2 vs Chronos-T5** comparison: how each model degrades when its
forecast *context* is corrupted (the held-out future stays clean), on 5 high-frequency
sensor-like datasets. Unlike the rest of this project it **runs both models**, reusing the
Chronos-2 harness in `../../Chronos2/src/` so the pipeline is identical.

| file | what |
| --- | --- |
| `perturbations.py` | spike / drift-ramp / level-shift / missing-chunk corruptions (seeded) |
| `run_edge_cases.py` | 2 models × 5 datasets × corruptions → `results/` CSV + report + figures |
| `make_edgecase_notebook.py` | builds (and `--execute`) the deliverable notebook |
| `Chronos2_EdgeCase_Robustness.ipynb` | generated notebook (tables, curves, example figures) |
| `_mk_three_config.py`, `_mk_examples.py` | build the two report figures in `plots/` (win-rate heatmap; illustrative series) |
| `results/EDGE_CASE_REPORT.md`, `results/examples/<dataset>/` | report + per-series figures |
| `README.md` | folder guide: scripts, data deps, figure regeneration steps |

```powershell
conda activate chronos_bench
cd zero-shot\edge-case
python run_edge_cases.py                  # full sweep -> results/
python run_edge_cases.py --report-only    # rewrite report + degradation curve from the CSV
python run_edge_cases.py --examples-only   # regenerate the per-series example figures
python make_edgecase_notebook.py --execute # build + run the notebook
```

**Headline:** only corruptions reaching the **recent** context near the forecast origin hurt
(both models anchor to recent values). **Spikes split by axis** (dense severity sweeps): under
*intensity* Chronos-2 degrades smoothly/monotonically while Chronos-T5 is erratic — badly hit at
moderate magnitudes (~1.8× around ×8–20) but recovering to ~1.0× at extreme magnitudes (its
tokenizer clamps out-of-range spikes); under *density* both worsen but Chronos-2 worsens more
steeply (at 20% spiked C2 2.4× vs T5 1.4× MASE; WQL gap reaches 11.7× vs 3.2× at 40%). Density is
the more damaging axis for both — C2's strength is amplitude-robustness, not count-robustness.
The **gradual drift ramp is catastrophic for both** (~30×, per-series normalisation does not save
it); a **random-segment level shift** and a **random-position gap** are essentially harmless to
both (~1.0×) — though a gap **pinned to the most-recent points** is far more damaging (placement,
not size, decides the damage).

## one-shot/

Placeholder for the one-shot head-to-head (Chronos-T5 one-shot fine-tuning already
exists in `Chronos_benchmark`; the Chronos-2 fine-tuning counterpart is future work).
