# one-shot — Chronos-2 vs Chronos-T5 one-shot (LoRA) study

Files are grouped by kind:

| folder | contents |
| --- | --- |
| `scripts/` | all Python code (run from here) |
| `results/` | CSV outputs (per-dataset metrics + aggregates) |
| `reports/` | Markdown reports |
| `logs/` | captured run logs |
| `plots/` | all figures (PNG) |

## Scripts (run with `conda activate chronos_bench`, from this folder)

| script | phase | writes |
| --- | --- | --- |
| `scripts/hpo.py --model {c2,t5}` | 3 — val-based HPO (lr/rank/ctx) | `results/hpo_<model>_results.csv` |
| `scripts/final_run.py --model {c2,t5}` | 4 — train all 25 with best config + univariate eval | `results/oneshot_hpo_<model>.csv` |
| `scripts/final_run.py --model c2 --eval-only --cross-learning` | 4b — C2 cross-learning eval (ceiling ref) | `results/oneshot_hpo_c2_crosslearning.csv` |
| `scripts/head_to_head.py` | 5 — C2 vs T5 head-to-head (univariate) | `results/head_to_head.csv`, `reports/HEAD_TO_HEAD_REPORT.md`, `plots/h2h_*`, `plots/agg_*` |
| `scripts/phase6_cltrain.py` | 6 — train-time cross-learning C2 (C2-only, not head-to-head) | `results/oneshot_cltrain_c2.csv`, `reports/PHASE6_CLTRAIN_REPORT.md` |
| `scripts/summary_matrix.py` | all-7-settings per-dataset + aggregate heatmap | `results/summary_matrix.csv`, `plots/summary_matrix_*` |
| `scripts/plot_oneshot_loss.py --dataset <name>` | training-loss curve T5 vs C2 | `plots/loss_<dataset>.png` |

Adapters/checkpoints live outside this folder, under each model project's `models/` dir
(`Chronos2/models/…`, `Chronos_benchmark/models/…`); TensorBoard logs under each project's `runs/`.

## Headline result (aggregate, rel. Seasonal-Naive, lower=better, 25 datasets)

Best-use ranking: **C2 zero-shot+CL 0.563 (WQL)** > C2 zero-shot uni 0.588 > T5 one-shot 0.596 >
C2 one-shot CLtrain 0.609 > … > T5 zero-shot 0.687. C2's edge over T5 is **zero-shot + cross-learning**,
not one-shot fine-tuning (where the two are statistically tied).