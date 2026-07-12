# Benchmark II leaderboard (our reproduction, 25 datasets)

Chronos-2-report layout (arXiv:2510.15821 Tables 3/5). Win rate & skill score are with respect to each metric; higher is better for both. Leakage = 0 (no Benchmark-II pretraining; one-shot holds out the eval window). #Failures = datasets with non-finite MASE/WQL.

## WQL

| Model | Avg Win Rate (%) | Skill Score (%) | Median runtime (s) | Leakage (%) | #Failures |
| --- | --- | --- | --- | --- | --- |
| Chronos-2 (zs, CL) | 75.4 | 43.7 | 0.29 | 0 | 0 |
| Chronos-2 (1s, CL-eval) | 66.9 | 38.6 | 0.58 | 0 | 0 |
| Chronos-2 (zs, uni) | 62.3 | 41.2 | 0.22 | 0 | 0 |
| Chronos-2 (1s, CLtrain) | 59.4 | 39.1 | 0.58 | 0 | 0 |
| Chronos-2 (1s, uni) | 53.7 | 37.2 | 0.50 | 0 | 0 |
| Chronos-T5 (1s) | 52.0 | 40.4 | 3.76 | 0 | 0 |
| Chronos-T5 (zs) | 25.7 | 31.3 | 5.78 | 0 | 0 |
| Seasonal Naive | 4.6 | 0.0 | — | 0 | 0 |

## MASE

| Model | Avg Win Rate (%) | Skill Score (%) | Median runtime (s) | Leakage (%) | #Failures |
| --- | --- | --- | --- | --- | --- |
| Chronos-2 (zs, CL) | 72.0 | 24.9 | 0.29 | 0 | 0 |
| Chronos-T5 (1s) | 63.4 | 24.3 | 3.76 | 0 | 0 |
| Chronos-2 (zs, uni) | 60.6 | 22.7 | 0.22 | 0 | 0 |
| Chronos-2 (1s, CL-eval) | 55.4 | 19.1 | 0.58 | 0 | 0 |
| Chronos-2 (1s, CLtrain) | 53.1 | 19.8 | 0.58 | 0 | 0 |
| Chronos-2 (1s, uni) | 51.4 | 19.2 | 0.50 | 0 | 0 |
| Chronos-T5 (zs) | 25.7 | 14.8 | 5.78 | 0 | 0 |
| Seasonal Naive | 18.3 | 0.0 | — | 0 | 0 |
