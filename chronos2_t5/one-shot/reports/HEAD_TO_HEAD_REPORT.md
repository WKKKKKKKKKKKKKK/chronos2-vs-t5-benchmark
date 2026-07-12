# One-shot head-to-head: Chronos-2 vs Chronos-T5 (both LoRA-tuned)

Fair setting: identical HPO protocol + identical gluonts eval, **univariate on both sides**, over 25 Benchmark II datasets. C2-CL is a C2 self-ceiling reference only.

## Aggregate relative score (gmean of model / Seasonal-Naive, lower = better)

| line | MASE | WQL |
| --- | --- | --- |
| C2-uni | 0.808 | 0.628 |
| T5-uni | 0.757 | 0.596 |
| C2-CL | 0.809 | 0.614 |

## Head-to-head (C2-uni vs T5-uni)

| metric | gmean(C2/T5) | C2 win-rate | Wilcoxon p | verdict |
| --- | --- | --- | --- | --- |
| MASE | 1.067 | 10/25 | 0.120 | T5 better by 6.7%  (not significant) |
| WQL | 1.053 | 13/25 | 0.653 | T5 better by 5.3%  (not significant) |

## Reading

- In fair one-shot univariate fine-tuning the two models are **statistically tied** (Wilcoxon p > 0.05); any gap is not significant.
- Cross-learning on Benchmark II barely moves C2 in one-shot (see C2-CL), because these are weakly-related univariate series; C2's ICL benefit is largest on multivariate / covariate tasks, which are out of scope here.
- C2's demonstrated advantage over T5 is in the **zero-shot** setting (cross-learning), reported separately and consistent with the Chronos-2 technical report (which presents no one-shot fine-tuning comparison and does not benchmark against Chronos-T5 directly).

## Figures (plots/)

- `h2h_scatter_{mase,wql}.png` - per-dataset C2 vs T5 (below diagonal = C2 better)
- `h2h_ratio_{mase,wql}.png` - per-dataset C2/T5 ratio, sorted
- `agg_{mase,wql}.png` - aggregate C2-uni / T5-uni / C2-CL vs Seasonal-Naive