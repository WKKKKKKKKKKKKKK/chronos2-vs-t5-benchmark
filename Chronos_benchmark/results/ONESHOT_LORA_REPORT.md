# One-shot (fine-tuned) Zero-shot-benchmark Reproduction

chronos-t5-small fine-tuned per dataset (lr 1e-3 -> 0 over 1000 steps, explicit PyTorch loop), evaluated via the official gluonts pipeline, cap=1000, on 1 Benchmark II datasets.

## Aggregated relative score (gmean of model / Seasonal-Naive)

| scenario | WQL | MASE |
| --- | --- | --- |
| one-shot (ours) | 0.694 | 0.700 |
| zero-shot (ours) | 0.687 | 0.852 |
| one-shot (paper, Fig. 6) | 0.597 | 0.760 |

## Per-dataset (one-shot, ours)

| dataset | MASE | WQL | n_series |
| --- | --- | --- | --- |
| monash_m1_yearly | 3.4279 | 0.1452 | 181 |

## Inference efficiency (ours, fine-tuned chronos-t5-small, GPU (bf16))

Total forecast wall-time 1.2s over 1 datasets; peak GPU memory 680 MB; mean 6.9 ms/series.

| dataset | n_series | latency_s | ms/series | peak_mem_MB |
| --- | --- | --- | --- | --- |
| monash_m1_yearly | 181 | 1.24 | 6.9 | 680 |