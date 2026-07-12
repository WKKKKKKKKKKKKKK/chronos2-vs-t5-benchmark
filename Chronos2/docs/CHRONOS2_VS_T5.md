# Chronos-2 vs Chronos-T5 — input formatting, tokenization, scaling

This note is the conceptual half of the study. It pins down *why* the two models
need different evaluation plumbing (see `src/run_zeroshot_chronos2.py` vs the
sibling `Chronos_benchmark/src/run_zeroshot_official.py`) and what to look at when
the benchmark numbers diverge. Everything below is grounded in the installed
`chronos-forecasting==2.2.2` source and the Chronos-2 technical report
(arXiv:2510.15821) / the original Chronos paper (Ansari et al., 2024, TMLR).

## TL;DR table

| axis | Chronos-T5 (Small, zero-shot baseline) | Chronos-2 |
| --- | --- | --- |
| backbone | T5 encoder–decoder, autoregressive | Encoder-only; blocks alternate **time** and **group** attention |
| input unit | one scalar per time step | a **patch** of `input_patch_size` consecutive steps |
| "tokenization" | mean-scale → **quantize into a fixed vocab** of bins (discrete token IDs) | **no vocabulary**; each patch is embedded by a residual MLP (`input_patch_embedding`) |
| scaling | per-series **mean (abs) scaling** | per-series **InstanceNorm** (standardize) + optional `arcsinh`/`sinh⁻¹` |
| output | **sampled** trajectories from a categorical over the token vocab | **direct quantile head** (multi-step) |
| how we get quantiles | sample 20 paths → empirical quantiles | model emits quantiles directly (21 native; we request 9) |
| stochastic? | yes → needs a fixed seed for reproducibility | **no** → deterministic by construction |
| multivariate / covariates | unsupported (strictly univariate) | native, via **group attention** over a group of series |
| context length | 512 | 2048 (8192 after post-training) |
| gluonts forecast object | `SampleForecast` | `QuantileForecast` |

## 1. Input formatting

**Chronos-T5** consumes a 1-D sequence of scalars. The official evaluation
batches a list of `{start, target}` series, each `target` a 1-D array; the model
sees one value per position. There is no notion of "other series" — every series
is an independent sequence.

**Chronos-2** consumes either a 1-D series (`(history_length,)`), a 2-D
multivariate series (`(n_variates, history_length)`), or a dict carrying
`target` + `past_covariates` + `future_covariates`. Internally each series is
split into **non-overlapping patches** of length `input_patch_size`
(`chronos/chronos2/model.py: self.patch = Patch(...)`), and two meta-features are
concatenated onto every patch before embedding (`input_patch_embedding` has
`in_dim = input_patch_size * 3`):

  1. a **time-index** feature `[-T/C, …, 0, …, (H-1)/C]` scaled by the context
     length `C` (`time_encoding_scale`), and
  2. a binary **observed/missing mask**.

A learned **`[REG]` token** is inserted between context patches and the
(to-be-predicted) future patches as a separator / attention sink
(`use_reg_token`, `config.reg_token_id = 1`).

> Practical consequence for the benchmark: Benchmark II ships as univariate
> `{start, target}` series. In `univariate` mode we feed each `target` as a 1-D
> array — the apples-to-apples match with T5. In `multivariate` mode we stack
> series that share an identical `(start, length)` into a 2-D
> `(n_variates, length)` item so they form one **group** (see §4).

## 2. Tokenization — the biggest representational difference

**Chronos-T5** is literally "the language of time series": after scaling, each
real value is **quantized** into one of a fixed number of bins, i.e. mapped to a
**discrete token ID** from a vocabulary, exactly like text. The model is a
classifier over that vocabulary and is trained with cross-entropy. Forecasting =
autoregressively sampling token IDs and de-quantizing them back to real values.
This is why T5 forecasts are **stochastic** and why quantiles are obtained by
drawing many sample paths.

**Chronos-2** has **no vocabulary and no quantization**. A patch of raw (scaled)
values is mapped straight into the embedding space by a residual network
(`ResidualBlock`). There is nothing discrete; the model is trained with a
**quantile-regression** loss, not cross-entropy. Forecasting is a single
deterministic forward pass producing quantiles — no sampling, no de-quantization,
no seed.

> Why it matters for WQL/MASE: T5's quantiles are *empirical* (20 samples → noisy
> at the tails, seed-dependent). Chronos-2's quantiles are *parametric outputs*
> (stable, reproducible). We request the **same 9 quantile levels (0.1…0.9)** from
> both so `MeanWeightedSumQuantileLoss` is computed on an identical grid; Chronos-2
> natively predicts 21 levels including extremes (0.01/0.99) it could expose for
> richer tail coverage.

## 3. Scaling / normalization

**Chronos-T5** divides each series by its mean absolute value (per-series mean
scaling) before quantization, and rescales the de-quantized output back.

**Chronos-2** applies **`InstanceNorm`** (`chronos/chronos_bolt.py`, reused by
`chronos2/model.py`): subtract the per-series mean, divide by the per-series std
to get `(v - μ)/σ`, with an optional **`arcsinh` (`sinh⁻¹`) transform**
(`use_arcsinh`) that compresses outliers/heavy tails before the network sees
them. The `(loc, scale)` pair is cached and reused to un-normalize the outputs —
and, crucially, the **same** normalization is applied to known future covariates
(`_prepare_future`), so target and covariates live on comparable scales.

> Why it matters: standardization + `arcsinh` behaves differently from mean
> scaling on series with large dynamic range, near-zero means, or heavy tails
> (finance, intermittent retail). When Chronos-2 and T5 diverge most on a given
> Benchmark II dataset, scaling is usually the first suspect.

## 4. Multivariate: the group-attention mechanism

This is the capability Chronos-2 adds and the reason for the `multivariate` mode
in this study. Each encoder block alternates two attention types
(`chronos/chronos2/model.py: Chronos2EncoderBlock`):

  * **time attention** — across patches *within a single series* (temporal
    dynamics; the only thing T5 does), and
  * **group attention** — across *all series in the same group* at each patch
    index (cross-series / cross-variate dynamics).

A **group** is identified by integer `group_ids` mapped to a 2-D attention mask,
so information is shared *only within a group*. Per the technical report, a group
may be:

  * a single series → univariate (group attention is inert);
  * a set of **variates of a multivariate series** (shared dynamics);
  * a set of **related series** for in-context learning (ICL) / "cross-learning";
  * **target(s) + covariates** (past-only and/or known-future).

`Chronos2Pipeline.predict` exposes this as: 1-D list elements → univariate; 2-D
`(n_variates, L)` elements → one shared-dynamics multivariate group;
`cross_learning=True` → every item in the **batch** is forced into one group.

> How this study uses it — **exactly as the technical report does on Benchmark II**
> (§5.1). Benchmark II is a *univariate* benchmark: each task is a standalone 1-D
> series with no declared multivariate variates and no labelled covariates. So the
> report does **not** apply shared-dynamics multivariate here — that is reserved for
> fev-bench / GIFT-Eval. Instead its headline Benchmark II numbers use Chronos-2's
> **full cross-learning** mode: 1-D inputs, every item in a batch assigned the *same*
> group id, with a group/batch size of ~100. We mirror exactly that in the
> `cross_learning` mode (`cross_learning=True`, `batch_size=CROSS_LEARNING_BATCH=100`).
> Because the group **is** the batch, the result is batch-size dependent — intended,
> and matching the report. Cross-learning helps most when individual series have
> short histories (Benchmark II has many such tasks), which is why it can move the
> needle here even though genuine multivariate gains live on other benchmarks.
>
> (Note: an earlier version of this study grouped series by aligned `(start, length)`
> into 2-D shared-dynamics items. That is a defensible interpretation for genuinely
> related panels — `exchange_rate` currencies, `ercot` zones — but it is **not** how
> the report handles Benchmark II, so it was replaced by full cross-learning to stay
> faithful to the paper.)

## 5. Consequences for the evaluation code

| step | T5 script | Chronos-2 script |
| --- | --- | --- |
| forecast call | `predict_quantiles(...)` then treat 9 quantiles as 20-sample proxy | `predict_quantiles(...)` returns quantiles directly |
| reproducibility | `torch.manual_seed(SEED)` before sampling | none needed (deterministic) |
| gluonts object | `SampleForecast(samples=[B,Q,H])` | `QuantileForecast(forecast_arrays=[Q,H], forecast_keys=...)` |
| metrics | `MASE` + `MeanWeightedSumQuantileLoss` (identical) | **identical** — same gluonts pipeline, same split, same cap |

The metric definitions, the gluonts `split`, the `MAX_SERIES=1000` cap and the
dataset registry are byte-identical across the two projects, so any difference in
the headline aggregated relative score is attributable to the **model**, not the
harness.