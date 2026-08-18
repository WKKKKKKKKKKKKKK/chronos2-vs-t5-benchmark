"""E1 -- Direct measurement of Chronos-T5 quantiser clamping under corrupted context.

WHY THIS EXISTS
---------------
The edge-case sweep (`run_edge_cases.py`) shows Chronos-T5 degrading *non-monotonically*
under amplitude spikes: error peaks at moderate magnitudes and then RECOVERS at extreme
ones. A model that genuinely resists corruption does not get better as the corruption
gets worse, so the signature points at the input front-end rather than the forecaster.

Chronos-T5 mean-scales the context and buckets it into a FIXED, BOUNDED grid of value
tokens (`MeanScaleUniformBins`: centers = linspace(low_limit, high_limit, n_tokens -
n_special_tokens - 1); here [-15, +15] over 4093 bins). Any scaled value outside that
range is bucketed into the outermost bin -- i.e. CLAMPED. Past some magnitude a larger
spike is no longer larger *for the model*.

This script measures that directly instead of inferring it from the curve's shape. It
re-creates the sweep's corrupted contexts byte-for-byte (same `build_dataset`,
same seeded `perturb_contexts`) and, for every context, records what the tokeniser
actually admits. No model forward pass, no GPU -- this is tokeniser arithmetic only.

WHAT IS A "CLAMPED" TOKEN
-------------------------
`_input_transform` does `bucketize(scaled, boundaries, right=True) + n_special_tokens`,
where `boundaries = [-1e20, midpoints..., +1e20]`. For any finite value the bucket index
lands in [1, len(centers)], so value tokens occupy [n_special_tokens+1, n_tokens-1]
(here 3..4095). The two endpoints are exactly the bins that absorb out-of-range values:

    LO_TOKEN = n_special_tokens + 1   (3)     <- everything at or below centers[0]
    HI_TOKEN = n_tokens - 1           (4095)  <- everything above centers[-1]

NaN points are masked out (token 0) and excluded from both numerator and denominator.

OUTPUT (results/clamping_measurements.csv), one row per (dataset, family, severity):
  clamped_frac        mean over series of clamped value-tokens / valid value-tokens
  clamped_lo_frac     same, low end only          clamped_hi_frac   high end only
  series_any_clamped  fraction of series with at least one clamped point
  scale_mean          mean of the tokeniser's mean-scale (spikes inflate it -- a second,
                      independent suppression channel worth separating from clamping)
  nominal_p99/max     |scaled value| BEFORE bucketing  -- the excursion as injected
  realized_p99/max    |scaled value| AFTER clipping to the representable range
                      -- the excursion the model actually sees. The ratio of these two
                      is the basis for the "realised vs nominal severity" correction.

Usage:
    python measure_clamping.py                # all 25 datasets, all families
    python measure_clamping.py --smoke        # 2 datasets, spikes only (quick check)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Reuse the sweep's own dataset builder + seeded perturbation so the contexts measured
# here are the *same bytes* the models were scored on. Importing is safe: run_edge_cases
# guards its entry points behind `if __name__ == "__main__"`.
import run_edge_cases as RE  # noqa: E402
import perturbations as P    # noqa: E402  (severity constants, for the report header)

OUT = HERE / "results"
T5_ID = "amazon/chronos-t5-small"     # the model the sweep scored (RE.MODELS)


def load_tokenizer(model_id: str = T5_ID):
    """The exact tokeniser the sweep's Chronos-T5 used. CPU only; weights unused."""
    from chronos import BaseChronosPipeline

    pipe = BaseChronosPipeline.from_pretrained(model_id, device_map="cpu",
                                               torch_dtype=torch.float32)
    tok = pipe.tokenizer
    cfg = tok.config
    lo_token = cfg.n_special_tokens + 1        # first value bin  -> absorbs the low tail
    hi_token = cfg.n_tokens - 1                # last  value bin  -> absorbs the high tail
    meta = {
        "model": model_id,
        "n_tokens": cfg.n_tokens,
        "n_special_tokens": cfg.n_special_tokens,
        "context_length": cfg.context_length,
        "low_limit": float(tok.centers[0]),
        "high_limit": float(tok.centers[-1]),
        "lo_token": lo_token,
        "hi_token": hi_token,
        "n_bins": int(tok.centers.numel()),
    }
    del pipe.model                              # tokeniser is all we need; drop the weights
    return tok, meta


def measure_context(tok, meta, ctx: np.ndarray) -> dict | None:
    """Tokenise one context and report what the front-end admits.

    Mirrors `context_input_transform` (truncate to context_length, mean-scale, bucketise)
    and then drops the appended EOS column so it is not counted as a value token.
    """
    x = torch.tensor(np.asarray(ctx, dtype=np.float32)).unsqueeze(0)   # (1, L)
    token_ids, attn, scale = tok.context_input_transform(x)

    cfg = tok.config
    if cfg.use_eos_token and cfg.model_type == "seq2seq":
        token_ids, attn = token_ids[:, :-1], attn[:, :-1]              # strip EOS

    valid = attn[0]
    n_valid = int(valid.sum())
    if n_valid == 0:
        return None                                                    # all-NaN context

    tid = token_ids[0][valid]
    lo_hits = int((tid == meta["lo_token"]).sum())
    hi_hits = int((tid == meta["hi_token"]).sum())

    # Scaled values, recomputed from the returned scale (same quantity bucketize saw).
    # Truncate first so this matches the tokens exactly.
    xt = x[0]
    if xt.numel() > cfg.context_length:
        xt = xt[-cfg.context_length:]
    scaled = (xt / scale[0]).numpy()
    scaled = scaled[np.isfinite(scaled)]
    a = np.abs(scaled)

    # What survives the bounded grid: clipping to [centers[0], centers[-1]].
    clipped = np.abs(np.clip(scaled, meta["low_limit"], meta["high_limit"]))

    return {
        "n_valid": n_valid,
        "clamped": (lo_hits + hi_hits) / n_valid,
        "clamped_lo": lo_hits / n_valid,
        "clamped_hi": hi_hits / n_valid,
        "any_clamped": float((lo_hits + hi_hits) > 0),
        "scale": float(scale[0]),
        "nominal_p99": float(np.percentile(a, 99)),
        "nominal_max": float(a.max()),
        "realized_p99": float(np.percentile(clipped, 99)),
        "realized_max": float(clipped.max()),
    }


def run(smoke: bool = False):
    tok, meta = load_tokenizer()
    print(f"tokenizer: {meta['model']}  bins={meta['n_bins']} "
          f"range=[{meta['low_limit']:.1f}, {meta['high_limit']:.1f}] "
          f"clamp tokens={meta['lo_token']}/{meta['hi_token']} "
          f"context_length={meta['context_length']}", flush=True)

    datasets = RE.EDGE_DATASETS[:2] if smoke else RE.EDGE_DATASETS
    conditions = ([("clean", 0.0)]
                  + [(f, s) for f in ("spikes_intensity", "spikes_density")
                     for s in RE.SEVERITIES[f]]) if smoke else RE.CONDITIONS

    rows, t0 = [], time.perf_counter()
    for di, (config, horizon) in enumerate(datasets, 1):
        _, contexts, _ = RE.build_dataset(config, horizon)
        print(f"[{di}/{len(datasets)}] {config} (n={len(contexts)})", flush=True)
        for fam, sev in conditions:
            pc = RE.perturb_contexts(config, contexts, fam, sev)
            per = [m for m in (measure_context(tok, meta, c) for c in pc) if m is not None]
            if not per:
                continue
            agg = {k: float(np.mean([p[k] for p in per])) for k in
                   ("clamped", "clamped_lo", "clamped_hi", "any_clamped", "scale",
                    "nominal_p99", "nominal_max", "realized_p99", "realized_max")}
            rows.append({
                "dataset": config, "family": fam, "severity": sev,
                "n_series": len(per),
                "clamped_frac": agg["clamped"],
                "clamped_lo_frac": agg["clamped_lo"],
                "clamped_hi_frac": agg["clamped_hi"],
                "series_any_clamped": agg["any_clamped"],
                "scale_mean": agg["scale"],
                "nominal_p99": agg["nominal_p99"], "nominal_max": agg["nominal_max"],
                "realized_p99": agg["realized_p99"], "realized_max": agg["realized_max"],
            })
        del contexts

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / ("clamping_measurements_smoke.csv" if smoke else "clamping_measurements.csv")
    df.to_csv(path, index=False, lineterminator="\n")
    print(f"\n{len(df)} rows in {time.perf_counter() - t0:.1f}s -> {path}")

    # Headline: the intensity sweep, aggregated across datasets. This is the column the
    # non-monotonic degradation curve has to be read against.
    si = df[df.family == "spikes_intensity"].groupby("severity")
    if len(si):
        print("\nspikes_intensity (mean over datasets)")
        print(f"{'sev':>6} {'clamped':>9} {'anyser':>8} {'nom p99':>9} {'real p99':>9} {'scale':>9}")
        for sev, g in si:
            print(f"{sev:>6g} {g.clamped_frac.mean():>9.4f} {g.series_any_clamped.mean():>8.3f} "
                  f"{g.nominal_p99.mean():>9.2f} {g.realized_p99.mean():>9.2f} {g.scale_mean.mean():>9.3f}")
    return df


if __name__ == "__main__":
    run(smoke="--smoke" in sys.argv)
