"""How Chronos-2's cross-learning benefit varies with series length (context size).

Hypothesis (from the paper): in-context cross-learning helps most on SHORT series and least on
long/data-rich ones. Here we test it per-SERIES: for every series we compute its univariate and
its cross-learning WQL/MASE (the exact eval scores) and the relative benefit
    benefit = 100 * (1 - CL/uni)     (>0 = cross-learning better)
then plot benefit against the series' length (finite context points, log axis) with a
binned-median trend.

Eval-faithful: cross-learning forecasts use the full positional grouping
(cross_learning=True, batch_size=CROSS_LEARNING_BATCH), then we read each series' own forecast.
Series are sampled to span the length range (<= CAP per dataset) to bound the metric cost.
C2 zero-shot base model; needs a GPU.

Outputs (cl_length/):
  cl_benefit_by_series.csv                 per-series uni/CL WQL+MASE + length + dataset
  cl_benefit_vs_length_wql.png / _mase.png scatter + binned-median trend
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import datasets as hfds

HERE = Path(__file__).resolve().parent          # .../zero-shot/scripts
ZS = HERE.parent                                 # .../zero-shot (output root)
sys.path.insert(0, str(HERE))                    # import make_forecast_plots (sibling)
import make_forecast_plots as M                     # reuse loaders + series_metrics
from chronos import BaseChronosPipeline

hfds.logging.set_verbosity_error(); hfds.disable_progress_bars()
R2, D = M.R2, M.D
Qs = R2.QUANTILES
CAP = 100                                           # series per dataset for metrics (spanning lengths)
OUT = ZS / "cl_length"


def _arr(q):
    a = q.cpu().numpy() if hasattr(q, "cpu") else np.asarray(q)
    return a[0] if a.ndim == 3 else a


def collect():
    rows = []
    pipe = BaseChronosPipeline.from_pretrained(R2.MODEL_ID, device_map="cuda", torch_dtype=R2.DTYPE)
    for cfg, H in D.BENCHMARK_II:
        ds = hfds.load_dataset(R2.HF_REPO, cfg, split="train"); ds.set_format("numpy")
        gts = R2.to_gluonts_univariate(ds, R2.MAX_SERIES)
        targets = [np.asarray(e["target"], np.float32) for e in gts]
        starts = [e["start"] for e in gts]
        contexts = [t[: len(t) - H] for t in targets]
        lengths = np.array([int(np.isfinite(c).sum()) for c in contexts])
        # forecast ALL series both modes (CL grouping must be the full positional batching)
        qu, _ = pipe.predict_quantiles(contexts, prediction_length=H, quantile_levels=Qs,
                                       batch_size=R2.BATCH_SIZE, limit_prediction_length=False)
        qc, _ = pipe.predict_quantiles(contexts, prediction_length=H, quantile_levels=Qs,
                                       cross_learning=True, batch_size=R2.CROSS_LEARNING_BATCH,
                                       limit_prediction_length=False)
        order = np.argsort(lengths)                                   # sample across the length range
        idx = np.unique(order[np.linspace(0, len(order) - 1, min(CAP, len(order))).astype(int)])
        for i in idx:
            if lengths[i] <= H:
                continue
            entry = {"start": starts[i], "target": targets[i]}
            um, uw = M.series_metrics(entry, H, _arr(qu[i]))
            cm, cw = M.series_metrics(entry, H, _arr(qc[i]))
            rows.append(dict(dataset=cfg, length=int(lengths[i]),
                             uni_WQL=uw, cl_WQL=cw, uni_MASE=um, cl_MASE=cm))
        print(f"  {cfg:32s} {len(idx)} series (len {lengths.min()}-{lengths.max()})", flush=True)
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def plot(df, metric):
    ben = 100 * (1 - df[f"cl_{metric}"].to_numpy() / df[f"uni_{metric}"].to_numpy())   # >0 = CL better
    L = df["length"].to_numpy(float)
    ok = np.isfinite(ben) & np.isfinite(L) & (L > 0)
    ben, L = ben[ok], L[ok]
    fig, ax = plt.subplots(figsize=(8.2, 5))
    ax.scatter(L, np.clip(ben, -100, 100), s=10, alpha=0.28, color="C4", edgecolors="none",
               label="per series (clipped to ±100%)")
    ax.axhline(0, color="k", lw=1)
    # binned-median trend over log-length (robust to outliers -> uses UNclipped values)
    logL = np.log10(L)
    edges = np.linspace(logL.min(), logL.max(), 9)
    bi = np.digitize(logL, edges)
    xs, ys, q1, q3 = [], [], [], []
    for b in range(1, len(edges)):
        m = bi == b
        if m.sum() >= 5:
            xs.append(10 ** ((edges[b - 1] + edges[b]) / 2))
            ys.append(np.median(ben[m])); q1.append(np.percentile(ben[m], 25)); q3.append(np.percentile(ben[m], 75))
    ax.plot(xs, ys, "o-", color="C3", lw=2.2, label="binned median")
    ax.fill_between(xs, q1, q3, color="C3", alpha=0.15, label="IQR")
    r = np.corrcoef(logL, ben)[0, 1]
    ax.set_xscale("log")
    ax.set_xlabel("series length — finite context points (log)")
    ax.set_ylabel(f"cross-learning benefit: 100·(1 − CL/uni) {metric}   (>0 = CL better)")
    ax.set_title(f"Chronos-2 cross-learning benefit vs series length ({metric})\n"
                 f"{len(L)} series, {df['dataset'].nunique()} datasets; corr(log-length, benefit) = {r:.2f}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / f"cl_benefit_vs_length_{metric.lower()}.png", dpi=150); plt.close(fig)
    return r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = collect()
    df.to_csv(OUT / "cl_benefit_by_series.csv", index=False)
    print(f"\nseries={len(df)} datasets={df['dataset'].nunique()}")
    for metric in ["WQL", "MASE"]:
        r = plot(df, metric)
        share = 100 * (df[f"cl_{metric}"] < df[f"uni_{metric}"]).mean()
        print(f"{metric}: CL better on {share:.0f}% of series;  corr(log-length, benefit) = {r:.2f}")
    print(f"\nSaved -> {OUT}/")


if __name__ == "__main__":
    main()