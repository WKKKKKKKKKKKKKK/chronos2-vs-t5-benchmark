"""Count, on the exact REPRESENTATIVE series shown in forecasts/chronos2/cross_learning/,
how often cross-learning beats univariate (lower WQL / MASE = better).

This uses the same series pick + scoring as make_forecast_plots.py (one series per dataset =
the longest), so the counts match what the plot titles show -- distinct from the full-dataset
eval in run_zeroshot_chronos2 (which averages over all series). Writes the per-series scores
to a CSV so they're inspectable. C2 zero-shot base model; needs a GPU.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import datasets as hfds

HERE = Path(__file__).resolve().parent          # .../zero-shot/scripts
ZS = HERE.parent                                 # .../zero-shot (output root)
sys.path.insert(0, str(HERE))                    # import make_forecast_plots (sibling)
import make_forecast_plots as M                     # reuse the exact plot logic
from chronos import BaseChronosPipeline

hfds.logging.set_verbosity_error(); hfds.disable_progress_bars()
R2, D = M.R2, M.D

rows = []
pipe = BaseChronosPipeline.from_pretrained(R2.MODEL_ID, device_map="cuda", torch_dtype=R2.DTYPE)
for cfg, H in D.BENCHMARK_II:
    ds = hfds.load_dataset(R2.HF_REPO, cfg, split="train"); ds.set_format("numpy")
    gts = R2.to_gluonts_univariate(ds, R2.MAX_SERIES)
    tidx = M.pick_series_index(gts)
    targets = [np.asarray(e["target"], np.float32) for e in gts]
    entry = {"start": gts[tidx]["start"], "target": targets[tidx]}
    *_, um, uw = M.forecast_uni(pipe, entry, H, "c2")           # ctx,fut,a,MASE,WQL
    _, cm, cw, _ = M.forecast_cl(pipe, targets, tidx, entry, H)  # q,MASE,WQL,group
    rows.append(dict(dataset=cfg, uni_MASE=um, uni_WQL=uw, cl_MASE=cm, cl_WQL=cw))
    print(f"  {cfg:32s} uni {uw:.4f}/{um:.3f}  CL {cw:.4f}/{cm:.3f}", flush=True)

df = pd.DataFrame(rows)
out = ZS / "forecasts" / "chronos2" / "representative_series_scores.csv"
df.to_csv(out, index=False)
n = len(df)
print(f"\n=== representative series (1 per dataset, as in the plots), n={n} ===")
for m in ["WQL", "MASE"]:
    b = int((df[f"cl_{m}"] < df[f"uni_{m}"]).sum())
    w = int((df[f"cl_{m}"] > df[f"uni_{m}"]).sum())
    print(f"{m}: cross-learning better {b}/{n}  (worse {w}, tie {n-b-w})")
both = int(((df["cl_WQL"] < df["uni_WQL"]) & (df["cl_MASE"] < df["uni_MASE"])).sum())
print(f"CL better on BOTH metrics: {both}/{n}")
print(f"saved -> {out}")