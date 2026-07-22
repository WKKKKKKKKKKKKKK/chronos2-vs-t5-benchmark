"""Visualize Chronos-2 CROSS-LEARNING as inputs -> one output (extra experiment; standalone).

Cross-learning is a *batch* effect: a group of series is fed together and group-attention
shares information across them. A single-series forecast plot cannot show this. So here,
for one dataset, we draw BOTH halves of the mechanism:

  * TOP panel  -- the INPUTS: the target series' actual cross-learning group (the positional
    batch of up to CROSS_LEARNING_BATCH series it falls in), z-normalized so their shapes
    are comparable (that is how InstanceNorm presents them to the model). Target in bold.
  * BOTTOM panel -- the OUTPUT: the forecast produced for the TARGET *because it sat in that
    group* (cross-learning p50 + 10-90% band), with the univariate p50 (target alone)
    overlaid dashed so you can see what cross-learning changed.

FAITHFUL TO EVALUATION: the forecast is computed with the SAME call the eval uses --
all the dataset's series with `cross_learning=True, batch_size=CROSS_LEARNING_BATCH` --
and the target's forecast is extracted by index. So the target's plotted curve is the
*exact* forecast the eval produces for that series (Chronos-2 is deterministic). The plot
never computes metrics; the aggregate scores still come from run_zeroshot_chronos2.

Does NOT touch make_forecast_plots.py. Zero-shot base model. Needs a GPU.
Run:  python plot_crosslearning_io.py --dataset monash_m1_yearly --n-show 12
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import datasets as hfds

HERE = Path(__file__).resolve().parent          # .../zero-shot/scripts
ZS = HERE.parent                                 # .../zero-shot (output root)
ROOT = HERE.parents[2]                            # repo root
sys.path.insert(0, str(ROOT / "Chronos2" / "src"))                # config, datasets_lib (src root)
sys.path.insert(0, str(ROOT / "Chronos2" / "src" / "zero_shot"))  # run_zeroshot_chronos2.py (shared harness)
import run_zeroshot_chronos2 as R2          # loaders / MODEL_ID / DTYPE / CROSS_LEARNING_BATCH ...
import datasets_lib as D                    # BENCHMARK_II (dataset -> native horizon)

from chronos import BaseChronosPipeline

hfds.logging.set_verbosity_error(); hfds.disable_progress_bars()
Qs = [0.1, 0.5, 0.9]


def _arr(a):
    a = a.cpu().numpy() if hasattr(a, "cpu") else np.asarray(a)
    return a[0] if a.ndim == 3 else a


def _znorm(x):
    x = np.asarray(x, np.float32)
    m, s = np.nanmean(x), np.nanstd(x)
    return (x - m) / (s + 1e-8)


def main():
    ap = argparse.ArgumentParser(description="Chronos-2 cross-learning inputs -> output visualization.")
    ap.add_argument("--dataset", default="monash_m1_yearly", help="Benchmark II dataset name")
    ap.add_argument("--n-show", type=int, default=12, help="how many input series to draw in the top panel")
    ap.add_argument("--out", default=None, help="output PNG path")
    args = ap.parse_args()

    B = R2.CROSS_LEARNING_BATCH
    H = dict(D.BENCHMARK_II)[args.dataset]
    ds = hfds.load_dataset(R2.HF_REPO, args.dataset, split="train"); ds.set_format("numpy")
    gts = R2.to_gluonts_univariate(ds, R2.MAX_SERIES)
    targets = [np.asarray(e["target"], np.float32) for e in gts]         # eval order (do NOT reorder)
    contexts = [t[: len(t) - H] for t in targets]                        # each series' context (drop last-H)
    tidx = max(range(len(targets)), key=lambda i: int(np.isfinite(targets[i]).sum()))   # target = longest

    pipe = BaseChronosPipeline.from_pretrained(R2.MODEL_ID, device_map="cuda", torch_dtype=R2.DTYPE)
    # EXACTLY the eval's cross-learning call: all series, batch_size=CROSS_LEARNING_BATCH (batch == group).
    qx, _ = pipe.predict_quantiles(contexts, prediction_length=H, quantile_levels=Qs,
                                   cross_learning=True, batch_size=B, limit_prediction_length=False)
    xl = _arr(qx[tidx])                                                  # target's forecast = eval's for it
    qu, _ = pipe.predict_quantiles([contexts[tidx]], prediction_length=H, quantile_levels=Qs,
                                   limit_prediction_length=False)        # univariate (target alone)
    uni = _arr(qu[0])

    # the target's ACTUAL group = the positional batch it lands in during eval
    g0 = (tidx // B) * B
    batch_idx = list(range(g0, min(g0 + B, len(targets))))
    show_idx = ([tidx] + [j for j in batch_idx if j != tidx])[: args.n_show]

    rep = targets[tidx]
    ctx, fut = rep[: len(rep) - H], rep[len(rep) - H:]
    W = max(3 * H, 80)

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(11, 8))

    # TOP: the target's cross-learning group (inputs)
    for j in show_idx:
        z = _znorm(contexts[j])[-W:]
        x = np.arange(-len(z), 0)
        if j == tidx:
            axA.plot(x, z, color="C0", lw=2.0, zorder=3, label="target series (forecast below)")
        else:
            axA.plot(x, z, color="grey", lw=0.7, alpha=0.5,
                     label="other group series" if j == show_idx[1] else None)
    axA.set_title(f"INPUTS — target's cross-learning group in '{args.dataset}': "
                  f"batch of {len(batch_idx)} series (showing {len(show_idx)}), z-normalized")
    axA.set_xlabel("steps before forecast origin"); axA.legend(fontsize=8, loc="upper left")
    axA.grid(alpha=0.3)

    # BOTTOM: the one output forecast for the target (exact eval forecast)
    show = ctx[-W:]
    xc, xf = np.arange(-len(show), 0), np.arange(0, H)
    axB.plot(xc, show, color="C0", lw=1.2, label="target history")
    axB.plot(xf, fut, color="C2", lw=1.8, label="actual (held-out)")
    axB.plot(xf, xl[:, 1], color="C3", lw=1.9, label="cross-learning p50 (= eval)")
    axB.fill_between(xf, xl[:, 0], xl[:, 2], color="C3", alpha=0.25, label="cross-learning 10-90%")
    axB.plot(xf, uni[:, 1], color="C1", ls="--", lw=1.6, label="univariate p50 (target alone)")
    axB.axvline(0, color="grey", ls="--", lw=0.8)
    axB.set_title(f"OUTPUT — forecast for the target series (H={H}): cross-learning vs univariate")
    axB.set_xlabel("steps from forecast origin"); axB.legend(fontsize=8, loc="upper left")
    axB.grid(alpha=0.3)

    fig.suptitle(f"Chronos-2 cross-learning ('{args.dataset}'): group in (top) -> target's forecast out (bottom)\n"
                 f"target is series #{tidx}, grouped with its positional batch [{g0}:{g0 + len(batch_idx)}] "
                 f"-- exactly as in evaluation", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_dir = ZS / "crosslearning_io"; out_dir.mkdir(parents=True, exist_ok=True)   # dedicated folder
    out = Path(args.out) if args.out else out_dir / f"{args.dataset}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved -> {out}  (target #{tidx} in batch [{g0}:{g0 + len(batch_idx)}])")


if __name__ == "__main__":
    main()
