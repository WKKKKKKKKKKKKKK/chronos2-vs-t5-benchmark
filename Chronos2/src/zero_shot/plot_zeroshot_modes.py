"""Bar chart of Chronos-2 zero-shot on Benchmark II: univariate vs cross-learning.

Aggregated relative score = gmean(model / Seasonal-Naive) over the common datasets
(the paper's metric, lower is better). Reads the zero-shot result CSV -- no GPU.
This shows the cross-learning benefit, which a single-series forecast plot cannot.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gmean

SRC = Path(__file__).resolve().parent.parent   # src root (config.py, datasets_lib.py live here)
sys.path.insert(0, str(SRC))
from config import REFERENCE_DIR, RESULTS_DIR as RESULTS  # noqa: E402

BASE = pd.read_csv(REFERENCE_DIR / "seasonal-naive-zero-shot.csv").set_index("dataset")[["MASE", "WQL"]]


def agg(df):
    c = df.index.intersection(BASE.index)
    return (df.loc[c] / BASE.loc[c]).apply(gmean)


def main():
    zs = pd.read_csv(RESULTS / "zeroshot_chronos2_results.csv")
    configs = {
        "univariate": agg(zs[zs["mode"] == "univariate"].set_index("dataset")[["MASE", "WQL"]]),
        "cross-learning": agg(zs[zs["mode"] == "cross_learning"].set_index("dataset")[["MASE", "WQL"]]),
    }
    labels = list(configs)
    colors = ["#8ecae6", "#219ebc"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    for ax, metric in zip(axes, ["WQL", "MASE"]):
        vals = [configs[k][metric] for k in labels]
        bars = ax.bar(labels, vals, color=colors, width=0.6)
        best = min(range(len(vals)), key=lambda i: vals[i])
        bars[best].set_edgecolor("black"); bars[best].set_linewidth(2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
        drop = (vals[0] - vals[1]) / vals[0] * 100      # cross-learning improvement over univariate
        ax.set_title(f"zero-shot {metric}\n(cross-learning {drop:.1f}% lower than univariate)", fontsize=10)
        ax.set_ylim(0, max(vals) * 1.18)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Chronos-2 zero-shot on Benchmark II — cross-learning vs univariate\n"
                 "aggregated relative score = gmean(model / Seasonal-Naive), lower = better",
                 fontsize=10)
    fig.tight_layout()
    out = RESULTS / "zeroshot_modes.png"
    fig.savefig(out, dpi=130)
    print(f"saved -> {out}")
    for k in labels:
        print(f"  zero-shot {k:15s} WQL={configs[k]['WQL']:.3f}  MASE={configs[k]['MASE']:.3f}")


if __name__ == "__main__":
    main()
