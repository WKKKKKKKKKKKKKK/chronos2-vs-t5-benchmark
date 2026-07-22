"""Combined per-dataset + aggregate comparison of ALL 7 settings, one heatmap per metric.

Settings (relative-to-Seasonal-Naive score, lower = better):
  zero-shot:  C2 zs-CL, C2 zs-uni, T5 zs
  one-shot:   C2 1s-CLtrain, C2 1s-CLeval, C2 1s-uni, T5 1s

Each figure (WQL, MASE): rows = 25 datasets + an AGGREGATE (gmean) row; columns = the 7
settings sorted by that metric's aggregate (best left). Cell text = the rel.-to-SN score;
cell colour = rank WITHIN that row (green = best model for the dataset, red = worst), so the
model comparison is visible despite datasets living on very different score scales.

Outputs (one-shot plots folder, per convention):
  plots/summary_matrix_wql.png
  plots/summary_matrix_mase.png
  summary_matrix.csv                 combined per-dataset rel.-to-SN scores (all 7 lines)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gmean

ONESHOT = Path(__file__).resolve().parents[1]   # .../chronos2_t5/one-shot
ROOT = ONESHOT.parents[1]                        # repo root
RESULTS = ONESHOT / "results"
PLOTS = ONESHOT / "plots"
SN = pd.read_csv(ROOT / "Chronos_benchmark" / "reference" / "seasonal-naive-zero-shot.csv").set_index("dataset")


def load_lines():
    c2zs = pd.read_csv(ROOT / "Chronos2" / "results" / "zeroshot_chronos2_results.csv")
    t5zs = pd.read_csv(ROOT / "Chronos_benchmark" / "results" / "zeroshot_official_results.csv").set_index("dataset")
    return {
        "C2 zs-CL":      c2zs[c2zs["mode"] == "cross_learning"].set_index("dataset"),
        "C2 zs-uni":     c2zs[c2zs["mode"] == "univariate"].set_index("dataset"),
        "T5 zs":         t5zs,
        "C2 1s-CLtrain": pd.read_csv(RESULTS / "oneshot_cltrain_c2.csv").set_index("dataset"),
        "C2 1s-CLeval":  pd.read_csv(RESULTS / "oneshot_hpo_c2_crosslearning.csv").set_index("dataset"),
        "C2 1s-uni":     pd.read_csv(RESULTS / "oneshot_hpo_c2.csv").set_index("dataset"),
        "T5 1s":         pd.read_csv(RESULTS / "oneshot_hpo_t5.csv").set_index("dataset"),
    }


def rel_frame(lines, metric):
    """Per-dataset score / Seasonal-Naive, one column per setting; datasets common to all."""
    common = set(SN.index)
    for f in lines.values():
        common &= set(f.index)
    common = sorted(common)
    rel = pd.DataFrame({name: f.loc[common, metric] / SN.loc[common, metric] for name, f in lines.items()})
    agg = pd.Series({name: gmean(rel[name].values) for name in rel.columns}, name="AGGREGATE (gmean)")
    return rel, agg


def draw(metric, fname):
    lines = load_lines()
    rel, agg = rel_frame(lines, metric)
    order = agg.sort_values().index.tolist()          # columns best -> worst by aggregate
    rel = rel[order]; agg = agg[order]
    mat = pd.concat([rel, agg.to_frame().T])          # datasets + aggregate row at the bottom
    vals = mat.values                                 # (26, 7)
    nrow, ncol = vals.shape

    # colour by rank within each row (1=best .. 7=worst) -> normalise to [0,1] for RdYlGn_r
    ranks = np.argsort(np.argsort(vals, axis=1), axis=1)          # 0=best
    color = ranks / (ncol - 1)

    fig, ax = plt.subplots(figsize=(8.6, 12.2))
    ax.imshow(color, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    for i in range(nrow):
        for j in range(ncol):
            ax.text(j, i, f"{vals[i, j]:.3f}", ha="center", va="center", fontsize=7.5,
                    fontweight="bold" if ranks[i, j] == 0 else "normal")
    ax.set_xticks(range(ncol)); ax.set_xticklabels(order, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(nrow)); ax.set_yticklabels(mat.index, fontsize=8)
    # separate the aggregate row with a line
    ax.axhline(nrow - 1.5, color="black", lw=1.6)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_title(f"{metric}  relative to Seasonal-Naive (lower = better)\n"
                 f"cell = score  |  colour = rank within row (green best -> red worst)  |  "
                 f"{len(rel)} datasets, columns sorted by aggregate", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return rel, agg, order


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    rel_w, agg_w, order = draw("WQL", "summary_matrix_wql.png")
    rel_m, agg_m, _ = draw("MASE", "summary_matrix_mase.png")

    # combined per-dataset CSV (both metrics, all lines)
    out = pd.concat({"WQL": rel_w, "MASE": rel_m}, axis=1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS / "summary_matrix.csv")

    print("Aggregate (gmean rel. Seasonal-Naive), WQL order:")
    for i, name in enumerate(order):
        print(f"  {i+1}. {name:14s} WQL={agg_w[name]:.3f}  MASE={agg_m[name]:.3f}")
    print(f"\nSaved -> plots/summary_matrix_{{wql,mase}}.png  +  summary_matrix.csv")


if __name__ == "__main__":
    main()