"""Chronos-2-style benchmark leaderboard on Chronos Benchmark II (our reproduction).

Mirrors the Chronos-2 technical report's results tables (arXiv:2510.15821, Tables 3/5):
columns = Avg Win Rate (%), Skill Score (%), Median runtime (s), Leakage (%), #Failures,
computed with respect to WQL and MASE (the Benchmark II metrics), over the 25 datasets.

Definitions (from the report):
  * Skill Score  S = 100 * (1 - G),  G = geometric-mean relative error = gmean(model / Seasonal-Naive).
    (Seasonal-Naive therefore has skill 0, exactly as in the report's tables.)
  * Avg Win Rate W = 100 * (fraction of pairwise (task, other-model) comparisons the model wins);
    equivalent to average rank via R = 1 + (1 - W/100)(N-1). Ties count as half a win.
  * Median runtime = median per-dataset forecast wall-time (s); Seasonal-Naive is a reference (n/a).
  * Leakage = 0 for every model (nothing here pretrains on Benchmark II; one-shot holds out the
    eval window). #Failures = datasets with a non-finite MASE or WQL.

Compared models (all on the identical 25-dataset test split):
  zero-shot:  Chronos-2 CL, Chronos-2 uni, Chronos-T5, Seasonal-Naive
  one-shot :  Chronos-2 CLtrain, Chronos-2 CL-eval, Chronos-2 uni, Chronos-T5

Outputs (benchmark/):
  benchmark_leaderboard.csv        all metrics, all models
  BENCHMARK_LEADERBOARD.md         two tables (WQL, MASE), Chronos-2 table layout
  winrate_wql.png / winrate_mase.png   avg win-rate bar charts
Run:  python benchmark_table.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gmean

HERE = Path(__file__).resolve().parent          # .../zero-shot/scripts
ZS = HERE.parent                                 # .../zero-shot (output root)
ROOT = HERE.parents[2]                            # repo root
OUT = ZS / "benchmark"
ONESHOT = ROOT / "chronos2_t5" / "one-shot" / "results"
SN = pd.read_csv(ROOT / "Chronos_benchmark" / "reference" / "seasonal-naive-zero-shot.csv").set_index("dataset")


def load_models():
    c2zs = pd.read_csv(ROOT / "Chronos2" / "results" / "zeroshot_chronos2_results.csv")
    t5zs = pd.read_csv(ROOT / "Chronos_benchmark" / "results" / "zeroshot_official_results.csv").set_index("dataset")
    def one(name): return pd.read_csv(ONESHOT / name).set_index("dataset")
    sn = SN.copy(); sn["latency_s"] = np.nan                       # baseline: no runtime of our own
    return [
        ("Chronos-2 (zs, CL)",       c2zs[c2zs["mode"] == "cross_learning"].set_index("dataset")),
        ("Chronos-2 (zs, uni)",      c2zs[c2zs["mode"] == "univariate"].set_index("dataset")),
        ("Chronos-T5 (zs)",          t5zs),
        ("Chronos-2 (1s, CLtrain)",  one("oneshot_cltrain_c2.csv")),
        ("Chronos-2 (1s, CL-eval)",  one("oneshot_hpo_c2_crosslearning.csv")),
        ("Chronos-2 (1s, uni)",      one("oneshot_hpo_c2.csv")),
        ("Chronos-T5 (1s)",          one("oneshot_hpo_t5.csv")),
        ("Seasonal Naive",           sn),
    ]


def pairwise_winrate(mat, n_boot=2000, seed=0):
    """Pairwise win-rate matrix + 95% bootstrap CIs (Chronos-2 Fig. 12 style).

    mat: (n_models, n_tasks), lower better. WR[i,j] = 100 * fraction of tasks model i beats
    model j (ties = 0.5); diagonal = 50. CIs by resampling the tasks (datasets) with replacement.
    Assumes finite entries (our #Failures = 0)."""
    n, T = mat.shape
    A, B = mat[:, None, :], mat[None, :, :]                 # (n,1,T),(1,n,T)
    win = (A < B).astype(float) + 0.5 * (A == B)            # (n,n,T) win of i over j per task
    WR = 100 * win.mean(axis=2)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, n, n))
    for b in range(n_boot):
        idx = rng.integers(0, T, T)                         # resample tasks with replacement
        boots[b] = 100 * win[:, :, idx].mean(axis=2)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    for M in (WR, lo, hi):
        np.fill_diagonal(M, 50.0)
    return WR, lo, hi


def plot_pairwise(WR, lo, hi, names, metric, out):
    """Heatmap of the pairwise win-rate matrix with CIs (green=high, purple=low), like Fig. 12."""
    n = len(names)
    fig, ax = plt.subplots(figsize=(1.05 * n + 2.2, 1.0 * n + 1.8))
    ax.imshow(WR, cmap="PRGn", vmin=0, vmax=100, aspect="auto")
    for i in range(n):
        for j in range(n):
            v = WR[i, j]
            txt = f"{v:.0f}" if i == j else f"{v:.0f}\n({lo[i, j]:.0f},{hi[i, j]:.0f})"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                    color="white" if (v > 78 or v < 22) else "black")
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=40, ha="left", fontsize=8)
    ax.xaxis.set_ticks_position("top"); ax.xaxis.set_label_position("top")
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Model 2  (opponent)", fontsize=9); ax.set_ylabel("Model 1", fontsize=9)
    ax.set_title(f"Pairwise win rate (%) w.r.t. {metric} with 95% CIs\n"
                 f"cell = row beats column; green = wins, purple = loses; diagonal = 50", fontsize=10, pad=28)
    cb = fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04); cb.set_label("Win rate (%)", fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


def win_rate(mat):
    """mat: (n_models, n_tasks) metric matrix (lower better). Returns win rate % per model.
    Pairwise over every (task, other-model); ties = 0.5; comparisons with a NaN are skipped."""
    n = mat.shape[0]
    W = np.zeros(n)
    for i in range(n):
        wins = comps = 0.0
        for j in range(n):
            if i == j:
                continue
            a, b = mat[i], mat[j]
            ok = np.isfinite(a) & np.isfinite(b)
            comps += ok.sum()
            wins += (a[ok] < b[ok]).sum() + 0.5 * (a[ok] == b[ok]).sum()
        W[i] = 100 * wins / comps if comps else np.nan
    return W


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    models = load_models()
    common = set(SN.index)
    for _, f in models:
        common &= set(f.index)
    common = sorted(common)
    names = [n for n, _ in models]

    rows = {}
    winrates = {}
    mats = {}
    for metric in ["WQL", "MASE"]:
        mat = np.vstack([f.loc[common, metric].to_numpy(float) for _, f in models])   # (n_models, n_tasks)
        mats[metric] = mat
        W = win_rate(mat)
        winrates[metric] = pd.Series(W, index=names)
        sn_vec = SN.loc[common, metric].to_numpy(float)
        for k, (name, f) in enumerate(models):
            rel = mat[k] / sn_vec
            rel = rel[np.isfinite(rel) & (rel > 0)]
            skill = 100 * (1 - gmean(rel)) if rel.size else np.nan
            rows.setdefault(name, {})[f"WinRate_{metric}"] = round(W[k], 1)
            rows[name][f"Skill_{metric}"] = round(skill, 1)

    # runtime / leakage / failures (metric-independent)
    for name, f in models:
        lat = f.loc[common, "latency_s"] if "latency_s" in f.columns else pd.Series(np.nan, index=common)
        finite_bad = 0
        for m in ["WQL", "MASE"]:
            v = f.loc[common, m].to_numpy(float)
            finite_bad += int((~np.isfinite(v)).sum())
        rows[name]["MedianRuntime_s"] = round(float(np.nanmedian(lat)), 2) if lat.notna().any() else np.nan
        rows[name]["Leakage_pct"] = 0
        rows[name]["Failures"] = finite_bad

    df = pd.DataFrame(rows).T
    df = df.sort_values("WinRate_WQL", ascending=False)
    df.to_csv(OUT / "benchmark_leaderboard.csv")

    # ---- markdown: one table per metric, Chronos-2 Table 5 layout ----
    md = [f"# Benchmark II leaderboard (our reproduction, {len(common)} datasets)\n",
          "Chronos-2-report layout (arXiv:2510.15821 Tables 3/5). Win rate & skill score are with "
          "respect to each metric; higher is better for both. Leakage = 0 (no Benchmark-II pretraining; "
          "one-shot holds out the eval window). #Failures = datasets with non-finite MASE/WQL.\n"]
    for metric in ["WQL", "MASE"]:
        sub = df.sort_values(f"WinRate_{metric}", ascending=False)
        md += [f"## {metric}\n",
               "| Model | Avg Win Rate (%) | Skill Score (%) | Median runtime (s) | Leakage (%) | #Failures |",
               "| --- | --- | --- | --- | --- | --- |"]
        for name, r in sub.iterrows():
            rt = "—" if pd.isna(r["MedianRuntime_s"]) else f"{r['MedianRuntime_s']:.2f}"
            md.append(f"| {name} | {r[f'WinRate_{metric}']:.1f} | {r[f'Skill_{metric}']:.1f} | "
                      f"{rt} | {int(r['Leakage_pct'])} | {int(r['Failures'])} |")
        md.append("")
    (OUT / "BENCHMARK_LEADERBOARD.md").write_text("\n".join(md), encoding="utf-8")

    # ---- win-rate bar charts (one per metric) ----
    def fam_color(name):
        if name.startswith("Chronos-2"): return "#c0392b" if "CL" in name else "#e07b6a"  # C2: CL darker
        if name.startswith("Chronos-T5"): return "#2980b9"
        return "#9aa0a6"                                                                    # baseline grey
    for metric in ["WQL", "MASE"]:
        s = winrates[metric].sort_values()
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        ax.barh(range(len(s)), s.values, color=[fam_color(n) for n in s.index])
        for i, v in enumerate(s.values):
            ax.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=8)
        ax.set_yticks(range(len(s))); ax.set_yticklabels(s.index, fontsize=9)
        ax.set_xlabel(f"Avg win rate (%) w.r.t. {metric}  (higher = better)")
        ax.set_title(f"Benchmark II — average win rate ({metric})\n"
                     f"C2 = red, T5 = blue, baseline = grey; {len(common)} datasets, {len(names)} models")
        ax.set_xlim(0, 100)
        fig.tight_layout(); fig.savefig(OUT / f"winrate_{metric.lower()}.png", dpi=150); plt.close(fig)

    # ---- pairwise win-rate matrices with 95% CIs (Chronos-2 Fig. 12 style), best->worst order ----
    for metric in ["WQL", "MASE"]:
        order = winrates[metric].sort_values(ascending=False).index.tolist()
        pos = [names.index(n) for n in order]
        WR, lo, hi = pairwise_winrate(mats[metric][pos])
        plot_pairwise(WR, lo, hi, order, metric, OUT / f"pairwise_winrate_{metric.lower()}.png")

    print(f"datasets={len(common)}  models={len(names)}")
    print(df[["WinRate_WQL", "Skill_WQL", "WinRate_MASE", "Skill_MASE", "MedianRuntime_s", "Failures"]].to_string())
    print(f"\nSaved -> {OUT}/ (leaderboard.csv, MD, winrate_wql.png, winrate_mase.png)")


if __name__ == "__main__":
    main()