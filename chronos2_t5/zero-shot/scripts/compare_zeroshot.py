"""Chronos-2 vs Chronos-T5 — zero-shot head-to-head on Chronos Benchmark II.

This is the *comparison* project: it does not run any model itself, it reads the
per-dataset results already produced by the two sibling projects and builds the
head-to-head artifacts:

  inputs (read-only):
    ../../Chronos2/results/zeroshot_chronos2_results.csv          (Chronos-2: uni + cross-learning)
    ../../Chronos_benchmark/results/zeroshot_official_results.csv (Chronos-T5 measured, cap=1000)
    ../../Chronos_benchmark/reference/seasonal-naive-zero-shot.csv (relative-score denominator)
    ../../Chronos_benchmark/reference/chronos-t5-small-zero-shot.csv (T5 paper reference)

  outputs (this folder):
    c2_vs_t5_dashboard.png          accuracy scatter + aggregate score + speed + memory
    CHRONOS2_VS_T5_HEADTOHEAD.md    aggregated relative score + per-dataset + efficiency

Both projects use the IDENTICAL gluonts pipeline / 25 datasets / cap=1000 / bf16, so
the comparison is apples-to-apples. Chronos-T5 is univariate-only (a single result);
the two Chronos-2 series are its univariate and full-cross-learning modes.

Run:  python compare_zeroshot.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from scipy.stats import gmean

HERE = Path(__file__).resolve().parent          # .../zero-shot/scripts
ZS = HERE.parent                                 # .../zero-shot (output root)
ROOT = HERE.parents[2]                            # .../SAUDI_ARAMCO (repo root)
C2 = ROOT / "Chronos2"
BENCH = ROOT / "Chronos_benchmark"

# --- load the two projects' per-dataset results + references ---
c2 = pd.read_csv(C2 / "results" / "zeroshot_chronos2_results.csv")
uni = c2[c2["mode"] == "univariate"].set_index("dataset")
xl = c2[c2["mode"] == "cross_learning"].set_index("dataset")
t5 = pd.read_csv(BENCH / "results" / "zeroshot_official_results.csv").set_index("dataset")
base = pd.read_csv(BENCH / "reference" / "seasonal-naive-zero-shot.csv").set_index("dataset")[["MASE", "WQL"]]
t5p = pd.read_csv(BENCH / "reference" / "chronos-t5-small-zero-shot.csv").set_index("dataset")[["MASE", "WQL"]]
common = sorted(uni.index.intersection(xl.index).intersection(t5.index).intersection(base.index))

# Chronos-2 paper Benchmark II skill scores (arXiv:2510.15821, Table 5) -> G = 1 - skill/100
C2_PAPER = pd.Series({"MASE": 1 - 0.265, "WQL": 1 - 0.466})


def agg(df):
    return (df.loc[common, ["MASE", "WQL"]] / base.loc[common]).apply(gmean)


def write_dashboard():
    a_uni, a_xl, a_t5 = agg(uni), agg(xl), agg(t5)
    headline = pd.DataFrame({"C2 univariate": a_uni, "C2 cross-learning": a_xl,
                             "T5 measured": a_t5}).T[["WQL", "MASE"]]
    t5m = t5[["MASE", "WQL"]]
    modes = [("C2 univariate", uni, "C0", "o"), ("C2 cross-learning", xl, "C3", "^")]

    fig = plt.figure(figsize=(17, 12.5))
    gs = GridSpec(2, 6, figure=fig, hspace=0.55, wspace=1.4)
    axM = fig.add_subplot(gs[0, 0:3]); axW = fig.add_subplot(gs[0, 3:6])
    axA = fig.add_subplot(gs[1, 0:2]); axT = fig.add_subplot(gs[1, 2:4]); axP = fig.add_subplot(gs[1, 4:6])
    TITLE, LBL, TICK, LEG = 15, 13, 12, 11

    for a, metric in [(axM, "MASE"), (axW, "WQL")]:
        series = [m[1].loc[common, metric] for m in modes] + [t5m.loc[common, metric]]
        lo = max(min(s.min() for s in series) * 0.7, 1e-3); hi = max(s.max() for s in series) * 1.4
        for mname, mdf, col, mk in modes:
            y, x = mdf.loc[common, metric], t5m.loc[common, metric]
            a.scatter(x, y, s=54, marker=mk, alpha=0.8, zorder=3, color=col,
                      label=f"{mname} vs T5  ({int((y < x).sum())}/{len(common)} better)")
        a.plot([lo, hi], [lo, hi], "k--", lw=1); a.set_xscale("log"); a.set_yscale("log")
        a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_aspect("equal")
        a.set_xlabel(f"Chronos-T5 {metric}", fontsize=LBL); a.set_ylabel(f"Chronos-2 {metric}", fontsize=LBL)
        a.set_title(f"({'a' if metric == 'MASE' else 'b'}) {metric}: C2 vs T5", fontsize=TITLE)
        a.tick_params(labelsize=TICK)
        a.legend(fontsize=LEG, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=True)

    runs = list(headline.index); xa = np.arange(len(runs)); w = 0.38
    axA.bar(xa - w / 2, headline["WQL"], w, label="WQL"); axA.bar(xa + w / 2, headline["MASE"], w, label="MASE")
    axA.axhline(1.0, color="grey", lw=0.8, ls=":"); axA.set_xticks(xa)
    axA.set_xticklabels(runs, rotation=25, ha="right", fontsize=TICK); axA.tick_params(axis="y", labelsize=TICK)
    axA.set_ylabel("relative score (vs Seasonal-Naive)", fontsize=LBL)
    axA.set_title("(c) Aggregated relative score", fontsize=TITLE); axA.legend(fontsize=LEG)

    eff = {"C2 univariate":     (uni.loc[common, "latency_s"].sum(), uni.loc[common, "peak_mem_mb"].max()),
           "C2 cross-learning": (xl.loc[common, "latency_s"].sum(),  xl.loc[common, "peak_mem_mb"].max()),
           "T5 measured":       (t5.loc[common, "latency_s"].sum(),  t5.loc[common, "peak_mem_mb"].max())}
    er = list(eff); xe = np.arange(len(er))
    axT.bar(xe, [eff[k][0] for k in er], color="C2"); axT.set_xticks(xe)
    axT.set_xticklabels(er, rotation=25, ha="right", fontsize=TICK); axT.tick_params(axis="y", labelsize=TICK)
    axT.set_ylabel("seconds", fontsize=LBL); axT.set_title("(d) Total forecast time", fontsize=TITLE)
    axP.bar(xe, [eff[k][1] for k in er], color="C4"); axP.set_xticks(xe)
    axP.set_xticklabels(er, rotation=25, ha="right", fontsize=TICK); axP.tick_params(axis="y", labelsize=TICK)
    axP.set_ylabel("MB", fontsize=LBL); axP.set_title("(e) Peak GPU memory", fontsize=TITLE)

    (ZS / "headtohead").mkdir(parents=True, exist_ok=True)
    out = ZS / "headtohead" / "c2_vs_t5_dashboard.png"; fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)
    return headline


def write_headtohead(headline):
    a_uni, a_xl, a_t5 = agg(uni), agg(xl), agg(t5)
    a_t5p = (t5p.loc[common] / base.loc[common]).apply(gmean)
    um, xm = uni[["MASE", "WQL"]], xl[["MASE", "WQL"]]
    n = len(common)
    xl_w_mase = int((xm.loc[common, "MASE"] < t5.loc[common, "MASE"]).sum())
    xl_w_wql = int((xm.loc[common, "WQL"] < t5.loc[common, "WQL"]).sum())

    def eff_agg(df): return df.loc[common, "latency_s"].sum(), df.loc[common, "peak_mem_mb"].max()
    lu, lx, lt = eff_agg(uni), eff_agg(xl), eff_agg(t5)

    md = [
        "# Chronos-2 vs Chronos-T5 — zero-shot head-to-head (Benchmark II)\n",
        f"Same 25 datasets, same cap=1000, same gluonts pipeline (MASE + WQL), same machine, "
        f"both bf16. Chronos-T5 is univariate-only; C2 has univariate & full cross-learning.\n",
        "## Aggregated relative score (gmean of model / Seasonal-Naive; lower is better)\n",
        "| metric | C2 uni | C2 cross-learning | C2 paper¹ | T5 measured | T5 paper |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| WQL  | {a_uni['WQL']:.3f} | **{a_xl['WQL']:.3f}** | {C2_PAPER['WQL']:.3f} | {a_t5['WQL']:.3f} | {a_t5p['WQL']:.3f} |",
        f"| MASE | {a_uni['MASE']:.3f} | **{a_xl['MASE']:.3f}** | {C2_PAPER['MASE']:.3f} | {a_t5['MASE']:.3f} | {a_t5p['MASE']:.3f} |",
        "\n¹ `C2 paper` = Chronos-2 paper (arXiv:2510.15821, Table 5) skill scores via G=1-skill/100; "
        "aggregate over all 27 datasets/full data (ours: 25 + cap=1000).",
        f"\n**Win rate vs T5 measured (per-dataset):** C2 cross-learning MASE {xl_w_mase}/{n}, WQL {xl_w_wql}/{n}.\n",
        "## Inference efficiency (same machine, both bf16; lower is better)\n",
        "| run | total forecast time (s) | mean ms/series | peak GPU mem (MB) |",
        "| --- | --- | --- | --- |",
        f"| C2 univariate | {lu[0]:.1f} | {uni.loc[common,'ms_per_series'].mean():.1f} | {lu[1]:.0f} |",
        f"| C2 cross-learning | {lx[0]:.1f} | {xl.loc[common,'ms_per_series'].mean():.1f} | {lx[1]:.0f} |",
        f"| T5 measured | {lt[0]:.1f} | {t5.loc[common,'ms_per_series'].mean():.1f} | {lt[1]:.0f} |",
        f"\nC2 cross-learning vs T5: **{lt[0]/lx[0]:.1f}x** less total time, **{lt[1]/lx[1]:.1f}x** less peak memory.\n",
        "## Per-dataset MASE\n",
        "| dataset | C2 uni | C2 xl | T5 measured |",
        "| --- | --- | --- | --- |",
    ]
    for d in common:
        md.append(f"| {d} | {um.loc[d,'MASE']:.4f} | {xm.loc[d,'MASE']:.4f} | {t5.loc[d,'MASE']:.4f} |")
    md += ["\n## Per-dataset WQL\n", "| dataset | C2 uni | C2 xl | T5 measured |", "| --- | --- | --- | --- |"]
    for d in common:
        md.append(f"| {d} | {um.loc[d,'WQL']:.4f} | {xm.loc[d,'WQL']:.4f} | {t5.loc[d,'WQL']:.4f} |")

    (ZS / "headtohead").mkdir(parents=True, exist_ok=True)
    out = ZS / "headtohead" / "CHRONOS2_VS_T5_HEADTOHEAD.md"; out.write_text("\n".join(md), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    print(f"datasets compared: {len(common)}")
    hl = write_dashboard()
    write_headtohead(hl)
    print("\nAggregated relative score:")
    print(hl.round(3).to_string())
