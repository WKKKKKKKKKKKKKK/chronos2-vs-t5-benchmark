"""E2a analysis -- is Chronos-T5's flat severity slope sampling noise?

Reads `decode_ablation.csv`: the spike-magnitude sweep re-run under 8 decode seeds with the
corruption held FIXED at seed 0, so the only thing that moves between runs is
`torch.manual_seed` for Chronos-T5's 20-sample decode.

Three questions.

  Q1  How much does the slope move when only the decode moves? Per-decode-seed Spearman
      rho between severity and degradation. A wide spread means decode noise is a
      substantial part of what the single-seed curves were showing.

  Q2  THE DECISIVE ONE. Average the degradation over the 8 decode seeds first -- which cuts
      decode-induced noise by about sqrt(8) ~ 2.8x -- and recompute the slope. If the slope
      turns positive, the flat reading was sampling noise and Chronos-T5 does track
      magnitude after all. If it stays flat, decode is excluded alongside the two
      representation mechanisms E1 already rejected.

  Q3  A variance decomposition the study currently lacks. Compare the spread of degradation
      across DECODE seeds (corruption fixed, this run) against the spread across CORRUPTION
      seeds (both moving, `edge_case_seeds.csv`). The ratio says how much of the
      curve-to-curve variation is the sampler and how much is where the spikes landed.

Analysis unit is the dataset (n = 25); rho is computed per curve and averaged within
dataset; CIs bootstrap the datasets. Same conventions as `statistics.py`.

Outputs (results/): fig_decode_ablation_<metric>.png, DECODE_ABLATION_<metric>.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
FAMILY = "spikes_intensity"
NBOOT = 2000
RNG = np.random.default_rng(0)


def boot_ci(x, n=NBOOT):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return (np.nan, np.nan)
    d = [x[RNG.integers(0, x.size, x.size)].mean() for _ in range(n)]
    return tuple(np.percentile(d, [2.5, 97.5]))


def degr(df: pd.DataFrame, metric: str, seedcol: str) -> pd.DataFrame:
    """Degradation against the clean score from the SAME run, per (seed, dataset)."""
    clean = (df[df.family == "clean"]
             .set_index([seedcol, "model", "dataset"])[metric].rename("clean"))
    d = df[df.family == FAMILY].join(clean, on=[seedcol, "model", "dataset"])
    d["degr"] = d[metric] / d["clean"]
    return d


def rho_per_curve(d: pd.DataFrame, seedcol: str) -> pd.DataFrame:
    rows = []
    for (s, ds), g in d.groupby([seedcol, "dataset"]):
        g = g.dropna(subset=["degr"])
        if g.severity.nunique() < 3:
            continue
        rows.append({seedcol: s, "dataset": ds,
                     "rho": spearmanr(g.severity, g.degr).statistic})
    return pd.DataFrame(rows)


def main():
    metric = sys.argv[sys.argv.index("--metric") + 1] if "--metric" in sys.argv else "WQL"
    da = pd.read_csv(OUT / "decode_ablation.csv")
    d = degr(da, metric, "decode_seed")
    nseeds = d.decode_seed.nunique()
    print(f"metric = {metric};  {d.dataset.nunique()} datasets x {nseeds} decode seeds "
          f"x {d.severity.nunique()} severities;  corruption fixed at seed 0\n")

    # ---- Q1: slope per decode seed -----------------------------------------
    per = rho_per_curve(d, "decode_seed")
    q1 = (per.groupby("decode_seed").rho
             .agg(["mean", "median", "std"]).round(4))
    print("Q1 -- slope per decode seed (mean over the 25 datasets)")
    print(q1.to_string())
    print(f"  spread of the per-seed mean rho: "
          f"{per.groupby('decode_seed').rho.mean().min():+.3f} to "
          f"{per.groupby('decode_seed').rho.mean().max():+.3f}")

    # ---- Q2: slope after averaging decode noise down -----------------------
    avg = (d.groupby(["dataset", "severity"], as_index=False).degr.mean())
    rows = []
    for ds, g in avg.groupby("dataset"):
        rows.append({"dataset": ds,
                     "rho": spearmanr(g.severity, g.degr).statistic})
    q2 = pd.DataFrame(rows)
    lo, hi = boot_ci(q2.rho)
    # baseline: the same quantity WITHOUT decode averaging, i.e. a single decode seed,
    # averaged over seeds so the comparison is not a lucky draw
    base = per.groupby("dataset", as_index=False).rho.mean()
    blo, bhi = boot_ci(base.rho)
    w = wilcoxon(q2.set_index("dataset").rho.loc[base.dataset],
                 base.set_index("dataset").rho)
    print(f"\nQ2 -- slope after averaging over {nseeds} decode seeds  (THE decisive number)")
    print(f"  decode-averaged : rho = {q2.rho.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
          f"positive on {int((q2.rho > 0).sum())}/{len(q2)} datasets")
    print(f"  single-seed mean: rho = {base.rho.mean():+.4f}  [{blo:+.4f}, {bhi:+.4f}]  "
          f"positive on {int((base.rho > 0).sum())}/{len(base)}")
    print(f"  paired change, n={len(base)}: p = {w.pvalue:.4g}")
    tracks = bool(lo > 0)
    verdict = ("DECODE EXPLAINS IT: averaging the sampler down turns the slope positive; "
               "the flat reading was sampling noise" if tracks else
               "DECODE IS EXCLUDED: the slope stays indistinguishable from zero once "
               "sampling noise is averaged down")
    print(f"\nVERDICT ({metric}): {verdict}")

    # ---- Q3: decode noise vs corruption placement --------------------------
    q3 = None
    sp = OUT / "edge_case_seeds.csv"
    if sp.exists():
        cs = degr(pd.read_csv(sp), metric, "seed")
        cs = cs[cs.model == "chronos-t5"]
        # spread across seeds within (dataset, severity), for each source of variation
        sd_dec = d.groupby(["dataset", "severity"]).degr.std()
        sd_cor = cs.groupby(["dataset", "severity"]).degr.std()
        j = pd.concat([sd_dec.rename("sd_decode_only"),
                       sd_cor.rename("sd_decode_plus_corruption")], axis=1).dropna()
        share = (j.sd_decode_only / j.sd_decode_plus_corruption).replace(
            [np.inf, -np.inf], np.nan).dropna()
        q3 = {"n_cells": len(share), "median_sd_ratio": float(share.median()),
              "median_sd_decode": float(j.sd_decode_only.median()),
              "median_sd_both": float(j.sd_decode_plus_corruption.median())}
        print("\nQ3 -- how much of the curve-to-curve spread is the sampler?")
        print(f"  median s.d. across decode seeds only      : {q3['median_sd_decode']:.4f}")
        print(f"  median s.d. across corruption+decode seeds: {q3['median_sd_both']:.4f}")
        print(f"  ratio (median over {q3['n_cells']} cells)        : "
              f"{q3['median_sd_ratio']:.3f}")
        print("  Note: the corruption-seed run has 3 seeds vs 8 here, so the ratio is a "
              "rough decomposition, not an exact variance split.")

    # ---- figure ------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))

    for s, g in d.groupby("decode_seed"):
        c = g.groupby("severity").degr.mean()
        axes[0].plot(c.index, c.values, lw=0.9, alpha=0.5, color="0.55",
                     label="single decode seed" if s == 0 else None)
    ca = avg.groupby("severity").degr.mean()
    axes[0].plot(ca.index, ca.values, lw=2.6, color="C3", marker="o",
                 label=f"mean over {nseeds} decode seeds")
    axes[0].axhline(1.0, color="grey", ls=":", lw=1.2)
    axes[0].set_xlabel("spike magnitude (x MAD), corruption fixed")
    axes[0].set_ylabel(f"{metric} (x clean)")
    axes[0].set_title("Chronos-T5: does averaging the sampler\nrestore a slope?",
                      fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    for i, (lab, v) in enumerate([("single\ndecode seed", base.rho.to_numpy()),
                                  (f"averaged over\n{nseeds} seeds", q2.rho.to_numpy())]):
        axes[1].scatter(np.full(len(v), i) + RNG.normal(0, 0.05, len(v)), v,
                        s=26, alpha=0.55, color="C0", zorder=3)
        l_, h_ = boot_ci(v)
        axes[1].plot([i - 0.22, i + 0.22], [v.mean()] * 2, color="C3", lw=2.6, zorder=4)
        axes[1].plot([i, i], [l_, h_], color="C3", lw=1.6, zorder=4)
    axes[1].axhline(0.0, color="grey", ls=":", lw=1.2)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["single\ndecode seed", f"averaged over\n{nseeds} seeds"])
    axes[1].set_ylabel(r"Spearman $\rho$(severity, degradation)")
    axes[1].set_title("Per-dataset slope", fontsize=12, fontweight="bold")
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = OUT / f"fig_decode_ablation_{metric}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)

    md = [f"# E2a -- decode-only ablation ({metric})", "",
          "The spike-magnitude sweep re-run under 8 decode seeds with the corruption held "
          "fixed at seed 0. Only `torch.manual_seed` moves, so this is a genuine "
          "single-variable manipulation of the model; the corrupted contexts were "
          "fingerprinted and verified byte-identical across runs.", "",
          "## Q1 -- slope per decode seed", "", q1.to_markdown(), "",
          "## Q2 -- slope after averaging the sampler down", "",
          f"- decode-averaged: rho = {q2.rho.mean():+.4f} [{lo:+.4f}, {hi:+.4f}], "
          f"positive on {int((q2.rho > 0).sum())}/{len(q2)} datasets",
          f"- single decode seed: rho = {base.rho.mean():+.4f} [{blo:+.4f}, {bhi:+.4f}], "
          f"positive on {int((base.rho > 0).sum())}/{len(base)}",
          f"- paired change over {len(base)} datasets: p = {w.pvalue:.4g}", "",
          f"**{verdict}**", ""]
    if q3:
        md += ["## Q3 -- sampler noise vs corruption placement", "",
               f"- median s.d. across decode seeds only: {q3['median_sd_decode']:.4f}",
               f"- median s.d. across corruption+decode seeds: {q3['median_sd_both']:.4f}",
               f"- ratio: {q3['median_sd_ratio']:.3f} (median over {q3['n_cells']} cells)",
               "", "The corruption-seed run supplies 3 seeds against 8 here, so treat this "
               "as an indicative decomposition rather than an exact variance split.", ""]
    md += [f"![decode]({p.name})", ""]
    mdp = OUT / f"DECODE_ABLATION_{metric}.md"
    mdp.write_text("\n".join(md), encoding="utf-8", newline="\n")
    print(f"\n-> {p}\n-> {mdp}")


if __name__ == "__main__":
    main()
