"""E9 analysis -- what does Chronos-2's cross-learning actually transmit?

Reads `crosslearning_shuffle.csv` and resolves the 2x2 the experiment was designed for.
Gain is defined per dataset as

    gain = metric(univariate) / metric(cross-learning)        > 1 means grouping helped

measured under three sibling conditions: the target's own dataset (`native`), foreign
datasets at the same frequency (`foreign_samefreq`, arm A), and foreign datasets at a
different frequency (`foreign_difffreq`, arm B).

    gain survives A and B  ->  generic pooling of level/scale; sibling identity irrelevant
    survives A, dies in B  ->  shared seasonality is the active ingredient
    dies in A and B        ->  a dataset-level prior, not pooling

Datasets whose same-frequency sibling pool held fewer than 99 series are absent from arm
A by construction; they are reported rather than silently dropped.

Analysis unit is the dataset (n = 25); paired Wilcoxon signed-rank across datasets, with
Holm correction over the family of arm contrasts and bootstrap CIs resampling datasets.

Outputs (results/): fig_cl_shuffle_<metric>.png, CL_SHUFFLE_ANALYSIS_<metric>.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gmean, wilcoxon, binomtest

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
ARMS = ["native", "foreign_samefreq", "foreign_difffreq"]
NBOOT = 2000
RNG = np.random.default_rng(0)


def holm(ps):
    ps = np.asarray(ps, float)
    m, order, adj, run = len(ps), np.argsort(ps), np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(run, 1.0)
    return adj


def boot_ci(x, stat=gmean, n=NBOOT):
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < 3:
        return (np.nan, np.nan)
    d = [stat(x[RNG.integers(0, x.size, x.size)]) for _ in range(n)]
    return tuple(np.percentile(d, [2.5, 97.5]))


def load(metric: str) -> pd.DataFrame:
    df = pd.read_csv(OUT / "crosslearning_shuffle.csv")
    piv = df.pivot_table(index=["dataset", "freq"], columns="arm", values=metric)
    piv = piv.reset_index()
    for arm in ARMS:
        if arm in piv:
            piv[f"gain_{arm}"] = piv["univariate"] / piv[arm]
    return piv


def main():
    metric = sys.argv[sys.argv.index("--metric") + 1] if "--metric" in sys.argv else "WQL"
    g = load(metric)
    present = [a for a in ARMS if f"gain_{a}" in g]
    print(f"{len(g)} datasets; arms present: {present}\n")

    # ---- per-arm gain -----------------------------------------------------
    rows = []
    for arm in present:
        v = g[f"gain_{arm}"].dropna()
        lo, hi = boot_ci(v)
        wins = int((v > 1).sum())
        bt = binomtest(wins, len(v), 0.5, alternative="greater")
        rows.append({"arm": arm, "n": len(v), "gain_gmean": gmean(v),
                     "ci_lo": lo, "ci_hi": hi, "datasets_helped": f"{wins}/{len(v)}",
                     "sign_p": bt.pvalue})
    per_arm = pd.DataFrame(rows)
    print("Gain per sibling condition (>1 = grouping helped)")
    print(per_arm.round(4).to_string(index=False))

    # ---- paired arm contrasts --------------------------------------------
    contrasts, ps = [], []
    for a, b in (("native", "foreign_samefreq"), ("native", "foreign_difffreq"),
                 ("foreign_samefreq", "foreign_difffreq")):
        if f"gain_{a}" not in g or f"gain_{b}" not in g:
            continue
        j = g[[f"gain_{a}", f"gain_{b}"]].dropna()
        w = wilcoxon(j[f"gain_{a}"], j[f"gain_{b}"])          # two-sided
        contrasts.append({"contrast": f"{a} vs {b}", "n": len(j),
                          "median_diff": float((j[f"gain_{a}"] - j[f"gain_{b}"]).median()),
                          "p": float(w.pvalue)})
        ps.append(w.pvalue)
    C = pd.DataFrame(contrasts)
    if len(C):
        C["p_holm"] = holm(ps)
        print("\nPaired contrasts between sibling conditions (dataset as unit)")
        print(C.round(4).to_string(index=False))

    # ---- verdict ----------------------------------------------------------
    # Two questions, not one. Whether an arm's own CI clears 1 tells you if that sibling
    # set is sufficient on its own; the A-vs-B contrast tells you whether frequency
    # matching contributes. Reading only the first (an earlier version of this function)
    # collapses a three-rung ladder into a single label and loses the contrast entirely.
    def clears_one(arm):
        if f"gain_{arm}" not in g:
            return None
        lo, _ = boot_ci(g[f"gain_{arm}"].dropna())
        return bool(lo > 1.0)

    suff = {a: clears_one(a) for a in present}
    ab = next((c for c in contrasts
               if c["contrast"] == "foreign_samefreq vs foreign_difffreq"), None)
    ab_p = C.loc[C.contrast == "foreign_samefreq vs foreign_difffreq", "p_holm"].iloc[0] \
        if len(C) and (C.contrast == "foreign_samefreq vs foreign_difffreq").any() else np.nan

    parts = [
        f"native sufficient on its own: {suff.get('native')}",
        f"foreign same-frequency sufficient: {suff.get('foreign_samefreq')}",
        f"foreign different-frequency sufficient: {suff.get('foreign_difffreq')}",
    ]
    if ab is not None:
        direction = "same-freq > diff-freq" if ab["median_diff"] > 0 else "diff-freq > same-freq"
        sig = "significant" if ab_p < 0.05 else "not significant"
        parts.append(f"frequency matching contributes: {direction}, Holm p={ab_p:.3g} ({sig})")
    verdict = "; ".join(parts)
    print(f"\nVERDICT ({metric}): {verdict}")

    # ---- figure -----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    xs = np.arange(len(present))
    for i, arm in enumerate(present):
        v = g[f"gain_{arm}"].dropna()
        ax.scatter(np.full(len(v), i) + RNG.normal(0, 0.05, len(v)), v,
                   s=26, alpha=0.55, color="C0", zorder=3)
        m = gmean(v)
        lo, hi = boot_ci(v)
        ax.plot([i - 0.22, i + 0.22], [m, m], color="C3", lw=2.6, zorder=4)
        ax.plot([i, i], [lo, hi], color="C3", lw=1.6, zorder=4)
    ax.axhline(1.0, color="grey", ls=":", lw=1.3)
    ax.set_xticks(xs)
    ax.set_xticklabels([a.replace("_", "\n") for a in present], fontsize=11)
    ax.set_ylabel(f"gain = univariate {metric} / cross-learning {metric}", fontsize=12)
    ax.set_title("What do the group-mates have to be?", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = OUT / f"fig_cl_shuffle_{metric}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)

    lines = [f"# E9 -- cross-learning sibling shuffle ({metric})", "",
             f"Gain = univariate {metric} divided by cross-learning {metric}; above 1 "
             "means grouping helped. Analysis unit is the dataset; CIs bootstrap the "
             f"datasets ({NBOOT} draws); contrasts are paired Wilcoxon with Holm "
             "correction.", "",
             "## Gain per sibling condition", "", per_arm.round(4).to_markdown(index=False), ""]
    if len(C):
        lines += ["## Paired contrasts", "", C.round(4).to_markdown(index=False), ""]
    missing = sorted(set(g.dataset) - set(g.dropna(subset=["gain_foreign_samefreq"]).dataset)) \
        if "gain_foreign_samefreq" in g else []
    if missing:
        lines += ["## Datasets absent from arm A", "",
                  "Their same-frequency sibling pool held fewer than 99 series: "
                  + ", ".join(f"`{d}`" for d in missing) + ".", ""]
    lines += [f"## Verdict", "", verdict, "", f"![shuffle]({p.name})", ""]
    md = OUT / f"CL_SHUFFLE_ANALYSIS_{metric}.md"
    md.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"\n-> {p}\n-> {md}")


if __name__ == "__main__":
    main()
