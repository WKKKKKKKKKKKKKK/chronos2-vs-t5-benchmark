"""Are the four small datasets really the most internally homogeneous?

OUTCOME: no, and the claim this was written to check has been REMOVED from the paper as a
result. A draft of E9b explained the drop in own-dataset cross-learning gain by arguing the
four sub-100-series datasets are unusually homogeneous collections, so within-collection
similarity rather than group size is what the gain is made of. Measured here, three of the
four do rank near the top (`ercot` 0.93, `monash_australian_electricity` 0.60,
`exchange_rate` 0.37 against a suite median near 0.05) but the fourth,
`monash_cif_2016`, ranks 23rd of 25 at 0.004 -- and `monash_cif_2016` is precisely the
dataset carrying the effect, with the largest own-dataset gain in the suite (1.395). The
homogeneity explanation is therefore refuted by its own strongest case. The manuscript now
reports the phenomenon without a mechanism.

The script is kept because a removed claim should leave a trace of why it was removed, and
because the per-dataset homogeneity numbers are useful on their own.

Definition. For each dataset, take the clean context of every series, align them at the
forecast origin (the last point each model actually sees) and keep the last L = 512 steps,
which is Chronos-T5's context window. Correlate on first differences rather than levels:
two series that merely share a trend correlate near 1 in levels regardless of whether they
move together, which would score every trending collection as "homogeneous". Report the
mean pairwise Pearson correlation, and the share of pairs above 0.5 as a shape-free check
that the mean is not carried by a few extreme pairs.

Series shorter than 32 usable steps after alignment are skipped; datasets left with fewer
than two series report NaN.

Reads:   the shared dataset registry (no corruption, no GPU)
Writes:  results/HOMOGENEITY.md
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_edge_cases as RE  # noqa: E402

OUT = HERE / "results"
CTX = 512          # Chronos-T5's context window
MIN_STEPS = 16   # yearly collections have short series; 32 excluded them entirely
MAX_PAIRS = 20_000  # cap for the few datasets with 100 series (4950 pairs -- never hit)


def homogeneity(contexts) -> tuple[float, float, int]:
    """Mean pairwise correlation of first differences over each PAIR's overlapping tail.

    Alignment is per pair, not global. An earlier version truncated every series in a
    dataset to the length of its shortest member, so one short series collapsed the usable
    window for the whole collection -- six datasets returned NaN and a yearly collection
    returned no usable series at all. A short series should only cost the pairs it is in.
    """
    arrs = [np.asarray(c, dtype=np.float64)[-CTX:] for c in contexts]
    arrs = [a for a in arrs if np.isfinite(a).sum() >= MIN_STEPS + 1]
    if len(arrs) < 2:
        return (np.nan, np.nan, len(arrs))
    diffs = [np.diff(a) for a in arrs]          # first differences: co-movement, not trend
    cors = []
    for i, j in combinations(range(len(diffs)), 2):
        L = min(diffs[i].size, diffs[j].size)   # overlap for THIS pair, at the origin
        if L < MIN_STEPS:
            continue
        x, y = diffs[i][-L:], diffs[j][-L:]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < MIN_STEPS:
            continue
        xs, ys = x[m], y[m]
        if xs.std() == 0 or ys.std() == 0:
            continue
        cors.append(np.corrcoef(xs, ys)[0, 1])
        if len(cors) >= MAX_PAIRS:
            break
    if not cors:
        return (np.nan, np.nan, len(arrs))
    cors = np.asarray(cors)
    return (float(cors.mean()), float((cors > 0.5).mean()), len(arrs))


def main():
    rows = []
    for ds, horizon in RE.EDGE_DATASETS:
        _, contexts, _ = RE.build_dataset(ds, horizon)
        mean_r, frac_hi, n = homogeneity(contexts)
        rows.append({"dataset": ds, "n_series": n,
                     "mean_pairwise_r": mean_r, "frac_pairs_r_gt_0.5": frac_hi})
        print(f"{ds:32s} n={n:4d}  mean r={mean_r:+.3f}  frac>0.5={frac_hi:.3f}",
              flush=True)

    t = pd.DataFrame(rows).sort_values("mean_pairwise_r", ascending=False)
    t["rank"] = range(1, len(t) + 1)
    small = {"monash_australian_electricity", "ercot", "exchange_rate", "monash_cif_2016"}
    t["sub100"] = t.dataset.isin(small)

    ranks = t.loc[t.sub100, "rank"].tolist()
    verdict = (f"the four sub-100 datasets rank {sorted(ranks)} of {len(t)} by mean "
               f"pairwise correlation")
    print("\n" + t.round(4).to_string(index=False))
    print("\n" + verdict)

    md = ["# Are the four small datasets the most internally homogeneous?", "",
          "E9b attributes the cross-learning gain to within-collection homogeneity rather "
          "than to group size. This measures the homogeneity instead of asserting it.", "",
          f"Mean pairwise Pearson correlation of **first differences** over the last {CTX} "
          "aligned context steps. First differences rather than levels: two series that "
          "merely share a trend correlate near 1 in levels whether or not they move "
          "together, which would score every trending collection as homogeneous.", "",
          t.round(4).to_markdown(index=False), "",
          f"**{verdict}.**", ""]
    p = OUT / "HOMOGENEITY.md"
    p.write_text("\n".join(md), encoding="utf-8", newline="\n")
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
