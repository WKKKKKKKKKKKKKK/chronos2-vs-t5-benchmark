"""Plot the one-shot training-loss curve: Chronos-T5 vs Chronos-2, per dataset.

Reads the per-step `loss_history.csv` that each project's finetune script writes into
its per-dataset checkpoint dir:
    Chronos_benchmark/models/ft_oneshot/<dataset>/loss_history.csv   (Chronos-T5)
    Chronos2/models/ft_oneshot/<dataset>/loss_history.csv            (Chronos-2 LoRA)
and overlays them on a shared step axis.

IMPORTANT: the two losses are DIFFERENT objectives on DIFFERENT scales -- Chronos-T5
is token cross-entropy, Chronos-2 is the quantile (pinball) loss -- so each gets its
OWN y-axis. Compare the convergence *shape* (how fast / how smoothly it drops), not the
absolute values.

Usage:
    conda activate chronos_bench
    python plot_oneshot_loss.py --dataset monash_m1_yearly
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ONESHOT = Path(__file__).resolve().parents[1]       # .../chronos2_t5/one-shot
ROOT = ONESHOT.parents[1]                            # d:/KAUST/SAUDI_ARAMCO
PLOTS = ONESHOT / "plots"
T5_FT = ROOT / "Chronos_benchmark" / "models" / "ft_oneshot"
C2_FT = ROOT / "Chronos2" / "models" / "ft_oneshot"


def _load(ft_root: Path, dataset: str):
    p = ft_root / dataset / "loss_history.csv"
    return pd.read_csv(p) if p.exists() else None


C2_COLORS = ["tab:red", "tab:orange", "tab:purple", "tab:green"]


def _resolve_c2_curves(args):
    """Return [(label, DataFrame), ...] for the C2 (right) axis.

    Default: the single canonical loss_history.csv. With one or more --c2-csv
    'label=path' (path relative to the per-dataset checkpoint dir, or absolute),
    plot those instead -- e.g. to compare learning rates."""
    if not args.c2_csv:
        df = _load(C2_FT, args.dataset)
        return [("Chronos-2 (quantile loss, LoRA)", df)] if df is not None else []
    curves = []
    for spec in args.c2_csv:
        label, _, path = spec.partition("=")
        p = Path(path)
        if not p.is_absolute():
            p = C2_FT / args.dataset / path
        curves.append((label, pd.read_csv(p) if p.exists() else None))
    return curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="monash_m1_yearly", help="Benchmark II dataset name")
    ap.add_argument("--window", type=int, default=25, help="rolling-mean window for smoothing")
    ap.add_argument("--c2-csv", action="append", default=None,
                    help="C2 curve as 'label=path' (repeatable); path relative to the dataset's "
                         "checkpoint dir or absolute. Overrides the default single C2 curve.")
    ap.add_argument("--out", default=None, help="output PNG path (default: loss_<dataset>.png here)")
    args = ap.parse_args()

    t5 = _load(T5_FT, args.dataset)
    c2_curves = [(lbl, df) for lbl, df in _resolve_c2_curves(args) if df is not None]
    if t5 is None and not c2_curves:
        raise SystemExit(
            f"no loss curves for '{args.dataset}'. Run the finetune scripts with "
            f"--only {args.dataset} --force first.")

    fig, axL = plt.subplots(figsize=(9, 5))
    axR = axL.twinx()
    handles = []

    if t5 is not None:
        axL.plot(t5["step"], t5["loss"], color="tab:blue", alpha=0.20, lw=0.8)
        h, = axL.plot(t5["step"], t5["loss"].rolling(args.window, min_periods=1).mean(),
                      color="tab:blue", lw=2, label="Chronos-T5 (cross-entropy, full FT)")
        handles.append(h)
        axL.set_ylabel("Chronos-T5 loss — token cross-entropy", color="tab:blue")
        axL.tick_params(axis="y", labelcolor="tab:blue")

    for (label, df), color in zip(c2_curves, C2_COLORS):
        axR.plot(df["step"], df["loss"], color=color, alpha=0.18, lw=0.8)
        h, = axR.plot(df["step"], df["loss"].rolling(args.window, min_periods=1).mean(),
                      color=color, lw=2, label=label)
        handles.append(h)
    if c2_curves:
        axR.set_ylabel("Chronos-2 loss — quantile / pinball", color="tab:red")
        axR.tick_params(axis="y", labelcolor="tab:red")

    axL.set_xlabel("training step")
    axL.set_title(f"One-shot fine-tuning loss — {args.dataset}\n"
                  f"(different objectives & scales; compare convergence shape, not absolute values)")
    axL.legend(handles=handles, loc="upper right")
    fig.tight_layout()

    PLOTS.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else PLOTS / f"loss_{args.dataset}.png"
    fig.savefig(out, dpi=130)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()