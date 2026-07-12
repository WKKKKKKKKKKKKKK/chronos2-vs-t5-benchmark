"""Phase 6 (SEPARATE, C2-only): TRAIN-TIME cross-learning Chronos-2 one-shot.

Unlike Phase 4 (univariate) and Phase 4b (univariate-trained, CL only at eval), here group
attention is active DURING TRAINING -- every batch shares one group id, so the LoRA learns to
exploit cross-series structure and gradients flow through group attention. Eval is cross-learning
too. This is Chronos-2's full form; it is NOT part of the fair head-to-head with Chronos-T5
(T5 has no group axis). Best C2 config from HPO (lr1e-3 / rank16 / ctx512), 1000 steps, val
early-stop (cross-learning val).

Everything goes to INDEPENDENT locations -- Phase 4/4b/5 artifacts are never touched:
    models/ft_oneshot_cltrain/<dataset>/   train-time-CL adapters
    runs/cltrain/<dataset>                  TensorBoard curves
    oneshot_cltrain_c2.csv                  per-dataset MASE/WQL (CL eval)
    PHASE6_CLTRAIN_REPORT.md                C2-CLtrain vs C2-uni / C2-CL(eval) / T5-uni

  python phase6_cltrain.py                # train all 25 + eval
  python phase6_cltrain.py --eval-only    # skip training, eval existing ft_oneshot_cltrain
  python phase6_cltrain.py --only ercot   # single dataset
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from gluonts.dataset.split import split
from scipy.stats import gmean

ONESHOT = Path(__file__).resolve().parents[1]   # .../chronos2_t5/one-shot
ROOT = ONESHOT.parents[1]                        # .../SAUDI_ARAMCO
RESULTS = ONESHOT / "results"
REPORTS = ONESHOT / "reports"
BEST = dict(lr=1e-3, rank=16, ctx=512)     # C2 HPO winner (same as Phase 4)


def setup(use_tb):
    base = ROOT / "Chronos2" / "src"
    for p in (base, base / "zero_shot", base / "one_shot"):
        sys.path.insert(0, str(p))
    import finetune_oneshot_chronos2 as F
    import run_oneshot_chronos2 as E
    import run_zeroshot_chronos2 as R
    F.USE_VAL = True
    F.USE_TB = use_tb
    F.TRAIN_CROSS_LEARNING = True          # <-- the Phase 6 switch
    F.STEPS = 1000
    F.LR, F.LORA_R, F.LORA_ALPHA, F.CONTEXT_LENGTH = BEST["lr"], BEST["rank"], 2 * BEST["rank"], BEST["ctx"]
    F.TB_ROOT = F.MODELS_DIR.parent / "runs" / "cltrain"
    out_root = F.MODELS_DIR / "ft_oneshot_cltrain"

    def do_eval(adapter_dir, config, horizon):
        import datasets as hfds
        pipe = E.load_oneshot_pipeline(adapter_dir)
        ds = hfds.load_dataset(R.HF_REPO, config, split="train"); ds.set_format("numpy")
        gts = R.to_gluonts_univariate(ds, R.MAX_SERIES)
        _, tt = split(gts, offset=-horizon); td = tt.generate_instances(horizon, windows=1)
        t0 = time.perf_counter()
        fc, _, _ = R.forecast_cross_learning(pipe, list(td.input), horizon)   # CL eval
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lat = time.perf_counter() - t0
        mase, wql = R.evaluate(fc, td)
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return mase, wql, len(gts), lat

    return F, R, do_eval, out_root


def write_report(df, n):
    """Compare C2-CLtrain against the existing C2/T5 lines (rel. Seasonal-Naive gmean)."""
    ref = ROOT / "Chronos_benchmark" / "reference" / "seasonal-naive-zero-shot.csv"
    sn = pd.read_csv(ref).set_index("dataset")[["MASE", "WQL"]]
    lines = {"C2-CLtrain (Phase 6)": df.set_index("dataset")[["MASE", "WQL"]]}
    for label, fname in [("C2-uni (Phase 4)", "oneshot_hpo_c2.csv"),
                         ("C2-CL eval-only (Phase 4b)", "oneshot_hpo_c2_crosslearning.csv"),
                         ("T5-uni (Phase 4)", "oneshot_hpo_t5.csv")]:
        p = RESULTS / fname
        if p.exists():
            lines[label] = pd.read_csv(p).set_index("dataset")[["MASE", "WQL"]]

    md = ["# Phase 6: train-time cross-learning Chronos-2 (C2-only, not head-to-head)\n",
          f"Group attention active during TRAINING (best config lr{BEST['lr']:g}/r{BEST['rank']}/"
          f"ctx{BEST['ctx']}, 1000 steps, CL val), CL eval, over {n} datasets.\n",
          "## Aggregate relative score (gmean of model / Seasonal-Naive, lower = better)\n",
          "| line | MASE | WQL |", "| --- | --- | --- |"]
    for label, frame in lines.items():
        common = frame.index.intersection(sn.index)
        rel = (frame.loc[common] / sn.loc[common]).apply(gmean)
        md.append(f"| {label} | {rel['MASE']:.3f} | {rel['WQL']:.3f} |")
    md += ["\n## Per-dataset (C2-CLtrain)\n", "| dataset | MASE | WQL | n_series |",
           "| --- | --- | --- | --- |"]
    for _, r in df.iterrows():
        md.append(f"| {r['dataset']} | {r['MASE']:.4f} | {r['WQL']:.4f} | {int(r['n_series'])} |")
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "PHASE6_CLTRAIN_REPORT.md").write_text("\n".join(md), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Phase 6: train-time cross-learning C2 one-shot (C2-only).")
    ap.add_argument("--eval-only", action="store_true", help="skip training; eval existing ft_oneshot_cltrain")
    ap.add_argument("--no-tb", action="store_true", help="disable TensorBoard during training")
    ap.add_argument("--only", default=None, help="only datasets whose name contains this substring")
    ap.add_argument("--force", action="store_true", help="retrain even if an adapter exists")
    args = ap.parse_args()

    F, R, do_eval, out_root = setup(use_tb=not args.no_tb)
    todo = [(c, h) for c, h in F.BENCHMARK_II if args.only is None or args.only in c]
    print(f"[phase6] train-time cross-learning C2, {len(todo)} dataset(s), "
          f"lr={BEST['lr']:g} r={BEST['rank']} ctx={BEST['ctx']} steps=1000 (CL val)", flush=True)

    rows = []
    for config, horizon in todo:
        out = out_root / config
        if not args.eval_only:
            if (out / "adapter_config.json").exists() and not args.force:
                print(f"[skip-train] {config}: adapter exists", flush=True)
            else:
                print(f"[train-CL] {config} ...", flush=True)
                F.finetune_one(config, horizon, out)
        if not (out / "adapter_config.json").exists():
            print(f"[skip-eval] {config}: no adapter", flush=True)
            continue
        mase, wql, nseries, lat = do_eval(out, config, horizon)
        rows.append({"dataset": config, "MASE": mase, "WQL": wql, "n_series": nseries,
                     "latency_s": round(lat, 3)})
        print(f"  {config:30s} MASE={mase:.4f} WQL={wql:.4f} n={nseries}", flush=True)

    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS / "oneshot_cltrain_c2.csv"
    df.to_csv(out_csv, index=False)
    if args.only is None and not df.empty:
        write_report(df, len(df))
        print("  wrote PHASE6_CLTRAIN_REPORT.md", flush=True)
    print(f"\nSaved -> {out_csv}  ({len(df)} datasets)")


if __name__ == "__main__":
    main()