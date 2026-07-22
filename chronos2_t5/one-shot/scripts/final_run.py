"""Phase 4: final tuned one-shot on ALL 25 Benchmark II datasets, per model, with its
HPO-selected best config (1000 steps, val early-stopping). Then evaluate (univariate).

Uses each model's finetune_one + eval loaders directly (same tested code as the CLI), so the
result is on the identical gluonts pipeline. Writes to NEW locations only:
    models/ft_oneshot_hpo/<dataset>/         (tuned adapters)
    <chronos2_t5>/one-shot/oneshot_hpo_<model>.csv   (per-dataset MASE/WQL + efficiency)
The earlier default-config one-shot (ft_oneshot / ft_oneshot_lora) and full-FT are untouched.

  python final_run.py --model c2      # Chronos-2, best: lr1e-3 r16 ctx512
  python final_run.py --model t5      # Chronos-T5, best: lr1e-3 r32 ctx512
  python final_run.py --model c2 --eval-only   # skip training, just eval existing ft_oneshot_hpo
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from gluonts.dataset.split import split

ONESHOT = Path(__file__).resolve().parents[1]   # .../chronos2_t5/one-shot
ROOT = ONESHOT.parents[1]                        # repo root
RESULTS = ONESHOT / "results"

BEST = {"c2": dict(lr=1e-3, rank=16, ctx=512),
        "t5": dict(lr=1e-3, rank=32, ctx=512)}


def setup(model, use_tb, cross=False):
    """Import the model's modules, set the best config globals, return handles + an eval fn.

    cross=True (C2 only) evaluates in cross-learning mode instead of univariate -- a C2 ceiling
    reference, NOT part of the fair head-to-head (T5 has no cross-learning).
    """
    best = BEST[model]
    if model == "c2":
        base = ROOT / "Chronos2" / "src"
        for p in (base, base / "zero_shot", base / "one_shot"):
            sys.path.insert(0, str(p))
        import finetune_oneshot_chronos2 as F
        import run_oneshot_chronos2 as E
        import run_zeroshot_chronos2 as R
        F.USE_TB = use_tb

        def do_eval(adapter_dir, config, horizon):
            import datasets as hfds
            pipe = E.load_oneshot_pipeline(adapter_dir)          # base + merged LoRA
            ds = hfds.load_dataset(R.HF_REPO, config, split="train"); ds.set_format("numpy")
            gts = R.to_gluonts_univariate(ds, R.MAX_SERIES)
            _, tt = split(gts, offset=-horizon); td = tt.generate_instances(horizon, windows=1)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            if cross:
                fc, _, _ = R.forecast_cross_learning(pipe, list(td.input), horizon)   # C2 group attention
            else:
                fc = R.forecast_univariate(pipe, list(td.input), horizon)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            lat = time.perf_counter() - t0
            mase, wql = R.evaluate(fc, td)
            del pipe
            return mase, wql, len(gts), lat
    else:
        base = ROOT / "Chronos_benchmark" / "src"
        for p in (base, base / "zero_shot", base / "one_shot"):
            sys.path.insert(0, str(p))
        import finetune_oneshot as F
        import run_oneshot_official as E
        import run_zeroshot_official as R
        from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
        from gluonts.model.evaluation import evaluate_forecasts
        F.USE_LORA = True
        F.USE_TB = use_tb

        def do_eval(adapter_dir, config, horizon):
            import datasets as hfds
            pipe = E.load_pipe(adapter_dir, True)                # base + merged LoRA
            ds = hfds.load_dataset(R.HF_REPO, config, split="train"); ds.set_format("numpy")
            gts = R.to_gluonts_univariate(ds, R.MAX_SERIES)
            _, tt = split(gts, offset=-horizon); td = tt.generate_instances(horizon, windows=1)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            fc = R.chronos_forecasts(pipe, td.input, horizon)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            lat = time.perf_counter() - t0
            m = (evaluate_forecasts(fc, test_data=td, metrics=[MASE(), MeanWeightedSumQuantileLoss(R.QUANTILES)],
                                    batch_size=5000).reset_index(drop=True).to_dict("records")[0])
            del pipe
            return m["MASE[0.5]"], m["mean_weighted_sum_quantile_loss"], len(gts), lat

    F.USE_VAL = True
    F.STEPS = 1000
    F.LR, F.LORA_R, F.LORA_ALPHA, F.CONTEXT_LENGTH = best["lr"], best["rank"], 2 * best["rank"], best["ctx"]
    F.TB_ROOT = F.MODELS_DIR.parent / "runs" / "final"        # runs/final/<dataset>
    ft_hpo = F.MODELS_DIR / "ft_oneshot_hpo"
    return F, R, do_eval, ft_hpo


def main():
    ap = argparse.ArgumentParser(description="Phase 4: final tuned one-shot (all 25) per model.")
    ap.add_argument("--model", choices=["c2", "t5"], required=True)
    ap.add_argument("--eval-only", action="store_true", help="skip training; eval existing ft_oneshot_hpo")
    ap.add_argument("--no-tb", action="store_true", help="disable TensorBoard during training")
    ap.add_argument("--cross-learning", action="store_true",
                    help="(C2 only) eval the tuned adapters in cross-learning mode -- a C2 ceiling ref, not head-to-head")
    args = ap.parse_args()
    if args.cross_learning and args.model != "c2":
        sys.exit("--cross-learning is C2-only (Chronos-T5 has no cross-learning)")

    F, R, do_eval, ft_hpo = setup(args.model, use_tb=not args.no_tb, cross=args.cross_learning)
    horizons = dict(F.BENCHMARK_II)
    best = BEST[args.model]
    print(f"[final {args.model}] best config: lr={best['lr']:g} rank={best['rank']} "
          f"alpha={2 * best['rank']} ctx={best['ctx']} steps=1000 (val early-stop)", flush=True)

    rows = []
    for config, horizon in F.BENCHMARK_II:
        out = ft_hpo / config
        if not args.eval_only:
            marker = out / ("adapter_config.json")
            if marker.exists():
                print(f"[skip-train] {config}: adapter exists", flush=True)
            else:
                print(f"[train] {config} ...", flush=True)
                F.finetune_one(config, horizon, out)
        if not (out / "adapter_config.json").exists():
            print(f"[skip-eval] {config}: no adapter", flush=True)
            continue
        mase, wql, n, lat = do_eval(out, config, horizon)
        rows.append({"dataset": config, "MASE": mase, "WQL": wql, "n_series": n,
                     "latency_s": round(lat, 3)})
        print(f"  {config:30s} MASE={mase:.4f} WQL={wql:.4f} n={n}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    suffix = "_crosslearning" if args.cross_learning else ""
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS / f"oneshot_hpo_{args.model}{suffix}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv}  ({len(df)} datasets)")


if __name__ == "__main__":
    main()