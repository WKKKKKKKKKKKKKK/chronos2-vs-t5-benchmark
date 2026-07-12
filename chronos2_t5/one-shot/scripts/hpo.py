"""Shared val-based HPO for one-shot LoRA -- IDENTICAL protocol for Chronos-2 and Chronos-T5.

Searches a grid over (learning_rate, lora_rank, context_length); alpha is coupled to 2*rank
and the number of steps is handled by --val early-stopping (best-val checkpoint), so those are
not extra search dimensions. Each config is trained on a representative SUBSET of datasets with
val early-stopping at reduced steps, and scored by its VALIDATION loss (never test):

  * per (config, dataset): best (min) val_loss over training, read from loss_history.csv;
  * to compare datasets of different loss-scales, normalise each dataset's column by that
    dataset's best-across-configs val_loss, then take the geometric mean across datasets ->
    a scale-free relative val score (lower = better);
  * the config with the lowest relative score is the chosen GLOBAL config for that model.

Fairness: both models use this same driver, grid, subset, and step budget. Trials are written
under models/hpo/<model>/... so the canonical one-shot adapters are NEVER touched. The held-out
TEST window is never used here.

TensorBoard: every trial logs live curves to runs/hpo/<model>/<config>/<dataset> (train/loss,
val/loss, GPU). Watch with:  tensorboard --logdir <project>/runs/hpo

  python hpo.py --model c2            # Chronos-2 LoRA
  python hpo.py --model t5            # Chronos-T5 LoRA
  python hpo.py --model c2 --smoke    # 1 config x 1 dataset x few steps (plumbing check)
"""
import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import gmean

ONESHOT = Path(__file__).resolve().parents[1]   # .../chronos2_t5/one-shot
ROOT = ONESHOT.parents[1]                        # .../SAUDI_ARAMCO
RESULTS = ONESHOT / "results"

# --- search space (same for both models) ---
GRID_LR = [1e-4, 3e-4, 1e-3]
GRID_RANK = [8, 16, 32]          # alpha = 2 * rank
GRID_CTX = [512, 1024]
HPO_STEPS = 500                  # reduced steps for search; final refit uses full 1000
SUBSET = ["monash_m1_yearly", "m4_quarterly", "monash_m3_monthly", "m5", "ercot"]   # spans freqs


def load_finetune(model):
    """Import the model's finetune module, put it in LoRA + val + TB mode; return (module, project_src)."""
    if model == "c2":
        base = ROOT / "Chronos2" / "src"
        for p in (base, base / "zero_shot", base / "one_shot"):
            sys.path.insert(0, str(p))
        import finetune_oneshot_chronos2 as F
    elif model == "t5":
        base = ROOT / "Chronos_benchmark" / "src"
        for p in (base, base / "zero_shot", base / "one_shot"):
            sys.path.insert(0, str(p))
        import finetune_oneshot as F
        F.USE_LORA = True                      # LoRA one-shot (matched method)
    else:
        raise SystemExit(f"unknown --model {model!r} (use c2 or t5)")
    F.USE_VAL = True                           # val early-stopping (gives the val score)
    F.USE_TB = True                            # live TensorBoard curves
    return F


def main():
    ap = argparse.ArgumentParser(description="Shared val-based HPO for one-shot LoRA.")
    ap.add_argument("--model", choices=["c2", "t5"], required=True)
    ap.add_argument("--steps", type=int, default=HPO_STEPS, help=f"HPO training steps (default {HPO_STEPS})")
    ap.add_argument("--smoke", action="store_true", help="tiny run (1 config, 1 dataset, few steps)")
    args = ap.parse_args()

    F = load_finetune(args.model)
    F.STEPS = 20 if args.smoke else args.steps
    hpo_models = F.MODELS_DIR / "hpo" / args.model     # trial adapters (throwaway)
    tb_base = F.TB_ROOT / "hpo" / args.model           # trial TensorBoard curves
    horizons = dict(F.BENCHMARK_II)

    grid = ([(GRID_LR[0], GRID_RANK[0], GRID_CTX[0])] if args.smoke
            else list(itertools.product(GRID_LR, GRID_RANK, GRID_CTX)))
    subset = SUBSET[:1] if args.smoke else SUBSET

    print(f"HPO {args.model}: {len(grid)} configs x {len(subset)} datasets, steps={F.STEPS}\n"
          f"  TensorBoard: tensorboard --logdir {F.TB_ROOT / 'hpo'}", flush=True)
    scores, meta = {}, {}
    for lr, rank, ctx in grid:
        F.LR, F.LORA_R, F.LORA_ALPHA, F.CONTEXT_LENGTH = lr, rank, 2 * rank, ctx
        tag = f"lr{lr:g}_r{rank}_c{ctx}"
        meta[tag] = (lr, rank, ctx)
        scores[tag] = {}
        for ds in subset:
            F.TB_ROOT = tb_base / tag                  # -> runs/hpo/<model>/<tag>/<dataset>
            out = hpo_models / tag / ds
            F.finetune_one(ds, horizons[ds], out)      # trains with the globals set above (val + TB)
            vl = float(pd.read_csv(out / "loss_history.csv")["val_loss"].min())   # best val loss
            scores[tag][ds] = vl
            print(f"  [{tag}] {ds}: best_val={vl:.4f}", flush=True)

    mat = pd.DataFrame(scores).T[subset]               # rows = config, cols = dataset
    rel = mat / mat.min(axis=0)                         # normalise each dataset by its best config
    mat["val_score"] = rel.apply(gmean, axis=1)         # scale-free relative score (lower=better)
    for tag, (lr, rank, ctx) in meta.items():
        mat.loc[tag, ["lr", "rank", "context"]] = [lr, rank, ctx]
    mat = mat.sort_values("val_score")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS / f"hpo_{args.model}_results.csv"
    mat.to_csv(out_csv)
    best = mat.iloc[0]
    print(f"\nSaved -> {out_csv}")
    print(f"BEST {args.model}: lr={best['lr']:g}  rank={int(best['rank'])}  context={int(best['context'])}  "
          f"(alpha={2 * int(best['rank'])})  val_score={best['val_score']:.4f}")


if __name__ == "__main__":
    main()