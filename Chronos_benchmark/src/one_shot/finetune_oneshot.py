"""One-shot fine-tuning of chronos-t5-small on each Benchmark II dataset.

Explicit plain-PyTorch loop (NOT the official train.py / HF Trainer): sample a window,
tokenize it with the model's own ChronosTokenizer, compute the seq2seq cross-entropy,
step AdamW with a manual linear LR decay. Fine-tuning prediction_length = the dataset's
eval horizon (train/eval horizons match; short yearly series can still form windows).

Two modes:
  * FULL fine-tuning (default) -- the Chronos paper one-shot (Section 5.5.2); writes a full
    checkpoint to models/ft_oneshot/<dataset>/. UNCHANGED behavior (paper reproduction).
  * --lora -- LoRA (peft) fine-tuning for the FAIR C2-vs-T5 head-to-head (both models LoRA);
    writes an adapter to models/ft_oneshot_lora/<dataset>/. Never touches ft_oneshot/.

--val keeps the best-val-loss checkpoint (anti-overfit). HPO knobs (--lr/--steps/
--context-length/--lora-rank/--lora-alpha) are exposed so the shared HPO driver can sweep
them identically for both models. steps are handled by --val early selection when --val is on.
"""
import argparse
import copy
import gc
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForSeq2SeqLM

SRC = Path(__file__).resolve().parent.parent   # src root (config.py, datasets_lib.py)
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "zero_shot"))     # run_zeroshot_official.py (shared harness)
from config import MODELS_DIR  # noqa: E402
from datasets_lib import BENCHMARK_II, HF_REPO, MAX_SERIES  # noqa: E402
import run_zeroshot_official as R  # reuse the official-style data loading  # noqa: E402

from chronos import ChronosConfig  # noqa: E402
from peft import (LoraConfig, get_peft_model,  # noqa: E402
                  get_peft_model_state_dict, set_peft_model_state_dict)

MODEL_ID = "amazon/chronos-t5-small"
LR = 1e-3                 # paper one-shot initial LR (annealed linearly to 0)
STEPS = 1000             # paper one-shot steps
CONTEXT_LENGTH = 512
BATCH = 16
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0      # transformers default, used by the paper
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LoRA (only used with --lora); target the T5 attention projections q/k/v/o (the matched
# counterpart of Chronos-2's attention LoRA). alpha defaults to 2*rank via the CLI.
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = ["q", "k", "v", "o"]

USE_LORA = False         # --lora
USE_VAL = False          # --val (best-val-loss checkpoint selection)
USE_TB = False           # --tb (live TensorBoard curves)
EVAL_EVERY = 50
VAL_BATCH = 64           # T5 tokenizes to full-length sequences (no patching), so the val forward
                         # is attention-heavy; keep the val batch modest to fit the 12GB GPU
OUT_FULL = MODELS_DIR / "ft_oneshot"        # full fine-tuning (paper) -- preserved
OUT_LORA = MODELS_DIR / "ft_oneshot_lora"   # LoRA one-shot (head-to-head)
TB_ROOT = MODELS_DIR.parent / "runs"        # tensorboard --logdir Chronos_benchmark/runs


def _gpu_stats():
    """(gpu_util%, mem_used_mb, power_w) via nvidia-smi, or NaNs. Call sparsely (subprocess)."""
    try:
        out = subprocess.check_output(                                               # Query nvidia-smi for GPU util%, memory (MB), and power (W); take the first GPU's CSV row and parse the three numbers.
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL)                                
        u, m, p = (x.strip() for x in out.strip().splitlines()[0].split(","))    
        return float(u), float(m), float(p)                                          # Split the first GPU's row on commas and strip whitespace
    except Exception:
        return float("nan"), float("nan"), float("nan")


def training_series(config: str, horizon: int) -> list[np.ndarray]:
    """Capped series with the last `horizon` (eval window) removed -> no leakage. NaNs kept."""
    import datasets as hfds
    ds = hfds.load_dataset(HF_REPO, config, split="train")
    ds.set_format("numpy")
    gts = R.to_gluonts_univariate(ds, MAX_SERIES)
    out = []
    for e in gts:
        t = np.asarray(e["target"][: len(e["target"]) - horizon], dtype=np.float32)    # use the last `horizon` for eval, not training
        if np.isfinite(t).sum() > horizon:   # enough non-NaN points for one window
            out.append(t)
    return out


def load_model_and_tokenizer(horizon: int):
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)         # load the Chronos T5 model from Hugging Face
    cfg = dict(model.config.chronos_config)                         # get the ChronosConfig from the model's config
    cfg["prediction_length"] = horizon                              # set the prediction length to the horizon for one-shot fine-tuning
    cfg["context_length"] = CONTEXT_LENGTH                          # set the context length to the global CONTEXT_LENGTH
    tokenizer = ChronosConfig(**cfg).create_tokenizer()             # create a ChronosTokenizer with the updated config
    model = model.to(DEVICE)
    if USE_LORA:                              # wrap with LoRA adapters (base frozen)
        model = get_peft_model(model, LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES, task_type="SEQ_2_SEQ_LM"))
        n_tr, n_tot = model.get_nb_trainable_parameters()           # get the number of trainable parameters in the LoRA model
        print(f"    LoRA trainable params: {n_tr:,} / {n_tot:,} ({100 * n_tr / n_tot:.3f}%)", flush=True)
    return model, tokenizer


def _tok_window(ctx, fut, tokenizer):
    """Tokenize one (context, future) window -> (input_ids, attention_mask, labels)."""
    c = ctx[-CONTEXT_LENGTH:]                                       # truncate context to the last CONTEXT_LENGTH points                               
    if len(c) < CONTEXT_LENGTH:                                     # pad with NaNs to the left if the context is shorter than CONTEXT_LENGTH
        c = np.concatenate([np.full(CONTEXT_LENGTH - len(c), np.nan, np.float32), c])
    ii, am, scale = tokenizer.context_input_transform(torch.tensor(c).unsqueeze(0))      # tokenize the context and get the scale for the future
    lab, lab_mask = tokenizer.label_input_transform(torch.tensor(np.asarray(fut, np.float32)).unsqueeze(0), scale)    # tokenize the future and get the label mask
    lab[lab_mask == 0] = -100
    return ii, am, lab                                              # return the tokenized input_ids, attention_mask, and labels for the model


def sample_batch(series, horizon, tokenizer, rng):
    """Sample BATCH random (context, future=horizon) windows -> tokenized tensors."""
    ids, masks, labels = [], [], []
    while len(ids) < BATCH:
        s = series[rng.integers(len(series))]              # randomly select a series from the list of series
        end = int(rng.integers(horizon + 1, len(s) + 1))   # future = s[end-horizon:end]
        ii, am, lab = _tok_window(s[max(0, end - horizon - CONTEXT_LENGTH): end - horizon], s[end - horizon: end], tokenizer)  # tokenize the context and future windows
        ids.append(ii); masks.append(am); labels.append(lab)   # append the tokenized input_ids, attention_mask, and labels to the respective lists
    return (torch.cat(ids).to(DEVICE), torch.cat(masks).to(DEVICE), torch.cat(labels).to(DEVICE))


def _val_split(series, horizon, tokenizer):
    """Reserve a val window (last `horizon` of each test-removed series) -> (train_pool, val_batch).

    train_pool drops the val window too (no leakage into val/test). val_batch is a fixed
    tokenized batch (<= VAL_BATCH series) for the validation cross-entropy. Returns
    (None, None) if no series is long enough to spare a val window.
    """
    train_pool, ids, masks, labels = [], [], [], []
    for t in series:
        pool = t[: len(t) - horizon]                           # the training pool is the series without the last `horizon` points (reserved for validation)
        if np.isfinite(pool).sum() > horizon:                  # only keep series that have enough non-NaN points to form a training window
            train_pool.append(pool)
        if len(ids) < VAL_BATCH and np.isfinite(pool).sum() > 0: # only keep series that have at least one non-NaN point in the training pool for validation
            ii, am, lab = _tok_window(pool, t[len(t) - horizon:], tokenizer) # tokenize the training pool and the reserved validation window (last `horizon` points) for the model
            ids.append(ii); masks.append(am); labels.append(lab) # append the tokenized input_ids, attention_mask, and labels to the respective lists for validation
    if not train_pool or not ids:
        return None, None
    val = (torch.cat(ids).to(DEVICE), torch.cat(masks).to(DEVICE), torch.cat(labels).to(DEVICE))
    return train_pool, val


def finetune_one(config: str, horizon: int, out_dir: Path):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    series = training_series(config, horizon)                           # get the training series for the given config and horizon, removing the last `horizon` points for validation
    if not series:
        print(f"[skip] {config}: no trainable series", flush=True)
        return None

    model, tokenizer = load_model_and_tokenizer(horizon)                
    model.train()

    train_series, val = series, None
    if USE_VAL:
        tp, val = _val_split(series, horizon, tokenizer)
        if tp is None:
            print(f"    {config}: series too short for a val window -> training without selection", flush=True)
        else:
            train_series = tp

    opt = AdamW((p for p in model.parameters() if p.requires_grad), lr=LR, weight_decay=WEIGHT_DECAY)
    best_vloss, best_state = float("inf"), None
    writer = SummaryWriter(log_dir=str(TB_ROOT / config), flush_secs=10) if USE_TB else None  # live curves

    history = []
    for step in range(STEPS):                                               # training loop for the specified number of steps
        ii, am, lab = sample_batch(train_series, horizon, tokenizer, rng)
        loss = model(input_ids=ii, attention_mask=am, labels=lab).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), MAX_GRAD_NORM)
        lr = LR * (1.0 - step / STEPS)       # explicit linear LR anneal to 0
        for g in opt.param_groups:
            g["lr"] = lr
        opt.step()
        opt.zero_grad(set_to_none=True)
        lval = loss.item()

        vloss = float("nan")
        if val is not None and (step % EVAL_EVERY == 0 or step == STEPS - 1):   # evaluate on the validation set every EVAL_EVERY steps or on the last step
            model.eval()
            with torch.no_grad():
                vloss = model(input_ids=val[0], attention_mask=val[1], labels=val[2]).loss.item()
            model.train()
            if vloss < best_vloss:              # keep the best-val-loss checkpoint (anti-overfit) if --val is specified
                best_vloss = vloss
                best_state = copy.deepcopy(get_peft_model_state_dict(model) if USE_LORA else model.state_dict())

        history.append((step, lr, lval, vloss))
        if writer is not None:                  # live TensorBoard
            writer.add_scalar("train/loss", lval, step)
            writer.add_scalar("train/lr", lr, step)
            if val is not None and not np.isnan(vloss):
                writer.add_scalar("val/loss", vloss, step)
            if DEVICE == "cuda":
                writer.add_scalar("sys/gpu_mem_alloc_mb", torch.cuda.memory_allocated() / 1e6, step)
            if step % 25 == 0:
                gu, gm, gp = _gpu_stats()
                writer.add_scalar("sys/gpu_util_pct", gu, step)
                writer.add_scalar("sys/gpu_mem_used_mb", gm, step)
                writer.add_scalar("sys/power_w", gp, step)
        if (step + 1) % 200 == 0:
            print(f"    {config} step {step + 1}/{STEPS} loss={lval:.4f}"
                  + (f" best_val={best_vloss:.4f}" if val is not None else ""), flush=True)

    if best_state is not None:               # restore best-val checkpoint before saving
        if USE_LORA:
            set_peft_model_state_dict(model, best_state)
        else:
            model.load_state_dict(best_state)
        print(f"    {config}: kept best-val checkpoint (val_loss={best_vloss:.4f})", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)           # full: model.safetensors(+config); LoRA: adapter files
    pd.DataFrame(history, columns=["step", "lr", "loss", "val_loss"]).to_csv(
        out_dir / "loss_history.csv", index=False)
    if writer is not None:
        writer.close()
    del model, opt                             # free GPU between trials (critical for the HPO loop)
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out_dir


def main():
    global LR, STEPS, CONTEXT_LENGTH, LORA_R, LORA_ALPHA, USE_LORA, USE_VAL, USE_TB
    ap = argparse.ArgumentParser(description="One-shot fine-tune chronos-t5-small per Benchmark II dataset.")
    ap.add_argument("--only", default=None, help="only datasets whose name contains this substring")
    ap.add_argument("--force", action="store_true", help="re-train even if a checkpoint exists")
    ap.add_argument("--lora", action="store_true", help="LoRA fine-tuning -> ft_oneshot_lora/ (else full FT -> ft_oneshot/)")
    ap.add_argument("--val", action="store_true", help="keep the best-val-loss checkpoint (anti-overfit)")
    ap.add_argument("--tb", action="store_true", help="log live loss + GPU curves to TensorBoard (runs/<dataset>)")
    # HPO knobs (same as Chronos-2's finetune, so the HPO driver drives both identically)
    ap.add_argument("--lr", type=float, default=LR, help=f"initial LR, annealed to 0 (default {LR})")
    ap.add_argument("--steps", type=int, default=STEPS, help=f"training steps (default {STEPS})")
    ap.add_argument("--context-length", type=int, default=CONTEXT_LENGTH, help=f"context window (default {CONTEXT_LENGTH})")
    ap.add_argument("--lora-rank", type=int, default=LORA_R, help=f"LoRA rank (default {LORA_R})")
    ap.add_argument("--lora-alpha", type=int, default=LORA_ALPHA, help=f"LoRA alpha (default {LORA_ALPHA})")
    args = ap.parse_args()
    USE_LORA, USE_VAL, USE_TB = args.lora, args.val, args.tb
    LR, STEPS, CONTEXT_LENGTH = args.lr, args.steps, args.context_length
    LORA_R, LORA_ALPHA = args.lora_rank, args.lora_alpha

    out_root = OUT_LORA if USE_LORA else OUT_FULL
    marker = "adapter_config.json" if USE_LORA else "model.safetensors"
    out_root.mkdir(parents=True, exist_ok=True)
    todo = [(c, h) for c, h in BENCHMARK_II if args.only is None or args.only in c]
    if not todo:
        sys.exit(f"--only={args.only!r} matched no Benchmark II dataset")

    manifest = []
    for config, horizon in todo:
        out_dir = out_root / config
        rel = out_dir.relative_to(MODELS_DIR).as_posix()
        if (out_dir / marker).exists() and not args.force:
            print(f"[skip] {config}: checkpoint exists", flush=True)
            manifest.append({"dataset": config, "horizon": horizon, "checkpoint": rel})
            continue
        mode = f"LoRA r={LORA_R}" if USE_LORA else "full"
        print(f"[finetune] {config} ({mode}, horizon={horizon}, lr={LR}, steps={STEPS}, ctx={CONTEXT_LENGTH}) ...", flush=True)
        ckpt = finetune_one(config, horizon, out_dir)
        manifest.append({"dataset": config, "horizon": horizon, "checkpoint": rel if ckpt else None})
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    if args.only is None:                    # don't clobber the full manifest on a targeted run
        pd.DataFrame(manifest).to_csv(out_root / "manifest.csv", index=False)
    print(f"\nFine-tuned {len(manifest)} dataset(s) -> {out_root}")


if __name__ == "__main__":
    main()
