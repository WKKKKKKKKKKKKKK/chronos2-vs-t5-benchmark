"""Explicit one-shot LoRA fine-tuning of amazon/chronos-2 on each Benchmark II dataset.

The Chronos-2 counterpart of the Chronos-T5 one-shot
(`Chronos_benchmark/src/finetune_oneshot.py`). It keeps the SAME paper-faithful
one-shot *recipe* so the two are comparable, and -- exactly like the T5 script --
writes the optimization out as an EXPLICIT plain-PyTorch loop instead of handing it
to the library (`Chronos2Pipeline.fit` / HuggingFace `Trainer`):

  * per dataset (one adapter per Benchmark II config);
  * train prediction_length = the dataset's evaluation horizon (so train and eval
    horizons match, and short yearly series can still form a training window);
  * the last `horizon` points (the eval window) are held out -> no leakage;
  * lr 1e-3 annealed linearly to 0 over 1000 steps, AdamW, grad-clip 1.0, seed 0
    -- the T5 one-shot recipe (Chronos paper Section 5.5.2).

What necessarily differs from the T5 loop -- architecture, NOT recipe:
  * Chronos-2 has NO token vocabulary / ChronosTokenizer. Its `forward()` takes the
    RAW `context` + `future_target` and returns the quantile loss directly -- patching,
    InstanceNorm scaling and the quantile objective are all internal. So there is no
    "tokenize -> seq2seq cross-entropy" step (the heart of the T5 loop); we feed raw
    windows and read `out.loss`, which is exactly the model's own training objective.
  * We adapt with LoRA (peft) rather than full fine-tuning, so the model fine-tunes
    on a local GPU without memory exhaustion (only the small adapters carry grads).
    Rank / alpha / target modules mirror the official Chronos-2 LoRA defaults
    (attention q/k/v/o + the output patch-embedding layer).

Training windows are univariate and independent (each item is its own group via
`group_ids = arange(batch)`), matching the `univariate` zero-shot mode and the T5
one-shot. NaNs are kept IN PLACE in both context and future window; the model masks
them (context attention mask + loss mask), exactly as gluonts does at eval time.

Outputs:
  models/ft_oneshot/<dataset>/   -- per-dataset LoRA adapter (peft save_pretrained)
  models/ft_oneshot/manifest.csv -- dataset, horizon, repo-relative adapter path
"""
import argparse
import copy
import gc
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW

try:
    import psutil                      # optional: logs process RAM (RSS); skipped if not installed
    _PROC = psutil.Process()
except Exception:
    psutil, _PROC = None, None

from torch.utils.tensorboard import SummaryWriter   # live loss/resource curves for TensorBoard

SRC = Path(__file__).resolve().parent.parent   # src root (config.py, datasets_lib.py)
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "zero_shot"))     # run_zeroshot_chronos2.py (shared harness)
from config import MODELS_DIR  # noqa: E402
from datasets_lib import BENCHMARK_II, HF_REPO, MAX_SERIES  # noqa: E402
import run_zeroshot_chronos2 as R  # reuse the zero-shot data loading (identical harness)  # noqa: E402

from chronos import BaseChronosPipeline  # noqa: E402
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict  # noqa: E402

MODEL_ID = "amazon/chronos-2"
LR = 1e-3                 # T5 one-shot initial LR (annealed linearly to 0); LoRA-friendly
STEPS = 1000             # T5 one-shot steps
CONTEXT_LENGTH = 512     # training-window context (mirrors T5; <= model's native context)
BATCH = 16               # T5 one-shot batch
WEIGHT_DECAY = 0.0       # regularization to reduce overfitting
MAX_GRAD_NORM = 1.0      # gradient clipping threshold
SEED = 0
DTYPE = torch.bfloat16   # match the zero-shot study + keep memory low (only adapters carry grads)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_ROOT = MODELS_DIR / "ft_oneshot"

# Official Chronos-2 LoRA defaults (chronos.chronos2.pipeline.Chronos2Pipeline.fit).
LORA_R = 8                  # LoRA rank; controls adapter capacity
LORA_ALPHA = 16             # LoRA scaling factor (scale = alpha / r)
LORA_DROPOUT = 0.0          # dropout applied to LoRA layers

# Unlike T5 (attention q/k/v/o only), C2 also adapts the output layer: it emits continuous.
# quantiles (no fixed token vocab), so the output head needs per-dataset recalibration too.
LORA_TARGET_MODULES = [
    "self_attention.q",     # query projection
    "self_attention.k",     # key projection
    "self_attention.v",     # value projection
    "self_attention.o",     # attention output projection
    "output_patch_embedding.output_layer",   # output embedding layer
]

USE_TB = False                          # toggled on by --tb; per-step curves -> TB_ROOT/<dataset>
TB_ROOT = MODELS_DIR.parent / "runs"    # view with: tensorboard --logdir Chronos2/runs

USE_VAL = False                         # toggled on by --val; keep the best-val-loss adapter (anti-overfit)
EVAL_EVERY = 50                         # steps between validation evaluations
VAL_BATCH = 256                         # number of series used to estimate validation loss

# Train-time cross-learning (Phase 6, C2-only). Default False -> univariate training (group_ids
# = arange, each item its own group), matching Phase 4 and the T5 head-to-head. When True, every
# item in a batch shares ONE group id, so GroupSelfAttention mixes across the batch's series AND
# gradients flow through it -- the training-time mirror of eval's `cross_learning=True`. This is a
# C2 full-form variant, NOT part of the fair head-to-head with T5.
TRAIN_CROSS_LEARNING = False


def _gpu_stats():
    """Device-wide (gpu_util%, mem_used_mb, power_w) via nvidia-smi, or NaNs.

    Uses a subprocess, so call it sparsely (every N steps), not every step.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
             "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL)
        u, m, p = (x.strip() for x in out.strip().splitlines()[0].split(","))
        return float(u), float(m), float(p)
    except Exception:
        return float("nan"), float("nan"), float("nan")


def training_series(config: str, horizon: int) -> list[np.ndarray]:
    """Capped series with the last `horizon` (the eval window) removed -> no leakage.

    Verbatim policy of the T5 one-shot: reuse the zero-shot loader so the series set /
    order / cap are byte-identical, drop the eval window, and keep a series only if it
    still has more than `horizon` finite points (enough real data for one window). NaNs
    are kept in place so temporal spacing is preserved; the model masks them downstream.
    """
    import datasets as hfds

    ds = hfds.load_dataset(HF_REPO, config, split="train")
    ds.set_format("numpy")
    gts = R.to_gluonts_univariate(ds, MAX_SERIES)
    out = []
    for e in gts:
        t = np.asarray(e["target"][: len(e["target"]) - horizon], dtype=np.float32)
        if np.isfinite(t).sum() > horizon:       # Keep series with more finite points than the forecast horizon after removing the eval window.
            out.append(t)
    return out


def build_lora_model():
    """Load amazon/chronos-2 and wrap its core model with a LoRA adapter.

    Returns (peft_model, output_patch_size). `peft_model` forwards straight to the
    underlying `Chronos2Model.forward`, so we call it with raw (context, future_target).
    """
    pipe = BaseChronosPipeline.from_pretrained(MODEL_ID, device_map=DEVICE, torch_dtype=DTYPE)
    base = pipe.model  # Chronos2Model (nn.Module)
    output_patch_size = base.chronos_config.output_patch_size
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
    )
    peft_model = get_peft_model(base, lora_config)                            # wrap the base model with a LoRA adapter (peft)
    n_trainable, n_total = peft_model.get_nb_trainable_parameters()           # count trainable vs total params
    print(f"    LoRA trainable params: {n_trainable:,} / {n_total:,} "
          f"({100 * n_trainable / n_total:.3f}%)", flush=True)                # log LoRA parameter ratio
    return peft_model, output_patch_size


def sample_batch(series, horizon, rng):
    """Sample BATCH random (context, future=horizon) windows -> raw float tensors.

    Mirrors the T5 sampler, but emits RAW values (no tokenization): the context is
    left-padded to CONTEXT_LENGTH with NaN (the model's instance-norm/patching builds
    the attention mask from NaNs), and the future window is exactly `horizon` long.
    """
    ctxs, futs = [], []
    while len(ctxs) < BATCH:
        s = series[rng.integers(len(series))]              # random select a series
        end = int(rng.integers(horizon + 1, len(s) + 1))   # future = s[end-horizon:end]
        ctx = s[max(0, end - horizon - CONTEXT_LENGTH): end - horizon] #cut historical context window
        if len(ctx) < CONTEXT_LENGTH:
            ctx = np.concatenate([np.full(CONTEXT_LENGTH - len(ctx), np.nan, np.float32), ctx])
        ctxs.append(ctx)
        futs.append(s[end - horizon: end])
    context = torch.as_tensor(np.stack(ctxs), dtype=torch.float32, device=DEVICE)
    future = torch.as_tensor(np.stack(futs), dtype=torch.float32, device=DEVICE)
    return context, future


def _val_split(series, horizon):
    """Reserve a validation window: the last `horizon` of the (already test-removed) series.

    Returns (train_pool, val_context, val_future):
      * train_pool: series with their last `horizon` ALSO removed, so sampled training
        windows never touch the val window (which itself precedes the sacred test window)
        -> no leakage into val or test.
      * val_context / val_future: a fixed batch (<= VAL_BATCH series) for the validation loss;
        context = series[:-horizon] (NaN-left-padded to CONTEXT_LENGTH), future = last horizon.
    Returns (None, None, None) if no series is long enough to spare a val window.
    """
    train_pool, ctxs, futs = [], [], []
    for t in series:
        pool = t[: len(t) - horizon]                     # drop the val window from the training pool
        if np.isfinite(pool).sum() > horizon:            # enough real data left for a training window
            train_pool.append(pool)
        if len(ctxs) < VAL_BATCH and np.isfinite(pool).sum() > 0:
            ctx = pool[-CONTEXT_LENGTH:]
            if len(ctx) < CONTEXT_LENGTH:
                ctx = np.concatenate([np.full(CONTEXT_LENGTH - len(ctx), np.nan, np.float32), ctx])
            ctxs.append(ctx)
            futs.append(t[len(t) - horizon:])
    if not train_pool or not ctxs:
        return None, None, None
    val_ctx = torch.as_tensor(np.stack(ctxs), dtype=torch.float32, device=DEVICE)
    val_fut = torch.as_tensor(np.stack(futs), dtype=torch.float32, device=DEVICE)
    return train_pool, val_ctx, val_fut


def finetune_one(config: str, horizon: int, out_dir: Path):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    series = training_series(config, horizon)
    if not series:
        print(f"[skip] {config}: no trainable series", flush=True)
        return None

    # validation split for anti-overfit checkpoint selection (--val). Falls back to plain
    # training (take the final adapter) if series are too short to spare a val window.
    train_series, val_ctx, val_fut, val_gids = series, None, None, None
    if USE_VAL:
        tp, val_ctx, val_fut = _val_split(series, horizon)
        if tp is None:
            print(f"    {config}: series too short for a val window -> training without selection", flush=True)
        else:
            train_series = tp
            if TRAIN_CROSS_LEARNING:
                # cross-learning val: one shared group. Cap the group to the eval group size
                # (CROSS_LEARNING_BATCH) so the val forward stays within tested memory limits.
                cap = min(val_ctx.shape[0], R.CROSS_LEARNING_BATCH)
                val_ctx, val_fut = val_ctx[:cap], val_fut[:cap]
                val_gids = torch.zeros(cap, dtype=torch.long, device=DEVICE)
            else:
                val_gids = torch.arange(val_ctx.shape[0], device=DEVICE)

    peft_model, output_patch_size = build_lora_model()
    peft_model.train()
    num_output_patches = math.ceil(horizon / output_patch_size)   # enough patches to cover the horizon
    group_ids = (torch.zeros(BATCH, dtype=torch.long, device=DEVICE)   # cross-learning: one shared group
                 if TRAIN_CROSS_LEARNING                               #   -> group attention active across the batch
                 else torch.arange(BATCH, device=DEVICE))             # univariate: each item is its own group
    opt = AdamW((p for p in peft_model.parameters() if p.requires_grad),  # optimizer
                lr=LR, weight_decay=WEIGHT_DECAY)
    writer = SummaryWriter(log_dir=str(TB_ROOT / config), flush_secs=10) if USE_TB else None  # live curves
    best_vloss, best_state = float("inf"), None   # best-val-loss checkpoint (used only when val_ctx is set)

    history = []                             # per-step (step, lr, loss, val_loss) -> the loss curve
    for step in range(STEPS):
        context, future = sample_batch(train_series, horizon, rng)
        out = peft_model(context=context, future_target=future,
                         group_ids=group_ids, num_output_patches=num_output_patches)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in peft_model.parameters() if p.requires_grad), MAX_GRAD_NORM)    # gradient clipping
        lr = LR * (1.0 - step / STEPS)       # explicit linear LR anneal to 0
        for g in opt.param_groups:
            g["lr"] = lr
        opt.step()
        opt.zero_grad(set_to_none=True) # explicit zero_grad (faster than default)
        lval = loss.item()

        vloss = float("nan")
        if val_ctx is not None and (step % EVAL_EVERY == 0 or step == STEPS - 1):   # validation eval
            peft_model.eval()
            with torch.no_grad():
                vloss = peft_model(context=val_ctx, future_target=val_fut,
                                   group_ids=val_gids, num_output_patches=num_output_patches).loss.item()
            peft_model.train()
            if vloss < best_vloss:           # keep the best-val-loss adapter (anti-overfit selection)
                best_vloss = vloss
                best_state = copy.deepcopy(get_peft_model_state_dict(peft_model))

        history.append((step, lr, lval, vloss))
        if writer is not None:                  # live TensorBoard
            writer.add_scalar("train/loss", lval, step)
            writer.add_scalar("train/lr", lr, step)
            if val_ctx is not None and not math.isnan(vloss):
                writer.add_scalar("val/loss", vloss, step)
            if DEVICE == "cuda":                # GPU mem this process holds: cheap (torch), every step
                writer.add_scalar("sys/gpu_mem_alloc_mb", torch.cuda.memory_allocated() / 1e6, step)
            if step % 25 == 0:                  # device-wide util/mem/power via nvidia-smi: sample sparsely
                gutil, gmem, gpow = _gpu_stats()
                writer.add_scalar("sys/gpu_util_pct", gutil, step)
                writer.add_scalar("sys/gpu_mem_used_mb", gmem, step)
                writer.add_scalar("sys/power_w", gpow, step)
        if (step + 1) % 200 == 0:
            print(f"    {config} step {step + 1}/{STEPS} loss={lval:.4f}"
                  + (f" best_val={best_vloss:.4f}" if val_ctx is not None else ""), flush=True)

    if best_state is not None:               # restore the best-val adapter before saving (anti-overfit)
        set_peft_model_state_dict(peft_model, best_state)
        print(f"    {config}: kept best-val adapter (val_loss={best_vloss:.4f})", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(out_dir))   # saves adapter_config.json + adapter weights
    pd.DataFrame(history, columns=["step", "lr", "loss", "val_loss"]).to_csv(   # per-step loss curve
        out_dir / "loss_history.csv", index=False)
    if writer is not None:
        writer.close()
    del peft_model, opt                        # free GPU between trials (critical for the HPO loop)
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out_dir


def main():
    # ── B-1: let CLI flags override the module-level constants (also enables HPO sweeps) ──
    global LR, STEPS, CONTEXT_LENGTH, LORA_R, LORA_ALPHA, USE_TB, USE_VAL, TRAIN_CROSS_LEARNING   # assigning below mutates the MODULE globals
                          # (not new locals); must precede any use, else "used prior to global declaration"

    # ── B-2: define the command-line arguments ──
    ap = argparse.ArgumentParser(description="One-shot LoRA fine-tune amazon/chronos-2 per Benchmark II dataset.")         # build the parser; `description` shows up in `-h` help
    ap.add_argument("--only", default=None,                                                                                # optional, takes a string; None when omitted
                    help="only fine-tune datasets whose name contains this substring (e.g. monash_m1_yearly)")
    ap.add_argument("--force", action="store_true",                                                                        # boolean flag: present=True, absent=False;
                    help="re-train even if an adapter already exists (overwrites it)")                                     # store_true = "true if it appears", takes no value
    ap.add_argument("--lr", type=float, default=LR,                                                                        # takes a float; defaults to the file constant LR
                    help=f"initial LR, annealed linearly to 0 (default {LR}; LoRA often prefers ~1e-4)")
    # --- HPO knobs: each defaults to its module constant; override to sweep hyper-parameters ---
    ap.add_argument("--steps", type=int, default=STEPS, help=f"training steps (default {STEPS})")
    ap.add_argument("--context-length", type=int, default=CONTEXT_LENGTH,
                    help=f"training context window length (default {CONTEXT_LENGTH})")
    ap.add_argument("--lora-r", type=int, default=LORA_R, help=f"LoRA rank (default {LORA_R})")
    ap.add_argument("--lora-alpha", type=int, default=LORA_ALPHA, help=f"LoRA alpha (default {LORA_ALPHA})")
    ap.add_argument("--tb", action="store_true", help="log live loss + GPU curves to TensorBoard (runs/<dataset>)")
    ap.add_argument("--val", action="store_true",
                    help="reserve a val window per series and keep the best-val-loss adapter (anti-overfit)")
    ap.add_argument("--train-cross-learning", action="store_true",
                    help="Phase 6 (C2-only): share one group id across the batch so group attention is "
                         "active DURING training; NOT part of the head-to-head with T5")
    args = ap.parse_args()   # parse sys.argv -> args.only / args.force / args.lr / args.steps / ...
    LR = args.lr             # write parsed overrides back to the module globals
    STEPS, CONTEXT_LENGTH = args.steps, args.context_length
    LORA_R, LORA_ALPHA = args.lora_r, args.lora_alpha
    USE_TB = args.tb
    USE_VAL = args.val
    TRAIN_CROSS_LEARNING = args.train_cross_learning

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ── B-3: filter which datasets to train, per --only ──
    # no --only -> keep all 25; given -> keep names CONTAINING the substring (not an exact match)
    todo = [(c, h) for c, h in BENCHMARK_II if args.only is None or args.only in c]
    if not todo:                                       # --only mistyped / matched nothing
        sys.exit(f"--only={args.only!r} matched no Benchmark II dataset")   # !r -> repr(), clearer error

    manifest, resources = [], []
    for config, horizon in todo:
        out_dir = OUT_ROOT / config
        rel = out_dir.relative_to(MODELS_DIR).as_posix()   # portable, repo-relative
        # ── B-4: --force decides whether to skip an already-trained adapter ──
        # exists AND no --force -> skip (resumable); with --force, fall through and retrain/overwrite
        if (out_dir / "adapter_config.json").exists() and not args.force:
            print(f"[skip] {config}: adapter exists", flush=True)
            manifest.append({"dataset": config, "horizon": horizon, "adapter": rel})
            continue
        print(f"[finetune] {config} (horizon={horizon}, lr={LR}, steps={STEPS}, "
              f"ctx={CONTEXT_LENGTH}, LoRA r={LORA_R}/a={LORA_ALPHA}) ...", flush=True)
        # resource logging: reset GPU peak + start a wall-clock timer around this dataset's training
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        ckpt = finetune_one(config, horizon, out_dir)      # train LoRA adapter for this dataset
        train_s = time.perf_counter() - t0
        peak_gpu_mb = torch.cuda.max_memory_allocated() / 1e6 if DEVICE == "cuda" else float("nan")
        rss_mb = _PROC.memory_info().rss / 1e6 if _PROC else float("nan")
        manifest.append({"dataset": config, "horizon": horizon,
                         "adapter": rel if ckpt else None})
        resources.append({"dataset": config, "train_s": round(train_s, 1),
                          "peak_gpu_mb": round(peak_gpu_mb, 1), "rss_mb": round(rss_mb, 1),
                          "steps": STEPS, "lr": LR, "lora_r": LORA_R, "context_length": CONTEXT_LENGTH})
        print(f"    -> trained in {train_s:.1f}s, peak GPU {peak_gpu_mb:.0f} MB", flush=True)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # ── B-5: a targeted --only run must NOT clobber the 25-row manifest ──
    # a --only run of 1 dataset would collapse manifest.csv to 1 row, so only a full run rewrites it
    if args.only is None:
        pd.DataFrame(manifest).to_csv(OUT_ROOT / "manifest.csv", index=False)
    if resources:                                       # per-run training resource log (GPU peak, time, RAM)
        res_path = OUT_ROOT / "train_resources.csv"
        df_new = pd.DataFrame(resources)
        if res_path.exists():                           # merge, don't clobber: a --only backfill keeps prior rows
            old = pd.read_csv(res_path)
            old = old[~old["dataset"].isin(df_new["dataset"])]   # drop datasets we just re-measured
            df_new = pd.concat([old, df_new], ignore_index=True)
        df_new.to_csv(res_path, index=False)
    print(f"\nFine-tuned {len(manifest)} dataset(s) -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
