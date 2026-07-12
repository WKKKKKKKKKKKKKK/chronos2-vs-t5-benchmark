"""Sample GPU (and RAM) usage over time -> CSV, to monitor a long fine-tuning run.

Two modes:
  * sample: poll `nvidia-smi` every `--interval` seconds and append rows to a CSV.
            Run it in a second terminal (or the background) WHILE fine-tuning:
                python src/monitor_resources.py --out results/gpu_usage.csv --interval 3
            Stop with Ctrl-C.
  * plot  : render the collected CSV as a GPU-util + GPU-mem curve:
                python src/monitor_resources.py --plot results/gpu_usage.csv

Only stdlib + nvidia-smi are needed for sampling; plotting needs matplotlib/pandas.
This is the *time-series* companion to the per-dataset peak GPU / train-time that
finetune_oneshot_chronos2.py already writes to train_resources.csv.
"""
import argparse
import csv
import subprocess
import time
from pathlib import Path

GPU_FIELDS = ["utilization.gpu", "memory.used", "memory.total", "power.draw", "temperature.gpu"]
COLS = ["elapsed_s", "gpu_util_pct", "mem_used_mb", "mem_total_mb", "power_w", "temp_c"]


def _sample_gpu():
    """One nvidia-smi reading -> list of strings (NaNs if the call fails)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL)
        return [x.strip() for x in out.strip().splitlines()[0].split(",")]
    except Exception:
        return ["nan"] * len(GPU_FIELDS)


def record(out_path: Path, interval: float):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"sampling every {interval}s -> {out_path}  (Ctrl-C to stop)", flush=True)
    t0 = time.perf_counter()
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        try:
            while True:
                w.writerow([round(time.perf_counter() - t0, 1), *_sample_gpu()])
                f.flush()                      # flush each row so a live tail / plot sees it
                time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\nstopped -> {out_path}", flush=True)


def plot(csv_path: Path):
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()
    ax1.plot(df["elapsed_s"] / 60, df["gpu_util_pct"], color="tab:blue", lw=1, label="GPU util %")
    ax2.plot(df["elapsed_s"] / 60, df["mem_used_mb"], color="tab:red", lw=1, label="GPU mem MB")
    ax1.set_xlabel("elapsed (min)")
    ax1.set_ylabel("GPU utilization %", color="tab:blue")
    ax2.set_ylabel("GPU memory used (MB)", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    peak = df["mem_used_mb"].max()
    ax1.set_title(f"GPU usage over time — {csv_path.name}  (peak mem {peak:.0f} MB)")
    out = csv_path.with_suffix(".png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved -> {out}")


def main():
    ap = argparse.ArgumentParser(description="Sample or plot GPU/RAM usage over time.")
    ap.add_argument("--out", default="results/gpu_usage.csv", help="CSV to write while sampling")
    ap.add_argument("--interval", type=float, default=3.0, help="seconds between samples")
    ap.add_argument("--plot", default=None, help="instead of sampling, plot an existing CSV")
    args = ap.parse_args()
    if args.plot:
        plot(Path(args.plot))
    else:
        record(Path(args.out), args.interval)


if __name__ == "__main__":
    main()