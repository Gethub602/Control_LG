"""
Train every gain-chunk model on one dataset and collect the results in one table.

Covers the family the original study compared, plus flow matching:

  supervised baselines : mlp, cnn, cnn_residual, cnn_attention
  generative           : diffusion (DDIM), flow matching (rectified flow)

All of them share the dataset, the trajectory-grouped split, the scalers and the
accuracy metrics, so differences reflect the modelling choice.

Not covered here: the horizon-cost family (rf_cost, mlp_cost, multitask_mlp,
direct_policy_mlp). Those consume a different dataset built by
build_esp32_horizon_cost_dataset.py from gain-sweep logs, which requires running
esp32_gain_sweep.py first.
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))
sys.path.append(str(CURRENT_DIR))

from config import RESULTS_DIR  # noqa: E402

SUMMARY_DIR = RESULTS_DIR / "summary"

BASELINES = ["mlp", "cnn", "cnn_residual", "cnn_attention"]


def parse_args():
    p = argparse.ArgumentParser(description="Train all gain-chunk models.")
    p.add_argument("--dataset", default="")
    p.add_argument("--profile", default="tracking_first")
    p.add_argument("--quality", choices=["all", "top_k", "best"], default="top_k")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-count", type=int, default=4)
    p.add_argument("--ddim-eval-steps", default="5,10,20,30")
    p.add_argument("--flow-eval-steps", default="1,2,4,8")
    p.add_argument("--flow-solver", default="midpoint")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--skip-baselines", action="store_true")
    p.add_argument("--run-label", default="real")
    return p.parse_args()


def run(cmd, name):
    print("=" * 78)
    print(f"### {name}")
    print("=" * 78, flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(MOTOR_DIR))
    dt = time.perf_counter() - t0
    status = "ok" if proc.returncode == 0 else f"FAILED({proc.returncode})"
    print(f"--- {name}: {status} in {dt:.1f}s", flush=True)
    return proc.returncode == 0


def newest(pattern):
    paths = sorted(SUMMARY_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    return paths[-1] if paths else None


def main():
    args = parse_args()
    py = sys.executable
    common = [
        "--profile", args.profile,
        "--quality", args.quality,
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--seed", str(args.seed),
        "--run-label", args.run_label,
    ]
    if args.dataset:
        common += ["--dataset", args.dataset]
    if args.cpu:
        common += ["--cpu"]

    started = datetime.now()
    ok = {}

    if not args.skip_baselines:
        for m in BASELINES:
            ok[f"baseline_{m}"] = run(
                [py, "-u", "src/train_torch_gain_chunk_baseline.py",
                 "--model-type", m] + common,
                f"baseline {m}",
            )

    ok["diffusion"] = run(
        [py, "-u", "src/train_torch_gain_chunk.py", "--method", "diffusion",
         "--eval-steps", args.ddim_eval_steps,
         "--sample-count", str(args.sample_count)] + common,
        "diffusion (DDIM)",
    )

    ok["flow"] = run(
        [py, "-u", "src/train_torch_gain_chunk.py", "--method", "flow",
         "--solver", args.flow_solver,
         "--eval-steps", args.flow_eval_steps,
         "--sample-count", str(args.sample_count)] + common,
        f"flow matching ({args.flow_solver})",
    )

    # ---- gather results from the saved payloads rather than by globbing CSVs
    #
    # Each payload records the dataset it was trained on, so rows from a
    # concurrent run on a different dataset cannot silently land in the table.
    import joblib

    from config import MODEL_DIR

    rows = []
    for path in MODEL_DIR.glob("torch_*.joblib"):
        if path.stat().st_mtime < started.timestamp():
            continue
        try:
            payload = joblib.load(path)
        except Exception:
            continue
        evals = payload.get("eval")
        if not evals:
            continue
        ds = Path(str(payload.get("dataset_path", ""))).name
        frame = pd.DataFrame(evals)
        frame["dataset"] = ds
        frame["params"] = payload.get("args", {}).get("model_type", "")
        rows.append(frame)
    if not rows:
        print("No eval files were produced.")
        return 1

    table = pd.concat(rows, ignore_index=True)
    if "dataset" in table.columns and table["dataset"].nunique() > 1:
        print("Multiple datasets present:", sorted(table["dataset"].unique()))
    keep = [c for c in [
        "method", "solver", "steps", "nfe", "device", "dataset",
        "sample_mean_mae", "best_of_n_mae", "chunk_rmse",
        "kp_mae", "ki_mae", "kd_mae", "sample_diversity_std",
        "latency_p50_sec", "latency_p90_sec",
    ] if c in table.columns]
    table = table[keep].sort_values("sample_mean_mae").reset_index(drop=True)

    timestamp = started.strftime("%Y%m%d_%H%M%S")
    out = SUMMARY_DIR / f"gain_chunk_model_comparison_{args.run_label}_{timestamp}.csv"
    table.to_csv(out, index=False)

    print()
    print("=" * 78)
    print("ALL GAIN-CHUNK MODELS")
    print("=" * 78)
    print(table.to_string(index=False))
    print()
    for name, good in ok.items():
        print(f"  {name:<24}{'ok' if good else 'FAILED'}")
    print()
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
