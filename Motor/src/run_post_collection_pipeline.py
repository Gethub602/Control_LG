"""Wait for gain-chunk collection, validate it, then run the real gain sweep.

The chain is deliberately fail-closed: a partial/aborted/clock-corrupted
collection does not release the next motor experiment, and an incomplete sweep
does not rebuild the horizon dataset.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


MOTOR_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = MOTOR_DIR / "results" / "esp32_gain_sweep"
STATUS_PATH = MOTOR_DIR / "results" / "summary" / "post_collection_pipeline_status.json"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collector-pid", type=int, required=True)
    p.add_argument("--collection-run-dir", type=Path, required=True)
    p.add_argument("--collection-target", type=int, default=709)
    p.add_argument("--max-time-gap", type=float, default=0.2)
    p.add_argument("--poll-seconds", type=float, default=30.0)
    p.add_argument("--sweep-targets", default="65,85,105")
    p.add_argument("--sweep-cases", type=int, default=420)
    p.add_argument("--run-label", default="post_real1000")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def write_status(stage, **details):
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        **details,
    }
    STATUS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[pipeline] stage={stage} details={details}", flush=True)


def process_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def validate_collection(run_dir, target, max_gap):
    run_dir = run_dir if run_dir.is_absolute() else MOTOR_DIR / run_dir
    metadata_paths = sorted(run_dir.glob("metadata_*.csv"))
    # Ignore the checkpoint once the final metadata exists.
    metadata_paths = [p for p in metadata_paths if "checkpoint" not in p.name]
    if not metadata_paths:
        raise RuntimeError(f"final collection metadata not found in {run_dir}")
    with metadata_paths[-1].open(encoding="utf-8-sig", newline="") as f:
        meta = list(csv.DictReader(f))
    if len(meta) != target:
        raise RuntimeError(f"collection incomplete: {len(meta)}/{target}")
    aborted = [r for r in meta if str(r["aborted"]).lower() == "true"]
    if aborted:
        raise RuntimeError(f"collection contains {len(aborted)} aborted trajectories")

    files = sorted(run_dir.glob("trajectory_*.csv"))
    if len(files) != target:
        raise RuntimeError(f"trajectory file mismatch: {len(files)}/{target}")
    max_observed_gap = 0.0
    for path in files:
        raw = pd.read_csv(path, usecols=["time"])
        if len(raw) != 120:
            raise RuntimeError(f"{path.name} has {len(raw)} rows, expected 120")
        dt = raw["time"].astype(float).diff().dropna().to_numpy()
        if len(dt):
            max_observed_gap = max(max_observed_gap, float(np.max(dt)))
    if max_observed_gap > max_gap:
        raise RuntimeError(
            f"collection time gap {max_observed_gap:.6f}s exceeds {max_gap:.6f}s"
        )
    return {"trajectories": len(files), "max_time_gap": max_observed_gap}


def latest_labeled_metrics(label):
    paths = sorted(RESULT_DIR.glob(f"esp32_gain_sweep_metrics_{label}_*.csv"),
                   key=lambda p: p.stat().st_mtime)
    if not paths:
        raise RuntimeError(f"no final sweep metrics found for label {label}")
    return paths[-1]


def run_checked(command, env):
    print("[pipeline] exec:", " ".join(map(str, command)), flush=True)
    completed = subprocess.run(command, cwd=MOTOR_DIR, env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit code {completed.returncode}")


def main():
    args = parse_args()
    sweep_command = [
        sys.executable,
        "src/esp32_gain_sweep.py",
        "--targets", args.sweep_targets,
        "--test-time", "20",
        "--rest-time", "2",
        "--pwm-max", "200",
        "--pwm-rate-limit", "20",
        "--run-label", args.run_label,
    ]
    dataset_command = [
        sys.executable,
        "src/build_esp32_horizon_cost_dataset.py",
        "--horizon-steps", "20",
        "--lag-steps", "5",
        "--latest-only",
        "--output-name", f"esp32_horizon_cost_dataset_{args.run_label}.csv",
        "--save-latest",
    ]
    if args.dry_run:
        print("wait for PID:", args.collector_pid)
        print("validate:", args.collection_run_dir)
        print("sweep:", " ".join(map(str, sweep_command)))
        print("dataset:", " ".join(map(str, dataset_command)))
        return 0

    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["MPLBACKEND"] = "Agg"
    write_status("waiting_for_collection", collector_pid=args.collector_pid)
    while process_alive(args.collector_pid):
        time.sleep(args.poll_seconds)

    try:
        collection = validate_collection(
            args.collection_run_dir, args.collection_target, args.max_time_gap
        )
        write_status("collection_validated", **collection)

        write_status("gain_sweep_running", targets=args.sweep_targets,
                     expected_cases=args.sweep_cases)
        run_checked(sweep_command, env)
        metrics_path = latest_labeled_metrics(args.run_label)
        metrics = pd.read_csv(metrics_path)
        if len(metrics) != args.sweep_cases:
            raise RuntimeError(f"sweep incomplete: {len(metrics)}/{args.sweep_cases}")
        aborted = int(metrics["aborted"].astype(str).str.lower().eq("true").sum())
        if aborted:
            raise RuntimeError(f"sweep completed with {aborted} aborted cases")
        write_status("gain_sweep_validated", cases=len(metrics),
                     metrics_path=str(metrics_path))

        run_checked(dataset_command, env)
        write_status("complete", cases=len(metrics), metrics_path=str(metrics_path),
                     horizon_dataset=f"esp32_horizon_cost_dataset_{args.run_label}.csv")
        return 0
    except Exception as exc:
        write_status("failed", error=str(exc))
        print(f"[pipeline] FAILED: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
