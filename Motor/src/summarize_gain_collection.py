"""Summarize an active or completed real gain-chunk collection run."""

import argparse
import csv
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


MOTOR_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = MOTOR_DIR / "data" / "raw" / "diffusion_gain_chunk_db"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, help="Run directory; defaults to latest real* run")
    p.add_argument("--target-runs", type=int, default=950)
    return p.parse_args()


def latest_run(root):
    runs = [p for p in root.glob("real*") if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"no real collection run under {root}")
    # Labels such as real_pilot50 sort after real950, so lexical order is not
    # chronological.  An active run updates its checkpoint after every run.
    def checkpoint_mtime(path):
        checkpoints = list(path.glob("metadata_checkpoint_*.csv"))
        return max((p.stat().st_mtime for p in checkpoints), default=path.stat().st_mtime)
    return max(runs, key=checkpoint_mtime)


def main():
    args = parse_args()
    run_dir = args.run_dir or latest_run(DEFAULT_ROOT)
    metadata_paths = sorted(run_dir.glob("metadata_checkpoint_*.csv"))
    if not metadata_paths:
        raise FileNotFoundError(f"metadata checkpoint not found in {run_dir}")

    with metadata_paths[-1].open(encoding="utf-8-sig", newline="") as f:
        meta = list(csv.DictReader(f))
    files = sorted(run_dir.glob("trajectory_*.csv"))
    completed = len(meta)
    pct = completed / args.target_runs * 100

    print(f"run: {run_dir}")
    print(f"progress: {completed}/{args.target_runs} ({pct:.1f}%), trajectory files={len(files)}")
    print(f"aborted: {dict(Counter(r['aborted'] for r in meta))}")
    print(f"scenarios: {dict(Counter(r['scenario_type'] for r in meta))}")
    print(f"gain profiles: {dict(Counter(r['gain_profile_type'] for r in meta))}")

    stamp = re.search(r"(\d{8}_\d{6})", run_dir.name)
    if stamp and completed:
        started = datetime.strptime(stamp.group(1), "%Y%m%d_%H%M%S")
        elapsed = datetime.now() - started
        seconds_per_run = elapsed.total_seconds() / completed
        eta = datetime.now() + timedelta(seconds=(args.target_runs - completed) * seconds_per_run)
        print(f"pace: {seconds_per_run:.1f} sec/trajectory, ETA {eta:%Y-%m-%d %H:%M:%S}")

    row_counts, durations, max_dts, long_gaps = [], [], [], []
    loop_ms, motor_ms = [], []
    nan_rows = near_limit = 0
    for path in files:
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        row_counts.append(len(rows))
        times = np.asarray([float(r["time"]) for r in rows])
        dts = np.diff(times)
        durations.append(times[-1] - times[0])
        max_dts.append(dts.max(initial=0.0))
        long_gaps.extend(dts[dts > 0.2])
        for row in rows:
            numeric = ["time", "control_dt", "loop_elapsed_sec", "motor_step_elapsed_sec",
                       "target", "rpm", "error", "pwm", "kp", "ki", "kd"]
            nan_rows += any(not np.isfinite(float(row[k])) for k in numeric)
            near_limit += row["pwm_near_limit"].lower() == "true"
            loop_ms.append(float(row["loop_elapsed_sec"]) * 1000)
            motor_ms.append(float(row["motor_step_elapsed_sec"]) * 1000)

    if row_counts:
        print(f"rows/trajectory: min={min(row_counts)}, median={np.median(row_counts):.0f}, max={max(row_counts)}")
        print("duration sec (min/median/p90/max): " + "/".join(
            f"{x:.3f}" for x in (min(durations), np.median(durations), np.percentile(durations, 90), max(durations))))
        print("max dt sec (p50/p90/max): " + "/".join(
            f"{x:.3f}" for x in np.percentile(max_dts, (50, 90, 100))))
        print(f"gaps > 0.2 sec: {len(long_gaps)}, max={max(long_gaps, default=0):.3f} sec")
        print("loop ms (p50/p90/p99): " + "/".join(f"{x:.3f}" for x in np.percentile(loop_ms, (50, 90, 99))))
        print("serial step ms (p50/p90/p99): " + "/".join(f"{x:.3f}" for x in np.percentile(motor_ms, (50, 90, 99))))
        print(f"NaN rows: {nan_rows}, PWM-near-limit rows: {near_limit}")


if __name__ == "__main__":
    main()
