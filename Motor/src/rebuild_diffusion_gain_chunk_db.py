import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))

from collect_diffusion_gain_chunk_db import (  # noqa: E402
    PROCESSED_ROOT,
    RAW_ROOT,
    build_chunk_rows,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild diffusion gain chunk raw metrics from saved raw trajectories."
    )
    parser.add_argument(
        "--raw-dir-glob",
        default="diffusion_balanced100_20260506_160618,diffusion_balanced800_extra_to1000_20260508_143525",
        help=(
            "Comma-separated raw directory names or glob patterns under "
            "data/raw/diffusion_gain_chunk_db."
        ),
    )
    parser.add_argument("--obs-steps", type=int, default=10)
    parser.add_argument("--horizon-steps", type=int, default=30)
    parser.add_argument("--pwm-max", type=float, default=140.0)
    parser.add_argument("--run-label", default="diffusion_balanced1000_horizon30")
    parser.add_argument("--max-trajectories", type=int, default=0)
    return parser.parse_args()


def safe_label(value: str):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value))


def resolve_raw_dirs(raw_dir_glob: str):
    dirs = []
    for token in str(raw_dir_glob).split(","):
        token = token.strip()
        if not token:
            continue
        path = Path(token)
        if path.is_absolute() and path.exists():
            dirs.append(path)
            continue
        matches = sorted(RAW_ROOT.glob(token))
        dirs.extend([item for item in matches if item.is_dir()])
    unique = []
    seen = set()
    for item in dirs:
        resolved = item.resolve()
        if resolved not in seen:
            unique.append(item)
            seen.add(resolved)
    return unique


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = safe_label(args.run_label)
    raw_dirs = resolve_raw_dirs(args.raw_dir_glob)
    if not raw_dirs:
        raise FileNotFoundError(f"No raw dirs matched: {args.raw_dir_glob}")

    trajectory_paths = []
    for raw_dir in raw_dirs:
        trajectory_paths.extend(sorted(raw_dir.glob("trajectory_*.csv")))
    if args.max_trajectories and len(trajectory_paths) > int(args.max_trajectories):
        trajectory_paths = trajectory_paths[: int(args.max_trajectories)]
    if not trajectory_paths:
        raise FileNotFoundError(f"No trajectory_*.csv files found in {raw_dirs}")

    frames = []
    for idx, path in enumerate(trajectory_paths, start=1):
        if idx % 100 == 0 or idx == 1:
            print(f"[LOAD] {idx}/{len(trajectory_paths)} {path.name}")
        frames.append(pd.read_csv(path))
    raw_df = pd.concat(frames, ignore_index=True)

    chunk_args = SimpleNamespace(
        obs_steps=int(args.obs_steps),
        horizon_steps=int(args.horizon_steps),
        pwm_max=float(args.pwm_max),
    )
    chunks = build_chunk_rows(raw_df, chunk_args, timestamp)

    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_ROOT / f"chunk_raw_metrics_{label}_{timestamp}.csv"
    chunks.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary = {
        "timestamp": timestamp,
        "run_label": label,
        "raw_dirs": [str(path) for path in raw_dirs],
        "trajectory_count": int(len(trajectory_paths)),
        "raw_rows": int(len(raw_df)),
        "chunk_rows": int(len(chunks)),
        "obs_steps": int(args.obs_steps),
        "horizon_steps": int(args.horizon_steps),
        "output_path": str(output_path),
    }
    summary_path = PROCESSED_ROOT / f"chunk_raw_metrics_{label}_{timestamp}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
