"""
Merge several chunk_raw_metrics CSVs into one before labelling.

Collection happens in separate runs (a pilot, then the main batch), each writing
its own chunk_raw_metrics file. Labelling takes a single input, so the runs have
to be concatenated first. They share a schema by construction -- the same
collector wrote them -- but this checks rather than assumes, because a silent
column mismatch would produce a dataset whose feature set differs from what the
models expect.

sample_id and trajectory_id already carry each run's timestamp, so rows stay
distinguishable and the trajectory-grouped split keeps working across runs.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))

PROCESSED_ROOT = MOTOR_DIR / "data" / "processed" / "diffusion_gain_chunk_db"
RAW_ROOT = MOTOR_DIR / "data" / "raw" / "diffusion_gain_chunk_db"


def parse_args():
    p = argparse.ArgumentParser(description="Merge chunk_raw_metrics CSVs.")
    p.add_argument("inputs", nargs="*",
                   help="Input CSVs. Defaults to every chunk_raw_metrics_*.csv.")
    p.add_argument("--pattern", default="chunk_raw_metrics_*.csv",
                   help="Glob used when no inputs are given.")
    p.add_argument("--exclude", default="sim",
                   help="Skip inputs whose name contains this substring. "
                        "Defaults to 'sim' so simulation data is not mixed in.")
    p.add_argument("--label", default="merged")
    p.add_argument(
        "--max-time-gap",
        type=float,
        help="Keep only trajectories whose raw time axis has no gap above this many seconds. "
             "Use 0.2 to reject the pre-monotonic WSL clock jumps.",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.inputs:
        paths = [Path(p) if Path(p).is_absolute() else MOTOR_DIR / p
                 for p in args.inputs]
    else:
        paths = sorted(PROCESSED_ROOT.glob(args.pattern))
        if args.exclude:
            paths = [p for p in paths if args.exclude not in p.name]

    if not paths:
        print("No input files matched.")
        return 1

    print("Inputs:")
    frames, columns = [], None
    for path in paths:
        if not path.exists():
            print(f"  MISSING {path}")
            return 1
        df = pd.read_csv(path)
        trajectories = df["trajectory_id"].nunique() if "trajectory_id" in df else 0
        print(f"  {path.name}  rows={len(df):>7}  trajectories={trajectories:>5}")
        if columns is None:
            columns = list(df.columns)
        elif list(df.columns) != columns:
            only_a = set(columns) - set(df.columns)
            only_b = set(df.columns) - set(columns)
            print("  COLUMN MISMATCH -- refusing to merge.")
            if only_a:
                print(f"    missing here : {sorted(only_a)[:8]}")
            if only_b:
                print(f"    extra here   : {sorted(only_b)[:8]}")
            return 1
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)

    if args.max_time_gap is not None:
        wanted = set(merged["trajectory_id"].astype(str).unique())
        seen, rejected = set(), set()
        for raw_path in RAW_ROOT.glob("*/trajectory_*.csv"):
            # The trajectory id is also inside the CSV; reading only two small
            # columns avoids loading the wide raw schema during filtering.
            raw = pd.read_csv(raw_path, usecols=["trajectory_id", "time"])
            if raw.empty:
                continue
            trajectory_id = str(raw["trajectory_id"].iloc[0])
            if trajectory_id not in wanted:
                continue
            seen.add(trajectory_id)
            dt = raw["time"].astype(float).diff().dropna()
            if bool((dt > float(args.max_time_gap)).any()):
                rejected.add(trajectory_id)
        missing = wanted - seen
        if missing:
            print(f"Missing raw trajectory files for {len(missing)} ids; refusing to guess.")
            return 1
        before = len(merged)
        merged = merged[~merged["trajectory_id"].astype(str).isin(rejected)].reset_index(drop=True)
        print(f"clock-gap filter   : rejected {len(rejected)} trajectories, "
              f"{before - len(merged)} chunk rows (threshold={args.max_time_gap}s)")

    dupes = merged["sample_id"].duplicated().sum() if "sample_id" in merged else 0
    if dupes:
        print(f"WARNING: {dupes} duplicate sample_id rows; dropping duplicates.")
        merged = merged.drop_duplicates(subset="sample_id").reset_index(drop=True)

    print()
    print(f"merged rows        : {len(merged)}")
    if "trajectory_id" in merged:
        print(f"merged trajectories: {merged['trajectory_id'].nunique()}")
    if "scenario_type" in merged:
        print("scenario mix       :")
        for k, v in merged["scenario_type"].value_counts().items():
            print(f"    {k:<14}{v}")

    if args.dry_run:
        print("\n(dry run, nothing written)")
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = PROCESSED_ROOT / f"chunk_raw_metrics_{args.label}_{timestamp}.csv"
    merged.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out}")
    print("\nNext:")
    print(f"  python src/label_diffusion_gain_chunks.py --chunk-path {out.relative_to(MOTOR_DIR)} \\")
    print(f"      --profile tracking_first --top-k 5 --label {args.label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
