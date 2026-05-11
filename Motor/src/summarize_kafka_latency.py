import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
sys.path.append(str(ROOT_DIR))

from config import RESULTS_DIR


KAFKA_DIR = RESULTS_DIR / "kafka_control"
SUMMARY_DIR = RESULTS_DIR / "summary"


LATENCY_COLUMNS = [
    "schedule_source_to_accept_sec",
    "schedule_source_to_apply_sec",
    "schedule_generated_to_accept_sec",
    "schedule_generated_to_apply_sec",
    "schedule_publish_to_accept_sec",
    "schedule_generator_duration_sec",
    "schedule_state_to_generation_sec",
    "schedule_timing_slack_sec",
    "schedule_accepted_control_lag_sec",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize measured Kafka schedule chunk latency."
    )
    parser.add_argument(
        "--pattern",
        default="local_kafka_controller_log_esp32_delay_aware_*latency*.csv",
        help="Glob pattern under results/kafka_control.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Controller sample time used to round recommended delay.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.9,
        help="Quantile used for recommended inference delay.",
    )
    return parser.parse_args()


def load_logs(pattern: str):
    paths = sorted(KAFKA_DIR.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not paths:
        raise FileNotFoundError(f"No logs found for pattern: {pattern}")

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["source_file"] = path.name
        frames.append(df)

    return pd.concat(frames, ignore_index=True), paths


def stats(values):
    values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "p50": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }

    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def round_up_to_dt(value: float, dt: float):
    if not np.isfinite(value):
        return np.nan
    return float(np.ceil(value / dt) * dt)


def summarize(df: pd.DataFrame, dt: float, quantile: float):
    schedule_df = df[df.get("schedule_source", "") == "schedule_chunk"].copy()

    group_cols = []
    for col in ["schedule_generator_mode", "schedule_model_family"]:
        if col in schedule_df.columns:
            group_cols.append(col)

    rows = []
    groups = [(("all", "all"), schedule_df)]

    if group_cols:
        groups = []
        for key, group in schedule_df.groupby(group_cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            while len(key) < 2:
                key = (*key, "all")
            groups.append((key, group))

    for key, group in groups:
        row = {
            "schedule_generator_mode": key[0],
            "schedule_model_family": key[1],
            "rows": int(len(group)),
        }

        for col in LATENCY_COLUMNS:
            if col in group.columns:
                col_stats = stats(group[col])
                for stat_name, value in col_stats.items():
                    row[f"{col}_{stat_name}"] = value

        source_to_accept = pd.to_numeric(
            group.get("schedule_source_to_accept_sec", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()

        if len(source_to_accept) > 0:
            q_value = float(source_to_accept.quantile(quantile))
            row["recommended_inference_delay_quantile"] = float(quantile)
            row["recommended_inference_delay_sec"] = round_up_to_dt(q_value, dt)
            row["recommended_inference_delay_raw_sec"] = q_value
        else:
            row["recommended_inference_delay_quantile"] = float(quantile)
            row["recommended_inference_delay_sec"] = np.nan
            row["recommended_inference_delay_raw_sec"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df, paths = load_logs(args.pattern)
    summary = summarize(df, dt=args.dt, quantile=args.quantile)

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = SUMMARY_DIR / f"kafka_latency_summary_{timestamp}.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"Loaded logs: {[path.name for path in paths]}")
    print(summary.to_string(index=False))
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
