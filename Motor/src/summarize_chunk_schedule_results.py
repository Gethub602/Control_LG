import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))

from config import RESULTS_DIR


KAFKA_RESULT_DIR = RESULTS_DIR / "kafka_control"
SUMMARY_DIR = RESULTS_DIR / "summary"


PRIMARY_METRICS = [
    "IAE",
    "mean_abs_error",
    "final_error",
    "after_change_IAE",
    "after_change_max_error",
    "settling_time_after_change",
    "mean_pwm",
    "total_pwm",
    "max_pwm",
    "saturation_ratio_percent",
    "near_high_saturation_ratio_percent",
    "schedule_gain_applied_count",
    "schedule_chunk_accepted_count",
    "schedule_fallback_count",
    "server_gain_applied_count",
    "local_gain_reduction_count",
    "local_gain_recovery_count",
    "min_kp_scale",
    "min_ki_scale",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize chunk schedule Kafka controller experiments."
    )
    parser.add_argument(
        "--backend",
        default="esp32",
        help="Backend label to include, e.g. esp32 or simulation.",
    )
    parser.add_argument(
        "--latest-per-mode",
        action="store_true",
        help="Keep only the newest metrics row for each schedule apply mode.",
    )
    parser.add_argument(
        "--latest-per-label",
        action="store_true",
        help="Keep only the newest metrics row for each run label.",
    )
    parser.add_argument(
        "--since",
        default="",
        help="Optional timestamp prefix filter, e.g. 20260430.",
    )
    return parser.parse_args()


def list_metric_files(backend: str, since: str = ""):
    patterns = [
        f"local_kafka_controller_metrics_{backend}_*.csv",
        f"local_kafka_controller_metrics_{backend}_delay_aware_*.csv",
        f"local_kafka_controller_metrics_{backend}_naive_*.csv",
    ]

    files = []
    for pattern in patterns:
        files.extend(KAFKA_RESULT_DIR.glob(pattern))

    unique_files = sorted(set(files), key=lambda path: path.stat().st_mtime)

    if since:
        unique_files = [path for path in unique_files if since in path.name]

    return unique_files


def infer_mode_from_filename(path: Path):
    name = path.name
    if "_delay_aware_" in name:
        return "delay_aware"
    if "_naive_" in name:
        return "naive"
    return "legacy_or_unknown"


def infer_run_label_from_filename(path: Path, backend: str, mode: str):
    stem = path.stem
    prefix = f"local_kafka_controller_metrics_{backend}_"

    if not stem.startswith(prefix):
        return ""

    rest = stem[len(prefix) :]

    if mode in ["delay_aware", "naive"] and rest.startswith(f"{mode}_"):
        rest = rest[len(mode) + 1 :]

    parts = rest.split("_")
    if len(parts) <= 2:
        return ""

    # Last two parts are usually YYYYMMDD_HHMMSS.
    return "_".join(parts[:-2])


def load_metrics_table(files):
    rows = []

    for path in files:
        df = pd.read_csv(path)
        if df.empty:
            continue

        row = df.iloc[0].to_dict()
        row["metrics_file"] = path.name
        row["metrics_mtime"] = path.stat().st_mtime
        row["metrics_datetime"] = datetime.fromtimestamp(path.stat().st_mtime).isoformat(
            timespec="seconds"
        )

        if "schedule_apply_mode" not in row or pd.isna(row["schedule_apply_mode"]):
            row["schedule_apply_mode"] = infer_mode_from_filename(path)

        row["run_label"] = infer_run_label_from_filename(
            path,
            backend=str(row.get("backend", "")) if "backend" in row else "esp32",
            mode=str(row["schedule_apply_mode"]),
        )

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def keep_latest_per_mode(metrics_df: pd.DataFrame):
    if metrics_df.empty:
        return metrics_df

    return (
        metrics_df.sort_values("metrics_mtime")
        .groupby("schedule_apply_mode", as_index=False)
        .tail(1)
        .sort_values("schedule_apply_mode")
        .reset_index(drop=True)
    )


def keep_latest_per_label(metrics_df: pd.DataFrame):
    if metrics_df.empty:
        return metrics_df

    group_cols = ["schedule_apply_mode", "run_label"]

    return (
        metrics_df.sort_values("metrics_mtime")
        .groupby(group_cols, as_index=False)
        .tail(1)
        .sort_values(group_cols)
        .reset_index(drop=True)
    )


def calc_improvement(reference_value, candidate_value, lower_is_better=True):
    if pd.isna(reference_value) or pd.isna(candidate_value):
        return np.nan

    if abs(float(reference_value)) <= 1e-12:
        return np.nan

    if lower_is_better:
        return (float(reference_value) - float(candidate_value)) / abs(float(reference_value)) * 100.0

    return (float(candidate_value) - float(reference_value)) / abs(float(reference_value)) * 100.0


def build_comparison_table(metrics_df: pd.DataFrame):
    if metrics_df.empty:
        return pd.DataFrame()

    table_cols = [
        "schedule_apply_mode",
        "run_label",
        "backend",
        "control_dt",
        "target_before",
        "target_after",
        "metrics_datetime",
        "metrics_file",
    ]
    table_cols.extend(metric for metric in PRIMARY_METRICS if metric in metrics_df.columns)

    return metrics_df[table_cols].copy()


def build_delay_aware_vs_naive(metrics_df: pd.DataFrame):
    if metrics_df.empty:
        return pd.DataFrame()

    mode_df = metrics_df.set_index("schedule_apply_mode", drop=False)

    if "naive" not in mode_df.index or "delay_aware" not in mode_df.index:
        return pd.DataFrame()

    naive = mode_df.loc["naive"]
    delay_aware = mode_df.loc["delay_aware"]

    if isinstance(naive, pd.DataFrame):
        naive = naive.sort_values("metrics_mtime").iloc[-1]
    if isinstance(delay_aware, pd.DataFrame):
        delay_aware = delay_aware.sort_values("metrics_mtime").iloc[-1]

    lower_is_better_metrics = {
        "IAE",
        "mean_abs_error",
        "final_error",
        "after_change_IAE",
        "after_change_max_error",
        "settling_time_after_change",
        "mean_pwm",
        "total_pwm",
        "max_pwm",
        "saturation_ratio_percent",
        "near_high_saturation_ratio_percent",
        "schedule_fallback_count",
    }

    rows = []
    for metric in PRIMARY_METRICS:
        if metric not in metrics_df.columns:
            continue

        naive_value = naive.get(metric, np.nan)
        delay_value = delay_aware.get(metric, np.nan)
        lower_is_better = metric in lower_is_better_metrics

        rows.append(
            {
                "metric": metric,
                "naive": naive_value,
                "delay_aware": delay_value,
                "difference_delay_aware_minus_naive": delay_value - naive_value
                if not pd.isna(naive_value) and not pd.isna(delay_value)
                else np.nan,
                "delay_aware_improvement_percent": calc_improvement(
                    naive_value,
                    delay_value,
                    lower_is_better=lower_is_better,
                ),
            }
        )

    return pd.DataFrame(rows)


def save_markdown_summary(
    comparison_df: pd.DataFrame,
    improvement_df: pd.DataFrame,
    timestamp: str,
):
    path = SUMMARY_DIR / f"chunk_schedule_summary_{timestamp}.md"

    lines = []
    lines.append("# Chunk Schedule Experiment Summary")
    lines.append("")

    if comparison_df.empty:
        lines.append("No chunk schedule metrics found.")
    else:
        lines.append("## Latest Metrics")
        lines.append("")
        lines.append(comparison_df.to_markdown(index=False))
        lines.append("")

    if not improvement_df.empty:
        lines.append("## Delay-Aware vs Naive")
        lines.append("")
        lines.append(improvement_df.to_markdown(index=False))
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {path}")
    return path


def main():
    args = parse_args()

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    files = list_metric_files(backend=args.backend, since=args.since)

    if not files:
        raise FileNotFoundError(
            f"No chunk schedule metrics found in {KAFKA_RESULT_DIR}"
        )

    metrics_df = load_metrics_table(files)

    if args.latest_per_mode:
        metrics_df = keep_latest_per_mode(metrics_df)

    if args.latest_per_label:
        metrics_df = keep_latest_per_label(metrics_df)

    comparison_df = build_comparison_table(metrics_df)
    improvement_df = build_delay_aware_vs_naive(metrics_df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    comparison_path = SUMMARY_DIR / f"chunk_schedule_metrics_{timestamp}.csv"
    improvement_path = SUMMARY_DIR / f"chunk_schedule_delayaware_vs_naive_{timestamp}.csv"

    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    improvement_df.to_csv(improvement_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {comparison_path}")
    print(f"Saved: {improvement_path}")

    save_markdown_summary(comparison_df, improvement_df, timestamp)

    print("")
    print("Comparison")
    print(comparison_df.to_string(index=False))

    if not improvement_df.empty:
        print("")
        print("Delay-aware vs naive")
        print(improvement_df.to_string(index=False))


if __name__ == "__main__":
    main()
