import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))

from config import PROCESSED_DATA_DIR, RESULTS_DIR, REAL_PWM_MAX


ESP32_GAIN_SWEEP_DIR = RESULTS_DIR / "esp32_gain_sweep"
ESP32_CONTROL_COMPARISON_DIR = RESULTS_DIR / "esp32_control_comparison"
KAFKA_CONTROL_DIR = RESULTS_DIR / "kafka_control"


BASE_FEATURE_COLUMNS = [
    "target",
    "previous_target",
    "target_delta",
    "abs_target_delta",
    "target_direction",
    "target_change_count",
    "current",
    "error",
    "error_derivative",
    "pwm",
    "prev_pwm",
    "kp",
    "ki",
    "kd",
    "integral",
    "kp_scale",
    "ki_scale",
    "time_since_start",
    "time_since_target_change",
    "error_ratio",
    "pwm_ratio",
]

LAG_COLUMNS = [
    "current",
    "error",
    "pwm",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build ESP32 state/gain -> short-horizon cost dataset."
    )
    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=10,
        help="Future horizon length in rows.",
    )
    parser.add_argument(
        "--lag-steps",
        type=int,
        default=3,
        help="Number of lagged state rows to include as features.",
    )
    parser.add_argument(
        "--include-kafka",
        action="store_true",
        help="Include Kafka controller logs in addition to ESP32 sweep/control logs.",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Use only latest aggregate gain sweep log and latest controller logs.",
    )
    parser.add_argument(
        "--output-name",
        default="",
        help="Optional output filename. Defaults to timestamped CSV.",
    )
    parser.add_argument(
        "--save-latest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also overwrite esp32_horizon_cost_dataset_latest.csv.",
    )
    return parser.parse_args()


def get_latest_file(folder: Path, pattern: str):
    files = sorted(folder.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not files:
        return None
    return files[-1]


def list_source_files(include_kafka: bool, latest_only: bool):
    files = []

    if latest_only:
        latest_sweep = get_latest_file(
            ESP32_GAIN_SWEEP_DIR,
            "esp32_gain_sweep_log_*.csv",
        )
        if latest_sweep is not None:
            files.append(("esp32_gain_sweep", latest_sweep))

        latest_fixed = get_latest_file(
            ESP32_CONTROL_COMPARISON_DIR,
            "esp32_fixed_pid_log_*.csv",
        )
        if latest_fixed is not None:
            files.append(("esp32_fixed_pid", latest_fixed))

        latest_db = get_latest_file(
            ESP32_CONTROL_COMPARISON_DIR,
            "esp32_local_gain_db_log_*.csv",
        )
        if latest_db is not None:
            files.append(("esp32_local_gain_db", latest_db))

        if include_kafka:
            latest_kafka = get_latest_file(
                KAFKA_CONTROL_DIR,
                "local_kafka_controller_log_esp32*.csv",
            )
            if latest_kafka is not None:
                files.append(("esp32_kafka", latest_kafka))

        return files

    files.extend(
        ("esp32_gain_sweep", path)
        for path in sorted(ESP32_GAIN_SWEEP_DIR.glob("esp32_gain_sweep_log_*.csv"))
    )
    files.extend(
        ("esp32_fixed_pid", path)
        for path in sorted(ESP32_CONTROL_COMPARISON_DIR.glob("esp32_fixed_pid_log_*.csv"))
    )
    files.extend(
        ("esp32_local_gain_db", path)
        for path in sorted(ESP32_CONTROL_COMPARISON_DIR.glob("esp32_local_gain_db_log_*.csv"))
    )

    if include_kafka:
        files.extend(
            ("esp32_kafka", path)
            for path in sorted(KAFKA_CONTROL_DIR.glob("local_kafka_controller_log_esp32*.csv"))
        )

    return files


def standardize_log(df: pd.DataFrame, source_kind: str, source_file: Path):
    out = pd.DataFrame()

    if "case_id" in df.columns:
        out["trajectory_id"] = df["case_id"].astype(str)
    elif "run_id" in df.columns:
        out["trajectory_id"] = df["run_id"].astype(str)
    else:
        out["trajectory_id"] = pd.Series(
            [source_file.stem] * len(df),
            index=df.index,
        )

    out["source_kind"] = pd.Series([source_kind] * len(df), index=df.index)
    out["source_file"] = pd.Series([source_file.name] * len(df), index=df.index)

    out["step"] = df["step"].astype(int) if "step" in df.columns else np.arange(len(df))
    out["time"] = df["time"].astype(float) if "time" in df.columns else out["step"].astype(float)
    out["target"] = df["target"].astype(float)

    if "current" in df.columns:
        out["current"] = df["current"].astype(float)
    elif "rpm" in df.columns:
        out["current"] = df["rpm"].astype(float)
    else:
        raise ValueError(f"No current/rpm column in {source_file}")

    if "error" in df.columns:
        out["error"] = df["error"].astype(float)
    elif "measured_error" in df.columns:
        out["error"] = df["measured_error"].astype(float)
    else:
        out["error"] = out["target"] - out["current"]

    if "error_derivative" in df.columns:
        out["error_derivative"] = df["error_derivative"].astype(float)
    else:
        out["error_derivative"] = np.nan

    out["pwm"] = df["pwm"].astype(float)

    if "prev_pwm" in df.columns:
        out["prev_pwm"] = df["prev_pwm"].astype(float)
    else:
        out["prev_pwm"] = out.groupby("trajectory_id")["pwm"].shift(1).fillna(0.0)

    if "base_kp" in df.columns:
        out["kp"] = df["base_kp"].astype(float)
    elif "kp" in df.columns:
        out["kp"] = df["kp"].astype(float)
    else:
        out["kp"] = np.nan

    if "base_ki" in df.columns:
        out["ki"] = df["base_ki"].astype(float)
    elif "ki" in df.columns:
        out["ki"] = df["ki"].astype(float)
    else:
        out["ki"] = np.nan

    if "base_kd" in df.columns:
        out["kd"] = df["base_kd"].astype(float)
    elif "kd" in df.columns:
        out["kd"] = df["kd"].astype(float)
    else:
        out["kd"] = 0.0

    out["integral"] = df["integral"].astype(float) if "integral" in df.columns else 0.0
    out["kp_scale"] = df["kp_scale"].astype(float) if "kp_scale" in df.columns else 1.0
    out["ki_scale"] = df["ki_scale"].astype(float) if "ki_scale" in df.columns else 1.0

    out["aborted"] = df["aborted"].astype(bool) if "aborted" in df.columns else False

    out = out.sort_values(["trajectory_id", "step"]).reset_index(drop=True)

    if out["error_derivative"].isna().any():
        out["error_derivative"] = (
            out.groupby("trajectory_id")["error"].diff().fillna(0.0)
            / out.groupby("trajectory_id")["time"].diff().replace(0.0, np.nan).fillna(0.1)
        )

    out["time_since_start"] = out.groupby("trajectory_id")["time"].transform(
        lambda s: s - s.iloc[0]
    )
    out["time_since_target_change"] = calc_time_since_target_change(out)
    add_transition_context_features(out)
    out["error_ratio"] = out["error"].abs() / out["target"].abs().clip(lower=1e-6)
    out["pwm_ratio"] = out["pwm"].abs() / float(REAL_PWM_MAX)

    return out


def calc_time_since_target_change(df: pd.DataFrame):
    values = np.zeros(len(df), dtype=float)

    for _, group in df.groupby("trajectory_id", sort=False):
        times = group["time"].to_numpy(dtype=float)
        targets = group["target"].to_numpy(dtype=float)
        idx_values = group.index.to_numpy()

        last_change_time = times[0] if len(times) else 0.0
        previous_target = targets[0] if len(targets) else 0.0

        for local_idx, global_idx in enumerate(idx_values):
            if abs(targets[local_idx] - previous_target) > 1e-9:
                last_change_time = times[local_idx]
                previous_target = targets[local_idx]

            values[global_idx] = times[local_idx] - last_change_time

    return values


def add_transition_context_features(df: pd.DataFrame):
    previous_target_values = np.zeros(len(df), dtype=float)
    target_delta_values = np.zeros(len(df), dtype=float)
    abs_target_delta_values = np.zeros(len(df), dtype=float)
    target_direction_values = np.zeros(len(df), dtype=float)
    target_change_count_values = np.zeros(len(df), dtype=float)

    for _, group in df.groupby("trajectory_id", sort=False):
        ordered = group.sort_values("step")
        targets = ordered["target"].to_numpy(dtype=float)
        idx_values = ordered.index.to_numpy()
        if len(targets) == 0:
            continue

        previous_segment_target = float(targets[0])
        current_segment_target = float(targets[0])
        target_change_count = 0

        for local_idx, global_idx in enumerate(idx_values):
            target = float(targets[local_idx])
            if abs(target - current_segment_target) > 1e-9:
                previous_segment_target = current_segment_target
                current_segment_target = target
                target_change_count += 1

            delta = current_segment_target - previous_segment_target
            previous_target_values[global_idx] = previous_segment_target
            target_delta_values[global_idx] = delta
            abs_target_delta_values[global_idx] = abs(delta)
            target_direction_values[global_idx] = np.sign(delta)
            target_change_count_values[global_idx] = float(target_change_count)

    df["previous_target"] = previous_target_values
    df["target_delta"] = target_delta_values
    df["abs_target_delta"] = abs_target_delta_values
    df["target_direction"] = target_direction_values
    df["target_change_count"] = target_change_count_values


def add_lag_features(df: pd.DataFrame, lag_steps: int):
    out = df.copy()

    for lag in range(1, lag_steps + 1):
        for column in LAG_COLUMNS:
            out[f"{column}_lag_{lag}"] = (
                out.groupby("trajectory_id")[column].shift(lag)
            )

    return out


def build_horizon_rows(df: pd.DataFrame, horizon_steps: int, lag_steps: int):
    df = add_lag_features(df, lag_steps=lag_steps)
    rows = []

    for _, group in df.groupby("trajectory_id", sort=False):
        group = group.sort_values("step").reset_index(drop=True)

        for i in range(len(group) - horizon_steps):
            row = group.iloc[i]
            future = group.iloc[i + 1 : i + horizon_steps + 1]

            if row.get("aborted", False):
                continue

            if future.empty:
                continue

            if future["aborted"].any():
                continue

            if any(pd.isna(row.get(f"{column}_lag_{lag}", np.nan)) for lag in range(1, lag_steps + 1) for column in LAG_COLUMNS):
                continue

            target = float(row["target"])
            future_current = future["current"].to_numpy(dtype=float)
            future_error = future["target"].to_numpy(dtype=float) - future_current
            future_pwm = future["pwm"].to_numpy(dtype=float)

            dt_values = future["time"].diff().fillna(
                max(float(future["time"].iloc[0] - row["time"]), 0.1)
            ).replace(0.0, 0.1)

            abs_error = np.abs(future_error)
            overshoot = np.maximum(future_current - target, 0.0)
            pwm_variation = np.abs(np.diff(future_pwm, prepend=float(row["pwm"])))
            saturation = future_pwm >= float(REAL_PWM_MAX) * 0.98
            near_saturation = future_pwm >= float(REAL_PWM_MAX) * 0.90

            horizon_iae = float(np.sum(abs_error * dt_values.to_numpy(dtype=float)))
            horizon_mean_abs_error = float(np.mean(abs_error))
            horizon_max_abs_error = float(np.max(abs_error))
            horizon_overshoot = float(np.max(overshoot))
            horizon_overshoot_ratio = horizon_overshoot / max(abs(target), 1e-6)
            horizon_mean_pwm = float(np.mean(np.abs(future_pwm)))
            horizon_pwm_variation = float(np.sum(pwm_variation))
            horizon_saturation_ratio = float(np.mean(saturation))
            horizon_near_saturation_ratio = float(np.mean(near_saturation))

            horizon_cost = (
                horizon_iae
                + 20.0 * horizon_overshoot_ratio
                + 5.0 * horizon_saturation_ratio
                + 2.0 * horizon_near_saturation_ratio
                + 0.002 * horizon_mean_pwm
                + 0.001 * horizon_pwm_variation
            )

            sample = {
                "source_kind": row["source_kind"],
                "source_file": row["source_file"],
                "trajectory_id": row["trajectory_id"],
                "step": int(row["step"]),
                "time": float(row["time"]),
                "horizon_steps": int(horizon_steps),
                "future_time_end": float(future["time"].iloc[-1]),
                "horizon_iae": horizon_iae,
                "horizon_mean_abs_error": horizon_mean_abs_error,
                "horizon_max_abs_error": horizon_max_abs_error,
                "horizon_overshoot": horizon_overshoot,
                "horizon_overshoot_ratio": horizon_overshoot_ratio,
                "horizon_mean_pwm": horizon_mean_pwm,
                "horizon_pwm_variation": horizon_pwm_variation,
                "horizon_saturation_ratio": horizon_saturation_ratio,
                "horizon_near_saturation_ratio": horizon_near_saturation_ratio,
                "horizon_cost": float(horizon_cost),
            }

            for column in BASE_FEATURE_COLUMNS:
                sample[column] = float(row[column])

            for lag in range(1, lag_steps + 1):
                for column in LAG_COLUMNS:
                    sample[f"{column}_lag_{lag}"] = float(row[f"{column}_lag_{lag}"])

            rows.append(sample)

    return pd.DataFrame(rows)


def build_dataset(source_files, horizon_steps: int, lag_steps: int):
    frames = []

    for source_kind, path in source_files:
        try:
            raw_df = pd.read_csv(path)
            standardized = standardize_log(raw_df, source_kind, path)
            frame = build_horizon_rows(
                standardized,
                horizon_steps=horizon_steps,
                lag_steps=lag_steps,
            )

            if not frame.empty:
                frames.append(frame)
                print(f"Loaded {path.name}: {len(frame)} samples")
            else:
                print(f"Skipped {path.name}: no usable samples")

        except Exception as exc:
            print(f"Skipped {path}: {exc}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def main():
    args = parse_args()

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    source_files = list_source_files(
        include_kafka=args.include_kafka,
        latest_only=args.latest_only,
    )

    if not source_files:
        raise FileNotFoundError("No ESP32 logs found for dataset construction.")

    dataset = build_dataset(
        source_files=source_files,
        horizon_steps=args.horizon_steps,
        lag_steps=args.lag_steps,
    )

    if dataset.empty:
        raise RuntimeError("No usable dataset rows were generated.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = args.output_name
    if not output_name:
        output_name = f"esp32_horizon_cost_dataset_{timestamp}.csv"

    output_path = PROCESSED_DATA_DIR / output_name
    latest_path = PROCESSED_DATA_DIR / "esp32_horizon_cost_dataset_latest.csv"

    dataset.to_csv(output_path, index=False, encoding="utf-8-sig")
    if args.save_latest:
        dataset.to_csv(latest_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print(f"Saved dataset: {output_path}")
    if args.save_latest:
        print(f"Saved latest copy: {latest_path}")
    else:
        print("Skipped latest copy update")
    print(f"Rows: {len(dataset)}")
    print(f"Columns: {len(dataset.columns)}")
    print("Source counts:")
    print(dataset["source_kind"].value_counts().to_string())
    print("Horizon cost summary:")
    print(dataset["horizon_cost"].describe().to_string())


if __name__ == "__main__":
    main()
