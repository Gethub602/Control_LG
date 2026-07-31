import argparse
import json
import random
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))

from config import (  # noqa: E402
    ESP32_BAUDRATE,
    ESP32_PORT,
    ESP32_REAL_GAIN_DB_MODE,
    ESP32_REAL_PID_GAIN_DB,
    ESP32_TIMEOUT,
    REAL_PWM_MAX,
    REAL_PWM_MIN,
)
from motor_interface import ESP32MotorInterface  # noqa: E402
from pid_controller import PIDController  # noqa: E402
from local_kafka_controller import interpolate_gain_from_db  # noqa: E402


RAW_ROOT = MOTOR_DIR / "data" / "raw" / "diffusion_gain_chunk_db"
PROCESSED_ROOT = MOTOR_DIR / "data" / "processed" / "diffusion_gain_chunk_db"
SUMMARY_DIR = MOTOR_DIR / "results" / "summary"
CONTROL_DT = 0.10
V1_POLICY_FEATURE_COLS = [
    "target",
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
    "current_lag_1",
    "error_lag_1",
    "pwm_lag_1",
    "current_lag_2",
    "error_lag_2",
    "pwm_lag_2",
    "current_lag_3",
    "error_lag_3",
    "pwm_lag_3",
    "abs_error",
    "signed_error_ratio",
    "accel_demand",
    "decel_demand",
    "speed_ratio",
    "abs_error_derivative",
    "previous_target",
    "target_delta",
    "abs_target_delta",
    "target_direction",
    "target_change_count",
]
SEQUENCE_RAW_COLS = [
    "target",
    "current",
    "rpm",
    "error",
    "error_derivative",
    "pwm",
    "raw_pwm",
    "kp",
    "ki",
    "kd",
    "integral",
    "pid_p_term",
    "pid_i_term",
    "pid_d_term",
    "time_since_start",
    "time_since_target_change",
    "loop_elapsed_sec",
    "motor_step_elapsed_sec",
]

SAFE_GAIN_BOUNDS = {
    "kp": (0.55, 1.45),
    "ki": (0.70, 2.50),
    "kd": (0.00, 0.12),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect local PID-only raw trajectories for diffusion gain chunk DB."
    )
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--run-time", type=float, default=12.0)
    parser.add_argument("--rest-time", type=float, default=3.0)
    parser.add_argument("--cooldown-every", type=int, default=10)
    parser.add_argument("--cooldown-time", type=float, default=30.0)
    parser.add_argument("--target-min", type=float, default=65.0)
    parser.add_argument("--target-max", type=float, default=105.0)
    parser.add_argument("--pwm-max", type=float, default=140.0)
    parser.add_argument("--pwm-rate-limit", type=float, default=30.0)
    parser.add_argument("--obs-steps", type=int, default=10)
    parser.add_argument("--horizon-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sampling-mode",
        choices=["balanced", "random"],
        default="balanced",
        help="balanced cycles scenario/gain profile types to reduce accidental DB skew.",
    )
    parser.add_argument("--run-label", default="diffusion_pilot")
    return parser.parse_args()


def safe_label(value: str):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value))


def apply_pwm_rate_limit(pwm_cmd: float, prev_pwm: float, rate_limit: float):
    delta = float(pwm_cmd) - float(prev_pwm)
    limit = abs(float(rate_limit))
    if limit <= 0:
        return float(pwm_cmd)
    delta = max(-limit, min(limit, delta))
    return float(prev_pwm + delta)


def random_gain(rng: random.Random):
    return {
        "kp": rng.uniform(*SAFE_GAIN_BOUNDS["kp"]),
        "ki": rng.uniform(*SAFE_GAIN_BOUNDS["ki"]),
        "kd": rng.choice([0.0, 0.03, 0.06, 0.10, 0.12]),
    }


def db_gain(target: float):
    kp, ki, kd = interpolate_gain_from_db(
        target=float(target),
        gain_db=ESP32_REAL_PID_GAIN_DB,
        fallback_gain=(0.8, 1.6, 0.03),
        mode=ESP32_REAL_GAIN_DB_MODE,
    )
    return {"kp": float(kp), "ki": float(ki), "kd": float(kd)}


def target_change_context(t: float, scenario: dict, segment_idx: int):
    targets = scenario["targets"]
    change_times = scenario["change_times"]
    if segment_idx <= 0:
        previous_target = float(targets[0])
        current_target = float(targets[0])
        last_change_time = 0.0
        target_change_count = 0
    else:
        previous_target = float(targets[segment_idx - 1])
        current_target = float(targets[segment_idx])
        last_change_time = float(change_times[segment_idx - 1])
        target_change_count = int(segment_idx)

    target_delta = current_target - previous_target
    return {
        "previous_target": previous_target,
        "target_delta": float(target_delta),
        "abs_target_delta": float(abs(target_delta)),
        "target_direction": float(np.sign(target_delta)),
        "target_change_count": float(target_change_count),
        "time_since_target_change": float(max(0.0, t - last_change_time)),
    }


def add_raw_state_features(df: pd.DataFrame, pwm_max: float):
    if df.empty:
        return df

    out = df.copy()
    out = out.sort_values(["trajectory_id", "step"]).reset_index(drop=True)
    out["current"] = out.get("current", out["rpm"])
    out["error"] = out.get("error", out["target"] - out["current"])
    out["abs_error"] = out["error"].abs()
    target_abs = out["target"].abs().clip(lower=1e-6)
    out["signed_error_ratio"] = out["error"] / target_abs
    out["error_ratio"] = out["error"] / target_abs
    out["speed_ratio"] = out["current"] / target_abs
    out["pwm_ratio"] = out["pwm"] / max(float(pwm_max), 1e-6)
    out["abs_error_derivative"] = out["error_derivative"].abs()
    out["accel_demand"] = (out["error"] > 0.0).astype(float)
    out["decel_demand"] = (out["error"] < 0.0).astype(float)

    db_kp = []
    db_ki = []
    db_kd = []
    for target in out["target"].to_numpy(dtype=float):
        gain = db_gain(float(target))
        db_kp.append(gain["kp"])
        db_ki.append(gain["ki"])
        db_kd.append(gain["kd"])
    out["db_base_kp"] = db_kp
    out["db_base_ki"] = db_ki
    out["db_base_kd"] = db_kd
    out["kp_scale"] = out["kp"] / out["db_base_kp"].replace(0.0, np.nan)
    out["ki_scale"] = out["ki"] / out["db_base_ki"].replace(0.0, np.nan)
    out["kd_scale"] = out["kd"] / out["db_base_kd"].replace(0.0, np.nan)
    out[["kp_scale", "ki_scale", "kd_scale"]] = out[
        ["kp_scale", "ki_scale", "kd_scale"]
    ].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for lag in range(1, 6):
        for col in ["current", "rpm", "error", "pwm", "kp", "ki", "kd", "target"]:
            out[f"{col}_lag_{lag}"] = out.groupby("trajectory_id")[col].shift(lag)

    for col in out.columns:
        if col.endswith(tuple(f"_lag_{lag}" for lag in range(1, 6))):
            base_col = col.rsplit("_lag_", 1)[0]
            out[col] = out[col].fillna(out[base_col])

    return out


def make_scenario(run_idx: int, args, rng: random.Random):
    scenario_types = ["step", "multistep3", "multistep4", "zigzag", "edge"]
    gain_profile_types = ["fixed_random", "piecewise_random", "smooth_random", "db"]
    if args.sampling_mode == "balanced":
        scenario_type = scenario_types[(int(run_idx) - 1) % len(scenario_types)]
        gain_profile_type = gain_profile_types[(int(run_idx) - 1) % len(gain_profile_types)]
    else:
        scenario_type = rng.choice(scenario_types)
        gain_profile_type = rng.choice(gain_profile_types)
    run_time = float(args.run_time)

    if scenario_type == "step":
        targets = [
            rng.uniform(args.target_min, args.target_max),
            rng.uniform(args.target_min, args.target_max),
        ]
        change_times = [run_time * 0.5]
    elif scenario_type == "multistep3":
        targets = [
            rng.uniform(args.target_min, args.target_max),
            rng.uniform(args.target_min, args.target_max),
            rng.uniform(args.target_min, args.target_max),
        ]
        change_times = [run_time / 3.0, run_time * 2.0 / 3.0]
    elif scenario_type == "multistep4":
        targets = [
            rng.uniform(args.target_min, args.target_max),
            rng.uniform(args.target_min, args.target_max),
            rng.uniform(args.target_min, args.target_max),
            rng.uniform(args.target_min, args.target_max),
        ]
        change_times = [run_time * 0.25, run_time * 0.50, run_time * 0.75]
    elif scenario_type == "zigzag":
        low = rng.uniform(args.target_min, min(args.target_min + 12.0, args.target_max))
        high = rng.uniform(max(args.target_max - 15.0, args.target_min), args.target_max)
        mid = rng.uniform(args.target_min + 8.0, args.target_max - 8.0)
        targets = [low, high, mid, rng.choice([low, high])]
        change_times = [run_time * 0.25, run_time * 0.50, run_time * 0.75]
    else:
        targets = [
            rng.choice([args.target_min, args.target_max]),
            rng.uniform(args.target_min + 8.0, args.target_max - 8.0),
            rng.choice([args.target_max, args.target_min]),
        ]
        change_times = [run_time / 3.0, run_time * 2.0 / 3.0]

    targets = [round(float(value), 1) for value in targets]
    if gain_profile_type == "fixed_random":
        params = {"gain": random_gain(rng)}
    elif gain_profile_type == "piecewise_random":
        params = {"gains": [random_gain(rng) for _ in targets]}
    elif gain_profile_type == "smooth_random":
        params = {"start_gain": random_gain(rng), "end_gain": random_gain(rng)}
    else:
        params = {"mode": "target_interpolated_db"}

    return {
        "run_idx": int(run_idx),
        "scenario_type": scenario_type,
        "targets": targets,
        "change_times": [round(float(value), 3) for value in change_times],
        "gain_profile_type": gain_profile_type,
        "gain_profile_params": params,
        "run_time": run_time,
    }


def get_target_at(t: float, scenario: dict):
    targets = scenario["targets"]
    change_times = scenario["change_times"]
    idx = 0
    for change_time in change_times:
        if t >= change_time:
            idx += 1
    return float(targets[min(idx, len(targets) - 1)]), int(min(idx, len(targets) - 1))


def get_gain_at(t: float, scenario: dict, target: float, segment_idx: int):
    profile = scenario["gain_profile_type"]
    params = scenario["gain_profile_params"]
    if profile == "fixed_random":
        return params["gain"]
    if profile == "piecewise_random":
        return params["gains"][min(segment_idx, len(params["gains"]) - 1)]
    if profile == "smooth_random":
        alpha = min(max(t / max(float(scenario["run_time"]), 1e-6), 0.0), 1.0)
        start = params["start_gain"]
        end = params["end_gain"]
        return {
            key: float(start[key] * (1.0 - alpha) + end[key] * alpha)
            for key in ["kp", "ki", "kd"]
        }
    return db_gain(target)


def run_trajectory(motor, scenario: dict, args, trajectory_id: str):
    motor.stop()
    time.sleep(float(args.rest_time))

    n_steps = int(float(scenario["run_time"]) / CONTROL_DT)
    first_target, first_segment = get_target_at(0.0, scenario)
    first_gain = get_gain_at(0.0, scenario, first_target, first_segment)
    pid = PIDController(
        kp=first_gain["kp"],
        ki=first_gain["ki"],
        kd=first_gain["kd"],
        dt=CONTROL_DT,
        output_min=REAL_PWM_MIN,
        output_max=float(args.pwm_max),
    )

    rows = []
    current_rpm = motor.get_state().current
    prev_error = 0.0
    prev_pwm = 0.0
    aborted = False
    abort_reason = "none"
    # WSL's realtime clock can jump forward when it synchronizes with the
    # Windows host.  Use a monotonic clock for every elapsed-time decision so
    # target transitions and 10 Hz sample timestamps cannot jump with it.
    start_monotonic = time.monotonic()

    try:
        for step in range(n_steps):
            loop_start = time.monotonic()
            t = time.monotonic() - start_monotonic
            target, segment_idx = get_target_at(t, scenario)
            transition = target_change_context(t, scenario, segment_idx)
            gain = get_gain_at(t, scenario, target, segment_idx)
            pid.set_gains(gain["kp"], gain["ki"], gain["kd"])

            current_before = float(current_rpm)
            error = target - current_before
            error_derivative = (error - prev_error) / CONTROL_DT
            pid_state_before = pid.get_state()
            tentative_integral = float(pid_state_before["integral"] + error * CONTROL_DT)
            pid_p_term = float(gain["kp"] * error)
            pid_i_term = float(gain["ki"] * tentative_integral)
            pid_d_term = float(gain["kd"] * error_derivative)
            compute_start = time.monotonic()
            raw_pwm = pid.compute(target, current_rpm)
            compute_elapsed_sec = time.monotonic() - compute_start
            pwm = apply_pwm_rate_limit(raw_pwm, prev_pwm, float(args.pwm_rate_limit))
            pwm = float(np.clip(pwm, REAL_PWM_MIN, float(args.pwm_max)))

            motor_step_start = time.monotonic()
            state = motor.step(pwm)
            motor_step_elapsed_sec = time.monotonic() - motor_step_start
            current_rpm = float(state.current)
            measured_error_post = float(target - current_rpm)
            encoder = np.nan
            esp32_reported_pwm = np.nan
            if state.raw is not None:
                encoder = state.raw.get("encoder", np.nan)
                esp32_reported_pwm = state.raw.get("pwm", np.nan)

            pid_state = pid.get_state()
            loop_elapsed_sec = time.monotonic() - loop_start
            sleep_time = max(0.0, CONTROL_DT - loop_elapsed_sec)
            rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "run_idx": int(scenario["run_idx"]),
                    "step": int(step),
                    "time": float(t),
                    "wall_time": float(time.time()),
                    "state_timestamp": float(state.timestamp),
                    "control_dt": float(CONTROL_DT),
                    "loop_elapsed_sec": float(loop_elapsed_sec),
                    "pid_compute_elapsed_sec": float(compute_elapsed_sec),
                    "motor_step_elapsed_sec": float(motor_step_elapsed_sec),
                    "sleep_time_sec": float(sleep_time),
                    "target": float(target),
                    "target_segment_idx": int(segment_idx),
                    "previous_target": float(transition["previous_target"]),
                    "target_delta": float(transition["target_delta"]),
                    "abs_target_delta": float(transition["abs_target_delta"]),
                    "target_direction": float(transition["target_direction"]),
                    "target_change_count": float(transition["target_change_count"]),
                    "time_since_start": float(t),
                    "time_since_target_change": float(
                        transition["time_since_target_change"]
                    ),
                    "current": float(current_before),
                    "rpm_before": float(current_before),
                    "rpm": float(current_rpm),
                    "rpm_after": float(current_rpm),
                    "error": float(error),
                    "control_error": float(error),
                    "measured_error_post": float(measured_error_post),
                    "error_derivative": float(error_derivative),
                    "pwm": float(pwm),
                    "raw_pwm": float(raw_pwm),
                    "esp32_reported_pwm": float(esp32_reported_pwm),
                    "prev_pwm": float(prev_pwm),
                    "pwm_delta": float(pwm - prev_pwm),
                    "pwm_saturated": bool(abs(raw_pwm - pwm) > 1e-9),
                    "pwm_near_limit": bool(pwm >= float(args.pwm_max) * 0.90),
                    "encoder": encoder,
                    "kp": float(gain["kp"]),
                    "ki": float(gain["ki"]),
                    "kd": float(gain["kd"]),
                    "pid_p_term": float(pid_p_term),
                    "pid_i_term": float(pid_i_term),
                    "pid_d_term": float(pid_d_term),
                    "pid_integral_before": float(pid_state_before["integral"]),
                    "integral": float(pid_state["integral"]),
                    "pid_prev_error_before": float(pid_state_before["prev_error"]),
                    "pid_prev_error_after": float(pid_state["prev_error"]),
                    "pid_prev_output_before": float(pid_state_before["prev_output"]),
                    "pid_prev_output_after": float(pid_state["prev_output"]),
                    "scenario_type": scenario["scenario_type"],
                    "gain_profile_type": scenario["gain_profile_type"],
                    "aborted": bool(aborted),
                    "abort_reason": abort_reason,
                }
            )

            if step % 10 == 0:
                print(
                    f"step={step:04d}, t={t:.2f}, target={target:.1f}, "
                    f"rpm={current_rpm:.2f}, error={error:.2f}, pwm={pwm:.2f}, "
                    f"gain=({gain['kp']:.3f},{gain['ki']:.3f},{gain['kd']:.3f})"
                )

            if abs(current_rpm) > 180.0:
                aborted = True
                abort_reason = "rpm_exceeded_safe_limit"
                print(f"[ABORT] {abort_reason}: rpm={current_rpm:.2f}")
                break

            prev_error = error
            prev_pwm = pwm
            time.sleep(sleep_time)
    finally:
        motor.stop()
        time.sleep(float(args.rest_time))

    df = pd.DataFrame(rows)
    if not df.empty:
        df["aborted"] = bool(aborted)
        df["abort_reason"] = abort_reason
    return df, aborted, abort_reason


def sequence_to_json(values):
    return json.dumps([float(value) for value in values], ensure_ascii=True)


def add_sequence_payload(row: dict, prefix: str, frame: pd.DataFrame, columns):
    for col in columns:
        if col in frame.columns:
            row[f"{prefix}_{col}_seq"] = sequence_to_json(
                frame[col].to_numpy(dtype=float)
            )


def build_chunk_rows(raw_df: pd.DataFrame, args, timestamp: str):
    rows = []
    obs_steps = int(args.obs_steps)
    horizon_steps = int(args.horizon_steps)
    pwm_max = float(args.pwm_max)

    for trajectory_id, group in raw_df.groupby("trajectory_id", sort=False):
        group = group.sort_values("step").reset_index(drop=True)
        for start in range(obs_steps, len(group) - horizon_steps):
            obs = group.iloc[start - obs_steps : start]
            future = group.iloc[start : start + horizon_steps]
            if obs.empty or future.empty:
                continue
            if future["aborted"].astype(bool).any():
                continue

            target = future["target"].to_numpy(dtype=float)
            current = future["current"].to_numpy(dtype=float)
            rpm = future["rpm"].to_numpy(dtype=float)
            error = future["error"].to_numpy(dtype=float)
            post_error = future["measured_error_post"].to_numpy(dtype=float)
            pwm = future["pwm"].to_numpy(dtype=float)
            kp = future["kp"].to_numpy(dtype=float)
            ki = future["ki"].to_numpy(dtype=float)
            kd = future["kd"].to_numpy(dtype=float)
            time_values = future["time"].to_numpy(dtype=float)
            dt_values = np.diff(time_values, prepend=float(group.iloc[start - 1]["time"]))
            dt_values = np.where(dt_values <= 0.0, CONTROL_DT, dt_values)

            abs_error = np.abs(error)
            overshoot = np.maximum(rpm - target, 0.0)
            pwm_variation = np.sum(np.abs(np.diff(pwm, prepend=float(group.iloc[start - 1]["pwm"]))))
            gain_variation = (
                np.sum(np.abs(np.diff(kp, prepend=float(group.iloc[start - 1]["kp"]))))
                + np.sum(np.abs(np.diff(ki, prepend=float(group.iloc[start - 1]["ki"]))))
                + np.sum(np.abs(np.diff(kd, prepend=float(group.iloc[start - 1]["kd"]))))
            )

            rows.append(
                {
                    "sample_id": f"{timestamp}_{trajectory_id}_chunk_{start:04d}",
                    "trajectory_id": trajectory_id,
                    "run_idx": int(group.iloc[start]["run_idx"]),
                    "chunk_start_step": int(group.iloc[start]["step"]),
                    "chunk_start_time": float(group.iloc[start]["time"]),
                    "obs_steps": obs_steps,
                    "horizon_steps": horizon_steps,
                    "horizon_sec": float(horizon_steps * CONTROL_DT),
                    "scenario_type": str(group.iloc[start]["scenario_type"]),
                    "gain_profile_type": str(group.iloc[start]["gain_profile_type"]),
                    "target_start": float(target[0]),
                    "target_end": float(target[-1]),
                    "target_delta_in_chunk": float(target[-1] - target[0]),
                    "target_change_inside_chunk": bool(
                        np.any(np.abs(target - target[0]) > 1e-9)
                    ),
                    "current_start": float(current[0]),
                    "rpm_start": float(rpm[0]),
                    "error_start": float(error[0]),
                    "chunk_iae": float(np.sum(abs_error * dt_values)),
                    "chunk_post_iae": float(np.sum(np.abs(post_error) * dt_values)),
                    "chunk_mean_abs_error": float(np.mean(abs_error)),
                    "chunk_max_abs_error": float(np.max(abs_error)),
                    "chunk_overshoot": float(np.max(overshoot)),
                    "chunk_overshoot_ratio": float(np.max(overshoot / np.maximum(np.abs(target), 1e-6))),
                    "chunk_pwm_mean": float(np.mean(np.abs(pwm))),
                    "chunk_pwm_max": float(np.max(pwm)),
                    "chunk_pwm_variation": float(pwm_variation),
                    "chunk_saturation_ratio": float(np.mean(pwm >= pwm_max * 0.98)),
                    "chunk_near_saturation_ratio": float(np.mean(pwm >= pwm_max * 0.90)),
                    "chunk_gain_variation": float(gain_variation),
                }
            )
            row = rows[-1]

            for feature in V1_POLICY_FEATURE_COLS:
                if feature in group.columns:
                    row[f"state_{feature}"] = float(group.iloc[start][feature])

            add_sequence_payload(row, "obs", obs, SEQUENCE_RAW_COLS)
            add_sequence_payload(row, "future", future, SEQUENCE_RAW_COLS)
            add_sequence_payload(row, "future", future, ["kp", "ki", "kd"])

    return pd.DataFrame(rows)


def write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    args = parse_args()
    rng = random.Random(int(args.seed))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = safe_label(args.run_label)
    raw_dir = RAW_ROOT / f"{label}_{timestamp}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    raw_frames = []
    interrupted = False

    run_config = {
        "timestamp": timestamp,
        "run_label": label,
        "args": vars(args),
        "control_dt": CONTROL_DT,
        "elapsed_clock": "time.monotonic",
        "wall_clock": "time.time",
        "esp32_port": ESP32_PORT,
        "esp32_baudrate": ESP32_BAUDRATE,
        "esp32_timeout": ESP32_TIMEOUT,
        "real_pwm_min": REAL_PWM_MIN,
        "real_pwm_max_config": REAL_PWM_MAX,
        "safe_gain_bounds": SAFE_GAIN_BOUNDS,
        "v1_policy_feature_cols": V1_POLICY_FEATURE_COLS,
        "sequence_raw_cols": SEQUENCE_RAW_COLS,
    }
    run_config_path = raw_dir / f"run_config_{label}_{timestamp}.json"
    write_json(run_config_path, run_config)

    print("=" * 80)
    print("Diffusion gain chunk DB pilot collection")
    print(f"Runs: {args.num_runs}")
    print(f"Output raw dir: {raw_dir}")
    print("=" * 80)

    motor = ESP32MotorInterface(
        port=ESP32_PORT,
        baudrate=ESP32_BAUDRATE,
        timeout=ESP32_TIMEOUT,
        pwm_min=REAL_PWM_MIN,
        pwm_max=float(args.pwm_max),
    )

    try:
        try:
            for run_idx in range(1, int(args.num_runs) + 1):
                scenario = make_scenario(run_idx, args, rng)
                trajectory_id = f"{label}_{timestamp}_{run_idx:04d}_{uuid.uuid4().hex[:8]}"
                print("-" * 80)
                print(f"[{run_idx}/{args.num_runs}] trajectory_id={trajectory_id}")
                print(
                    f"scenario={scenario['scenario_type']}, targets={scenario['targets']}, "
                    f"change_times={scenario['change_times']}, gain_profile={scenario['gain_profile_type']}"
                )
                df, aborted, abort_reason = run_trajectory(
                    motor=motor,
                    scenario=scenario,
                    args=args,
                    trajectory_id=trajectory_id,
                )
                df = add_raw_state_features(df, pwm_max=float(args.pwm_max))

                trajectory_path = raw_dir / f"trajectory_{run_idx:04d}_{trajectory_id}.csv"
                df.to_csv(trajectory_path, index=False, encoding="utf-8-sig")
                raw_frames.append(df)

                metadata_rows.append(
                    {
                        "trajectory_id": trajectory_id,
                        "run_idx": int(run_idx),
                        "scenario_type": scenario["scenario_type"],
                        "targets": json.dumps(scenario["targets"], ensure_ascii=True),
                        "change_times": json.dumps(scenario["change_times"], ensure_ascii=True),
                        "gain_profile_type": scenario["gain_profile_type"],
                        "gain_profile_params": json.dumps(scenario["gain_profile_params"], ensure_ascii=True),
                        "run_time": float(scenario["run_time"]),
                        "rest_time": float(args.rest_time),
                        "pwm_max": float(args.pwm_max),
                        "pwm_rate_limit": float(args.pwm_rate_limit),
                        "rows": int(len(df)),
                        "aborted": bool(aborted),
                        "abort_reason": abort_reason,
                        "trajectory_path": str(trajectory_path),
                    }
                )

                metadata_checkpoint_path = raw_dir / f"metadata_checkpoint_{label}_{timestamp}.csv"
                pd.DataFrame(metadata_rows).to_csv(
                    metadata_checkpoint_path, index=False, encoding="utf-8-sig"
                )

                if (
                    int(args.cooldown_every) > 0
                    and run_idx % int(args.cooldown_every) == 0
                    and run_idx < int(args.num_runs)
                ):
                    print(f"[COOLDOWN] {args.cooldown_time:.1f}s")
                    motor.stop()
                    time.sleep(float(args.cooldown_time))
        except KeyboardInterrupt:
            interrupted = True
            print("[INTERRUPTED] Saving completed trajectories and partial summary.")
    finally:
        motor.close()

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_path = raw_dir / f"metadata_{label}_{timestamp}.csv"
    metadata_df.to_csv(metadata_path, index=False, encoding="utf-8-sig")

    if raw_frames:
        raw_df = pd.concat(raw_frames, ignore_index=True)
    else:
        raw_df = pd.DataFrame()

    raw_aggregate_path = raw_dir / f"raw_trajectories_{label}_{timestamp}.csv"
    raw_df.to_csv(raw_aggregate_path, index=False, encoding="utf-8-sig")

    chunk_df = build_chunk_rows(raw_df, args, timestamp=timestamp) if not raw_df.empty else pd.DataFrame()
    chunk_path = PROCESSED_ROOT / f"chunk_raw_metrics_{label}_{timestamp}.csv"
    chunk_df.to_csv(chunk_path, index=False, encoding="utf-8-sig")

    schema = {
        "control_dt": CONTROL_DT,
        "elapsed_clock": "time.monotonic",
        "wall_clock": "time.time",
        "v1_policy_feature_cols": V1_POLICY_FEATURE_COLS,
        "sequence_raw_cols": SEQUENCE_RAW_COLS,
        "safe_gain_bounds": SAFE_GAIN_BOUNDS,
        "notes": [
            "current/rpm_before are the decision-time measured speed values.",
            "rpm/rpm_after are the post-command measured speed values.",
            "error/control_error are computed at decision time.",
            "measured_error_post is computed after applying the PWM command.",
            "chunk rows store raw metrics only; downstream label weights should be applied later.",
        ],
    }
    schema_path = raw_dir / f"schema_{label}_{timestamp}.json"
    schema_path.write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "timestamp": timestamp,
        "run_label": label,
        "num_runs_requested": int(args.num_runs),
        "num_runs_completed": int(len(metadata_df)),
        "interrupted": bool(interrupted),
        "num_aborted": int(metadata_df["aborted"].sum()) if not metadata_df.empty else 0,
        "raw_rows": int(len(raw_df)),
        "chunk_rows": int(len(chunk_df)),
        "raw_dir": str(raw_dir),
        "run_config_path": str(run_config_path),
        "metadata_path": str(metadata_path),
        "raw_aggregate_path": str(raw_aggregate_path),
        "chunk_raw_metrics_path": str(chunk_path),
        "schema_path": str(schema_path),
        "v1_policy_feature_count": int(len(V1_POLICY_FEATURE_COLS)),
        "sequence_raw_col_count": int(len(SEQUENCE_RAW_COLS)),
    }
    summary_path = SUMMARY_DIR / f"diffusion_gain_chunk_db_pilot_summary_{label}_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 80)
    print("Diffusion gain chunk DB pilot finished")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
