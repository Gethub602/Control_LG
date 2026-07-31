"""
Simulation-backed gain-chunk dataset collector.

collect_diffusion_gain_chunk_db.py requires real ESP32 hardware. This script
produces the *same* chunk_raw_metrics schema from SimpleMotorEnv so the rest of
the pipeline (label_diffusion_gain_chunks.py -> train_* ) can run without a motor.

It reuses build_chunk_rows / add_raw_state_features from the real collector, so
the produced columns stay in sync with the hardware path by construction.
"""

import argparse
import json
import random
import sys
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))
sys.path.append(str(CURRENT_DIR))

from motor_env import SimpleMotorEnv  # noqa: E402
from pid_controller import PIDController  # noqa: E402
from collect_diffusion_gain_chunk_db import (  # noqa: E402
    CONTROL_DT,
    SAFE_GAIN_BOUNDS,
    add_raw_state_features,
    build_chunk_rows,
    db_gain,
    make_scenario,
    get_target_at,
    get_gain_at,
    target_change_context,
    apply_pwm_rate_limit,
)

RAW_ROOT = MOTOR_DIR / "data" / "raw" / "diffusion_gain_chunk_db"
PROCESSED_ROOT = MOTOR_DIR / "data" / "processed" / "diffusion_gain_chunk_db"
SUMMARY_DIR = MOTOR_DIR / "results" / "summary"


def parse_args():
    p = argparse.ArgumentParser(description="Collect gain-chunk DB from simulation.")
    p.add_argument("--runs", type=int, default=120)
    p.add_argument("--run-time", type=float, default=20.0)
    p.add_argument("--target-min", type=float, default=30.0)
    p.add_argument("--target-max", type=float, default=100.0)
    p.add_argument("--obs-steps", type=int, default=10)
    p.add_argument("--horizon-steps", type=int, default=20)
    p.add_argument("--pwm-min", type=float, default=0.0)
    p.add_argument("--pwm-max", type=float, default=140.0)
    p.add_argument("--pwm-rate-limit", type=float, default=20.0)
    p.add_argument("--sampling-mode", default="balanced", choices=["balanced", "random"])
    p.add_argument("--rest-time", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label", default="sim")
    # motor plant variation, so the dataset is not a single fixed plant
    p.add_argument("--tau-min", type=float, default=0.30)
    p.add_argument("--tau-max", type=float, default=0.50)
    p.add_argument("--k-min", type=float, default=0.85)
    p.add_argument("--k-max", type=float, default=1.15)
    return p.parse_args()


def run_trajectory_sim(scenario, args, trajectory_id, rng):
    """Mirror of collect_diffusion_gain_chunk_db.run_trajectory on SimpleMotorEnv."""
    env = SimpleMotorEnv(
        dt=CONTROL_DT,
        k_motor=rng.uniform(args.k_min, args.k_max),
        tau_motor=rng.uniform(args.tau_min, args.tau_max),
        pwm_min=args.pwm_min,
        pwm_max=args.pwm_max,
        use_disturbance=False,
    )
    env.reset()

    pid = PIDController(dt=CONTROL_DT, output_min=args.pwm_min, output_max=args.pwm_max)

    n_steps = int(args.run_time / CONTROL_DT)
    rows = []
    prev_error = 0.0
    prev_pwm = 0.0
    aborted = False
    abort_reason = ""

    for step in range(n_steps):
        t = step * CONTROL_DT
        target, segment_idx = get_target_at(t, scenario)
        transition = target_change_context(t, scenario, segment_idx)

        gain = get_gain_at(t, scenario, target, segment_idx)
        pid.set_gains(gain["kp"], gain["ki"], gain["kd"])

        current_before = env.get_state()
        error = target - current_before
        error_derivative = (error - prev_error) / CONTROL_DT

        state_before = pid.get_state()
        raw_pwm = pid.compute(target, current_before)
        pwm = apply_pwm_rate_limit(raw_pwm, prev_pwm, args.pwm_rate_limit)
        pwm = float(np.clip(pwm, args.pwm_min, args.pwm_max))

        env.step(pwm)
        current_rpm = env.get_state()
        measured_error_post = target - current_rpm

        state_after = pid.get_state()
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "run_idx": scenario["run_idx"],
                "step": step,
                "time": float(t),
                "target": float(target),
                "target_segment_idx": int(segment_idx),
                "previous_target": float(transition["previous_target"]),
                "target_delta": float(transition["target_delta"]),
                "abs_target_delta": float(transition["abs_target_delta"]),
                "target_direction": float(transition["target_direction"]),
                "target_change_count": float(transition["target_change_count"]),
                "time_since_start": float(t),
                "time_since_target_change": float(transition["time_since_target_change"]),
                "current": float(current_before),
                "rpm": float(current_rpm),
                "error": float(error),
                "measured_error_post": float(measured_error_post),
                "error_derivative": float(error_derivative),
                "pwm": float(pwm),
                "raw_pwm": float(raw_pwm),
                "prev_pwm": float(prev_pwm),
                "kp": float(gain["kp"]),
                "ki": float(gain["ki"]),
                "kd": float(gain["kd"]),
                "pid_p_term": float(gain["kp"] * error),
                "pid_i_term": float(gain["ki"] * state_after["integral"]),
                "pid_d_term": float(gain["kd"] * error_derivative),
                "integral": float(state_after["integral"]),
                "pid_integral_before": float(state_before["integral"]),
                "loop_elapsed_sec": float(CONTROL_DT),
                "motor_step_elapsed_sec": float(CONTROL_DT),
                "scenario_type": scenario["scenario_type"],
                "gain_profile_type": scenario["gain_profile_type"],
                "aborted": False,
                "abort_reason": "",
            }
        )

        if abs(current_rpm) > 180.0:
            aborted = True
            abort_reason = "rpm_exceeded_safe_limit"
            break

        prev_error = error
        prev_pwm = pwm

    df = pd.DataFrame(rows)
    if not df.empty:
        df["aborted"] = bool(aborted)
        df["abort_reason"] = abort_reason
    return df, aborted, abort_reason


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    aborted_count = 0
    for run_idx in range(1, int(args.runs) + 1):
        scenario = make_scenario(run_idx, args, rng)
        trajectory_id = f"{timestamp}_sim_{run_idx:04d}_{uuid.uuid4().hex[:6]}"
        df, aborted, _ = run_trajectory_sim(scenario, args, trajectory_id, rng)
        if aborted:
            aborted_count += 1
        if not df.empty:
            frames.append(df)
        if run_idx % 20 == 0:
            print(f"  collected {run_idx}/{args.runs} trajectories")

    raw_df = pd.concat(frames, ignore_index=True)
    raw_df = add_raw_state_features(raw_df, pwm_max=args.pwm_max)

    raw_path = RAW_ROOT / f"raw_steps_{args.label}_{timestamp}.csv"
    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")

    chunk_df = build_chunk_rows(raw_df, args, timestamp)
    chunk_path = PROCESSED_ROOT / f"chunk_raw_metrics_{args.label}_{timestamp}.csv"
    chunk_df.to_csv(chunk_path, index=False, encoding="utf-8-sig")

    summary = {
        "timestamp": timestamp,
        "runs": int(args.runs),
        "aborted_runs": int(aborted_count),
        "raw_steps": int(len(raw_df)),
        "chunks": int(len(chunk_df)),
        "raw_path": str(raw_path),
        "chunk_path": str(chunk_path),
        "obs_steps": int(args.obs_steps),
        "horizon_steps": int(args.horizon_steps),
        "backend": "simulation",
    }
    (SUMMARY_DIR / f"sim_gain_chunk_db_summary_{args.label}_{timestamp}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
