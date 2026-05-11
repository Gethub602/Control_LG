import sys
from pathlib import Path
from datetime import datetime
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Path setting
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))

from motor_interface import ESP32MotorInterface
from pid_controller import PIDController
from config import (
    ESP32_PORT,
    ESP32_BAUDRATE,
    ESP32_TIMEOUT,
    REAL_PWM_MIN,
    REAL_PWM_MAX,
    RESULTS_DIR,
    FIGURE_DIR,
)


# ============================================================
# Experiment settings
# ============================================================

MODE_NAME = "esp32_fixed_pid"
RESULT_DIR = RESULTS_DIR / "esp32_control_comparison"

CONTROL_DT = 0.10
SIM_TIME = 10.0
N_STEPS = int(SIM_TIME / CONTROL_DT)

TARGET_BEFORE = 70.0
TARGET_AFTER = 100.0
TARGET_CHANGE_TIME = 3.0

# Fixed PID gain
# ESP32 gain sweep 결과 기반 baseline
FIXED_KP = 1.2
FIXED_KI = 0.9
FIXED_KD = 0.0

PWM_MIN = REAL_PWM_MIN
PWM_MAX = REAL_PWM_MAX

# 실물 모터 보호용 PWM 변화율 제한
PWM_RATE_LIMIT = 50.0

# settling 기준
SETTLING_TOLERANCE_RATIO = 0.02


# ============================================================
# Utility
# ============================================================

def get_target_at_time(t: float) -> float:
    if t < TARGET_CHANGE_TIME:
        return TARGET_BEFORE
    return TARGET_AFTER


def apply_pwm_rate_limit(pwm_cmd: float, prev_pwm: float, rate_limit: float) -> float:
    delta = pwm_cmd - prev_pwm
    delta = np.clip(delta, -rate_limit, rate_limit)
    return float(prev_pwm + delta)


def compute_dt(time_arr: np.ndarray):
    if len(time_arr) <= 1:
        return np.zeros_like(time_arr)

    dt_arr = np.diff(time_arr, prepend=time_arr[0])
    dt_arr[0] = 0.0
    return dt_arr


def calculate_metrics(df: pd.DataFrame) -> dict:
    time_arr = df["time"].to_numpy(dtype=float)
    target = df["target"].to_numpy(dtype=float)
    rpm = df["current"].to_numpy(dtype=float)
    pwm = df["pwm"].to_numpy(dtype=float)

    error = target - rpm
    abs_error = np.abs(error)
    dt_arr = compute_dt(time_arr)

    iae = float(np.sum(abs_error * dt_arr))
    mean_abs_error = float(np.mean(abs_error))
    final_error = float(abs_error[-1])

    after_change_mask = time_arr >= TARGET_CHANGE_TIME

    if after_change_mask.any():
        after_change_iae = float(
            np.sum(abs_error[after_change_mask] * dt_arr[after_change_mask])
        )
        after_change_max_error = float(np.max(abs_error[after_change_mask]))
    else:
        after_change_iae = np.nan
        after_change_max_error = np.nan

    max_rpm = float(np.max(rpm))
    overshoot = max(0.0, max_rpm - TARGET_AFTER)
    overshoot_percent = overshoot / max(abs(TARGET_AFTER), 1e-6) * 100.0

    high_saturation = pwm >= PWM_MAX - 1e-9
    saturation_ratio_percent = float(np.mean(high_saturation) * 100.0)
    saturation_duration = float(np.sum(dt_arr[high_saturation]))

    tolerance = SETTLING_TOLERANCE_RATIO * abs(TARGET_AFTER)
    settling_time_after_change = np.nan

    after_indices = np.where(after_change_mask)[0]

    for idx in after_indices:
        if np.all(np.abs(target[idx:] - rpm[idx:]) <= tolerance):
            settling_time_after_change = float(time_arr[idx] - TARGET_CHANGE_TIME)
            break

    return {
        "mode": MODE_NAME,
        "backend": "esp32",
        "control_dt": CONTROL_DT,
        "target_before": TARGET_BEFORE,
        "target_after": TARGET_AFTER,
        "target_change_time": TARGET_CHANGE_TIME,

        "kp": FIXED_KP,
        "ki": FIXED_KI,
        "kd": FIXED_KD,

        "IAE": iae,
        "mean_abs_error": mean_abs_error,
        "final_error": final_error,
        "after_change_IAE": after_change_iae,
        "after_change_max_error": after_change_max_error,
        "settling_time_after_change": settling_time_after_change,
        "overshoot_percent": overshoot_percent,

        "mean_pwm": float(np.mean(np.abs(pwm))),
        "max_pwm": float(np.max(pwm)),
        "min_pwm": float(np.min(pwm)),
        "saturation_ratio_percent": saturation_ratio_percent,
        "saturation_duration": saturation_duration,

        "server_gain_applied_count": 0,
        "server_gain_discarded_count": 0,
        "duplicate_gain_discard_count": 0,
        "unsafe_gain_discard_count": 0,
        "local_gain_reduction_count": 0,
        "local_gain_recovery_count": 0,
    }


def plot_result(df: pd.DataFrame, timestamp: str):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # RPM response
    plt.figure(figsize=(10, 5))
    plt.plot(df["time"], df["current"], label="measured RPM")
    plt.plot(df["time"], df["target"], linestyle="--", label="target RPM")
    plt.axvline(TARGET_CHANGE_TIME, linestyle="--", linewidth=1.2, label="target change")
    plt.xlabel("Time [s]")
    plt.ylabel("RPM")
    plt.title("ESP32 Fixed PID RPM Response")
    plt.legend()
    plt.grid(True)

    save_path = FIGURE_DIR / f"{MODE_NAME}_rpm_response_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()

    # PWM command
    plt.figure(figsize=(10, 5))
    plt.plot(df["time"], df["raw_pwm"], linestyle="--", label="raw PWM")
    plt.plot(df["time"], df["pwm"], label="applied PWM")
    plt.axvline(TARGET_CHANGE_TIME, linestyle="--", linewidth=1.2, label="target change")
    plt.xlabel("Time [s]")
    plt.ylabel("PWM")
    plt.title("ESP32 Fixed PID PWM Command")
    plt.legend()
    plt.grid(True)

    save_path = FIGURE_DIR / f"{MODE_NAME}_pwm_command_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()

    # Error response
    plt.figure(figsize=(10, 5))
    plt.plot(df["time"], df["error"], label="measured error")
    plt.axhline(0.0, linewidth=1.0)
    plt.axvline(TARGET_CHANGE_TIME, linestyle="--", linewidth=1.2, label="target change")
    plt.xlabel("Time [s]")
    plt.ylabel("Error [RPM]")
    plt.title("ESP32 Fixed PID Error Response")
    plt.legend()
    plt.grid(True)

    save_path = FIGURE_DIR / f"{MODE_NAME}_error_response_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ESP32 Fixed PID Baseline")
    print("=" * 80)
    print(f"Port: {ESP32_PORT}")
    print(f"Baudrate: {ESP32_BAUDRATE}")
    print(f"Control DT: {CONTROL_DT}")
    print(f"Target: {TARGET_BEFORE} -> {TARGET_AFTER} at {TARGET_CHANGE_TIME}s")
    print(f"Fixed PID: Kp={FIXED_KP}, Ki={FIXED_KI}, Kd={FIXED_KD}")
    print(f"PWM limit: {PWM_MIN} ~ {PWM_MAX}")
    print(f"PWM rate limit: {PWM_RATE_LIMIT}")
    print("=" * 80)

    motor = ESP32MotorInterface(
        port=ESP32_PORT,
        baudrate=ESP32_BAUDRATE,
        timeout=ESP32_TIMEOUT,
        pwm_min=PWM_MIN,
        pwm_max=PWM_MAX,
    )

    pid = PIDController(
        kp=FIXED_KP,
        ki=FIXED_KI,
        kd=FIXED_KD,
        dt=CONTROL_DT,
        output_min=PWM_MIN,
        output_max=PWM_MAX,
    )

    rows = []

    prev_error = 0.0
    prev_pwm = 0.0

    try:
        print("PING:", motor.ping())

        motor.stop()
        time.sleep(0.5)

        state = motor.get_state()
        current = state.current
        prev_error = TARGET_BEFORE - current

        print(f"Initial RPM: {current:.3f}")

        for step in range(N_STEPS):
            loop_start = time.time()

            t = step * CONTROL_DT
            target = get_target_at_time(t)

            control_current = current
            control_error = target - control_current
            error_derivative = (control_error - prev_error) / CONTROL_DT

            raw_pwm = pid.compute(target, control_current)

            pwm = apply_pwm_rate_limit(
                pwm_cmd=raw_pwm,
                prev_pwm=prev_pwm,
                rate_limit=PWM_RATE_LIMIT,
            )
            pwm = float(np.clip(pwm, PWM_MIN, PWM_MAX))

            state = motor.step(pwm)
            measured_current = state.current
            measured_error = target - measured_current

            encoder = np.nan
            if state.raw is not None:
                encoder = state.raw.get("encoder", np.nan)

            rows.append(
                {
                    "mode": MODE_NAME,
                    "backend": "esp32",
                    "step": step,
                    "time": t,

                    "target": target,

                    "control_current": control_current,
                    "control_error": control_error,

                    "current": measured_current,
                    "measured_error": measured_error,
                    "error": measured_error,
                    "error_derivative": error_derivative,

                    "raw_pwm": raw_pwm,
                    "pwm": pwm,
                    "prev_pwm": prev_pwm,

                    "encoder": encoder,

                    "kp": FIXED_KP,
                    "ki": FIXED_KI,
                    "kd": FIXED_KD,
                }
            )

            if step % 10 == 0:
                print(
                    f"step={step:04d}, "
                    f"t={t:.2f}, "
                    f"target={target:.1f}, "
                    f"rpm={measured_current:.2f}, "
                    f"err={measured_error:.2f}, "
                    f"raw_pwm={raw_pwm:.2f}, "
                    f"pwm={pwm:.2f}"
                )

            current = measured_current
            prev_error = measured_error
            prev_pwm = pwm

            elapsed = time.time() - loop_start
            sleep_time = max(0.0, CONTROL_DT - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Stopping motor...")

    finally:
        motor.stop()
        time.sleep(0.3)
        motor.close()

    result_df = pd.DataFrame(rows)
    metrics = calculate_metrics(result_df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_path = RESULT_DIR / f"{MODE_NAME}_log_{timestamp}.csv"
    metric_path = RESULT_DIR / f"{MODE_NAME}_metrics_{timestamp}.csv"

    result_df.to_csv(log_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([metrics]).to_csv(metric_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("ESP32 Fixed PID Baseline Finished")
    print(f"Saved log: {log_path}")
    print(f"Saved metrics: {metric_path}")
    print("=" * 80)
    print(metrics)

    plot_result(result_df, timestamp)


if __name__ == "__main__":
    main()