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

    ESP32_REAL_PID_GAIN_DB,
    ESP32_REAL_GAIN_DB_MODE,
)


# ============================================================
# Experiment settings
# ============================================================

MODE_NAME = "esp32_local_gain_db"
RESULT_DIR = RESULTS_DIR / "esp32_control_comparison"

CONTROL_DT = 0.10
SIM_TIME = 10.0
N_STEPS = int(SIM_TIME / CONTROL_DT)

TARGET_BEFORE = 70.0
TARGET_AFTER = 100.0
TARGET_CHANGE_TIME = 3.0

PWM_MIN = REAL_PWM_MIN
PWM_MAX = REAL_PWM_MAX

PWM_RATE_LIMIT = 50.0
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


def interpolate_gain_from_db(
    target: float,
    gain_db: dict,
    mode: str,
    fallback_gain: tuple,
):
    if not gain_db:
        return fallback_gain

    target = float(target)
    db_targets = sorted([float(k) for k in gain_db.keys()])

    if target in db_targets:
        gains = gain_db[target]
        return gains["kp"], gains["ki"], gains["kd"]

    if mode == "nearest":
        nearest_target = min(db_targets, key=lambda x: abs(x - target))
        gains = gain_db[nearest_target]
        return gains["kp"], gains["ki"], gains["kd"]

    if mode == "linear":
        if target <= db_targets[0]:
            gains = gain_db[db_targets[0]]
            return gains["kp"], gains["ki"], gains["kd"]

        if target >= db_targets[-1]:
            gains = gain_db[db_targets[-1]]
            return gains["kp"], gains["ki"], gains["kd"]

        for i in range(len(db_targets) - 1):
            t_low = db_targets[i]
            t_high = db_targets[i + 1]

            if t_low <= target <= t_high:
                ratio = (target - t_low) / (t_high - t_low)

                g_low = gain_db[t_low]
                g_high = gain_db[t_high]

                kp = g_low["kp"] + ratio * (g_high["kp"] - g_low["kp"])
                ki = g_low["ki"] + ratio * (g_high["ki"] - g_low["ki"])
                kd = g_low["kd"] + ratio * (g_high["kd"] - g_low["kd"])

                return kp, ki, kd

    nearest_target = min(db_targets, key=lambda x: abs(x - target))
    gains = gain_db[nearest_target]
    return gains["kp"], gains["ki"], gains["kd"]


def get_gain_from_db(target: float):
    return interpolate_gain_from_db(
        target=target,
        gain_db=ESP32_REAL_PID_GAIN_DB,
        mode=ESP32_REAL_GAIN_DB_MODE,
        fallback_gain=(1.2, 0.7, 0.0),
    )


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

    gain_update_count = int((df["gain_update_reason"] == "local_gain_db_update").sum())

    return {
        "mode": MODE_NAME,
        "backend": "esp32",
        "control_dt": CONTROL_DT,
        "target_before": TARGET_BEFORE,
        "target_after": TARGET_AFTER,
        "target_change_time": TARGET_CHANGE_TIME,

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
        "local_gain_db_update_count": gain_update_count,
        "local_gain_reduction_count": 0,
        "local_gain_recovery_count": 0,
    }


def plot_result(df: pd.DataFrame, timestamp: str):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(df["time"], df["current"], label="measured RPM")
    plt.plot(df["time"], df["target"], linestyle="--", label="target RPM")
    plt.axvline(TARGET_CHANGE_TIME, linestyle="--", linewidth=1.2, label="target change")
    plt.xlabel("Time [s]")
    plt.ylabel("RPM")
    plt.title("ESP32 Local Gain DB PID RPM Response")
    plt.legend()
    plt.grid(True)

    save_path = FIGURE_DIR / f"{MODE_NAME}_rpm_response_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(df["time"], df["raw_pwm"], linestyle="--", label="raw PWM")
    plt.plot(df["time"], df["pwm"], label="applied PWM")
    plt.axvline(TARGET_CHANGE_TIME, linestyle="--", linewidth=1.2, label="target change")
    plt.xlabel("Time [s]")
    plt.ylabel("PWM")
    plt.title("ESP32 Local Gain DB PID PWM Command")
    plt.legend()
    plt.grid(True)

    save_path = FIGURE_DIR / f"{MODE_NAME}_pwm_command_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(df["time"], df["kp"], label="Kp")
    plt.plot(df["time"], df["ki"], label="Ki")
    plt.axvline(TARGET_CHANGE_TIME, linestyle="--", linewidth=1.2, label="target change")
    plt.xlabel("Time [s]")
    plt.ylabel("Gain")
    plt.title("ESP32 Local Gain DB PID Gain History")
    plt.legend()
    plt.grid(True)

    save_path = FIGURE_DIR / f"{MODE_NAME}_gain_history_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ESP32 Local Gain DB PID Baseline")
    print("=" * 80)
    print(f"Port: {ESP32_PORT}")
    print(f"Baudrate: {ESP32_BAUDRATE}")
    print(f"Control DT: {CONTROL_DT}")
    print(f"Target: {TARGET_BEFORE} -> {TARGET_AFTER} at {TARGET_CHANGE_TIME}s")
    print(f"PWM limit: {PWM_MIN} ~ {PWM_MAX}")
    print(f"PWM rate limit: {PWM_RATE_LIMIT}")
    print(f"Gain DB mode: {ESP32_REAL_GAIN_DB_MODE}")
    print("=" * 80)

    motor = ESP32MotorInterface(
        port=ESP32_PORT,
        baudrate=ESP32_BAUDRATE,
        timeout=ESP32_TIMEOUT,
        pwm_min=PWM_MIN,
        pwm_max=PWM_MAX,
    )

    target0 = TARGET_BEFORE
    kp, ki, kd = get_gain_from_db(target0)

    pid = PIDController(
        kp=kp,
        ki=ki,
        kd=kd,
        dt=CONTROL_DT,
        output_min=PWM_MIN,
        output_max=PWM_MAX,
    )

    rows = []

    prev_error = 0.0
    prev_pwm = 0.0
    last_gain_target = target0

    try:
        print("PING:", motor.ping())

        motor.stop()
        time.sleep(0.5)

        state = motor.get_state()
        current = state.current
        prev_error = target0 - current

        print(f"Initial RPM: {current:.3f}")
        print(f"Initial gain: Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f}")

        for step in range(N_STEPS):
            loop_start = time.time()

            t = step * CONTROL_DT
            target = get_target_at_time(t)

            gain_update_reason = "none"

            if abs(target - last_gain_target) > 1e-9:
                kp, ki, kd = get_gain_from_db(target)
                pid.set_gains(kp, ki, kd)
                last_gain_target = target
                gain_update_reason = "local_gain_db_update"

                print(
                    f"[LOCAL GAIN UPDATE] t={t:.2f}, "
                    f"target={target:.1f}, "
                    f"Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f}"
                )

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

                    "kp": kp,
                    "ki": ki,
                    "kd": kd,

                    "gain_update_reason": gain_update_reason,
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
                    f"pwm={pwm:.2f}, "
                    f"Kp={kp:.3f}, Ki={ki:.3f}, "
                    f"gain_reason={gain_update_reason}"
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
    print("ESP32 Local Gain DB PID Baseline Finished")
    print(f"Saved log: {log_path}")
    print(f"Saved metrics: {metric_path}")
    print("=" * 80)
    print(metrics)

    plot_result(result_df, timestamp)


if __name__ == "__main__":
    main()