import sys
from pathlib import Path
from datetime import datetime
import time
import itertools
import argparse

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

    RESULTS_DIR,
    FIGURE_DIR,

    ESP32_SWEEP_TARGET_LIST,
    ESP32_SWEEP_KP_LIST,
    ESP32_SWEEP_KI_LIST,
    ESP32_SWEEP_KD_LIST,

    ESP32_SWEEP_TEST_TIME,
    ESP32_SWEEP_REST_TIME,
    ESP32_SWEEP_PWM_MIN,
    ESP32_SWEEP_PWM_MAX,
    ESP32_SWEEP_PWM_RATE_LIMIT,
)


# ============================================================
# Sweep settings
# ============================================================

RESULT_DIR = RESULTS_DIR / "esp32_gain_sweep"

CONTROL_DT = 0.10
N_STEPS = int(ESP32_SWEEP_TEST_TIME / CONTROL_DT)

SETTLING_TOLERANCE_RATIO = 0.05

# 너무 위험한 경우 case 중단 기준
MAX_SAFE_RPM = 150.0
MAX_ABS_ERROR_FOR_ABORT = 200.0

# score weight
SCORE_IAE_WEIGHT = 1.0
SCORE_FINAL_ERROR_WEIGHT = 3.0
SCORE_OVERSHOOT_WEIGHT = 2.0
SCORE_PWM_WEIGHT = 0.02
SCORE_SATURATION_WEIGHT = 10.0


# ============================================================
# Utility
# ============================================================

def parse_float_list(text: str):
    if text is None or str(text).strip() == "":
        return None

    return [
        float(item.strip())
        for item in str(text).split(",")
        if item.strip() != ""
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ESP32 real motor PID gain sweep."
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="",
        help="Comma-separated target RPM list, e.g. 30,50,70,100.",
    )
    parser.add_argument(
        "--kp-list",
        type=str,
        default="",
        help="Comma-separated Kp list.",
    )
    parser.add_argument(
        "--ki-list",
        type=str,
        default="",
        help="Comma-separated Ki list.",
    )
    parser.add_argument(
        "--kd-list",
        type=str,
        default="",
        help="Comma-separated Kd list.",
    )
    parser.add_argument(
        "--test-time",
        type=float,
        default=ESP32_SWEEP_TEST_TIME,
        help="Test duration per case in seconds.",
    )
    parser.add_argument(
        "--rest-time",
        type=float,
        default=ESP32_SWEEP_REST_TIME,
        help="Rest duration between cases in seconds.",
    )
    parser.add_argument(
        "--pwm-max",
        type=float,
        default=ESP32_SWEEP_PWM_MAX,
        help="Maximum PWM during sweep.",
    )
    parser.add_argument(
        "--pwm-rate-limit",
        type=float,
        default=ESP32_SWEEP_PWM_RATE_LIMIT,
        help="Maximum PWM change per control step.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Limit number of cases for a pilot run. 0 means all cases.",
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default="",
        help="Optional label appended to output filenames.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned cases without opening serial or moving the motor.",
    )

    return parser.parse_args()


def apply_runtime_args(args):
    global ESP32_SWEEP_TEST_TIME
    global ESP32_SWEEP_REST_TIME
    global ESP32_SWEEP_PWM_MAX
    global ESP32_SWEEP_PWM_RATE_LIMIT
    global N_STEPS

    ESP32_SWEEP_TEST_TIME = float(args.test_time)
    ESP32_SWEEP_REST_TIME = float(args.rest_time)
    ESP32_SWEEP_PWM_MAX = float(args.pwm_max)
    ESP32_SWEEP_PWM_RATE_LIMIT = float(args.pwm_rate_limit)
    N_STEPS = int(ESP32_SWEEP_TEST_TIME / CONTROL_DT)


def build_case_lists(args):
    target_list = parse_float_list(args.targets)
    kp_list = parse_float_list(args.kp_list)
    ki_list = parse_float_list(args.ki_list)
    kd_list = parse_float_list(args.kd_list)

    if target_list is None:
        target_list = [float(x) for x in ESP32_SWEEP_TARGET_LIST]

    if kp_list is None:
        kp_list = [float(x) for x in ESP32_SWEEP_KP_LIST]

    if ki_list is None:
        ki_list = [float(x) for x in ESP32_SWEEP_KI_LIST]

    if kd_list is None:
        kd_list = [float(x) for x in ESP32_SWEEP_KD_LIST]

    cases = list(itertools.product(target_list, kp_list, ki_list, kd_list))

    if args.max_cases and args.max_cases > 0:
        cases = cases[: int(args.max_cases)]

    return target_list, kp_list, ki_list, kd_list, cases


def safe_label(text: str):
    if not text:
        return ""

    return "".join(
        ch if ch.isalnum() or ch in ["-", "_"] else "_"
        for ch in str(text)
    )

def apply_pwm_rate_limit(pwm_cmd, prev_pwm, rate_limit):
    delta = pwm_cmd - prev_pwm
    delta = np.clip(delta, -rate_limit, rate_limit)
    return prev_pwm + delta


def compute_dt(time_arr):
    if len(time_arr) <= 1:
        return np.zeros_like(time_arr)

    dt_arr = np.diff(time_arr, prepend=time_arr[0])
    dt_arr[0] = 0.0
    return dt_arr


def calculate_metrics(df: pd.DataFrame, target_rpm: float, kp: float, ki: float, kd: float):
    time_arr = df["time"].to_numpy(dtype=float)
    target = df["target"].to_numpy(dtype=float)
    rpm = df["rpm"].to_numpy(dtype=float)
    error = df["error"].to_numpy(dtype=float)
    pwm = df["pwm"].to_numpy(dtype=float)

    dt_arr = compute_dt(time_arr)
    abs_error = np.abs(error)
    abs_pwm = np.abs(pwm)

    iae = float(np.sum(abs_error * dt_arr))
    ise = float(np.sum((error ** 2) * dt_arr))
    mean_abs_error = float(np.mean(abs_error))
    final_error = float(abs_error[-1])

    max_rpm = float(np.max(rpm))
    min_rpm = float(np.min(rpm))

    overshoot = max(0.0, max_rpm - target_rpm)
    overshoot_percent = overshoot / max(abs(target_rpm), 1e-6) * 100.0

    high_saturation = pwm >= ESP32_SWEEP_PWM_MAX - 1e-9
    saturation_count = int(np.sum(high_saturation))
    saturation_ratio_percent = float(np.mean(high_saturation) * 100.0)
    saturation_duration = float(np.sum(dt_arr[high_saturation]))

    near_high_saturation = pwm >= ESP32_SWEEP_PWM_MAX - 0.1 * (
        ESP32_SWEEP_PWM_MAX - ESP32_SWEEP_PWM_MIN
    )
    near_high_saturation_ratio_percent = float(np.mean(near_high_saturation) * 100.0)

    tolerance = SETTLING_TOLERANCE_RATIO * abs(target_rpm)
    settling_time = np.nan

    for i in range(len(time_arr)):
        if np.all(np.abs(target[i:] - rpm[i:]) <= tolerance):
            settling_time = float(time_arr[i])
            break

    if np.isnan(settling_time):
        settling_penalty = ESP32_SWEEP_TEST_TIME
    else:
        settling_penalty = settling_time

    score = (
        SCORE_IAE_WEIGHT * iae
        + SCORE_FINAL_ERROR_WEIGHT * final_error
        + SCORE_OVERSHOOT_WEIGHT * overshoot_percent
        + SCORE_PWM_WEIGHT * float(np.mean(abs_pwm))
        + SCORE_SATURATION_WEIGHT * saturation_ratio_percent
        + 0.5 * settling_penalty
    )

    return {
        "target": float(target_rpm),
        "kp": float(kp),
        "ki": float(ki),
        "kd": float(kd),

        "IAE": iae,
        "ISE": ise,
        "mean_abs_error": mean_abs_error,
        "final_error": final_error,

        "max_rpm": max_rpm,
        "min_rpm": min_rpm,
        "overshoot_percent": overshoot_percent,
        "settling_time": settling_time,

        "mean_pwm": float(np.mean(abs_pwm)),
        "max_pwm": float(np.max(pwm)),
        "min_pwm": float(np.min(pwm)),

        "saturation_count": saturation_count,
        "saturation_ratio_percent": saturation_ratio_percent,
        "saturation_duration": saturation_duration,
        "near_high_saturation_ratio_percent": near_high_saturation_ratio_percent,

        "score": float(score),
    }


def run_single_case(
    motor: ESP32MotorInterface,
    target_rpm: float,
    kp: float,
    ki: float,
    kd: float,
    case_id: int,
    total_cases: int,
):
    print("=" * 80)
    print(
        f"[{case_id}/{total_cases}] "
        f"target={target_rpm:.1f}, Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f}"
    )
    print("=" * 80)

    # case 시작 전 정지
    motor.stop()
    time.sleep(ESP32_SWEEP_REST_TIME)

    pid = PIDController(
        kp=kp,
        ki=ki,
        kd=kd,
        dt=CONTROL_DT,
        output_min=ESP32_SWEEP_PWM_MIN,
        output_max=ESP32_SWEEP_PWM_MAX,
    )

    rows = []

    prev_error = 0.0
    prev_pwm = 0.0

    state = motor.get_state()
    current_rpm = state.current

    # WSL can correct CLOCK_REALTIME by about one second while a run is active.
    # Elapsed control time must therefore use a monotonic clock.
    start_time = time.monotonic()
    aborted = False
    abort_reason = "none"

    try:
        for step in range(N_STEPS):
            loop_start = time.monotonic()
            t = time.monotonic() - start_time

            target = float(target_rpm)
            error = target - current_rpm
            error_derivative = (error - prev_error) / CONTROL_DT

            pwm_cmd = pid.compute(target, current_rpm)

            pwm_cmd = apply_pwm_rate_limit(
                pwm_cmd=pwm_cmd,
                prev_pwm=prev_pwm,
                rate_limit=ESP32_SWEEP_PWM_RATE_LIMIT,
            )
            pwm_cmd = float(np.clip(pwm_cmd, ESP32_SWEEP_PWM_MIN, ESP32_SWEEP_PWM_MAX))

            state = motor.step(pwm_cmd)
            current_rpm = state.current

            encoder = np.nan
            if state.raw is not None:
                encoder = state.raw.get("encoder", np.nan)

            rows.append(
                {
                    "case_id": case_id,
                    "step": step,
                    "time": t,
                    "wall_time": time.time(),

                    "target": target,
                    "rpm": current_rpm,
                    "error": error,
                    "error_derivative": error_derivative,

                    "pwm": pwm_cmd,
                    "encoder": encoder,

                    "kp": kp,
                    "ki": ki,
                    "kd": kd,

                    "aborted": aborted,
                    "abort_reason": abort_reason,
                }
            )

            if step % 10 == 0:
                print(
                    f"step={step:04d}, "
                    f"t={t:.2f}, "
                    f"target={target:.1f}, "
                    f"rpm={current_rpm:.2f}, "
                    f"error={error:.2f}, "
                    f"pwm={pwm_cmd:.2f}"
                )

            if abs(current_rpm) > MAX_SAFE_RPM:
                aborted = True
                abort_reason = "rpm_exceeded_safe_limit"
                print(f"[ABORT] {abort_reason}: rpm={current_rpm:.2f}")
                break

            if abs(error) > MAX_ABS_ERROR_FOR_ABORT:
                aborted = True
                abort_reason = "error_exceeded_safe_limit"
                print(f"[ABORT] {abort_reason}: error={error:.2f}")
                break

            prev_error = error
            prev_pwm = pwm_cmd

            elapsed = time.monotonic() - loop_start
            sleep_time = max(0.0, CONTROL_DT - elapsed)
            time.sleep(sleep_time)

    finally:
        motor.stop()
        time.sleep(ESP32_SWEEP_REST_TIME)

    df = pd.DataFrame(rows)

    if len(df) == 0:
        metrics = {
            "target": float(target_rpm),
            "kp": float(kp),
            "ki": float(ki),
            "kd": float(kd),
            "IAE": np.inf,
            "ISE": np.inf,
            "mean_abs_error": np.inf,
            "final_error": np.inf,
            "max_rpm": np.nan,
            "min_rpm": np.nan,
            "overshoot_percent": np.inf,
            "settling_time": np.nan,
            "mean_pwm": np.nan,
            "max_pwm": np.nan,
            "min_pwm": np.nan,
            "saturation_count": np.nan,
            "saturation_ratio_percent": np.inf,
            "saturation_duration": np.nan,
            "near_high_saturation_ratio_percent": np.inf,
            "score": np.inf,
        }
    else:
        metrics = calculate_metrics(
            df=df,
            target_rpm=target_rpm,
            kp=kp,
            ki=ki,
            kd=kd,
        )

    metrics["case_id"] = case_id
    metrics["aborted"] = aborted
    metrics["abort_reason"] = abort_reason

    return df, metrics


def select_best_gains(metrics_df: pd.DataFrame):
    """
    target별 best gain 선택.
    aborted case는 제외.
    saturation이 심한 case도 뒤로 밀림.
    """

    valid_df = metrics_df.copy()

    valid_df = valid_df[valid_df["aborted"] == False]  # noqa: E712

    if valid_df.empty:
        return pd.DataFrame()

    best_rows = []

    for target, group in valid_df.groupby("target"):
        group_sorted = group.sort_values(
            by=[
                "score",
                "IAE",
                "final_error",
                "overshoot_percent",
                "saturation_ratio_percent",
            ],
            ascending=True,
        )

        best_rows.append(group_sorted.iloc[0].to_dict())

    best_df = pd.DataFrame(best_rows)
    best_df = best_df.sort_values("target").reset_index(drop=True)

    return best_df


def save_best_gain_db_text(best_df: pd.DataFrame, timestamp: str):
    """
    config.py에 복사할 수 있는 gain DB text 생성.
    """

    save_path = RESULT_DIR / f"esp32_real_gain_db_{timestamp}.txt"

    lines = []
    lines.append("# Copy this block into config.py")
    lines.append("ESP32_REAL_PID_GAIN_DB = {")

    for _, row in best_df.iterrows():
        target = float(row["target"])
        kp = float(row["kp"])
        ki = float(row["ki"])
        kd = float(row["kd"])

        lines.append(
            f"    {target:.1f}: "
            f'{{"kp": {kp:.4f}, "ki": {ki:.4f}, "kd": {kd:.4f}}},'
        )

    lines.append("}")
    lines.append("")
    lines.append('ESP32_REAL_GAIN_DB_MODE = "linear"')

    save_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved gain DB text: {save_path}")

    return save_path


def plot_best_summary(metrics_df: pd.DataFrame, best_df: pd.DataFrame, timestamp: str):
    if metrics_df.empty:
        return

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # target별 score scatter
    plt.figure(figsize=(10, 5))

    for target, group in metrics_df.groupby("target"):
        plt.scatter(
            group["case_id"],
            group["score"],
            label=f"target={target:.0f}",
        )

    plt.xlabel("Case ID")
    plt.ylabel("Score")
    plt.title("ESP32 PID Gain Sweep Score")
    plt.legend()
    plt.grid(True)

    save_path = FIGURE_DIR / f"esp32_gain_sweep_score_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()

    if not best_df.empty:
        plt.figure(figsize=(8, 5))
        plt.plot(best_df["target"], best_df["kp"], marker="o", label="best Kp")
        plt.plot(best_df["target"], best_df["ki"], marker="o", label="best Ki")
        plt.xlabel("Target RPM")
        plt.ylabel("Gain")
        plt.title("Best ESP32 PID Gains by Target")
        plt.legend()
        plt.grid(True)

        save_path = FIGURE_DIR / f"esp32_best_gains_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.show()


# ============================================================
# Main
# ============================================================

def main(args=None):
    if args is None:
        args = parse_args()

    apply_runtime_args(args)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    target_list, kp_list, ki_list, kd_list, cases = build_case_lists(args)
    total_cases = len(cases)
    label = safe_label(args.run_label)
    file_suffix = f"_{label}_{timestamp}" if label else f"_{timestamp}"

    print("=" * 80)
    print("ESP32 real motor PID gain sweep")
    print("=" * 80)
    print(f"Port: {ESP32_PORT}")
    print(f"Baudrate: {ESP32_BAUDRATE}")
    print(f"Control DT: {CONTROL_DT}")
    print(f"Targets: {target_list}")
    print(f"Kp list: {kp_list}")
    print(f"Ki list: {ki_list}")
    print(f"Kd list: {kd_list}")
    print(f"Total cases: {total_cases}")
    print(f"Test time per case: {ESP32_SWEEP_TEST_TIME} s")
    print(f"Rest time per case: {ESP32_SWEEP_REST_TIME} s")
    print(f"PWM limit: {ESP32_SWEEP_PWM_MIN} ~ {ESP32_SWEEP_PWM_MAX}")
    print(f"PWM rate limit: {ESP32_SWEEP_PWM_RATE_LIMIT}")
    print(f"Max safe RPM: {MAX_SAFE_RPM}")
    print(f"Max abs error for abort: {MAX_ABS_ERROR_FOR_ABORT}")
    print("=" * 80)

    if args.dry_run:
        print("Dry run only. Planned cases:")
        for case_id, (target, kp, ki, kd) in enumerate(cases, start=1):
            print(
                f"{case_id:03d}: "
                f"target={target:.1f}, kp={kp:.3f}, ki={ki:.3f}, kd={kd:.3f}"
            )
        return

    motor = ESP32MotorInterface(
        port=ESP32_PORT,
        baudrate=ESP32_BAUDRATE,
        timeout=ESP32_TIMEOUT,
        pwm_min=ESP32_SWEEP_PWM_MIN,
        pwm_max=ESP32_SWEEP_PWM_MAX,
    )

    all_logs = []
    metric_rows = []

    try:
        print("PING:", motor.ping())

        for case_id, (target, kp, ki, kd) in enumerate(cases, start=1):
            case_df, metrics = run_single_case(
                motor=motor,
                target_rpm=target,
                kp=kp,
                ki=ki,
                kd=kd,
                case_id=case_id,
                total_cases=total_cases,
            )

            all_logs.append(case_df)
            metric_rows.append(metrics)

            # case별 중간 저장
            case_log_path = RESULT_DIR / (
                f"esp32_gain_sweep_case_{case_id:03d}_"
                f"target_{target:.0f}_kp_{kp:.2f}_ki_{ki:.2f}{file_suffix}.csv"
            )
            case_df.to_csv(case_log_path, index=False, encoding="utf-8-sig")

            metrics_df_partial = pd.DataFrame(metric_rows)
            partial_metrics_path = RESULT_DIR / f"esp32_gain_sweep_metrics_partial{file_suffix}.csv"
            metrics_df_partial.to_csv(
                partial_metrics_path,
                index=False,
                encoding="utf-8-sig",
            )

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Stopping sweep safely...")

    finally:
        motor.stop()
        time.sleep(1.0)
        motor.close()

    if all_logs:
        log_df = pd.concat(all_logs, ignore_index=True)
    else:
        log_df = pd.DataFrame()

    metrics_df = pd.DataFrame(metric_rows)

    log_path = RESULT_DIR / f"esp32_gain_sweep_log{file_suffix}.csv"
    metrics_path = RESULT_DIR / f"esp32_gain_sweep_metrics{file_suffix}.csv"

    log_df.to_csv(log_path, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("ESP32 gain sweep finished")
    print(f"Saved log: {log_path}")
    print(f"Saved metrics: {metrics_path}")

    best_df = select_best_gains(metrics_df)

    best_path = RESULT_DIR / f"esp32_gain_sweep_best{file_suffix}.csv"
    best_df.to_csv(best_path, index=False, encoding="utf-8-sig")

    print(f"Saved best gains: {best_path}")

    if not best_df.empty:
        gain_db_path = save_best_gain_db_text(best_df, timestamp)
        print("\nBest gains:")
        print(best_df[["target", "kp", "ki", "kd", "score", "IAE", "final_error", "settling_time"]])
        print(f"\nGain DB text: {gain_db_path}")

    plot_best_summary(metrics_df, best_df, timestamp)


if __name__ == "__main__":
    main()
