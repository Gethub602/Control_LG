"""
Run an arbitrary multi-target scenario on the real ESP32 motor.

The repository's run_esp32_fixed_pid.py / run_esp32_local_gain_db.py hard-code a
single 70 -> 100 step. The final study in the README uses a four-segment
scenario instead:

    70 -> 95 -> 90 -> 73 RPM, changing at 5, 10, 15 s

That sequence covers one acceleration and two decelerations, and its top target
sits right at what PWM 140 can actually deliver (about 95 rpm). This script runs
that scenario, or any other, so the local baselines are measured on exactly the
same trajectory the server-assisted results are reported on.

Gain sources:
    --gain-source db      target-interpolated gains from ESP32_REAL_PID_GAIN_DB
    --gain-source fixed   one constant gain triple

No Kafka is involved; this is the local-only baseline.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))
sys.path.append(str(CURRENT_DIR))

from config import (  # noqa: E402
    ESP32_BAUDRATE,
    ESP32_PORT,
    ESP32_REAL_GAIN_DB_MODE,
    ESP32_REAL_PID_GAIN_DB,
    ESP32_TIMEOUT,
    REAL_PWM_MAX,
    REAL_PWM_MIN,
    RESULTS_DIR,
)
from motor_interface import ESP32MotorInterface  # noqa: E402
from pid_controller import PIDController  # noqa: E402
from run_esp32_local_gain_db import (  # noqa: E402
    apply_pwm_rate_limit,
    get_gain_from_db,
)

RESULT_DIR = RESULTS_DIR / "esp32_control_comparison"


def parse_args():
    p = argparse.ArgumentParser(description="Real-motor multi-target scenario run.")
    p.add_argument("--targets", default="70,95,90,73")
    p.add_argument("--change-times", default="5,10,15")
    p.add_argument("--sim-time", type=float, default=20.0)
    p.add_argument("--control-dt", type=float, default=0.10)
    p.add_argument("--gain-source", default="db", choices=["db", "fixed"])
    p.add_argument("--kp", type=float, default=1.2)
    p.add_argument("--ki", type=float, default=0.9)
    p.add_argument("--kd", type=float, default=0.0)
    p.add_argument("--pwm-rate-limit", type=float, default=50.0)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--rest", type=float, default=3.0,
                   help="Seconds of coast between repeats.")
    p.add_argument("--run-label", default="")
    return p.parse_args()


def parse_floats(text):
    return [float(v) for v in str(text).split(",") if str(v).strip()]


def target_at(t, targets, change_times):
    idx = 0
    for i, ct in enumerate(change_times):
        if t >= ct:
            idx = i + 1
    return targets[min(idx, len(targets) - 1)]


def gains_for(target, args):
    if args.gain_source == "fixed":
        return args.kp, args.ki, args.kd
    return get_gain_from_db(target)


def set_gains_preserving_integral(pid, kp, ki, kd):
    """Keep Ki * integral continuous across a gain switch."""
    old_ki = pid.ki
    if abs(ki) > 1e-9 and abs(old_ki) > 1e-9:
        pid.integral = pid.integral * (old_ki / ki)
    pid.set_gains(kp, ki, kd)


def run_once(motor, args, targets, change_times, run_idx):
    n_steps = int(args.sim_time / args.control_dt)
    pid = PIDController(
        dt=args.control_dt, output_min=REAL_PWM_MIN, output_max=REAL_PWM_MAX
    )
    kp, ki, kd = gains_for(targets[0], args)
    pid.set_gains(kp, ki, kd)

    state = motor.get_state()
    current = state.current
    prev_pwm = 0.0
    prev_err = targets[0] - current
    last_gain = (kp, ki, kd)

    rows = []
    t0 = time.time()
    for step in range(n_steps):
        t = step * args.control_dt
        target = target_at(t, targets, change_times)

        new_gain = gains_for(target, args)
        if any(abs(a - b) > 1e-9 for a, b in zip(new_gain, last_gain)):
            set_gains_preserving_integral(pid, *new_gain)
            last_gain = new_gain

        err = target - current
        raw_pwm = pid.compute(target, current)
        pwm = apply_pwm_rate_limit(raw_pwm, prev_pwm, args.pwm_rate_limit)
        pwm = float(np.clip(pwm, REAL_PWM_MIN, REAL_PWM_MAX))

        st = motor.step(pwm)
        current = st.current

        rows.append({
            "run_idx": run_idx,
            "step": step,
            "time": t,
            "target": target,
            "current": current,
            "error": target - current,
            "error_derivative": (err - prev_err) / args.control_dt,
            "raw_pwm": raw_pwm,
            "pwm": pwm,
            "kp": pid.kp,
            "ki": pid.ki,
            "kd": pid.kd,
        })

        if step % 20 == 0:
            print(f"  step={step:04d} t={t:5.2f} target={target:6.1f} "
                  f"rpm={current:7.2f} err={target - current:7.2f} "
                  f"pwm={pwm:6.1f} Kp={pid.kp:.3f} Ki={pid.ki:.3f}", flush=True)

        prev_pwm, prev_err = pwm, err

        # keep the loop close to real time
        target_wall = t0 + (step + 1) * args.control_dt
        sleep = target_wall - time.time()
        if sleep > 0:
            time.sleep(sleep)

    motor.stop()
    return pd.DataFrame(rows)


def metrics_for(df, args, targets, change_times):
    dt = args.control_dt
    err = df["error"].to_numpy()
    t = df["time"].to_numpy()

    after = np.zeros_like(t, dtype=bool)
    for ct in change_times:
        after |= (t >= ct) & (t < ct + 2.0)

    pwm = df["pwm"].to_numpy()
    over = []
    for i, ct in enumerate([0.0] + list(change_times)):
        seg = (t >= ct) & (t < (change_times[i] if i < len(change_times) else t[-1] + 1))
        if seg.sum() == 0:
            continue
        tgt = df["target"].to_numpy()[seg][0]
        peak = df["current"].to_numpy()[seg].max()
        over.append(max(0.0, (peak - tgt) / max(abs(tgt), 1e-6) * 100.0))

    return {
        "IAE": float(np.sum(np.abs(err)) * dt),
        "after_change_IAE": float(np.sum(np.abs(err[after])) * dt),
        "mean_abs_error": float(np.mean(np.abs(err))),
        "final_error": float(abs(err[-1])),
        "max_overshoot_percent": float(max(over)) if over else 0.0,
        "mean_pwm": float(np.mean(pwm)),
        "max_pwm": float(np.max(pwm)),
        "saturation_ratio_percent": float(np.mean(pwm >= REAL_PWM_MAX - 1e-6) * 100.0),
    }


def main():
    args = parse_args()
    targets = parse_floats(args.targets)
    change_times = parse_floats(args.change_times)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = args.run_label or f"scenario_{args.gain_source}"

    print("=" * 78)
    print("ESP32 scenario run")
    print("=" * 78)
    print(f"Port: {ESP32_PORT}")
    print(f"Targets: {targets} at {change_times} s over {args.sim_time} s")
    print(f"Gain source: {args.gain_source}")
    if args.gain_source == "fixed":
        print(f"Fixed gains: Kp={args.kp} Ki={args.ki} Kd={args.kd}")
    print(f"PWM limit: {REAL_PWM_MIN} ~ {REAL_PWM_MAX}  rate_limit={args.pwm_rate_limit}")
    print(f"Repeats: {args.repeats}")
    print("=" * 78)

    motor = ESP32MotorInterface(
        port=ESP32_PORT,
        baudrate=ESP32_BAUDRATE,
        timeout=ESP32_TIMEOUT,
        pwm_min=REAL_PWM_MIN,
        pwm_max=REAL_PWM_MAX,
    )

    frames, mets = [], []
    try:
        print(f"PING: {motor.ping()}")
        for r in range(args.repeats):
            print(f"\n--- run {r + 1}/{args.repeats} ---")
            motor.stop()
            time.sleep(args.rest)
            df = run_once(motor, args, targets, change_times, r)
            frames.append(df)
            m = metrics_for(df, args, targets, change_times)
            m["run_idx"] = r
            mets.append(m)
            print(f"  IAE={m['IAE']:.2f}  after={m['after_change_IAE']:.2f}  "
                  f"final_err={m['final_error']:.2f}  sat={m['saturation_ratio_percent']:.1f}%")
    finally:
        # never leave the motor running
        try:
            motor.stop()
            motor.close()
        except Exception:
            pass

    log = pd.concat(frames, ignore_index=True)
    met = pd.DataFrame(mets)
    log_path = RESULT_DIR / f"esp32_{label}_log_{timestamp}.csv"
    met_path = RESULT_DIR / f"esp32_{label}_metrics_{timestamp}.csv"
    log.to_csv(log_path, index=False)
    met.to_csv(met_path, index=False)

    print()
    print("=" * 78)
    print(met.to_string(index=False))
    print()
    if len(met) > 1:
        print(f"IAE          : {met['IAE'].mean():.2f} +- {met['IAE'].std():.2f}")
        print(f"after_change : {met['after_change_IAE'].mean():.2f} "
              f"+- {met['after_change_IAE'].std():.2f}")
    print(f"Saved log    : {log_path}")
    print(f"Saved metrics: {met_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
