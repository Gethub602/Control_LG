"""
First powered spin test.

Drives the motor briefly at a few low PWM levels and reports the resulting RPM,
so three things get verified at once:

  1. the TB6612 wiring actually drives the motor (PWMA/AIN1/AIN2 pins)
  2. a positive PWM produces a positive RPM (sign convention matches the PID)
  3. the RPM magnitude is consistent with the 1:56 gearbox and 178 RPM nameplate

Safety:
  - the firmware clamps PWM to its own limit (default 140) regardless of this script
  - every level runs for a couple of seconds, then stops
  - STOP is sent from a finally block, so Ctrl-C or an exception still stops the motor
  - the firmware watchdog stops the motor if this script dies without sending STOP

Keep the motor body held or clamped, with nothing attached to the shaft.
"""

import argparse
import sys
import time

import serial


def parse_args():
    p = argparse.ArgumentParser(description="Low-power motor spin test.")
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--baudrate", type=int, default=115200)
    p.add_argument("--levels", default="40,60,80,100",
                   help="Forward PWM levels to try, comma separated.")
    p.add_argument("--seconds", type=float, default=2.5,
                   help="Drive time per level.")
    p.add_argument("--rest", type=float, default=1.5,
                   help="Coast time between levels.")
    p.add_argument("--reverse", action="store_true",
                   help="Also run one reverse step to check direction handling.")
    p.add_argument("--max-level", type=int, default=110,
                   help="Refuse to send anything above this, as a host-side guard.")
    return p.parse_args()


def ask(ser, cmd):
    ser.reset_input_buffer()
    ser.write((cmd + "\n").encode())
    ser.flush()
    return ser.readline().decode(errors="ignore").strip()


def ask_multi(ser, cmd, settle=0.4):
    """Some commands (VERSION) answer with several lines; drain all of them so
    the next single-line read does not pick up a leftover."""
    ser.reset_input_buffer()
    ser.write((cmd + "\n").encode())
    ser.flush()
    time.sleep(settle)
    lines = []
    while ser.in_waiting:
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            lines.append(line)
    return lines


def read_state(ser):
    line = ask(ser, "GET_STATE")
    if not line.startswith("STATE"):
        return None
    out = {}
    for item in line.split()[1:]:
        if "=" in item:
            k, v = item.split("=", 1)
            try:
                out[k] = float(v)
            except ValueError:
                pass
    return out


def drive(ser, level, seconds, label):
    """Hold one PWM level and return the RPM averaged over the final second."""
    resp = ask(ser, f"SET_PWM {level}")
    print(f"  {label:>8}  SET_PWM {level:>4} -> {resp}")

    samples = []
    t_end = time.time() + seconds
    while time.time() < t_end:
        st = read_state(ser)
        if st:
            remaining = t_end - time.time()
            samples.append((remaining, st.get("rpm", float("nan")), st.get("encoder", 0)))
            print(f"           rpm={st.get('rpm', 0):8.2f}  "
                  f"encoder={int(st.get('encoder', 0)):>9}", end="\r", flush=True)
        time.sleep(0.1)
    print(" " * 60, end="\r")

    steady = [r for rem, r, _ in samples if rem <= 1.0]
    if not steady:
        steady = [r for _, r, _ in samples]
    mean_rpm = sum(steady) / len(steady) if steady else float("nan")
    last_count = samples[-1][2] if samples else 0
    return mean_rpm, last_count


def main():
    args = parse_args()

    levels = [int(v) for v in args.levels.split(",") if v.strip()]
    over = [v for v in levels if abs(v) > args.max_level]
    if over:
        print(f"Refusing to run: {over} exceed --max-level {args.max_level}")
        return 1

    try:
        ser = serial.Serial(args.port, args.baudrate, timeout=1.0)
    except FileNotFoundError:
        print(f"{args.port} does not exist -- attach the device to WSL first:")
        print("  usbipd attach --wsl --busid 3-3   (admin PowerShell)")
        return 1

    time.sleep(2.2)
    ser.reset_input_buffer()

    if ask(ser, "PING").upper() != "PONG":
        print("No PONG. Firmware not responding.")
        ser.close()
        return 1

    for line in ask_multi(ser, "VERSION"):
        print("  " + line)
    print()
    print("=" * 66)
    print("  POWERED TEST -- the motor will spin.")
    print("  Hold the motor body. Nothing should be attached to the shaft.")
    print("=" * 66)
    print()

    results = []
    try:
        ask(ser, "RESET_ENCODER")

        for level in levels:
            rpm, count = drive(ser, level, args.seconds, "forward")
            results.append((level, rpm, count))
            print(f"  {'':>8}  pwm={level:>4}  ->  rpm={rpm:8.2f}   count={count}")
            ask(ser, "STOP")
            time.sleep(args.rest)

        if args.reverse:
            level = -abs(levels[len(levels) // 2])
            rpm, count = drive(ser, level, args.seconds, "reverse")
            results.append((level, rpm, count))
            print(f"  {'':>8}  pwm={level:>4}  ->  rpm={rpm:8.2f}   count={count}")
            ask(ser, "STOP")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        # The motor must not be left running under any exit path.
        try:
            print()
            print("STOP ->", ask(ser, "STOP"))
        except Exception:
            pass

    print()
    print("-" * 66)
    print(f"  {'pwm':>6} {'rpm':>10} {'rpm/pwm':>10}")
    for level, rpm, _ in results:
        ratio = rpm / level if level else float("nan")
        print(f"  {level:>6} {rpm:>10.2f} {ratio:>10.3f}")
    print()

    forward = [(l, r) for l, r, _ in results if l > 0]
    if not forward:
        print("No forward results.")
        ser.close()
        return 1

    if all(abs(r) < 1.0 for _, r in forward):
        print("RESULT: no rotation detected.")
        print("  Motor power on? Check VM, STBY on 3V3, and AO1/AO2 to red/white.")
    elif any(r < -1.0 for _, r in forward):
        print("RESULT: motor spins, but a positive PWM gives a NEGATIVE rpm.")
        print("  The sign convention is inverted. Either swap AO1/AO2 (or the red")
        print("  and white motor wires), or swap encoder A and B. The PID assumes")
        print("  positive PWM increases rpm; leaving this inverted makes the loop")
        print("  drive the wrong way and run away.")
    else:
        top_level, top_rpm = max(forward, key=lambda x: x[0])
        # 12V nameplate is 178 rpm; PWM is an 8-bit duty over that supply.
        expected = 178.0 * (top_level / 255.0)
        print(f"RESULT: motor spins, sign is correct.")
        print(f"  At pwm={top_level} measured {top_rpm:.1f} rpm; "
              f"a linear 12V/178rpm model predicts about {expected:.1f} rpm.")
        if top_rpm > expected * 2.0:
            print("  Measured rpm is far above expectation -- gear ratio may be too")
            print("  low. Re-check with calibrate_gear.py.")
        elif top_rpm < expected * 0.4:
            print("  Measured rpm is well below expectation. That can be normal at")
            print("  low duty (friction dominates), but if it persists at higher")
            print("  duty the gear ratio may be too high.")
        else:
            print("  Magnitude is in the expected range: the 1:56 ratio looks right.")

    ser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
