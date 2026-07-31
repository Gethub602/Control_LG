"""
Read-only ESP32 / encoder wiring check.

Sends only PING and GET_STATE. No SET_PWM is ever issued, so the motor is not
driven -- turn the shaft by hand to verify that the encoder counts.

Usage:
    python src/check_encoder.py [--port /dev/ttyUSB0] [--seconds 20]
"""

import argparse
import sys
import time
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))
sys.path.append(str(CURRENT_DIR))

import serial  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Read-only ESP32 encoder check.")
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--baudrate", type=int, default=115200)
    p.add_argument("--timeout", type=float, default=1.5)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--interval", type=float, default=0.25)
    return p.parse_args()


def ask(ser, command, timeout_note=""):
    ser.reset_input_buffer()
    ser.write((command + "\n").encode())
    ser.flush()
    line = ser.readline().decode(errors="ignore").strip()
    return line if line else f"<no response{timeout_note}>"


def parse_state(line):
    if not line.startswith("STATE"):
        return None
    values = {}
    for item in line.split()[1:]:
        if "=" in item:
            k, v = item.split("=", 1)
            values[k.strip()] = v.strip()
    try:
        return {
            "rpm": float(values.get("rpm", "nan")),
            "pwm": float(values.get("pwm", "nan")),
            "encoder": int(float(values.get("encoder", "nan"))),
        }
    except ValueError:
        return None


def main():
    args = parse_args()

    print(f"Opening {args.port} @ {args.baudrate} (read-only check, no PWM sent)")
    try:
        ser = serial.Serial(args.port, args.baudrate, timeout=args.timeout)
    except FileNotFoundError:
        print(f"OPEN FAILED: {args.port} does not exist.")
        print("The device is not attached to WSL. On Windows (admin PowerShell):")
        print("  usbipd list                          # find the BUSID")
        print("  usbipd attach --wsl --busid <BUSID>  # needed after every replug")
        return 1
    except PermissionError as exc:
        print(f"OPEN FAILED: {exc}")
        print("Permission problem. Add yourself to the dialout group:")
        print("  sudo usermod -aG dialout $USER   (then 'wsl --shutdown' on Windows)")
        return 1
    except Exception as exc:
        print(f"OPEN FAILED: {exc}")
        return 1

    time.sleep(2.2)  # ESP32 resets when DTR toggles on open
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    pong = ask(ser, "PING")
    print(f"PING -> {pong}")
    if pong.strip().upper() != "PONG":
        print("Firmware did not answer PING. Stopping here.")
        ser.close()
        return 1

    print()
    print("=" * 62)
    print(f"  TURN THE MOTOR SHAFT BY HAND for the next {args.seconds:.0f} seconds")
    print("=" * 62)
    print()

    encoders = []
    rpms = []
    t_end = time.time() + args.seconds
    first = None

    while time.time() < t_end:
        state = parse_state(ask(ser, "GET_STATE"))
        if state is None:
            print("  bad STATE line")
        else:
            if first is None:
                first = state["encoder"]
            encoders.append(state["encoder"])
            rpms.append(state["rpm"])
            delta = state["encoder"] - first
            print(f"  encoder={state['encoder']:>10d}  delta={delta:>+8d}  "
                  f"rpm={state['rpm']:>8.2f}  pwm={state['pwm']:>5.0f}")
        time.sleep(args.interval)

    ser.close()

    print()
    print("-" * 62)
    if not encoders:
        print("RESULT: no valid STATE lines received.")
        return 1

    span = max(encoders) - min(encoders)
    moved_both_ways = len(set(encoders)) > 2
    print(f"samples={len(encoders)}  min={min(encoders)}  max={max(encoders)}  span={span}")
    print(f"rpm range: {min(rpms):.2f} .. {max(rpms):.2f}")
    print()

    if span == 0:
        print("RESULT: encoder NEVER changed.")
        print("  Check, in order:")
        print("   1. firmware pin numbers match the A/B wiring (D18/D19)")
        print("   2. hall sensor supply voltage (some encoders need 5V, not 3V3)")
        print("   3. pull-ups on A/B (firmware INPUT_PULLUP or external 10k)")
        print("   4. hall GND actually connected")
    elif span < 20:
        print(f"RESULT: encoder changed only slightly (span={span}).")
        print("  Counting works but may be missing edges -- check pull-ups,")
        print("  wiring noise, or whether only one channel is connected.")
    else:
        print(f"RESULT: encoder is counting correctly (span={span}). Wiring OK.")
        if not moved_both_ways:
            print("  Note: very few distinct values; turn the shaft more next time.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
