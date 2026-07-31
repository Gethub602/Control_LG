"""
Live encoder wiring monitor.

Polls RAW (instantaneous A/B pin levels) and GET_STATE (accumulated count) while
you turn the shaft by hand, and prints a verdict that distinguishes the failure
modes from each other:

  A/B never change              -> encoder unpowered, wire broken, or wrong pins
  A/B stuck at 1,1              -> only the internal pull-ups are being read
  A/B stuck at 0,0              -> signal lines shorted to ground
  A/B change but count does not -> interrupts/decoding problem
  both change                   -> wiring is good

Read-only: no SET_PWM is ever sent, so the motor is never driven.
"""

import argparse
import sys
import time

import serial


def parse_args():
    p = argparse.ArgumentParser(description="Live encoder monitor (read-only).")
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--baudrate", type=int, default=115200)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--interval", type=float, default=0.1)
    return p.parse_args()


def ask(ser, cmd):
    ser.reset_input_buffer()
    ser.write((cmd + "\n").encode())
    ser.flush()
    return ser.readline().decode(errors="ignore").strip()


def parse_raw(line):
    # "RAW a=1 b=0"
    if not line.startswith("RAW"):
        return None
    vals = {}
    for item in line.split()[1:]:
        if "=" in item:
            k, v = item.split("=", 1)
            vals[k] = v
    try:
        return int(vals["a"]), int(vals["b"])
    except (KeyError, ValueError):
        return None


def parse_encoder(line):
    if not line.startswith("STATE"):
        return None
    for item in line.split()[1:]:
        if item.startswith("encoder="):
            try:
                return int(float(item.split("=", 1)[1]))
            except ValueError:
                return None
    return None


def main():
    args = parse_args()
    print(f"Opening {args.port} (read-only; no PWM is sent)")
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

    ask(ser, "RESET_ENCODER")

    print()
    print("=" * 66)
    print(f"  TURN THE SHAFT BY HAND for {args.seconds:.0f} seconds")
    print("=" * 66)
    print(f"  {'A':>3} {'B':>3}   {'count':>9}   state")
    print("-" * 66)

    seen_states = set()
    counts = []
    t_end = time.time() + args.seconds

    while time.time() < t_end:
        raw = parse_raw(ask(ser, "RAW"))
        cnt = parse_encoder(ask(ser, "GET_STATE"))
        if raw is not None:
            seen_states.add(raw)
            a, b = raw
            bar = "#" * (len(seen_states) * 2)
            print(f"  {a:>3} {b:>3}   {cnt if cnt is not None else '?':>9}   {bar}")
        if cnt is not None:
            counts.append(cnt)
        time.sleep(args.interval)

    ser.close()

    print("-" * 66)
    print(f"distinct (A,B) states seen: {sorted(seen_states)}")
    if counts:
        print(f"count range: {min(counts)} .. {max(counts)}  (span {max(counts) - min(counts)})")
    print()

    span = (max(counts) - min(counts)) if counts else 0

    if len(seen_states) <= 1:
        only = next(iter(seen_states)) if seen_states else None
        print("RESULT: encoder signals never changed.")
        if only == (1, 1):
            print("  A/B sit at 1,1 -- that is just the ESP32 internal pull-ups.")
            print("  The encoder is not driving the lines. Check, in order:")
            print("   1. blue wire actually on 3V3 (encoder supply)")
            print("   2. black wire actually on GND")
            print("   3. yellow on D18 / green on D19, seated firmly")
            print("   4. encoder itself may be faulty")
        elif only == (0, 0):
            print("  A/B sit at 0,0 -- signal lines look shorted to ground.")
            print("  Check that green/yellow are not on a GND rail.")
        else:
            print(f"  A/B stuck at {only}.")
    elif span == 0:
        print("RESULT: A/B toggle but the count never moves.")
        print("  Wiring is fine; decoding is not. This would be a firmware bug.")
    elif span < 20:
        print(f"RESULT: counting, but only {span} counts -- turn the shaft more,")
        print("  or one channel may be intermittent.")
    else:
        print(f"RESULT: encoder works. span={span} counts.")
        print(f"  One full output revolution should be about 2464 counts.")
        print("  Next: RESET_ENCODER, turn the shaft exactly one turn, GET_STATE,")
        print("  and compare -- that confirms the 1:56 gear ratio.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
