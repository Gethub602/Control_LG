"""
Read-only firmware probe.

Opening the port toggles DTR, which resets the ESP32, so anything the sketch
prints in setup() lands in the buffer. Many sketches announce their pin map
there. Afterwards a few harmless introspection commands are tried.

Never sends SET_PWM. The motor is not driven.
"""

import argparse
import sys
import time

import serial

SAFE_PROBE_COMMANDS = [
    "HELP",
    "?",
    "VERSION",
    "INFO",
    "STATUS",
    "CONFIG",
    "PINS",
    "GET_STATE",
]


def parse_args():
    p = argparse.ArgumentParser(description="Read-only ESP32 firmware probe.")
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--baudrate", type=int, default=115200)
    p.add_argument("--boot-seconds", type=float, default=4.0)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Opening {args.port} @ {args.baudrate} (read-only; no PWM is sent)")
    print("-" * 62)

    ser = serial.Serial(args.port, args.baudrate, timeout=0.4)

    # 1) capture whatever the sketch prints after the DTR-induced reset
    print(f"[boot output, {args.boot_seconds:.0f}s]")
    t_end = time.time() + args.boot_seconds
    boot_lines = 0
    while time.time() < t_end:
        raw = ser.readline()
        if raw:
            text = raw.decode(errors="replace").rstrip()
            if text:
                print(f"  | {text}")
                boot_lines += 1
    if boot_lines == 0:
        print("  (nothing printed on boot)")

    # 2) try harmless introspection commands
    print()
    print("[probe commands]")
    ser.reset_input_buffer()
    for cmd in SAFE_PROBE_COMMANDS:
        ser.reset_input_buffer()
        ser.write((cmd + "\n").encode())
        ser.flush()
        time.sleep(0.35)
        replies = []
        while ser.in_waiting:
            line = ser.readline().decode(errors="replace").strip()
            if line:
                replies.append(line)
        if replies:
            print(f"  {cmd:<10} -> " + " | ".join(replies))
        else:
            print(f"  {cmd:<10} -> <no response>")

    ser.close()
    print()
    print("-" * 62)
    print("If no pin map appeared, the sketch source is needed to confirm which")
    print("GPIOs it reads for encoder A/B.")


if __name__ == "__main__":
    sys.exit(main())
