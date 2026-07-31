"""
Measure counts-per-output-revolution and derive the real gearbox ratio.

The firmware's rpm output is only as good as GEAR_RATIO. That constant was
inferred from the nameplate (12V / 178 RPM -> roughly 1:56), which is an
estimate: the bare 520 motor's no-load speed varies between units. If the ratio
is wrong every rpm reading is off by the same factor, which silently invalidates
the tuned gain DB and any model trained against it.

Procedure: mark the output shaft, reset the counter, turn the shaft exactly N
full revolutions by hand during the measurement window, then stop.

Read-only with respect to the motor: no SET_PWM is sent.
"""

import argparse
import sys
import time

import serial

PULSES_PER_MOTOR_REV = 11.0
QUADRATURE_MULT = 4.0

# Gear ratios commonly offered for the JGB37-520, used only to suggest the
# nearest catalogue value alongside the measured one.
COMMON_RATIOS = [4.4, 9.6, 19.2, 30.0, 34.0, 56.0, 90.0, 131.0, 171.0, 270.0]


def parse_args():
    p = argparse.ArgumentParser(description="Calibrate gearbox ratio from encoder counts.")
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--baudrate", type=int, default=115200)
    p.add_argument("--turns", type=float, default=5.0,
                   help="How many full output-shaft revolutions you will turn.")
    p.add_argument("--seconds", type=float, default=30.0,
                   help="Measurement window length.")
    p.add_argument("--apply", action="store_true",
                   help="Send SET_GEAR with the measured ratio when finished.")
    return p.parse_args()


def ask(ser, cmd):
    ser.reset_input_buffer()
    ser.write((cmd + "\n").encode())
    ser.flush()
    return ser.readline().decode(errors="ignore").strip()


def read_count(ser):
    line = ask(ser, "GET_STATE")
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

    print(ask(ser, "VERSION"))
    print()
    print("=" * 66)
    print(f"  Mark the output shaft, then turn it EXACTLY {args.turns:g} full")
    print(f"  revolutions during the next {args.seconds:.0f} seconds, and stop.")
    print("  Direction does not matter. Accuracy of the final position does.")
    print("=" * 66)

    ask(ser, "RESET_ENCODER")
    time.sleep(0.2)

    t_end = time.time() + args.seconds
    while time.time() < t_end:
        remaining = t_end - time.time()
        count = read_count(ser)
        print(f"  {remaining:5.1f}s left   count={count}", end="\r", flush=True)
        time.sleep(0.2)

    print()
    print()
    final = read_count(ser)
    if final is None:
        print("Could not read the final count.")
        ser.close()
        return 1

    counts = abs(final)
    print(f"final count: {final}  (|{counts}| over {args.turns:g} turns)")

    if counts < 50:
        print()
        print("Almost nothing counted. Did the shaft actually turn?")
        ser.close()
        return 1

    counts_per_rev = counts / float(args.turns)
    ratio = counts_per_rev / (PULSES_PER_MOTOR_REV * QUADRATURE_MULT)
    nearest = min(COMMON_RATIOS, key=lambda r: abs(r - ratio))

    print(f"counts per output revolution: {counts_per_rev:.1f}")
    print(f"implied gear ratio:           1:{ratio:.2f}")
    print(f"nearest catalogue ratio:      1:{nearest:g}"
          f"   (off by {abs(nearest - ratio) / nearest * 100:.1f}%)")
    print()

    if abs(nearest - ratio) / nearest < 0.05:
        chosen = nearest
        print(f"Measurement matches the catalogue value 1:{nearest:g} within 5%.")
        print(f"Recommend GEAR_RATIO = {chosen:g}")
    else:
        chosen = ratio
        print("Measurement does not match a catalogue value closely.")
        print("Either the turn count was imprecise, or this is a non-standard")
        print(f"gearbox. Re-run with more turns to reduce error, or use {ratio:.2f}.")

    if args.apply:
        print()
        print(ask(ser, f"SET_GEAR {chosen:.4f}"))
        print(ask(ser, "VERSION"))
        print()
        print("Note: SET_GEAR is runtime-only and resets on reboot. To make it")
        print("permanent, edit GEAR_RATIO in firmware/motor_controller and reflash.")

    ser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
