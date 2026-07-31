"""
Dissect where the time goes in a sequential sampler, using tinygrad's tooling.

Context: the inherited TF DDIM20 sampler cost ~815 ms per chunk. Wrapping the
20-step loop in tf.function cut it to 27 ms. That 30x came from somewhere, but
TensorFlow only lets it be inferred from wall-clock timing.

tinygrad exposes the layer underneath. DEBUG=2 prints every kernel it launches
with its own timing, so the question "how much of this is dispatch overhead
rather than arithmetic?" can be answered by counting instead of guessing.

Run plainly for the timing comparison:

    python src/tinygrad_overhead_probe.py

Run with DEBUG=2 to see each kernel, or GRAPH=1 (needs graphviz) for the graph:

    DEBUG=2 python src/tinygrad_overhead_probe.py --steps 3 --skip-timing
"""

import argparse
import os
import time

import numpy as np
from tinygrad import Tensor, TinyJit
from tinygrad.device import Device


def parse_args():
    p = argparse.ArgumentParser(description="tinygrad overhead probe.")
    p.add_argument("--steps", type=int, default=20,
                   help="Sequential steps, mirroring the DDIM step count.")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--skip-timing", action="store_true",
                   help="Run the loop once and exit; for DEBUG inspection.")
    return p.parse_args()


class TinyBlock:
    """
    Stand-in for one denoising step.

    Deliberately small, like the real model: two matmuls and an activation at
    batch size one. The arithmetic is trivial, so anything that dominates the
    measurement is overhead rather than compute -- which is the whole point.
    """

    def __init__(self, width, seed=0):
        Tensor.manual_seed(seed)
        self.w1 = Tensor.randn(width, width)
        self.w2 = Tensor.randn(width, width)
        self.b = Tensor.randn(width)

    def __call__(self, x):
        h = (x @ self.w1).relu()
        h = (h @ self.w2) + self.b
        return h.tanh()


def sequential(block, x, steps):
    for _ in range(steps):
        x = block(x)
    return x


def bench(fn, repeats):
    fn()
    lat = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        lat.append(time.perf_counter() - t0)
    return float(np.mean(lat)), float(np.percentile(lat, 90))


def main():
    args = parse_args()
    print(f"device={Device.DEFAULT}  steps={args.steps}  width={args.width}")
    print(f"DEBUG={os.environ.get('DEBUG', '0')}")
    print()

    block = TinyBlock(args.width)
    x0 = Tensor.randn(1, args.width)

    if args.skip_timing:
        # one pass only, so DEBUG output stays readable
        sequential(block, x0, args.steps).realize()
        return 0

    eager_mean, eager_p90 = bench(
        lambda: sequential(block, x0, args.steps).numpy(), args.repeats
    )

    jitted = TinyJit(lambda t: sequential(block, t, args.steps).realize())
    for _ in range(3):  # TinyJit captures on the third call
        jitted(x0).numpy()
    jit_mean, jit_p90 = bench(lambda: jitted(x0).numpy(), args.repeats)

    # per-step cost isolates dispatch: the arithmetic per step is identical
    print(f"{'variant':<12}{'mean ms':>10}{'p90 ms':>10}{'per-step us':>14}")
    print("-" * 46)
    print(f"{'eager':<12}{eager_mean * 1000:>10.2f}{eager_p90 * 1000:>10.2f}"
          f"{eager_mean / args.steps * 1e6:>14.1f}")
    print(f"{'TinyJit':<12}{jit_mean * 1000:>10.2f}{jit_p90 * 1000:>10.2f}"
          f"{jit_mean / args.steps * 1e6:>14.1f}")
    print()
    print(f"speedup: {eager_mean / jit_mean:.1f}x")
    print(f"overhead removed per step: "
          f"{(eager_mean - jit_mean) / args.steps * 1e6:.1f} us")
    print()
    print("The per-step arithmetic is identical in both rows, so the difference")
    print("is what the JIT removed: Python dispatch, graph rebuild and per-kernel")
    print("launch cost. Re-run with DEBUG=2 to see the individual kernels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
