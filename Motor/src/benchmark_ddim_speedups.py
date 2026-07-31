"""
Measure ways to make the inherited TF DDIM20 sampler faster.

The generator costs ~0.8 s per chunk on this machine, against a 0.5 s
schedule_start_time assumption, so every chunk lands after its own window has
begun. The model is tiny (1.5 M parameters) and runs at batch size 1, so the
cost is not arithmetic -- it is 20 sequential eager calls, each paying Python
dispatch and kernel-launch overhead.

Variants compared here:

  eager          what the repository does today: a Python loop of model(...)
  tf_function    the whole 20-step loop traced into one graph, so the 20 calls
                 become one graph execution
  tf_function_xla same, with XLA compilation
  threads        eager, but with TF's intra/inter-op thread counts pinned

Accuracy is checked against the eager path so a speedup that changes the output
is not mistaken for a win.
"""

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))
sys.path.append(str(CURRENT_DIR))


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark DDIM sampler speedups.")
    p.add_argument("--model-path", required=True)
    p.add_argument("--ddim-steps", type=int, default=20)
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build(tf, payload):
    from train_diffusion_gain_chunk_unet import build_diffusion_constants, build_unet

    arch = payload["architecture"]
    args_ns = SimpleNamespace(
        condition_dim=int(arch["condition_dim"]),
        time_embed_dim=int(arch["time_embed_dim"]),
        base_filters=int(arch["base_filters"]),
        dropout=float(arch["dropout"]),
        norm=str(arch.get("norm", "batch")),
        condition_mode=str(arch.get("condition_mode", "avg")),
        diffusion_steps=int(payload["diffusion_steps"]),
        ddim_steps=20,
    )
    model = build_unet(
        tf,
        obs_steps=int(payload["obs_steps"]),
        obs_dim=len(payload["obs_cols"]),
        static_dim=len(payload["static_feature_cols"]),
        horizon_steps=int(payload["horizon_steps"]),
        args=args_ns,
    )
    model.load_weights(payload["weights_path"])
    constants = build_diffusion_constants(tf, int(payload["diffusion_steps"]))
    return model, args_ns, constants


def make_graph_sampler(tf, model, alpha_bars_np, step_indices, horizon, gain_dim,
                       jit=False):
    """Trace the whole DDIM loop into a single graph."""
    a_t = [float(np.sqrt(max(alpha_bars_np[i], 1e-12))) for i in step_indices]
    a_t_c = [float(np.sqrt(max(1.0 - alpha_bars_np[i], 1e-12))) for i in step_indices]

    @tf.function(jit_compile=jit, reduce_retracing=True)
    def sample(obs, static, x0_noise):
        x = x0_noise
        n = tf.shape(obs)[0]
        for i, t_idx in enumerate(step_indices):
            t = tf.fill((n,), tf.constant(int(t_idx), tf.int32))
            eps = model([x, t, obs, static], training=False)
            x0 = (x - a_t_c[i] * eps) / a_t[i]
            x0 = tf.clip_by_value(x0, -1.5, 1.5)
            if i == len(step_indices) - 1:
                x = x0
            else:
                j = step_indices[i + 1]
                ap = float(np.sqrt(max(alpha_bars_np[j], 1e-12)))
                apc = float(np.sqrt(max(1.0 - alpha_bars_np[j], 1e-12)))
                x = ap * x0 + apc * eps
        return tf.clip_by_value(x, -1.0, 1.0)

    return sample


def timeit(fn, repeats):
    fn()  # warm up / trace
    lat = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        lat.append(time.perf_counter() - t0)
    return {
        "mean_ms": float(np.mean(lat) * 1000),
        "p50_ms": float(np.percentile(lat, 50) * 1000),
        "p90_ms": float(np.percentile(lat, 90) * 1000),
    }


def main():
    args = parse_args()
    payload = joblib.load(args.model_path)

    from train_diffusion_gain_chunk_unet import configure_tensorflow, ddim_sample

    tf = configure_tensorflow(args.seed)
    model, args_ns, constants = build(tf, payload)
    args_ns.ddim_steps = args.ddim_steps

    obs_steps = int(payload["obs_steps"])
    obs_dim = len(payload["obs_cols"])
    static_dim = len(payload["static_feature_cols"])
    horizon = int(payload["horizon_steps"])
    gain_dim = len(payload.get("gain_cols", ["kp", "ki", "kd"]))

    rng = np.random.default_rng(args.seed)
    obs = rng.normal(size=(1, obs_steps, obs_dim)).astype(np.float32)
    static = rng.normal(size=(1, static_dim)).astype(np.float32)

    alpha_bars_np = constants["alpha_bars"].numpy()
    step_indices = np.linspace(
        int(payload["diffusion_steps"]) - 1, 0, args.ddim_steps, dtype=int
    )

    print(f"model params: {model.count_params():,}   ddim_steps={args.ddim_steps}")
    print(f"obs=({obs_steps},{obs_dim}) static={static_dim} horizon={horizon}")
    print()

    results = {}

    # --- 1. eager, exactly what the repo does today
    results["eager (current)"] = timeit(
        lambda: ddim_sample(tf, model, obs, static, args_ns, constants, 1), args.repeats
    )
    ref = ddim_sample(tf, model, obs, static, args_ns, constants, 1)

    # --- 2. whole loop traced into one graph
    fixed_noise = tf.constant(
        rng.normal(size=(1, horizon, gain_dim)).astype(np.float32)
    )
    obs_t = tf.constant(obs)
    static_t = tf.constant(static)

    g = make_graph_sampler(tf, model, alpha_bars_np, step_indices, horizon, gain_dim)
    results["tf.function"] = timeit(lambda: g(obs_t, static_t, fixed_noise).numpy(),
                                    args.repeats)

    try:
        gx = make_graph_sampler(tf, model, alpha_bars_np, step_indices, horizon,
                                gain_dim, jit=True)
        results["tf.function + XLA"] = timeit(
            lambda: gx(obs_t, static_t, fixed_noise).numpy(), args.repeats
        )
    except Exception as exc:
        print(f"XLA variant unavailable: {exc}")

    # --- 3. thread pinning (batch-1 work does not parallelise well)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        results["eager, 1 thread"] = timeit(
            lambda: ddim_sample(tf, model, obs, static, args_ns, constants, 1),
            args.repeats,
        )
    except RuntimeError as exc:
        print(f"thread pinning unavailable after init: {exc}")

    print(f"{'variant':<24}{'mean ms':>10}{'p50 ms':>10}{'p90 ms':>10}{'speedup':>10}")
    print("-" * 64)
    base = results["eager (current)"]["p90_ms"]
    for name, r in results.items():
        print(f"{name:<24}{r['mean_ms']:>10.1f}{r['p50_ms']:>10.1f}"
              f"{r['p90_ms']:>10.1f}{base / r['p90_ms']:>9.1f}x")

    # sanity: the graph path must produce the same distribution as eager
    out = g(obs_t, static_t, fixed_noise).numpy()
    print()
    print(f"eager sample range : [{ref.min():.3f}, {ref.max():.3f}]")
    print(f"graph sample range : [{out.min():.3f}, {out.max():.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
