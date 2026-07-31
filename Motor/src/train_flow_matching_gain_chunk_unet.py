"""
Conditional Flow Matching (rectified flow) for 20-step PID gain chunks.

Same data pipeline, same conditional U-Net backbone, and same artifact layout as
train_diffusion_gain_chunk_unet.py. Only the generative objective changes:

  Diffusion (DDPM/DDIM)          Flow matching (rectified flow)
  --------------------------     ------------------------------------
  y_t = sqrt(a_t) y0             y_t = (1-t) * y_noise + t * y0
        + sqrt(1-a_t) eps        (straight line, t in [0,1])
  predict eps                    predict velocity v = y0 - y_noise
  sample: DDIM, ~20 net evals    sample: Euler/midpoint ODE, 1-4 net evals

The motivation here is latency. The final DDIM20 result was bounded by the
asynchronous chunk window (p90 generator time ~0.61 s against a 0.5 s
schedule_start_time assumption), and DDIM30 lost to DDIM20 purely because it was
slower. Rectified flow targets that bottleneck directly: a straight probability
path can be integrated in very few steps.
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))
sys.path.append(str(CURRENT_DIR))

from config import FIGURE_DIR, MODEL_DIR, RESULTS_DIR  # noqa: E402
from train_diffusion_gain_chunk_baselines import (  # noqa: E402
    DEFAULT_OBS_COLS,
    GAIN_BOUNDS,
    GAIN_COLS,
    denormalize_gain_sequence,
    latest_label_path,
    load_chunk_labels,
    make_arrays,
    normalize_gain_sequence,
    scale_inputs,
    split_by_trajectory,
    static_cols,
)
from train_diffusion_gain_chunk_unet import (  # noqa: E402
    build_unet,
    configure_tensorflow,
    diffusion_to_gain_space,
    gain_to_diffusion_space,
    plot_history,
)

SUMMARY_DIR = RESULTS_DIR / "summary"


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a conditional flow-matching U-Net for PID gain chunks."
    )
    p.add_argument("--dataset", default="")
    p.add_argument("--profile", default="tracking_first")
    p.add_argument("--quality", choices=["all", "top_k", "best"], default="top_k")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows", type=int, default=0)

    # U-Net backbone (kept identical to the diffusion trainer)
    p.add_argument("--base-filters", type=int, default=64)
    p.add_argument("--condition-dim", type=int, default=128)
    p.add_argument("--time-embed-dim", type=int, default=64)
    p.add_argument("--condition-mode", default="avg", choices=["avg", "flatten", "gru"])
    p.add_argument("--norm", default="batch", choices=["batch", "layer"])

    # Flow-matching specific
    p.add_argument(
        "--sigma-min",
        type=float,
        default=0.0,
        help="Optional noise floor of the conditional path (0 = pure rectified flow).",
    )
    p.add_argument(
        "--time-sampling",
        default="uniform",
        choices=["uniform", "logit_normal"],
        help="Distribution of t during training. logit_normal emphasises mid-path.",
    )
    p.add_argument("--logit-normal-mean", type=float, default=0.0)
    p.add_argument("--logit-normal-std", type=float, default=1.0)
    p.add_argument(
        "--solver", default="midpoint", choices=["euler", "midpoint", "heun"]
    )
    p.add_argument("--eval-steps", default="1,2,4,8,20")
    p.add_argument("--sample-count", type=int, default=4)
    # `diffusion_steps` is unused by flow matching but kept so the saved payload
    # stays shape-compatible with build_unet / the diffusion generator loader.
    p.add_argument("--diffusion-steps", type=int, default=1000)
    p.add_argument("--run-label", default="")
    return p.parse_args()


def prepare_data(args):
    dataset_path = Path(args.dataset) if args.dataset else latest_label_path(args.profile)
    if not dataset_path.is_absolute():
        dataset_path = MOTOR_DIR / dataset_path
    df = load_chunk_labels(dataset_path, args.quality, args.max_rows, args.seed)
    train_df, test_df = split_by_trajectory(df, args.test_size, args.seed)
    static_feature_cols = static_cols(df)
    train_seq, train_static, y_train = make_arrays(
        train_df, DEFAULT_OBS_COLS, static_feature_cols
    )
    test_seq, test_static, y_test = make_arrays(
        test_df, DEFAULT_OBS_COLS, static_feature_cols
    )
    (
        train_seq,
        test_seq,
        train_static,
        test_static,
        seq_scaler,
        static_scaler,
    ) = scale_inputs(train_seq, test_seq, train_static, test_static)
    return {
        "dataset_path": dataset_path,
        "train_seq": train_seq,
        "test_seq": test_seq,
        "train_static": train_static,
        "test_static": test_static,
        "y_train_gain": y_train,
        "y_test_gain": y_test,
        "y_train_flow": gain_to_diffusion_space(y_train),
        "y_test_flow": gain_to_diffusion_space(y_test),
        "seq_scaler": seq_scaler,
        "static_scaler": static_scaler,
        "static_feature_cols": static_feature_cols,
    }


def sample_time(tf, batch_size, args):
    """t ~ U(0,1) or logit-normal, matching common rectified-flow practice."""
    if args.time_sampling == "logit_normal":
        z = tf.random.normal(
            (batch_size,), mean=args.logit_normal_mean, stddev=args.logit_normal_std
        )
        return tf.sigmoid(z)
    return tf.random.uniform((batch_size,), minval=0.0, maxval=1.0)


def train_model(tf, model, data, args):
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    train_ds = (
        tf.data.Dataset.from_tensor_slices(
            (
                data["train_seq"].astype(np.float32),
                data["train_static"].astype(np.float32),
                data["y_train_flow"].astype(np.float32),
            )
        )
        .shuffle(min(len(data["train_seq"]), 20000), seed=args.seed)
        .batch(args.batch_size)
    )
    val_ds = tf.data.Dataset.from_tensor_slices(
        (
            data["test_seq"].astype(np.float32),
            data["test_static"].astype(np.float32),
            data["y_test_flow"].astype(np.float32),
        )
    ).batch(args.batch_size)

    sigma_min = float(args.sigma_min)
    # The U-Net expects an int32 timestep for its sinusoidal embedding, so the
    # continuous t in [0,1] is mapped onto the same integer grid the diffusion
    # model used. This keeps the backbone byte-identical between both methods.
    time_scale = float(args.diffusion_steps - 1)

    @tf.function
    def step(obs, static, y0, training):
        batch_size = tf.shape(y0)[0]
        t = sample_time(tf, batch_size, args)
        t_b = tf.reshape(t, (-1, 1, 1))

        y_noise = tf.random.normal(tf.shape(y0), dtype=tf.float32)

        # Conditional optimal-transport path: straight line from noise to data.
        #   y_t = (1 - (1 - sigma_min) t) * y_noise + t * y0
        #   v*  = y0 - (1 - sigma_min) * y_noise
        y_t = (1.0 - (1.0 - sigma_min) * t_b) * y_noise + t_b * y0
        v_target = y0 - (1.0 - sigma_min) * y_noise

        t_idx = tf.cast(tf.round(t * time_scale), tf.int32)
        with tf.GradientTape() as tape:
            v_pred = model([y_t, t_idx, obs, static], training=training)
            loss = tf.reduce_mean(tf.square(v_target - v_pred))
        if training:
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    history = {"loss": [], "val_loss": []}
    best_weights = None
    best_val = np.inf
    wait = 0
    for epoch in range(1, int(args.epochs) + 1):
        tr = [float(step(o, s, y, True).numpy()) for o, s, y in train_ds]
        va = [float(step(o, s, y, False).numpy()) for o, s, y in val_ds]
        loss, val_loss = float(np.mean(tr)), float(np.mean(va))
        history["loss"].append(loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch:03d}: loss={loss:.6f}, val_loss={val_loss:.6f}")
        if val_loss < best_val - 1e-6:
            best_val, best_weights, wait = val_loss, model.get_weights(), 0
        else:
            wait += 1
            if wait >= int(args.patience):
                print(f"Early stopping at epoch {epoch}. Best val_loss={best_val:.6f}")
                break
    if best_weights is not None:
        model.set_weights(best_weights)
    return pd.DataFrame(history)


def flow_sample(tf, model, obs, static, args, num_steps: int, sample_count: int = 1):
    """
    Integrate dy/dt = v_theta(y, t, cond) from t=0 (noise) to t=1 (data).

    num_steps is the number of network evaluations for euler; midpoint/heun use
    two evaluations per step, which is accounted for by the caller's benchmark.
    """
    n = obs.shape[0]
    horizon_steps = int(model.output_shape[1])
    obs_rep = np.repeat(obs, sample_count, axis=0).astype(np.float32)
    static_rep = np.repeat(static, sample_count, axis=0).astype(np.float32)
    total = obs_rep.shape[0]

    y = tf.random.normal((total, horizon_steps, len(GAIN_COLS)), dtype=tf.float32)
    time_scale = float(args.diffusion_steps - 1)
    dt = 1.0 / float(num_steps)

    def velocity(y_cur, t_scalar):
        t_idx = tf.fill((total,), int(round(float(t_scalar) * time_scale)))
        return model([y_cur, t_idx, obs_rep, static_rep], training=False)

    for i in range(int(num_steps)):
        t0 = i * dt
        if args.solver == "euler":
            y = y + dt * velocity(y, t0)
        elif args.solver == "midpoint":
            k1 = velocity(y, t0)
            y_mid = y + 0.5 * dt * k1
            y = y + dt * velocity(y_mid, t0 + 0.5 * dt)
        else:  # heun
            k1 = velocity(y, t0)
            y_end = y + dt * k1
            k2 = velocity(y_end, min(t0 + dt, 1.0))
            y = y + 0.5 * dt * (k1 + k2)

    y = tf.clip_by_value(y, -1.0, 1.0).numpy()
    return y.reshape(n, sample_count, horizon_steps, len(GAIN_COLS))


def evaluate_steps(tf, model, data, args):
    """MAE and wall-clock latency across ODE step budgets."""
    step_list = [int(v) for v in str(args.eval_steps).split(",") if v.strip()]
    y_true = data["y_test_gain"]
    obs, static = data["test_seq"], data["test_static"]
    rows = []

    for num_steps in step_list:
        # warm up once so tracing cost is excluded from the timing
        flow_sample(tf, model, obs[:1], static[:1], args, num_steps, 1)

        t0 = time.perf_counter()
        samples = flow_sample(tf, model, obs, static, args, num_steps, args.sample_count)
        batch_sec = time.perf_counter() - t0

        # single-condition latency, which is what the server actually pays
        lat = []
        for _ in range(12):
            t1 = time.perf_counter()
            flow_sample(tf, model, obs[:1], static[:1], args, num_steps, 1)
            lat.append(time.perf_counter() - t1)

        gain = diffusion_to_gain_space(
            samples.reshape((-1, samples.shape[2], samples.shape[3]))
        ).reshape(samples.shape)
        mean_gain = np.mean(gain, axis=1)
        all_mae = np.mean(np.abs(gain - y_true[:, None, :, :]), axis=(2, 3))
        best_gain = gain[np.arange(len(gain)), np.argmin(all_mae, axis=1)]

        nfe_per_step = 1 if args.solver == "euler" else 2
        rows.append(
            {
                "num_steps": num_steps,
                "solver": args.solver,
                "nfe": num_steps * nfe_per_step,
                "sample_mean_mae": float(np.mean(np.abs(mean_gain - y_true))),
                "best_of_n_mae": float(np.mean(np.abs(best_gain - y_true))),
                "kp_mae": float(np.mean(np.abs(mean_gain[:, :, 0] - y_true[:, :, 0]))),
                "ki_mae": float(np.mean(np.abs(mean_gain[:, :, 1] - y_true[:, :, 1]))),
                "kd_mae": float(np.mean(np.abs(mean_gain[:, :, 2] - y_true[:, :, 2]))),
                "batch_sec": batch_sec,
                "single_mean_sec": float(np.mean(lat)),
                "single_p90_sec": float(np.percentile(lat, 90)),
            }
        )
        print(
            f"  steps={num_steps:>3} nfe={rows[-1]['nfe']:>3}  "
            f"mae={rows[-1]['sample_mean_mae']:.5f}  "
            f"p90={rows[-1]['single_p90_sec'] * 1000:.1f} ms"
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    tf = configure_tensorflow(args.seed)
    data = prepare_data(args)

    obs_steps, obs_dim = data["train_seq"].shape[1], data["train_seq"].shape[2]
    static_dim = data["train_static"].shape[1]
    horizon_steps = data["y_train_flow"].shape[1]

    print(
        f"Dataset: {data['dataset_path'].name}\n"
        f"train={len(data['train_seq'])}, test={len(data['test_seq'])}, "
        f"obs=({obs_steps},{obs_dim}), static={static_dim}, horizon={horizon_steps}"
    )

    model = build_unet(tf, obs_steps, obs_dim, static_dim, horizon_steps, args)
    print(f"Trainable params: {model.count_params():,}")

    history = train_model(tf, model, data, args)

    print("Evaluating ODE step budgets...")
    eval_df = evaluate_steps(tf, model, data, args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = args.run_label or "flow_matching_gain_chunk"
    stem = f"flow_matching_gain_chunk_unet_{label}_{timestamp}"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    weights_path = MODEL_DIR / f"{stem}.weights.h5"
    model.save_weights(weights_path)

    # Payload layout mirrors train_diffusion_gain_chunk_unet.py so the server-side
    # generator can load either model with the same reader.
    payload = {
        "model_type": "conditional_flow_matching_unet_gain_chunk",
        "weights_path": str(weights_path),
        "architecture": {
            "time_embed_dim": int(args.time_embed_dim),
            "condition_dim": int(args.condition_dim),
            "base_filters": int(args.base_filters),
            "dropout": float(args.dropout),
            "norm": str(args.norm),
            "condition_mode": str(args.condition_mode),
        },
        "dataset_path": str(data["dataset_path"]),
        "label_profile": args.profile,
        "quality": args.quality,
        "obs_cols": list(DEFAULT_OBS_COLS),
        "static_feature_cols": list(data["static_feature_cols"]),
        "gain_cols": list(GAIN_COLS),
        "gain_bounds": GAIN_BOUNDS,
        "obs_steps": int(obs_steps),
        "obs_dim": int(obs_dim),
        "static_dim": int(static_dim),
        "horizon_steps": int(horizon_steps),
        # kept for backbone compatibility: the sinusoidal embedding grid size
        "diffusion_steps": int(args.diffusion_steps),
        "flow_solver": str(args.solver),
        "flow_sigma_min": float(args.sigma_min),
        "flow_time_sampling": str(args.time_sampling),
        "sample_count": int(args.sample_count),
        "seq_scaler": data["seq_scaler"],
        "static_scaler": data["static_scaler"],
        "args": vars(args),
        "eval": eval_df.to_dict(orient="records"),
    }
    joblib_path = MODEL_DIR / f"{stem}.joblib"
    joblib.dump(payload, joblib_path)

    eval_path = SUMMARY_DIR / f"{stem}_step_sweep.csv"
    eval_df.to_csv(eval_path, index=False)
    history.to_csv(SUMMARY_DIR / f"{stem}_history.csv", index=False)
    plot_history(history, FIGURE_DIR / f"{stem}_history.png")

    print(json.dumps({"model": str(joblib_path), "eval": str(eval_path)}, indent=2))


if __name__ == "__main__":
    main()
