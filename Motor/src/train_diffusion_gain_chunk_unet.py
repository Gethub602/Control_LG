import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))

from config import FIGURE_DIR, MODEL_DIR, RESULTS_DIR  # noqa: E402
from train_diffusion_gain_chunk_baselines import (  # noqa: E402
    DEFAULT_OBS_COLS,
    GAIN_BOUNDS,
    GAIN_COLS,
    PROCESSED_ROOT,
    denormalize_gain_sequence,
    latest_label_path,
    load_chunk_labels,
    make_arrays,
    normalize_gain_sequence,
    scale_inputs,
    split_by_trajectory,
    static_cols,
)


SUMMARY_DIR = RESULTS_DIR / "summary"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a conditional 1D U-Net diffusion model for PID gain chunks."
    )
    parser.add_argument("--dataset", default="")
    parser.add_argument("--profile", default="tracking_first")
    parser.add_argument("--quality", choices=["all", "top_k", "best"], default="top_k")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=28)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--time-embed-dim", type=int, default=64)
    parser.add_argument("--condition-dim", type=int, default=128)
    parser.add_argument("--base-filters", type=int, default=64)
    parser.add_argument(
        "--norm",
        choices=["batch", "layer"],
        default="batch",
        help="Normalization inside FiLM residual blocks.",
    )
    parser.add_argument(
        "--condition-mode",
        choices=["avg", "last_avg", "attention"],
        default="avg",
        help="How recent observation sequence is encoded.",
    )
    parser.add_argument("--run-label", default="")
    return parser.parse_args()


def configure_tensorflow(seed: int):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    return tf


def gain_to_diffusion_space(y_gain):
    y01 = normalize_gain_sequence(y_gain)
    return (2.0 * y01 - 1.0).astype(np.float32)


def diffusion_to_gain_space(y_diff):
    y01 = np.clip((y_diff + 1.0) / 2.0, 0.0, 1.0)
    return denormalize_gain_sequence(y01)


def cosine_beta_schedule(num_steps: int, s: float = 0.008):
    steps = np.arange(num_steps + 1, dtype=np.float64)
    x = steps / num_steps
    alphas_cumprod = np.cos(((x + s) / (1 + s)) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 1e-5, 0.999).astype(np.float32)


def sinusoidal_embedding(tf, t, dim: int):
    half = dim // 2
    t = tf.cast(tf.reshape(t, (-1, 1)), tf.float32)
    freqs = tf.exp(
        tf.linspace(
            tf.math.log(tf.constant(1.0, tf.float32)),
            tf.math.log(tf.constant(10000.0, tf.float32)),
            half,
        )
        * -1.0
    )
    args = t * tf.reshape(freqs, (1, -1))
    emb = tf.concat([tf.sin(args), tf.cos(args)], axis=1)
    if dim % 2 == 1:
        emb = tf.pad(emb, [[0, 0], [0, 1]])
    return emb


def norm_layer(tf, name: str, norm: str):
    if norm == "layer":
        return tf.keras.layers.LayerNormalization(name=name)
    return tf.keras.layers.BatchNormalization(name=name)


def film_res_block(tf, x, cond, filters, kernel_size, dropout, name, norm: str):
    shortcut = x
    x = tf.keras.layers.Conv1D(filters, kernel_size, padding="same", name=f"{name}_conv1")(x)
    x = norm_layer(tf, f"{name}_norm1", norm)(x)
    gamma_beta = tf.keras.layers.Dense(filters * 2, name=f"{name}_film")(cond)
    gamma, beta = tf.split(gamma_beta, 2, axis=-1)
    gamma = tf.expand_dims(gamma, axis=1)
    beta = tf.expand_dims(beta, axis=1)
    x = x * (1.0 + gamma) + beta
    x = tf.keras.layers.Activation("swish", name=f"{name}_act1")(x)
    x = tf.keras.layers.Dropout(dropout, name=f"{name}_drop")(x)
    x = tf.keras.layers.Conv1D(filters, kernel_size, padding="same", name=f"{name}_conv2")(x)
    x = norm_layer(tf, f"{name}_norm2", norm)(x)
    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv1D(filters, 1, padding="same", name=f"{name}_skip")(shortcut)
    x = tf.keras.layers.Add(name=f"{name}_add")([x, shortcut])
    return tf.keras.layers.Activation("swish", name=f"{name}_act2")(x)


def match_temporal_length(tf, x, target_length: int, name: str):
    current_length = x.shape[1]
    target_length = int(target_length)
    if current_length == target_length:
        return x
    if current_length is not None and current_length < target_length:
        return tf.keras.layers.ZeroPadding1D(
            padding=(0, target_length - int(current_length)),
            name=f"{name}_pad",
        )(x)
    if current_length is not None and current_length > target_length:
        return tf.keras.layers.Cropping1D(
            cropping=(0, int(current_length) - target_length),
            name=f"{name}_crop",
        )(x)
    return tf.keras.layers.Lambda(
        lambda z: z[:, :target_length, :],
        name=f"{name}_dynamic_crop",
    )(x)


def build_condition_encoder(tf, obs_steps, obs_dim, static_dim, condition_dim, dropout, condition_mode):
    obs_input = tf.keras.Input(shape=(obs_steps, obs_dim), name="obs_sequence")
    static_input = tf.keras.Input(shape=(static_dim,), name="state_features")
    x = tf.keras.layers.Conv1D(64, 3, padding="causal", activation="swish")(obs_input)
    x = tf.keras.layers.Conv1D(96, 3, padding="causal", activation="swish")(x)
    if condition_mode == "last_avg":
        avg = tf.keras.layers.GlobalAveragePooling1D(name="obs_avg_pool")(x)
        last = tf.keras.layers.Lambda(lambda z: z[:, -1, :], name="obs_last_token")(x)
        x = tf.keras.layers.Concatenate(name="obs_last_avg_concat")([last, avg])
    elif condition_mode == "attention":
        score = tf.keras.layers.Dense(1, name="obs_attention_score")(x)
        weight = tf.keras.layers.Softmax(axis=1, name="obs_attention_weight")(score)
        weighted = tf.keras.layers.Multiply(name="obs_attention_multiply")([x, weight])
        attn = tf.keras.layers.Lambda(
            lambda z: tf.reduce_sum(z, axis=1),
            name="obs_attention_pool",
        )(weighted)
        last = tf.keras.layers.Lambda(lambda z: z[:, -1, :], name="obs_last_token")(x)
        x = tf.keras.layers.Concatenate(name="obs_attention_last_concat")([last, attn])
    else:
        x = tf.keras.layers.GlobalAveragePooling1D(name="obs_avg_pool")(x)
    x = tf.keras.layers.Concatenate()([x, static_input])
    x = tf.keras.layers.Dense(192, activation="swish")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    cond = tf.keras.layers.Dense(condition_dim, activation="swish", name="condition_embedding")(x)
    return obs_input, static_input, cond


def build_unet(tf, obs_steps, obs_dim, static_dim, horizon_steps, args):
    noisy_input = tf.keras.Input(shape=(horizon_steps, len(GAIN_COLS)), name="noisy_gain_chunk")
    t_input = tf.keras.Input(shape=(), dtype=tf.int32, name="diffusion_step")
    obs_input, static_input, cond_state = build_condition_encoder(
        tf,
        obs_steps,
        obs_dim,
        static_dim,
        args.condition_dim,
        args.dropout,
        getattr(args, "condition_mode", "avg"),
    )
    t_emb = tf.keras.layers.Lambda(
        lambda t: sinusoidal_embedding(tf, t, args.time_embed_dim),
        name="timestep_sinusoidal_embedding",
    )(t_input)
    t_emb = tf.keras.layers.Dense(args.condition_dim, activation="swish")(t_emb)
    cond = tf.keras.layers.Concatenate()([cond_state, t_emb])
    cond = tf.keras.layers.Dense(args.condition_dim, activation="swish")(cond)

    f = int(args.base_filters)
    x0 = tf.keras.layers.Conv1D(f, 3, padding="same", name="input_projection")(noisy_input)
    norm = getattr(args, "norm", "batch")
    x1 = film_res_block(tf, x0, cond, f, 3, args.dropout, "down_block_1", norm)
    d1 = tf.keras.layers.AveragePooling1D(pool_size=2, name="downsample_1")(x1)
    x2 = film_res_block(tf, d1, cond, f * 2, 3, args.dropout, "down_block_2", norm)
    d2 = tf.keras.layers.AveragePooling1D(pool_size=2, name="downsample_2")(x2)
    mid = film_res_block(tf, d2, cond, f * 4, 3, args.dropout, "mid_block_1", norm)
    mid = film_res_block(tf, mid, cond, f * 4, 3, args.dropout, "mid_block_2", norm)

    u2 = tf.keras.layers.UpSampling1D(size=2, name="upsample_2")(mid)
    u2 = match_temporal_length(tf, u2, int(x2.shape[1]), "upsample_2_match")
    u2 = tf.keras.layers.Concatenate(name="skip_2")([u2, x2])
    u2 = film_res_block(tf, u2, cond, f * 2, 3, args.dropout, "up_block_2", norm)
    u1 = tf.keras.layers.UpSampling1D(size=2, name="upsample_1")(u2)
    u1 = match_temporal_length(tf, u1, int(x1.shape[1]), "upsample_1_match")
    u1 = tf.keras.layers.Concatenate(name="skip_1")([u1, x1])
    u1 = film_res_block(tf, u1, cond, f, 3, args.dropout, "up_block_1", norm)
    output = tf.keras.layers.Conv1D(len(GAIN_COLS), 1, padding="same", name="predicted_noise")(u1)
    return tf.keras.Model([noisy_input, t_input, obs_input, static_input], output)


def prepare_data(args):
    dataset_path = Path(args.dataset) if args.dataset else latest_label_path(args.profile)
    if not dataset_path.is_absolute():
        dataset_path = MOTOR_DIR / dataset_path
    df = load_chunk_labels(dataset_path, args.quality, args.max_rows, args.seed)
    train_df, test_df = split_by_trajectory(df, args.test_size, args.seed)
    static_feature_cols = static_cols(df)
    train_seq, train_static, y_train = make_arrays(train_df, DEFAULT_OBS_COLS, static_feature_cols)
    test_seq, test_static, y_test = make_arrays(test_df, DEFAULT_OBS_COLS, static_feature_cols)
    (
        train_seq,
        test_seq,
        train_static,
        test_static,
        seq_scaler,
        static_scaler,
    ) = scale_inputs(train_seq, test_seq, train_static, test_static)
    y_train_diff = gain_to_diffusion_space(y_train)
    y_test_diff = gain_to_diffusion_space(y_test)
    return {
        "dataset_path": dataset_path,
        "train_df": train_df,
        "test_df": test_df,
        "train_seq": train_seq,
        "test_seq": test_seq,
        "train_static": train_static,
        "test_static": test_static,
        "y_train_gain": y_train,
        "y_test_gain": y_test,
        "y_train_diff": y_train_diff,
        "y_test_diff": y_test_diff,
        "seq_scaler": seq_scaler,
        "static_scaler": static_scaler,
        "static_feature_cols": static_feature_cols,
    }


def build_diffusion_constants(tf, diffusion_steps: int):
    betas = cosine_beta_schedule(diffusion_steps)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas).astype(np.float32)
    return {
        "betas": tf.constant(betas, dtype=tf.float32),
        "alphas": tf.constant(alphas.astype(np.float32), dtype=tf.float32),
        "alpha_bars": tf.constant(alpha_bars, dtype=tf.float32),
    }


def gather_alpha(tf, alpha_bars, t, like):
    values = tf.gather(alpha_bars, t)
    return tf.reshape(values, (-1, 1, 1)) * tf.ones_like(like[:, :, :1])


def train_model(tf, model, data, args, constants):
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    train_ds = tf.data.Dataset.from_tensor_slices(
        (
            data["train_seq"].astype(np.float32),
            data["train_static"].astype(np.float32),
            data["y_train_diff"].astype(np.float32),
        )
    ).shuffle(min(len(data["train_seq"]), 20000), seed=args.seed).batch(args.batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices(
        (
            data["test_seq"].astype(np.float32),
            data["test_static"].astype(np.float32),
            data["y_test_diff"].astype(np.float32),
        )
    ).batch(args.batch_size)

    alpha_bars = constants["alpha_bars"]
    diffusion_steps = int(args.diffusion_steps)

    @tf.function
    def step(obs, static, y0, training):
        batch_size = tf.shape(y0)[0]
        t = tf.random.uniform((batch_size,), minval=0, maxval=diffusion_steps, dtype=tf.int32)
        noise = tf.random.normal(tf.shape(y0), dtype=tf.float32)
        alpha_bar = gather_alpha(tf, alpha_bars, t, y0)
        yt = tf.sqrt(alpha_bar) * y0 + tf.sqrt(1.0 - alpha_bar) * noise
        with tf.GradientTape() as tape:
            pred = model([yt, t, obs, static], training=training)
            loss = tf.reduce_mean(tf.square(noise - pred))
        if training:
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    history = {"loss": [], "val_loss": []}
    best_weights = None
    best_val = np.inf
    wait = 0
    for epoch in range(1, int(args.epochs) + 1):
        train_losses = []
        val_losses = []
        for obs, static, y0 in train_ds:
            train_losses.append(float(step(obs, static, y0, True).numpy()))
        for obs, static, y0 in val_ds:
            val_losses.append(float(step(obs, static, y0, False).numpy()))
        loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["loss"].append(loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch:03d}: loss={loss:.6f}, val_loss={val_loss:.6f}")
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_weights = model.get_weights()
            wait = 0
        else:
            wait += 1
            if wait >= int(args.patience):
                print(f"Early stopping at epoch {epoch}. Best val_loss={best_val:.6f}")
                break
    if best_weights is not None:
        model.set_weights(best_weights)
    return pd.DataFrame(history)


def ddim_sample(tf, model, obs, static, args, constants, sample_count: int):
    n = obs.shape[0]
    horizon_steps = int(model.output_shape[1])
    obs_rep = np.repeat(obs, sample_count, axis=0).astype(np.float32)
    static_rep = np.repeat(static, sample_count, axis=0).astype(np.float32)
    total = obs_rep.shape[0]
    x = tf.random.normal((total, horizon_steps, len(GAIN_COLS)), dtype=tf.float32)
    alpha_bars = constants["alpha_bars"].numpy()
    step_indices = np.linspace(args.diffusion_steps - 1, 0, int(args.ddim_steps), dtype=int)

    for i, t_idx in enumerate(step_indices):
        t = tf.fill((total,), int(t_idx))
        eps = model([x, t, obs_rep, static_rep], training=False)
        a_t = float(alpha_bars[t_idx])
        x0 = (x - math.sqrt(max(1.0 - a_t, 1e-12)) * eps) / math.sqrt(max(a_t, 1e-12))
        x0 = tf.clip_by_value(x0, -1.5, 1.5)
        if i == len(step_indices) - 1:
            x = x0
        else:
            t_prev = int(step_indices[i + 1])
            a_prev = float(alpha_bars[t_prev])
            x = math.sqrt(max(a_prev, 1e-12)) * x0 + math.sqrt(max(1.0 - a_prev, 1e-12)) * eps
    x = tf.clip_by_value(x, -1.0, 1.0).numpy()
    x = x.reshape(n, sample_count, horizon_steps, len(GAIN_COLS))
    return x


def evaluate_samples(y_true_gain, sample_diff):
    sample_gain = diffusion_to_gain_space(sample_diff.reshape((-1, sample_diff.shape[2], sample_diff.shape[3])))
    sample_gain = sample_gain.reshape(sample_diff.shape)
    mean_gain = np.mean(sample_gain, axis=1)
    all_mae = np.mean(np.abs(sample_gain - y_true_gain[:, None, :, :]), axis=(2, 3))
    best_idx = np.argmin(all_mae, axis=1)
    best_gain = sample_gain[np.arange(len(sample_gain)), best_idx]

    rows = []
    for label, pred in [("sample_mean", mean_gain), ("best_of_n_oracle_mae", best_gain)]:
        true_flat = y_true_gain.reshape((-1, len(GAIN_COLS)))
        pred_flat = pred.reshape((-1, len(GAIN_COLS)))
        for idx, col in enumerate(GAIN_COLS):
            rows.append(
                {
                    "prediction": label,
                    "target": col,
                    "mae": float(mean_absolute_error(true_flat[:, idx], pred_flat[:, idx])),
                    "rmse": float(np.sqrt(mean_squared_error(true_flat[:, idx], pred_flat[:, idx]))),
                    "r2": float(r2_score(true_flat[:, idx], pred_flat[:, idx])),
                }
            )
        rows.append(
            {
                "prediction": label,
                "target": "all_gains",
                "mae": float(mean_absolute_error(true_flat, pred_flat)),
                "rmse": float(np.sqrt(mean_squared_error(true_flat, pred_flat))),
                "r2": float(r2_score(true_flat, pred_flat, multioutput="variance_weighted")),
            }
        )

    diversity = np.mean(np.std(sample_gain, axis=1), axis=(0, 1))
    for idx, col in enumerate(GAIN_COLS):
        rows.append(
            {
                "prediction": "sample_diversity_std",
                "target": col,
                "mae": float(diversity[idx]),
                "rmse": np.nan,
                "r2": np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_history(history: pd.DataFrame, path: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["loss"], label="train")
    ax.plot(history["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("noise MSE")
    ax.set_title("Diffusion U-Net Training")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tf = configure_tensorflow(args.seed)
    print("=" * 80)
    print("Train conditional diffusion U-Net gain chunk generator")
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPUs: {tf.config.list_physical_devices('GPU')}")
    print("=" * 80)

    data = prepare_data(args)
    constants = build_diffusion_constants(tf, int(args.diffusion_steps))
    obs_steps, obs_dim = data["train_seq"].shape[1:]
    horizon_steps = data["y_train_gain"].shape[1]
    model = build_unet(
        tf,
        obs_steps=obs_steps,
        obs_dim=obs_dim,
        static_dim=data["train_static"].shape[1],
        horizon_steps=horizon_steps,
        args=args,
    )
    history = train_model(tf, model, data, args, constants)

    eval_count = min(1024, len(data["test_seq"]))
    sample_diff = ddim_sample(
        tf,
        model,
        data["test_seq"][:eval_count],
        data["test_static"][:eval_count],
        args,
        constants,
        sample_count=int(args.sample_count),
    )
    metrics = evaluate_samples(data["y_test_gain"][:eval_count], sample_diff)

    run_label = args.run_label or data["dataset_path"].stem.replace("chunk_labels_", "")
    name = f"diffusion_gain_chunk_unet_{run_label}_{timestamp}"
    weights_path = MODEL_DIR / f"{name}.weights.h5"
    metadata_path = MODEL_DIR / f"{name}.joblib"
    history_path = SUMMARY_DIR / f"{name}_history.csv"
    metrics_path = SUMMARY_DIR / f"{name}_metrics.csv"
    figure_path = FIGURE_DIR / f"{name}_training.png"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    model.save_weights(weights_path)
    history.to_csv(history_path, index=False, encoding="utf-8-sig")
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    plot_history(history, figure_path)

    payload = {
        "model_type": "conditional_diffusion_unet_gain_chunk",
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
        "obs_cols": DEFAULT_OBS_COLS,
        "static_feature_cols": data["static_feature_cols"],
        "gain_cols": GAIN_COLS,
        "gain_bounds": GAIN_BOUNDS,
        "obs_steps": int(obs_steps),
        "horizon_steps": int(horizon_steps),
        "diffusion_steps": int(args.diffusion_steps),
        "ddim_steps": int(args.ddim_steps),
        "sample_count": int(args.sample_count),
        "seq_scaler": data["seq_scaler"],
        "static_scaler": data["static_scaler"],
        "metrics_path": str(metrics_path),
        "history_path": str(history_path),
        "figure_path": str(figure_path),
        "betas": cosine_beta_schedule(int(args.diffusion_steps)),
    }
    joblib.dump(payload, metadata_path)

    print(metrics.to_string(index=False))
    print(f"Saved weights: {weights_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
