import argparse
import json
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


PROCESSED_ROOT = MOTOR_DIR / "data" / "processed" / "diffusion_gain_chunk_db"
SUMMARY_DIR = RESULTS_DIR / "summary"

GAIN_COLS = ["kp", "ki", "kd"]
DEFAULT_OBS_COLS = [
    "target",
    "current",
    "error",
    "error_derivative",
    "pwm",
    "raw_pwm",
    "kp",
    "ki",
    "kd",
    "integral",
    "pid_p_term",
    "pid_i_term",
    "pid_d_term",
    "time_since_start",
    "time_since_target_change",
]
STATIC_PREFIX = "state_"
STATIC_EXCLUDE = {"state_time_since_start"}
GAIN_BOUNDS = {
    "kp": (0.55, 1.45),
    "ki": (0.70, 2.50),
    "kd": (0.00, 0.12),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train supervised MLP/CNN baselines for 20-step PID gain chunks."
    )
    parser.add_argument("--dataset", default="")
    parser.add_argument("--profile", default="tracking_first")
    parser.add_argument(
        "--quality",
        choices=["all", "top_k", "best"],
        default="top_k",
        help="Which labeled chunks to imitate.",
    )
    parser.add_argument(
        "--model-type",
        choices=["mlp", "cnn", "cnn_residual", "cnn_attention"],
        default="mlp",
    )
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=22)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=0)
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


def latest_label_path(profile: str):
    paths = sorted(PROCESSED_ROOT.glob(f"chunk_labels_*_{profile}_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No label file found for profile={profile}")
    return paths[-1]


def parse_seq(value):
    if isinstance(value, str):
        return np.asarray(json.loads(value), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def normalize_gain_sequence(y):
    out = np.empty_like(y, dtype=np.float32)
    for idx, col in enumerate(GAIN_COLS):
        lo, hi = GAIN_BOUNDS[col]
        out[:, :, idx] = (y[:, :, idx] - lo) / max(hi - lo, 1e-9)
    return np.clip(out, 0.0, 1.0)


def denormalize_gain_sequence(y_norm):
    out = np.empty_like(y_norm, dtype=np.float32)
    for idx, col in enumerate(GAIN_COLS):
        lo, hi = GAIN_BOUNDS[col]
        out[:, :, idx] = lo + y_norm[:, :, idx] * (hi - lo)
    return out


def load_chunk_labels(path: Path, quality: str, max_rows: int, seed: int):
    df = pd.read_csv(path)
    if quality == "top_k":
        df = df[df["is_top_k_in_condition"].astype(bool)].copy()
    elif quality == "best":
        df = df[df["is_condition_best"].astype(bool)].copy()
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).copy()
    return df.reset_index(drop=True)


def static_cols(df: pd.DataFrame):
    cols = [
        col
        for col in df.columns
        if col.startswith(STATIC_PREFIX)
        and col not in STATIC_EXCLUDE
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    return cols


def make_arrays(df: pd.DataFrame, obs_cols, static_feature_cols):
    obs_arrays = []
    for col in obs_cols:
        seq_col = f"obs_{col}_seq"
        if seq_col not in df.columns:
            raise ValueError(f"Missing sequence column: {seq_col}")
        obs_arrays.append(np.stack(df[seq_col].apply(parse_seq).to_numpy()))
    x_seq = np.stack(obs_arrays, axis=2).astype(np.float32)
    x_static = df[static_feature_cols].to_numpy(dtype=np.float32)

    y_arrays = []
    for col in GAIN_COLS:
        y_arrays.append(np.stack(df[f"future_{col}_seq"].apply(parse_seq).to_numpy()))
    y = np.stack(y_arrays, axis=2).astype(np.float32)
    return x_seq, x_static, y


def split_by_trajectory(df: pd.DataFrame, test_size: float, seed: int):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df["trajectory_id"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def scale_inputs(train_seq, test_seq, train_static, test_static):
    seq_scaler = StandardScaler()
    static_scaler = StandardScaler()

    n_train, obs_steps, obs_dim = train_seq.shape
    n_test = test_seq.shape[0]
    train_seq_scaled = seq_scaler.fit_transform(
        train_seq.reshape(n_train, obs_steps * obs_dim)
    ).reshape(n_train, obs_steps, obs_dim)
    test_seq_scaled = seq_scaler.transform(
        test_seq.reshape(n_test, obs_steps * obs_dim)
    ).reshape(n_test, obs_steps, obs_dim)

    train_static_scaled = static_scaler.fit_transform(train_static).astype(np.float32)
    test_static_scaled = static_scaler.transform(test_static).astype(np.float32)
    return (
        train_seq_scaled.astype(np.float32),
        test_seq_scaled.astype(np.float32),
        train_static_scaled,
        test_static_scaled,
        seq_scaler,
        static_scaler,
    )


def build_mlp(tf, obs_steps, obs_dim, static_dim, horizon_steps, dropout):
    seq_input = tf.keras.Input(shape=(obs_steps, obs_dim), name="obs_sequence")
    static_input = tf.keras.Input(shape=(static_dim,), name="state_features")
    x_seq = tf.keras.layers.Flatten()(seq_input)
    x = tf.keras.layers.Concatenate()([x_seq, static_input])
    for units in [256, 192, 128]:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(horizon_steps * 96, activation="relu")(x)
    x = tf.keras.layers.Reshape((horizon_steps, 96))(x)
    x = tf.keras.layers.Conv1D(96, 3, padding="same", activation="relu")(x)
    output = tf.keras.layers.Conv1D(3, 1, activation="sigmoid", name="gain_chunk")(x)
    return tf.keras.Model([seq_input, static_input], output)


def build_cnn(tf, obs_steps, obs_dim, static_dim, horizon_steps, dropout):
    seq_input = tf.keras.Input(shape=(obs_steps, obs_dim), name="obs_sequence")
    static_input = tf.keras.Input(shape=(static_dim,), name="state_features")
    x = tf.keras.layers.Conv1D(64, 3, padding="causal", activation="relu")(seq_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(96, 3, padding="causal", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Concatenate()([x, static_input])
    x = tf.keras.layers.Dense(192, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(horizon_steps * 96, activation="relu")(x)
    x = tf.keras.layers.Reshape((horizon_steps, 96))(x)
    x = tf.keras.layers.Conv1D(96, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    output = tf.keras.layers.Conv1D(3, 1, activation="sigmoid", name="gain_chunk")(x)
    return tf.keras.Model([seq_input, static_input], output)


def residual_block(tf, x, filters, kernel_size, dropout):
    shortcut = x
    x = tf.keras.layers.Conv1D(filters, kernel_size, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv1D(filters, kernel_size, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv1D(filters, 1, padding="same")(shortcut)
    x = tf.keras.layers.Add()([x, shortcut])
    return tf.keras.layers.Activation("relu")(x)


def build_cnn_residual(tf, obs_steps, obs_dim, static_dim, horizon_steps, dropout):
    seq_input = tf.keras.Input(shape=(obs_steps, obs_dim), name="obs_sequence")
    static_input = tf.keras.Input(shape=(static_dim,), name="state_features")
    x = tf.keras.layers.Conv1D(64, 3, padding="causal", activation="relu")(seq_input)
    x = residual_block(tf, x, 64, 3, dropout)
    x = residual_block(tf, x, 96, 3, dropout)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Concatenate()([x, static_input])
    x = tf.keras.layers.Dense(224, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(horizon_steps * 96, activation="relu")(x)
    x = tf.keras.layers.Reshape((horizon_steps, 96))(x)
    x = residual_block(tf, x, 96, 3, dropout)
    x = residual_block(tf, x, 64, 3, dropout)
    output = tf.keras.layers.Conv1D(3, 1, activation="sigmoid", name="gain_chunk")(x)
    return tf.keras.Model([seq_input, static_input], output)


def build_cnn_attention(tf, obs_steps, obs_dim, static_dim, horizon_steps, dropout):
    seq_input = tf.keras.Input(shape=(obs_steps, obs_dim), name="obs_sequence")
    static_input = tf.keras.Input(shape=(static_dim,), name="state_features")
    x = tf.keras.layers.Conv1D(64, 3, padding="causal", activation="relu")(seq_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(96, 3, padding="causal", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=24, dropout=dropout)(
        x, x
    )
    x = tf.keras.layers.Add()([x, attn])
    x = tf.keras.layers.LayerNormalization()(x)
    ff = tf.keras.layers.Dense(128, activation="relu")(x)
    ff = tf.keras.layers.Dropout(dropout)(ff)
    ff = tf.keras.layers.Dense(96)(ff)
    x = tf.keras.layers.Add()([x, ff])
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Concatenate()([x, static_input])
    x = tf.keras.layers.Dense(224, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(horizon_steps * 96, activation="relu")(x)
    x = tf.keras.layers.Reshape((horizon_steps, 96))(x)
    attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=24, dropout=dropout)(
        x, x
    )
    x = tf.keras.layers.Add()([x, attn])
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    output = tf.keras.layers.Conv1D(3, 1, activation="sigmoid", name="gain_chunk")(x)
    return tf.keras.Model([seq_input, static_input], output)


def gain_smoothness(y):
    return np.mean(np.abs(np.diff(y, axis=1)), axis=(0, 1))


def evaluate_predictions(y_true, y_pred):
    rows = []
    true_flat = y_true.reshape((-1, len(GAIN_COLS)))
    pred_flat = y_pred.reshape((-1, len(GAIN_COLS)))
    for idx, col in enumerate(GAIN_COLS):
        rows.append(
            {
                "target": col,
                "mae": float(mean_absolute_error(true_flat[:, idx], pred_flat[:, idx])),
                "rmse": float(np.sqrt(mean_squared_error(true_flat[:, idx], pred_flat[:, idx]))),
                "r2": float(r2_score(true_flat[:, idx], pred_flat[:, idx])),
            }
        )
    rows.append(
        {
            "target": "all_gains",
            "mae": float(mean_absolute_error(true_flat, pred_flat)),
            "rmse": float(np.sqrt(mean_squared_error(true_flat, pred_flat))),
            "r2": float(r2_score(true_flat, pred_flat, multioutput="variance_weighted")),
        }
    )
    smooth_true = gain_smoothness(y_true)
    smooth_pred = gain_smoothness(y_pred)
    for idx, col in enumerate(GAIN_COLS):
        rows.append(
            {
                "target": f"{col}_mean_abs_step_delta",
                "mae": float(abs(smooth_true[idx] - smooth_pred[idx])),
                "rmse": np.nan,
                "r2": np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_training(history, path: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history.history["loss"], label="train")
    ax.plot(history.history["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Huber loss")
    ax.set_title("Gain Chunk Baseline Training")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tf = configure_tensorflow(args.seed)
    dataset_path = Path(args.dataset) if args.dataset else latest_label_path(args.profile)
    if not dataset_path.is_absolute():
        dataset_path = MOTOR_DIR / dataset_path

    print("=" * 80)
    print("Train diffusion gain chunk supervised baseline")
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPUs: {tf.config.list_physical_devices('GPU')}")
    print(f"Dataset: {dataset_path}")
    print(f"Model: {args.model_type}, quality={args.quality}")
    print("=" * 80)

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
    y_train_norm = normalize_gain_sequence(y_train)
    y_test_norm = normalize_gain_sequence(y_test)

    obs_steps, obs_dim = train_seq.shape[1:]
    horizon_steps = y_train.shape[1]
    if args.model_type == "mlp":
        model = build_mlp(tf, obs_steps, obs_dim, train_static.shape[1], horizon_steps, args.dropout)
    elif args.model_type == "cnn":
        model = build_cnn(tf, obs_steps, obs_dim, train_static.shape[1], horizon_steps, args.dropout)
    elif args.model_type == "cnn_residual":
        model = build_cnn_residual(
            tf, obs_steps, obs_dim, train_static.shape[1], horizon_steps, args.dropout
        )
    else:
        model = build_cnn_attention(
            tf, obs_steps, obs_dim, train_static.shape[1], horizon_steps, args.dropout
        )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=tf.keras.losses.Huber(),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        [train_seq, train_static],
        y_train_norm,
        validation_data=([test_seq, test_static], y_test_norm),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    pred_norm = model.predict([test_seq, test_static], batch_size=args.batch_size, verbose=0)
    y_pred = denormalize_gain_sequence(pred_norm)
    metrics = evaluate_predictions(y_test, y_pred)

    run_label = args.run_label or dataset_path.stem.replace("chunk_labels_", "")
    model_name = f"diffusion_gain_chunk_{args.model_type}_{run_label}_{timestamp}"
    keras_path = MODEL_DIR / f"{model_name}.keras"
    metadata_path = MODEL_DIR / f"{model_name}.joblib"
    metrics_path = SUMMARY_DIR / f"{model_name}_metrics.csv"
    history_path = SUMMARY_DIR / f"{model_name}_history.csv"
    figure_path = FIGURE_DIR / f"{model_name}_training.png"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    model.save(keras_path)
    pd.DataFrame(history.history).to_csv(history_path, index=False, encoding="utf-8-sig")
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    plot_training(history, figure_path)

    payload = {
        "model_type": f"supervised_gain_chunk_{args.model_type}",
        "keras_model_path": str(keras_path),
        "dataset_path": str(dataset_path),
        "label_profile": args.profile,
        "quality": args.quality,
        "obs_cols": DEFAULT_OBS_COLS,
        "static_feature_cols": static_feature_cols,
        "gain_cols": GAIN_COLS,
        "gain_bounds": GAIN_BOUNDS,
        "obs_steps": int(obs_steps),
        "horizon_steps": int(horizon_steps),
        "seq_scaler": seq_scaler,
        "static_scaler": static_scaler,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "metrics_path": str(metrics_path),
        "history_path": str(history_path),
        "figure_path": str(figure_path),
    }
    joblib.dump(payload, metadata_path)

    print(metrics.to_string(index=False))
    print(f"Saved model: {keras_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
