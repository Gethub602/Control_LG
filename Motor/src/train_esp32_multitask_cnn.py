import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))

from config import FIGURE_DIR, MODEL_DIR, PROCESSED_DATA_DIR, RESULTS_DIR
from train_esp32_horizon_cost_model import FEATURE_COLS, load_dataset, split_dataset
from train_esp32_multitask_mlp import (
    DEFAULT_SCORE_WEIGHTS,
    TARGET_COLS,
    inverse_targets,
    score_from_metrics,
    transform_targets,
)


DATASET_PATH = PROCESSED_DATA_DIR / "esp32_horizon_cost_dataset_latest.csv"
SUMMARY_DIR = RESULTS_DIR / "summary"

SEQUENCE_BASE_COLS = ["current", "error", "pwm"]
SEQUENCE_COLS = [
    ["current_lag_3", "error_lag_3", "pwm_lag_3"],
    ["current_lag_2", "error_lag_2", "pwm_lag_2"],
    ["current_lag_1", "error_lag_1", "pwm_lag_1"],
    ["current", "error", "pwm"],
]
SEQUENCE_FEATURE_COLS = [col for step_cols in SEQUENCE_COLS for col in step_cols]
STATIC_FEATURE_COLS = [
    col for col in FEATURE_COLS if col not in set(SEQUENCE_FEATURE_COLS)
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a 1D CNN multi-task surrogate for ESP32 horizon metrics."
    )
    parser.add_argument("--dataset", type=str, default=str(DATASET_PATH))
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.03)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-split", action="store_true")
    parser.add_argument(
        "--target-log1p",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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


def load_multitask_dataset(path: Path):
    df = load_dataset(path)
    required = FEATURE_COLS + TARGET_COLS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df.dropna(subset=required).reset_index(drop=True)


def make_inputs(df: pd.DataFrame, seq_scaler=None, static_scaler=None, fit=False):
    seq_flat = df[SEQUENCE_FEATURE_COLS].to_numpy(dtype=np.float32)
    static = df[STATIC_FEATURE_COLS].to_numpy(dtype=np.float32)

    if fit:
        seq_scaler = StandardScaler()
        static_scaler = StandardScaler()
        seq_flat = seq_scaler.fit_transform(seq_flat).astype(np.float32)
        static = static_scaler.fit_transform(static).astype(np.float32)
    else:
        seq_flat = seq_scaler.transform(seq_flat).astype(np.float32)
        static = static_scaler.transform(static).astype(np.float32)

    seq = seq_flat.reshape((-1, len(SEQUENCE_COLS), len(SEQUENCE_BASE_COLS)))
    return seq, static, seq_scaler, static_scaler


def build_model(tf, learning_rate: float, dropout: float):
    seq_input = tf.keras.Input(
        shape=(len(SEQUENCE_COLS), len(SEQUENCE_BASE_COLS)),
        name="state_sequence",
    )
    static_input = tf.keras.Input(shape=(len(STATIC_FEATURE_COLS),), name="static_features")

    x = tf.keras.layers.Conv1D(48, kernel_size=2, padding="causal", activation="relu")(
        seq_input
    )
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(64, kernel_size=2, padding="causal", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Concatenate()([x, static_input])
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(96, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(48, activation="relu")(x)
    outputs = tf.keras.layers.Dense(len(TARGET_COLS), name="horizon_metrics")(x)

    model = tf.keras.Model(inputs=[seq_input, static_input], outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.Huber(),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def predict_metrics(model, seq, static, target_log1p: bool, batch_size: int):
    pred = model.predict([seq, static], batch_size=batch_size, verbose=0)
    return inverse_targets(pred, target_log1p)


def evaluate(model, train_df, test_df, seq_scaler, static_scaler, target_log1p, batch_size):
    train_seq, train_static, _, _ = make_inputs(
        train_df, seq_scaler=seq_scaler, static_scaler=static_scaler
    )
    test_seq, test_static, _, _ = make_inputs(
        test_df, seq_scaler=seq_scaler, static_scaler=static_scaler
    )
    train_pred = predict_metrics(model, train_seq, train_static, target_log1p, batch_size)
    test_pred = predict_metrics(model, test_seq, test_static, target_log1p, batch_size)
    train_true = train_df[TARGET_COLS].to_numpy(dtype=float)
    test_true = test_df[TARGET_COLS].to_numpy(dtype=float)

    rows = []
    for idx, col in enumerate(TARGET_COLS):
        rows.append(
            {
                "target": col,
                "train_mae": float(mean_absolute_error(train_true[:, idx], train_pred[:, idx])),
                "test_mae": float(mean_absolute_error(test_true[:, idx], test_pred[:, idx])),
                "train_rmse": float(np.sqrt(mean_squared_error(train_true[:, idx], train_pred[:, idx]))),
                "test_rmse": float(np.sqrt(mean_squared_error(test_true[:, idx], test_pred[:, idx]))),
                "train_r2": float(r2_score(train_true[:, idx], train_pred[:, idx])),
                "test_r2": float(r2_score(test_true[:, idx], test_pred[:, idx])),
            }
        )

    true_score = score_from_metrics(test_true)
    pred_score = score_from_metrics(test_pred)
    rows.append(
        {
            "target": "risk_aware_score",
            "train_mae": np.nan,
            "test_mae": float(mean_absolute_error(true_score, pred_score)),
            "train_rmse": np.nan,
            "test_rmse": float(np.sqrt(mean_squared_error(true_score, pred_score))),
            "train_r2": np.nan,
            "test_r2": float(r2_score(true_score, pred_score)),
        }
    )
    return pd.DataFrame(rows), test_pred


def benchmark_latency(model, seq, static):
    sample_seq = seq[:1]
    sample_static = static[:1]
    batch32_seq = seq[:32]
    batch32_static = static[:32]
    batch1024_seq = seq[:1024]
    batch1024_static = static[:1024]

    for _ in range(10):
        model.predict([sample_seq, sample_static], verbose=0)

    rows = []
    for name, x_seq, x_static, repeats in [
        ("batch_1", sample_seq, sample_static, 200),
        ("batch_32", batch32_seq, batch32_static, 80),
        ("batch_1024", batch1024_seq, batch1024_static, 30),
    ]:
        durations = []
        for _ in range(repeats):
            start = time.perf_counter()
            model.predict([x_seq, x_static], verbose=0)
            durations.append(time.perf_counter() - start)
        arr = np.array(durations, dtype=float)
        rows.append(
            {
                "batch": name,
                "samples": int(len(x_seq)),
                "mean_sec": float(np.mean(arr)),
                "p50_sec": float(np.percentile(arr, 50)),
                "p90_sec": float(np.percentile(arr, 90)),
                "max_sec": float(np.max(arr)),
                "mean_ms_per_sample": float(np.mean(arr) / max(len(x_seq), 1) * 1000.0),
            }
        )
    return pd.DataFrame(rows)


def save_outputs(model, seq_scaler, static_scaler, metrics_df, latency_df, history, args, dataset_path, timestamp, y_test, test_pred):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    keras_path = MODEL_DIR / f"esp32_horizon_multitask_cnn_{timestamp}.keras"
    metadata_path = MODEL_DIR / f"esp32_horizon_multitask_cnn_{timestamp}.joblib"
    metrics_path = SUMMARY_DIR / f"esp32_horizon_multitask_cnn_metrics_{timestamp}.csv"
    latency_path = SUMMARY_DIR / f"esp32_horizon_multitask_cnn_latency_{timestamp}.csv"

    model.save(keras_path)
    payload = {
        "keras_model_path": str(keras_path),
        "seq_scaler": seq_scaler,
        "static_scaler": static_scaler,
        "feature_cols": FEATURE_COLS,
        "sequence_feature_cols": SEQUENCE_FEATURE_COLS,
        "static_feature_cols": STATIC_FEATURE_COLS,
        "sequence_cols": SEQUENCE_COLS,
        "target_cols": TARGET_COLS,
        "target_log1p": bool(args.target_log1p),
        "score_weights": DEFAULT_SCORE_WEIGHTS,
        "dataset_path": str(dataset_path),
        "metrics": metrics_df.to_dict(orient="records"),
        "latency": latency_df.to_dict(orient="records"),
        "model_type": "tensorflow_multitask_1d_cnn",
    }
    joblib.dump(payload, metadata_path)
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    latency_df.to_csv(latency_path, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.history["loss"], label="train loss")
    ax.plot(history.history["val_loss"], label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber loss")
    ax.set_title("ESP32 Multi-task 1D CNN Training")
    ax.grid(True, alpha=0.3)
    ax.legend()
    train_path = FIGURE_DIR / f"esp32_horizon_multitask_cnn_training_{timestamp}.png"
    fig.savefig(train_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    true_score = score_from_metrics(y_test.to_numpy(dtype=float))
    pred_score = score_from_metrics(test_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_score, pred_score, alpha=0.35, s=12)
    min_value = min(float(np.min(true_score)), float(np.min(pred_score)))
    max_value = max(float(np.max(true_score)), float(np.max(pred_score)))
    ax.plot([min_value, max_value], [min_value, max_value], linestyle="--")
    ax.set_xlabel("True risk-aware score")
    ax.set_ylabel("Predicted risk-aware score")
    ax.set_title("1D CNN Score Prediction")
    ax.grid(True, alpha=0.3)
    pred_path = FIGURE_DIR / f"esp32_horizon_multitask_cnn_score_prediction_{timestamp}.png"
    fig.savefig(pred_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved model: {keras_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved latency: {latency_path}")
    print(f"Saved figure: {train_path}")
    print(f"Saved figure: {pred_path}")


def main():
    args = parse_args()
    tf = configure_tensorflow(seed=int(args.seed))
    dataset_path = Path(args.dataset)
    df = load_multitask_dataset(dataset_path)

    print("=" * 80)
    print("Train ESP32 horizon multi-task 1D CNN")
    print("=" * 80)
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPUs: {tf.config.list_physical_devices('GPU')}")
    print(f"Dataset: {dataset_path}")
    print(f"Rows: {len(df)}")
    print(f"Sequence shape: ({len(SEQUENCE_COLS)}, {len(SEQUENCE_BASE_COLS)})")
    print(f"Static features: {len(STATIC_FEATURE_COLS)}")
    print("=" * 80)

    _, _, _, _, train_df, test_df = split_dataset(
        df,
        test_size=float(args.test_size),
        random_split=bool(args.random_split),
    )
    y_train = train_df[TARGET_COLS]
    y_train_fit = transform_targets(y_train, target_log1p=bool(args.target_log1p))
    train_seq, train_static, seq_scaler, static_scaler = make_inputs(train_df, fit=True)

    indices = np.arange(len(train_df))
    fit_idx, val_idx = train_test_split(
        indices,
        test_size=float(args.validation_split),
        random_state=int(args.seed),
        shuffle=True,
    )

    model = build_model(
        tf=tf,
        learning_rate=float(args.learning_rate),
        dropout=float(args.dropout),
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(args.patience),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(5, int(args.patience) // 3),
            min_lr=1e-5,
        ),
    ]
    history = model.fit(
        [train_seq[fit_idx], train_static[fit_idx]],
        y_train_fit[fit_idx],
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        validation_data=(
            [train_seq[val_idx], train_static[val_idx]],
            y_train_fit[val_idx],
        ),
        callbacks=callbacks,
        verbose=2,
    )

    metrics_df, test_pred = evaluate(
        model,
        train_df,
        test_df,
        seq_scaler,
        static_scaler,
        bool(args.target_log1p),
        int(args.batch_size),
    )
    test_seq, test_static, _, _ = make_inputs(
        test_df, seq_scaler=seq_scaler, static_scaler=static_scaler
    )
    latency_df = benchmark_latency(model, test_seq, test_static)

    print("Metrics:")
    print(metrics_df.to_string(index=False))
    print("Latency:")
    print(latency_df.to_string(index=False))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_outputs(
        model,
        seq_scaler,
        static_scaler,
        metrics_df,
        latency_df,
        history,
        args,
        dataset_path,
        timestamp,
        test_df[TARGET_COLS],
        test_pred,
    )


if __name__ == "__main__":
    main()
