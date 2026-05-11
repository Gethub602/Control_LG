import argparse
import sys
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
from train_esp32_horizon_cost_model import (
    FEATURE_COLS,
    TARGET_COL,
    load_dataset,
    split_dataset,
)


DATASET_PATH = PROCESSED_DATA_DIR / "esp32_horizon_cost_dataset_latest.csv"
SUMMARY_DIR = RESULTS_DIR / "summary"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train TensorFlow MLP model for ESP32 horizon cost prediction."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DATASET_PATH),
        help="Path to horizon cost dataset CSV.",
    )
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--random-split",
        action="store_true",
        help="Use row-wise random split instead of trajectory-group split.",
    )
    parser.add_argument(
        "--target-log1p",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train on log1p(horizon_cost) and invert predictions for metrics.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.05,
        help="Dropout rate after hidden layers.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Early stopping patience.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
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


def build_model(tf, input_dim: int, learning_rate: float, dropout: float):
    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    x = tf.keras.layers.Dense(128, activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, name="horizon_cost")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.Huber(),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def predict_to_cost(model, scaler: StandardScaler, X, target_log1p: bool, batch_size: int):
    X_scaled = scaler.transform(X).astype(np.float32)
    pred = model.predict(X_scaled, batch_size=batch_size, verbose=0).reshape(-1)
    if target_log1p:
        pred = np.expm1(pred)
    return np.maximum(pred, 0.0)


def evaluate(model, scaler, X_train, y_train, X_test, y_test, target_log1p, batch_size):
    train_pred = predict_to_cost(model, scaler, X_train, target_log1p, batch_size)
    test_pred = predict_to_cost(model, scaler, X_test, target_log1p, batch_size)

    return {
        "train_mae": float(mean_absolute_error(y_train, train_pred)),
        "test_mae": float(mean_absolute_error(y_test, test_pred)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_pred))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
        "train_r2": float(r2_score(y_train, train_pred)),
        "test_r2": float(r2_score(y_test, test_pred)),
        "train_pred": train_pred,
        "test_pred": test_pred,
    }


def save_model(model, scaler, metrics, args, dataset_path: Path, timestamp: str):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    keras_path = MODEL_DIR / f"esp32_horizon_cost_mlp_{timestamp}.keras"
    latest_keras_path = MODEL_DIR / "esp32_horizon_cost_mlp_latest.keras"
    metadata_path = MODEL_DIR / f"esp32_horizon_cost_mlp_{timestamp}.joblib"
    latest_metadata_path = MODEL_DIR / "esp32_horizon_cost_mlp_latest.joblib"

    model.save(keras_path)
    model.save(latest_keras_path)

    payload = {
        "keras_model_path": str(keras_path),
        "latest_keras_model_path": str(latest_keras_path),
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "target_log1p": bool(args.target_log1p),
        "dataset_path": str(dataset_path),
        "metrics": {
            key: value
            for key, value in metrics.items()
            if not key.endswith("_pred")
        },
        "model_type": "tensorflow_mlp",
    }

    latest_payload = dict(payload)
    latest_payload["keras_model_path"] = str(latest_keras_path)

    joblib.dump(payload, metadata_path)
    joblib.dump(latest_payload, latest_metadata_path)

    print(f"Saved model: {keras_path}")
    print(f"Saved latest model: {latest_keras_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved latest metadata: {latest_metadata_path}")

    return keras_path, latest_keras_path, metadata_path, latest_metadata_path


def save_metrics(metrics, history, train_df, test_df, timestamp: str):
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    metric_row = {
        key: value
        for key, value in metrics.items()
        if not key.endswith("_pred")
    }
    metric_row["train_rows"] = int(len(train_df))
    metric_row["test_rows"] = int(len(test_df))
    metric_row["best_epoch"] = int(np.argmin(history.history["val_loss"]) + 1)
    metric_row["final_val_loss"] = float(history.history["val_loss"][-1])

    if "source_kind" in train_df.columns:
        metric_row["train_sources"] = train_df["source_kind"].value_counts().to_dict()
    if "source_kind" in test_df.columns:
        metric_row["test_sources"] = test_df["source_kind"].value_counts().to_dict()

    metrics_path = SUMMARY_DIR / f"esp32_horizon_cost_mlp_metrics_{timestamp}.csv"
    pd.DataFrame([metric_row]).to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"Saved metrics: {metrics_path}")
    return metrics_path


def save_training_plot(history, timestamp: str):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.history["loss"], label="train loss")
    ax.plot(history.history["val_loss"], label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber loss")
    ax.set_title("ESP32 Horizon Cost MLP Training")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig_path = FIGURE_DIR / f"esp32_horizon_cost_mlp_training_{timestamp}.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {fig_path}")
    return fig_path


def save_prediction_plot(y_test, y_pred, timestamp: str):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.35, s=12)
    min_value = min(float(np.min(y_test)), float(np.min(y_pred)))
    max_value = max(float(np.max(y_test)), float(np.max(y_pred)))
    ax.plot([min_value, max_value], [min_value, max_value], linestyle="--")
    ax.set_xlabel("True horizon cost")
    ax.set_ylabel("Predicted horizon cost")
    ax.set_title("ESP32 Horizon Cost MLP Prediction")
    ax.grid(True, alpha=0.3)

    fig_path = FIGURE_DIR / f"esp32_horizon_cost_mlp_prediction_{timestamp}.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {fig_path}")
    return fig_path


def main():
    args = parse_args()
    tf = configure_tensorflow(seed=int(args.seed))

    dataset_path = Path(args.dataset)
    df = load_dataset(dataset_path)

    print("=" * 80)
    print("Train ESP32 horizon cost TensorFlow MLP")
    print("=" * 80)
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPUs: {tf.config.list_physical_devices('GPU')}")
    print(f"Dataset: {dataset_path}")
    print(f"Rows: {len(df)}")
    print(f"Features: {len(FEATURE_COLS)}")
    print(f"Target: {TARGET_COL}")
    if "source_kind" in df.columns:
        print("Source counts:")
        print(df["source_kind"].value_counts().to_string())
    print("=" * 80)

    X_train, X_test, y_train, y_test, train_df, test_df = split_dataset(
        df,
        test_size=float(args.test_size),
        random_split=bool(args.random_split),
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    y_train_fit = y_train.to_numpy(dtype=np.float32)
    if args.target_log1p:
        y_train_fit = np.log1p(y_train_fit)

    model = build_model(
        tf=tf,
        input_dim=len(FEATURE_COLS),
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

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train_scaled,
        y_train_fit,
        test_size=float(args.validation_split),
        random_state=int(args.seed),
        shuffle=True,
    )

    history = model.fit(
        X_fit,
        y_fit,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=2,
    )

    metrics = evaluate(
        model,
        scaler,
        X_train,
        y_train,
        X_test,
        y_test,
        target_log1p=bool(args.target_log1p),
        batch_size=int(args.batch_size),
    )

    print("Metrics:")
    for key, value in metrics.items():
        if key.endswith("_pred"):
            continue
        print(f"{key}: {value:.6f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_model(model, scaler, metrics, args, dataset_path, timestamp)
    save_metrics(metrics, history, train_df, test_df, timestamp)
    save_training_plot(history, timestamp)
    save_prediction_plot(y_test.to_numpy(dtype=float), metrics["test_pred"], timestamp)


if __name__ == "__main__":
    main()
