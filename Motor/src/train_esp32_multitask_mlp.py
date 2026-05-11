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
from train_esp32_horizon_cost_model import FEATURE_COLS, load_dataset, split_dataset


DATASET_PATH = PROCESSED_DATA_DIR / "esp32_horizon_cost_dataset_latest.csv"
SUMMARY_DIR = RESULTS_DIR / "summary"

TARGET_COLS = [
    "horizon_iae",
    "horizon_overshoot_ratio",
    "horizon_saturation_ratio",
    "horizon_near_saturation_ratio",
    "horizon_mean_pwm",
    "horizon_pwm_variation",
    "horizon_max_abs_error",
]

DEFAULT_SCORE_WEIGHTS = {
    "horizon_iae": 1.0,
    "horizon_overshoot_ratio": 20.0,
    "horizon_saturation_ratio": 5.0,
    "horizon_near_saturation_ratio": 2.0,
    "horizon_mean_pwm": 0.002,
    "horizon_pwm_variation": 0.001,
    "horizon_max_abs_error": 0.0,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train multi-task TensorFlow MLP for risk-aware ESP32 gain scheduling."
    )
    parser.add_argument("--dataset", type=str, default=str(DATASET_PATH))
    parser.add_argument("--epochs", type=int, default=260)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.02)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--random-split",
        action="store_true",
        help="Use row-wise random split instead of trajectory-group split.",
    )
    parser.add_argument(
        "--target-log1p",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train on log1p of non-negative target metrics.",
    )
    parser.add_argument(
        "--save-latest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also overwrite esp32_horizon_multitask_mlp_latest.*.",
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
    missing = [col for col in TARGET_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing multi-task target columns: {missing}")
    required = FEATURE_COLS + TARGET_COLS
    return df.dropna(subset=required).reset_index(drop=True)


def build_model(tf, input_dim: int, output_dim: int, learning_rate: float, dropout: float):
    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    x = tf.keras.layers.Dense(160, activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(160, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(96, activation="relu")(x)
    x = tf.keras.layers.Dense(48, activation="relu")(x)
    outputs = tf.keras.layers.Dense(output_dim, name="horizon_metrics")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.Huber(),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def transform_targets(y: pd.DataFrame, target_log1p: bool):
    values = y.to_numpy(dtype=np.float32)
    if target_log1p:
        values = np.log1p(np.maximum(values, 0.0))
    return values


def inverse_targets(pred: np.ndarray, target_log1p: bool):
    if target_log1p:
        pred = np.expm1(pred)
    return np.maximum(pred, 0.0)


def predict_metrics(model, x_scaler, X, target_log1p: bool, batch_size: int):
    X_scaled = x_scaler.transform(X).astype(np.float32)
    pred = model.predict(X_scaled, batch_size=batch_size, verbose=0)
    return inverse_targets(pred, target_log1p)


def evaluate(model, x_scaler, X_train, y_train, X_test, y_test, target_log1p, batch_size):
    train_pred = predict_metrics(model, x_scaler, X_train, target_log1p, batch_size)
    test_pred = predict_metrics(model, x_scaler, X_test, target_log1p, batch_size)
    train_true = y_train.to_numpy(dtype=float)
    test_true = y_test.to_numpy(dtype=float)

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

    return pd.DataFrame(rows), train_pred, test_pred


def score_from_metrics(metrics: np.ndarray):
    weights = np.array([DEFAULT_SCORE_WEIGHTS[col] for col in TARGET_COLS], dtype=float)
    return metrics @ weights


def save_model(model, x_scaler, metrics_df, args, dataset_path: Path, timestamp: str):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    keras_path = MODEL_DIR / f"esp32_horizon_multitask_mlp_{timestamp}.keras"
    latest_keras_path = MODEL_DIR / "esp32_horizon_multitask_mlp_latest.keras"
    metadata_path = MODEL_DIR / f"esp32_horizon_multitask_mlp_{timestamp}.joblib"
    latest_metadata_path = MODEL_DIR / "esp32_horizon_multitask_mlp_latest.joblib"

    model.save(keras_path)
    if args.save_latest:
        model.save(latest_keras_path)

    payload = {
        "keras_model_path": str(keras_path),
        "latest_keras_model_path": str(latest_keras_path),
        "scaler": x_scaler,
        "feature_cols": FEATURE_COLS,
        "target_cols": TARGET_COLS,
        "target_log1p": bool(args.target_log1p),
        "score_weights": DEFAULT_SCORE_WEIGHTS,
        "dataset_path": str(dataset_path),
        "metrics": metrics_df.to_dict(orient="records"),
        "model_type": "tensorflow_multitask_mlp",
    }
    joblib.dump(payload, metadata_path)
    if args.save_latest:
        latest_payload = dict(payload)
        latest_payload["keras_model_path"] = str(latest_keras_path)
        joblib.dump(latest_payload, latest_metadata_path)

    print(f"Saved model: {keras_path}")
    print(f"Saved metadata: {metadata_path}")
    if args.save_latest:
        print(f"Saved latest model: {latest_keras_path}")
        print(f"Saved latest metadata: {latest_metadata_path}")
    else:
        print("Skipped latest model update")


def save_outputs(metrics_df, history, y_test, test_pred, timestamp: str):
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = SUMMARY_DIR / f"esp32_horizon_multitask_mlp_metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"Saved metrics: {metrics_path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.history["loss"], label="train loss")
    ax.plot(history.history["val_loss"], label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber loss")
    ax.set_title("ESP32 Multi-task MLP Training")
    ax.grid(True, alpha=0.3)
    ax.legend()
    train_path = FIGURE_DIR / f"esp32_horizon_multitask_mlp_training_{timestamp}.png"
    fig.savefig(train_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {train_path}")

    true_score = score_from_metrics(y_test.to_numpy(dtype=float))
    pred_score = score_from_metrics(test_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_score, pred_score, alpha=0.35, s=12)
    min_value = min(float(np.min(true_score)), float(np.min(pred_score)))
    max_value = max(float(np.max(true_score)), float(np.max(pred_score)))
    ax.plot([min_value, max_value], [min_value, max_value], linestyle="--")
    ax.set_xlabel("True risk-aware score")
    ax.set_ylabel("Predicted risk-aware score")
    ax.set_title("Multi-task MLP Score Prediction")
    ax.grid(True, alpha=0.3)
    pred_path = FIGURE_DIR / f"esp32_horizon_multitask_mlp_score_prediction_{timestamp}.png"
    fig.savefig(pred_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {pred_path}")


def main():
    args = parse_args()
    tf = configure_tensorflow(seed=int(args.seed))
    dataset_path = Path(args.dataset)
    df = load_multitask_dataset(dataset_path)

    print("=" * 80)
    print("Train ESP32 horizon multi-task TensorFlow MLP")
    print("=" * 80)
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPUs: {tf.config.list_physical_devices('GPU')}")
    print(f"Dataset: {dataset_path}")
    print(f"Rows: {len(df)}")
    print(f"Features: {len(FEATURE_COLS)}")
    print(f"Targets: {TARGET_COLS}")
    if "source_kind" in df.columns:
        print("Source counts:")
        print(df["source_kind"].value_counts().to_string())
    print("=" * 80)

    X_train, X_test, _, _, train_df, test_df = split_dataset(
        df,
        test_size=float(args.test_size),
        random_split=bool(args.random_split),
    )
    y_train = train_df[TARGET_COLS]
    y_test = test_df[TARGET_COLS]

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train).astype(np.float32)
    y_train_fit = transform_targets(y_train, target_log1p=bool(args.target_log1p))

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train_scaled,
        y_train_fit,
        test_size=float(args.validation_split),
        random_state=int(args.seed),
        shuffle=True,
    )

    model = build_model(
        tf=tf,
        input_dim=len(FEATURE_COLS),
        output_dim=len(TARGET_COLS),
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
        X_fit,
        y_fit,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=2,
    )

    metrics_df, _, test_pred = evaluate(
        model,
        x_scaler,
        X_train,
        y_train,
        X_test,
        y_test,
        target_log1p=bool(args.target_log1p),
        batch_size=int(args.batch_size),
    )

    print("Metrics:")
    print(metrics_df.to_string(index=False))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_model(model, x_scaler, metrics_df, args, dataset_path, timestamp)
    save_outputs(metrics_df, history, y_test, test_pred, timestamp)


if __name__ == "__main__":
    main()
