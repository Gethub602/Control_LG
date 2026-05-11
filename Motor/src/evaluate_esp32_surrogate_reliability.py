import argparse
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))

from config import RESULTS_DIR, MODEL_DIR, PROCESSED_DATA_DIR
from train_esp32_horizon_cost_model import FEATURE_COLS
from train_esp32_multitask_mlp import TARGET_COLS, inverse_targets, score_from_metrics


SUMMARY_DIR = RESULTS_DIR / "summary"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ESP32 multi-task surrogate reliability on a shared dataset."
    )
    parser.add_argument(
        "--dataset",
        default=str(PROCESSED_DATA_DIR / "esp32_horizon_cost_dataset_latest.csv"),
        help="Horizon cost dataset CSV.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[str(MODEL_DIR / "esp32_horizon_multitask_mlp_latest.joblib")],
        help="One or more surrogate metadata .joblib paths.",
    )
    parser.add_argument(
        "--model-names",
        nargs="*",
        default=[],
        help="Optional display names for models.",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def configure_tensorflow():
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    return tf


def load_dataset(path: Path):
    df = pd.read_csv(path)
    required = FEATURE_COLS + TARGET_COLS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df.dropna(subset=required).reset_index(drop=True)


def load_model(tf, metadata_path: Path):
    payload = joblib.load(metadata_path)
    keras_path = Path(payload["keras_model_path"])
    model = tf.keras.models.load_model(keras_path)
    return payload, model


def predict(payload: dict, model, df: pd.DataFrame, batch_size: int):
    feature_cols = list(payload["feature_cols"])
    scaler = payload["scaler"]
    target_log1p = bool(payload.get("target_log1p", True))
    x = scaler.transform(df[feature_cols]).astype(np.float32)
    pred = model.predict(x, batch_size=batch_size, verbose=0)
    return inverse_targets(pred, target_log1p)


def metric_rows(model_name: str, df: pd.DataFrame, pred: np.ndarray, group_name: str, group_value: str):
    true = df[TARGET_COLS].to_numpy(dtype=float)
    rows = []
    for idx, target_col in enumerate(TARGET_COLS):
        y_true = true[:, idx]
        y_pred = pred[:, idx]
        rows.append(
            {
                "model": model_name,
                "group": group_name,
                "value": str(group_value),
                "target": target_col,
                "rows": int(len(df)),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "r2": float(r2_score(y_true, y_pred)) if len(df) > 1 else np.nan,
                "bias": float(np.mean(y_pred - y_true)),
                "p90_abs_error": float(np.percentile(np.abs(y_pred - y_true), 90)),
            }
        )

    true_score = score_from_metrics(true)
    pred_score = score_from_metrics(pred)
    rows.append(
        {
            "model": model_name,
            "group": group_name,
            "value": str(group_value),
            "target": "risk_aware_score",
            "rows": int(len(df)),
            "mae": float(mean_absolute_error(true_score, pred_score)),
            "rmse": float(np.sqrt(mean_squared_error(true_score, pred_score))),
            "r2": float(r2_score(true_score, pred_score)) if len(df) > 1 else np.nan,
            "bias": float(np.mean(pred_score - true_score)),
            "p90_abs_error": float(np.percentile(np.abs(pred_score - true_score), 90)),
        }
    )
    return rows


def add_target_bins(df: pd.DataFrame):
    out = df.copy()
    if "target" not in out.columns:
        out["target_bin"] = "unknown"
        return out
    bins = [-np.inf, 70, 85, 100, np.inf]
    labels = ["<=70", "70-85", "85-100", ">100"]
    out["target_bin"] = pd.cut(out["target"], bins=bins, labels=labels, include_lowest=True)
    out["target_bin"] = out["target_bin"].astype(str)
    return out


def main():
    args = parse_args()
    dataset_path = Path(args.dataset)
    model_paths = [Path(path) for path in args.models]
    model_names = list(args.model_names)
    while len(model_names) < len(model_paths):
        model_names.append(model_paths[len(model_names)].stem)

    tf = configure_tensorflow()
    df = add_target_bins(load_dataset(dataset_path))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("Evaluate ESP32 surrogate reliability")
    print(f"Dataset: {dataset_path}")
    print(f"Rows: {len(df)}")
    print(f"Models: {', '.join(model_names)}")
    print("=" * 80)

    rows = []
    for model_name, model_path in zip(model_names, model_paths):
        payload, model = load_model(tf, model_path)
        pred = predict(payload, model, df, int(args.batch_size))
        rows.extend(metric_rows(model_name, df, pred, "all", "all"))

        if "source_kind" in df.columns:
            for value, group_df in df.groupby("source_kind", sort=True):
                group_pred = pred[group_df.index.to_numpy()]
                rows.extend(metric_rows(model_name, group_df, group_pred, "source_kind", value))

        for value, group_df in df.groupby("target_bin", sort=True):
            group_pred = pred[group_df.index.to_numpy()]
            rows.extend(metric_rows(model_name, group_df, group_pred, "target_bin", value))

    reliability_df = pd.DataFrame(rows)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SUMMARY_DIR / f"esp32_surrogate_reliability_{timestamp}.csv"
    reliability_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved reliability: {output_path}")
    print("Overall metrics:")
    print(
        reliability_df[reliability_df["group"].eq("all")]
        .sort_values(["model", "target"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
