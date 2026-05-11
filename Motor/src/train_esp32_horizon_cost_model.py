import argparse
import sys
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split


CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))

from config import MODEL_DIR, FIGURE_DIR, PROCESSED_DATA_DIR, RESULTS_DIR


DATASET_PATH = PROCESSED_DATA_DIR / "esp32_horizon_cost_dataset_latest.csv"
SUMMARY_DIR = RESULTS_DIR / "summary"

TARGET_COL = "horizon_cost"

FEATURE_COLS = [
    "target",
    "current",
    "error",
    "error_derivative",
    "pwm",
    "prev_pwm",
    "kp",
    "ki",
    "kd",
    "integral",
    "kp_scale",
    "ki_scale",
    "time_since_start",
    "time_since_target_change",
    "error_ratio",
    "pwm_ratio",
    "current_lag_1",
    "error_lag_1",
    "pwm_lag_1",
    "current_lag_2",
    "error_lag_2",
    "pwm_lag_2",
    "current_lag_3",
    "error_lag_3",
    "pwm_lag_3",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RandomForest model for ESP32 horizon cost prediction."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DATASET_PATH),
        help="Path to horizon cost dataset CSV.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=400,
        help="Number of trees.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=0,
        help="Max tree depth. 0 means None.",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=2,
        help="Minimum samples per leaf.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel jobs for RandomForest. Use 1 on restricted Windows environments.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--random-split",
        action="store_true",
        help="Use row-wise random split instead of trajectory-group split.",
    )
    parser.add_argument(
        "--target-log1p",
        action="store_true",
        help="Train on log1p(horizon_cost) and invert predictions for metrics.",
    )
    return parser.parse_args()


def load_dataset(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    required_cols = FEATURE_COLS + [TARGET_COL]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=required_cols).reset_index(drop=True)

    return df


def split_dataset(df: pd.DataFrame, test_size: float, random_split: bool):
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    if random_split or "trajectory_id" not in df.columns:
        return train_test_split(
            X,
            y,
            df,
            test_size=test_size,
            random_state=42,
        )

    groups = (
        df["source_kind"].astype(str)
        + "::"
        + df["trajectory_id"].astype(str)
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=42,
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    return (
        X.iloc[train_idx],
        X.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[test_idx],
        df.iloc[train_idx],
        df.iloc[test_idx],
    )


def build_model(args):
    max_depth = None if args.max_depth == 0 else int(args.max_depth)

    return RandomForestRegressor(
        n_estimators=int(args.n_estimators),
        max_depth=max_depth,
        min_samples_leaf=int(args.min_samples_leaf),
        random_state=42,
        n_jobs=int(args.n_jobs),
    )


def predict_to_cost(model, X, target_log1p: bool):
    pred = model.predict(X)
    if target_log1p:
        pred = np.expm1(pred)
    return np.maximum(pred, 0.0)


def evaluate(model, X_train, y_train, X_test, y_test, target_log1p: bool):
    train_pred = predict_to_cost(model, X_train, target_log1p)
    test_pred = predict_to_cost(model, X_test, target_log1p)

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


def save_model(model, metrics: dict, args, dataset_path: Path, timestamp: str):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / f"esp32_horizon_cost_random_forest_{timestamp}.joblib"
    latest_path = MODEL_DIR / "esp32_horizon_cost_random_forest_latest.joblib"

    payload = {
        "model": model,
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "target_log1p": bool(args.target_log1p),
        "dataset_path": str(dataset_path),
        "metrics": {
            key: value
            for key, value in metrics.items()
            if not key.endswith("_pred")
        },
        "model_type": "random_forest",
    }

    joblib.dump(payload, model_path)
    joblib.dump(payload, latest_path)

    print(f"Saved model: {model_path}")
    print(f"Saved latest model: {latest_path}")

    return model_path, latest_path


def save_metrics(metrics: dict, train_df, test_df, timestamp: str):
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    metric_row = {
        key: value
        for key, value in metrics.items()
        if not key.endswith("_pred")
    }
    metric_row["train_rows"] = int(len(train_df))
    metric_row["test_rows"] = int(len(test_df))

    if "source_kind" in train_df.columns:
        metric_row["train_sources"] = train_df["source_kind"].value_counts().to_dict()
    if "source_kind" in test_df.columns:
        metric_row["test_sources"] = test_df["source_kind"].value_counts().to_dict()

    metrics_path = SUMMARY_DIR / f"esp32_horizon_cost_model_metrics_{timestamp}.csv"
    pd.DataFrame([metric_row]).to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"Saved metrics: {metrics_path}")

    return metrics_path


def save_feature_importance(model, timestamp: str):
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance_path = SUMMARY_DIR / f"esp32_horizon_cost_feature_importance_{timestamp}.csv"
    importance_df.to_csv(importance_path, index=False, encoding="utf-8-sig")
    print(f"Saved feature importance: {importance_path}")

    top_df = importance_df.head(20).sort_values("importance", ascending=True)

    plt.figure(figsize=(8, 6))
    plt.barh(top_df["feature"], top_df["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("ESP32 Horizon Cost RandomForest Feature Importance")
    plt.grid(True, axis="x")

    fig_path = FIGURE_DIR / f"esp32_horizon_cost_feature_importance_{timestamp}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {fig_path}")

    return importance_df, importance_path


def save_prediction_plot(y_test, y_pred, timestamp: str):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.35, s=12)

    min_value = min(float(np.min(y_test)), float(np.min(y_pred)))
    max_value = max(float(np.max(y_test)), float(np.max(y_pred)))
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")

    plt.xlabel("True horizon cost")
    plt.ylabel("Predicted horizon cost")
    plt.title("ESP32 Horizon Cost Prediction")
    plt.grid(True)

    fig_path = FIGURE_DIR / f"esp32_horizon_cost_prediction_{timestamp}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {fig_path}")

    return fig_path


def main():
    args = parse_args()
    dataset_path = Path(args.dataset)
    df = load_dataset(dataset_path)

    print("=" * 80)
    print("Train ESP32 horizon cost RandomForest")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"Rows: {len(df)}")
    print(f"Features: {len(FEATURE_COLS)}")
    print(f"Target: {TARGET_COL}")
    print("Source counts:")
    if "source_kind" in df.columns:
        print(df["source_kind"].value_counts().to_string())
    print("=" * 80)

    X_train, X_test, y_train, y_test, train_df, test_df = split_dataset(
        df,
        test_size=float(args.test_size),
        random_split=bool(args.random_split),
    )

    y_train_fit = np.log1p(y_train) if args.target_log1p else y_train

    model = build_model(args)
    model.fit(X_train, y_train_fit)

    metrics = evaluate(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        target_log1p=bool(args.target_log1p),
    )

    print("Metrics:")
    for key, value in metrics.items():
        if key.endswith("_pred"):
            continue
        print(f"{key}: {value:.6f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_model(model, metrics, args, dataset_path, timestamp)
    save_metrics(metrics, train_df, test_df, timestamp)
    importance_df, _ = save_feature_importance(model, timestamp)
    save_prediction_plot(y_test.to_numpy(dtype=float), metrics["test_pred"], timestamp)

    print("Top feature importance:")
    print(importance_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
