import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

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

from config import (
    ESP32_SWEEP_KD_LIST,
    ESP32_SWEEP_KI_LIST,
    ESP32_SWEEP_KP_LIST,
    ESP32_REAL_PID_GAIN_DB,
    FIGURE_DIR,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
)
from train_esp32_horizon_cost_model import FEATURE_COLS, load_dataset
from train_esp32_multitask_mlp import DEFAULT_SCORE_WEIGHTS, TARGET_COLS


DATASET_PATH = PROCESSED_DATA_DIR / "esp32_horizon_cost_dataset_latest.csv"
SUMMARY_DIR = RESULTS_DIR / "summary"
OUTPUT_COLS = ["kp", "ki", "kd"]
DERIVED_POLICY_FEATURE_COLS = [
    "abs_error",
    "signed_error_ratio",
    "accel_demand",
    "decel_demand",
    "speed_ratio",
    "abs_error_derivative",
]
TRANSITION_CONTEXT_FEATURE_COLS = [
    "previous_target",
    "target_delta",
    "abs_target_delta",
    "target_direction",
    "target_change_count",
]
DB_REFERENCE_FEATURE_COLS = [
    "target_db_nearest",
    "target_db_lower",
    "target_db_upper",
    "target_db_nearest_distance",
    "target_db_interval_width",
    "target_db_alpha",
    "target_db_is_interpolation",
]
DB_PRIOR_FEATURE_COLS = [
    "db_base_kp",
    "db_base_ki",
    "db_base_kd",
]
BASE_POLICY_FEATURE_COLS = [
    col for col in FEATURE_COLS if col not in OUTPUT_COLS
] + DERIVED_POLICY_FEATURE_COLS + TRANSITION_CONTEXT_FEATURE_COLS
POLICY_FEATURE_COLS = BASE_POLICY_FEATURE_COLS
PARETO_OBJECTIVE_COLS = [
    "horizon_iae",
    "horizon_overshoot_ratio",
    "horizon_near_saturation_ratio",
    "horizon_mean_pwm",
    "horizon_pwm_variation",
]
DB_REFERENCE_TARGETS = sorted(float(target) for target in ESP32_REAL_PID_GAIN_DB.keys())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a direct state -> PID gain TensorFlow MLP policy."
    )
    parser.add_argument("--dataset", type=str, default=str(DATASET_PATH))
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.03)
    parser.add_argument(
        "--hidden-layers",
        type=str,
        default="128,96,48",
        help="Comma-separated hidden layer widths for the direct policy MLP.",
    )
    parser.add_argument(
        "--output-mode",
        choices=["absolute", "residual_db"],
        default="absolute",
        help="Predict absolute gains or residuals around ESP32_REAL_PID_GAIN_DB interpolation.",
    )
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--label-selection",
        choices=[
            "weighted_score",
            "pareto_utopia",
            "iae_pwm_tiebreak",
            "rank_weighted",
        ],
        default="weighted_score",
        help="Pseudo-label selection strategy inside each context bin.",
    )
    parser.add_argument("--mean-pwm-weight", type=float, default=0.002)
    parser.add_argument("--pwm-variation-weight", type=float, default=0.001)
    parser.add_argument("--near-saturation-weight", type=float, default=2.0)
    parser.add_argument("--saturation-weight", type=float, default=5.0)
    parser.add_argument("--pareto-performance-weight", type=float, default=0.72)
    parser.add_argument("--pareto-pwm-weight", type=float, default=0.16)
    parser.add_argument("--pareto-variation-weight", type=float, default=0.07)
    parser.add_argument("--pareto-risk-weight", type=float, default=0.05)
    parser.add_argument(
        "--iae-tolerance-abs",
        type=float,
        default=0.05,
        help="Absolute IAE tolerance for the IAE-first PWM tiebreak label selection.",
    )
    parser.add_argument(
        "--iae-tolerance-ratio",
        type=float,
        default=0.08,
        help="Relative IAE tolerance for the IAE-first PWM tiebreak label selection.",
    )
    parser.add_argument("--tiebreak-pwm-weight", type=float, default=1.0)
    parser.add_argument("--tiebreak-variation-weight", type=float, default=0.25)
    parser.add_argument("--tiebreak-risk-weight", type=float, default=0.5)
    parser.add_argument(
        "--rank-top-k",
        type=int,
        default=3,
        help="Number of ranked candidates to keep per context for rank_weighted labels.",
    )
    parser.add_argument(
        "--rank-temperature",
        type=float,
        default=0.35,
        help="Softmax temperature for rank_weighted labels in normalized context score units.",
    )
    parser.add_argument(
        "--rank-score-col",
        choices=["horizon_iae", "policy_score"],
        default="horizon_iae",
        help="Objective used to rank candidates inside each context group.",
    )
    parser.add_argument(
        "--use-db-reference-features",
        action="store_true",
        help="Add target distance/interpolation features derived from ESP32_REAL_PID_GAIN_DB.",
    )
    parser.add_argument(
        "--use-db-prior-features",
        action="store_true",
        help="Add interpolated ESP32_REAL_PID_GAIN_DB gains as model input features.",
    )
    parser.add_argument(
        "--no-transition-context-features",
        dest="use_transition_context_features",
        action="store_false",
        help="Disable previous-target and target-delta features for clean-policy ablations.",
    )
    parser.set_defaults(use_transition_context_features=True)
    parser.add_argument(
        "--max-labels",
        type=int,
        default=30000,
        help="Maximum pseudo-label rows after context binning. 0 keeps all.",
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


def gain_bounds():
    return {
        "kp": (float(min(ESP32_SWEEP_KP_LIST)), float(max(ESP32_SWEEP_KP_LIST))),
        "ki": (float(min(ESP32_SWEEP_KI_LIST)), float(max(ESP32_SWEEP_KI_LIST))),
        "kd": (float(min(ESP32_SWEEP_KD_LIST)), float(max(ESP32_SWEEP_KD_LIST))),
    }


def add_direct_policy_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    add_transition_context_features(data)
    target_abs = data["target"].abs().clip(lower=1e-6)
    data["abs_error"] = data["error"].abs()
    data["signed_error_ratio"] = data["error"] / target_abs
    data["accel_demand"] = (data["error"] > 0.0).astype(float)
    data["decel_demand"] = (data["error"] < 0.0).astype(float)
    data["speed_ratio"] = data["current"] / target_abs
    data["abs_error_derivative"] = data["error_derivative"].abs()
    add_db_reference_features(data)
    add_db_prior_gain_features(data)
    return data


def add_transition_context_features(data: pd.DataFrame):
    required_cols = [
        "previous_target",
        "target_delta",
        "abs_target_delta",
        "target_direction",
        "target_change_count",
    ]
    if all(col in data.columns for col in required_cols):
        return

    for col in required_cols:
        data[col] = 0.0

    if "trajectory_id" not in data.columns:
        data["previous_target"] = data["target"]
        return

    sort_cols = ["trajectory_id"]
    if "step" in data.columns:
        sort_cols.append("step")
    elif "time" in data.columns:
        sort_cols.append("time")

    for _, group in data.sort_values(sort_cols).groupby("trajectory_id", sort=False):
        targets = group["target"].to_numpy(dtype=float)
        idx_values = group.index.to_numpy()
        if len(targets) == 0:
            continue

        previous_segment_target = float(targets[0])
        current_segment_target = float(targets[0])
        target_change_count = 0

        for local_idx, global_idx in enumerate(idx_values):
            target = float(targets[local_idx])
            if abs(target - current_segment_target) > 1e-9:
                previous_segment_target = current_segment_target
                current_segment_target = target
                target_change_count += 1

            delta = current_segment_target - previous_segment_target
            data.at[global_idx, "previous_target"] = previous_segment_target
            data.at[global_idx, "target_delta"] = delta
            data.at[global_idx, "abs_target_delta"] = abs(delta)
            data.at[global_idx, "target_direction"] = float(np.sign(delta))
            data.at[global_idx, "target_change_count"] = float(target_change_count)


def get_policy_feature_cols(args=None):
    cols = [col for col in FEATURE_COLS if col not in OUTPUT_COLS]
    cols += DERIVED_POLICY_FEATURE_COLS
    if args is None or getattr(args, "use_transition_context_features", True):
        cols += TRANSITION_CONTEXT_FEATURE_COLS
    if args is not None and getattr(args, "use_db_reference_features", False):
        cols += DB_REFERENCE_FEATURE_COLS
    if args is not None and (
        getattr(args, "use_db_prior_features", False)
        or getattr(args, "output_mode", "absolute") == "residual_db"
    ):
        cols += DB_PRIOR_FEATURE_COLS
    return cols


def interpolate_db_gain(target: float, db_targets=None) -> dict:
    targets = list(DB_REFERENCE_TARGETS if db_targets is None else db_targets)
    if not targets:
        return {"kp": 0.0, "ki": 0.0, "kd": 0.0}

    target = float(target)
    if target in targets:
        gains = ESP32_REAL_PID_GAIN_DB[float(target)]
        return {key: float(gains[key]) for key in OUTPUT_COLS}

    if target <= targets[0]:
        gains = ESP32_REAL_PID_GAIN_DB[float(targets[0])]
        return {key: float(gains[key]) for key in OUTPUT_COLS}
    if target >= targets[-1]:
        gains = ESP32_REAL_PID_GAIN_DB[float(targets[-1])]
        return {key: float(gains[key]) for key in OUTPUT_COLS}

    for lower, upper in zip(targets[:-1], targets[1:]):
        if lower <= target <= upper:
            ratio = (target - lower) / max(upper - lower, 1e-9)
            lower_gain = ESP32_REAL_PID_GAIN_DB[float(lower)]
            upper_gain = ESP32_REAL_PID_GAIN_DB[float(upper)]
            return {
                key: float(lower_gain[key])
                + ratio * (float(upper_gain[key]) - float(lower_gain[key]))
                for key in OUTPUT_COLS
            }

    nearest = min(targets, key=lambda value: abs(value - target))
    gains = ESP32_REAL_PID_GAIN_DB[float(nearest)]
    return {key: float(gains[key]) for key in OUTPUT_COLS}


def add_db_prior_gain_features(data: pd.DataFrame):
    for key in DB_PRIOR_FEATURE_COLS:
        data[key] = 0.0
    if "target" not in data.columns:
        return
    for idx, target in data["target"].items():
        gains = interpolate_db_gain(float(target))
        data.at[idx, "db_base_kp"] = gains["kp"]
        data.at[idx, "db_base_ki"] = gains["ki"]
        data.at[idx, "db_base_kd"] = gains["kd"]


def db_reference_values(target: float, db_targets=None) -> dict:
    targets = list(DB_REFERENCE_TARGETS if db_targets is None else db_targets)
    if not targets:
        return {
            "target_db_nearest": 0.0,
            "target_db_lower": 0.0,
            "target_db_upper": 0.0,
            "target_db_nearest_distance": 0.0,
            "target_db_interval_width": 0.0,
            "target_db_alpha": 0.0,
            "target_db_is_interpolation": 0.0,
        }

    nearest = min(targets, key=lambda value: abs(value - target))
    lower_candidates = [value for value in targets if value <= target]
    upper_candidates = [value for value in targets if value >= target]
    lower = max(lower_candidates) if lower_candidates else targets[0]
    upper = min(upper_candidates) if upper_candidates else targets[-1]
    width = max(upper - lower, 0.0)
    alpha = 0.0 if width <= 1e-9 else (target - lower) / width
    is_interpolation = float(width > 1e-9 and lower < target < upper)

    return {
        "target_db_nearest": float(nearest),
        "target_db_lower": float(lower),
        "target_db_upper": float(upper),
        "target_db_nearest_distance": float(abs(target - nearest)),
        "target_db_interval_width": float(width),
        "target_db_alpha": float(np.clip(alpha, 0.0, 1.0)),
        "target_db_is_interpolation": is_interpolation,
    }


def add_db_reference_features(data: pd.DataFrame):
    values = data["target"].apply(lambda target: db_reference_values(float(target)))
    for key in DB_REFERENCE_FEATURE_COLS:
        data[key] = values.apply(lambda item: item[key])


def score_rows(df: pd.DataFrame, args) -> pd.Series:
    weights = dict(DEFAULT_SCORE_WEIGHTS)
    weights["horizon_mean_pwm"] = float(args.mean_pwm_weight)
    weights["horizon_pwm_variation"] = float(args.pwm_variation_weight)
    weights["horizon_near_saturation_ratio"] = float(args.near_saturation_weight)
    weights["horizon_saturation_ratio"] = float(args.saturation_weight)

    score = np.zeros(len(df), dtype=float)
    for col in TARGET_COLS:
        score += float(weights[col]) * df[col].to_numpy(dtype=float)
    return pd.Series(score, index=df.index)


def pareto_utopia_choice(group: pd.DataFrame, args) -> int:
    if len(group) == 1:
        return int(group.index[0])

    objectives = group[PARETO_OBJECTIVE_COLS].to_numpy(dtype=float)
    mins = objectives.min(axis=0)
    spans = np.maximum(objectives.max(axis=0) - mins, 1e-9)
    norm = (objectives - mins) / spans

    dominated = np.zeros(len(group), dtype=bool)
    for idx in range(len(group)):
        if dominated[idx]:
            continue
        other = np.delete(norm, idx, axis=0)
        if len(other) == 0:
            continue
        dominated[idx] = np.any(
            np.all(other <= norm[idx], axis=1)
            & np.any(other < norm[idx], axis=1)
        )

    pareto_positions = np.where(~dominated)[0]
    pareto_norm = norm[pareto_positions]
    weights = np.array(
        [
            float(args.pareto_performance_weight),
            float(args.pareto_risk_weight),
            float(args.pareto_risk_weight),
            float(args.pareto_pwm_weight),
            float(args.pareto_variation_weight),
        ],
        dtype=float,
    )
    weights = weights / max(weights.sum(), 1e-9)
    distance = np.sqrt(np.sum(weights * pareto_norm * pareto_norm, axis=1))
    best_position = int(pareto_positions[int(np.argmin(distance))])
    return int(group.index[best_position])


def iae_pwm_tiebreak_choice(group: pd.DataFrame, args) -> int:
    if len(group) == 1:
        return int(group.index[0])

    min_iae = float(group["horizon_iae"].min())
    tolerance = float(args.iae_tolerance_abs) + float(args.iae_tolerance_ratio) * max(
        abs(min_iae), 1.0
    )
    candidates = group[group["horizon_iae"] <= min_iae + tolerance].copy()
    if candidates.empty:
        return int(group["horizon_iae"].idxmin())

    objective_cols = [
        "horizon_iae",
        "horizon_mean_pwm",
        "horizon_pwm_variation",
        "horizon_near_saturation_ratio",
        "horizon_saturation_ratio",
    ]
    objectives = candidates[objective_cols].to_numpy(dtype=float)
    mins = objectives.min(axis=0)
    spans = np.maximum(objectives.max(axis=0) - mins, 1e-9)
    norm = (objectives - mins) / spans
    weights = np.array(
        [
            1.0,
            float(args.tiebreak_pwm_weight),
            float(args.tiebreak_variation_weight),
            float(args.tiebreak_risk_weight),
            float(args.tiebreak_risk_weight) * 2.0,
        ],
        dtype=float,
    )
    score = norm @ weights
    return int(candidates.index[int(np.argmin(score))])


def make_policy_labels(df: pd.DataFrame, args):
    df = add_direct_policy_features(df)
    feature_cols = get_policy_feature_cols(args)
    required = feature_cols + OUTPUT_COLS + TARGET_COLS
    data = df.dropna(subset=required).copy()
    data["policy_score"] = score_rows(data, args)

    # Coarse context bins turn observed candidate evaluations into a compact
    # pseudo-label set: for each state neighborhood, keep the best observed gain.
    data["target_bin"] = data["target"].round(0)
    data["current_bin"] = data["current"].round(0)
    data["error_bin"] = data["error"].round(0)
    data["derr_bin"] = (data["error_derivative"] / 10.0).round(0)
    data["pwm_bin"] = (data["pwm"] / 2.0).round(0)
    data["change_bin"] = (data["time_since_target_change"] / 0.5).round(0)
    data["direction_bin"] = np.sign(data["error"]).astype(int)

    group_cols = [
        "target_bin",
        "current_bin",
        "error_bin",
        "derr_bin",
        "pwm_bin",
        "change_bin",
        "direction_bin",
    ]
    grouped = data.groupby(group_cols, dropna=False)
    if args.label_selection == "rank_weighted":
        labels = make_rank_weighted_labels(grouped, args)
    elif args.label_selection == "pareto_utopia":
        best_idx = grouped.apply(lambda group: pareto_utopia_choice(group, args))
        labels = data.loc[best_idx].sort_values("policy_score").reset_index(drop=True)
    elif args.label_selection == "iae_pwm_tiebreak":
        best_idx = grouped.apply(lambda group: iae_pwm_tiebreak_choice(group, args))
        labels = data.loc[best_idx].sort_values("policy_score").reset_index(drop=True)
    else:
        best_idx = grouped["policy_score"].idxmin()
        labels = data.loc[best_idx].sort_values("policy_score").reset_index(drop=True)

    if args.max_labels and len(labels) > args.max_labels:
        labels = labels.sample(n=args.max_labels, random_state=args.seed).reset_index(drop=True)

    if "sample_weight" not in labels.columns:
        labels["sample_weight"] = 1.0

    return labels


def make_rank_weighted_labels(grouped, args) -> pd.DataFrame:
    frames = []
    top_k = max(1, int(args.rank_top_k))
    temperature = max(float(args.rank_temperature), 1e-6)
    score_col = str(args.rank_score_col)

    for _, group in grouped:
        score = group[score_col].to_numpy(dtype=float)
        order = np.argsort(score)[: min(top_k, len(group))]
        selected = group.iloc[order].copy()

        selected_score = selected[score_col].to_numpy(dtype=float)
        score_min = float(np.nanmin(score))
        score_span = max(float(np.nanmax(score) - score_min), 1e-9)
        normalized_gap = (selected_score - score_min) / score_span
        weights = np.exp(-normalized_gap / temperature)
        weights = weights / max(float(weights.sum()), 1e-9)
        selected["sample_weight"] = weights
        frames.append(selected)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(
        [score_col, "policy_score"]
    ).reset_index(drop=True)


def normalize_gains(y: pd.DataFrame, bounds: dict):
    values = []
    for col in OUTPUT_COLS:
        lo, hi = bounds[col]
        values.append(((y[col].to_numpy(dtype=float) - lo) / max(hi - lo, 1e-9)).clip(0.0, 1.0))
    return np.vstack(values).T.astype(np.float32)


def residual_scales_from_training(y_train: pd.DataFrame, X_train: pd.DataFrame):
    scales = {}
    for col in OUTPUT_COLS:
        residual = (
            y_train[col].to_numpy(dtype=float)
            - X_train[f"db_base_{col}"].to_numpy(dtype=float)
        )
        scale = float(np.nanpercentile(np.abs(residual), 99.0))
        scales[col] = max(scale, 1e-6)
    return scales


def normalize_residual_gains(
    y: pd.DataFrame,
    X: pd.DataFrame,
    residual_scales: dict,
):
    values = []
    for col in OUTPUT_COLS:
        residual = (
            y[col].to_numpy(dtype=float)
            - X[f"db_base_{col}"].to_numpy(dtype=float)
        )
        scale = max(float(residual_scales[col]), 1e-9)
        values.append(np.clip(residual / scale, -1.0, 1.0))
    return np.vstack(values).T.astype(np.float32)


def denormalize_gains(y_norm: np.ndarray, bounds: dict):
    values = []
    for idx, col in enumerate(OUTPUT_COLS):
        lo, hi = bounds[col]
        values.append(lo + np.clip(y_norm[:, idx], 0.0, 1.0) * (hi - lo))
    return np.vstack(values).T


def decode_predictions(
    y_pred: np.ndarray,
    X: pd.DataFrame,
    bounds: dict,
    output_mode: str,
    residual_scales: Optional[dict] = None,
):
    if output_mode == "residual_db":
        values = []
        residual_scales = residual_scales or {}
        for idx, col in enumerate(OUTPUT_COLS):
            lo, hi = bounds[col]
            base = X[f"db_base_{col}"].to_numpy(dtype=float)
            scale = max(float(residual_scales.get(col, 1.0)), 1e-9)
            value = base + np.clip(y_pred[:, idx], -1.0, 1.0) * scale
            values.append(np.clip(value, lo, hi))
        return np.vstack(values).T
    return denormalize_gains(y_pred, bounds)


def parse_hidden_layers(hidden_layers: str):
    values = []
    for raw in hidden_layers.split(","):
        raw = raw.strip()
        if not raw:
            continue
        width = int(raw)
        if width <= 0:
            raise ValueError("--hidden-layers values must be positive integers")
        values.append(width)
    if not values:
        raise ValueError("--hidden-layers must include at least one width")
    return values


def build_model(
    tf,
    input_dim: int,
    learning_rate: float,
    dropout: float,
    hidden_layers: str,
    output_mode: str,
):
    inputs = tf.keras.Input(shape=(input_dim,), name="policy_features")
    x = inputs
    widths = parse_hidden_layers(hidden_layers)
    for layer_idx, width in enumerate(widths):
        x = tf.keras.layers.Dense(width, activation="relu", name=f"dense_{layer_idx + 1}")(x)
        if layer_idx < len(widths) - 1:
            x = tf.keras.layers.BatchNormalization(name=f"batch_norm_{layer_idx + 1}")(x)
            x = tf.keras.layers.Dropout(dropout, name=f"dropout_{layer_idx + 1}")(x)
    output_activation = "tanh" if output_mode == "residual_db" else "sigmoid"
    outputs = tf.keras.layers.Dense(
        3,
        activation=output_activation,
        name="normalized_gain",
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.Huber(),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def evaluate(
    model,
    scaler,
    X_train,
    y_train,
    X_test,
    y_test,
    bounds,
    batch_size,
    output_mode: str,
    residual_scales: Optional[dict] = None,
):
    train_pred = model.predict(
        scaler.transform(X_train).astype(np.float32),
        batch_size=batch_size,
        verbose=0,
    )
    test_pred = model.predict(
        scaler.transform(X_test).astype(np.float32),
        batch_size=batch_size,
        verbose=0,
    )

    train_pred_gain = decode_predictions(
        train_pred,
        X_train,
        bounds,
        output_mode,
        residual_scales,
    )
    test_pred_gain = decode_predictions(
        test_pred,
        X_test,
        bounds,
        output_mode,
        residual_scales,
    )
    train_true_gain = y_train[OUTPUT_COLS].to_numpy(dtype=float)
    test_true_gain = y_test[OUTPUT_COLS].to_numpy(dtype=float)

    rows = []
    for idx, col in enumerate(OUTPUT_COLS):
        rows.append(
            {
                "target": col,
                "train_mae": float(mean_absolute_error(train_true_gain[:, idx], train_pred_gain[:, idx])),
                "test_mae": float(mean_absolute_error(test_true_gain[:, idx], test_pred_gain[:, idx])),
                "train_rmse": float(np.sqrt(mean_squared_error(train_true_gain[:, idx], train_pred_gain[:, idx]))),
                "test_rmse": float(np.sqrt(mean_squared_error(test_true_gain[:, idx], test_pred_gain[:, idx]))),
                "train_r2": float(r2_score(train_true_gain[:, idx], train_pred_gain[:, idx])),
                "test_r2": float(r2_score(test_true_gain[:, idx], test_pred_gain[:, idx])),
            }
        )
    return pd.DataFrame(rows), test_pred_gain


def save_model(
    model,
    scaler,
    metrics_df,
    bounds,
    args,
    dataset_path: Path,
    label_count: int,
    timestamp: str,
    feature_cols,
    residual_scales,
):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    keras_path = MODEL_DIR / f"esp32_direct_policy_mlp_{timestamp}.keras"
    latest_keras_path = MODEL_DIR / "esp32_direct_policy_mlp_latest.keras"
    metadata_path = MODEL_DIR / f"esp32_direct_policy_mlp_{timestamp}.joblib"
    latest_metadata_path = MODEL_DIR / "esp32_direct_policy_mlp_latest.joblib"

    model.save(keras_path)
    model.save(latest_keras_path)

    payload = {
        "keras_model_path": str(keras_path),
        "latest_keras_model_path": str(latest_keras_path),
        "scaler": scaler,
        "feature_cols": list(feature_cols),
        "output_cols": OUTPUT_COLS,
        "gain_bounds": bounds,
        "db_reference_targets": DB_REFERENCE_TARGETS,
        "use_db_reference_features": bool(args.use_db_reference_features),
        "use_db_prior_features": bool(args.use_db_prior_features),
        "use_transition_context_features": bool(args.use_transition_context_features),
        "hidden_layers": str(args.hidden_layers),
        "output_mode": str(args.output_mode),
        "residual_scales": dict(residual_scales or {}),
        "label_selection": str(args.label_selection),
        "pareto_objective_cols": PARETO_OBJECTIVE_COLS,
        "pareto_weights": {
            "horizon_iae": float(args.pareto_performance_weight),
            "horizon_overshoot_ratio": float(args.pareto_risk_weight),
            "horizon_near_saturation_ratio": float(args.pareto_risk_weight),
            "horizon_mean_pwm": float(args.pareto_pwm_weight),
            "horizon_pwm_variation": float(args.pareto_variation_weight),
        },
        "iae_pwm_tiebreak": {
            "iae_tolerance_abs": float(args.iae_tolerance_abs),
            "iae_tolerance_ratio": float(args.iae_tolerance_ratio),
            "tiebreak_pwm_weight": float(args.tiebreak_pwm_weight),
            "tiebreak_variation_weight": float(args.tiebreak_variation_weight),
            "tiebreak_risk_weight": float(args.tiebreak_risk_weight),
        },
        "rank_weighted": {
            "rank_top_k": int(args.rank_top_k),
            "rank_temperature": float(args.rank_temperature),
            "rank_score_col": str(args.rank_score_col),
        },
        "weighted_score_weights": {
            "horizon_mean_pwm": float(args.mean_pwm_weight),
            "horizon_pwm_variation": float(args.pwm_variation_weight),
            "horizon_near_saturation_ratio": float(args.near_saturation_weight),
            "horizon_saturation_ratio": float(args.saturation_weight),
        },
        "dataset_path": str(dataset_path),
        "label_count": int(label_count),
        "metrics": metrics_df.to_dict(orient="records"),
        "model_type": "tensorflow_direct_policy_mlp",
    }
    latest_payload = dict(payload)
    latest_payload["keras_model_path"] = str(latest_keras_path)

    joblib.dump(payload, metadata_path)
    joblib.dump(latest_payload, latest_metadata_path)
    print(f"Saved model: {keras_path}")
    print(f"Saved latest model: {latest_keras_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved latest metadata: {latest_metadata_path}")


def save_outputs(metrics_df, history, y_test, test_pred_gain, timestamp: str):
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = SUMMARY_DIR / f"esp32_direct_policy_mlp_metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"Saved metrics: {metrics_path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.history["loss"], label="train loss")
    ax.plot(history.history["val_loss"], label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber loss")
    ax.set_title("ESP32 Direct Policy MLP Training")
    ax.grid(True, alpha=0.3)
    ax.legend()
    train_path = FIGURE_DIR / f"esp32_direct_policy_mlp_training_{timestamp}.png"
    fig.savefig(train_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {train_path}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for idx, col in enumerate(OUTPUT_COLS):
        axes[idx].scatter(y_test[col], test_pred_gain[:, idx], s=8, alpha=0.35)
        lo = min(float(y_test[col].min()), float(test_pred_gain[:, idx].min()))
        hi = max(float(y_test[col].max()), float(test_pred_gain[:, idx].max()))
        axes[idx].plot([lo, hi], [lo, hi], "k--", linewidth=1)
        axes[idx].set_title(col)
        axes[idx].set_xlabel("Pseudo-label")
        axes[idx].set_ylabel("Predicted")
        axes[idx].grid(True, alpha=0.3)
    fig.tight_layout()
    pred_path = FIGURE_DIR / f"esp32_direct_policy_mlp_prediction_{timestamp}.png"
    fig.savefig(pred_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {pred_path}")


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_path = Path(args.dataset)

    print("=" * 80)
    print("Train ESP32 direct-policy TensorFlow MLP")
    print("=" * 80)
    tf = configure_tensorflow(args.seed)
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPUs: {tf.config.list_physical_devices('GPU')}")

    df = load_dataset(dataset_path)
    labels = make_policy_labels(df, args)
    bounds = gain_bounds()

    print(f"Dataset: {dataset_path}")
    print(f"Rows: {len(df)}")
    print(f"Pseudo-labels: {len(labels)}")
    feature_cols = get_policy_feature_cols(args)
    print(f"Features: {len(feature_cols)}")

    X = labels[feature_cols]
    y = labels[OUTPUT_COLS]
    sample_weight = labels["sample_weight"].to_numpy(dtype=np.float32)
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X,
        y,
        sample_weight,
        test_size=args.test_size,
        random_state=args.seed,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    residual_scales = {}
    if args.output_mode == "residual_db":
        residual_scales = residual_scales_from_training(y_train, X_train)
        y_train_norm = normalize_residual_gains(y_train, X_train, residual_scales)
        y_test_norm = normalize_residual_gains(y_test, X_test, residual_scales)
        print(f"Residual scales: {residual_scales}")
    else:
        y_train_norm = normalize_gains(y_train, bounds)
        y_test_norm = normalize_gains(y_test, bounds)

    model = build_model(
        tf,
        X_train_scaled.shape[1],
        args.learning_rate,
        args.dropout,
        args.hidden_layers,
        args.output_mode,
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(5, args.patience // 3),
            min_lr=1e-5,
        ),
    ]
    history = model.fit(
        X_train_scaled,
        y_train_norm,
        sample_weight=w_train,
        validation_data=(X_test_scaled, y_test_norm),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    metrics_df, test_pred_gain = evaluate(
        model,
        scaler,
        X_train,
        y_train,
        X_test,
        y_test,
        bounds,
        args.batch_size,
        args.output_mode,
        residual_scales,
    )
    print("Metrics:")
    print(metrics_df.to_string(index=False))

    save_model(
        model,
        scaler,
        metrics_df,
        bounds,
        args,
        dataset_path,
        len(labels),
        timestamp,
        feature_cols,
        residual_scales,
    )
    save_outputs(metrics_df, history, y_test, test_pred_gain, timestamp)


if __name__ == "__main__":
    main()
