import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))


PROCESSED_ROOT = MOTOR_DIR / "data" / "processed" / "diffusion_gain_chunk_db"
SUMMARY_DIR = MOTOR_DIR / "results" / "summary"

RAW_METRIC_COLS = [
    "chunk_iae",
    "chunk_post_iae",
    "chunk_mean_abs_error",
    "chunk_max_abs_error",
    "chunk_overshoot",
    "chunk_overshoot_ratio",
    "chunk_pwm_mean",
    "chunk_pwm_max",
    "chunk_pwm_variation",
    "chunk_saturation_ratio",
    "chunk_near_saturation_ratio",
    "chunk_gain_variation",
]

DEFAULT_WEIGHT_PROFILES = {
    "tracking_first": {
        "chunk_iae": 1.0,
        "chunk_overshoot_ratio": 25.0,
        "chunk_saturation_ratio": 15.0,
        "chunk_near_saturation_ratio": 4.0,
        "chunk_pwm_mean": 0.001,
        "chunk_pwm_variation": 0.0005,
        "chunk_gain_variation": 0.02,
    },
    "smooth_control": {
        "chunk_iae": 1.0,
        "chunk_overshoot_ratio": 18.0,
        "chunk_saturation_ratio": 20.0,
        "chunk_near_saturation_ratio": 6.0,
        "chunk_pwm_mean": 0.004,
        "chunk_pwm_variation": 0.004,
        "chunk_gain_variation": 0.10,
    },
    "iae_only_safe": {
        "chunk_iae": 1.0,
        "chunk_saturation_ratio": 50.0,
        "chunk_near_saturation_ratio": 8.0,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create re-weightable labels from diffusion gain chunk raw metrics."
    )
    parser.add_argument(
        "--chunk-path",
        default="",
        help="Input chunk_raw_metrics CSV. Defaults to latest processed diffusion chunk DB.",
    )
    parser.add_argument("--profile", default="tracking_first", choices=sorted(DEFAULT_WEIGHT_PROFILES))
    parser.add_argument(
        "--weights-json",
        default="",
        help="Optional JSON object overriding the selected weight profile.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--target-bin", type=float, default=5.0)
    parser.add_argument("--current-bin", type=float, default=5.0)
    parser.add_argument("--error-bin", type=float, default=5.0)
    parser.add_argument("--delta-bin", type=float, default=10.0)
    parser.add_argument("--label", default="")
    return parser.parse_args()


def latest_chunk_path():
    paths = sorted(PROCESSED_ROOT.glob("chunk_raw_metrics_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No chunk raw metrics found under {PROCESSED_ROOT}")
    return paths[-1]


def rounded_bin(series: pd.Series, width: float):
    width = max(float(width), 1e-9)
    return (series.astype(float) / width).round(0) * width


def load_weights(args):
    weights = dict(DEFAULT_WEIGHT_PROFILES[args.profile])
    if args.weights_json:
        overrides = json.loads(args.weights_json)
        for key, value in overrides.items():
            weights[str(key)] = float(value)
    return weights


def compute_cost(df: pd.DataFrame, weights: dict):
    cost = np.zeros(len(df), dtype=float)
    missing = []
    for col, weight in weights.items():
        if col not in df.columns:
            missing.append(col)
            continue
        cost += float(weight) * df[col].astype(float).to_numpy()
    return cost, missing


def add_condition_bins(df: pd.DataFrame, args):
    data = df.copy()
    data["condition_target_bin"] = rounded_bin(data["target_start"], args.target_bin)
    data["condition_current_bin"] = rounded_bin(data["current_start"], args.current_bin)
    data["condition_error_bin"] = rounded_bin(data["error_start"], args.error_bin)
    data["condition_delta_bin"] = rounded_bin(data["target_delta_in_chunk"], args.delta_bin)
    data["condition_transition"] = data["target_change_inside_chunk"].astype(bool).astype(int)
    data["condition_key"] = (
        data["scenario_type"].astype(str)
        + "|T"
        + data["condition_target_bin"].astype(str)
        + "|C"
        + data["condition_current_bin"].astype(str)
        + "|E"
        + data["condition_error_bin"].astype(str)
        + "|D"
        + data["condition_delta_bin"].astype(str)
        + "|X"
        + data["condition_transition"].astype(str)
    )
    return data


def label_chunks(df: pd.DataFrame, args, weights: dict):
    data = add_condition_bins(df, args)
    cost, missing = compute_cost(data, weights)
    data["label_cost"] = cost
    data["label_profile"] = args.profile
    data["label_weight_json"] = json.dumps(weights, sort_keys=True, ensure_ascii=True)

    data["global_rank"] = data["label_cost"].rank(method="first", ascending=True)
    data["condition_rank"] = (
        data.groupby("condition_key")["label_cost"].rank(method="first", ascending=True)
    )
    data["condition_count"] = data.groupby("condition_key")["sample_id"].transform("count")
    data["is_top_k_in_condition"] = data["condition_rank"] <= int(args.top_k)
    data["is_condition_best"] = data["condition_rank"] == 1

    data["label_quality"] = "candidate"
    data.loc[data["is_top_k_in_condition"], "label_quality"] = "top_k"
    data.loc[data["is_condition_best"], "label_quality"] = "best"
    data.loc[data["chunk_saturation_ratio"].astype(float) > 0.0, "label_quality"] = (
        data["label_quality"] + "_saturated"
    )

    return data, missing


def summarize(labels: pd.DataFrame, args, weights: dict, input_path: Path, output_path: Path, missing):
    summary = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "profile": args.profile,
        "weights": weights,
        "missing_weight_columns": missing,
        "rows": int(len(labels)),
        "condition_count": int(labels["condition_key"].nunique()) if not labels.empty else 0,
        "top_k": int(args.top_k),
        "top_k_rows": int(labels["is_top_k_in_condition"].sum()) if not labels.empty else 0,
        "condition_best_rows": int(labels["is_condition_best"].sum()) if not labels.empty else 0,
        "cost_mean": float(labels["label_cost"].mean()) if not labels.empty else None,
        "cost_min": float(labels["label_cost"].min()) if not labels.empty else None,
        "cost_max": float(labels["label_cost"].max()) if not labels.empty else None,
    }
    return summary


def main():
    args = parse_args()
    input_path = Path(args.chunk_path) if args.chunk_path else latest_chunk_path()
    if not input_path.is_absolute():
        input_path = MOTOR_DIR / input_path
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    label = args.label or input_path.stem.replace("chunk_raw_metrics_", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    weights = load_weights(args)
    df = pd.read_csv(input_path)
    labels, missing = label_chunks(df, args, weights)

    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_ROOT / f"chunk_labels_{label}_{args.profile}_{timestamp}.csv"
    labels.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary = summarize(labels, args, weights, input_path, output_path, missing)
    summary_path = SUMMARY_DIR / f"diffusion_gain_chunk_label_summary_{label}_{args.profile}_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
