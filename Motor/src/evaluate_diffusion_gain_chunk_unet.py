import argparse
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import joblib
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))

from config import RESULTS_DIR  # noqa: E402
from train_diffusion_gain_chunk_unet import (  # noqa: E402
    build_diffusion_constants,
    build_unet,
    ddim_sample,
    evaluate_samples,
    make_arrays,
    prepare_data,
)


SUMMARY_DIR = RESULTS_DIR / "summary"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved diffusion U-Net gain chunk model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--profile", default="tracking_first")
    parser.add_argument("--quality", choices=["all", "top_k", "best"], default="top_k")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--eval-count", type=int, default=1024)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--cost-surrogate-model", default="")
    parser.add_argument("--cost-selection-metric", default="label_cost")
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


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tf = configure_tensorflow(args.seed)
    payload = joblib.load(args.model)
    arch = payload["architecture"]
    model_args = SimpleNamespace(
        condition_dim=int(arch["condition_dim"]),
        time_embed_dim=int(arch["time_embed_dim"]),
        base_filters=int(arch["base_filters"]),
        dropout=float(arch["dropout"]),
        norm=str(arch.get("norm", "batch")),
        condition_mode=str(arch.get("condition_mode", "avg")),
        diffusion_steps=int(payload["diffusion_steps"]),
        ddim_steps=int(args.ddim_steps),
    )
    if not args.dataset:
        args.dataset = payload["dataset_path"]
    data = prepare_data(args)
    model = build_unet(
        tf,
        obs_steps=int(payload["obs_steps"]),
        obs_dim=len(payload["obs_cols"]),
        static_dim=len(payload["static_feature_cols"]),
        horizon_steps=int(payload["horizon_steps"]),
        args=model_args,
    )
    model.load_weights(payload["weights_path"])
    constants = build_diffusion_constants(tf, int(payload["diffusion_steps"]))
    eval_count = min(int(args.eval_count), len(data["test_seq"]))
    sample_diff = ddim_sample(
        tf,
        model,
        data["test_seq"][:eval_count],
        data["test_static"][:eval_count],
        model_args,
        constants,
        sample_count=int(args.sample_count),
    )
    metrics = evaluate_samples(data["y_test_gain"][:eval_count], sample_diff)
    if args.cost_surrogate_model and int(args.sample_count) > 1:
        cost_payload = joblib.load(args.cost_surrogate_model)
        cost_model = tf.keras.models.load_model(cost_payload["keras_model_path"])
        target_cols = list(cost_payload["target_cols"])
        if args.cost_selection_metric not in target_cols:
            raise ValueError(
                f"cost_selection_metric={args.cost_selection_metric!r} not in {target_cols}"
            )

        cost_df = data["test_df"].iloc[:eval_count].reset_index(drop=True)
        raw_seq, raw_static, _ = make_arrays(
            cost_df,
            list(cost_payload["obs_cols"]),
            list(cost_payload["static_feature_cols"]),
        )
        seq_flat = raw_seq.reshape((len(raw_seq), -1))
        x_seq = cost_payload["seq_scaler"].transform(seq_flat).reshape(raw_seq.shape)
        x_static = cost_payload["static_scaler"].transform(raw_static)

        candidate_norm = ((sample_diff + 1.0) / 2.0).clip(0.0, 1.0)
        n, sample_n, horizon_steps, gain_dim = candidate_norm.shape
        gain_flat = candidate_norm.reshape((n * sample_n, horizon_steps * gain_dim))
        gain_scaled = cost_payload["gain_scaler"].transform(gain_flat)
        gain_scaled = gain_scaled.reshape((n * sample_n, horizon_steps, gain_dim))
        x_seq_rep = x_seq[:, None, :, :].repeat(sample_n, axis=1).reshape(
            (n * sample_n, x_seq.shape[1], x_seq.shape[2])
        )
        x_static_rep = x_static[:, None, :].repeat(sample_n, axis=1).reshape(
            (n * sample_n, x_static.shape[1])
        )
        pred_scaled = cost_model.predict(
            [x_seq_rep.astype("float32"), x_static_rep.astype("float32"), gain_scaled.astype("float32")],
            batch_size=512,
            verbose=0,
        )
        pred = cost_payload["target_scaler"].inverse_transform(pred_scaled)
        pred = pred.reshape((n, sample_n, len(target_cols)))
        metric_idx = target_cols.index(args.cost_selection_metric)
        selected_idx = pred[:, :, metric_idx].argmin(axis=1)
        selected_diff = sample_diff[
            pd.RangeIndex(n).to_numpy(),
            selected_idx,
        ][:, None, :, :]
        selected_metrics = evaluate_samples(data["y_test_gain"][:eval_count], selected_diff)
        selected_metrics = selected_metrics[
            selected_metrics["prediction"].eq("sample_mean")
        ].copy()
        selected_metrics["prediction"] = selected_metrics["prediction"].replace(
            {
                "sample_mean": f"cost_selected_{args.cost_selection_metric}",
            }
        )
        metrics = pd.concat([metrics, selected_metrics], ignore_index=True)

        cost_rows = []
        selected_pred = pred[pd.RangeIndex(n).to_numpy(), selected_idx]
        for col_idx, col in enumerate(target_cols):
            cost_rows.append(
                {
                    "prediction": f"cost_selected_{args.cost_selection_metric}_predicted_metric",
                    "target": col,
                    "mae": float(selected_pred[:, col_idx].mean()),
                    "rmse": float(selected_pred[:, col_idx].std()),
                    "r2": float(selected_pred[:, col_idx].min()),
                }
            )
        metrics = pd.concat([metrics, pd.DataFrame(cost_rows)], ignore_index=True)

    metrics.insert(0, "model_path", str(args.model))
    metrics.insert(1, "ddim_steps", int(args.ddim_steps))
    metrics.insert(2, "sample_count", int(args.sample_count))
    metrics.insert(3, "eval_count", int(eval_count))
    if args.cost_surrogate_model:
        metrics.insert(4, "cost_surrogate_model", str(args.cost_surrogate_model))
        metrics.insert(5, "cost_selection_metric", str(args.cost_selection_metric))
    label = args.run_label or Path(args.model).stem
    out_path = SUMMARY_DIR / f"diffusion_gain_chunk_unet_eval_{label}_ddim{args.ddim_steps}_n{args.sample_count}_{timestamp}.csv"
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(metrics.to_string(index=False))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
