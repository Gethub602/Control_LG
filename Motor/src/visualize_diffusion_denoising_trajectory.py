import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))

from config import FIGURE_DIR, RESULTS_DIR  # noqa: E402
from train_diffusion_gain_chunk_unet import (  # noqa: E402
    GAIN_COLS,
    build_diffusion_constants,
    build_unet,
    diffusion_to_gain_space,
    prepare_data,
)


SUMMARY_DIR = RESULTS_DIR / "summary"
FIGURE_OUT_DIR = FIGURE_DIR / "diffusion_denoising"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize DDIM denoising trajectory for 20-step gain chunks."
    )
    parser.add_argument(
        "--model",
        default=(
            "results/models/"
            "diffusion_gain_chunk_unet_balanced1000_global_topk_light_bn_attention_"
            "20260508_200449.joblib"
        ),
    )
    parser.add_argument("--dataset", default="")
    parser.add_argument("--profile", default="tracking_first")
    parser.add_argument("--quality", choices=["all", "top_k", "best"], default="top_k")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--ddim-steps", type=int, default=10)
    parser.add_argument("--run-label", default="light_attention_ddim10")
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


def build_model_args(payload, ddim_steps: int):
    arch = payload["architecture"]
    return SimpleNamespace(
        condition_dim=int(arch["condition_dim"]),
        time_embed_dim=int(arch["time_embed_dim"]),
        base_filters=int(arch["base_filters"]),
        dropout=float(arch["dropout"]),
        norm=str(arch.get("norm", "batch")),
        condition_mode=str(arch.get("condition_mode", "avg")),
        diffusion_steps=int(payload["diffusion_steps"]),
        ddim_steps=int(ddim_steps),
    )


def ddim_trace(tf, model, obs, static, args, constants):
    horizon_steps = 20
    x = tf.random.normal((1, horizon_steps, len(GAIN_COLS)), dtype=tf.float32)
    alpha_bars = constants["alpha_bars"].numpy()
    step_indices = np.linspace(args.diffusion_steps - 1, 0, int(args.ddim_steps), dtype=int)

    rows = []
    snapshots = []
    for denoise_idx, t_idx in enumerate(step_indices, start=1):
        t = tf.fill((1,), int(t_idx))
        eps = model([x, t, obs, static], training=False)
        a_t = float(alpha_bars[t_idx])
        x0 = (x - math.sqrt(max(1.0 - a_t, 1e-12)) * eps) / math.sqrt(max(a_t, 1e-12))
        x0 = tf.clip_by_value(x0, -1.5, 1.5)

        gain_preview = diffusion_to_gain_space(
            tf.clip_by_value(x0, -1.0, 1.0).numpy()
        )[0]
        snapshots.append((denoise_idx, int(t_idx), gain_preview.copy()))
        for step_idx, values in enumerate(gain_preview):
            rows.append(
                {
                    "denoise_index": denoise_idx,
                    "diffusion_t": int(t_idx),
                    "chunk_step": step_idx,
                    "chunk_time_sec": step_idx * 0.1,
                    "kp": float(values[0]),
                    "ki": float(values[1]),
                    "kd": float(values[2]),
                }
            )

        if denoise_idx == len(step_indices):
            x = x0
        else:
            t_prev = int(step_indices[denoise_idx])
            a_prev = float(alpha_bars[t_prev])
            x = math.sqrt(max(a_prev, 1e-12)) * x0 + math.sqrt(max(1.0 - a_prev, 1e-12)) * eps

    final_gain = diffusion_to_gain_space(tf.clip_by_value(x, -1.0, 1.0).numpy())[0]
    return pd.DataFrame(rows), snapshots, final_gain


def plot_trace(trace_df, final_gain, y_true, sample_meta, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.5), sharex=True)
    cmap = plt.get_cmap("viridis")
    max_idx = int(trace_df["denoise_index"].max())

    for ax, gain_col, true_idx in zip(axes, GAIN_COLS, range(len(GAIN_COLS))):
        for denoise_idx, group in trace_df.groupby("denoise_index"):
            alpha = 0.22 + 0.70 * denoise_idx / max(max_idx, 1)
            color = cmap((denoise_idx - 1) / max(max_idx - 1, 1))
            label = f"DDIM {denoise_idx}" if denoise_idx in {1, max_idx} else None
            ax.plot(
                group["chunk_time_sec"],
                group[gain_col],
                color=color,
                alpha=alpha,
                linewidth=1.2,
                label=label,
            )
        ax.plot(
            np.arange(len(final_gain)) * 0.1,
            final_gain[:, true_idx],
            color="black",
            linewidth=2.2,
            label="final generated",
        )
        if y_true is not None:
            ax.plot(
                np.arange(len(y_true)) * 0.1,
                y_true[:, true_idx],
                color="tab:red",
                linestyle="--",
                linewidth=1.8,
                label="label chunk",
            )
        ax.set_ylabel(gain_col)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    title = (
        "DDIM Denoising Trajectory for 20-step PID Gain Chunk\n"
        f"sample={sample_meta.get('sample_index')}, "
        f"target={sample_meta.get('target'):.1f}, "
        f"current={sample_meta.get('current'):.2f}, "
        f"error={sample_meta.get('error'):.2f}"
    )
    fig.suptitle(title)
    axes[-1].set_xlabel("chunk time [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_heatmap(trace_df, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, gain_col in zip(axes, GAIN_COLS):
        pivot = trace_df.pivot(
            index="denoise_index",
            columns="chunk_step",
            values=gain_col,
        )
        im = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower", cmap="magma")
        ax.set_title(gain_col)
        ax.set_xlabel("chunk step")
        ax.set_xticks([0, 5, 10, 15, 19])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index.tolist())
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("DDIM denoise index")
    fig.suptitle("Gain Chunk Values During DDIM Denoising")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tf = configure_tensorflow(args.seed)
    payload = joblib.load(args.model)
    model_args = build_model_args(payload, args.ddim_steps)
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

    idx = int(np.clip(args.sample_index, 0, len(data["test_seq"]) - 1))
    obs = data["test_seq"][idx : idx + 1].astype(np.float32)
    static = data["test_static"][idx : idx + 1].astype(np.float32)
    y_true = data["y_test_gain"][idx]
    row = data["test_df"].iloc[idx]
    sample_meta = {
        "sample_index": idx,
        "target": float(row.get("state_target", np.nan)),
        "current": float(row.get("state_current", np.nan)),
        "error": float(row.get("state_error", np.nan)),
    }

    trace_df, _, final_gain = ddim_trace(tf, model, obs, static, model_args, constants)
    for key, value in sample_meta.items():
        trace_df[key] = value

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    label = f"{args.run_label}_sample{idx}_ddim{args.ddim_steps}_{timestamp}"
    csv_path = SUMMARY_DIR / f"diffusion_denoising_trace_{label}.csv"
    line_path = FIGURE_OUT_DIR / f"diffusion_denoising_lines_{label}.png"
    heatmap_path = FIGURE_OUT_DIR / f"diffusion_denoising_heatmap_{label}.png"
    trace_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    plot_trace(trace_df, final_gain, y_true, sample_meta, line_path)
    plot_heatmap(trace_df, heatmap_path)

    print(f"Saved trace CSV: {csv_path}")
    print(f"Saved line plot: {line_path}")
    print(f"Saved heatmap: {heatmap_path}")
    print(trace_df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
