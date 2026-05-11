import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))

from config import RESULTS_DIR  # noqa: E402
from train_diffusion_gain_chunk_unet import (  # noqa: E402
    build_diffusion_constants,
    build_unet,
    ddim_sample,
)


SUMMARY_DIR = RESULTS_DIR / "summary"


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark diffusion gain chunk latency.")
    parser.add_argument("--model", required=True, help="Diffusion U-Net metadata joblib path.")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--label", default="")
    return parser.parse_args()


def configure_tensorflow():
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    return tf


def main():
    args = parse_args()
    tf = configure_tensorflow()
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
    obs_steps = int(payload["obs_steps"])
    obs_dim = len(payload["obs_cols"])
    static_dim = len(payload["static_feature_cols"])
    horizon_steps = int(payload["horizon_steps"])
    model = build_unet(
        tf,
        obs_steps=obs_steps,
        obs_dim=obs_dim,
        static_dim=static_dim,
        horizon_steps=horizon_steps,
        args=model_args,
    )
    model.load_weights(payload["weights_path"])
    constants = build_diffusion_constants(tf, int(payload["diffusion_steps"]))

    obs = np.zeros((int(args.batch_size), obs_steps, obs_dim), dtype=np.float32)
    static = np.zeros((int(args.batch_size), static_dim), dtype=np.float32)

    for _ in range(int(args.warmup)):
        ddim_sample(
            tf,
            model,
            obs,
            static,
            model_args,
            constants,
            sample_count=int(args.sample_count),
        )

    times = []
    for _ in range(int(args.runs)):
        start = time.perf_counter()
        ddim_sample(
            tf,
            model,
            obs,
            static,
            model_args,
            constants,
            sample_count=int(args.sample_count),
        )
        times.append(time.perf_counter() - start)

    values = np.asarray(times, dtype=float)
    row = {
        "label": args.label or Path(args.model).stem,
        "model_path": str(args.model),
        "batch_size": int(args.batch_size),
        "runs": int(args.runs),
        "ddim_steps": int(args.ddim_steps),
        "sample_count": int(args.sample_count),
        "mean_sec": float(values.mean()),
        "p50_sec": float(np.quantile(values, 0.50)),
        "p90_sec": float(np.quantile(values, 0.90)),
        "p99_sec": float(np.quantile(values, 0.99)),
        "max_sec": float(values.max()),
    }
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUMMARY_DIR / f"diffusion_gain_chunk_latency_{row['label']}.csv"
    pd.DataFrame([row]).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(json.dumps(row, indent=2, ensure_ascii=False))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
