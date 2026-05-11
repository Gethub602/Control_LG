import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))

from config import RESULTS_DIR  # noqa: E402


SUMMARY_DIR = RESULTS_DIR / "summary"


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark gain chunk baseline latency.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
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
    model = tf.keras.models.load_model(payload["keras_model_path"])

    obs_steps = int(payload["obs_steps"])
    obs_dim = len(payload["obs_cols"])
    static_dim = len(payload["static_feature_cols"])
    batch_size = int(args.batch_size)
    x_seq = np.zeros((batch_size, obs_steps, obs_dim), dtype=np.float32)
    x_static = np.zeros((batch_size, static_dim), dtype=np.float32)

    for _ in range(int(args.warmup)):
        model.predict([x_seq, x_static], batch_size=batch_size, verbose=0)

    times = []
    for _ in range(int(args.runs)):
        start = time.perf_counter()
        model.predict([x_seq, x_static], batch_size=batch_size, verbose=0)
        times.append(time.perf_counter() - start)

    values = np.asarray(times, dtype=float)
    row = {
        "label": args.label or Path(args.model).stem,
        "model_path": str(args.model),
        "batch_size": batch_size,
        "runs": int(args.runs),
        "mean_sec": float(values.mean()),
        "p50_sec": float(np.quantile(values, 0.50)),
        "p90_sec": float(np.quantile(values, 0.90)),
        "p99_sec": float(np.quantile(values, 0.99)),
        "max_sec": float(values.max()),
    }
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUMMARY_DIR / f"gain_chunk_baseline_latency_{row['label']}.csv"
    pd.DataFrame([row]).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(json.dumps(row, indent=2, ensure_ascii=False))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
