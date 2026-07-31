"""
Head-to-head benchmark: diffusion (DDIM) vs flow matching (ODE) gain chunks.

Both models share the dataset, the conditioning columns, the scalers, and the
U-Net backbone, so the comparison isolates the generative method and its
sampling budget. Reports chunk-reconstruction accuracy and, more importantly for
this project, single-condition inference latency -- the quantity that decided
DDIM20 vs DDIM30 in the original study.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))
sys.path.append(str(CURRENT_DIR))

from config import RESULTS_DIR  # noqa: E402
from train_diffusion_gain_chunk_baselines import (  # noqa: E402
    DEFAULT_OBS_COLS,
    GAIN_COLS,
    load_chunk_labels,
    make_arrays,
    split_by_trajectory,
    static_cols,
)
from train_diffusion_gain_chunk_unet import (  # noqa: E402
    build_diffusion_constants,
    build_unet,
    configure_tensorflow,
    ddim_sample,
    diffusion_to_gain_space,
)
from train_flow_matching_gain_chunk_unet import flow_sample  # noqa: E402

SUMMARY_DIR = RESULTS_DIR / "summary"


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark diffusion vs flow matching.")
    p.add_argument("--diffusion-model", required=True)
    p.add_argument("--flow-model", required=True)
    p.add_argument("--dataset", default="")
    p.add_argument("--quality", choices=["all", "top_k", "best"], default="top_k")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-count", type=int, default=1024)
    p.add_argument("--sample-count", type=int, default=4)
    p.add_argument("--ddim-steps", default="5,10,20,30")
    p.add_argument("--flow-steps", default="1,2,4,8")
    p.add_argument("--flow-solvers", default="euler,midpoint")
    p.add_argument("--latency-repeats", type=int, default=15)
    p.add_argument("--run-label", default="")
    return p.parse_args()


def load_model(tf, payload_path):
    payload = joblib.load(payload_path)
    arch = payload["architecture"]
    args_ns = SimpleNamespace(
        condition_dim=int(arch["condition_dim"]),
        time_embed_dim=int(arch["time_embed_dim"]),
        base_filters=int(arch["base_filters"]),
        dropout=float(arch["dropout"]),
        norm=str(arch.get("norm", "batch")),
        condition_mode=str(arch.get("condition_mode", "avg")),
        diffusion_steps=int(payload["diffusion_steps"]),
        ddim_steps=20,
        solver="midpoint",
    )
    model = build_unet(
        tf,
        obs_steps=int(payload["obs_steps"]),
        obs_dim=len(payload["obs_cols"]),
        static_dim=len(payload["static_feature_cols"]),
        horizon_steps=int(payload["horizon_steps"]),
        args=args_ns,
    )
    model.load_weights(payload["weights_path"])
    return payload, model, args_ns


def build_eval_data(payload, args):
    dataset_path = Path(args.dataset or payload["dataset_path"])
    if not dataset_path.is_absolute():
        dataset_path = MOTOR_DIR / dataset_path
    df = load_chunk_labels(dataset_path, args.quality, 0, args.seed)
    _, test_df = split_by_trajectory(df, args.test_size, args.seed)
    static_feature_cols = list(payload["static_feature_cols"])
    seq, static, y = make_arrays(test_df, DEFAULT_OBS_COLS, static_feature_cols)

    n, steps, dim = seq.shape
    seq_scaled = payload["seq_scaler"].transform(seq.reshape(n, steps * dim)).reshape(
        n, steps, dim
    )
    static_scaled = payload["static_scaler"].transform(static)

    k = min(int(args.eval_count), n)
    return (
        seq_scaled[:k].astype(np.float32),
        static_scaled[:k].astype(np.float32),
        y[:k],
        dataset_path,
    )


def accuracy(samples, y_true):
    gain = diffusion_to_gain_space(
        samples.reshape((-1, samples.shape[2], samples.shape[3]))
    ).reshape(samples.shape)
    mean_gain = np.mean(gain, axis=1)
    all_mae = np.mean(np.abs(gain - y_true[:, None, :, :]), axis=(2, 3))
    best_gain = gain[np.arange(len(gain)), np.argmin(all_mae, axis=1)]
    return {
        "sample_mean_mae": float(np.mean(np.abs(mean_gain - y_true))),
        "best_of_n_mae": float(np.mean(np.abs(best_gain - y_true))),
        "kp_mae": float(np.mean(np.abs(mean_gain[:, :, 0] - y_true[:, :, 0]))),
        "ki_mae": float(np.mean(np.abs(mean_gain[:, :, 1] - y_true[:, :, 1]))),
        "kd_mae": float(np.mean(np.abs(mean_gain[:, :, 2] - y_true[:, :, 2]))),
        "chunk_rmse": float(np.sqrt(np.mean((mean_gain - y_true) ** 2))),
    }


def time_single(fn, repeats):
    fn()  # warm-up / trace
    lat = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        fn()
        lat.append(time.perf_counter() - t0)
    return {
        "latency_mean_sec": float(np.mean(lat)),
        "latency_p50_sec": float(np.percentile(lat, 50)),
        "latency_p90_sec": float(np.percentile(lat, 90)),
        "latency_max_sec": float(np.max(lat)),
    }


def main():
    args = parse_args()
    tf = configure_tensorflow(args.seed)

    diff_payload, diff_model, diff_args = load_model(tf, args.diffusion_model)
    flow_payload, flow_model, flow_args = load_model(tf, args.flow_model)
    constants = build_diffusion_constants(tf, int(diff_payload["diffusion_steps"]))

    seq, static, y_true, dataset_path = build_eval_data(diff_payload, args)
    print(f"Eval conditions: {len(seq)}  dataset={dataset_path.name}")

    rows = []

    for steps in [int(v) for v in args.ddim_steps.split(",") if v.strip()]:
        diff_args.ddim_steps = steps
        samples = ddim_sample(
            tf, diff_model, seq, static, diff_args, constants, args.sample_count
        )
        lat = time_single(
            lambda: ddim_sample(
                tf, diff_model, seq[:1], static[:1], diff_args, constants, 1
            ),
            args.latency_repeats,
        )
        row = {"method": "diffusion_ddim", "steps": steps, "solver": "ddim", "nfe": steps}
        row.update(accuracy(samples, y_true))
        row.update(lat)
        rows.append(row)
        print(
            f"  DDIM{steps:<3} mae={row['sample_mean_mae']:.5f} "
            f"p90={row['latency_p90_sec'] * 1000:7.1f} ms"
        )

    solvers = [s.strip() for s in args.flow_solvers.split(",") if s.strip()]
    for solver in solvers:
        flow_args.solver = solver
        nfe_per = 1 if solver == "euler" else 2
        for steps in [int(v) for v in args.flow_steps.split(",") if v.strip()]:
            samples = flow_sample(
                tf, flow_model, seq, static, flow_args, steps, args.sample_count
            )
            lat = time_single(
                lambda: flow_sample(
                    tf, flow_model, seq[:1], static[:1], flow_args, steps, 1
                ),
                args.latency_repeats,
            )
            row = {
                "method": f"flow_{solver}",
                "steps": steps,
                "solver": solver,
                "nfe": steps * nfe_per,
            }
            row.update(accuracy(samples, y_true))
            row.update(lat)
            rows.append(row)
            print(
                f"  FLOW-{solver}{steps:<3} mae={row['sample_mean_mae']:.5f} "
                f"p90={row['latency_p90_sec'] * 1000:7.1f} ms"
            )

    out = pd.DataFrame(rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = args.run_label or "flow_vs_diffusion"
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    path = SUMMARY_DIR / f"{label}_{timestamp}.csv"
    out.to_csv(path, index=False)

    print()
    print(
        out[
            [
                "method",
                "steps",
                "nfe",
                "sample_mean_mae",
                "best_of_n_mae",
                "latency_p90_sec",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
