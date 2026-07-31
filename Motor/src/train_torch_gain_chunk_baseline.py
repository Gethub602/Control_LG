"""
Train a supervised gain-chunk baseline (mlp / cnn / cnn_residual / cnn_attention).

Companion to train_torch_gain_chunk.py, which handles the generative methods.
Same dataset, same split, same scalers, same accuracy metrics, so the whole
family can be compared on one table.

These models are deterministic: one input gives one chunk. `best_of_n` is
therefore identical to `sample_mean` and sample diversity is zero by
construction -- reported anyway so the columns line up with the generative runs.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))
sys.path.append(str(CURRENT_DIR))

from config import MODEL_DIR, RESULTS_DIR  # noqa: E402
from train_diffusion_gain_chunk_baselines import (  # noqa: E402
    DEFAULT_OBS_COLS,
    GAIN_BOUNDS,
    GAIN_COLS,
    denormalize_gain_sequence,
    normalize_gain_sequence,
)
from torch_gain_chunk_baselines import BASELINE_MODELS, build_baseline  # noqa: E402
from torch_gain_chunk_common import chunk_accuracy, get_device, prepare_data  # noqa: E402

SUMMARY_DIR = RESULTS_DIR / "summary"


def parse_args():
    p = argparse.ArgumentParser(description="Train a torch gain-chunk baseline.")
    p.add_argument("--model-type", choices=sorted(BASELINE_MODELS), required=True)
    p.add_argument("--dataset", default="")
    p.add_argument("--profile", default="tracking_first")
    p.add_argument("--quality", choices=["all", "top_k", "best"], default="top_k")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--eval-count", type=int, default=1024)
    p.add_argument("--latency-repeats", type=int, default=30)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--run-label", default="")
    return p.parse_args()


def make_loader(seq, static, y, batch_size, shuffle, seed):
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(seq), torch.from_numpy(static), torch.from_numpy(y)
    )
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, generator=g if shuffle else None
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device(prefer_cuda=not args.cpu)

    data = prepare_data(args.dataset, args.profile, args.quality, args.max_rows,
                        args.test_size, args.seed)
    obs_steps, obs_dim = data["train_seq"].shape[1:]
    static_dim = data["train_static"].shape[1]

    # baselines regress the normalised chunk directly, in [0, 1]
    y_train = normalize_gain_sequence(data["y_train_gain"]).astype(np.float32)
    y_test = normalize_gain_sequence(data["y_test_gain"]).astype(np.float32)
    horizon = y_train.shape[1]

    print(f"model={args.model_type} device={device} dataset={data['dataset_path'].name}")
    print(f"train={len(data['train_seq'])} test={len(data['test_seq'])} "
          f"obs=({obs_steps},{obs_dim}) static={static_dim} horizon={horizon}")

    model = build_baseline(args.model_type, obs_steps, obs_dim, static_dim, horizon,
                           len(GAIN_COLS), args.dropout).to(device)
    print(f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    tr = make_loader(data["train_seq"], data["train_static"], y_train,
                     args.batch_size, True, args.seed)
    va = make_loader(data["test_seq"], data["test_static"], y_test,
                     args.batch_size, False, args.seed)

    history = {"loss": [], "val_loss": []}
    best_val, best_state, wait = np.inf, None, 0
    t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tl = []
        for obs, static, y in tr:
            obs, static, y = obs.to(device), static.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(obs, static), y)
            loss.backward()
            opt.step()
            tl.append(loss.item())

        model.eval()
        vl = []
        with torch.no_grad():
            for obs, static, y in va:
                obs, static, y = obs.to(device), static.to(device), y.to(device)
                vl.append(F.mse_loss(model(obs, static), y).item())

        loss_v, val_v = float(np.mean(tl)), float(np.mean(vl))
        history["loss"].append(loss_v)
        history["val_loss"].append(val_v)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}: loss={loss_v:.6f}, val_loss={val_v:.6f}",
                  flush=True)
        if val_v < best_val - 1e-7:
            best_val, wait = val_v, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {epoch}. Best val_loss={best_val:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    train_sec = time.perf_counter() - t0
    print(f"train_time={train_sec:.1f}s best_val={best_val:.6f}")

    # ---- accuracy on the same slice the generative runs use
    model.eval()
    k = min(args.eval_count, len(data["test_seq"]))
    obs = torch.from_numpy(data["test_seq"][:k]).to(device)
    static = torch.from_numpy(data["test_static"][:k]).to(device)
    with torch.no_grad():
        pred_norm = model(obs, static).cpu().numpy()
    pred_gain = denormalize_gain_sequence(np.clip(pred_norm, 0.0, 1.0))
    # single deterministic candidate -> shape (N, 1, H, G)
    metrics = chunk_accuracy(pred_gain[:, None, :, :], data["y_test_gain"][:k])

    # ---- single-condition latency, the number that matters for the server
    obs1, st1 = obs[:1], static[:1]
    with torch.no_grad():
        model(obs1, st1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        lat = []
        for _ in range(args.latency_repeats):
            t = time.perf_counter()
            model(obs1, st1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            lat.append(time.perf_counter() - t)

    row = {
        "method": f"baseline_{args.model_type}",
        "solver": "deterministic",
        "steps": 1,
        "nfe": 1,
        "device": device.type,
        **metrics,
        "latency_mean_sec": float(np.mean(lat)),
        "latency_p50_sec": float(np.percentile(lat, 50)),
        "latency_p90_sec": float(np.percentile(lat, 90)),
        "latency_max_sec": float(np.max(lat)),
    }
    print(f"  mae={row['sample_mean_mae']:.5f}  "
          f"p90={row['latency_p90_sec'] * 1000:.2f} ms")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = args.run_label or args.model_type
    stem = f"torch_baseline_{args.model_type}_{label}_{timestamp}"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    weights_path = MODEL_DIR / f"{stem}.pt"
    torch.save(model.state_dict(), weights_path)
    payload = {
        "model_type": f"torch_baseline_{args.model_type}",
        "framework": "pytorch",
        "weights_path": str(weights_path),
        "baseline_type": args.model_type,
        "architecture": {"dropout": args.dropout},
        "obs_cols": list(DEFAULT_OBS_COLS),
        "static_feature_cols": list(data["static_feature_cols"]),
        "gain_cols": list(GAIN_COLS),
        "gain_bounds": GAIN_BOUNDS,
        "obs_steps": int(obs_steps),
        "obs_dim": int(obs_dim),
        "static_dim": int(static_dim),
        "horizon_steps": int(horizon),
        "seq_scaler": data["seq_scaler"],
        "static_scaler": data["static_scaler"],
        "dataset_path": str(data["dataset_path"]),
        "best_val_loss": float(best_val),
        "train_seconds": float(train_sec),
        "args": vars(args),
        "eval": [row],
    }
    joblib.dump(payload, MODEL_DIR / f"{stem}.joblib")
    pd.DataFrame([row]).to_csv(SUMMARY_DIR / f"{stem}_eval.csv", index=False)
    pd.DataFrame(history).to_csv(SUMMARY_DIR / f"{stem}_history.csv", index=False)
    print(json.dumps({"model": str(MODEL_DIR / f'{stem}.joblib')}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
