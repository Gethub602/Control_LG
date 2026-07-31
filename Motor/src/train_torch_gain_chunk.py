"""
Train diffusion (DDPM/DDIM) or flow matching (rectified flow) gain-chunk models
on GPU with PyTorch.

Both methods share GainChunkUNet, the dataset, the scalers and the split, so a
comparison isolates the generative objective and its sampling budget:

  method=diffusion   y_t = sqrt(a_t) y0 + sqrt(1-a_t) eps ; predict eps
                     sample with DDIM, one network eval per step

  method=flow        y_t = (1-t) y_noise + t y0           ; predict y0 - y_noise
                     sample by integrating the ODE, 1 eval (euler) or
                     2 evals (midpoint/heun) per step

Latency is the quantity that decided DDIM20 vs DDIM30 in the original study, so
the evaluation sweeps step budgets and records single-condition inference time.
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

from config import FIGURE_DIR, MODEL_DIR, RESULTS_DIR  # noqa: E402
from train_diffusion_gain_chunk_baselines import (  # noqa: E402
    DEFAULT_OBS_COLS,
    GAIN_BOUNDS,
    GAIN_COLS,
)
from torch_gain_chunk_common import (  # noqa: E402
    GainChunkUNet,
    chunk_accuracy,
    cosine_beta_schedule,
    get_device,
    model_to_gain_space,
    prepare_data,
)

SUMMARY_DIR = RESULTS_DIR / "summary"


def parse_args():
    p = argparse.ArgumentParser(description="Train torch gain-chunk generator.")
    p.add_argument("--method", choices=["diffusion", "flow"], required=True)
    p.add_argument("--dataset", default="")
    p.add_argument("--profile", default="tracking_first")
    p.add_argument("--quality", choices=["all", "top_k", "best"], default="top_k")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows", type=int, default=0)

    p.add_argument("--base-filters", type=int, default=64)
    p.add_argument("--condition-dim", type=int, default=128)
    p.add_argument("--time-embed-dim", type=int, default=64)
    p.add_argument("--condition-mode", default="avg", choices=["avg", "gru"])
    p.add_argument("--norm", default="batch", choices=["batch", "layer"])

    # 100 matches the original study's payload (diffusion_steps=100). The noise
    # schedule differs substantially from the more common 1000, so this has to
    # match for a comparison against the inherited model to mean anything.
    p.add_argument("--diffusion-steps", type=int, default=100)
    p.add_argument("--sigma-min", type=float, default=0.0)
    p.add_argument("--time-sampling", default="uniform",
                   choices=["uniform", "logit_normal"])
    p.add_argument("--solver", default="midpoint",
                   choices=["euler", "midpoint", "heun"])

    p.add_argument("--eval-steps", default="")
    p.add_argument("--eval-count", type=int, default=1024)
    p.add_argument("--sample-count", type=int, default=4)
    p.add_argument("--latency-repeats", type=int, default=20)
    p.add_argument("--cpu", action="store_true", help="Force CPU (for A/B timing).")
    p.add_argument("--run-label", default="")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

def make_loader(seq, static, y, batch_size, shuffle, seed):
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(seq), torch.from_numpy(static), torch.from_numpy(y)
    )
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, generator=g if shuffle else None,
        drop_last=False,
    )


def sample_t(batch_size, args, device):
    if args.time_sampling == "logit_normal":
        return torch.sigmoid(torch.randn(batch_size, device=device))
    return torch.rand(batch_size, device=device)


def batch_loss(model, obs, static, y0, args, alpha_bars, device):
    b = y0.shape[0]
    if args.method == "diffusion":
        t = torch.randint(0, args.diffusion_steps, (b,), device=device)
        noise = torch.randn_like(y0)
        ab = alpha_bars[t].view(-1, 1, 1)
        y_t = ab.sqrt() * y0 + (1.0 - ab).sqrt() * noise
        pred = model(y_t, t.float(), obs, static)
        return F.mse_loss(pred, noise)

    # flow matching: straight conditional path from noise to data
    t = sample_t(b, args, device)
    tb = t.view(-1, 1, 1)
    y_noise = torch.randn_like(y0)
    sm = float(args.sigma_min)
    y_t = (1.0 - (1.0 - sm) * tb) * y_noise + tb * y0
    v_target = y0 - (1.0 - sm) * y_noise
    # scale t onto the same embedding grid the diffusion path uses
    pred = model(y_t, t * (args.diffusion_steps - 1), obs, static)
    return F.mse_loss(pred, v_target)


def train(model, data, args, alpha_bars, device):
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    tr = make_loader(data["train_seq"], data["train_static"], data["y_train_model"],
                     args.batch_size, True, args.seed)
    va = make_loader(data["test_seq"], data["test_static"], data["y_test_model"],
                     args.batch_size, False, args.seed)

    history = {"loss": [], "val_loss": []}
    best_val, best_state, wait = np.inf, None, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        tl = []
        for obs, static, y0 in tr:
            obs, static, y0 = obs.to(device), static.to(device), y0.to(device)
            opt.zero_grad(set_to_none=True)
            loss = batch_loss(model, obs, static, y0, args, alpha_bars, device)
            loss.backward()
            opt.step()
            tl.append(loss.item())

        model.eval()
        vl = []
        with torch.no_grad():
            for obs, static, y0 in va:
                obs, static, y0 = obs.to(device), static.to(device), y0.to(device)
                vl.append(batch_loss(model, obs, static, y0, args, alpha_bars, device).item())

        loss_v, val_v = float(np.mean(tl)), float(np.mean(vl))
        history["loss"].append(loss_v)
        history["val_loss"].append(val_v)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}: loss={loss_v:.6f}, val_loss={val_v:.6f}", flush=True)

        if val_v < best_val - 1e-6:
            best_val, wait = val_v, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {epoch}. Best val_loss={best_val:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return pd.DataFrame(history), best_val


# ----------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------

@torch.no_grad()
def ddim_sample(model, obs, static, num_steps, diffusion_steps, alpha_bars,
                sample_count, horizon, device):
    n = obs.shape[0]
    obs_r = obs.repeat_interleave(sample_count, dim=0)
    st_r = static.repeat_interleave(sample_count, dim=0)
    total = obs_r.shape[0]
    x = torch.randn(total, horizon, len(GAIN_COLS), device=device)
    idx = np.linspace(diffusion_steps - 1, 0, num_steps, dtype=int)

    for i, t_idx in enumerate(idx):
        t = torch.full((total,), float(t_idx), device=device)
        eps = model(x, t, obs_r, st_r)
        a_t = float(alpha_bars[int(t_idx)])
        x0 = (x - math_sqrt(1.0 - a_t) * eps) / math_sqrt(a_t)
        x0 = x0.clamp(-1.5, 1.5)
        if i == len(idx) - 1:
            x = x0
        else:
            a_prev = float(alpha_bars[int(idx[i + 1])])
            x = math_sqrt(a_prev) * x0 + math_sqrt(1.0 - a_prev) * eps

    return x.clamp(-1.0, 1.0).reshape(n, sample_count, horizon, len(GAIN_COLS))


def math_sqrt(v):
    return float(np.sqrt(max(v, 1e-12)))


@torch.no_grad()
def flow_sample(model, obs, static, num_steps, solver, diffusion_steps,
                sample_count, horizon, device):
    n = obs.shape[0]
    obs_r = obs.repeat_interleave(sample_count, dim=0)
    st_r = static.repeat_interleave(sample_count, dim=0)
    total = obs_r.shape[0]
    y = torch.randn(total, horizon, len(GAIN_COLS), device=device)
    dt = 1.0 / float(num_steps)
    scale = float(diffusion_steps - 1)

    def v(y_cur, t_scalar):
        t = torch.full((total,), float(t_scalar) * scale, device=device)
        return model(y_cur, t, obs_r, st_r)

    for i in range(num_steps):
        t0 = i * dt
        if solver == "euler":
            y = y + dt * v(y, t0)
        elif solver == "midpoint":
            k1 = v(y, t0)
            y = y + dt * v(y + 0.5 * dt * k1, t0 + 0.5 * dt)
        else:  # heun
            k1 = v(y, t0)
            k2 = v(y + dt * k1, min(t0 + dt, 1.0))
            y = y + 0.5 * dt * (k1 + k2)

    return y.clamp(-1.0, 1.0).reshape(n, sample_count, horizon, len(GAIN_COLS))


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def evaluate(model, data, args, alpha_bars, device, horizon):
    step_list = [int(v) for v in (args.eval_steps or default_steps(args)).split(",")]
    obs = torch.from_numpy(data["test_seq"][: args.eval_count]).to(device)
    static = torch.from_numpy(data["test_static"][: args.eval_count]).to(device)
    y_true = data["y_test_gain"][: args.eval_count]
    obs1, st1 = obs[:1], static[:1]
    rows = []

    for steps in step_list:
        def run(o, s, sc):
            if args.method == "diffusion":
                return ddim_sample(model, o, s, steps, args.diffusion_steps,
                                   alpha_bars, sc, horizon, device)
            return flow_sample(model, o, s, steps, args.solver,
                               args.diffusion_steps, sc, horizon, device)

        samples = run(obs, static, args.sample_count).cpu().numpy()
        gain = model_to_gain_space(
            samples.reshape(-1, samples.shape[2], samples.shape[3])
        ).reshape(samples.shape)

        run(obs1, st1, 1)  # warm up
        sync(device)
        lat = []
        for _ in range(args.latency_repeats):
            t0 = time.perf_counter()
            run(obs1, st1, 1)
            sync(device)
            lat.append(time.perf_counter() - t0)

        nfe_per = 1 if (args.method == "diffusion" or args.solver == "euler") else 2
        row = {
            "method": args.method,
            "solver": "ddim" if args.method == "diffusion" else args.solver,
            "steps": steps,
            "nfe": steps * nfe_per,
            "device": device.type,
        }
        row.update(chunk_accuracy(gain, y_true))
        row.update({
            "latency_mean_sec": float(np.mean(lat)),
            "latency_p50_sec": float(np.percentile(lat, 50)),
            "latency_p90_sec": float(np.percentile(lat, 90)),
            "latency_max_sec": float(np.max(lat)),
        })
        rows.append(row)
        print(f"  steps={steps:>3} nfe={row['nfe']:>3} "
              f"mae={row['sample_mean_mae']:.5f} best={row['best_of_n_mae']:.5f} "
              f"p90={row['latency_p90_sec'] * 1000:7.2f} ms", flush=True)
    return pd.DataFrame(rows)


def default_steps(args):
    return "5,10,20,30" if args.method == "diffusion" else "1,2,4,8,20"


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device(prefer_cuda=not args.cpu)

    data = prepare_data(args.dataset, args.profile, args.quality, args.max_rows,
                        args.test_size, args.seed)
    obs_steps, obs_dim = data["train_seq"].shape[1:]
    static_dim = data["train_static"].shape[1]
    horizon = data["y_train_model"].shape[1]

    print(f"method={args.method} device={device} dataset={data['dataset_path'].name}")
    print(f"train={len(data['train_seq'])} test={len(data['test_seq'])} "
          f"obs=({obs_steps},{obs_dim}) static={static_dim} horizon={horizon}")

    model = GainChunkUNet(
        obs_dim=obs_dim, static_dim=static_dim, horizon_steps=horizon,
        gain_dim=len(GAIN_COLS), base_filters=args.base_filters,
        cond_dim=args.condition_dim, time_embed_dim=args.time_embed_dim,
        dropout=args.dropout, norm=args.norm, condition_mode=args.condition_mode,
    ).to(device)
    print(f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    betas = cosine_beta_schedule(args.diffusion_steps)
    alpha_bars = torch.from_numpy(np.cumprod(1.0 - betas).astype(np.float32)).to(device)

    t0 = time.perf_counter()
    history, best_val = train(model, data, args, alpha_bars, device)
    train_sec = time.perf_counter() - t0
    print(f"train_time={train_sec:.1f}s best_val={best_val:.6f}")

    print("Evaluating step budgets...")
    model.eval()
    eval_df = evaluate(model, data, args, alpha_bars, device, horizon)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = args.run_label or args.method
    stem = f"torch_{args.method}_gain_chunk_{label}_{timestamp}"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    weights_path = MODEL_DIR / f"{stem}.pt"
    torch.save(model.state_dict(), weights_path)

    payload = {
        "model_type": f"torch_{args.method}_gain_chunk_unet",
        "framework": "pytorch",
        "weights_path": str(weights_path),
        "architecture": {
            "base_filters": args.base_filters,
            "condition_dim": args.condition_dim,
            "time_embed_dim": args.time_embed_dim,
            "dropout": args.dropout,
            "norm": args.norm,
            "condition_mode": args.condition_mode,
        },
        "obs_cols": list(DEFAULT_OBS_COLS),
        "static_feature_cols": list(data["static_feature_cols"]),
        "gain_cols": list(GAIN_COLS),
        "gain_bounds": GAIN_BOUNDS,
        "obs_steps": int(obs_steps),
        "obs_dim": int(obs_dim),
        "static_dim": int(static_dim),
        "horizon_steps": int(horizon),
        "diffusion_steps": int(args.diffusion_steps),
        "flow_solver": args.solver,
        "flow_sigma_min": args.sigma_min,
        "seq_scaler": data["seq_scaler"],
        "static_scaler": data["static_scaler"],
        "dataset_path": str(data["dataset_path"]),
        "best_val_loss": float(best_val),
        "train_seconds": float(train_sec),
        "args": vars(args),
        "eval": eval_df.to_dict(orient="records"),
    }
    joblib_path = MODEL_DIR / f"{stem}.joblib"
    joblib.dump(payload, joblib_path)

    eval_df.to_csv(SUMMARY_DIR / f"{stem}_step_sweep.csv", index=False)
    history.to_csv(SUMMARY_DIR / f"{stem}_history.csv", index=False)

    print(json.dumps({"model": str(joblib_path),
                      "sweep": str(SUMMARY_DIR / f"{stem}_step_sweep.csv")}, indent=2))


if __name__ == "__main__":
    main()
