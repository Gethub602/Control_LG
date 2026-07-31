"""
Train the horizon-cost model family on the ESP32 horizon dataset.

Covers the generations that came before the gain-chunk work:

  rf          RandomForest regression of the scalar horizon cost (sklearn; kept
              as-is since there is no framework question for a forest)
  mlp_cost    the same target, as a PyTorch MLP
  multitask   seven horizon metrics at once
  direct      regress the gain triple directly, skipping candidate scoring

The dataset is the one build_esp32_horizon_cost_dataset.py writes from gain
sweep logs plus control logs, which is a different table from the gain-chunk
labels. Splits are grouped so rows from one trajectory cannot straddle the
train/test boundary.
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))
sys.path.append(str(CURRENT_DIR))

from config import MODEL_DIR, PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402
from torch_horizon_models import (  # noqa: E402
    DirectPolicyMlp,
    HorizonCostMlp,
    HorizonMultiTaskMlp,
    parse_hidden_layers,
)

SUMMARY_DIR = RESULTS_DIR / "summary"

COST_TARGET = "horizon_cost"
MULTITASK_TARGETS = [
    "horizon_iae",
    "horizon_overshoot_ratio",
    "horizon_saturation_ratio",
    "horizon_near_saturation_ratio",
    "horizon_mean_pwm",
    "horizon_pwm_variation",
    "horizon_max_abs_error",
]
GAIN_TARGETS = ["kp", "ki", "kd"]
GAIN_BOUNDS = {"kp": (0.55, 1.45), "ki": (0.70, 2.50), "kd": (0.00, 0.12)}


def parse_args():
    p = argparse.ArgumentParser(description="Train horizon-cost family models.")
    p.add_argument("--model", required=True,
                   choices=["rf", "mlp_cost", "multitask", "direct"])
    p.add_argument("--dataset", default="")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden-layers", default="256,192,128")
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-depth", type=int, default=0, help="0 means unlimited.")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--run-label", default="real")
    return p.parse_args()


def load_dataset(path_arg):
    path = Path(path_arg) if path_arg else (
        PROCESSED_DATA_DIR / "esp32_horizon_cost_dataset_latest.csv"
    )
    if not path.is_absolute():
        path = MOTOR_DIR / path
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run build_esp32_horizon_cost_dataset.py first "
            "(which needs gain-sweep logs from esp32_gain_sweep.py)."
        )
    return pd.read_csv(path), path


def feature_columns(df, targets):
    """Numeric columns that are not targets and not identifiers."""
    exclude = set(targets) | {COST_TARGET} | set(MULTITASK_TARGETS)
    exclude |= {c for c in df.columns if c.endswith("_id") or c in {
        "trajectory_id", "run_idx", "case_idx", "step", "mode", "backend",
        "scenario_type", "gain_profile_type", "abort_reason", "label_quality",
    }}
    return [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


def split(df, test_size, seed):
    """Group by trajectory when the column exists, otherwise fall back to random."""
    for key in ("trajectory_id", "case_id", "run_idx"):
        if key in df.columns and df[key].nunique() > 3:
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                         random_state=seed)
            tr, te = next(splitter.split(df, groups=df[key]))
            return df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
    return train_test_split(df, test_size=test_size, random_state=seed)


def normalize_gains(y):
    out = np.empty_like(y, dtype=np.float32)
    for i, col in enumerate(GAIN_TARGETS):
        lo, hi = GAIN_BOUNDS[col]
        out[:, i] = (y[:, i] - lo) / max(hi - lo, 1e-9)
    return np.clip(out, 0.0, 1.0)


def denormalize_gains(y):
    out = np.empty_like(y, dtype=np.float32)
    for i, col in enumerate(GAIN_TARGETS):
        lo, hi = GAIN_BOUNDS[col]
        out[:, i] = lo + y[:, i] * (hi - lo)
    return out


def train_torch(model, xtr, ytr, xte, yte, args, device, loss_fn):
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    tr = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr)),
        batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    va = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(xte), torch.from_numpy(yte)),
        batch_size=args.batch_size,
    )
    hist = {"loss": [], "val_loss": []}
    best, best_state, wait = np.inf, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tl = []
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tl.append(loss.item())
        model.eval()
        vl = []
        with torch.no_grad():
            for xb, yb in va:
                xb, yb = xb.to(device), yb.to(device)
                vl.append(loss_fn(model(xb), yb).item())
        l, v = float(np.mean(tl)), float(np.mean(vl))
        hist["loss"].append(l)
        hist["val_loss"].append(v)
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}: loss={l:.6f}, val_loss={v:.6f}", flush=True)
        if v < best - 1e-7:
            best, wait = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {epoch}. Best val_loss={best:.6f}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return pd.DataFrame(hist), best


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")

    df, dataset_path = load_dataset(args.dataset)
    print(f"model={args.model} device={device} dataset={dataset_path.name} rows={len(df)}")

    if args.model == "direct":
        targets = GAIN_TARGETS
    elif args.model == "multitask":
        targets = [c for c in MULTITASK_TARGETS if c in df.columns]
    else:
        targets = [COST_TARGET]

    missing = [t for t in targets if t not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing target columns: {missing}")

    feats = feature_columns(df, targets)
    if args.model == "direct":
        # a direct policy must not be handed the gains it is meant to predict
        feats = [c for c in feats if c not in GAIN_TARGETS]
    print(f"features={len(feats)} targets={targets}")

    train_df, test_df = split(df.dropna(subset=feats + targets), args.test_size, args.seed)
    xs = StandardScaler().fit(train_df[feats].to_numpy(np.float32))
    xtr = xs.transform(train_df[feats].to_numpy(np.float32)).astype(np.float32)
    xte = xs.transform(test_df[feats].to_numpy(np.float32)).astype(np.float32)
    ytr_raw = train_df[targets].to_numpy(np.float32)
    yte_raw = test_df[targets].to_numpy(np.float32)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"torch_horizon_{args.model}_{args.run_label}_{timestamp}"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    payload = {
        "model_type": f"horizon_{args.model}",
        "feature_cols": feats,
        "target_cols": targets,
        "feature_scaler": xs,
        "dataset_path": str(dataset_path),
        "args": vars(args),
    }
    hist = None

    if args.model == "rf":
        rf = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=None if args.max_depth == 0 else args.max_depth,
            random_state=args.seed,
            n_jobs=-1,
        )
        rf.fit(xtr, ytr_raw.ravel())
        pred = rf.predict(xte).reshape(-1, 1)
        payload["framework"] = "sklearn"
        payload["model"] = rf
    else:
        if args.model == "direct":
            ytr = normalize_gains(ytr_raw)
            yte = normalize_gains(yte_raw)
            model = DirectPolicyMlp(
                len(feats), parse_hidden_layers(args.hidden_layers), args.dropout
            ).to(device)
            loss_fn = F.mse_loss
        else:
            ys = StandardScaler().fit(ytr_raw)
            ytr = ys.transform(ytr_raw).astype(np.float32)
            yte = ys.transform(yte_raw).astype(np.float32)
            payload["target_scaler"] = ys
            model = (
                HorizonCostMlp(len(feats), args.dropout)
                if args.model == "mlp_cost"
                else HorizonMultiTaskMlp(len(feats), len(targets), args.dropout)
            ).to(device)
            # Huber: horizon_iae has a long right tail, and squared error would
            # let a handful of bad candidates dominate the fit.
            loss_fn = torch.nn.HuberLoss()

        print(f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)
        hist, best = train_torch(model, xtr, ytr, xte, yte, args, device, loss_fn)
        model.eval()
        with torch.no_grad():
            pred_scaled = model(torch.from_numpy(xte).to(device)).cpu().numpy()
        pred = (
            denormalize_gains(np.clip(pred_scaled, 0.0, 1.0))
            if args.model == "direct"
            else payload["target_scaler"].inverse_transform(pred_scaled)
        )
        weights_path = MODEL_DIR / f"{stem}.pt"
        torch.save(model.state_dict(), weights_path)
        payload["framework"] = "pytorch"
        payload["weights_path"] = str(weights_path)
        payload["best_val_loss"] = float(best)

    train_sec = time.perf_counter() - t0
    rows = []
    for i, col in enumerate(targets):
        rows.append({
            "model": args.model,
            "target": col,
            "mae": float(mean_absolute_error(yte_raw[:, i], pred[:, i])),
            "r2": float(r2_score(yte_raw[:, i], pred[:, i])),
        })
    metrics = pd.DataFrame(rows)
    payload["train_seconds"] = float(train_sec)
    payload["metrics"] = metrics.to_dict(orient="records")
    joblib.dump(payload, MODEL_DIR / f"{stem}.joblib")
    metrics.to_csv(SUMMARY_DIR / f"{stem}_metrics.csv", index=False)
    if hist is not None:
        hist.to_csv(SUMMARY_DIR / f"{stem}_history.csv", index=False)

    print()
    print(metrics.to_string(index=False))
    print(f"train_time={train_sec:.1f}s")
    print(json.dumps({"model": str(MODEL_DIR / f'{stem}.joblib')}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
