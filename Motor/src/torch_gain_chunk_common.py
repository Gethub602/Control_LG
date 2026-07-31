"""
Shared PyTorch backbone and data pipeline for gain-chunk generative models.

The repository's original models are TensorFlow/Keras 2. That stack cannot run on
this machine's GPU (RTX 5070 Ti, compute capability 12.0): TF ships no sm_120
kernel binaries and its bundled PTX fails to JIT (CUDA_ERROR_INVALID_PTX). This
module mirrors the same architecture in PyTorch, which has native sm_120 support,
so diffusion and flow matching can be compared on the GPU.

The data contract is unchanged: the same labeled chunk CSVs, the same observation
and static feature columns, the same gain bounds and trajectory-grouped split as
train_diffusion_gain_chunk_baselines.py.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))
sys.path.append(str(CURRENT_DIR))

from train_diffusion_gain_chunk_baselines import (  # noqa: E402
    DEFAULT_OBS_COLS,
    GAIN_BOUNDS,
    GAIN_COLS,
    latest_label_path,
    load_chunk_labels,
    make_arrays,
    normalize_gain_sequence,
    denormalize_gain_sequence,
    scale_inputs,
    split_by_trajectory,
    static_cols,
)


def get_device(prefer_cuda: bool = True):
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def gain_to_model_space(y_gain):
    """Gain sequence -> [-1, 1], identical to the TF trainer's convention."""
    return (2.0 * normalize_gain_sequence(y_gain) - 1.0).astype(np.float32)


def model_to_gain_space(y_model):
    y01 = np.clip((np.asarray(y_model) + 1.0) / 2.0, 0.0, 1.0)
    return denormalize_gain_sequence(y01.astype(np.float32))


def newest_label_path(profile: str) -> Path:
    """
    Most recently written label file for `profile`.

    The repository's latest_label_path sorts paths lexicographically, and since
    the run label sits before the timestamp in the filename, an older run whose
    label happens to sort later wins. Sorting by mtime picks the file that was
    actually produced last.
    """
    root = MOTOR_DIR / "data" / "processed" / "diffusion_gain_chunk_db"
    paths = sorted(root.glob(f"chunk_labels_*_{profile}_*.csv"),
                   key=lambda p: p.stat().st_mtime)
    if not paths:
        raise FileNotFoundError(f"No label file found for profile={profile}")
    return paths[-1]


def prepare_data(dataset, profile, quality, max_rows, test_size, seed):
    dataset_path = Path(dataset) if dataset else newest_label_path(profile)
    if not dataset_path.is_absolute():
        dataset_path = MOTOR_DIR / dataset_path
    df = load_chunk_labels(dataset_path, quality, max_rows, seed)
    train_df, test_df = split_by_trajectory(df, test_size, seed)
    static_feature_cols = static_cols(df)

    tr_seq, tr_static, y_tr = make_arrays(train_df, DEFAULT_OBS_COLS, static_feature_cols)
    te_seq, te_static, y_te = make_arrays(test_df, DEFAULT_OBS_COLS, static_feature_cols)
    tr_seq, te_seq, tr_static, te_static, seq_scaler, static_scaler = scale_inputs(
        tr_seq, te_seq, tr_static, te_static
    )
    return {
        "dataset_path": dataset_path,
        "train_seq": tr_seq.astype(np.float32),
        "test_seq": te_seq.astype(np.float32),
        "train_static": tr_static.astype(np.float32),
        "test_static": te_static.astype(np.float32),
        "y_train_gain": y_tr,
        "y_test_gain": y_te,
        "y_train_model": gain_to_model_space(y_tr),
        "y_test_model": gain_to_model_space(y_te),
        "seq_scaler": seq_scaler,
        "static_scaler": static_scaler,
        "static_feature_cols": static_feature_cols,
    }


def sinusoidal_embedding(t, dim: int):
    """t is a float tensor of shape (B,) carrying a continuous or integer step."""
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(
            math.log(1.0), math.log(10000.0), half, device=t.device, dtype=torch.float32
        )
        * -1.0
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class FiLMResBlock(nn.Module):
    """Conv1d residual block whose normalisation is modulated by the condition."""

    def __init__(self, in_ch, out_ch, cond_dim, kernel_size=3, dropout=0.05, norm="batch"):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.norm1 = (
            nn.BatchNorm1d(out_ch) if norm == "batch" else nn.GroupNorm(1, out_ch)
        )
        self.norm2 = (
            nn.BatchNorm1d(out_ch) if norm == "batch" else nn.GroupNorm(1, out_ch)
        )
        self.film = nn.Linear(cond_dim, out_ch * 2)
        self.dropout = nn.Dropout(dropout)
        self.skip = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, cond):
        residual = self.skip(x)
        h = self.norm1(self.conv1(x))
        scale, shift = self.film(cond).chunk(2, dim=1)
        h = h * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        h = F.silu(h)
        h = self.dropout(h)
        h = F.silu(self.norm2(self.conv2(h)))
        return h + residual


class ConditionEncoder(nn.Module):
    def __init__(self, obs_dim, static_dim, cond_dim, dropout, mode="avg"):
        super().__init__()
        self.mode = mode
        self.obs_proj = nn.Conv1d(obs_dim, cond_dim, 3, padding=1)
        if mode == "gru":
            self.gru = nn.GRU(cond_dim, cond_dim, batch_first=True)
        self.static_proj = nn.Linear(static_dim, cond_dim)
        self.out = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim), nn.SiLU(), nn.Dropout(dropout)
        )

    def forward(self, obs, static):
        # obs: (B, T, C) -> conv expects (B, C, T)
        h = F.silu(self.obs_proj(obs.transpose(1, 2)))
        if self.mode == "gru":
            h, _ = self.gru(h.transpose(1, 2))
            h = h[:, -1, :]
        else:
            h = h.mean(dim=2)
        s = F.silu(self.static_proj(static))
        return self.out(torch.cat([h, s], dim=1))


class GainChunkUNet(nn.Module):
    """
    1D conditional U-Net over the 20-step gain chunk.

    Predicts noise (diffusion) or velocity (flow matching); the head is the same,
    only the training target differs.
    """

    def __init__(
        self,
        obs_dim,
        static_dim,
        horizon_steps,
        gain_dim=3,
        base_filters=64,
        cond_dim=128,
        time_embed_dim=64,
        dropout=0.05,
        norm="batch",
        condition_mode="avg",
    ):
        super().__init__()
        self.horizon_steps = horizon_steps
        self.gain_dim = gain_dim
        self.time_embed_dim = time_embed_dim

        self.cond_encoder = ConditionEncoder(
            obs_dim, static_dim, cond_dim, dropout, condition_mode
        )
        self.time_mlp = nn.Sequential(nn.Linear(time_embed_dim, cond_dim), nn.SiLU())
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim * 2, cond_dim), nn.SiLU())

        f = base_filters
        self.in_proj = nn.Conv1d(gain_dim, f, 3, padding=1)
        self.down1 = FiLMResBlock(f, f, cond_dim, 3, dropout, norm)
        self.down2 = FiLMResBlock(f, f * 2, cond_dim, 3, dropout, norm)
        self.mid1 = FiLMResBlock(f * 2, f * 4, cond_dim, 3, dropout, norm)
        self.mid2 = FiLMResBlock(f * 4, f * 4, cond_dim, 3, dropout, norm)
        self.up2 = FiLMResBlock(f * 4 + f * 2, f * 2, cond_dim, 3, dropout, norm)
        self.up1 = FiLMResBlock(f * 2 + f, f, cond_dim, 3, dropout, norm)
        self.out_proj = nn.Conv1d(f, gain_dim, 1)

    def forward(self, y, t, obs, static):
        """y: (B, H, G)  t: (B,)  obs: (B, T, C)  static: (B, S)"""
        cond_state = self.cond_encoder(obs, static)
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_embed_dim))
        cond = self.cond_mlp(torch.cat([cond_state, t_emb], dim=1))

        x = self.in_proj(y.transpose(1, 2))
        x1 = self.down1(x, cond)
        d1 = F.avg_pool1d(x1, 2)
        x2 = self.down2(d1, cond)
        d2 = F.avg_pool1d(x2, 2)

        m = self.mid2(self.mid1(d2, cond), cond)

        u2 = F.interpolate(m, size=x2.shape[-1], mode="nearest")
        u2 = self.up2(torch.cat([u2, x2], dim=1), cond)
        u1 = F.interpolate(u2, size=x1.shape[-1], mode="nearest")
        u1 = self.up1(torch.cat([u1, x1], dim=1), cond)

        return self.out_proj(u1).transpose(1, 2)


def cosine_beta_schedule(num_steps: int, s: float = 0.008):
    steps = np.arange(num_steps + 1, dtype=np.float64)
    x = steps / num_steps
    ac = np.cos(((x + s) / (1 + s)) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1.0 - (ac[1:] / ac[:-1])
    return np.clip(betas, 1e-5, 0.999).astype(np.float32)


def chunk_accuracy(sample_gain, y_true):
    """sample_gain: (N, S, H, G) candidates per condition."""
    mean_gain = sample_gain.mean(axis=1)
    all_mae = np.mean(np.abs(sample_gain - y_true[:, None, :, :]), axis=(2, 3))
    best = sample_gain[np.arange(len(sample_gain)), np.argmin(all_mae, axis=1)]
    return {
        "sample_mean_mae": float(np.mean(np.abs(mean_gain - y_true))),
        "best_of_n_mae": float(np.mean(np.abs(best - y_true))),
        "kp_mae": float(np.mean(np.abs(mean_gain[:, :, 0] - y_true[:, :, 0]))),
        "ki_mae": float(np.mean(np.abs(mean_gain[:, :, 1] - y_true[:, :, 1]))),
        "kd_mae": float(np.mean(np.abs(mean_gain[:, :, 2] - y_true[:, :, 2]))),
        "chunk_rmse": float(np.sqrt(np.mean((mean_gain - y_true) ** 2))),
        "sample_diversity_std": float(np.mean(np.std(sample_gain, axis=1))),
    }
