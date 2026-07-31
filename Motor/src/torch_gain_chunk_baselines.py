"""
PyTorch ports of the supervised gain-chunk baselines.

These mirror the four architectures in train_diffusion_gain_chunk_baselines.py
(mlp, cnn, cnn_residual, cnn_attention) layer for layer, so that a comparison
against the diffusion and flow-matching generators reflects the modelling
choice rather than a framework difference.

All of them are deterministic regressors: given the observation window and the
static state features they emit one 20x3 gain chunk in [0, 1], trained with MSE
against the normalised label chunk. That is the key contrast with the
generative models, which produce a distribution and can offer several
candidates for a downstream selector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MlpChunk(nn.Module):
    """Flatten the window, three dense blocks, then shape into the chunk."""

    def __init__(self, obs_steps, obs_dim, static_dim, horizon_steps, gain_dim=3,
                 dropout=0.05):
        super().__init__()
        self.horizon_steps = horizon_steps
        in_dim = obs_steps * obs_dim + static_dim
        blocks = []
        prev = in_dim
        for units in (256, 192, 128):
            blocks += [
                nn.Linear(prev, units),
                nn.ReLU(),
                nn.BatchNorm1d(units),
                nn.Dropout(dropout),
            ]
            prev = units
        self.trunk = nn.Sequential(*blocks)
        self.expand = nn.Sequential(nn.Linear(prev, horizon_steps * 96), nn.ReLU())
        self.head = nn.Sequential(
            nn.Conv1d(96, 96, 3, padding=1), nn.ReLU(), nn.Conv1d(96, gain_dim, 1)
        )

    def forward(self, obs, static):
        x = torch.cat([obs.flatten(1), static], dim=1)
        x = self.trunk(x)
        x = self.expand(x).view(-1, self.horizon_steps, 96).transpose(1, 2)
        return torch.sigmoid(self.head(x)).transpose(1, 2)


def _causal_conv(in_ch, out_ch, k=3):
    # left-pad so the convolution cannot see the future within the window
    return nn.Sequential(nn.ConstantPad1d((k - 1, 0), 0.0), nn.Conv1d(in_ch, out_ch, k))


class CnnChunk(nn.Module):
    def __init__(self, obs_steps, obs_dim, static_dim, horizon_steps, gain_dim=3,
                 dropout=0.05):
        super().__init__()
        self.horizon_steps = horizon_steps
        self.c1 = _causal_conv(obs_dim, 64)
        self.b1 = nn.BatchNorm1d(64)
        self.c2 = _causal_conv(64, 96)
        self.b2 = nn.BatchNorm1d(96)
        self.fc1 = nn.Linear(96 + static_dim, 192)
        self.drop = nn.Dropout(dropout)
        self.expand = nn.Linear(192, horizon_steps * 96)
        self.head = nn.Sequential(
            nn.Conv1d(96, 96, 3, padding=1), nn.ReLU(),
            nn.Conv1d(96, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, gain_dim, 1),
        )

    def forward(self, obs, static):
        x = obs.transpose(1, 2)
        x = self.b1(F.relu(self.c1(x)))
        x = self.b2(F.relu(self.c2(x)))
        x = x.mean(dim=2)
        x = torch.cat([x, static], dim=1)
        x = self.drop(F.relu(self.fc1(x)))
        x = F.relu(self.expand(x)).view(-1, self.horizon_steps, 96).transpose(1, 2)
        return torch.sigmoid(self.head(x)).transpose(1, 2)


class ResidualBlock1d(nn.Module):
    def __init__(self, in_ch, filters, k=3, dropout=0.05):
        super().__init__()
        self.c1 = nn.Conv1d(in_ch, filters, k, padding=k // 2)
        self.b1 = nn.BatchNorm1d(filters)
        self.drop = nn.Dropout(dropout)
        self.c2 = nn.Conv1d(filters, filters, k, padding=k // 2)
        self.b2 = nn.BatchNorm1d(filters)
        self.skip = nn.Conv1d(in_ch, filters, 1) if in_ch != filters else nn.Identity()

    def forward(self, x):
        s = self.skip(x)
        h = self.drop(self.b1(F.relu(self.c1(x))))
        h = self.b2(self.c2(h))
        return F.relu(h + s)


class CnnResidualChunk(nn.Module):
    def __init__(self, obs_steps, obs_dim, static_dim, horizon_steps, gain_dim=3,
                 dropout=0.05):
        super().__init__()
        self.horizon_steps = horizon_steps
        self.stem = _causal_conv(obs_dim, 64)
        self.r1 = ResidualBlock1d(64, 64, 3, dropout)
        self.r2 = ResidualBlock1d(64, 96, 3, dropout)
        self.fc1 = nn.Linear(96 + static_dim, 224)
        self.bn1 = nn.BatchNorm1d(224)
        self.drop = nn.Dropout(dropout)
        self.expand = nn.Linear(224, horizon_steps * 96)
        self.r3 = ResidualBlock1d(96, 96, 3, dropout)
        self.r4 = ResidualBlock1d(96, 64, 3, dropout)
        self.out = nn.Conv1d(64, gain_dim, 1)

    def forward(self, obs, static):
        x = F.relu(self.stem(obs.transpose(1, 2)))
        x = self.r2(self.r1(x)).mean(dim=2)
        x = torch.cat([x, static], dim=1)
        x = self.drop(self.bn1(F.relu(self.fc1(x))))
        x = F.relu(self.expand(x)).view(-1, self.horizon_steps, 96).transpose(1, 2)
        x = self.r4(self.r3(x))
        return torch.sigmoid(self.out(x)).transpose(1, 2)


class _AttnBlock(nn.Module):
    """Self-attention + feed-forward, both residual, matching the Keras version."""

    def __init__(self, dim, heads=4, key_dim=24, ff=128, dropout=0.05):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.n1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ff, dim)
        )
        self.n2 = nn.LayerNorm(dim)

    def forward(self, x):
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.n1(x + a)
        return self.n2(x + self.ff(x))


class CnnAttentionChunk(nn.Module):
    def __init__(self, obs_steps, obs_dim, static_dim, horizon_steps, gain_dim=3,
                 dropout=0.05):
        super().__init__()
        self.horizon_steps = horizon_steps
        self.c1 = _causal_conv(obs_dim, 64)
        self.b1 = nn.BatchNorm1d(64)
        self.c2 = _causal_conv(64, 96)
        self.b2 = nn.BatchNorm1d(96)
        self.attn_enc = _AttnBlock(96, dropout=dropout)
        self.fc1 = nn.Linear(96 + static_dim, 224)
        self.drop = nn.Dropout(dropout)
        self.expand = nn.Linear(224, horizon_steps * 96)
        self.attn_dec = _AttnBlock(96, dropout=dropout)
        self.out = nn.Conv1d(96, gain_dim, 1)

    def forward(self, obs, static):
        x = obs.transpose(1, 2)
        x = self.b1(F.relu(self.c1(x)))
        x = self.b2(F.relu(self.c2(x)))
        x = self.attn_enc(x.transpose(1, 2))
        x = x.mean(dim=1)
        x = torch.cat([x, static], dim=1)
        x = self.drop(F.relu(self.fc1(x)))
        x = F.relu(self.expand(x)).view(-1, self.horizon_steps, 96)
        x = self.attn_dec(x)
        return torch.sigmoid(self.out(x.transpose(1, 2))).transpose(1, 2)


BASELINE_MODELS = {
    "mlp": MlpChunk,
    "cnn": CnnChunk,
    "cnn_residual": CnnResidualChunk,
    "cnn_attention": CnnAttentionChunk,
}


def build_baseline(model_type, obs_steps, obs_dim, static_dim, horizon_steps,
                   gain_dim=3, dropout=0.05):
    if model_type not in BASELINE_MODELS:
        raise ValueError(
            f"Unknown baseline {model_type!r}. Choose from {sorted(BASELINE_MODELS)}."
        )
    return BASELINE_MODELS[model_type](
        obs_steps, obs_dim, static_dim, horizon_steps, gain_dim, dropout
    )
