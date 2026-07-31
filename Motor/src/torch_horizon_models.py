"""
PyTorch ports of the horizon-cost model family.

These consume the tabular dataset built by build_esp32_horizon_cost_dataset.py
(gain-sweep logs plus control logs), not the gain-chunk dataset. They are the
earlier generations of the study:

  cost        one scalar horizon cost per candidate gain; the server scores a
              candidate grid and picks the argmin
  multitask   the same, but predicting seven horizon metrics at once, so the
              server can select on any one of them
  direct      skips scoring entirely and regresses the gain triple directly

Layer widths mirror the Keras originals so a comparison reflects the modelling
choice. Huber loss is kept for the cost models: horizon_iae has a long right
tail and squared error lets a few bad candidates dominate the fit.
"""

import torch
import torch.nn as nn


def _block(in_dim, out_dim, dropout, norm=True):
    layers = [nn.Linear(in_dim, out_dim), nn.ReLU()]
    if norm:
        layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.Dropout(dropout))
    return layers


class HorizonCostMlp(nn.Module):
    """128-128-64-32 -> 1, matching train_esp32_horizon_cost_mlp.py."""

    def __init__(self, input_dim, dropout=0.05):
        super().__init__()
        layers = []
        layers += _block(input_dim, 128, dropout)
        layers += _block(128, 128, dropout)
        layers += _block(128, 64, dropout, norm=False)
        layers += _block(64, 32, dropout, norm=False)
        layers += [nn.Linear(32, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class HorizonMultiTaskMlp(nn.Module):
    """160-160-96-48 -> len(targets), matching train_esp32_multitask_mlp.py."""

    def __init__(self, input_dim, output_dim, dropout=0.05):
        super().__init__()
        layers = []
        layers += _block(input_dim, 160, dropout)
        layers += _block(160, 160, dropout)
        layers += _block(160, 96, dropout, norm=False)
        layers += _block(96, 48, dropout, norm=False)
        layers += [nn.Linear(48, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DirectPolicyMlp(nn.Module):
    """
    Configurable trunk -> 3 gains.

    output_mode 'residual_db' predicts a signed correction to the DB gain and
    so ends in tanh; anything else predicts the normalised gain directly and
    ends in sigmoid. Matches train_esp32_direct_policy_mlp.py.
    """

    def __init__(self, input_dim, hidden_layers=(256, 192, 128), dropout=0.05,
                 output_mode="normalized"):
        super().__init__()
        layers = []
        prev = input_dim
        widths = list(hidden_layers)
        for i, width in enumerate(widths):
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            if i < len(widths) - 1:
                layers.append(nn.BatchNorm1d(width))
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, 3))
        self.net = nn.Sequential(*layers)
        self.output_mode = output_mode

    def forward(self, x):
        out = self.net(x)
        if self.output_mode == "residual_db":
            return torch.tanh(out)
        return torch.sigmoid(out)


def parse_hidden_layers(text):
    if not text:
        return (256, 192, 128)
    return tuple(int(v) for v in str(text).split(",") if str(v).strip())
