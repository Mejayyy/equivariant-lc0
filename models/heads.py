"""Baseline LCZero heads.

Implements the three output heads of the BaselineLCZero model exactly per
CLAUDE.md "Architecture Quick-Reference":

- BaselinePolicyHead: Conv -> BN -> ReLU -> Conv(80 ch) -> BN -> ReLU
                     -> gather to (B, 1858) via the lc0 80-channel layout.
                     Returns RAW logits (no softmax).
- BaselineValueHead:  Conv(32) -> BN -> ReLU -> Conv(128, 8x8) -> BN -> ReLU
                     -> FC(128 -> 3) -> softmax.  Returns (B, 3) [W, D, L].
- BaselineMovesLeftHead: Conv(MLH_CH) -> BN -> ReLU -> Flatten
                       -> FC(MLH_CH*64 -> FC_SIZE) -> ReLU
                       -> FC(FC_SIZE -> 1) -> ReLU.

Each (Conv, BN) pair exposes a `.fold_bn()` that fuses the BN into the
preceding Conv and replaces the BN with `nn.Identity` — the network must
be in `eval()` mode for the fold to be mathematically equivalent (eval
uses the BN running stats; train would use batch stats which differ).
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .policy_map import POLICY_INDEX_SIZE, build_policy_gather_indices


# ---------------------------------------------------------------------------
# BN folding utility
# ---------------------------------------------------------------------------


def _fold_conv_bn_(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> None:
    """Fuse `bn` into `conv` IN PLACE (modifies `conv.weight` and `conv.bias`).

    After this call the (conv -> bn) sub-graph is mathematically equivalent
    to applying just `conv` with the new weights; the caller should replace
    `bn` with `nn.Identity()` so it is no longer applied.
    """
    if not isinstance(conv, nn.Conv2d) or not isinstance(bn, nn.BatchNorm2d):
        raise TypeError("fold expects (Conv2d, BatchNorm2d)")

    bn_var_eps = bn.running_var + bn.eps
    bn_std = torch.sqrt(bn_var_eps)
    factor = bn.weight / bn_std  # (C,)

    new_weight = conv.weight.data * factor.view(-1, 1, 1, 1)
    if conv.bias is None:
        new_bias = bn.bias.data - bn.running_mean * factor
        conv.bias = nn.Parameter(new_bias)
    else:
        new_bias = bn.bias.data + (conv.bias.data - bn.running_mean) * factor
        conv.bias.data.copy_(new_bias)

    conv.weight.data.copy_(new_weight)


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------


class BaselinePolicyHead(nn.Module):
    """LCZero AZ-policy-head: 80-channel conv stack + fixed gather to 1858."""

    def __init__(self, filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, 80, 3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(80)
        self.register_buffer("policy_gather", build_policy_gather_indices(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))   # (B, 80, 8, 8)
        flat = x.view(x.shape[0], -1)         # (B, 5120)
        return flat.index_select(dim=1, index=self.policy_gather)  # (B, 1858)

    def fold_bn(self) -> None:
        _fold_conv_bn_(self.conv1, self.bn1)
        self.bn1 = nn.Identity()
        _fold_conv_bn_(self.conv2, self.bn2)
        self.bn2 = nn.Identity()


class BaselineValueHead(nn.Module):
    """LCZero value head: Conv(32) -> Conv(128, 8x8) -> FC(3) -> softmax."""

    def __init__(self, filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, 32, 3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(32)
        # 8x8 kernel collapses spatial to 1x1 — equivalent to FC(32*64 -> 128)
        # but keeps a Conv2d so it folds with BN cleanly.
        self.conv2 = nn.Conv2d(32, 128, 8, padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))    # (B, 32, 8, 8)
        x = F.relu(self.bn2(self.conv2(x)))    # (B, 128, 1, 1)
        x = x.view(x.shape[0], -1)             # (B, 128)
        return F.softmax(self.fc(x), dim=1)    # (B, 3) — [W, D, L]

    def fold_bn(self) -> None:
        _fold_conv_bn_(self.conv1, self.bn1)
        self.bn1 = nn.Identity()
        _fold_conv_bn_(self.conv2, self.bn2)
        self.bn2 = nn.Identity()


class BaselineMovesLeftHead(nn.Module):
    """LCZero MLH head: small conv -> flatten -> FC stack -> ReLU output."""

    def __init__(self, filters: int, mlh_channels: int = 8, fc_size: int = 128):
        super().__init__()
        self.conv = nn.Conv2d(filters, mlh_channels, 3, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(mlh_channels)
        self.fc1 = nn.Linear(mlh_channels * 64, fc_size)
        self.fc2 = nn.Linear(fc_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))      # (B, mlh_channels, 8, 8)
        x = x.view(x.shape[0], -1)             # (B, mlh_channels*64)
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))             # (B, 1) — non-negative

    def fold_bn(self) -> None:
        _fold_conv_bn_(self.conv, self.bn)
        self.bn = nn.Identity()


__all__ = [
    "BaselinePolicyHead",
    "BaselineValueHead",
    "BaselineMovesLeftHead",
]
