"""BaselineLCZero — faithful SE-residual reimplementation of the LCZero
network (no equivariance). Body + 3 heads share the model interface

    policy_logits, value_wdl, moves_left = model(x)
    # x: (B, 34, 8, 8)
    # policy_logits: (B, 1858)  raw logits
    # value_wdl:     (B, 3)     softmax [W, D, L]
    # moves_left:    (B, 1)     ReLU output

Body structure:

    Input:  Conv2d(34, FILTERS, 3x3, padding=1, bias=True) -> BN -> ReLU
    Tower:  BLOCKS x ResBlock(FILTERS, SE_CHANNELS)
        ResBlock:
            Conv2d(F, F, 3x3, padding=1, bias=True) -> BN -> ReLU
            Conv2d(F, F, 3x3, padding=1, bias=True) -> BN
            SELayer(F, SE_CHANNELS):
                GAP -> FC(F -> SE_C) -> ReLU -> FC(SE_C -> 2*F)
                split into (W, B); out = sigmoid(W) * pre_SE + B
            skip-add input -> ReLU

The baseline uses standard Conv2d / BatchNorm2d primitives. BN is foldable
post-training: `model.fold_bn()` fuses each (Conv2d, BatchNorm2d) pair and
replaces the BN with `nn.Identity`. Eval mode only.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import (
    BaselineMovesLeftHead,
    BaselinePolicyHead,
    BaselineValueHead,
    _fold_conv_bn_,
)


# ---------------------------------------------------------------------------
# SE block + residual block
# ---------------------------------------------------------------------------


class SELayer(nn.Module):
    """Squeeze-Excitation a la LCZero: spatial GAP -> 2 FC layers -> split
    output into a sigmoid-gating channel and a per-channel bias.

    out = sigmoid(W) * pre_SE + B,   where (W, B) = split(FC2(FC1(GAP(x))))
    """

    def __init__(self, channels: int, se_channels: int):
        super().__init__()
        self.channels = channels
        self.fc1 = nn.Linear(channels, se_channels)
        self.fc2 = nn.Linear(se_channels, 2 * channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) — the "pre-SE" tensor that gets gated.
        b, c, _, _ = x.shape
        s = x.mean(dim=[2, 3])                  # (B, C)
        s = F.relu(self.fc1(s))                 # (B, SE_C)
        s = self.fc2(s)                         # (B, 2C)
        gate, bias = s[:, :c], s[:, c:]
        gate = torch.sigmoid(gate).view(b, c, 1, 1)
        bias = bias.view(b, c, 1, 1)
        return gate * x + bias


class ResBlock(nn.Module):
    def __init__(self, channels: int, se_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SELayer(channels, se_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + residual
        return F.relu(out)

    def fold_bn(self) -> None:
        _fold_conv_bn_(self.conv1, self.bn1)
        self.bn1 = nn.Identity()
        _fold_conv_bn_(self.conv2, self.bn2)
        self.bn2 = nn.Identity()


# ---------------------------------------------------------------------------
# Body + full model
# ---------------------------------------------------------------------------


class BaselineBody(nn.Module):
    def __init__(self, blocks: int, filters: int, se_channels: int):
        super().__init__()
        self.input_conv = nn.Conv2d(34, filters, 3, padding=1, bias=True)
        self.input_bn = nn.BatchNorm2d(filters)
        self.blocks = nn.ModuleList(
            [ResBlock(filters, se_channels) for _ in range(blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.blocks:
            x = block(x)
        return x  # (B, FILTERS, 8, 8)

    def fold_bn(self) -> None:
        _fold_conv_bn_(self.input_conv, self.input_bn)
        self.input_bn = nn.Identity()
        for block in self.blocks:
            block.fold_bn()


class BaselineLCZero(nn.Module):
    """Faithful LCZero SE-residual model — body + 3 heads.

    Hyper-params:
        blocks:        residual tower depth.
        filters:       channels in the body.
        se_channels:   bottleneck width inside SE blocks (LCZero uses 32).
        mlh_channels:  channel count in the moves-left head's first conv.
        fc_size:       hidden width in the moves-left head's FC stack.
    """

    def __init__(
        self,
        blocks: int = 10,
        filters: int = 128,
        se_channels: int = 32,
        mlh_channels: int = 8,
        fc_size: int = 128,
    ):
        super().__init__()
        self.body = BaselineBody(blocks, filters, se_channels)
        self.policy_head = BaselinePolicyHead(filters)
        self.value_head = BaselineValueHead(filters)
        self.moves_left_head = BaselineMovesLeftHead(filters, mlh_channels, fc_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.shape[1:] != (34, 8, 8):
            raise ValueError(f"BaselineLCZero expects (B, 34, 8, 8), got {tuple(x.shape)}")
        h = self.body(x)
        return self.policy_head(h), self.value_head(h), self.moves_left_head(h)

    def fold_bn(self) -> None:
        """Fuse every (Conv2d, BatchNorm2d) pair in the model. Call only in
        eval mode — folding mathematically reproduces the eval-mode forward
        but NOT the train-mode forward (which uses batch stats)."""
        if self.training:
            raise RuntimeError("fold_bn() must be called in eval() mode")
        self.body.fold_bn()
        self.policy_head.fold_bn()
        self.value_head.fold_bn()
        self.moves_left_head.fold_bn()


__all__ = [
    "BaselineLCZero",
    "BaselineBody",
    "ResBlock",
    "SELayer",
]
