"""LCZero output heads — both baseline and Phase-8 equivariant variants.

Baseline (Model 1) heads (no equivariance, faithful LCZero re-implementation):
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

Equivariant (Models 2/3) heads — Phase 8:
- EquivariantPolicyHead: 2x [GConv -> GBN -> ReLU], sum over group dim,
                        orbit-based gather (built from
                        build_orbit_policy_gather) to (B, 1858).
                        Equivariant under build_policy_permutation.
                        See policy_map.py for the orbit-gather rationale —
                        the lc0 AZ-style 80-channel layout is NOT
                        sigma_h/sigma_c-equivariant for diagonal moves, so
                        we use a different cell assignment.
- EquivariantValueHead: 2x [GConv -> GBN -> ReLU] (last 8x8 collapses
                        spatial), squeeze -> (B, 128, |G|), build
                        sigma_h-invariant W/L/D feature combinations from
                        the group axis, run W and L through a SHARED
                        nn.Linear (one instance applied twice — critical
                        for sigma_c-equivariance), softmax to (B, 3).
- InvariantMovesLeftHead: GConv -> GBN -> ReLU, sum over group, mean over
                          spatial, FC stack -> (B, 1) ReLU. Fully invariant.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from symmetry.group import GCHESS_GROUP, GChess, GroupDef

from .gcnn import GROUP_SIZE, GBatchNorm, GConv, GSELayer
from .policy_map import (
    POLICY_INDEX_SIZE,
    POLICY_NUM_CHANNELS,
    build_orbit_policy_gather,
    build_policy_gather_indices,
    build_sgchess_orbit_policy_gather,
)


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


# ---------------------------------------------------------------------------
# Equivariant (Phase 8) heads
# ---------------------------------------------------------------------------


class EquivariantPolicyHead(nn.Module):
    """G-equivariant policy head producing (B, 1858) logits.

    Pipeline:
        GConv(F, F)        -> GBN -> ReLU
        GConv(F, num_chan) -> GBN -> ReLU       # num_chan default 80
        Sum over group dim    -> (B, num_chan, 8, 8)
        Orbit-based gather    -> (B, 1858, 2)
        Mean over the inner-2 axis -> (B, 1858) raw logits

    Equivariance: under sigma in the chosen group, the body output
    transforms as (spatial_flip + regular_rep on group axis). Sum over
    group makes the (num_chan, 8, 8) tensor sigma-equivariant on its
    spatial axis only. The orbit gather then permutes the 1858-vector
    exactly by ``build_policy_permutation(sigma)``. See
    ``models/policy_map.py`` for the derivation.

    Two gather flavours, one per supported group:

    * ``GCHESS_GROUP`` -> ``build_orbit_policy_gather``: Klein-4 orbits
      placed on 2x2 (rank-pair, file-pair) blocks per channel.
    * ``SGCHESS_GROUP`` -> ``build_sgchess_orbit_policy_gather``: Z/2
      orbits placed on rank-pair cells per channel.
    """

    POLICY_NUM_CHANNELS = POLICY_NUM_CHANNELS  # 80

    def __init__(self, filters: int, group: GroupDef = GCHESS_GROUP):
        super().__init__()
        self.group = group
        self.filters = filters
        self.conv1 = GConv(filters, filters, 3, padding=1, bias=True, group=group)
        self.bn1 = GBatchNorm(filters, group=group)
        self.conv2 = GConv(
            filters, self.POLICY_NUM_CHANNELS, 3, padding=1, bias=True, group=group
        )
        self.bn2 = GBatchNorm(self.POLICY_NUM_CHANNELS, group=group)
        # Non-trainable buffer; persistent=False keeps it out of state_dict
        # since it can be rebuilt deterministically from the policy_map
        # + the group descriptor.
        if group.name == "gchess":
            gather = build_orbit_policy_gather(self.POLICY_NUM_CHANNELS)
        elif group.name == "sgchess":
            gather = build_sgchess_orbit_policy_gather(self.POLICY_NUM_CHANNELS)
        else:
            raise ValueError(
                f"EquivariantPolicyHead: no orbit gather available for group "
                f"{group.name!r}; expected 'gchess' or 'sgchess'."
            )
        self.register_buffer("policy_gather", gather, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F * |G|, 8, 8) — folded G-signal from the body
        x = F.relu(self.bn1(self.conv1(x)))                 # (B, F * |G|, 8, 8)
        x = F.relu(self.bn2(self.conv2(x)))                 # (B, C * |G|, 8, 8)
        b = x.shape[0]
        c = self.POLICY_NUM_CHANNELS
        order = self.group.order
        # Channel-major folded layout: idx = logical_c * |G| + g.
        # Sum over the |G| axis collapses regular-rep, leaving only
        # spatial-flip equivariance.
        x = x.view(b, c, order, 8, 8).sum(dim=2)            # (B, C, 8, 8)
        flat = x.reshape(b, c * 64)                         # (B, C*64)
        gathered = flat.index_select(1, self.policy_gather.view(-1))  # (B, 1858*2)
        return gathered.view(b, POLICY_INDEX_SIZE, 2).mean(dim=2)     # (B, 1858)


class EquivariantValueHead(nn.Module):
    """G-equivariant value head producing (B, 3) softmax [W, D, L].

    Pipeline (parameterized by ``group``):
        GConv(F, 32, 3x3)      -> GBN -> ReLU
        GConv(32, hidden, 8x8) -> GBN -> ReLU       # collapses spatial to 1x1
        squeeze spatial       -> (B, hidden, |G|)
        Partition the group axis into a "W-half" and an "L-half" via
        the SIGMA_C action — within each half, sum the group fibers:
            W_feat = sum_{g in W-half} h[:, :, g]
            L_feat = sum_{g in L-half} h[:, :, g]
            D_feat = sum_{g in G} h[:, :, g]      # fully invariant
        SHARED ``wl_linear`` applied to BOTH W_feat and L_feat
            -> (W_logit, L_logit) — a single nn.Linear instance applied twice.
        Separate ``d_linear(D_feat)`` -> D_logit
        cat(W, D, L), softmax -> (B, 3)

    W/L half partition. Under SIGMA_C on the group axis, the GChess
    element at local index ``g`` maps to the GChess element at the
    index whose ``to_gchess`` value is ``SIGMA_C . to_gchess[g]``.
    The 4-element GChess: orbits are {E, SIGMA_C} and {SIGMA_H,
    SIGMA_HC}. We label as "W-half" the indices whose ``to_gchess`` is
    E or SIGMA_H (i.e. the SIGMA_H-orbit of E in G_chess), and as
    "L-half" the indices whose ``to_gchess`` is SIGMA_C or SIGMA_HC.
    Under SIGMA_C these halves swap.

    For ``GCHESS_GROUP`` (|G|=4): W-half = {0, 1}, L-half = {2, 3}, so
    ``W_feat = h[:,:,0] + h[:,:,1]``, ``L_feat = h[:,:,2] + h[:,:,3]``.
    For ``SGCHESS_GROUP`` (|G|=2, local indices: 0=E, 1=SIGMA_C):
    W-half = {0}, L-half = {1}, so ``W_feat = h[:,:,0]``,
    ``L_feat = h[:,:,1]``.

    Equivariance under SIGMA_C in either group: the group-axis perm
    swaps W-half indices with L-half indices, so W_feat and L_feat
    swap. Because ``wl_linear`` is SHARED (one instance applied twice),
    ``W_logit_new = wl_linear(L_feat_old) = L_logit_old`` and vice
    versa. ``D_feat`` is invariant (sum over the whole group axis).
    Output ``[W, D, L]`` swaps to ``[L, D, W]`` — matches
    ``group_action_value(SIGMA_C)``.

    Equivariance under SIGMA_H (GChess only). The group-axis perm
    swaps within each W/L half, so the sums W_feat and L_feat are
    unchanged. Output unchanged — matches ``group_action_value(SIGMA_H)
    = identity``.

    Critical implementation note: ``self.wl_linear`` MUST be a single
    nn.Linear instance applied twice — not two separate nn.Linears
    with copied weights. Two instances would receive different
    gradients from each forward path and diverge during training,
    breaking equivariance.
    """

    def __init__(
        self,
        filters: int,
        hidden: int = 128,
        group: GroupDef = GCHESS_GROUP,
    ):
        super().__init__()
        self.group = group
        self.hidden = hidden
        self.conv1 = GConv(filters, 32, 3, padding=1, bias=True, group=group)
        self.bn1 = GBatchNorm(32, group=group)
        self.conv2 = GConv(32, hidden, 8, padding=0, bias=True, group=group)
        self.bn2 = GBatchNorm(hidden, group=group)
        # SHARED across W and L outputs (CRITICAL for sigma_c-equivariance).
        self.wl_linear = nn.Linear(hidden, 1, bias=True)
        self.d_linear = nn.Linear(hidden, 1, bias=True)

        # Build W/L index buffers based on the group's GChess mapping.
        # W-half: local indices g such that to_gchess[g] is E or SIGMA_H.
        # L-half: local indices g such that to_gchess[g] is SIGMA_C or SIGMA_HC.
        # Under SIGMA_C action on the group axis, these halves swap
        # exactly (verified by composition table).
        w_half = [
            g
            for g in range(group.order)
            if group.to_gchess[g] in (int(GChess.E), int(GChess.SIGMA_H))
        ]
        l_half = [
            g
            for g in range(group.order)
            if group.to_gchess[g] in (int(GChess.SIGMA_C), int(GChess.SIGMA_HC))
        ]
        if not w_half or not l_half:
            raise ValueError(
                f"EquivariantValueHead requires a group that contains BOTH a "
                f"W-half element (E or SIGMA_H) and an L-half element "
                f"(SIGMA_C or SIGMA_HC). Group {group.name!r} has "
                f"w_half={w_half}, l_half={l_half}."
            )
        self.register_buffer(
            "w_half_indices",
            torch.tensor(w_half, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "l_half_indices",
            torch.tensor(l_half, dtype=torch.long),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))   # (B, 32 * |G|, 8, 8)
        x = F.relu(self.bn2(self.conv2(x)))   # (B, hidden * |G|, 1, 1)
        b = x.shape[0]
        # Squeeze spatial; folded c-major -> (B, hidden, |G|) where the
        # group axis enumerates local indices 0..|G|-1.
        h = x.view(b, self.hidden, self.group.order)

        W_feat = h.index_select(dim=2, index=self.w_half_indices).sum(dim=2)  # (B, hidden)
        L_feat = h.index_select(dim=2, index=self.l_half_indices).sum(dim=2)  # (B, hidden)
        D_feat = h.sum(dim=2)                                                  # (B, hidden)

        W_logit = self.wl_linear(W_feat)
        L_logit = self.wl_linear(L_feat)
        D_logit = self.d_linear(D_feat)

        logits = torch.cat([W_logit, D_logit, L_logit], dim=1)  # (B, 3)
        return F.softmax(logits, dim=1)


class InvariantMovesLeftHead(nn.Module):
    """G-invariant moves-left head producing (B, 1) non-negative output.

    Pipeline:
        GConv(F, mlh_channels, 3x3) -> GBN -> ReLU      # (B, mlh*|G|, 8, 8)
        Sum over group dim                               # (B, mlh, 8, 8)
        Mean over (H, W)                                 # (B, mlh)
        FC(mlh, fc_size) -> ReLU
        FC(fc_size, 1) -> ReLU                            # non-negative

    Sum over group dim is invariant to the regular-rep permutation;
    mean over spatial axes is invariant to spatial flips. Combined the
    (B, mlh) vector is fully invariant under any group that consists
    of spatial flips + regular-rep permutations (GCHESS_GROUP and
    SGCHESS_GROUP both qualify); the FC stack on a scalar-per-channel
    vector preserves invariance.
    """

    def __init__(
        self,
        filters: int,
        mlh_channels: int = 8,
        fc_size: int = 128,
        group: GroupDef = GCHESS_GROUP,
    ):
        super().__init__()
        self.group = group
        self.mlh_channels = mlh_channels
        self.fc_size = fc_size
        self.conv = GConv(filters, mlh_channels, 3, padding=1, bias=True, group=group)
        self.bn = GBatchNorm(mlh_channels, group=group)
        self.fc1 = nn.Linear(mlh_channels, fc_size)
        self.fc2 = nn.Linear(fc_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))      # (B, mlh * |G|, 8, 8)
        b, _, h, w = x.shape
        c = self.mlh_channels
        order = self.group.order
        x = x.view(b, c, order, h, w).sum(dim=2)        # (B, mlh, 8, 8)
        x = x.mean(dim=[2, 3])                          # (B, mlh) — fully invariant
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))                      # (B, 1)


__all__ = [
    "BaselinePolicyHead",
    "BaselineValueHead",
    "BaselineMovesLeftHead",
    "EquivariantPolicyHead",
    "EquivariantValueHead",
    "InvariantMovesLeftHead",
]
