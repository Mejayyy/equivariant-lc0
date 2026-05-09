"""Group actions of G_chess on input planes (B, 34, 8, 8) and on body
feature maps (B, C, 8, 8).

Conventions:
- Tensor shape is (B, C, ranks, files), so dim -2 is ranks, dim -1 is files.
- sigma_h = file flip = torch.flip(x, [-1])           (A <-> H)
- sigma_c = rank flip + channel permutation + plane-30 negation
- sigma_hc = sigma_h composed with sigma_c

Input plane semantics (from CLAUDE.md):
- 0..5  our pieces (current),   6..11 opponent pieces (current)
- 12 current repetition flag
- 13..18 our pieces (1 ply ago), 19..24 opp pieces (1 ply ago)
- 25 1-ply-ago repetition flag
- 26 we OOO, 27 we OO, 28 opp OOO, 29 opp OO
- 30 black-to-move
- 31 rule50/100, 32 zeros, 33 ones

Body feature maps have unstructured channels (no chess interpretation),
so only the spatial flips apply at that level.
"""

import torch

from .group import GChess

# After sigma_c, channel i takes its value from old channel
# SIGMA_C_CHANNEL_PERM[i].
SIGMA_C_CHANNEL_PERM = [
    6, 7, 8, 9, 10, 11,
    0, 1, 2, 3, 4, 5,
    12,
    19, 20, 21, 22, 23, 24,
    13, 14, 15, 16, 17, 18,
    25,
    28, 29, 26, 27,
    30,
    31, 32, 33,
]
assert len(SIGMA_C_CHANNEL_PERM) == 34


def _apply_sigma_c_input(x: torch.Tensor) -> torch.Tensor:
    """sigma_c on (B, 34, 8, 8): rank flip + channel swap + plane-30 negate."""
    x = torch.flip(x, dims=[-2])
    perm = torch.tensor(SIGMA_C_CHANNEL_PERM, device=x.device, dtype=torch.long)
    x = x.index_select(dim=1, index=perm)
    x[:, 30] = 1.0 - x[:, 30]
    return x


def group_action_input(g, x: torch.Tensor) -> torch.Tensor:
    """Apply G_chess element g to the 34-plane input tensor x of shape (B, 34, 8, 8)."""
    if x.shape[1] != 34:
        raise ValueError(f"group_action_input expects 34 channels, got {x.shape[1]}")
    g = int(g)
    if g == int(GChess.E):
        return x.clone()
    if g == int(GChess.SIGMA_H):
        return torch.flip(x, dims=[-1])
    if g == int(GChess.SIGMA_C):
        return _apply_sigma_c_input(x)
    if g == int(GChess.SIGMA_HC):
        return torch.flip(_apply_sigma_c_input(x), dims=[-1])
    raise ValueError(f"unknown group element {g}")


def group_action_output(g, fx: torch.Tensor) -> torch.Tensor:
    """Apply G_chess element g to a body feature map (B, C, H, W).

    Body feature maps have unstructured channels, so only spatial flips apply.
    """
    g = int(g)
    if g == int(GChess.E):
        return fx.clone()
    if g == int(GChess.SIGMA_H):
        return torch.flip(fx, dims=[-1])
    if g == int(GChess.SIGMA_C):
        return torch.flip(fx, dims=[-2])
    if g == int(GChess.SIGMA_HC):
        return torch.flip(fx, dims=[-2, -1])
    raise ValueError(f"unknown group element {g}")


# ---------------------------------------------------------------------------
# Policy permutation
# ---------------------------------------------------------------------------
#
# For each policy index i with move (from_sq, to_sq, promo), the permutation
# under g maps i to the policy index of the spatially-transformed move:
#   - sigma_h: file flip on both squares.
#   - sigma_c: rank flip on both squares (NON-promotion moves only).
#   - sigma_hc: file flip + rank flip (rank flip skipped for promotions).
#
# Edge case for under-promotions under sigma_c: rank-flipping a (rank 6 -> rank
# 7) promotion would produce a (rank 1 -> rank 0) move, which has no slot in
# our 1858 set (LCZero canonical only encodes side-to-move-perspective
# promotions). We treat sigma_c as identity on under-promotion slots — this
# preserves perm[perm] == identity and matches the canonicalization argument
# (in canonical view, a single promotion index represents promotion regardless
# of which side is to move).


def _flip_file_sq(sq: int) -> int:
    return (sq // 8) * 8 + (7 - sq % 8)


def _flip_rank_sq(sq: int) -> int:
    return (7 - sq // 8) * 8 + sq % 8


def build_policy_permutation(g) -> torch.Tensor:
    """Permutation of the 1858 policy indices induced by group element g.

    Returns a torch.LongTensor of shape (1858,) where perm[i] = j means
    "move i transforms to move j under g" — equivalently, after applying g
    to the input, the logit value originally at index i appears at index j
    in the new output.
    """
    # Imported locally to keep symmetry/ free of a hard dependency on models/
    # at import time (matters only for tooling that scans modules in isolation).
    from models.policy_map import (
        MOVE_TO_POLICY_INDEX,
        POLICY_INDEX_SIZE,
        POLICY_INDEX_TO_MOVE,
    )

    g_int = int(g)
    apply_h = g_int in (int(GChess.SIGMA_H), int(GChess.SIGMA_HC))
    apply_c_nonpromo = g_int in (int(GChess.SIGMA_C), int(GChess.SIGMA_HC))

    perm = [0] * POLICY_INDEX_SIZE
    for i, (from_sq, to_sq, promo) in enumerate(POLICY_INDEX_TO_MOVE):
        new_from, new_to = from_sq, to_sq
        if apply_h:
            new_from = _flip_file_sq(new_from)
            new_to = _flip_file_sq(new_to)
        if apply_c_nonpromo and not promo:
            new_from = _flip_rank_sq(new_from)
            new_to = _flip_rank_sq(new_to)
        perm[i] = MOVE_TO_POLICY_INDEX[(new_from, new_to, promo)]
    return torch.tensor(perm, dtype=torch.long)


# ---------------------------------------------------------------------------
# Group action on head outputs
# ---------------------------------------------------------------------------


def group_action_policy(g, policy: torch.Tensor) -> torch.Tensor:
    """Apply g to a policy logits tensor (..., 1858).

    Each element is a self-inverse permutation, so index_select with the
    permutation yields the action; applying it twice returns the original.
    """
    if policy.shape[-1] != 1858:
        raise ValueError(f"group_action_policy expects 1858 trailing dim, got {policy.shape[-1]}")
    g = int(g)
    if g == int(GChess.E):
        return policy.clone()
    perm = build_policy_permutation(g).to(policy.device)
    return policy.index_select(dim=-1, index=perm)


# Value WDL output: [W, D, L]. Under sigma_c (and sigma_h composed with it)
# W <-> L; D unchanged. Under sigma_h alone: identity.
_VALUE_SWAP_INDEX = torch.tensor([2, 1, 0], dtype=torch.long)


def group_action_value(g, value: torch.Tensor) -> torch.Tensor:
    """Apply g to a value WDL tensor (..., 3)."""
    if value.shape[-1] != 3:
        raise ValueError(f"group_action_value expects 3 trailing dim, got {value.shape[-1]}")
    g = int(g)
    if g in (int(GChess.E), int(GChess.SIGMA_H)):
        return value.clone()
    if g in (int(GChess.SIGMA_C), int(GChess.SIGMA_HC)):
        return value.index_select(dim=-1, index=_VALUE_SWAP_INDEX.to(value.device))
    raise ValueError(f"unknown group element {g}")


def group_action_mlh(g, mlh: torch.Tensor) -> torch.Tensor:
    """Apply g to a moves-left tensor — invariant under all g."""
    int(g)  # validate-only
    return mlh.clone()
