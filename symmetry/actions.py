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
