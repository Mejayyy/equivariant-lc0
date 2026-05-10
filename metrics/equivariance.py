"""Equivariance error metric for evaluating G_chess-equivariant models.

For a 3-headed model `model(x) -> (policy, value, mlh)`, computes the
relative-Frobenius deviation between f(g . x) and g . f(x) per head,
averaged across heads and samples, for each non-identity g in G_chess.

API matches CLAUDE.md "Equivariance Error Metric":

    equivariance_error(model, n_samples=64, device='cpu') -> {
        "sigma_h":      <mean relative Frobenius error>,
        "sigma_c":      <mean relative Frobenius error>,
        "sigma_h_c":    <mean relative Frobenius error>,
        "overall_mean": <mean across non-identity elements>,
    }

Runs in `model.eval()` + `torch.no_grad()`. For Models 2 and 3, every
head is equivariant by construction; expected error magnitude is
~float32-roundoff (a few times 1e-6, well under the 1e-5 threshold).
For Model 1 (the baseline), this metric quantifies the "implicit
equivariance" learned from canonicalized data after training.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from symmetry.actions import (
    group_action_input,
    group_action_mlh,
    group_action_policy,
    group_action_value,
)
from symmetry.group import GChess


def sample_chess_input(batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
    """Sample a (batch_size, 34, 8, 8) tensor matching the CLAUDE.md
    sampling spec.

    Produces values plausible for our 34-plane encoding:
      - Piece + repetition + castling planes: binary sparse 0/1.
      - Plane 30 (side-to-move): a single 0/1 per batch element broadcast
        over the 8x8 — MUST be exactly 0 or 1 for sigma_c bit-exact
        invertibility (sigma_c flips it as 1 - x[30]).
      - Plane 31 (rule50): scalar in [0, 1] broadcast spatially.
      - Planes 32, 33: constants 0 and 1.
    """
    x = torch.zeros(batch_size, 34, 8, 8, device=device)
    x[:, 0:12] = (torch.rand(batch_size, 12, 8, 8, device=device) > 0.85).float()
    x[:, 12] = (torch.rand(batch_size, 8, 8, device=device) > 0.9).float()
    x[:, 13:25] = (torch.rand(batch_size, 12, 8, 8, device=device) > 0.85).float()
    x[:, 25] = (torch.rand(batch_size, 8, 8, device=device) > 0.9).float()
    x[:, 26:30] = (torch.rand(batch_size, 4, 8, 8, device=device) > 0.5).float()
    side = (torch.rand(batch_size, device=device) > 0.5).float()
    x[:, 30] = side.view(batch_size, 1, 1).expand(batch_size, 8, 8)
    rule50 = torch.rand(batch_size, device=device)
    x[:, 31] = rule50.view(batch_size, 1, 1).expand(batch_size, 8, 8)
    x[:, 32] = 0.0
    x[:, 33] = 1.0
    return x


def equivariance_error(
    model: nn.Module,
    n_samples: int = 64,
    device: str = "cpu",
    eps: float = 1e-9,
) -> Dict[str, float]:
    """Per-element relative-Frobenius equivariance error for a 3-head
    model with the standard `(policy, value, mlh) = model(x)` interface.

    Args:
        model: a `nn.Module` with the standard 3-head signature.
        n_samples: number of random chess inputs to evaluate.
        device: torch device for sampling and inference.
        eps: numerator + eps in the relative-error denominator to avoid
            division by zero when an output happens to be all-zero (e.g.
            an under-trained MLH head whose final ReLU saturates to 0).

    Returns dict with keys "sigma_h", "sigma_c", "sigma_h_c",
    "overall_mean".
    """
    was_training = model.training
    model.eval()

    elements = (GChess.SIGMA_H, GChess.SIGMA_C, GChess.SIGMA_HC)
    head_actions = (group_action_policy, group_action_value, group_action_mlh)

    sums = {g: 0.0 for g in elements}

    with torch.no_grad():
        for _ in range(n_samples):
            x = sample_chess_input(batch_size=1, device=device)
            fx = model(x)
            for g in elements:
                gx = group_action_input(g, x)
                fgx = model(gx)
                gfx = tuple(act(g, h) for act, h in zip(head_actions, fx))

                head_errs = []
                for f_after_g, target, original in zip(fgx, gfx, fx):
                    diff = (f_after_g - target).norm()
                    norm = original.norm()
                    head_errs.append((diff / (norm + eps)).item())

                sums[g] += sum(head_errs) / len(head_errs)

    if was_training:
        model.train()

    means = {g: sums[g] / n_samples for g in elements}
    return {
        "sigma_h": means[GChess.SIGMA_H],
        "sigma_c": means[GChess.SIGMA_C],
        "sigma_h_c": means[GChess.SIGMA_HC],
        "overall_mean": sum(means.values()) / len(means),
    }


__all__ = [
    "sample_chess_input",
    "equivariance_error",
]
