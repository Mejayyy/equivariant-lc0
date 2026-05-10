"""Loss functions for the three LCZero heads.

Each loss returns a SCALAR mean over the batch:
    - policy_cross_entropy(logits, target): policy CE over the V6 1858
      probability distribution. V6 stores -1 in slots for moves that are
      illegal in the current position; this loss masks them out so they
      contribute nothing AND so the softmax does not leak probability
      mass to illegal moves.
    - value_wdl_loss(pred_softmax, target): cross-entropy over the
      [W, D, L] WDL distribution. Our value head outputs softmax already.
    - mlh_l1_loss(pred, target): mean absolute error in plies.
    - combined_loss(model_output, targets, weights): weighted sum.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def policy_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy of the LCZero policy distribution.

    Args:
        logits: ``(B, 1858)`` raw model logits (no softmax).
        target: ``(B, 1858)`` V6 probability target. Illegal-move slots
            store ``-1.0`` as a sentinel; legal-move slots store
            non-negative visit-frequency values that sum to ~1.0 per
            sample.

    Returns: scalar mean cross-entropy across the batch.

    Implementation notes:
    - Illegal slots are MASKED OUT in two places:
        (a) target_clean = target.masked_fill(~mask, 0). Their
            contribution to the sum is zeroed.
        (b) Logits at illegal slots are set to -inf BEFORE softmax so the
            model is never rewarded for distributing probability there.
            Without this masking, a model could trivially reduce loss by
            assigning ~1/1858 to every slot — most of that probability
            being "wasted" on illegal moves the V6 target ignores.
    - IEEE: ``0 * (-inf) = nan``. After log_softmax, illegal entries are
      ``-inf`` (since their logit was -inf). We re-mask log_probs to 0
      for those entries before multiplying by target_clean, avoiding nan.
    """
    if logits.shape != target.shape:
        raise ValueError(
            f"shape mismatch: logits {tuple(logits.shape)} vs target "
            f"{tuple(target.shape)}"
        )

    mask = target >= 0
    target_clean = target.masked_fill(~mask, 0.0)
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    log_probs = F.log_softmax(masked_logits, dim=-1)
    log_probs = log_probs.masked_fill(~mask, 0.0)
    ce = -(target_clean * log_probs).sum(dim=-1)  # (B,)
    return ce.mean()


def value_wdl_loss(
    pred_softmax: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Cross-entropy of the [W, D, L] value distribution.

    Args:
        pred_softmax: ``(B, 3)`` model output AFTER softmax.
        target: ``(B, 3)`` WDL target (each row sums to 1).
        eps: clamp on `pred_softmax` before log to avoid -inf when a
            class probability is exactly 0.
    """
    if pred_softmax.shape != target.shape:
        raise ValueError(
            f"shape mismatch: pred {tuple(pred_softmax.shape)} vs target "
            f"{tuple(target.shape)}"
        )
    log_pred = pred_softmax.clamp(min=eps).log()
    return -(target * log_pred).sum(dim=-1).mean()


def mlh_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 loss on moves-left.

    Args:
        pred: ``(B, 1)`` model output (non-negative, after final ReLU).
        target: ``(B,)`` or ``(B, 1)`` plies-left target.
    """
    target = target.view_as(pred)
    return (pred - target).abs().mean()


DEFAULT_WEIGHTS = {"policy": 1.0, "value": 1.0, "mlh": 0.05}


def combined_loss(
    model_output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    weights: Dict[str, float] = None,
) -> Dict[str, torch.Tensor]:
    """Compute the three head losses and a weighted sum.

    Args:
        model_output: ``(policy_logits, value_softmax, mlh_pred)`` —
            matches the BaselineLCZero / GCNNChess forward signature.
        targets: dict with ``policy`` (``(B, 1858)``), ``value``
            (``(B, 3)``), ``mlh`` (``(B,)`` or ``(B, 1)``) entries.
        weights: per-head loss weights. Defaults: policy=1.0, value=1.0,
            mlh=0.05 (matches LCZero's typical weighting; MLH is rarely
            the limiting signal so it gets a small weight).

    Returns dict with keys ``total``, ``policy``, ``value``, ``mlh``.
    The ``total`` entry has gradients; the per-head entries are detached
    for logging.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    p_logits, v_pred, m_pred = model_output
    p = policy_cross_entropy(p_logits, targets["policy"])
    v = value_wdl_loss(v_pred, targets["value"])
    m = mlh_l1_loss(m_pred, targets["mlh"])
    total = (
        weights["policy"] * p + weights["value"] * v + weights["mlh"] * m
    )
    return {
        "total": total,
        "policy": p.detach(),
        "value": v.detach(),
        "mlh": m.detach(),
    }


__all__ = [
    "policy_cross_entropy",
    "value_wdl_loss",
    "mlh_l1_loss",
    "combined_loss",
    "DEFAULT_WEIGHTS",
]
