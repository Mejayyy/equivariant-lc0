"""Extract supervised training targets (policy, value WDL, MLH) from a V6Record."""

import math
from typing import Dict

import numpy as np

from .v6_parser import V6Record


def v6_to_targets(record: V6Record) -> Dict[str, np.ndarray]:
    """Return {'policy', 'value', 'mlh'} for one V6 record.

    policy: (1858,) float32. V6 stores -1.0 in slots for moves that are not
        legal in this position; legal-move slots store an MCTS visit
        probability (non-negative; legal-only entries sum to ~1.0).

    value: (3,) float32 in [W, D, L] order. Computed from best_q/best_d when
        finite, falling back to result_q/result_d if best is NaN (the V6 spec
        permits NaN there for cache-miss positions).

    mlh: scalar float32, expected plies remaining until end of game.
    """
    policy = record.probabilities.astype(np.float32, copy=True)

    if math.isnan(record.best_q) or math.isnan(record.best_d):
        q, d = record.result_q, record.result_d
    else:
        q, d = record.best_q, record.best_d

    W = (1.0 + q - d) / 2.0
    L = (1.0 - q - d) / 2.0
    D = d
    value = np.array([W, D, L], dtype=np.float32)

    mlh = np.float32(record.plies_left)

    return {"policy": policy, "value": value, "mlh": mlh}


__all__ = ["v6_to_targets"]
