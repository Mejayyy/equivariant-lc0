"""Pooled, game-level train/val splitting over all of `raw_data`.

Every `.gz` chunk is one game. We pool games from ALL sub-folders of the
data root (any naming — `chunck_set_*`, `training-run2-*`, ...), shuffle the
GAMES with a fixed seed, and reserve a disjoint slice for validation. The
split is therefore:

- **at the game level** → no positions from a single game appear in both
  train and val (no leakage);
- **filter-independent** → the partition depends only on the file list and the
  seed, NOT on `turn_start` / `max_plies_left`, so the validation games are the
  same across all filter settings;
- **reproducible** → the file list is sorted before the seeded shuffle, so the
  partition is identical across runs and machines.

The number of *plies* each split actually yields is then bounded by a cap
(`max_samples`) applied during dataset iteration — that is what makes the
filtered-ply count constant regardless of the filter (see
`LCZeroV6Dataset`).

The validation fraction of games is set proportional to the ply budgets,
`VAL_SAMPLES / (TRAIN_SAMPLES + VAL_SAMPLES)`, so both splits get the same
headroom relative to their target (games are near-uniform in length, ~118
plies, so a file-count fraction tracks a ply-count fraction closely).
"""

from __future__ import annotations

import glob
import os
import random
from pathlib import Path
from typing import List, Sequence, Tuple, Union

PathLike = Union[str, Path]


def list_all_chunk_files(root: PathLike) -> List[str]:
    """Return every `.gz` chunk under `root` (recursively), sorted for
    deterministic ordering prior to any seeded shuffle."""
    files = glob.glob(os.path.join(str(root), "**", "*.gz"), recursive=True)
    return sorted(files)


def partition_files(
    files: Sequence[PathLike],
    train_target: int,
    val_target: int,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """Partition a list of game files into (train_files, val_files).

    The input is sorted then shuffled with `seed` (so the result is
    deterministic and independent of the input order). The first
    `n_val = round(f * N)` games — with `f = val_target / (train_target +
    val_target)` — are the validation games; the rest are training games.
    Disjoint by construction.

    Returns the two lists in shuffled order (so downstream iteration over a
    prefix is already a random sample of games).
    """
    if train_target <= 0 or val_target <= 0:
        raise ValueError("train_target and val_target must be positive")
    shuffled = [str(f) for f in sorted(files)]
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n = len(shuffled)
    if n < 2:
        raise ValueError(f"need at least 2 games to split, got {n}")
    frac_val = val_target / float(train_target + val_target)
    n_val = max(1, min(n - 1, round(frac_val * n)))
    val_files = shuffled[:n_val]
    train_files = shuffled[n_val:]
    return train_files, val_files


def build_pooled_splits(
    root: PathLike,
    train_target: int,
    val_target: int,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """Glob all games under `root` and partition them into
    (train_files, val_files). See module docstring for the guarantees."""
    files = list_all_chunk_files(root)
    if not files:
        raise FileNotFoundError(f"no .gz chunk files found under {root!r}")
    return partition_files(files, train_target, val_target, seed)


__all__ = [
    "list_all_chunk_files",
    "partition_files",
    "build_pooled_splits",
]
