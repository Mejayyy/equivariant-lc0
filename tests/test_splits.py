"""Pooled game-level train/val split (data/splits.py).

The partition must be: deterministic given the seed, disjoint (no game in
both splits), cover all games, size the val fraction by the ply budgets, and
be filter-independent (it depends only on files + seed, never on
turn_start / max_plies_left) so the validation games are stable across all
filter settings.
"""

import pytest

from data.splits import partition_files


FILES = [f"g{i:05d}.gz" for i in range(1000)]
TRAIN_TARGET = 3_600_000
VAL_TARGET = 512_000


def test_partition_deterministic():
    a = partition_files(FILES, TRAIN_TARGET, VAL_TARGET, seed=1234)
    b = partition_files(FILES, TRAIN_TARGET, VAL_TARGET, seed=1234)
    assert a == b


def test_partition_disjoint_and_complete():
    train, val = partition_files(FILES, TRAIN_TARGET, VAL_TARGET, seed=1234)
    assert set(train).isdisjoint(set(val))
    assert set(train) | set(val) == set(FILES)
    assert len(train) + len(val) == len(FILES)


def test_partition_val_fraction_matches_budget():
    train, val = partition_files(FILES, TRAIN_TARGET, VAL_TARGET, seed=1234)
    frac = VAL_TARGET / (TRAIN_TARGET + VAL_TARGET)  # ~0.1245
    expected = round(frac * len(FILES))
    assert len(val) == expected


def test_partition_seed_changes_split():
    _, val1 = partition_files(FILES, TRAIN_TARGET, VAL_TARGET, seed=1)
    _, val2 = partition_files(FILES, TRAIN_TARGET, VAL_TARGET, seed=2)
    assert set(val1) != set(val2)


def test_partition_independent_of_input_order():
    """Sorting happens inside partition_files, so input order must not
    matter (reproducible across machines / glob orderings)."""
    import random as _r
    shuffled = FILES[:]
    _r.Random(99).shuffle(shuffled)
    a = partition_files(FILES, TRAIN_TARGET, VAL_TARGET, seed=1234)
    b = partition_files(shuffled, TRAIN_TARGET, VAL_TARGET, seed=1234)
    assert a == b


def test_partition_filter_independent_by_construction():
    """partition_files takes no filter argument — the same (files, seed,
    targets) always yields the same val games, so turn_start / max_plies_left
    cannot move a game between splits. Lock this in as a regression."""
    base_train, base_val = partition_files(FILES, TRAIN_TARGET, VAL_TARGET, seed=1234)
    # Re-derive with the same inputs (as the trainer does for every filter).
    again_train, again_val = partition_files(FILES, TRAIN_TARGET, VAL_TARGET, seed=1234)
    assert base_val == again_val
    assert base_train == again_train


def test_partition_rejects_tiny_input():
    with pytest.raises(ValueError):
        partition_files(["only.gz"], TRAIN_TARGET, VAL_TARGET, seed=0)


def test_partition_rejects_nonpositive_targets():
    with pytest.raises(ValueError):
        partition_files(FILES, 0, VAL_TARGET, seed=0)
    with pytest.raises(ValueError):
        partition_files(FILES, TRAIN_TARGET, -1, seed=0)
