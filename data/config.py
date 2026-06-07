"""Data configuration.

Two selection schemes coexist:

1. **Pooled random split (current default for training).** All `.gz` games
   under `RAW_DATA_ROOT` (across every sub-folder, regardless of name —
   `chunck_set_*`, `training-run2-*`, ...) are pooled, the GAMES are shuffled
   with a fixed seed, and a disjoint set is reserved for validation vs
   training. A fixed budget of *filtered* plies is then drawn:
   `TRAIN_SAMPLES` for training, `VAL_SAMPLES` for validation — **the same
   counts regardless of `turn_start` / `max_plies_left`** (the per-game ply
   filter is applied first, then capped to the budget). The split is at the
   GAME level (no positions from one game leak across train/val), and the
   game partition is filter-independent so the validation GAMES are stable
   across different filter settings.

2. **Legacy fixed-folder split** (`DATA_SPLITS`) — kept for reference /
   backward compatibility. No longer used by the trainer.
"""

# --- Pooled random split (default) ---------------------------------------
RAW_DATA_ROOT = "raw_data"

# Target number of *filtered* plies per split (capped during iteration so the
# count is constant for any turn_start / max_plies_left, provided the data
# pool is large enough; otherwise the trainer warns and uses all available).
TRAIN_SAMPLES = 3_600_000
VAL_SAMPLES = 512_000

# Fixed seed for the GAME-level shuffle that partitions files into train vs
# val. Keeping it fixed (independent of the run / model seed) guarantees the
# SAME validation games and the SAME training pool across all model runs and
# all filter settings — essential for a controlled ablation.
SPLIT_SEED = 1234

# Fixed seed for the per-position shuffle buffer during training. Decoupled
# from the model/run seed so the *data order* is identical across model runs.
DATA_SEED = 1234


# --- Legacy fixed-folder split (unused by the trainer; kept for reference) -
DATA_SPLITS = {
    "train": [
        "raw_data/chunck_set_1",
        "raw_data/chunck_set_2",
        "raw_data/chunck_set_3",
        "raw_data/chunck_set_4",
    ],
    "val": ["raw_data/chunck_set_5"],
}
