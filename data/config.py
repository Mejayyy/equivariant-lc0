"""Train/val/test split — file-level (not random per-position) to prevent
correlated positions from the same game leaking across splits.

Folder names match the on-disk layout (note "chunck" is the actual spelling
in our raw_data, not "chunk" as in CLAUDE.md's example).
"""

DATA_SPLITS = {
    "train": [
        "raw_data/chunck_set_1",
        "raw_data/chunck_set_2",
        "raw_data/chunck_set_3",
    ],
    "val": ["raw_data/chunck_set_4"],
    "test": ["raw_data/chunck_set_5"],
}
