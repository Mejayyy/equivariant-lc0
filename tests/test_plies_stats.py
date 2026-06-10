"""data/plies_stats.py — the pure `filtered_count` helper must mirror
`LCZeroV6Dataset._keep_record` exactly (keep iff in-game index >=
turn_start AND plies_left <= max_plies_left, inclusive), because the
yield analysis stands in for the real dataset filter."""

from types import SimpleNamespace

from data.dataset import LCZeroV6Dataset
from data.plies_stats import filtered_count

# One synthetic game, 10 records, plies_left counting down 10 -> 1.
GAME = [float(x) for x in range(10, 0, -1)]


def test_no_filter_counts_all():
    assert filtered_count(GAME) == 10


def test_turn_start_drops_prefix():
    assert filtered_count(GAME, turn_start=3) == 7


def test_turn_start_past_end_yields_zero():
    assert filtered_count(GAME, turn_start=10) == 0
    assert filtered_count(GAME, turn_start=99) == 0


def test_max_plies_left_inclusive():
    # plies_left values 3.0, 2.0, 1.0 pass an inclusive <= 3.0.
    assert filtered_count(GAME, max_plies_left=3.0) == 3


def test_both_filters_intersect():
    # idx >= 5 keeps plies_left 5..1; plies_left <= 7 keeps 7..1; overlap 5.
    assert filtered_count(GAME, turn_start=5, max_plies_left=7.0) == 5


def test_matches_dataset_keep_record_on_grid():
    """Bit-for-bit agreement with the dataset's own filter across a grid of
    settings (the dataset is never iterated, so the fake path is fine)."""
    for ts in (0, 2, 5, 11):
        for mpl in (None, 1.0, 4.5, 100.0):
            ds = LCZeroV6Dataset(
                files=["never_opened.gz"], shuffle_buffer=0,
                turn_start=ts, max_plies_left=mpl,
            )
            expected = sum(
                ds._keep_record(i, SimpleNamespace(plies_left=pl))
                for i, pl in enumerate(GAME)
            )
            assert filtered_count(GAME, ts, mpl) == expected
