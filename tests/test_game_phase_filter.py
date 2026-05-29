"""Game-phase filtering for LCZeroV6Dataset: turn_start + max_plies_left.

Each .gz is one game in ply order (record index == ply-from-start), so:
  - turn_start=N  keeps records with in-game index >= N (skip first N plies).
  - max_plies_left=K keeps records with plies_left <= K (within K plies of
    the game END).
A record is kept iff BOTH pass.

Most tests use synthetic records via monkeypatching so they run without the
real V6 data on disk; a couple of integration checks use raw_data when it
is present (skipped otherwise).
"""

import types
from pathlib import Path

import pytest

import data.dataset as dataset_mod
from data.dataset import LCZeroV6Dataset


# ---------------------------------------------------------------------------
# Synthetic-record harness
# ---------------------------------------------------------------------------


def _fake_record(plies_left: float):
    return types.SimpleNamespace(plies_left=float(plies_left))


# Two synthetic "games" (one per .gz). plies_left is monotonically
# non-increasing within each game, matching the real data.
_GAME_A = [20.0, 19.0, 18.0, 17.0, 16.0, 15.0]  # indices 0..5
_GAME_B = [8.0, 7.0, 6.0, 5.0]                   # indices 0..3


def _install_fake_stream(monkeypatch, ds: LCZeroV6Dataset):
    """Wire ds so that _iter_raw walks two fake games and yields each
    surviving record's plies_left (so tests can assert exactly which
    records passed the filter)."""
    files = [Path("gameA.gz"), Path("gameB.gz")]
    monkeypatch.setattr(ds, "list_files", lambda: files)

    def fake_iter(path):
        seq = _GAME_A if Path(path).name == "gameA.gz" else _GAME_B
        return [_fake_record(p) for p in seq]

    # Patch the name as used inside data.dataset._iter_raw.
    monkeypatch.setattr(dataset_mod, "iter_v6_records_from_file", fake_iter)
    # Make the yielded "sample" just the record's plies_left.
    monkeypatch.setattr(ds, "_record_to_sample", lambda rec: rec.plies_left)


# ---------------------------------------------------------------------------
# Constructor validation + defaults
# ---------------------------------------------------------------------------


def test_defaults_are_noop(monkeypatch):
    ds = LCZeroV6Dataset(folders=[], shuffle_buffer=0)
    assert ds.turn_start == 0
    assert ds.max_plies_left is None
    _install_fake_stream(monkeypatch, ds)
    out = list(ds._iter_raw())
    assert out == _GAME_A + _GAME_B  # nothing filtered


def test_turn_start_negative_raises():
    with pytest.raises(ValueError):
        LCZeroV6Dataset(folders=[], turn_start=-1)


def test_max_plies_left_negative_raises():
    with pytest.raises(ValueError):
        LCZeroV6Dataset(folders=[], max_plies_left=-5.0)


# ---------------------------------------------------------------------------
# _keep_record unit logic
# ---------------------------------------------------------------------------


def test_keep_record_turn_start_only():
    ds = LCZeroV6Dataset(folders=[], turn_start=3)
    rec = _fake_record(100.0)
    assert ds._keep_record(0, rec) is False
    assert ds._keep_record(2, rec) is False
    assert ds._keep_record(3, rec) is True
    assert ds._keep_record(10, rec) is True


def test_keep_record_max_plies_left_only():
    ds = LCZeroV6Dataset(folders=[], max_plies_left=7.0)
    assert ds._keep_record(0, _fake_record(8.0)) is False   # too far from end
    assert ds._keep_record(0, _fake_record(7.0)) is True    # boundary kept
    assert ds._keep_record(0, _fake_record(1.0)) is True


def test_keep_record_requires_both():
    ds = LCZeroV6Dataset(folders=[], turn_start=2, max_plies_left=7.0)
    # idx ok, plies too high -> drop
    assert ds._keep_record(3, _fake_record(8.0)) is False
    # plies ok, idx too low -> drop
    assert ds._keep_record(1, _fake_record(5.0)) is False
    # both ok -> keep
    assert ds._keep_record(3, _fake_record(5.0)) is True


# ---------------------------------------------------------------------------
# _iter_raw end-to-end over two synthetic games
# ---------------------------------------------------------------------------


def test_turn_start_skips_first_n_of_each_game(monkeypatch):
    """turn_start resets PER GAME (per .gz), since each file restarts the
    ply index at 0."""
    ds = LCZeroV6Dataset(folders=[], shuffle_buffer=0, turn_start=2)
    _install_fake_stream(monkeypatch, ds)
    out = list(ds._iter_raw())
    # Game A keeps indices 2..5; game B keeps indices 2..3.
    assert out == _GAME_A[2:] + _GAME_B[2:]


def test_max_plies_left_keeps_near_end(monkeypatch):
    ds = LCZeroV6Dataset(folders=[], shuffle_buffer=0, max_plies_left=7.0)
    _install_fake_stream(monkeypatch, ds)
    out = list(ds._iter_raw())
    # Only records with plies_left <= 7: game A none; game B [7,6,5].
    assert out == [7.0, 6.0, 5.0]


def test_both_filters_compose(monkeypatch):
    ds = LCZeroV6Dataset(
        folders=[], shuffle_buffer=0, turn_start=2, max_plies_left=7.0
    )
    _install_fake_stream(monkeypatch, ds)
    out = list(ds._iter_raw())
    # Game A: idx>=2 gives plies [18,17,16,15], all > 7 -> none.
    # Game B: idx>=2 gives plies [6,5], both <= 7 -> kept.
    assert out == [6.0, 5.0]


def test_filter_can_empty_the_stream(monkeypatch):
    ds = LCZeroV6Dataset(folders=[], shuffle_buffer=0, max_plies_left=0.5)
    _install_fake_stream(monkeypatch, ds)
    assert list(ds._iter_raw()) == []


# ---------------------------------------------------------------------------
# Trainer CLI wiring
# ---------------------------------------------------------------------------


def test_trainer_cli_accepts_game_phase_args():
    from training.trainer import parse_args

    args = parse_args([
        "--model", "baseline",
        "--blocks", "2", "--filters", "16",
        "--max-steps", "1", "--device", "cpu",
        "--turn-start", "40", "--max-plies-left", "20",
    ])
    assert args.turn_start == 40
    assert args.max_plies_left == 20.0


def test_trainer_cli_game_phase_defaults():
    from training.trainer import parse_args

    args = parse_args([
        "--model", "baseline", "--max-steps", "1", "--device", "cpu",
    ])
    assert args.turn_start == 0
    assert args.max_plies_left is None


# ---------------------------------------------------------------------------
# Integration with real V6 data (skipped when absent)
# ---------------------------------------------------------------------------


REAL_DATA_DIR = Path("raw_data/chunck_set_1")
skip_no_data = pytest.mark.skipif(
    not (REAL_DATA_DIR.exists() and any(REAL_DATA_DIR.glob("*.gz"))),
    reason=f"No raw V6 data found at {REAL_DATA_DIR}",
)


@skip_no_data
def test_turn_start_reduces_count_on_real_data():
    folders = [REAL_DATA_DIR]
    base = LCZeroV6Dataset(folders=folders, shuffle_buffer=0)
    filt = LCZeroV6Dataset(folders=folders, shuffle_buffer=0, turn_start=20)

    # Count over the first few files only (cap work).
    def count_first_files(ds, n_files=5):
        files = ds.list_files()[:n_files]
        c = 0
        for f in files:
            for idx, rec in enumerate(dataset_mod.iter_v6_records_from_file(f)):
                if ds._keep_record(idx, rec):
                    c += 1
        return c

    assert count_first_files(filt) < count_first_files(base)


@skip_no_data
def test_max_plies_left_selects_endgame_on_real_data():
    """Every record surviving the max_plies_left filter must have a small
    plies_left (genuine endgame proximity)."""
    ds = LCZeroV6Dataset(folders=[REAL_DATA_DIR], shuffle_buffer=0, max_plies_left=15.0)
    files = ds.list_files()[:10]
    kept = 0
    for f in files:
        for idx, rec in enumerate(dataset_mod.iter_v6_records_from_file(f)):
            if ds._keep_record(idx, rec):
                kept += 1
                assert rec.plies_left <= 15.0
    # At least some endgame positions exist in the first 10 games.
    assert kept > 0
