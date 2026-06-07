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

import shutil
import types
from pathlib import Path

import pytest

import data.dataset as dataset_mod
import data.v6_parser as v6_parser_mod
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
# count_samples() — synthetic
# ---------------------------------------------------------------------------


def _install_fake_counters(monkeypatch):
    """Patch the v6_parser helpers count_samples() lazily imports, so we can
    drive it with the two synthetic games without touching disk."""
    counts = {"gameA.gz": len(_GAME_A), "gameB.gz": len(_GAME_B)}
    plies = {"gameA.gz": _GAME_A, "gameB.gz": _GAME_B}
    monkeypatch.setattr(
        v6_parser_mod, "count_records_in_file",
        lambda p: counts[Path(p).name],
    )
    monkeypatch.setattr(
        v6_parser_mod, "iter_plies_left_from_file",
        lambda p: iter(plies[Path(p).name]),
    )


def _ds_with_files(monkeypatch, **kw):
    ds = LCZeroV6Dataset(folders=[], shuffle_buffer=0, **kw)
    monkeypatch.setattr(ds, "list_files", lambda: [Path("gameA.gz"), Path("gameB.gz")])
    _install_fake_counters(monkeypatch)
    return ds


def test_count_samples_no_filter_uses_record_counts(monkeypatch):
    ds = _ds_with_files(monkeypatch)
    assert ds.count_samples() == len(_GAME_A) + len(_GAME_B)  # 6 + 4


def test_count_samples_turn_start_subtracts_prefix_per_game(monkeypatch):
    ds = _ds_with_files(monkeypatch, turn_start=2)
    # max(0, 6-2) + max(0, 4-2) = 4 + 2
    assert ds.count_samples() == 6


def test_count_samples_turn_start_clamps_at_zero(monkeypatch):
    ds = _ds_with_files(monkeypatch, turn_start=100)
    assert ds.count_samples() == 0


def test_count_samples_max_plies_left(monkeypatch):
    ds = _ds_with_files(monkeypatch, max_plies_left=7.0)
    # plies_left <= 7: game A none; game B [7,6,5] -> 3
    assert ds.count_samples() == 3


def test_count_samples_both_filters(monkeypatch):
    ds = _ds_with_files(monkeypatch, turn_start=2, max_plies_left=7.0)
    # Only game B indices 2,3 (plies 6,5) survive -> 2
    assert ds.count_samples() == 2


def test_count_samples_matches_iter_raw(monkeypatch):
    """count_samples() must equal the number of items _iter_raw() yields."""
    for kw in (
        {},
        {"turn_start": 2},
        {"max_plies_left": 7.0},
        {"turn_start": 2, "max_plies_left": 7.0},
    ):
        ds = LCZeroV6Dataset(folders=[], shuffle_buffer=0, **kw)
        monkeypatch.setattr(ds, "list_files", lambda: [Path("gameA.gz"), Path("gameB.gz")])
        _install_fake_counters(monkeypatch)        # for count_samples
        _install_fake_stream(monkeypatch, ds)      # for _iter_raw (overrides _record_to_sample + iter)
        n_iter = len(list(ds._iter_raw()))
        # _install_fake_stream patched iter_v6_records_from_file; counting
        # uses the separately-patched count_records_in_file / plies iter.
        n_count = ds.count_samples()
        assert n_count == n_iter, f"{kw}: count={n_count} iter={n_iter}"


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


def test_trainer_cli_count_samples_flag():
    from training.trainer import parse_args

    base = parse_args(["--model", "baseline", "--max-steps", "1", "--device", "cpu"])
    assert base.count_samples is True  # default on
    off = parse_args([
        "--model", "baseline", "--max-steps", "1", "--device", "cpu",
        "--no-count-samples",
    ])
    assert off.count_samples is False


def test_trainer_cli_pooled_split_defaults_and_overrides():
    from training.trainer import parse_args
    from data.config import (
        RAW_DATA_ROOT, TRAIN_SAMPLES, VAL_SAMPLES, SPLIT_SEED, DATA_SEED,
    )

    d = parse_args(["--model", "baseline", "--max-steps", "1", "--device", "cpu"])
    assert d.data_root == RAW_DATA_ROOT
    assert d.train_samples == TRAIN_SAMPLES
    assert d.val_samples == VAL_SAMPLES
    assert d.split_seed == SPLIT_SEED
    assert d.data_seed == DATA_SEED

    o = parse_args([
        "--model", "baseline", "--max-steps", "1", "--device", "cpu",
        "--data-root", "some/dir", "--train-samples", "1000",
        "--val-samples", "200", "--split-seed", "7", "--data-seed", "9",
    ])
    assert o.data_root == "some/dir"
    assert o.train_samples == 1000
    assert o.val_samples == 200
    assert o.split_seed == 7
    assert o.data_seed == 9


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


def _copy_first_files(src_dir: Path, dst_dir: Path, n: int) -> int:
    """Copy the first n .gz files from src_dir into dst_dir; return count."""
    files = sorted(src_dir.glob("*.gz"))[:n]
    for f in files:
        shutil.copy(f, dst_dir / f.name)
    return len(files)


@skip_no_data
def test_count_samples_matches_full_iteration_real_data(tmp_path):
    """On a small real-data folder, count_samples() (fast ISIZE path)
    must equal the number of records actually parsed."""
    n_files = _copy_first_files(REAL_DATA_DIR, tmp_path, 4)
    assert n_files > 0
    ds = LCZeroV6Dataset(folders=[tmp_path], shuffle_buffer=0)
    manual = sum(
        1
        for f in ds.list_files()
        for _ in dataset_mod.iter_v6_records_from_file(f)
    )
    assert ds.count_samples() == manual


@skip_no_data
def test_count_samples_with_plies_filter_real_data(tmp_path):
    """The slow (decompress) plies-filter path must agree with a manual
    parse-and-filter over the same files."""
    _copy_first_files(REAL_DATA_DIR, tmp_path, 4)
    ds = LCZeroV6Dataset(folders=[tmp_path], shuffle_buffer=0,
                         turn_start=5, max_plies_left=40.0)
    manual = 0
    for f in ds.list_files():
        for idx, rec in enumerate(dataset_mod.iter_v6_records_from_file(f)):
            if idx >= 5 and rec.plies_left <= 40.0:
                manual += 1
    assert ds.count_samples() == manual


@skip_no_data
def test_count_records_in_file_fast_matches_parse_real_data():
    """The gzip-trailer fast count equals the parsed record count."""
    from data.v6_parser import count_records_in_file, iter_v6_records_from_file

    for f in sorted(REAL_DATA_DIR.glob("*.gz"))[:6]:
        fast = count_records_in_file(f)
        actual = sum(1 for _ in iter_v6_records_from_file(f))
        assert fast == actual, f"{f.name}: fast={fast} actual={actual}"


# ---------------------------------------------------------------------------
# max_samples cap (the constant-budget mechanism for pooled splits)
# ---------------------------------------------------------------------------


def test_constructor_requires_exactly_one_source():
    # neither folders nor files
    with pytest.raises(ValueError):
        LCZeroV6Dataset()
    # both
    with pytest.raises(ValueError):
        LCZeroV6Dataset(folders=[], files=[])


def test_explicit_files_order_preserved():
    files = [Path("z.gz"), Path("a.gz"), Path("m.gz")]
    ds = LCZeroV6Dataset(files=files, shuffle_buffer=0)
    # list_files must NOT sort explicit files (the caller already game-shuffled).
    assert ds.list_files() == files


def test_max_samples_cap_no_shuffle(monkeypatch):
    """With shuffle_buffer=0, __iter__ yields exactly max_samples (the cap),
    even though the unfiltered stream has 10 samples."""
    ds = LCZeroV6Dataset(folders=[], shuffle_buffer=0, max_samples=4)
    _install_fake_stream(monkeypatch, ds)  # 6 + 4 = 10 samples available
    out = list(ds)
    assert len(out) == 4
    # No shuffle -> the first 4 in stream order (game A indices 0..3).
    assert out == _GAME_A[:4]


def test_max_samples_cap_with_shuffle(monkeypatch):
    """With a shuffle buffer, __iter__ still yields exactly max_samples."""
    ds = LCZeroV6Dataset(folders=[], shuffle_buffer=3, seed=0, max_samples=5)
    _install_fake_stream(monkeypatch, ds)
    out = list(ds)
    assert len(out) == 5
    # All yielded values come from the real pool (10 samples).
    assert set(out).issubset(set(_GAME_A + _GAME_B))


def test_max_samples_cap_larger_than_pool_yields_all(monkeypatch):
    """If the cap exceeds the (filtered) pool, yield everything available."""
    ds = LCZeroV6Dataset(folders=[], shuffle_buffer=0, max_samples=999)
    _install_fake_stream(monkeypatch, ds)
    out = list(ds)
    assert out == _GAME_A + _GAME_B  # all 10


def test_max_samples_cap_composes_with_filter(monkeypatch):
    """Filter first, then cap: max_plies_left keeps [7,6,5] (3), cap at 2."""
    ds = LCZeroV6Dataset(folders=[], shuffle_buffer=0, max_plies_left=7.0, max_samples=2)
    _install_fake_stream(monkeypatch, ds)
    out = list(ds)
    assert out == [7.0, 6.0]


def test_count_samples_respects_cap(monkeypatch):
    """count_samples returns min(available, cap) and stops early."""
    ds = LCZeroV6Dataset(folders=[], max_samples=4)
    monkeypatch.setattr(ds, "list_files", lambda: [Path("gameA.gz"), Path("gameB.gz")])
    _install_fake_counters(monkeypatch)  # gameA=6, gameB=4 records
    # No plies filter -> ISIZE path; total available = 10, cap = 4.
    assert ds.count_samples() == 4

    ds2 = LCZeroV6Dataset(folders=[], max_samples=999)
    monkeypatch.setattr(ds2, "list_files", lambda: [Path("gameA.gz"), Path("gameB.gz")])
    assert ds2.count_samples() == 10  # cap exceeds pool -> actual
