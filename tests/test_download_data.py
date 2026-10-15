"""scripts/download_data.py: archive streaming + extraction, without network
access (a local tar shaped like the LCZero archives is read through a
file:// URL, which urllib handles exactly like the HTTP stream)."""

import gzip
import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"
_SPEC = importlib.util.spec_from_file_location("download_data", _SCRIPT)
download_data = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(download_data)

STEM = "training-run2-test91-20990101-0017"


def _make_archive(folder: Path, members) -> int:
    """Write folder/STEM.tar with a top-level directory entry followed by
    `members` (name, bytes). Returns the archive size in bytes."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{STEM}.tar"
    with tarfile.open(path, "w") as tar:
        top = tarfile.TarInfo(STEM)
        top.type = tarfile.DIRTYPE
        tar.addfile(top)
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path.stat().st_size


def _games(n):
    return [(f"{STEM}/training.{1000 + i}.gz", gzip.compress(b"x" * (100 + i)))
            for i in range(n)]


def test_download_archive_writes_only_game_files(tmp_path):
    server = tmp_path / "server"
    members = _games(5) + [(f"{STEM}/README.txt", b"not a game")]
    size = _make_archive(server, members)
    root = tmp_path / "raw_data"

    n = download_data.download_archive(
        root, "chunck_set_9", STEM, size, games=5, base_url=server.as_uri())

    assert n == 5
    written = sorted(p.name for p in (root / "chunck_set_9").iterdir())
    assert written == [f"training.{1000 + i}.gz" for i in range(5)]
    # Content round-trips byte-for-byte.
    first = (root / "chunck_set_9" / "training.1000.gz").read_bytes()
    assert gzip.decompress(first) == b"x" * 100
    assert not (root / "chunck_set_9.partial").exists()


def test_download_archive_never_writes_outside_target(tmp_path):
    server = tmp_path / "server"
    evil = ("../../training.666.gz", gzip.compress(b"evil"))
    size = _make_archive(server, _games(2) + [evil])
    root = tmp_path / "data" / "raw_data"

    download_data.download_archive(
        root, "chunck_set_9", STEM, size, games=3, base_url=server.as_uri())

    # The traversal member is reduced to its basename inside the target.
    assert (root / "chunck_set_9" / "training.666.gz").exists()
    assert not (tmp_path / "training.666.gz").exists()
    assert not (tmp_path / "data" / "training.666.gz").exists()


def test_download_archive_rejects_truncated_size(tmp_path):
    server = tmp_path / "server"
    size = _make_archive(server, _games(3))
    root = tmp_path / "raw_data"

    with pytest.raises(RuntimeError, match="expected"):
        download_data.download_archive(
            root, "chunck_set_9", STEM, size + 512, games=3,
            base_url=server.as_uri())
    # No complete-looking folder is left behind.
    assert not (root / "chunck_set_9").exists()


def test_main_skips_folders_already_present(tmp_path, capsys):
    root = tmp_path / "raw_data"
    (root / "chunck_set_1").mkdir(parents=True)
    # base-url points nowhere: any download attempt would fail loudly.
    rc = download_data.main(["--root", str(root), "--limit", "1",
                             "--base-url", (tmp_path / "missing").as_uri()])
    assert rc == 0
    assert "already present" in capsys.readouterr().out


def test_manifest_matches_the_experiment_dataset():
    """18 archives, 121,201 games, chunck_set_1 first (tests read it)."""
    assert len(download_data.ARCHIVES) == 18
    assert sum(a[3] for a in download_data.ARCHIVES) == 121_201
    assert download_data.ARCHIVES[0][0] == "chunck_set_1"
    folders = [a[0] for a in download_data.ARCHIVES]
    assert len(set(folders)) == len(folders)
