"""Download the LCZero V6 self-play data used in this project.

The experiments use 18 consecutive hourly archives of the public LCZero
`training-run2-test91` run (121,201 games, 14.3M positions, 3.0 GB).
Each archive is streamed from storage.lczero.org and its per-game `.gz`
files are written to `raw_data/<folder>/` -- the layout the data pipeline
expects (the trainer pools every `.gz` under `raw_data/`, whatever the
sub-folder names).

The five archives that formed the original fixed train/val/test split
keep their historical folder names `chunck_set_1..5` (the spelling is
what the code and tests reference); the other 13 use the archive name.

Usage:
    python scripts/download_data.py            # all 18 archives (3.0 GB)
    python scripts/download_data.py --limit 1  # only raw_data/chunck_set_1
                                               # (179 MB): enough for the
                                               # tests and the sanity check

Already-downloaded folders are skipped, so an interrupted download can
simply be restarted.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

BASE_URL = "https://storage.lczero.org/files/training_data/test91"

# (target folder, archive stem, archive size in bytes, games in archive).
# chunck_set_1 comes first because the tests and data/sanity_check.py read it.
ARCHIVES = [
    ("chunck_set_1", "training-run2-test91-20260509-0817", 178_667_520, 7198),
    ("chunck_set_2", "training-run2-test91-20260509-0917", 190_730_240, 7628),
    ("chunck_set_3", "training-run2-test91-20260509-1017", 170_188_800, 6908),
    ("chunck_set_4", "training-run2-test91-20260509-1117", 214_067_200, 8762),
    ("chunck_set_5", "training-run2-test91-20260509-1217", 233_820_160, 9419),
] + [
    (stem, stem, size, games)
    for stem, size, games in [
        ("training-run2-test91-20260508-1917", 179_374_080, 7139),
        ("training-run2-test91-20260508-2017", 91_576_320, 3993),
        ("training-run2-test91-20260508-2117", 182_712_320, 7317),
        ("training-run2-test91-20260508-2217", 182_927_360, 7353),
        ("training-run2-test91-20260508-2317", 183_511_040, 7329),
        ("training-run2-test91-20260509-0017", 164_208_640, 6551),
        ("training-run2-test91-20260509-0117", 21_975_040, 1041),
        ("training-run2-test91-20260509-0217", 181_493_760, 7302),
        ("training-run2-test91-20260509-0317", 182_743_040, 7308),
        ("training-run2-test91-20260509-0417", 183_050_240, 7329),
        ("training-run2-test91-20260509-0517", 180_807_680, 7176),
        ("training-run2-test91-20260509-0617", 98_406_400, 4198),
        ("training-run2-test91-20260509-0717", 181_411_840, 7250),
    ]
]

# Archive members are "<archive-stem>/training.<game id>.gz"; only the
# basename is used, so nothing can be written outside the target folder.
_GAME_FILE = re.compile(r"training\.\d+\.gz")


class _ProgressReader:
    """File-like wrapper around an HTTP response that counts bytes read
    and prints a progress line."""

    def __init__(self, raw, total: int, label: str):
        self._raw = raw
        self.total = total
        self.read_bytes = 0
        self._label = label
        self._last_pct = -1

    def read(self, n: int = -1) -> bytes:
        chunk = self._raw.read(n)
        self.read_bytes += len(chunk)
        pct = int(100 * self.read_bytes / self.total) if self.total else 100
        if pct != self._last_pct:
            self._last_pct = pct
            mb = self.read_bytes / 1e6
            print(f"\r  {self._label}: {mb:7.1f} / {self.total / 1e6:.1f} MB ({pct:3d}%)",
                  end="", flush=True)
        return chunk


def download_archive(root: Path, folder: str, stem: str, size: int, games: int,
                     base_url: str = BASE_URL) -> int:
    """Stream one archive into ``root/folder``. Returns the number of games
    written. Writes into a ``.partial`` folder first and renames it on
    success, so a folder that exists is always complete."""
    target = root / folder
    partial = root / f"{folder}.partial"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)

    url = f"{base_url}/{stem}.tar"
    n_games = 0
    with urllib.request.urlopen(url) as response:
        reader = _ProgressReader(response, size, folder)
        with tarfile.open(fileobj=reader, mode="r|") as archive:
            for member in archive:
                name = Path(member.name).name
                if not member.isfile() or not _GAME_FILE.fullmatch(name):
                    continue
                src = archive.extractfile(member)
                with open(partial / name, "wb") as out:
                    shutil.copyfileobj(src, out)
                n_games += 1
    print()

    if reader.read_bytes != size:
        raise RuntimeError(
            f"{stem}.tar: received {reader.read_bytes:,} bytes, expected {size:,}"
        )
    if n_games != games:
        print(f"  warning: {stem}.tar has {n_games} games; the experiments "
              f"used {games} (the archive may have changed on the server)")
    partial.rename(target)
    return n_games


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default="raw_data",
                        help="output folder (default: raw_data)")
    parser.add_argument("--limit", type=int, default=None,
                        help="download only the first N archives "
                             "(--limit 1 = chunck_set_1, for tests)")
    parser.add_argument("--base-url", default=BASE_URL, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    root = Path(args.root)
    archives = ARCHIVES[: args.limit] if args.limit else ARCHIVES
    todo = [a for a in archives if not (root / a[0]).is_dir()]
    if not todo:
        print(f"all {len(archives)} archive(s) already present in {root}/")
        return 0
    total_mb = sum(a[2] for a in todo) / 1e6
    print(f"downloading {len(todo)} archive(s), {total_mb:,.0f} MB -> {root}/")
    for folder, stem, size, games in todo:
        n = download_archive(root, folder, stem, size, games, args.base_url)
        print(f"  {folder}: {n} games")
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
