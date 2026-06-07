"""LCZero V6 IterableDataset.

Walks the configured folders for `.gz` chunk files (each holds one game's
positions in V6 binary format), parses records, and yields
`(planes_tensor, targets_dict)` pairs.

Shapes:
    planes_tensor:  torch.float32 (34, 8, 8)
    targets["policy"]: torch.float32 (1858,)   — V6 -1 sentinels preserved
    targets["value"]:  torch.float32 (3,)      — [W, D, L]
    targets["mlh"]:    torch.float32 ()        — plies remaining (scalar)

Set `shuffle_buffer=0` to iterate files-and-records deterministically (val
and test). For training, set a positive integer (default 8192) for
reservoir shuffle across the stream.

Game-phase filtering (for training models focused on a slice of the game,
e.g. endgames). Each `.gz` is exactly one game with records stored in ply
order, and record 0 is always the game start (verified: side-to-move =
white, rule50 = 0, plies_left maximal). So the record index within a file
IS the ply-from-game-start (0-indexed, half-moves). Two optional knobs,
applied BEFORE the shuffle buffer:

    turn_start:     skip the first `turn_start` records (plies) of every
                    game. A record is kept iff its in-game index
                    >= turn_start. Counted in HALF-MOVES (plies), so
                    turn_start=40 starts at the 41st half-move.
    max_plies_left: keep a record only if `plies_left <= max_plies_left`,
                    i.e. it is within that many plies of the game's END.
                    This is the truer "endgame" selector (distance to the
                    finish, independent of game length).

A record is yielded iff it passes BOTH filters. With both set you keep a
mid-to-end window: at least `turn_start` plies into the game AND at most
`max_plies_left` plies from the finish. Defaults (turn_start=0,
max_plies_left=None) are a no-op — the full stream is yielded.
"""

import random
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .extract_planes import v6_to_34_planes
from .extract_targets import v6_to_targets
from .v6_parser import V6Record, iter_v6_records_from_file

Sample = Tuple[torch.Tensor, dict]


class LCZeroV6Dataset(IterableDataset):
    """Iterates `(planes, targets)` over V6 chunk games.

    Provide EXACTLY ONE of:
      * ``folders`` — directories; their `*.gz` files are used in sorted
        order (legacy fixed-folder mode), or
      * ``files`` — an explicit list of `.gz` paths, used IN THE GIVEN
        ORDER (pooled-split mode; the caller has already shuffled games).

    ``max_samples`` caps how many *filtered* samples a single pass
    (epoch) yields. Combined with a large file pool this delivers a
    constant number of plies regardless of the `turn_start` /
    `max_plies_left` filters (the filter runs first, then the cap). If the
    filtered pool is smaller than ``max_samples`` the pass simply yields
    everything available (the trainer warns about the shortfall).
    """

    def __init__(
        self,
        folders: Optional[Iterable[Union[str, Path]]] = None,
        files: Optional[Iterable[Union[str, Path]]] = None,
        shuffle_buffer: int = 8192,
        seed: int = 0,
        turn_start: int = 0,
        max_plies_left: Optional[float] = None,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        if (folders is None) == (files is None):
            raise ValueError(
                "provide exactly one of `folders` or `files`"
            )
        self.folders: Optional[List[Path]] = (
            [Path(f) for f in folders] if folders is not None else None
        )
        # Explicit files keep their given order (already game-shuffled by the
        # caller); we do NOT re-sort them.
        self._files: Optional[List[Path]] = (
            [Path(f) for f in files] if files is not None else None
        )
        self.shuffle_buffer = int(shuffle_buffer)
        self.seed = int(seed)
        self.turn_start = int(turn_start)
        if self.turn_start < 0:
            raise ValueError(f"turn_start must be >= 0, got {self.turn_start}")
        self.max_plies_left = (
            None if max_plies_left is None else float(max_plies_left)
        )
        if self.max_plies_left is not None and self.max_plies_left < 0:
            raise ValueError(
                f"max_plies_left must be >= 0 or None, got {self.max_plies_left}"
            )
        self.max_samples = None if max_samples is None else int(max_samples)
        if self.max_samples is not None and self.max_samples < 0:
            raise ValueError(
                f"max_samples must be >= 0 or None, got {self.max_samples}"
            )

    def list_files(self) -> List[Path]:
        if self._files is not None:
            return self._files  # explicit order preserved (game-shuffled)
        files: List[Path] = []
        for folder in self.folders:
            files.extend(sorted(folder.glob("*.gz")))
        return files

    def _record_to_sample(self, rec: V6Record) -> Sample:
        planes = torch.from_numpy(v6_to_34_planes(rec))  # (34, 8, 8) float32
        t = v6_to_targets(rec)
        targets = {
            "policy": torch.from_numpy(t["policy"]),
            "value": torch.from_numpy(t["value"]),
            "mlh": torch.tensor(float(t["mlh"]), dtype=torch.float32),
        }
        return planes, targets

    def _keep_record(self, idx: int, rec: V6Record) -> bool:
        """Game-phase filter. `idx` is the record's in-game ply index
        (0-based). Keep iff at/after turn_start AND within max_plies_left
        of the game end."""
        if idx < self.turn_start:
            return False
        if self.max_plies_left is not None and rec.plies_left > self.max_plies_left:
            return False
        return True

    def _iter_raw(self) -> Iterator[Sample]:
        for f in self.list_files():
            # enumerate() gives the in-game ply index since each .gz is one
            # game in ply order starting at ply 0.
            for idx, rec in enumerate(iter_v6_records_from_file(f)):
                if not self._keep_record(idx, rec):
                    continue
                yield self._record_to_sample(rec)

    def count_samples(self) -> int:
        """Number of samples this dataset yields in one pass (honoring the
        turn_start / max_plies_left filters AND the ``max_samples`` cap),
        WITHOUT building tensors.

        Two paths:
          * No plies filter (max_plies_left is None): count records per
            file from the gzip trailer (no decompression) and subtract the
            skipped ``turn_start`` prefix. Near-instant.
          * Plies filter active: decompress each file and read only the
            ``plies_left`` float per record (no plane/target tensors), then
            apply both filters. Pays the gzip decompression cost.

        In both paths we stop as soon as the ``max_samples`` cap is reached,
        so when the pool is large this returns quickly (the cap, not the
        whole pool). The return value is ``min(available_filtered, cap)``.
        """
        # Local import keeps the dataset module's import surface small and
        # mirrors the lazy-import style used elsewhere.
        from .v6_parser import count_records_in_file, iter_plies_left_from_file

        cap = self.max_samples
        total = 0
        if self.max_plies_left is None:
            for f in self.list_files():
                n = count_records_in_file(f)
                if self.turn_start:
                    n = max(0, n - self.turn_start)
                total += n
                if cap is not None and total >= cap:
                    return cap
            return total

        for f in self.list_files():
            for idx, plies_left in enumerate(iter_plies_left_from_file(f)):
                if idx < self.turn_start:
                    continue
                if plies_left > self.max_plies_left:
                    continue
                total += 1
                if cap is not None and total >= cap:
                    return cap
        return total

    def _iter_stream(self) -> Iterator[Sample]:
        """The filtered sample stream, optionally ply-shuffled via a
        reservoir buffer. Does NOT apply the max_samples cap (that is done
        in __iter__)."""
        if self.shuffle_buffer <= 0:
            yield from self._iter_raw()
            return

        rng = random.Random(self.seed)
        buf: List[Sample] = []
        for sample in self._iter_raw():
            if len(buf) < self.shuffle_buffer:
                buf.append(sample)
                continue
            j = rng.randrange(self.shuffle_buffer)
            yield buf[j]
            buf[j] = sample
        rng.shuffle(buf)
        yield from buf

    def __iter__(self) -> Iterator[Sample]:
        cap = self.max_samples
        if cap is None:
            yield from self._iter_stream()
            return
        n = 0
        for sample in self._iter_stream():
            if n >= cap:
                return
            yield sample
            n += 1


__all__ = ["LCZeroV6Dataset"]
