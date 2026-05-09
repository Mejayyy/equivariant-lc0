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
"""

import random
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .extract_planes import v6_to_34_planes
from .extract_targets import v6_to_targets
from .v6_parser import V6Record, iter_v6_records_from_file

Sample = Tuple[torch.Tensor, dict]


class LCZeroV6Dataset(IterableDataset):
    def __init__(
        self,
        folders: Iterable[Union[str, Path]],
        shuffle_buffer: int = 8192,
        seed: int = 0,
    ):
        super().__init__()
        self.folders: List[Path] = [Path(f) for f in folders]
        self.shuffle_buffer = int(shuffle_buffer)
        self.seed = int(seed)

    def list_files(self) -> List[Path]:
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

    def _iter_raw(self) -> Iterator[Sample]:
        for f in self.list_files():
            for rec in iter_v6_records_from_file(f):
                yield self._record_to_sample(rec)

    def __iter__(self) -> Iterator[Sample]:
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


__all__ = ["LCZeroV6Dataset"]
