"""LCZero V6 binary record parser.

V6 record layout (8356 bytes, little-endian):

    offset  size  field
    ------  ----  ----------------------------------------------------------
         0     4  version (uint32) — must be 6
         4     4  input_format (uint32)
         8  7432  probabilities[1858] (float32) — MCTS visit distribution.
                  Illegal-move slots store -1.0 as sentinel.
      7440   832  planes[104] (uint64) — 8 history positions x 13 bitboards.
                  13 planes per ply: 6 our pieces (P,N,B,R,Q,K),
                  6 opp pieces (P,N,B,R,Q,K), 1 repetition flag.
      8272     4  castling_us_ooo / castling_us_oo /
                  castling_them_ooo / castling_them_oo (4x uint8)
      8276     1  side_to_move_or_enpassant (uint8)
      8277     1  rule50_count (uint8)
      8278     1  invariance_info (uint8) — bitfield, semantics per
                  lczero-training/tf/chunkparser.py:
                    bit 0: flip transform     (input_format >= 3)
                    bit 1: mirror transform   (input_format >= 3)
                    bit 2: transpose transform (input_format >= 3)
                    bit 3: best_q is for proven best move
                    bit 4: max game length exceeded
                    bit 5: game adjudicated
                    bit 6: marked for deletion
                    bit 7: side-to-move        (input_format >= 3)
                  For input_format == 1, bits 0/1/2/7 are unused; side-to-
                  move comes from `side_to_move_or_enpassant` directly.
      8279     1  dep_result (int8) — deprecated.
      8280    60  15 floats: root_q, best_q, root_d, best_d, root_m, best_m,
                  plies_left, result_q, result_d, played_q, played_d,
                  played_m, orig_q, orig_d, orig_m
      8340     4  visits (uint32)
      8344     2  played_idx (uint16) — index of move actually played
      8346     2  best_idx (uint16) — index of move with best q
      8348     8  reserved
"""

import gzip
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Union

import numpy as np

V6_RECORD_SIZE = 8356
_V6_STRUCT_FMT = '<II7432s832sBBBBBBBbfffffffffffffffIHH8s'
assert struct.calcsize(_V6_STRUCT_FMT) == V6_RECORD_SIZE
_V6_STRUCT = struct.Struct(_V6_STRUCT_FMT)

# Byte offset of the `plies_left` float within a record. The 15 floats
# start after II + probabilities + planes + 7 uint8 + 1 int8; plies_left
# is the 7th float (root_q, best_q, root_d, best_d, root_m, best_m,
# plies_left). Derived (not hand-counted) so it stays correct if the
# prefix layout ever changes.
_PLIES_LEFT_OFFSET = struct.calcsize('<II7432s832sBBBBBBBbffffff')  # == 8304
assert _PLIES_LEFT_OFFSET == 8304


@dataclass
class V6Record:
    version: int
    input_format: int
    probabilities: np.ndarray   # (1858,) float32
    planes: np.ndarray          # (104,) uint64

    castling_us_ooo: int
    castling_us_oo: int
    castling_them_ooo: int
    castling_them_oo: int

    side_to_move_or_enpassant: int
    rule50_count: int
    invariance_info: int
    dep_result: int

    root_q: float
    best_q: float
    root_d: float
    best_d: float
    root_m: float
    best_m: float
    plies_left: float
    result_q: float
    result_d: float
    played_q: float
    played_d: float
    played_m: float
    orig_q: float
    orig_d: float
    orig_m: float

    visits: int
    played_idx: int
    best_idx: int


def parse_v6_record(buf: bytes) -> V6Record:
    if len(buf) != V6_RECORD_SIZE:
        raise ValueError(f"Expected {V6_RECORD_SIZE} bytes, got {len(buf)}")
    fields = _V6_STRUCT.unpack(buf)
    (version, input_format, probs_b, planes_b,
     c_us_ooo, c_us_oo, c_them_ooo, c_them_oo,
     stm_or_ep, rule50, invariance, dep_result,
     root_q, best_q, root_d, best_d, root_m, best_m,
     plies_left, result_q, result_d,
     played_q, played_d, played_m,
     orig_q, orig_d, orig_m,
     visits, played_idx, best_idx, _reserved) = fields

    if version != 6:
        raise ValueError(f"Unsupported V6 record version: {version}")

    return V6Record(
        version=version,
        input_format=input_format,
        probabilities=np.frombuffer(probs_b, dtype='<f4').copy(),
        planes=np.frombuffer(planes_b, dtype='<u8').copy(),
        castling_us_ooo=c_us_ooo, castling_us_oo=c_us_oo,
        castling_them_ooo=c_them_ooo, castling_them_oo=c_them_oo,
        side_to_move_or_enpassant=stm_or_ep,
        rule50_count=rule50,
        invariance_info=invariance,
        dep_result=dep_result,
        root_q=root_q, best_q=best_q, root_d=root_d, best_d=best_d,
        root_m=root_m, best_m=best_m,
        plies_left=plies_left,
        result_q=result_q, result_d=result_d,
        played_q=played_q, played_d=played_d, played_m=played_m,
        orig_q=orig_q, orig_d=orig_d, orig_m=orig_m,
        visits=visits, played_idx=played_idx, best_idx=best_idx,
    )


def iter_v6_records_from_file(path: Union[str, Path]) -> Iterator[V6Record]:
    """Yield V6Records from a single gzip-compressed chunk file (one game's positions)."""
    with gzip.open(path, 'rb') as f:
        while True:
            buf = f.read(V6_RECORD_SIZE)
            if not buf:
                return
            if len(buf) != V6_RECORD_SIZE:
                raise ValueError(
                    f"Truncated V6 record in {path}: got {len(buf)} bytes "
                    f"(expected {V6_RECORD_SIZE})"
                )
            yield parse_v6_record(buf)


def count_records_in_file(path: Union[str, Path]) -> int:
    """Number of V6 records in one `.gz` chunk, WITHOUT decompressing.

    Uses the gzip ISIZE trailer (last 4 bytes = uncompressed size mod
    2^32). LCZero chunks are single-member gzip streams of one game,
    far under 4 GB, and every record is exactly ``V6_RECORD_SIZE``
    bytes, so ``ISIZE // V6_RECORD_SIZE`` is the exact record count.

    Falls back to a full decompress-and-count if the trailer does not
    cleanly divide by the record size (e.g. a multi-member gzip file or
    a truncated trailer), so correctness never depends on the fast path.
    """
    try:
        with open(path, "rb") as fh:
            fh.seek(-4, 2)
            isize = struct.unpack("<I", fh.read(4))[0]
        if isize > 0 and isize % V6_RECORD_SIZE == 0:
            return isize // V6_RECORD_SIZE
    except OSError:
        pass

    # Fallback: decompress and count record-sized chunks.
    n = 0
    with gzip.open(path, "rb") as f:
        while True:
            buf = f.read(V6_RECORD_SIZE)
            if not buf:
                break
            if len(buf) != V6_RECORD_SIZE:
                raise ValueError(
                    f"Truncated V6 record in {path}: got {len(buf)} bytes "
                    f"(expected {V6_RECORD_SIZE})"
                )
            n += 1
    return n


def iter_plies_left_from_file(path: Union[str, Path]) -> Iterator[float]:
    """Yield only the ``plies_left`` float of each record in a chunk.

    Decompresses (so it pays the gzip cost) but parses just the single
    float at ``_PLIES_LEFT_OFFSET`` instead of building a full
    ``V6Record`` — much cheaper than ``iter_v6_records_from_file`` when
    counting positions under a ``max_plies_left`` filter.
    """
    with gzip.open(path, "rb") as f:
        while True:
            buf = f.read(V6_RECORD_SIZE)
            if not buf:
                return
            if len(buf) != V6_RECORD_SIZE:
                raise ValueError(
                    f"Truncated V6 record in {path}: got {len(buf)} bytes "
                    f"(expected {V6_RECORD_SIZE})"
                )
            yield struct.unpack_from("<f", buf, _PLIES_LEFT_OFFSET)[0]


__all__ = [
    "V6_RECORD_SIZE",
    "V6Record",
    "parse_v6_record",
    "iter_v6_records_from_file",
    "count_records_in_file",
    "iter_plies_left_from_file",
]
