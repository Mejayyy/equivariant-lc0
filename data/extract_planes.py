"""Convert V6Records into our 34-plane (34, 8, 8) input tensors.

Channel layout (from CLAUDE.md):
     0..5   our pieces (P, N, B, R, Q, K), current ply
     6..11  opp pieces, current ply
    12      repetition flag, current ply
    13..18  our pieces, 1 ply ago
    19..24  opp pieces, 1 ply ago
    25      repetition flag, 1 ply ago
    26      we can castle queenside    (we OOO)
    27      we can castle kingside     (we OO)
    28      opp can castle queenside
    29      opp can castle kingside
    30      side-to-move (1 if black, else 0)
    31      rule50_count / 100.0
    32      all zeros
    33      all ones

Bit-to-square convention (LCZero):
    Bit i of a uint64 plane sets square at (rank = i // 8, file = 7 - i % 8).
    I.e. files are MSB-first within each rank, ranks are LSB-first overall.

Canonicalization (verified empirically against the actual records on disk):
    The V6 records — even with input_format == 1 — store planes and
    castling already canonicalized so that the side-to-move's pieces are at
    the bottom (us-king is always at canonical e1). The "us"/"them" naming
    in V6 is always relative to side-to-move. We therefore copy planes
    directly without any swap-and-flip.

    Plane 30 still records the original side-to-move (1 if black) so the
    network can use that information; it does not require re-orienting the
    rest of the input.
"""

import numpy as np

from .v6_parser import V6Record


_BIT_INDICES = np.arange(64, dtype=np.uint64)
_BIT_RANKS = (_BIT_INDICES // 8).astype(np.int64)
_BIT_FILES = (7 - (_BIT_INDICES % 8)).astype(np.int64)


def _bitboard_to_8x8(bb: int) -> np.ndarray:
    out = np.zeros((8, 8), dtype=np.float32)
    if bb == 0:
        return out
    set_mask = ((np.uint64(bb) >> _BIT_INDICES) & np.uint64(1)).astype(bool)
    if set_mask.any():
        out[_BIT_RANKS[set_mask], _BIT_FILES[set_mask]] = 1.0
    return out


def _decode_13_planes(planes_u64: np.ndarray) -> np.ndarray:
    """Decode 13 V6 planes into a (13, 8, 8) float32 array."""
    out = np.zeros((13, 8, 8), dtype=np.float32)
    for i in range(12):
        out[i] = _bitboard_to_8x8(int(planes_u64[i]))
    # Repetition plane (index 12): all-zeros for non-repetition, all-ones
    # for repetition.
    rep_bb = int(planes_u64[12])
    out[12] = 1.0 if rep_bb else 0.0
    return out


def get_side_to_move(record: V6Record) -> int:
    """Return 1 if black-to-move, else 0.

    For input_format == 1, the V6 `side_to_move_or_enpassant` byte directly
    encodes side-to-move (0 = white, 1 = black). For canonical formats
    (input_format >= 3) the same information is in `invariance_info` bit 7.
    """
    if record.input_format == 1:
        return record.side_to_move_or_enpassant & 1
    return (record.invariance_info >> 7) & 1


def v6_to_34_planes(record: V6Record) -> np.ndarray:
    out = np.zeros((34, 8, 8), dtype=np.float32)

    out[:13] = _decode_13_planes(record.planes[:13])
    out[13:26] = _decode_13_planes(record.planes[13:26])

    out[26] = float(record.castling_us_ooo)
    out[27] = float(record.castling_us_oo)
    out[28] = float(record.castling_them_ooo)
    out[29] = float(record.castling_them_oo)

    # Plane 30 (side-to-move) is encoded as a SIGNED scalar:
    #   white-to-move -> -1.0,  black-to-move -> +1.0
    # This makes sigma_c act on plane 30 as a pure SIGN FLIP
    # (x[30] -> -x[30]) — a linear operation, much cleaner to reason
    # about than the prior `1 - x[30]` affine flip on a {0, 1}
    # encoding. See PROGRESS.md for the discussion and CLAUDE.md's
    # "Input Channel Specification" for the canonical layout.
    out[30] = 1.0 if get_side_to_move(record) else -1.0
    out[31] = record.rule50_count / 100.0
    out[32] = 0.0
    out[33] = 1.0
    return out


__all__ = [
    "v6_to_34_planes",
    "get_side_to_move",
]
