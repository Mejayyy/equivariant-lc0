"""LCZero 1858-move policy index + 80-channel gather indices.

Canonical move ordering, vendored from
    https://github.com/LeelaChessZero/lczero-training/blob/master/tf/policy_index.py
into ``models/_lczero_policy_index.py``. This is the SAME ordering used in
the V6 ``probabilities[1858]`` field, so policy targets line up with our
``POLICY_INDEX_TO_UCI`` directly without any remap.

Encoding (1858 = 1792 non-promo + 66 promo):
- Non-promotion: a UCI string ``<from><to>`` where ``<from>`` and ``<to>``
  are squares like ``e2``. These cover all queen-style and knight-style
  moves that are valid on an 8x8 board (regardless of whose piece is on
  the from-square; e.g. ``a1a8`` is in the list because some piece could
  legally make that move in some position).
- Promotion: a 5-char UCI ``<from><to><piece>`` where ``<piece> in
  {q, r, b}``. 22 file-direction combos (NW, N, NE for files b..g;
  only N and one diagonal for a/h) x 3 promotion pieces. Knight
  promotion is NOT separately encoded (it collides with the knight-move
  slot ``a7b8`` etc.); LCZero's policy does not predict knight-promotion
  as a distinct option, and the V6 data follows the same convention.

Each entry is converted to a ``(from_sq, to_sq, promo)`` tuple where
``from_sq, to_sq`` are 0..63 indices (``sq = rank * 8 + file``;
a1=0, h8=63) and ``promo`` is one of ``'', 'q', 'r', 'b'``.

80-channel policy-head gather
-----------------------------
The baseline policy head produces an ``(80, 8, 8)`` tensor; we then
gather 1858 specific cells out of the 80*8*8 = 5120-element grid to get
the policy logits in the canonical V6 ordering. The 80 planes are:

    Planes  0..55: queen-style moves (8 directions x 7 distances)
    Planes 56..63: knight moves (8 directions)
    Planes 64..72: under-promotions (3 directions x 3 pieces: r, b, q)
                   (knight-promotion uses the corresponding knight-move
                   slot in planes 56..63; queen-promotion lives in the
                   q-slot here, written as e.g. "e7e8q".)

The 5120-flat-index ordering iterates ``channel * 64 + rank * 8 + file``.
``build_policy_gather_indices()`` returns a ``(1858,)`` int64 tensor
where entry ``i`` is the flat index of move ``POLICY_INDEX_TO_UCI[i]``
in that 5120-grid.

Source / credit for the gather logic: ported from
    https://github.com/Rocketknight1/minimal_lczero/blob/master/lc0_az_policy_map.py
which derives from the original lc0 AZ-policy-head PR
    https://github.com/LeelaChessZero/lc0/pull/712
"""

from typing import Dict, List, Optional, Tuple

import torch

from ._lczero_policy_index import policy_index as _LCZERO_UCI_LIST

POLICY_INDEX_SIZE = 1858

Move = Tuple[int, int, str]


# ---------------------------------------------------------------------------
# 1858 UCI list <-> index <-> (from, to, promo)
# ---------------------------------------------------------------------------


def _file_to_idx(c: str) -> int:
    return ord(c) - ord('a')


def _square_from_uci(s: str) -> int:
    return (int(s[1]) - 1) * 8 + _file_to_idx(s[0])


def square_to_uci(sq: int) -> str:
    return 'abcdefgh'[sq % 8] + str(sq // 8 + 1)


def move_to_uci(move: Move) -> str:
    from_sq, to_sq, promo = move
    return square_to_uci(from_sq) + square_to_uci(to_sq) + promo


def _parse_uci(uci: str) -> Move:
    from_sq = _square_from_uci(uci[:2])
    to_sq = _square_from_uci(uci[2:4])
    promo = uci[4:] if len(uci) > 4 else ''
    return from_sq, to_sq, promo


POLICY_INDEX_TO_UCI: List[str] = list(_LCZERO_UCI_LIST)
assert len(POLICY_INDEX_TO_UCI) == POLICY_INDEX_SIZE, (
    f"vendored LCZero policy_index has {len(POLICY_INDEX_TO_UCI)} entries, "
    f"expected {POLICY_INDEX_SIZE}"
)
assert len(set(POLICY_INDEX_TO_UCI)) == POLICY_INDEX_SIZE, "non-unique UCI strings"

POLICY_INDEX_TO_MOVE: List[Move] = [_parse_uci(u) for u in POLICY_INDEX_TO_UCI]
MOVE_TO_POLICY_INDEX: Dict[Move, int] = {m: i for i, m in enumerate(POLICY_INDEX_TO_MOVE)}
UCI_TO_POLICY_INDEX: Dict[str, int] = {u: i for i, u in enumerate(POLICY_INDEX_TO_UCI)}

PROMO_PIECES = ('q', 'r', 'b')


# ---------------------------------------------------------------------------
# 80-channel policy-head gather (lc0 AZ-style)
# ---------------------------------------------------------------------------

POLICY_NUM_CHANNELS = 80
_POLICY_FLAT_SIZE = POLICY_NUM_CHANNELS * 8 * 8  # 5120

_COLUMNS = "abcdefgh"
_ROWS = "12345678"
_COL = {c: i for i, c in enumerate(_COLUMNS)}
_ROW = {r: i for i, r in enumerate(_ROWS)}

# Underpromotion piece order (lc0 convention): rook, bishop, queen.
# The "q" slot covers queen-promotions written e.g. "e7e8q".
# Knight-promotion is encoded via the knight-move plane, NOT here.
_UNDERPROMO_PIECES = "rbq"

_QUEEN_DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
_QUEEN_VEC = {
    "N": (0, 1),  "NE": (1, 1),  "E": (1, 0),   "SE": (1, -1),
    "S": (0, -1), "SW": (-1, -1), "W": (-1, 0), "NW": (-1, 1),
}
# Knight directions reuse the 8 compass names but with knight vectors.
_KNIGHT_VEC = {
    "N":  (1, 2),   "NE": (2, 1),   "E":  (2, -1),  "SE": (1, -2),
    "S":  (-1, -2), "SW": (-2, -1), "W":  (-2, 1),  "NW": (-1, 2),
}


def _idx_to_pos(col: int, row: int) -> Optional[str]:
    if 0 <= col <= 7 and 0 <= row <= 7:
        return _COLUMNS[col] + _ROWS[row]
    return None


def _pos_to_idx(sq: str) -> Tuple[int, int]:
    return _COL[sq[0]], _ROW[sq[1]]


def _queen_dest(start: str, direction: str, steps: int) -> Optional[str]:
    c, r = _pos_to_idx(start)
    dc, dr = _QUEEN_VEC[direction]
    return _idx_to_pos(c + dc * steps, r + dr * steps)


def _knight_dest(start: str, direction: str) -> Optional[str]:
    c, r = _pos_to_idx(start)
    dc, dr = _KNIGHT_VEC[direction]
    return _idx_to_pos(c + dc, r + dr)


def _build_move_list_5120() -> List[str]:
    """Build the length-5120 list of UCI strings (or 'illegal') for each
    cell ``channel * 64 + rank * 8 + file`` of the (80, 8, 8) policy grid.

    Iteration order: outer = move type / direction, inner = (rank, file)
    in row-major order. This MUST match the lc0 AZ-policy-head layout —
    the 1858 gather indices below depend on it.
    """
    moves: List[str] = []

    # Planes 0..55: queen-style (8 dirs x 7 steps).
    for direction in _QUEEN_DIRS:
        for steps in range(1, 8):
            for r0 in _ROWS:
                for c0 in _COLUMNS:
                    start = c0 + r0
                    dest = _queen_dest(start, direction, steps)
                    moves.append("illegal" if dest is None else start + dest)

    # Planes 56..63: knight (8 directions).
    for direction in _QUEEN_DIRS:
        for r0 in _ROWS:
            for c0 in _COLUMNS:
                start = c0 + r0
                dest = _knight_dest(start, direction)
                moves.append("illegal" if dest is None else start + dest)

    # Planes 64..72: under-promotion (3 dirs x 3 pieces).
    for direction in ["NW", "N", "NE"]:
        for piece in _UNDERPROMO_PIECES:
            for r0 in _ROWS:
                for c0 in _COLUMNS:
                    if r0 != "7":
                        moves.append("illegal")
                        continue
                    start = c0 + r0
                    dest = _queen_dest(start, direction, 1)
                    if dest is None:
                        moves.append("illegal")
                    else:
                        moves.append(start + dest + piece)

    # Planes 73..79 are unused padding so the conv output sits at a clean
    # 80 channels. Fill them with 'illegal' so the flat index range stays
    # 80*8*8 = 5120; no canonical move ever gathers from these positions.
    while len(moves) < _POLICY_FLAT_SIZE:
        moves.append("illegal")

    assert len(moves) == _POLICY_FLAT_SIZE, (
        f"Expected {_POLICY_FLAT_SIZE} entries, got {len(moves)}"
    )
    return moves


def build_policy_gather_indices() -> torch.Tensor:
    """Return a ``(1858,)`` int64 tensor of flat indices into the (80, 8, 8)
    policy-head output, ordered to match ``POLICY_INDEX_TO_UCI``.

    Usage in the policy head::

        flat = policy_conv_out.view(B, 80 * 8 * 8)   # (B, 5120)
        logits = flat[:, gather_idx]                 # (B, 1858)

    Equivalent to a non-trainable lookup table from the policy-head's
    spatial / channel layout into the canonical V6 1858-element ordering.
    """
    move_list = _build_move_list_5120()
    uci_to_flat = {uci: idx for idx, uci in enumerate(move_list) if uci != "illegal"}

    gather: List[int] = []
    for uci in POLICY_INDEX_TO_UCI:
        if uci not in uci_to_flat:
            raise ValueError(
                f"UCI move '{uci}' from POLICY_INDEX_TO_UCI not found in the "
                f"80-channel grid. The policy_index list and the gather "
                f"layout have drifted apart."
            )
        gather.append(uci_to_flat[uci])

    indices = torch.tensor(gather, dtype=torch.int64)
    assert indices.shape == (POLICY_INDEX_SIZE,)
    assert len(set(gather)) == POLICY_INDEX_SIZE, "duplicate flat indices"
    return indices


def build_flat_to_policy_index() -> Dict[int, int]:
    """Inverse of ``build_policy_gather_indices``: ``{flat_idx: policy_idx}``
    for the 1858 legal moves. Useful for debugging."""
    gather = build_policy_gather_indices().tolist()
    return {flat: policy for policy, flat in enumerate(gather)}


__all__ = [
    "POLICY_INDEX_SIZE",
    "POLICY_NUM_CHANNELS",
    "POLICY_INDEX_TO_MOVE",
    "MOVE_TO_POLICY_INDEX",
    "POLICY_INDEX_TO_UCI",
    "UCI_TO_POLICY_INDEX",
    "PROMO_PIECES",
    "square_to_uci",
    "move_to_uci",
    "build_policy_gather_indices",
    "build_flat_to_policy_index",
]
