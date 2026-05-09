"""LCZero 1858-move policy index.

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
- Under-promotion: a 5-char UCI ``<from><to><piece>`` where ``<piece> in
  {q, r, b}``. These are pawn promotions specifically — 22 file-direction
  combos (NW, N, NE for files b..g; only N and one diagonal for a/h) ×
  3 promotion pieces. Knight promotion is NOT separately encoded (it
  collides with the knight-move slot ``a7b8`` etc.); LCZero's policy
  simply does not predict knight-promotion as a distinct option, and the
  V6 data follows the same convention.

Each entry is converted to a ``(from_sq, to_sq, promo)`` tuple where
``from_sq, to_sq`` are 0..63 indices (``sq = rank * 8 + file``;
a1=0, h8=63) and ``promo`` is one of ``'', 'q', 'r', 'b'``.
"""

from typing import Dict, List, Tuple

from ._lczero_policy_index import policy_index as _LCZERO_UCI_LIST

POLICY_INDEX_SIZE = 1858

Move = Tuple[int, int, str]


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


# --- Vendored canonical mapping ---------------------------------------------

POLICY_INDEX_TO_UCI: List[str] = list(_LCZERO_UCI_LIST)
assert len(POLICY_INDEX_TO_UCI) == POLICY_INDEX_SIZE, (
    f"vendored LCZero policy_index has {len(POLICY_INDEX_TO_UCI)} entries, "
    f"expected {POLICY_INDEX_SIZE}"
)
assert len(set(POLICY_INDEX_TO_UCI)) == POLICY_INDEX_SIZE, "non-unique UCI strings"

POLICY_INDEX_TO_MOVE: List[Move] = [_parse_uci(u) for u in POLICY_INDEX_TO_UCI]
MOVE_TO_POLICY_INDEX: Dict[Move, int] = {m: i for i, m in enumerate(POLICY_INDEX_TO_MOVE)}
UCI_TO_POLICY_INDEX: Dict[str, int] = {u: i for i, u in enumerate(POLICY_INDEX_TO_UCI)}

# Promotion piece set (for tests / external use). LCZero encodes only q, r, b
# as separate slots; knight promotions share their slot with the knight move
# at the same (from, to).
PROMO_PIECES = ('q', 'r', 'b')


__all__ = [
    "POLICY_INDEX_SIZE",
    "POLICY_INDEX_TO_MOVE",
    "MOVE_TO_POLICY_INDEX",
    "POLICY_INDEX_TO_UCI",
    "UCI_TO_POLICY_INDEX",
    "PROMO_PIECES",
    "square_to_uci",
    "move_to_uci",
]
