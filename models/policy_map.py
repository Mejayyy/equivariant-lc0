"""LCZero-style 1858-move policy index.

Each entry is a unique chess move encoded as (from_sq, to_sq, promo), where
from_sq, to_sq are 0..63 (sq = rank*8 + file; a1=0, h8=63) and promo is one
of {'', 'n', 'b', 'r'}. Queen-promotion is encoded as the queen-direction
move (no explicit promotion slot) — only under-promotions get explicit
slots.

Counts:
    queen-style:    1456   (sum over squares of reachable distances in 8 dirs)
    knight:          336   (8 L-shapes per square, edge-clipped)
    under-promo:      66   (22 file directions on rank 7 x 3 pieces {n,b,r})
                    ----
    total:          1858

Order (self-consistent, used everywhere in this codebase):
    for from_sq in 0..63:
        for d in QUEEN_DIRS:                     # N, NE, E, SE, S, SW, W, NW
            for distance in 1..7: emit if in-board
        for d in KNIGHT_DIRS:                    # 8 L-shapes
            emit if in-board
        if from_rank == 6:                       # white-perspective rank 7
            for d in PROMO_DIRS:                 # NW, N, NE
                for piece in [n, b, r]:
                    emit if to-file in-board

DEFERRED to Phase 4: cross-check this ordering against LCZero V6
`probabilities[1858]`. If V6 uses a different ordering we will either
reorder this module to match OR apply a per-record remap when reading
V6 records. Tracked in PROGRESS.md "Open questions".
"""

from typing import Dict, List, Tuple

POLICY_INDEX_SIZE = 1858

QUEEN_DIRS: List[Tuple[int, int]] = [
    (1, 0),    # N
    (1, 1),    # NE
    (0, 1),    # E
    (-1, 1),   # SE
    (-1, 0),   # S
    (-1, -1),  # SW
    (0, -1),   # W
    (1, -1),   # NW
]

KNIGHT_DIRS: List[Tuple[int, int]] = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1),
]

PROMO_DIRS: List[Tuple[int, int]] = [(1, -1), (1, 0), (1, 1)]   # NW, N, NE
PROMO_PIECES: List[str] = ['n', 'b', 'r']

Move = Tuple[int, int, str]


def square_to_uci(sq: int) -> str:
    return 'abcdefgh'[sq % 8] + str(sq // 8 + 1)


def move_to_uci(move: Move) -> str:
    from_sq, to_sq, promo = move
    return square_to_uci(from_sq) + square_to_uci(to_sq) + promo


def _generate_move_list() -> List[Move]:
    moves: List[Move] = []
    for from_sq in range(64):
        from_r, from_f = from_sq // 8, from_sq % 8

        for dr, df in QUEEN_DIRS:
            for dist in range(1, 8):
                tr, tf = from_r + dr * dist, from_f + df * dist
                if 0 <= tr < 8 and 0 <= tf < 8:
                    moves.append((from_sq, tr * 8 + tf, ''))

        for dr, df in KNIGHT_DIRS:
            tr, tf = from_r + dr, from_f + df
            if 0 <= tr < 8 and 0 <= tf < 8:
                moves.append((from_sq, tr * 8 + tf, ''))

        if from_r == 6:
            for dr, df in PROMO_DIRS:
                tr, tf = from_r + dr, from_f + df
                if 0 <= tr < 8 and 0 <= tf < 8:
                    for piece in PROMO_PIECES:
                        moves.append((from_sq, tr * 8 + tf, piece))

    return moves


POLICY_INDEX_TO_MOVE: List[Move] = _generate_move_list()
assert len(POLICY_INDEX_TO_MOVE) == POLICY_INDEX_SIZE, (
    f"Generator produced {len(POLICY_INDEX_TO_MOVE)} moves, expected {POLICY_INDEX_SIZE}"
)

MOVE_TO_POLICY_INDEX: Dict[Move, int] = {
    m: i for i, m in enumerate(POLICY_INDEX_TO_MOVE)
}
assert len(MOVE_TO_POLICY_INDEX) == POLICY_INDEX_SIZE, "duplicate moves in generator"

POLICY_INDEX_TO_UCI: List[str] = [move_to_uci(m) for m in POLICY_INDEX_TO_MOVE]

UCI_TO_POLICY_INDEX: Dict[str, int] = {
    uci: i for i, uci in enumerate(POLICY_INDEX_TO_UCI)
}
assert len(UCI_TO_POLICY_INDEX) == POLICY_INDEX_SIZE, "duplicate UCI strings"


__all__ = [
    "POLICY_INDEX_SIZE",
    "POLICY_INDEX_TO_MOVE",
    "MOVE_TO_POLICY_INDEX",
    "POLICY_INDEX_TO_UCI",
    "UCI_TO_POLICY_INDEX",
    "square_to_uci",
    "move_to_uci",
]
