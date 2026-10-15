"""Render a small sample of V6 records as FEN strings + top-3 policy moves
for manual inspection.

Usage:
    python -m data.sanity_check

This script reads the first .gz file from raw_data/chunck_set_1/, takes the
first N=20 records, and for each prints:
    - FEN, by inverting our canonicalization to absolute orientation and
      constructing a python-chess Board (chess960=True since this dataset
      includes Fischer-random games).
    - Top-3 policy moves: probability, canonical UCI (per our
      models.policy_map ordering), absolute UCI (rank-flipped if
      black-to-move), and a legality flag against the rendered Board.
    - Value (W, D, L) and MLH.

At the end, an aggregate statistic is printed: fraction of top-1 / top-3
moves judged legal across the printed sample. If the canonical-1858
ordering of our policy_map matches V6's ordering, this fraction should be
high (close to 100 %). If it's near zero, the orderings differ.

Run it before training: plane-extraction bugs are silent and would train a
network on garbage. Inspect the FENs and confirm they look like real,
plausible chess positions and that the top-3 moves are sensible.
"""

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# python-chess is optional at import time; fail early with a clear message.
try:
    import chess  # type: ignore
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "python-chess is required for sanity_check. Install with:\n"
        "    pip install python-chess>=1.999"
    ) from e

from data.extract_planes import v6_to_34_planes
from data.extract_targets import v6_to_targets
from data.v6_parser import iter_v6_records_from_file
from models.policy_map import POLICY_INDEX_TO_UCI

DEFAULT_FOLDER = Path("raw_data/chunck_set_1")
N_TO_PRINT = 20


_PIECE_PLANES: List[Tuple[int, int]] = [
    (chess.PAWN, 0),
    (chess.KNIGHT, 1),
    (chess.BISHOP, 2),
    (chess.ROOK, 3),
    (chess.QUEEN, 4),
    (chess.KING, 5),
]


def planes_to_board(planes: np.ndarray) -> chess.Board:
    """Reconstruct a python-chess Board from our 34-plane canonical input.

    Inverts the canonicalization: when stm=black, the planes have black-
    perspective at the bottom, so we rank-flip and swap our<->opp piece
    planes to recover absolute orientation (white at rank 0).

    Castling rights become Chess960 rook-square bits, since FRC games
    appear in the data and standard a/h-file rook assumptions are wrong
    for those. The rook-finding heuristic walks each side's back rank to
    locate the queen-side and king-side rooks.
    """
    # Plane 30 is signed: +1 -> black-to-move, -1 -> white-to-move.
    black_to_move = float(planes[30, 0, 0]) > 0.0

    if not black_to_move:
        white_p = planes[0:6]
        black_p = planes[6:12]
        we_ooo, we_oo = bool(planes[26, 0, 0]), bool(planes[27, 0, 0])
        opp_ooo, opp_oo = bool(planes[28, 0, 0]), bool(planes[29, 0, 0])
        white_ooo, white_oo, black_ooo, black_oo = we_ooo, we_oo, opp_ooo, opp_oo
    else:
        white_p = np.flip(planes[6:12], axis=-2).copy()
        black_p = np.flip(planes[0:6], axis=-2).copy()
        we_ooo, we_oo = bool(planes[26, 0, 0]), bool(planes[27, 0, 0])
        opp_ooo, opp_oo = bool(planes[28, 0, 0]), bool(planes[29, 0, 0])
        # us = black; rank-flipping piece planes gave absolute orientation.
        # The flag still says "we (= black) can castle", so map "we" to black.
        black_ooo, black_oo, white_ooo, white_oo = we_ooo, we_oo, opp_ooo, opp_oo

    rule50 = int(round(float(planes[31, 0, 0]) * 100))
    board = chess.Board.empty(chess960=True)

    for piece_type, ch in _PIECE_PLANES:
        for r in range(8):
            for f in range(8):
                if white_p[ch, r, f] == 1.0:
                    board.set_piece_at(chess.square(f, r), chess.Piece(piece_type, chess.WHITE))
                if black_p[ch, r, f] == 1.0:
                    board.set_piece_at(chess.square(f, r), chess.Piece(piece_type, chess.BLACK))

    board.turn = chess.BLACK if black_to_move else chess.WHITE
    board.halfmove_clock = rule50
    board.fullmove_number = 1  # not in our data; arbitrary

    castling_mask = 0

    def _find_rook(rank: int, color: chess.Color, queenside: bool) -> int:
        rng = range(8) if queenside else range(7, -1, -1)
        for f in rng:
            sq = chess.square(f, rank)
            piece = board.piece_at(sq)
            if piece is not None and piece.piece_type == chess.ROOK and piece.color == color:
                return sq
        return -1

    if white_ooo:
        sq = _find_rook(0, chess.WHITE, queenside=True)
        if sq >= 0:
            castling_mask |= chess.BB_SQUARES[sq]
    if white_oo:
        sq = _find_rook(0, chess.WHITE, queenside=False)
        if sq >= 0:
            castling_mask |= chess.BB_SQUARES[sq]
    if black_ooo:
        sq = _find_rook(7, chess.BLACK, queenside=True)
        if sq >= 0:
            castling_mask |= chess.BB_SQUARES[sq]
    if black_oo:
        sq = _find_rook(7, chess.BLACK, queenside=False)
        if sq >= 0:
            castling_mask |= chess.BB_SQUARES[sq]

    board.castling_rights = castling_mask
    return board


def canonical_to_absolute_uci(uci: str, black_to_move: bool) -> str:
    """Convert a canonical-perspective UCI to absolute UCI by rank-flipping
    when stm=black. The from-file and to-file are unchanged."""
    if not black_to_move:
        return uci

    def _flip(sq: str) -> str:
        return sq[0] + str(9 - int(sq[1]))

    return _flip(uci[:2]) + _flip(uci[2:4]) + uci[4:]


def _move_legal(board: chess.Board, uci: str) -> bool:
    """Return True if `uci` parses to a legal move in `board`. Robust to
    Chess960 castling encoding edge cases."""
    try:
        move = board.parse_uci(uci)
    except (chess.IllegalMoveError, chess.InvalidMoveError, ValueError):
        return False
    return move in board.legal_moves


def main() -> int:
    folder = DEFAULT_FOLDER
    files = sorted(folder.glob("*.gz"))
    if not files:
        print(f"ERROR: no .gz files in {folder}", file=sys.stderr)
        return 1

    print(f"=== sanity_check: reading first {N_TO_PRINT} records of {files[0].name} ===\n")

    legal_top1 = 0
    legal_top3_any = 0
    legal_top3_count = 0
    rendered = 0

    for i, rec in enumerate(iter_v6_records_from_file(files[0])):
        if i >= N_TO_PRINT:
            break

        planes = v6_to_34_planes(rec)
        targets = v6_to_targets(rec)

        try:
            board = planes_to_board(planes)
            fen = board.fen()
        except Exception as exc:
            board = None
            fen = f"<RECONSTRUCTION ERROR: {exc}>"

        probs = targets["policy"]
        # argsort works correctly even with -1 sentinels (negatives sink).
        top3 = np.argsort(-probs)[:3].tolist()
        black_to_move = float(planes[30, 0, 0]) > 0.0

        top3_lines: List[str] = []
        top3_legality = []
        for rank, idx in enumerate(top3):
            canon = POLICY_INDEX_TO_UCI[idx]
            absolute = canonical_to_absolute_uci(canon, black_to_move)
            if board is not None:
                ok = _move_legal(board, absolute)
            else:
                ok = False
            top3_legality.append(ok)
            tag = "OK " if ok else "ILL"
            top3_lines.append(
                f"    [{tag}] rank{rank+1}: prob={probs[idx]:.4f}  "
                f"canon={canon:<6} abs={absolute}"
            )

        rendered += 1
        if top3_legality[0]:
            legal_top1 += 1
        if any(top3_legality):
            legal_top3_any += 1
        legal_top3_count += sum(top3_legality)

        value = targets["value"]
        mlh = targets["mlh"]

        print(f"--- Position {i} ---")
        print(f"  FEN: \n{fen}")
        print(f"  STM: {'black' if black_to_move else 'white'},  rule50: {rec.rule50_count}")
        print(f"  Top-3 policy:")
        for line in top3_lines:
            print(line)
        print(
            f"  Value (W,D,L): ({value[0]:.3f}, {value[1]:.3f}, {value[2]:.3f})"
            f"   MLH: {mlh:.2f} plies"
        )
        print()

    print("=" * 60)
    print("AGGREGATE LEGALITY (across the printed sample):")
    if rendered:
        print(f"  top-1 legal:        {legal_top1}/{rendered}  ({100*legal_top1/rendered:.0f}%)")
        print(f"  any-of-top-3 legal: {legal_top3_any}/{rendered}  ({100*legal_top3_any/rendered:.0f}%)")
        print(f"  total top-3 legal:  {legal_top3_count}/{3*rendered}  ({100*legal_top3_count/(3*rendered):.0f}%)")
    print()
    print("If most top-3 moves are flagged ILL, the models.policy_map ordering")
    print("does NOT match V6's ordering.")
    print()
    print("Inspect the FENs above: each should look like a real, plausible")
    print("chess position, with sensible (legal) top-3 moves.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
