"""FEN <-> 34-plane encoding for the chess equivariance viz app.

This module is the bridge between a recognizable chess position (a
``python-chess`` Board / FEN) and the network's ``(34, 8, 8)`` input
tensor. It has two halves that are exact inverses of each other:

* ``board_to_planes`` / ``fen_to_planes`` -- the ENCODER. It reproduces
  ``data/extract_planes.py`` (the encoding spec) exactly, but starting
  from a FEN instead of a V6 record.
* ``planes_to_board`` -- the round-trip ORACLE, reused from
  ``data/sanity_check.py`` (the renderer the training data was checked
  with by hand). ``tests/test_app.py`` asserts
  ``planes_to_board(fen_to_planes(F))`` reproduces ``F``.

Design choices (must match data/extract_planes.py):

* **Canonical frame.** Planes are side-to-move-relative ("our"/"their")
  in LCZero's canonical orientation (side to move at the bottom). We get
  there with ``board.mirror()`` when it's Black to move -- that is exactly
  sigma_c (swap colors + flip ranks + swap castling). In the canonical
  board the side to move is always White, so "our" = canonical White
  pieces, "their" = canonical Black pieces.

* **Plane 30 (signed side-to-move).** White-to-move = -1.0,
  Black-to-move = +1.0, taken from the ORIGINAL ``board.turn`` (not the
  canonical one, which is always White). This is the single field that
  carries absolute color; sigma_c maps ``x[30] -> -x[30]``.

* **History (planes 13..25).** "1 ply ago", canonicalized to the CURRENT
  side-to-move's frame. With no real history (a freshly pasted FEN) we
  offer ``history="repeat"`` (default: copy the current position into the
  history slot -- a reasonable stand-in) or ``history="zeros"``. The V6
  format has no en-passant plane, so we never invent one.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

import chess

from data.sanity_check import planes_to_board

# piece type -> plane offset within a 6-plane block (P, N, B, R, Q, K),
# matching extract_planes.py channel order.
_PIECE_PLANES: List[Tuple[int, int]] = [
    (chess.PAWN, 0),
    (chess.KNIGHT, 1),
    (chess.BISHOP, 2),
    (chess.ROOK, 3),
    (chess.QUEEN, 4),
    (chess.KING, 5),
]

NUM_PLANES = 34


# ---------------------------------------------------------------------------
# ENCODER: board / FEN -> (34, 8, 8)
# ---------------------------------------------------------------------------


def _fill_piece_planes(out: np.ndarray, base: int, board: chess.Board, color: chess.Color) -> None:
    """Set ``out[base + ch, rank, file] = 1`` for each ``color`` piece.

    Square -> array-cell convention matches the planes_to_board oracle:
    a piece at ``chess.square(file, rank)`` lands at ``out[ch, rank, file]``
    (so cell ``[r, f]`` is rank ``r``, file ``f`` -- a1 at ``[0, 0]``).
    """
    for piece_type, ch in _PIECE_PLANES:
        for sq in board.pieces(piece_type, color):
            r = chess.square_rank(sq)
            f = chess.square_file(sq)
            out[base + ch, r, f] = 1.0


def board_to_planes(
    board: chess.Board,
    history: str = "repeat",
    prev_board: Optional[chess.Board] = None,
) -> np.ndarray:
    """Encode a ``chess.Board`` into the canonical ``(34, 8, 8)`` tensor.

    Args:
        board: the absolute position (source of truth).
        history: ``"repeat"`` (fill the 1-ply-ago slot with the current
            position when ``prev_board`` is None) or ``"zeros"`` (leave the
            history planes empty).
        prev_board: the real previous (1-ply-ago) absolute position, if
            known. When given it is canonicalized to the CURRENT side-to-
            move's frame and written into planes 13..25, regardless of the
            ``history`` flag.

    Returns:
        ``np.ndarray`` of shape ``(34, 8, 8)``, dtype ``float32``.
    """
    if history not in ("repeat", "zeros"):
        raise ValueError(f"history must be 'repeat' or 'zeros', got {history!r}")

    out = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    black_to_move = board.turn == chess.BLACK
    # board.mirror() == sigma_c: swap colors + flip ranks + swap castling/EP.
    canonical = board if board.turn == chess.WHITE else board.mirror()

    # Current ply: our pieces (0..5), their pieces (6..11).
    _fill_piece_planes(out, 0, canonical, chess.WHITE)
    _fill_piece_planes(out, 6, canonical, chess.BLACK)
    # Plane 12 (current repetition flag) stays 0 at a root position.

    # Castling 26..29 from the canonical board: our OOO/OO, their OOO/OO.
    out[26] = 1.0 if canonical.has_queenside_castling_rights(chess.WHITE) else 0.0
    out[27] = 1.0 if canonical.has_kingside_castling_rights(chess.WHITE) else 0.0
    out[28] = 1.0 if canonical.has_queenside_castling_rights(chess.BLACK) else 0.0
    out[29] = 1.0 if canonical.has_kingside_castling_rights(chess.BLACK) else 0.0

    # Plane 30 (signed side-to-move) from the ORIGINAL turn.
    out[30] = 1.0 if black_to_move else -1.0
    # rule50 normalized; constant planes.
    out[31] = board.halfmove_clock / 100.0
    out[32] = 0.0
    out[33] = 1.0

    # History planes 13..25 ("1 ply ago"), canonicalized to current frame.
    if prev_board is not None:
        prev_canon = prev_board if board.turn == chess.WHITE else prev_board.mirror()
        _fill_piece_planes(out, 13, prev_canon, chess.WHITE)
        _fill_piece_planes(out, 19, prev_canon, chess.BLACK)
        # Plane 25 (1-ply-ago repetition) stays 0.
    elif history == "repeat":
        # No real history: stand in with the current position (our/their/rep).
        out[13:26] = out[0:13]
    # history == "zeros": leave planes 13..25 at zero.

    return out


def fen_to_planes(
    fen: str,
    history: str = "repeat",
    prev_board: Optional[chess.Board] = None,
) -> np.ndarray:
    """Encode a FEN string into the canonical ``(34, 8, 8)`` tensor."""
    return board_to_planes(chess.Board(fen), history=history, prev_board=prev_board)


# ---------------------------------------------------------------------------
# ORACLE: (34, 8, 8) -> board / FEN
# planes_to_board (imported above) is the training-data sanity check's
# renderer, so the encoder is tested against the same inverse that was
# used to verify the V6 data.
# ---------------------------------------------------------------------------


def planes_to_fen(planes: np.ndarray) -> str:
    """Convenience: render planes straight to a FEN string."""
    return planes_to_board(planes).fen()


def planes_to_board_canonical(planes: np.ndarray) -> chess.Board:
    """Render the planes at FACE VALUE in the canonical frame, WITHOUT
    inverting the canonicalization: "our" planes (0..5) become White pieces
    at their array squares, "their" planes (6..11) become Black pieces.

    Use this to *visualize the group action*. ``planes_to_board`` (the
    round-trip oracle) un-canonicalizes — when it sees Black-to-move it
    mirrors back to absolute orientation, which CANCELS sigma_c's color-swap
    + rank-flip, so a sigma_c-transformed board would look unchanged. The
    canonical (face-value) render instead shows exactly what g did to the
    tensor: sigma_h = file mirror, sigma_c = color swap + rank flip,
    sigma_h_c = 180deg rotation.

    For a White-to-move source position this matches the real board; for a
    Black-to-move source it shows the net's-eye (canonical) view.
    """
    board = chess.Board.empty(chess960=True)
    for piece_type, ch in _PIECE_PLANES:
        for r in range(8):
            for f in range(8):
                if planes[ch, r, f] == 1.0:
                    board.set_piece_at(chess.square(f, r), chess.Piece(piece_type, chess.WHITE))
                if planes[6 + ch, r, f] == 1.0:
                    board.set_piece_at(chess.square(f, r), chess.Piece(piece_type, chess.BLACK))
    board.turn = chess.WHITE
    return board


__all__ = [
    "NUM_PLANES",
    "board_to_planes",
    "fen_to_planes",
    "planes_to_board",
    "planes_to_board_canonical",
    "planes_to_fen",
]
