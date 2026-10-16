"""Tests for the visualisation app (app/):

1. Encoder round-trip: ``planes_to_board(fen_to_planes(F))`` reproduces F
   on piece placement + side-to-move + castling, for a battery of FENs
   including >= 3 Black-to-move positions and some partial/zero castling.
2. uci <-> index round-trip on the vendored 1858-move list (decode then
   re-encode == identity, in both directions).

Plus a couple of targeted checks (plane-30 sign, history toggle) that
guard the conventions most likely to silently break the equivariance demo.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import chess

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from viz.encoding import board_to_planes, fen_to_planes, planes_to_board  # noqa: E402
from models.policy_map import (  # noqa: E402
    POLICY_INDEX_SIZE,
    POLICY_INDEX_TO_UCI,
    UCI_TO_POLICY_INDEX,
)


# ---------------------------------------------------------------------------
# Test FENs: 12 positions, 6 Black-to-move, several partial/zero castling.
# ---------------------------------------------------------------------------
TEST_FENS = [
    # 1. start position, full castling, White to move
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # 2. after 1.e4 -- Black to move, full castling
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    # 3. Sicilian after 1.e4 c5 2.Nf3 -- Black to move
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    # 4. rooks+kings, full castling, White to move
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    # 5. partial castling: White both, Black none, White to move
    "r3k2r/8/8/8/8/8/8/R3K2R w KQ - 0 1",
    # 6. partial castling: Black kingside only, Black to move
    "r3k2r/8/8/8/8/8/8/R3K2R b k - 0 1",
    # 7. no castling, kings only, White to move
    "8/8/8/4k3/8/8/4K3/8 w - - 0 1",
    # 8. no castling, kings only, Black to move
    "8/8/8/4k3/8/8/4K3/8 b - - 0 1",
    # 9. Italian-ish, Black to move, full castling, nonzero rule50
    "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 3",
    # 10. K+P endgame, Black to move, no castling
    "8/5k2/8/8/3P4/8/5K2/8 b - - 7 40",
    # 11. partial castling: White queenside + Black queenside, White to move
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w Qq - 0 1",
    # 12. castled middlegame, no castling rights, White to move
    "r2q1rk1/ppp2ppp/2n1bn2/2bpp3/4P3/2NP1N2/PPP1BPPP/R1BQ1RK1 w - - 6 8",
]

BLACK_TO_MOVE_FENS = [f for f in TEST_FENS if f.split()[1] == "b"]


def _placement_stm_castling(board: chess.Board):
    """The three fields the round-trip must preserve, normalized so a
    standard board and the chess960 oracle board compare equal."""
    return (
        board.board_fen(),
        board.turn,
        board.clean_castling_rights(),
    )


@pytest.mark.parametrize("fen", TEST_FENS)
def test_encoder_roundtrip(fen):
    """planes_to_board(fen_to_planes(F)) == F on placement + stm + castling."""
    original = chess.Board(fen)
    planes = fen_to_planes(fen)
    recovered = planes_to_board(planes)
    assert _placement_stm_castling(recovered) == _placement_stm_castling(original), (
        f"round-trip mismatch for {fen}\n"
        f"  original : {_placement_stm_castling(original)}\n"
        f"  recovered: {_placement_stm_castling(recovered)}"
    )


def test_have_enough_black_to_move():
    """Guard: the battery must include >= 3 Black-to-move positions."""
    assert len(BLACK_TO_MOVE_FENS) >= 3


@pytest.mark.parametrize("fen", TEST_FENS)
def test_plane30_sign_convention(fen):
    """Plane 30 is signed: White-to-move = -1.0, Black-to-move = +1.0,
    constant across the 8x8, and from the ORIGINAL side to move."""
    planes = fen_to_planes(fen)
    black_to_move = chess.Board(fen).turn == chess.BLACK
    expected = 1.0 if black_to_move else -1.0
    assert np.all(planes[30] == expected)


def test_rule50_roundtrip():
    """rule50 survives the /100 encode + *100 decode."""
    fen = "8/5k2/8/8/3P4/8/5K2/8 b - - 7 40"
    recovered = planes_to_board(fen_to_planes(fen))
    assert recovered.halfmove_clock == 7


def test_history_repeat_vs_zeros():
    """history='repeat' fills the 1-ply-ago slot with the current position;
    history='zeros' leaves it empty."""
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    rep = fen_to_planes(fen, history="repeat")
    zer = fen_to_planes(fen, history="zeros")
    # planes 13..25 mirror 0..12 under "repeat"
    assert np.array_equal(rep[13:26], rep[0:13])
    # ...and are all zero under "zeros"
    assert np.all(zer[13:26] == 0.0)
    # the current-position planes (0..12) are identical either way
    assert np.array_equal(rep[0:13], zer[0:13])


def test_history_prev_board_canonicalized():
    """A real prev_board is written into planes 13..25 in the CURRENT
    side-to-move frame (Black to move -> mirrored)."""
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    prev = chess.Board()  # start position, the real 1-ply-ago board
    planes = board_to_planes(board, prev_board=prev)
    # Black to move => prev is mirrored; "our" history (13..18) are Black's
    # start pieces seen from Black's perspective == White's start pieces seen
    # from White's perspective. So planes[13:19] should equal the canonical
    # current "our" pawns/pieces of a start position.
    start_planes = board_to_planes(chess.Board(), history="zeros")
    assert np.array_equal(planes[13:19], start_planes[0:6])
    assert np.array_equal(planes[19:25], start_planes[6:12])


# ---------------------------------------------------------------------------
# uci <-> index round-trip
# ---------------------------------------------------------------------------


def test_canonical_render_shows_the_transform():
    """The canonical (face-value) renderer must SHOW g's action — sigma_c
    visibly changes the board — whereas the un-canonicalizing oracle cancels
    it (same placement, only the turn flips)."""
    from viz.encoding import planes_to_board_canonical, planes_to_board
    from viz.modelio import planes_to_tensor
    from symmetry.actions import group_action_input
    from symmetry.group import GChess

    fen = "2r5/2k5/3p2Rp/1p2p3/8/6P1/P7/3b2K1 w - - 2 34"
    planes = fen_to_planes(fen)
    gx = group_action_input(GChess.SIGMA_C, planes_to_tensor(planes))[0].numpy()
    # canonical render shows the transform
    assert planes_to_board_canonical(planes).board_fen() != planes_to_board_canonical(gx).board_fen()
    # the oracle (un-canonicalizing) would render the same placement
    assert planes_to_board(planes).board_fen() == planes_to_board(gx).board_fen()


def test_legal_masking_helpers():
    """legal_canonical_indices + transform_indices produce the legal moves of
    g·x (same count, valid squares)."""
    from viz.modelio import (legal_canonical_indices, transform_indices,
                             canonical_index_to_squares)

    board = chess.Board("2r5/2k5/3p2Rp/1p2p3/8/6P1/P7/3b2K1 w - - 2 34")
    L = legal_canonical_indices(board)
    assert len(L) == board.legal_moves.count()  # no promotions here, all map
    for g in ("sigma_h", "sigma_c", "sigma_h_c"):
        Lg = transform_indices(L, g)
        assert len(Lg) == len(L)
        assert all(0 <= i < 1858 for i in Lg)
        for idx in Lg:
            fr, to = canonical_index_to_squares(idx)
            assert 0 <= fr < 64 and 0 <= to < 64 and fr != to


def test_policy_index_size():
    assert len(POLICY_INDEX_TO_UCI) == POLICY_INDEX_SIZE == 1858
    assert len(UCI_TO_POLICY_INDEX) == POLICY_INDEX_SIZE


def test_uci_index_roundtrip_index_first():
    """index -> uci -> index is the identity for all 1858 slots."""
    for i in range(POLICY_INDEX_SIZE):
        uci = POLICY_INDEX_TO_UCI[i]
        assert UCI_TO_POLICY_INDEX[uci] == i


def test_uci_index_roundtrip_uci_first():
    """uci -> index -> uci is the identity for all 1858 moves."""
    for uci, i in UCI_TO_POLICY_INDEX.items():
        assert POLICY_INDEX_TO_UCI[i] == uci


# ---------------------------------------------------------------------------
# Conventions shared with the training data
# ---------------------------------------------------------------------------


def test_castling_uses_king_takes_rook_slot():
    """LCZero encodes castling as the king capturing its own rook (e1h1 /
    e1a1); e1g1 / e1c1 are never legal in the training targets."""
    from viz.modelio import legal_canonical_indices

    for fen in ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
                "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"):
        idx = set(legal_canonical_indices(chess.Board(fen)))
        assert {UCI_TO_POLICY_INDEX["e1h1"], UCI_TO_POLICY_INDEX["e1a1"]} <= idx, fen
        assert UCI_TO_POLICY_INDEX["e1g1"] not in idx, fen
        assert UCI_TO_POLICY_INDEX["e1c1"] not in idx, fen


def test_knight_promotion_uses_plain_move_slot():
    """A knight promotion is LCZero's plain pawn move (e7e8); queen, rook and
    bishop promotions keep their suffix. Every legal move gets an index."""
    from viz.modelio import legal_canonical_indices

    board = chess.Board("8/4P3/8/8/8/8/k7/4K3 w - - 0 1")
    idx = legal_canonical_indices(board)
    for uci in ("e7e8", "e7e8q", "e7e8r", "e7e8b"):
        assert UCI_TO_POLICY_INDEX[uci] in idx
    assert len(set(idx)) == len(idx) == board.legal_moves.count()


def test_prediction_wdl_is_the_value_head_output():
    """The value heads already end in a softmax; predict() must report the
    WDL unchanged (a second softmax would squash it toward uniform)."""
    import torch
    from training.trainer import build_model
    from viz.modelio import planes_to_tensor, predict, raw_forward

    torch.manual_seed(0)
    model = build_model("baseline", blocks=1, filters=8, se_channels=4,
                        mlh_channels=4, fc_size=8).eval()
    board = chess.Board()
    planes = fen_to_planes(board.fen())
    value = raw_forward(model, planes_to_tensor(planes))[1][0].numpy()
    pred = predict(model, planes, board)
    assert np.allclose(pred.wdl, value)
    assert abs(float(pred.wdl.sum()) - 1.0) < 1e-5
    assert abs(pred.q - float(pred.wdl[0] - pred.wdl[2])) < 1e-6


# ---------------------------------------------------------------------------
# The shipped checkpoints
# ---------------------------------------------------------------------------

CKPT_DIR = APP_DIR / "checkpoints"
CLAIMS = {
    "baseline": set(),
    "gcnn": {"sigma_h", "sigma_c", "sigma_h_c"},
    "sgcnn": {"sigma_c"},
    "fgcnn": {"sigma_h"},
}


@pytest.mark.skipif(not any(CKPT_DIR.glob("*.pt")), reason="no app checkpoints")
def test_shipped_checkpoints_match_exactly_their_symmetries():
    """On a real position, each shipped model is MATCH on exactly the group
    elements of its own symmetry group (none for the baseline)."""
    from viz.modelio import (GROUP_ELEMENTS, discover_checkpoints,
                             equivariance_compare, load_model, planes_to_tensor)

    paths = discover_checkpoints(str(CKPT_DIR))
    assert set(CLAIMS) <= set(paths)
    x = planes_to_tensor(fen_to_planes(
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"))
    for name, expected in CLAIMS.items():
        model = load_model(paths[name]).model
        matched = {g for g in GROUP_ELEMENTS if equivariance_compare(model, x, g).match}
        assert matched == expected, (name, matched)
