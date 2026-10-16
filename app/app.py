"""Streamlit app: SEE equivariance on a chess board.

Two modes:
  * Equivariance (headline) -- for a position x and a group element g, show
    f(g.x) next to g.f(x) for every model, with MATCH/MISMATCH badges,
    per-head residuals, side-by-side boards and overlaid policy arrows.
  * Prediction -- per-model top-5 legal moves, WDL + eval, moves-left, arrows.

Run from the repository root:  streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

# The model code (models/, symmetry/, metrics/, training/) lives at the
# repository root; the app's helpers (viz/) live next to this file.
_APP_DIR = Path(__file__).resolve().parent
for _path in (str(_APP_DIR), str(_APP_DIR.parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import streamlit as st

import chess

from viz.encoding import fen_to_planes, planes_to_board, planes_to_board_canonical
from viz.modelio import (
    GROUP_ELEMENTS, GROUP_LABELS, MATCH_THRESHOLD,
    discover_checkpoints, load_model, planes_to_tensor,
    predict, equivariance_compare, raw_forward, wdl_probs, top_indices,
    policy_index_to_abs_squares,
    legal_canonical_indices, transform_indices, topk_over_indices,
    canonical_index_to_squares,
)
from viz.render import (
    COLOR_ACTUAL, COLOR_EXPECTED, COLOR_TOP, COLOR_OTHER,
    board_svg, arrows_from_squares, wdl_bar_html, badge_html,
)
import metrics.equivariance as E
from viz.modelio import MODEL_CLAIMS

# Preferred display order; the actual MODEL_ORDER is the subset discovered.
PREFERRED_ORDER = ["baseline", "gcnn", "sgcnn", "fgcnn"]
MODEL_ORDER = list(PREFERRED_ORDER)  # rebound after discovery
START_FEN = chess.STARTING_FEN
# Shown in the sidebar; relative paths are resolved from the repository root.
DEFAULT_CKPT_DIR = "app/checkpoints"

st.set_page_config(page_title="Chess equivariance viz", layout="wide")


# ---------------------------------------------------------------------------
# Model loading (once)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading checkpoints...")
def get_models(ckpt_dir: str):
    paths = discover_checkpoints(ckpt_dir)
    loaded = {}
    for name in PREFERRED_ORDER:
        if name in paths:
            loaded[name] = load_model(paths[name])
    return loaded, paths


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_state():
    if "board" not in st.session_state:
        # Allow an initial position via ?fen=... (shareable links).
        fen_qp = st.query_params.get("fen")
        try:
            board = chess.Board(fen_qp) if fen_qp else chess.Board()
        except ValueError:
            board = chess.Board()
        st.session_state.board = board
        st.session_state.prev_boards: List[chess.Board] = []
        st.session_state.last_move: Optional[chess.Move] = None
    if "fen_box" not in st.session_state:
        st.session_state.fen_box = st.session_state.board.fen()


def _set_board(board: chess.Board, last_move: Optional[chess.Move] = None):
    """Install a new current board and sync dependent widget state. Safe to
    call only from a widget callback (runs before widgets are instantiated)."""
    st.session_state.board = board
    st.session_state.prev_boards = []
    st.session_state.last_move = last_move
    st.session_state.fen_box = board.fen()
    st.session_state.move_sel = 0


# --- widget callbacks (run before the rerun, so they may set widget keys) ---
def cb_load():
    try:
        board = chess.Board(st.session_state.fen_box)
    except ValueError as e:
        st.session_state.load_err = f"Invalid FEN: {e}"
        return
    st.session_state.load_err = None
    _set_board(board)


def cb_reset():
    st.session_state.load_err = None
    _set_board(chess.Board())


def cb_play():
    legal = list(st.session_state.board.legal_moves)
    i = st.session_state.get("move_sel", 0)
    if 0 <= i < len(legal):
        st.session_state.prev_boards.append(st.session_state.board.copy())
        st.session_state.board.push(legal[i])
        st.session_state.last_move = legal[i]
        st.session_state.fen_box = st.session_state.board.fen()
        st.session_state.move_sel = 0


def cb_undo():
    if st.session_state.prev_boards:
        board = st.session_state.prev_boards.pop()
        st.session_state.board = board
        st.session_state.last_move = None
        st.session_state.fen_box = board.fen()
        st.session_state.move_sel = 0


def cb_play_typed():
    board = st.session_state.board
    txt = st.session_state.get("typed_move", "").strip()
    mv = None
    try:
        mv = board.parse_san(txt)
    except Exception:
        try:
            mv = board.parse_uci(txt)
        except Exception:
            mv = None
    if mv is not None and mv in board.legal_moves:
        st.session_state.prev_boards.append(board.copy())
        board.push(mv)
        st.session_state.last_move = mv
        st.session_state.fen_box = board.fen()
        st.session_state.move_sel = 0
        st.session_state.typed_err = None
        st.session_state.typed_move = ""
    else:
        st.session_state.typed_err = f"Not a legal move: {txt!r}"


# ---------------------------------------------------------------------------
# Small UI helpers
# ---------------------------------------------------------------------------
def show_svg(svg: str, height: int):
    html = f'<div style="display:flex;justify-content:center;">{svg}</div>'
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:  # Streamlit versions before st.iframe
        import streamlit.components.v1 as components
        components.html(html, height=height)


def current_planes(board: chess.Board) -> np.ndarray:
    prev = st.session_state.prev_boards[-1] if st.session_state.prev_boards else None
    return fen_to_planes(
        board.fen(),
        history=st.session_state.get("history_fill", "repeat"),
        prev_board=prev,
    )


def pov_adjust(wdl: np.ndarray, q: float, board: chess.Board, white_pov: bool):
    """Return (wdl, q, label) adjusted for the chosen point of view."""
    if white_pov and board.turn == chess.BLACK:
        return wdl[::-1].copy(), -q, "White POV"
    return wdl, q, ("White POV" if white_pov else "side-to-move POV")


def top5_table_md(rows: List[Tuple[str, float]]) -> str:
    """rows = [(label, prob), ...] -> a compact markdown table."""
    out = ["| # | move | prob |", "|--:|:--|--:|"]
    for i, (label, prob) in enumerate(rows, 1):
        out.append(f"| {i} | `{label}` | {prob:.3f} |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
_init_state()

st.sidebar.title("♟ Equivariance viz")

ckpt_dir = st.sidebar.text_input(
    "Checkpoints dir", value=DEFAULT_CKPT_DIR,
    help="Relative to the repository root, e.g. runs/<run> for your own training runs.",
)
models, paths = get_models(str((_APP_DIR.parent / ckpt_dir).resolve()))

# The models actually present, in preferred order.
MODEL_ORDER = [n for n in PREFERRED_ORDER if n in models]

if not MODEL_ORDER:
    st.sidebar.error(f"No models found. Looked in: {ckpt_dir}")
    st.stop()

missing = [n for n in PREFERRED_ORDER if n not in models]
if missing:
    st.sidebar.warning("Not found (skipped): " + ", ".join(missing))

with st.sidebar.expander("Loaded models", expanded=True):
    for name in MODEL_ORDER:
        lm = models[name]
        st.markdown(
            f"**{name}** — {lm.arch_label}  \n"
            f"params {lm.n_params:,} · step {lm.step}  \n"
            f"<span style='font-size:11px;color:#888;'>equivariant under: "
            f"{MODEL_CLAIMS.get(name, '?')}</span>",
            unsafe_allow_html=True,
        )

mode = st.sidebar.radio("Mode", ["Equivariance (headline)", "Prediction"])

st.sidebar.markdown("---")
st.sidebar.subheader("Position")
st.sidebar.text_input("FEN to load", key="fen_box")
c1, c2 = st.sidebar.columns(2)
c1.button("Load FEN", on_click=cb_load, use_container_width=True)
c2.button("Reset start", on_click=cb_reset, use_container_width=True)
if st.session_state.get("load_err"):
    st.sidebar.error(st.session_state.load_err)

board = st.session_state.board

# Move entry: legal-move dropdown (always works) + free-text UCI/SAN.
legal = list(board.legal_moves)
if legal:
    labels = [f"{board.san(m)}  ({m.uci()})" for m in legal]
    if st.session_state.get("move_sel", 0) >= len(legal):
        st.session_state.move_sel = 0
    st.sidebar.selectbox("Legal move", options=list(range(len(legal))),
                         format_func=lambda i: labels[i], key="move_sel")
    mc1, mc2 = st.sidebar.columns(2)
    mc1.button("Play move", on_click=cb_play, use_container_width=True)
    mc2.button("Undo", on_click=cb_undo, use_container_width=True)
else:
    st.sidebar.info("No legal moves (game over / illegal position).")
    st.sidebar.button("Undo", on_click=cb_undo)

st.sidebar.text_input("...or type a move (SAN or UCI)", key="typed_move")
st.sidebar.button("Play typed move", on_click=cb_play_typed)
if st.session_state.get("typed_err"):
    st.sidebar.error(st.session_state.typed_err)

st.sidebar.markdown("---")
st.sidebar.subheader("Options")
st.session_state.history_fill = st.sidebar.radio(
    "History fill (root FEN has no real history)",
    options=["repeat", "zeros"],
    format_func=lambda s: {"repeat": "repeat current position", "zeros": "zeros"}[s],
    help="The net expects 2-ply history; at a root FEN predictions are approximate.",
)
flip = st.sidebar.checkbox("Flip board view", value=False)
orientation = chess.BLACK if flip else chess.WHITE
white_pov = st.sidebar.checkbox("White POV for eval", value=False)

g_key = None
if mode.startswith("Equivariance"):
    g_keys = list(GROUP_ELEMENTS.keys())
    if "g_sel" not in st.session_state:
        qpg = st.query_params.get("g", "sigma_h")
        st.session_state.g_sel = qpg if qpg in g_keys else "sigma_h"
    g_key = st.sidebar.selectbox(
        "Group element g", options=g_keys,
        format_func=lambda k: GROUP_LABELS[k], key="g_sel",
    )

planes = current_planes(board)
x = planes_to_tensor(planes)


# ---------------------------------------------------------------------------
# Equivariance mode
# ---------------------------------------------------------------------------
def render_equivariance():
    st.title("Equivariance:  f(g·x)  vs  g·f(x)")
    st.caption(
        "If a model is equivariant under g, its output on the transformed board "
        "**f(g·x)** equals the transformed output **g·f(x)** — the two arrow sets "
        "(blue = actual, red = expected) overlap and the badge reads MATCH."
    )
    turn = "White" if board.turn == chess.WHITE else "Black"
    st.caption(f"Source position: `{board.fen()}` — {turn} to move · "
               f"history fill = {st.session_state.history_fill}")

    g = GROUP_ELEMENTS[g_key]
    from symmetry.actions import group_action_input
    gx = group_action_input(g, x)
    gx_planes = gx[0].numpy()
    # Render at FACE VALUE in the canonical frame so g's action is visible
    # (planes_to_board would un-canonicalize and cancel sigma_c's flip).
    board_x = planes_to_board_canonical(planes)
    board_gx = planes_to_board_canonical(gx_planes)

    # Legal moves of x (canonical indices), and their images under g = the
    # legal moves of g·x. Restricting arrows to these keeps them legal and
    # makes them transform with g (instead of the raw 1858-wide argmax).
    legal_x = legal_canonical_indices(board)
    legal_gx = transform_indices(legal_x, g_key)

    # The boards showing the transform itself.
    st.subheader(f"The transform: g = {GROUP_LABELS[g_key]}")
    st.caption(
        "**Your position** is in real colors/orientation. **x** and **g·x** are the "
        "**canonical frame** — the net's input space, where the side-to-move's pieces are "
        "drawn as White at the bottom (so a Black-to-move position looks recolored). That "
        "is the only frame where g is visible: σ_h mirrors files, σ_c swaps colors + flips "
        "ranks, σ_h∘σ_c rotates 180°."
    )
    tb0, tb1, tb2 = st.columns(3)
    with tb0:
        show_svg(board_svg(board, orientation=orientation, size=260,
                           lastmove=st.session_state.last_move), height=285)
        turn = "White" if board.turn == chess.WHITE else "Black"
        st.caption(f"your position — {turn} to move (absolute)")
    with tb1:
        show_svg(board_svg(board_x, orientation=orientation, size=260), height=285)
        st.caption("x  (canonical input the net sees)")
    with tb2:
        show_svg(board_svg(board_gx, orientation=orientation, size=260), height=285)
        st.caption("g·x  (canonical, transformed)")

    st.markdown("---")
    st.subheader("Per-model: f(g·x) vs g·f(x) on the transformed board g·x")
    st.caption("Arrows are the top-5 over the **legal moves of g·x**. "
               "🔵 actual f(g·x) · 🔴 expected g·f(x). Equivariant ⇒ they coincide.")

    board_size = 300 if len(MODEL_ORDER) <= 3 else 240
    cols = st.columns(len(MODEL_ORDER))
    for col, name in zip(cols, MODEL_ORDER):
        lm = models[name]
        res = equivariance_compare(lm.model, x, g_key)
        with col:
            st.markdown(f"### {name}")
            st.caption(f"claims: {MODEL_CLAIMS.get(name, '?')}")
            st.markdown(badge_html(res.match, res.residual_overall), unsafe_allow_html=True)

            # Arrows: top-5 legal moves of g·x for actual f(g·x) and expected g·f(x).
            actual_top = topk_over_indices(res.f_gx[0][0], legal_gx, 5)
            expected_top = topk_over_indices(res.g_f_x[0][0], legal_gx, 5)
            actual_sq = [canonical_index_to_squares(i) for i, _ in actual_top]
            expected_sq = [canonical_index_to_squares(i) for i, _ in expected_top]
            arrows = (arrows_from_squares(expected_sq, COLOR_EXPECTED) +
                      arrows_from_squares(actual_sq, COLOR_ACTUAL))
            show_svg(board_svg(board_gx, arrows=arrows, orientation=orientation,
                               size=board_size), height=board_size + 20)
            st.caption("🔵 actual f(g·x)  ·  🔴 expected g·f(x)")

            # Per-head residuals (over the FULL policy — the exact metric).
            st.markdown(
                f"<span style='font-size:12px;color:#888;'>"
                f"residuals — policy {res.residual_policy:.2e} · "
                f"value {res.residual_value:.2e} · "
                f"mlh {res.residual_mlh:.2e}</span>",
                unsafe_allow_html=True,
            )

            # WDL bars: actual vs expected.
            wdl_actual = wdl_probs(res.f_gx[1][0])
            wdl_expected = wdl_probs(res.g_f_x[1][0])
            st.markdown(wdl_bar_html(wdl_actual, "WDL — actual f(g·x)"),
                        unsafe_allow_html=True)
            st.markdown(wdl_bar_html(wdl_expected, "WDL — expected g·f(x)"),
                        unsafe_allow_html=True)

            # Moves-left: actual vs expected.
            mlh_actual = float(res.f_gx[2][0].item())
            mlh_expected = float(res.g_f_x[2][0].item())
            st.markdown(
                f"<span style='font-size:12px;color:#888;'>moves-left — "
                f"actual {mlh_actual:.2f} · expected {mlh_expected:.2f} plies</span>",
                unsafe_allow_html=True,
            )

            # Top-5 tables side by side (canonical-frame UCI).
            ta, te = st.columns(2)
            with ta:
                rows = [(_canon_uci(i), p) for i, p in actual_top]
                st.markdown("**actual f(g·x)**\n\n" + top5_table_md(rows))
            with te:
                rows = [(_canon_uci(i), p) for i, p in expected_top]
                st.markdown("**expected g·f(x)**\n\n" + top5_table_md(rows))

    _quantitative_panel()


def _canon_uci(idx: int) -> str:
    fr, to = canonical_index_to_squares(idx)
    return chess.square_name(fr) + chess.square_name(to)


def _quantitative_panel():
    st.markdown("---")
    with st.expander("Quantitative equivariance over N random samples "
                     "(metrics/equivariance.py)"):
        n = st.slider("N random samples", 16, 256, 64, step=16)
        if st.button("Compute"):
            # Run the FULL GChess metric on every model so the complete
            # σ_h / σ_c / σ_h∘σ_c pattern is visible (each model only matches
            # the elements of its own group).
            data = []
            for name in MODEL_ORDER:
                r = E.equivariance_error(models[name].model, n_samples=n)
                data.append({
                    "model": name,
                    "claims": MODEL_CLAIMS.get(name, "?"),
                    "σ_h": r["sigma_h"],
                    "σ_c": r["sigma_c"],
                    "σ_h∘σ_c": r["sigma_h_c"],
                })
            st.dataframe(data, use_container_width=True)
            st.caption(
                "Full GChess metric (relative-Frobenius error). A model reads ~1e-6–1e-7 "
                "on the g's it's equivariant under and O(0.1–1) on the rest: gcnn matches "
                "all, sgcnn only σ_c, **fgcnn only σ_h**, baseline none."
            )


# ---------------------------------------------------------------------------
# Prediction mode
# ---------------------------------------------------------------------------
def render_prediction():
    st.title("Predictions")
    turn = "White" if board.turn == chess.WHITE else "Black"
    st.caption(f"Position: `{board.fen()}` — {turn} to move")

    default_sel = MODEL_ORDER.index("gcnn") if "gcnn" in MODEL_ORDER else 0
    sel_model = st.radio("Show policy arrows for", MODEL_ORDER, index=default_sel,
                         horizontal=True)

    preds = {name: predict(models[name].model, planes, board) for name in MODEL_ORDER}

    # Board with arrows for the selected model.
    p = preds[sel_model]
    arrows = []
    for rank, mv in enumerate(p.top_moves):
        color = COLOR_TOP if rank == 0 else COLOR_OTHER
        arrows.append(chess.svg.Arrow(mv["from_abs"], mv["to_abs"], color=color))
    bcol, _ = st.columns([1, 1])
    with bcol:
        show_svg(board_svg(board, arrows=arrows, orientation=orientation,
                           size=360, lastmove=st.session_state.last_move), height=380)
        st.caption(f"🟢 top move · 🔵 next moves — model: **{sel_model}**")

    cols = st.columns(len(MODEL_ORDER))
    for col, name in zip(cols, MODEL_ORDER):
        pr = preds[name]
        wdl, q, povlabel = pov_adjust(pr.wdl, pr.q, board, white_pov)
        with col:
            st.markdown(f"### {name}")
            st.caption(f"equivariant: {MODEL_CLAIMS.get(name, '?')}")
            rows = [(mv["san"], mv["prob"]) for mv in pr.top_moves]
            st.markdown(top5_table_md(rows))
            st.markdown(wdl_bar_html(wdl, f"WDL ({povlabel})"), unsafe_allow_html=True)
            st.metric("Eval  Q = P(win) − P(loss)", f"{q:+.3f}")
            st.markdown(
                f"<span style='font-size:12px;color:#888;'>moves-left "
                f"{pr.moves_left:.1f} plies (~{pr.moves_left/2:.1f} moves)</span>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
import chess.svg  # noqa: E402  (needed for Arrow in prediction mode)

if mode.startswith("Equivariance"):
    render_equivariance()
else:
    render_prediction()
