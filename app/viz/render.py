"""SVG board + small HTML widgets for the Streamlit app.

Kept Streamlit-free so it can be unit-rendered in isolation.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import chess
import chess.svg

# Arrow colors (RGBA hex) for the equivariance overlay.
COLOR_ACTUAL = "#1f77b4cc"    # f(g.x)   -- model's actual output, blue
COLOR_EXPECTED = "#d627284d"  # g.f(x)   -- equivariant expectation, red (~30% opacity)
COLOR_TOP = "#2ca02ccc"       # prediction top move, green
COLOR_OTHER = "#1f77b4aa"     # prediction other moves, blue


def board_svg(
    board: chess.Board,
    arrows: Optional[Sequence[chess.svg.Arrow]] = None,
    orientation: bool = chess.WHITE,
    size: int = 340,
    lastmove: Optional[chess.Move] = None,
) -> str:
    """Render a board to an SVG string. Works on any Board object regardless
    of legality (transformed positions may be technically illegal)."""
    return chess.svg.board(
        board,
        arrows=list(arrows) if arrows else [],
        orientation=orientation,
        size=size,
        lastmove=lastmove,
    )


def arrows_from_squares(
    squares: Sequence[Tuple[int, int]],
    color: str,
) -> List[chess.svg.Arrow]:
    """Build arrows from a list of (from_sq, to_sq) absolute squares."""
    return [chess.svg.Arrow(fr, to, color=color) for fr, to in squares]


def wdl_bar_html(wdl, title: str, height: int = 18) -> str:
    """A stacked horizontal W/D/L bar. ``wdl`` is [W, D, L] in [0, 1]."""
    w, d, l = float(wdl[0]), float(wdl[1]), float(wdl[2])
    total = max(w + d + l, 1e-9)
    wp, dp, lp = 100 * w / total, 100 * d / total, 100 * l / total
    return f"""
<div style="margin:2px 0 8px 0;">
  <div style="font-size:12px;color:#888;margin-bottom:2px;">{title}</div>
  <div style="display:flex;height:{height}px;border-radius:4px;overflow:hidden;
              font-size:10px;line-height:{height}px;color:#fff;text-align:center;">
    <div style="width:{wp:.1f}%;background:#3a9a3a;" title="Win {w:.3f}">{w:.2f}</div>
    <div style="width:{dp:.1f}%;background:#888;" title="Draw {d:.3f}">{d:.2f}</div>
    <div style="width:{lp:.1f}%;background:#b03a3a;" title="Loss {l:.3f}">{l:.2f}</div>
  </div>
</div>
"""


def badge_html(match: bool, residual: float) -> str:
    """A MATCH / MISMATCH pill with the overall residual."""
    if match:
        bg, fg, label = "#1f7a1f", "#fff", "MATCH"
    else:
        bg, fg, label = "#a11", "#fff", "MISMATCH"
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 10px;border-radius:10px;'
        f'font-weight:600;font-size:13px;">{label}</span>'
        f'<span style="color:#888;font-size:12px;margin-left:8px;">'
        f'residual = {residual:.2e}</span>'
    )


__all__ = [
    "COLOR_ACTUAL", "COLOR_EXPECTED", "COLOR_TOP", "COLOR_OTHER",
    "board_svg", "arrows_from_squares", "wdl_bar_html", "badge_html",
]
