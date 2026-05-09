"""G_chess = Z/2Z x Z/2Z = {e, sigma_h, sigma_c, sigma_h o sigma_c}.

Fixed enumeration (used by every other module):
    0 = e
    1 = sigma_h    (file flip; A <-> H)
    2 = sigma_c    (rank flip + channel swap of our/opp + side-to-move flip)
    3 = sigma_hc   (sigma_h composed with sigma_c)

Klein four-group: every element is its own inverse.
"""

from enum import IntEnum


class GChess(IntEnum):
    E = 0
    SIGMA_H = 1
    SIGMA_C = 2
    SIGMA_HC = 3


ELEMENTS = (GChess.E, GChess.SIGMA_H, GChess.SIGMA_C, GChess.SIGMA_HC)
ORDER = 4

# COMPOSITION_TABLE[i][j] = i o j (apply j first, then i).
COMPOSITION_TABLE = (
    (0, 1, 2, 3),
    (1, 0, 3, 2),
    (2, 3, 0, 1),
    (3, 2, 1, 0),
)


def compose(g1: int, g2: int) -> int:
    return COMPOSITION_TABLE[int(g1)][int(g2)]


def inverse(g: int) -> int:
    return int(g)
