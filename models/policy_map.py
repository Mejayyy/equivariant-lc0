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


# ---------------------------------------------------------------------------
# Orbit-based gather (used by EquivariantPolicyHead — Phase 8)
# ---------------------------------------------------------------------------
#
# Why a different gather for the equivariant head:
#   The lc0 AZ-style 80-channel layout (above) encodes move *direction* in
#   the channel axis. After the equivariant-policy GConv stack +
#   sum-over-group, the resulting (80, 8, 8) tensor is sigma_h/sigma_c-
#   equivariant on its SPATIAL axis only — channels stay put. Hence
#   "spatial-flip + canonical lc0 gather" does NOT reproduce
#   build_policy_permutation for moves whose direction is itself
#   permuted by the flip (e.g. NE <-> NW under sigma_h, NE <-> SE under
#   sigma_c). For diagonal moves the natural permutation differs from the
#   canonical one — equivariance fails by O(1).
#
# Orbit-based gather:
#   Group the 1858 moves into Klein-4 orbits (size 4 for non-promotions,
#   size 2 for promotions since sigma_c is identity on promotions per
#   Phase 3 design). Each orbit is assigned to a fixed 2x2 block of cells
#   in some output channel:
#       cell_E  = (c, r0,    f0   )  -- orbit representative
#       cell_H  = (c, r0,    7-f0 )  -- file-flipped (sigma_h)
#       cell_C  = (c, 7-r0,  f0   )  -- rank-flipped (sigma_c)
#       cell_HC = (c, 7-r0,  7-f0 )  -- both flipped (sigma_hc)
#   with r0 in {0,1,2,3}, f0 in {0,1,2,3}, giving 16 disjoint blocks per
#   channel. Orbits are placed sequentially: orbit k -> channel k//16,
#   block (k%16). 481 orbits fit in 31 channels, well within 80.
#
#   For NON-PROMO orbit {m_E, m_H, m_C, m_HC}: each move gets ONE cell;
#   gather[move_idx, 0] == gather[move_idx, 1] (the duplicate makes the
#   downstream `.mean(-1)` trivial — no rank-symmetrization needed).
#
#   For PROMO orbit {m_E, m_H} (where sigma_c(m) = m, so the orbit only
#   has 2 distinct moves but still occupies 4 cells): each move is
#   placed at a rank-pair, gather[m_E] = (cell_E, cell_C) and
#   gather[m_H] = (cell_H, cell_HC). The downstream `.mean(-1)` averages
#   the rank-pair, making the head sigma_c-invariant on promo channels
#   exactly as required (sigma_c rank-flip swaps the pair, the mean
#   is unchanged).
#
# Equivariance proof (sketch). Let z = sum_pool of the equivariant body's
# 80*|G|-channel output. z_after_g = (spatial_g)(z). For move i in a
# non-promo orbit at cell (c, r0, f0):
#   gather(z_after_h)[i] = z_after_h[(c, r0, f0)] = z[(c, r0, 7-f0)]
#                        = z[gather[sigma_h(i)]] = gather(z)[sigma_h(i)]
# which matches build_policy_permutation(sigma_h)(gather(z))[i].
# Same argument for sigma_c (rank-flip) and sigma_hc (both flips).
# For promo: gather(z)[i] = (z[cell_E] + z[cell_C]) / 2; sigma_c rank-
# flip swaps the two cells, average is invariant -> matches "sigma_c is
# identity on promotions". sigma_h spatially file-flips both cells of i
# into both cells of sigma_h(i); the (averaged) value moves correctly.


def _verify_orbit_invariants(perm_h, perm_c, perm_hc):
    """Sanity-check that for every move i, the orbit {i, h, c, hc} has
    size 4 (non-promo) or 2 (promo, where c==i and hc==h)."""
    for i in range(POLICY_INDEX_SIZE):
        h, c, hc = perm_h[i], perm_c[i], perm_hc[i]
        if c == i:
            # promo orbit
            if hc != h:
                raise ValueError(
                    f"promo orbit invariant violated at move {i}: "
                    f"sigma_c(i)=i but sigma_hc(i)={hc} != sigma_h(i)={h}"
                )
            # The other element must be distinct.
            if h == i:
                raise ValueError(
                    f"degenerate orbit at move {i}: sigma_h(i)=i "
                    f"(no chess move is sigma_h-fixed)"
                )
        else:
            if len({i, h, c, hc}) != 4:
                raise ValueError(
                    f"non-promo orbit at {i} has duplicates: "
                    f"{{i={i}, h={h}, c={c}, hc={hc}}}"
                )


def build_orbit_policy_gather(num_channels: int = POLICY_NUM_CHANNELS) -> "torch.Tensor":
    """Build the orbit-based ``(1858, 2)`` gather buffer for the
    equivariant policy head. See module-level comment for the rationale.

    The returned tensor is suitable for use as a non-trainable buffer; the
    head extracts logits via::

        flat   = sum_pool_tensor.view(B, num_channels * 8 * 8)        # (B, C*64)
        gathered = flat.index_select(1, gather.view(-1))             # (B, 1858*2)
        logits = gathered.view(B, 1858, 2).mean(dim=2)               # (B, 1858)

    Args:
        num_channels: the channel count of the policy-head's last conv.
            Must be at least ceil(481 / 16) = 31. Default 80 matches the
            baseline policy head's channel count for parameter parity.

    Returns:
        torch.LongTensor of shape (1858, 2) with values in
        [0, num_channels * 64).
    """
    # Lazy import to break the symmetry/actions <-> models/policy_map cycle.
    from symmetry.actions import build_policy_permutation
    from symmetry.group import GChess

    perm_h = build_policy_permutation(GChess.SIGMA_H).tolist()
    perm_c = build_policy_permutation(GChess.SIGMA_C).tolist()
    perm_hc = build_policy_permutation(GChess.SIGMA_HC).tolist()
    _verify_orbit_invariants(perm_h, perm_c, perm_hc)

    blocks_per_channel = 16  # 4 r0 choices * 4 f0 choices, all disjoint 2x2 blocks
    visited: List[bool] = [False] * POLICY_INDEX_SIZE
    gather = torch.full((POLICY_INDEX_SIZE, 2), -1, dtype=torch.int64)

    orbit_count = 0
    for i in range(POLICY_INDEX_SIZE):
        if visited[i]:
            continue
        h = perm_h[i]
        c = perm_c[i]
        hc = perm_hc[i]

        chan = orbit_count // blocks_per_channel
        if chan >= num_channels:
            raise ValueError(
                f"orbit-based gather needs {orbit_count + 1} orbits but only "
                f"{num_channels * blocks_per_channel} slots are available "
                f"({num_channels} channels x {blocks_per_channel} blocks/channel). "
                f"Increase num_channels."
            )
        block_idx = orbit_count % blocks_per_channel
        r0 = block_idx // 4
        f0 = block_idx % 4

        cell_E = chan * 64 + r0 * 8 + f0
        cell_H = chan * 64 + r0 * 8 + (7 - f0)
        cell_C = chan * 64 + (7 - r0) * 8 + f0
        cell_HC = chan * 64 + (7 - r0) * 8 + (7 - f0)

        is_promo = (c == i)
        if is_promo:
            # 2 distinct moves; each occupies a rank-pair so that the
            # downstream mean enforces sigma_c-rank-invariance.
            gather[i, 0] = cell_E
            gather[i, 1] = cell_C
            gather[h, 0] = cell_H
            gather[h, 1] = cell_HC
            visited[i] = True
            visited[h] = True
        else:
            gather[i, 0] = cell_E
            gather[i, 1] = cell_E
            gather[h, 0] = cell_H
            gather[h, 1] = cell_H
            gather[c, 0] = cell_C
            gather[c, 1] = cell_C
            gather[hc, 0] = cell_HC
            gather[hc, 1] = cell_HC
            visited[i] = True
            visited[h] = True
            visited[c] = True
            visited[hc] = True

        orbit_count += 1

    if not all(visited):
        missing = [i for i, v in enumerate(visited) if not v]
        raise RuntimeError(f"unvisited moves after orbit walk: {missing[:10]} ...")
    if (gather < 0).any():
        raise RuntimeError("uninitialized gather entries")
    if (gather >= num_channels * 64).any():
        raise RuntimeError(
            f"gather entry out of range; max={int(gather.max())} >= "
            f"{num_channels * 64}"
        )
    return gather


# ---------------------------------------------------------------------------
# SG_chess orbit-based gather (used by EquivariantPolicyHead when the model
# is built with SGCHESS_GROUP — Z/2 generated by sigma_c only).
# ---------------------------------------------------------------------------
#
# SG_chess has just two elements: e and sigma_c. The body's
# sum-over-group of an SGChess-equivariant policy head's output is a
# (B, num_channels, 8, 8) tensor whose only non-trivial group action is
# rank flip (sigma_c). Orbits of the 1858 moves under sigma_c have size
# 2 (non-promo) or 1 (promo, since sigma_c is identity on promotions).
#
# Each orbit gets a rank-pair of cells at (chan, r0, f0) and
# (chan, 7-r0, f0) in some channel, with r0 in {0,1,2,3} and f0 in
# {0..7} -> 4*8 = 32 distinct rank-pair slots per channel. For
# non-promo orbit {m_E, m_C}:
#     gather[m_E] = (cell_E, cell_E)   # single-cell, mean is trivial
#     gather[m_C] = (cell_C, cell_C)
# For promo orbit {m} (size 1): the move is averaged over the rank pair
#     gather[m] = (cell_E, cell_C)
# so the downstream `.mean(-1)` enforces sigma_c-rank-invariance for
# promotions, matching "sigma_c is identity on promotions".
#
# Equivariance under sigma_c spatial rank flip on the (B, C, 8, 8) sum-
# pool tensor: cell_E and cell_C swap. For non-promo, that swaps the
# values picked up for m_E and m_C — exactly the policy permutation.
# For promo, the mean over the rank pair is invariant. Sigma_h is NOT
# in SG_chess, so no constraint from it.


def build_sgchess_orbit_policy_gather(
    num_channels: int = POLICY_NUM_CHANNELS,
) -> torch.Tensor:
    """Build the orbit-based ``(1858, 2)`` gather for the SGChess-
    equivariant policy head.

    Returns a ``torch.LongTensor`` shape ``(POLICY_INDEX_SIZE, 2)``
    with values in ``[0, num_channels * 64)``. Suitable for use as a
    non-trainable buffer; the head extracts logits via::

        flat   = sum_pool_tensor.view(B, num_channels * 8 * 8)
        gathered = flat.index_select(1, gather.view(-1))
        logits = gathered.view(B, 1858, 2).mean(dim=2)

    With 1792 non-promo moves (-> 896 orbits) and 66 promo orbits, the
    total orbit count is 962. With 32 rank-pair slots per channel, we
    need at most ceil(962 / 32) = 31 channels — well within the
    default ``POLICY_NUM_CHANNELS = 80``.
    """
    from symmetry.actions import build_policy_permutation
    from symmetry.group import GChess

    perm_c = build_policy_permutation(GChess.SIGMA_C).tolist()

    slots_per_channel = 32  # 4 rank-pairs * 8 files
    visited: List[bool] = [False] * POLICY_INDEX_SIZE
    gather = torch.full((POLICY_INDEX_SIZE, 2), -1, dtype=torch.int64)

    orbit_count = 0
    for i in range(POLICY_INDEX_SIZE):
        if visited[i]:
            continue
        c_i = perm_c[i]

        chan = orbit_count // slots_per_channel
        if chan >= num_channels:
            raise ValueError(
                f"SGChess orbit gather needs {orbit_count + 1} orbits but only "
                f"{num_channels * slots_per_channel} slots are available "
                f"({num_channels} channels x {slots_per_channel} slots/channel). "
                f"Increase num_channels."
            )
        slot = orbit_count % slots_per_channel
        r0 = slot // 8
        f0 = slot % 8

        cell_E = chan * 64 + r0 * 8 + f0
        cell_C = chan * 64 + (7 - r0) * 8 + f0

        is_promo = (c_i == i)
        if is_promo:
            # 1 distinct move at a rank-pair, averaged downstream -> sigma_c invariant.
            gather[i, 0] = cell_E
            gather[i, 1] = cell_C
            visited[i] = True
        else:
            # 2 distinct moves, each occupying a single cell (duplicated).
            gather[i, 0] = cell_E
            gather[i, 1] = cell_E
            gather[c_i, 0] = cell_C
            gather[c_i, 1] = cell_C
            visited[i] = True
            visited[c_i] = True

        orbit_count += 1

    if not all(visited):
        missing = [i for i, v in enumerate(visited) if not v]
        raise RuntimeError(f"unvisited moves after orbit walk: {missing[:10]} ...")
    if (gather < 0).any():
        raise RuntimeError("uninitialized gather entries")
    if (gather >= num_channels * 64).any():
        raise RuntimeError(
            f"gather entry out of range; max={int(gather.max())} >= "
            f"{num_channels * 64}"
        )
    return gather


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
    "build_orbit_policy_gather",
    "build_sgchess_orbit_policy_gather",
]
