"""Phase 2: G_chess group + actions on input planes and body feature maps.

Definition-of-done: invertibility g(g(x)) == x exactly for all g; composition
table matches CLAUDE.md.
"""

import pytest
import torch

from symmetry.group import (
    COMPOSITION_TABLE,
    ELEMENTS,
    ORDER,
    GChess,
    compose,
    inverse,
)
from symmetry.actions import (
    SIGMA_C_CHANNEL_PERM,
    build_policy_permutation,
    group_action_input,
    group_action_mlh,
    group_action_output,
    group_action_policy,
    group_action_value,
)
from models.policy_map import (
    MOVE_TO_POLICY_INDEX,
    POLICY_INDEX_SIZE,
    POLICY_INDEX_TO_MOVE,
    POLICY_INDEX_TO_UCI,
    UCI_TO_POLICY_INDEX,
)


def make_random_input(batch=2, seed=0):
    """Sample inputs matching CLAUDE.md sampling convention.

    Plane 30 (side-to-move) is signed in {-1, +1} (white=-1, black=+1).
    sigma_c on plane 30 is now a pure sign flip ``-x[30]`` (linear),
    and ``-(-x) == x`` is bit-exact under IEEE-754 — so the
    invertibility tests still pass exactly.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(batch, 34, 8, 8)
    x[:, 0:12] = (torch.rand(batch, 12, 8, 8, generator=g) > 0.85).float()
    x[:, 12] = (torch.rand(batch, 8, 8, generator=g) > 0.9).float()
    x[:, 13:25] = (torch.rand(batch, 12, 8, 8, generator=g) > 0.85).float()
    x[:, 25] = (torch.rand(batch, 8, 8, generator=g) > 0.9).float()
    x[:, 26:30] = (torch.rand(batch, 4, 8, 8, generator=g) > 0.5).float()
    # Side-to-move: -1 (white) or +1 (black).
    side = 2.0 * (torch.rand(batch, generator=g) > 0.5).float() - 1.0
    x[:, 30] = side.view(batch, 1, 1).expand(batch, 8, 8)
    rule50 = torch.rand(batch, generator=g)
    x[:, 31] = rule50.view(batch, 1, 1).expand(batch, 8, 8)
    x[:, 32] = 0.0
    x[:, 33] = 1.0
    return x


# ---------------------------------------------------------------------------
# Group structure
# ---------------------------------------------------------------------------


def test_composition_table_matches_claude_md():
    expected = (
        (0, 1, 2, 3),
        (1, 0, 3, 2),
        (2, 3, 0, 1),
        (3, 2, 1, 0),
    )
    assert COMPOSITION_TABLE == expected
    for i in range(ORDER):
        for j in range(ORDER):
            assert compose(i, j) == expected[i][j]


def test_composition_is_associative():
    for a in ELEMENTS:
        for b in ELEMENTS:
            for c in ELEMENTS:
                lhs = compose(compose(a, b), c)
                rhs = compose(a, compose(b, c))
                assert lhs == rhs


def test_composition_is_abelian():
    for a in ELEMENTS:
        for b in ELEMENTS:
            assert compose(a, b) == compose(b, a)


def test_identity_element():
    e = int(GChess.E)
    for g in ELEMENTS:
        assert compose(e, int(g)) == int(g)
        assert compose(int(g), e) == int(g)


def test_every_element_self_inverse():
    e = int(GChess.E)
    for g in ELEMENTS:
        assert inverse(g) == int(g)
        assert compose(int(g), inverse(g)) == e


# ---------------------------------------------------------------------------
# Input action: invertibility (bit-exact)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("g", ELEMENTS)
def test_input_action_self_inverse(g):
    x = make_random_input()
    y = group_action_input(g, x)
    z = group_action_input(g, y)
    assert torch.equal(z, x), f"g(g(x)) != x exactly for {g.name}"


@pytest.mark.parametrize("g", ELEMENTS)
def test_input_action_preserves_shape_and_dtype(g):
    x = make_random_input()
    y = group_action_input(g, x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_input_action_is_homomorphism():
    """group_action_input(g1)(group_action_input(g2)(x)) == group_action_input(g1 o g2)(x)."""
    x = make_random_input()
    for g1 in ELEMENTS:
        for g2 in ELEMENTS:
            sequential = group_action_input(g1, group_action_input(g2, x))
            composed = group_action_input(compose(int(g1), int(g2)), x)
            assert torch.equal(sequential, composed), (
                f"{g1.name}({g2.name}(x)) != ({g1.name} o {g2.name})(x)"
            )


# ---------------------------------------------------------------------------
# Input action: per-element specifics
# ---------------------------------------------------------------------------


def test_sigma_h_is_pure_file_flip():
    """sigma_h flips the file axis only and does NOT touch channels."""
    x = torch.zeros(1, 34, 8, 8)
    x[0, 0, 0, 0] = 1.0
    y = group_action_input(GChess.SIGMA_H, x)
    assert y[0, 0, 0, 7] == 1.0
    assert y[0, 0, 0, 0] == 0.0
    assert torch.all(y[0, 6] == 0.0)


def test_sigma_h_against_torch_flip():
    """Sanity: sigma_h on a random input == torch.flip(x, [-1])."""
    x = make_random_input()
    y = group_action_input(GChess.SIGMA_H, x)
    assert torch.equal(y, torch.flip(x, dims=[-1]))


def test_sigma_c_swaps_pieces_and_flips_ranks():
    """Our pawn at (rank=1, file=0) becomes opp pawn at (rank=6, file=0)."""
    x = torch.zeros(1, 34, 8, 8)
    x[0, 0, 1, 0] = 1.0
    y = group_action_input(GChess.SIGMA_C, x)
    assert y[0, 6, 6, 0] == 1.0
    assert torch.all(y[0, 0] == 0.0)


def test_sigma_c_swaps_history_pieces():
    """Planes 13..18 swap with 19..24."""
    x = torch.zeros(1, 34, 8, 8)
    x[0, 13, 2, 3] = 1.0
    y = group_action_input(GChess.SIGMA_C, x)
    assert y[0, 19, 5, 3] == 1.0


def test_sigma_c_repetition_planes_unchanged_position():
    """Planes 12 and 25 keep their position (only the spatial flip applies)."""
    x = torch.zeros(1, 34, 8, 8)
    x[0, 12] = 1.0
    x[0, 25, 0, 0] = 1.0
    y = group_action_input(GChess.SIGMA_C, x)
    assert torch.all(y[0, 12] == 1.0)
    assert y[0, 25, 7, 0] == 1.0


def test_sigma_c_swaps_castling():
    """Planes 26 <-> 28, 27 <-> 29 under sigma_c."""
    x = torch.zeros(1, 34, 8, 8)
    x[0, 26] = 1.0
    x[0, 27] = 1.0
    y = group_action_input(GChess.SIGMA_C, x)
    assert torch.all(y[0, 26] == 0.0)
    assert torch.all(y[0, 27] == 0.0)
    assert torch.all(y[0, 28] == 1.0)
    assert torch.all(y[0, 29] == 1.0)


def test_sigma_c_flips_side_to_move():
    """Plane 30 is signed in {-1, +1}; sigma_c negates it (linear sign flip)."""
    x = torch.zeros(1, 34, 8, 8)
    x[0, 30] = 1.0
    y = group_action_input(GChess.SIGMA_C, x)
    assert torch.all(y[0, 30] == -1.0)

    x[0, 30] = -1.0
    y = group_action_input(GChess.SIGMA_C, x)
    assert torch.all(y[0, 30] == 1.0)


def test_sigma_c_constants_unchanged():
    """Planes 31, 32, 33 must not be modified by sigma_c."""
    x = make_random_input()
    y = group_action_input(GChess.SIGMA_C, x)
    # 31: scalar broadcast over 8x8 — equal after rank flip
    assert torch.equal(y[:, 31], torch.flip(x[:, 31], dims=[-2]))
    assert torch.equal(y[:, 32], torch.flip(x[:, 32], dims=[-2]))
    assert torch.equal(y[:, 33], torch.flip(x[:, 33], dims=[-2]))


def test_sigma_hc_is_180_rotation_plus_channel_swap():
    """sigma_h o sigma_c = both spatial flips + sigma_c channel work."""
    x = make_random_input()
    via_chain = group_action_input(
        GChess.SIGMA_H, group_action_input(GChess.SIGMA_C, x)
    )
    direct = group_action_input(GChess.SIGMA_HC, x)
    assert torch.equal(via_chain, direct)


def test_channel_perm_is_self_inverse():
    """SIGMA_C_CHANNEL_PERM applied twice equals identity (excluding plane 30 negate)."""
    perm = SIGMA_C_CHANNEL_PERM
    twice = [perm[perm[i]] for i in range(34)]
    assert twice == list(range(34))


# ---------------------------------------------------------------------------
# Output action on body feature maps (unstructured channels)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("g", ELEMENTS)
def test_output_action_self_inverse(g):
    fx = torch.randn(2, 64, 8, 8)
    y = group_action_output(g, fx)
    z = group_action_output(g, y)
    assert torch.equal(z, fx), f"g(g(fx)) != fx exactly for {g.name}"


@pytest.mark.parametrize(
    "g, expected_dims",
    [
        (GChess.E, None),
        (GChess.SIGMA_H, [-1]),
        (GChess.SIGMA_C, [-2]),
        (GChess.SIGMA_HC, [-2, -1]),
    ],
)
def test_output_action_matches_torch_flip(g, expected_dims):
    fx = torch.randn(2, 16, 8, 8)
    y = group_action_output(g, fx)
    if expected_dims is None:
        assert torch.equal(y, fx)
    else:
        assert torch.equal(y, torch.flip(fx, dims=expected_dims))


def test_output_action_is_homomorphism():
    fx = torch.randn(2, 16, 8, 8)
    for g1 in ELEMENTS:
        for g2 in ELEMENTS:
            sequential = group_action_output(g1, group_action_output(g2, fx))
            composed = group_action_output(compose(int(g1), int(g2)), fx)
            assert torch.equal(sequential, composed), (
                f"{g1.name}({g2.name}(fx)) != ({g1.name} o {g2.name})(fx)"
            )


# ---------------------------------------------------------------------------
# Phase 3: policy map structure
# ---------------------------------------------------------------------------


def test_policy_index_size_is_1858():
    assert POLICY_INDEX_SIZE == 1858
    assert len(POLICY_INDEX_TO_MOVE) == 1858
    assert len(POLICY_INDEX_TO_UCI) == 1858


def test_policy_uci_strings_are_unique():
    assert len(set(POLICY_INDEX_TO_UCI)) == POLICY_INDEX_SIZE


def test_policy_move_tuples_are_unique():
    assert len(set(POLICY_INDEX_TO_MOVE)) == POLICY_INDEX_SIZE


def test_policy_index_round_trip_uci():
    for i, uci in enumerate(POLICY_INDEX_TO_UCI):
        assert UCI_TO_POLICY_INDEX[uci] == i


def test_policy_index_round_trip_move():
    for i, m in enumerate(POLICY_INDEX_TO_MOVE):
        assert MOVE_TO_POLICY_INDEX[m] == i


def test_known_policy_entries_present():
    """Spot-check a handful of well-known moves. LCZero's canonical 1858
    list encodes promotions for q/r/b ONLY (knight-promotion shares its
    slot with the corresponding knight move)."""
    for uci in ["a1a8", "h1h8", "e2e4", "e7e5", "g1f3", "b1c3",
                "e7e8q", "e7e8r", "e7e8b"]:
        assert uci in UCI_TO_POLICY_INDEX, f"missing: {uci}"
    for uci in ["e7e8n"]:  # knight promotion is NOT a separate slot
        assert uci not in UCI_TO_POLICY_INDEX


def test_promotion_count_is_66():
    promo_moves = [m for m in POLICY_INDEX_TO_MOVE if m[2]]
    assert len(promo_moves) == 66


def test_queen_and_knight_counts():
    nonpromo_moves = [m for m in POLICY_INDEX_TO_MOVE if not m[2]]
    assert len(nonpromo_moves) == 1456 + 336


# ---------------------------------------------------------------------------
# 80-channel gather (lc0 AZ-policy-head layout)
# ---------------------------------------------------------------------------


def test_policy_gather_indices_shape_and_uniqueness():
    """build_policy_gather_indices returns 1858 unique int64 indices in
    the (80, 8, 8) flat range, all landing in the 73 used planes."""
    from models.policy_map import build_policy_gather_indices, POLICY_NUM_CHANNELS

    idx = build_policy_gather_indices()
    assert idx.shape == (POLICY_INDEX_SIZE,)
    assert idx.dtype == torch.int64
    assert idx.min().item() >= 0
    assert idx.max().item() < POLICY_NUM_CHANNELS * 8 * 8  # < 5120
    # All 1858 entries are distinct.
    assert idx.unique().shape[0] == POLICY_INDEX_SIZE
    # No move gathers from the 7 unused padding planes (73..79).
    assert idx.max().item() < 73 * 64


def test_policy_gather_known_positions():
    """Hand-checked flat-positions for a few canonical moves.

    Channel layout:
      planes  0..55: queen-style. Plane = direction*7 + (steps-1).
                     Direction order: N, NE, E, SE, S, SW, W, NW.
      planes 56..63: knight. Plane = 56 + direction.
      planes 64..72: under-promotion. Plane = 64 + direction*3 + piece_idx.
                     Direction order: NW, N, NE.  Piece order: r, b, q.
    Within a plane, flat-offset = rank * 8 + file.
    """
    from models.policy_map import build_policy_gather_indices

    idx = build_policy_gather_indices()

    # e2e4: queen N (dir 0), steps 2 -> plane 1; rank 1, file 4 -> 1*64 + 12 = 76.
    assert idx[UCI_TO_POLICY_INDEX["e2e4"]].item() == 76

    # g1f3: knight from g1 with vector (-1, 2) which is the NW knight slot
    # (8th direction = index 7) -> plane 56+7 = 63; rank 0, file 6 -> 63*64 + 6 = 4038.
    assert idx[UCI_TO_POLICY_INDEX["g1f3"]].item() == 4038

    # e7e8q: under-promo N (dir 1), piece q (idx 2) -> plane 64 + 1*3 + 2 = 69;
    # rank 6, file 4 -> 69*64 + 52 = 4468.
    assert idx[UCI_TO_POLICY_INDEX["e7e8q"]].item() == 4468


def test_policy_gather_inverse_round_trip():
    """build_flat_to_policy_index is the inverse of build_policy_gather_indices."""
    from models.policy_map import build_flat_to_policy_index, build_policy_gather_indices

    idx = build_policy_gather_indices()
    inv = build_flat_to_policy_index()
    assert len(inv) == POLICY_INDEX_SIZE
    for policy_i, flat in enumerate(idx.tolist()):
        assert inv[flat] == policy_i


# ---------------------------------------------------------------------------
# Phase 3: build_policy_permutation
# ---------------------------------------------------------------------------


def test_policy_permutation_identity_is_arange():
    perm = build_policy_permutation(GChess.E)
    assert torch.equal(perm, torch.arange(POLICY_INDEX_SIZE, dtype=torch.long))


@pytest.mark.parametrize("g", [GChess.SIGMA_H, GChess.SIGMA_C, GChess.SIGMA_HC])
def test_policy_permutation_is_self_inverse(g):
    """DOD: perm[perm] == identity for every non-identity g."""
    perm = build_policy_permutation(g)
    composed = perm[perm]
    expected = torch.arange(POLICY_INDEX_SIZE, dtype=torch.long)
    assert torch.equal(composed, expected), f"perm[perm] != identity for {g.name}"


@pytest.mark.parametrize("g", [GChess.SIGMA_H, GChess.SIGMA_C, GChess.SIGMA_HC])
def test_policy_permutation_is_a_permutation(g):
    """The 1858 indices must each appear exactly once."""
    perm = build_policy_permutation(g)
    sorted_perm, _ = torch.sort(perm)
    assert torch.equal(sorted_perm, torch.arange(POLICY_INDEX_SIZE, dtype=torch.long))


def test_sigma_h_e2e4_maps_to_d2d4():
    """DOD example: under sigma_h (file flip), e2-e4 -> d2-d4."""
    perm = build_policy_permutation(GChess.SIGMA_H)
    src = UCI_TO_POLICY_INDEX["e2e4"]
    dst = UCI_TO_POLICY_INDEX["d2d4"]
    assert perm[src].item() == dst


def test_sigma_h_corner_swap():
    """File flip swaps the a- and h-files."""
    perm = build_policy_permutation(GChess.SIGMA_H)
    assert perm[UCI_TO_POLICY_INDEX["a1a8"]].item() == UCI_TO_POLICY_INDEX["h1h8"]
    assert perm[UCI_TO_POLICY_INDEX["h1h8"]].item() == UCI_TO_POLICY_INDEX["a1a8"]


def test_sigma_h_knight_move():
    """Under sigma_h, b1-c3 -> g1-f3."""
    perm = build_policy_permutation(GChess.SIGMA_H)
    src = UCI_TO_POLICY_INDEX["b1c3"]
    dst = UCI_TO_POLICY_INDEX["g1f3"]
    assert perm[src].item() == dst


def test_sigma_c_e2e4_maps_to_e7e5():
    """Under sigma_c (rank flip on non-promotions), e2-e4 -> e7-e5."""
    perm = build_policy_permutation(GChess.SIGMA_C)
    src = UCI_TO_POLICY_INDEX["e2e4"]
    dst = UCI_TO_POLICY_INDEX["e7e5"]
    assert perm[src].item() == dst


def test_sigma_hc_e2e4_maps_to_d7d5():
    """Under sigma_hc (file + rank flips), e2-e4 -> d7-d5."""
    perm = build_policy_permutation(GChess.SIGMA_HC)
    src = UCI_TO_POLICY_INDEX["e2e4"]
    dst = UCI_TO_POLICY_INDEX["d7d5"]
    assert perm[src].item() == dst


def test_sigma_h_promotion_file_flips():
    """Under sigma_h, a7-a8q -> h7-h8q; a7-b8r -> h7-g8r."""
    perm = build_policy_permutation(GChess.SIGMA_H)
    assert perm[UCI_TO_POLICY_INDEX["a7a8q"]].item() == UCI_TO_POLICY_INDEX["h7h8q"]
    assert perm[UCI_TO_POLICY_INDEX["a7b8r"]].item() == UCI_TO_POLICY_INDEX["h7g8r"]


def test_sigma_c_promotion_is_identity():
    """sigma_c is treated as identity on promotion slots — see actions.py docstring."""
    perm = build_policy_permutation(GChess.SIGMA_C)
    for uci in ["a7a8q", "h7h8r", "e7e8b", "a7b8q", "h7g8r"]:
        idx = UCI_TO_POLICY_INDEX[uci]
        assert perm[idx].item() == idx, f"sigma_c should be identity on {uci}"


def test_sigma_hc_promotion_is_file_flip_only():
    """sigma_hc on promotions = sigma_h alone (since sigma_c is identity on promos)."""
    perm = build_policy_permutation(GChess.SIGMA_HC)
    assert perm[UCI_TO_POLICY_INDEX["a7a8q"]].item() == UCI_TO_POLICY_INDEX["h7h8q"]
    assert perm[UCI_TO_POLICY_INDEX["e7e8b"]].item() == UCI_TO_POLICY_INDEX["d7d8b"]


# ---------------------------------------------------------------------------
# Phase 3: group_action_policy / _value / _mlh
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("g", ELEMENTS)
def test_group_action_policy_self_inverse(g):
    p = torch.randn(2, 1858)
    twice = group_action_policy(g, group_action_policy(g, p))
    assert torch.equal(twice, p)


def test_group_action_policy_matches_permutation():
    """group_action_policy must use the build_policy_permutation indices."""
    p = torch.arange(1858, dtype=torch.float32).unsqueeze(0)
    for g in [GChess.SIGMA_H, GChess.SIGMA_C, GChess.SIGMA_HC]:
        perm = build_policy_permutation(g)
        out = group_action_policy(g, p)
        # out[0, j] == p[0, perm[j]] == perm[j]
        assert torch.equal(out[0], perm.float())


def test_group_action_policy_concrete_e2e4():
    """If logit at e2e4 = 7.0 and we apply sigma_h, that 7.0 ends up at d2d4."""
    p = torch.zeros(1, 1858)
    p[0, UCI_TO_POLICY_INDEX["e2e4"]] = 7.0
    out = group_action_policy(GChess.SIGMA_H, p)
    assert out[0, UCI_TO_POLICY_INDEX["d2d4"]].item() == 7.0
    # And the original slot should be 0 (since d2d4 was 0 originally and
    # under sigma_h the d2d4 logit comes from the e2e4 slot... let's just
    # check the e2e4 slot got overwritten by whatever was at sigma_h(e2e4) = d2d4).
    assert out[0, UCI_TO_POLICY_INDEX["e2e4"]].item() == 0.0


@pytest.mark.parametrize("g", ELEMENTS)
def test_group_action_value_self_inverse(g):
    v = torch.softmax(torch.randn(2, 3), dim=-1)
    twice = group_action_value(g, group_action_value(g, v))
    assert torch.allclose(twice, v)


def test_group_action_value_sigma_h_is_identity():
    v = torch.tensor([[0.6, 0.3, 0.1]])
    assert torch.equal(group_action_value(GChess.SIGMA_H, v), v)


def test_group_action_value_sigma_c_swaps_W_and_L():
    v = torch.tensor([[0.6, 0.3, 0.1]])  # W=0.6, D=0.3, L=0.1
    out = group_action_value(GChess.SIGMA_C, v)
    assert torch.equal(out, torch.tensor([[0.1, 0.3, 0.6]]))


def test_group_action_value_sigma_hc_swaps_W_and_L():
    v = torch.tensor([[0.6, 0.3, 0.1]])
    out = group_action_value(GChess.SIGMA_HC, v)
    assert torch.equal(out, torch.tensor([[0.1, 0.3, 0.6]]))


@pytest.mark.parametrize("g", ELEMENTS)
def test_group_action_mlh_invariant(g):
    m = torch.tensor([[5.0], [12.0]])
    out = group_action_mlh(g, m)
    assert torch.equal(out, m)
