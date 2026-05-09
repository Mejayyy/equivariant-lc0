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
    group_action_input,
    group_action_output,
)


def make_random_input(batch=2, seed=0):
    """Sample inputs matching CLAUDE.md sampling convention.

    Plane 30 (side-to-move) MUST be exactly 0.0 or 1.0 — otherwise
    1 - (1 - x) is not bit-exact under IEEE-754, breaking the
    invertibility test.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(batch, 34, 8, 8)
    x[:, 0:12] = (torch.rand(batch, 12, 8, 8, generator=g) > 0.85).float()
    x[:, 12] = (torch.rand(batch, 8, 8, generator=g) > 0.9).float()
    x[:, 13:25] = (torch.rand(batch, 12, 8, 8, generator=g) > 0.85).float()
    x[:, 25] = (torch.rand(batch, 8, 8, generator=g) > 0.9).float()
    x[:, 26:30] = (torch.rand(batch, 4, 8, 8, generator=g) > 0.5).float()
    side = (torch.rand(batch, generator=g) > 0.5).float()
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
    """Plane 30 must be 1 - x[30] under sigma_c."""
    x = torch.zeros(1, 34, 8, 8)
    x[0, 30] = 1.0
    y = group_action_input(GChess.SIGMA_C, x)
    assert torch.all(y[0, 30] == 0.0)

    x[0, 30] = 0.0
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
