"""Phase 8: per-head equivariance/invariance with random body-output tensors.

DOD (CLAUDE.md):
> per-head equivariance with random body-output tensors

Each equivariant head is tested in isolation: we feed a random folded
G-signal (B, F*|G|, 8, 8), apply group_action_g_signal(g, ...) as the
input transform, and compare the head's output against the head-specific
output action:
- EquivariantPolicyHead   <-> group_action_policy
- EquivariantValueHead    <-> group_action_value
- InvariantMovesLeftHead  <-> group_action_mlh (identity)

We also include a few hand-checked tests for the orbit-based policy
gather (build_orbit_policy_gather) to lock in its shape, range, and
duplicate-handling.
"""

import pytest
import torch

from models.gcnn import (
    GCHESS_GROUP,
    GROUP_SIZE,
    SGCHESS_GROUP,
    group_action_g_signal,
)
from models.heads import (
    EquivariantPolicyHead,
    EquivariantValueHead,
    InvariantMovesLeftHead,
)
from models.policy_map import (
    POLICY_INDEX_SIZE,
    POLICY_NUM_CHANNELS,
    build_orbit_policy_gather,
    build_sgchess_orbit_policy_gather,
)
from symmetry.actions import (
    build_policy_permutation,
    group_action_mlh,
    group_action_policy,
    group_action_value,
)
from symmetry.group import GChess, SGChess


NON_IDENTITY = [GChess.SIGMA_H, GChess.SIGMA_C, GChess.SIGMA_HC]
EQUIV_TOL = 1e-5
# Per-head policy: 2 GConvs into 80 channels + GBN + sum-pool + gather +
# mean accumulates more float32 roundoff than the per-primitive tests in
# test_gcnn.py. The end-to-end test_equivariance.py tolerance stays at
# 1e-5 (and passes by a comfortable margin); only this isolated-head
# test uses the slightly looser bound.
POLICY_PER_HEAD_TOL = 5e-5

SGCHESS_ORDER = 2


# ---------------------------------------------------------------------------
# Orbit-based gather: structural sanity
# ---------------------------------------------------------------------------


def test_orbit_gather_shape_and_dtype():
    g = build_orbit_policy_gather()
    assert g.shape == (POLICY_INDEX_SIZE, 2)
    assert g.dtype == torch.int64


def test_orbit_gather_range():
    g = build_orbit_policy_gather()
    flat_range = POLICY_NUM_CHANNELS * 64
    assert int(g.min()) >= 0
    assert int(g.max()) < flat_range


def test_orbit_gather_uses_few_channels():
    """481 orbits * 16 blocks/channel needs 31 channels; the rest unused."""
    g = build_orbit_policy_gather()
    used_channels = (g // 64).unique().tolist()
    # 481 orbits across 16 blocks per channel -> ceil(481/16) = 31 channels.
    assert len(used_channels) <= 31
    assert max(used_channels) < 31


def test_orbit_gather_non_promo_columns_are_equal():
    """For non-promo moves, gather[i, 0] == gather[i, 1] (single-cell)."""
    from symmetry.actions import build_policy_permutation
    perm_c = build_policy_permutation(GChess.SIGMA_C).tolist()
    g = build_orbit_policy_gather()
    for i in range(POLICY_INDEX_SIZE):
        is_promo = (perm_c[i] == i)
        if not is_promo:
            assert g[i, 0].item() == g[i, 1].item(), (
                f"non-promo move {i} has differing gather columns"
            )


def test_orbit_gather_promo_columns_form_rank_pair():
    """Promo gather: (chan, r, f) and (chan, 7-r, f) — same channel and file."""
    perm_c = build_policy_permutation(GChess.SIGMA_C).tolist()
    g = build_orbit_policy_gather()
    for i in range(POLICY_INDEX_SIZE):
        if perm_c[i] != i:
            continue  # non-promo
        a, b = int(g[i, 0]), int(g[i, 1])
        chan_a, sq_a = a // 64, a % 64
        chan_b, sq_b = b // 64, b % 64
        ra, fa = sq_a // 8, sq_a % 8
        rb, fb = sq_b // 8, sq_b % 8
        assert chan_a == chan_b, f"promo {i}: differing channels"
        assert fa == fb, f"promo {i}: differing files"
        assert ra + rb == 7, f"promo {i}: ranks {ra}, {rb} not a pair"


def test_orbit_gather_under_spatial_flip_matches_policy_permutation():
    """Hand-check the head's equivariance: applying spatial flip to a
    random (80, 8, 8) tensor and gathering+averaging gives the same
    permutation as build_policy_permutation, modulo float roundoff."""
    torch.manual_seed(0)
    z = torch.randn(1, POLICY_NUM_CHANNELS, 8, 8)
    gather = build_orbit_policy_gather()  # (1858, 2)

    def head_gather(z_):
        flat = z_.view(1, -1)
        g = flat.index_select(1, gather.view(-1)).view(1, POLICY_INDEX_SIZE, 2)
        return g.mean(dim=2)

    base = head_gather(z)
    for g in NON_IDENTITY:
        if g == GChess.SIGMA_H:
            z_g = torch.flip(z, dims=[-1])
        elif g == GChess.SIGMA_C:
            z_g = torch.flip(z, dims=[-2])
        else:  # SIGMA_HC
            z_g = torch.flip(z, dims=[-2, -1])
        natural = head_gather(z_g)
        canonical = group_action_policy(g, base)
        diff = (natural - canonical).abs().max().item()
        assert diff < EQUIV_TOL, (
            f"orbit-gather sigma_{g.name} disagreement: max diff = {diff}"
        )


# ---------------------------------------------------------------------------
# EquivariantPolicyHead
# ---------------------------------------------------------------------------


def test_equivariant_policy_head_shape():
    F = 8
    head = EquivariantPolicyHead(filters=F)
    x = torch.randn(2, F * GROUP_SIZE, 8, 8)
    out = head(x)
    assert out.shape == (2, POLICY_INDEX_SIZE)
    assert out.dtype == torch.float32


@pytest.mark.parametrize("g", NON_IDENTITY)
def test_equivariant_policy_head_equivariance(g):
    """head(group_action_g_signal(g, x)) == group_action_policy(g, head(x))."""
    torch.manual_seed(0)
    F = 8
    head = EquivariantPolicyHead(filters=F)
    head.eval()

    x = torch.randn(2, F * GROUP_SIZE, 8, 8)
    gx = group_action_g_signal(g, x, num_logical_channels=F)

    with torch.no_grad():
        out_x = head(x)
        out_gx = head(gx)
        g_out_x = group_action_policy(g, out_x)

    diff = (out_gx - g_out_x).abs().max().item()
    assert diff < POLICY_PER_HEAD_TOL, (
        f"EquivariantPolicyHead {g.name}: max diff = {diff}"
    )


# ---------------------------------------------------------------------------
# EquivariantValueHead
# ---------------------------------------------------------------------------


def test_equivariant_value_head_shape_and_simplex():
    F = 8
    head = EquivariantValueHead(filters=F)
    x = torch.randn(3, F * GROUP_SIZE, 8, 8)
    out = head(x)
    assert out.shape == (3, 3)
    assert torch.allclose(out.sum(dim=1), torch.ones(3), atol=1e-5)
    assert (out >= 0).all() and (out <= 1).all()


def test_equivariant_value_head_shares_wl_linear():
    """Sanity: there is a SINGLE shared nn.Linear instance for W and L,
    not two separate-with-copied-weights — this matters for sigma_c
    equivariance under training (gradients accumulate into one weight)."""
    head = EquivariantValueHead(filters=8)
    # The same instance is referenced under one name; assert it's not
    # duplicated under another internal attribute.
    linear_modules = [
        m for m in head.modules() if isinstance(m, torch.nn.Linear)
    ]
    # Expect exactly two: wl_linear (shared) + d_linear.
    assert len(linear_modules) == 2
    assert head.wl_linear is linear_modules[0] or head.wl_linear is linear_modules[1]


@pytest.mark.parametrize("g", NON_IDENTITY)
def test_equivariant_value_head_equivariance(g):
    torch.manual_seed(0)
    F = 8
    head = EquivariantValueHead(filters=F)
    head.eval()

    x = torch.randn(2, F * GROUP_SIZE, 8, 8)
    gx = group_action_g_signal(g, x, num_logical_channels=F)

    with torch.no_grad():
        out_x = head(x)
        out_gx = head(gx)
        g_out_x = group_action_value(g, out_x)

    diff = (out_gx - g_out_x).abs().max().item()
    assert diff < EQUIV_TOL, (
        f"EquivariantValueHead {g.name}: max diff = {diff}"
    )


# ---------------------------------------------------------------------------
# InvariantMovesLeftHead
# ---------------------------------------------------------------------------


def test_invariant_mlh_head_shape_and_nonneg():
    F = 8
    head = InvariantMovesLeftHead(filters=F, mlh_channels=8, fc_size=16)
    x = torch.randn(3, F * GROUP_SIZE, 8, 8)
    out = head(x)
    assert out.shape == (3, 1)
    assert (out >= 0).all()


@pytest.mark.parametrize("g", NON_IDENTITY)
def test_invariant_mlh_head_invariance(g):
    """head(group_action_g_signal(g, x)) == head(x) (mlh is fully invariant)."""
    torch.manual_seed(0)
    F = 8
    head = InvariantMovesLeftHead(filters=F, mlh_channels=8, fc_size=16)
    head.eval()

    x = torch.randn(2, F * GROUP_SIZE, 8, 8)
    gx = group_action_g_signal(g, x, num_logical_channels=F)

    with torch.no_grad():
        out_x = head(x)
        out_gx = head(gx)
        g_out_x = group_action_mlh(g, out_x)  # identity

    diff = (out_gx - g_out_x).abs().max().item()
    assert diff < EQUIV_TOL, (
        f"InvariantMovesLeftHead {g.name}: max diff = {diff}"
    )


# ===========================================================================
# SG_chess subgroup ({e, sigma_c}, |G|=2): heads
# ===========================================================================


# ---------------------------------------------------------------------------
# SGChess orbit-based policy gather: structural sanity
# ---------------------------------------------------------------------------


def test_sgchess_orbit_gather_shape_and_dtype():
    g = build_sgchess_orbit_policy_gather()
    assert g.shape == (POLICY_INDEX_SIZE, 2)
    assert g.dtype == torch.int64


def test_sgchess_orbit_gather_range():
    g = build_sgchess_orbit_policy_gather()
    flat_range = POLICY_NUM_CHANNELS * 64
    assert int(g.min()) >= 0
    assert int(g.max()) < flat_range


def test_sgchess_orbit_gather_uses_few_channels():
    """962 orbits at 32 slots/channel -> ceil(962/32) = 31 channels."""
    g = build_sgchess_orbit_policy_gather()
    used_channels = (g // 64).unique().tolist()
    assert len(used_channels) <= 31
    assert max(used_channels) < 31


def test_sgchess_orbit_gather_non_promo_columns_are_equal():
    """For non-promo moves under SGChess, gather[i, 0] == gather[i, 1]
    (single-cell duplicated — the mean is trivial)."""
    perm_c = build_policy_permutation(GChess.SIGMA_C).tolist()
    g = build_sgchess_orbit_policy_gather()
    for i in range(POLICY_INDEX_SIZE):
        is_promo = (perm_c[i] == i)
        if not is_promo:
            assert g[i, 0].item() == g[i, 1].item(), (
                f"SGChess non-promo move {i} has differing gather columns"
            )


def test_sgchess_orbit_gather_promo_columns_form_rank_pair():
    """Promo gather: (chan, r, f) and (chan, 7-r, f) — same channel, file."""
    perm_c = build_policy_permutation(GChess.SIGMA_C).tolist()
    g = build_sgchess_orbit_policy_gather()
    for i in range(POLICY_INDEX_SIZE):
        if perm_c[i] != i:
            continue  # non-promo
        a, b = int(g[i, 0]), int(g[i, 1])
        chan_a, sq_a = a // 64, a % 64
        chan_b, sq_b = b // 64, b % 64
        ra, fa = sq_a // 8, sq_a % 8
        rb, fb = sq_b // 8, sq_b % 8
        assert chan_a == chan_b, f"SGChess promo {i}: differing channels"
        assert fa == fb, f"SGChess promo {i}: differing files"
        assert ra + rb == 7, f"SGChess promo {i}: ranks {ra}, {rb} not a pair"


def test_sgchess_orbit_gather_under_rank_flip_matches_policy_permutation():
    """The head's equivariance under sigma_c (rank flip): gather of the
    rank-flipped sum-pool tensor equals build_policy_permutation(sigma_c)
    applied to the original gather output."""
    torch.manual_seed(0)
    z = torch.randn(1, POLICY_NUM_CHANNELS, 8, 8)
    gather = build_sgchess_orbit_policy_gather()

    def head_gather(z_):
        flat = z_.view(1, -1)
        g = flat.index_select(1, gather.view(-1)).view(1, POLICY_INDEX_SIZE, 2)
        return g.mean(dim=2)

    base = head_gather(z)
    z_c = torch.flip(z, dims=[-2])  # rank flip on the (B, C, H, W) sum-pool
    natural = head_gather(z_c)
    canonical = group_action_policy(GChess.SIGMA_C, base)
    diff = (natural - canonical).abs().max().item()
    assert diff < EQUIV_TOL, f"SGChess orbit-gather sigma_c: max diff = {diff}"


# ---------------------------------------------------------------------------
# EquivariantPolicyHead (SGChess)
# ---------------------------------------------------------------------------


def test_equivariant_policy_head_sgchess_shape():
    F = 8
    head = EquivariantPolicyHead(filters=F, group=SGCHESS_GROUP)
    x = torch.randn(2, F * SGCHESS_ORDER, 8, 8)
    out = head(x)
    assert out.shape == (2, POLICY_INDEX_SIZE)
    assert out.dtype == torch.float32


def test_equivariant_policy_head_sgchess_uses_sgchess_gather():
    """Selecting group=SGCHESS_GROUP wires the head's policy_gather to
    the SGChess-orbit gather (not the GChess one). Verified by direct
    equality of the buffer."""
    head = EquivariantPolicyHead(filters=4, group=SGCHESS_GROUP)
    expected = build_sgchess_orbit_policy_gather(EquivariantPolicyHead.POLICY_NUM_CHANNELS)
    assert torch.equal(head.policy_gather, expected)


def test_equivariant_policy_head_sgchess_equivariance():
    """head(group_action_g_signal(sigma_c, x; SGChess)) ==
       group_action_policy(sigma_c, head(x))."""
    torch.manual_seed(0)
    F = 8
    head = EquivariantPolicyHead(filters=F, group=SGCHESS_GROUP)
    head.eval()

    x = torch.randn(2, F * SGCHESS_ORDER, 8, 8)
    gx = group_action_g_signal(
        SGChess.SIGMA_C, x, num_logical_channels=F, group=SGCHESS_GROUP,
    )

    with torch.no_grad():
        out_x = head(x)
        out_gx = head(gx)
        g_out_x = group_action_policy(GChess.SIGMA_C, out_x)

    diff = (out_gx - g_out_x).abs().max().item()
    assert diff < POLICY_PER_HEAD_TOL, (
        f"SGChess EquivariantPolicyHead sigma_c: max diff = {diff}"
    )


# ---------------------------------------------------------------------------
# EquivariantValueHead (SGChess)
# ---------------------------------------------------------------------------


def test_equivariant_value_head_sgchess_shape_and_simplex():
    F = 8
    head = EquivariantValueHead(filters=F, group=SGCHESS_GROUP)
    x = torch.randn(3, F * SGCHESS_ORDER, 8, 8)
    out = head(x)
    assert out.shape == (3, 3)
    assert torch.allclose(out.sum(dim=1), torch.ones(3), atol=1e-5)
    assert (out >= 0).all() and (out <= 1).all()


def test_equivariant_value_head_sgchess_w_l_indices_partition_group():
    """W-half: local index 0 (E); L-half: local index 1 (SIGMA_C).
    Together they cover the whole 2-element SGChess group."""
    head = EquivariantValueHead(filters=4, group=SGCHESS_GROUP)
    w = head.w_half_indices.tolist()
    l = head.l_half_indices.tolist()
    assert w == [0]
    assert l == [1]


def test_equivariant_value_head_sgchess_shares_wl_linear():
    head = EquivariantValueHead(filters=8, group=SGCHESS_GROUP)
    linear_modules = [
        m for m in head.modules() if isinstance(m, torch.nn.Linear)
    ]
    assert len(linear_modules) == 2  # wl_linear (shared) + d_linear
    assert head.wl_linear is linear_modules[0] or head.wl_linear is linear_modules[1]


def test_equivariant_value_head_sgchess_equivariance():
    """Under SGChess sigma_c, the value WDL output swaps W <-> L."""
    torch.manual_seed(0)
    F = 8
    head = EquivariantValueHead(filters=F, group=SGCHESS_GROUP)
    head.eval()

    x = torch.randn(2, F * SGCHESS_ORDER, 8, 8)
    gx = group_action_g_signal(
        SGChess.SIGMA_C, x, num_logical_channels=F, group=SGCHESS_GROUP,
    )

    with torch.no_grad():
        out_x = head(x)
        out_gx = head(gx)
        g_out_x = group_action_value(GChess.SIGMA_C, out_x)

    diff = (out_gx - g_out_x).abs().max().item()
    assert diff < EQUIV_TOL, (
        f"SGChess EquivariantValueHead sigma_c: max diff = {diff}"
    )


# ---------------------------------------------------------------------------
# InvariantMovesLeftHead (SGChess)
# ---------------------------------------------------------------------------


def test_invariant_mlh_head_sgchess_shape_and_nonneg():
    F = 8
    head = InvariantMovesLeftHead(filters=F, mlh_channels=8, fc_size=16, group=SGCHESS_GROUP)
    x = torch.randn(3, F * SGCHESS_ORDER, 8, 8)
    out = head(x)
    assert out.shape == (3, 1)
    assert (out >= 0).all()


def test_invariant_mlh_head_sgchess_invariance():
    torch.manual_seed(0)
    F = 8
    head = InvariantMovesLeftHead(filters=F, mlh_channels=8, fc_size=16, group=SGCHESS_GROUP)
    head.eval()

    x = torch.randn(2, F * SGCHESS_ORDER, 8, 8)
    gx = group_action_g_signal(
        SGChess.SIGMA_C, x, num_logical_channels=F, group=SGCHESS_GROUP,
    )

    with torch.no_grad():
        out_x = head(x)
        out_gx = head(gx)
        g_out_x = group_action_mlh(GChess.SIGMA_C, out_x)  # identity

    diff = (out_gx - g_out_x).abs().max().item()
    assert diff < EQUIV_TOL, (
        f"SGChess InvariantMovesLeftHead sigma_c: max diff = {diff}"
    )
