"""Phase 7: G-CNN primitives — shape + per-layer equivariance tests.

DOD (CLAUDE.md):
> per-layer shape tests + per-layer equivariance: each primitive in
> isolation has equivariance error < 1e-5

Equivariance is checked AT INIT (no training). With Kaiming-init kernels
and default BN running stats, every primitive in this file is equivariant
by construction; the test asserts this against the regular-representation
group action on G-signals.
"""

import pytest
import torch

from models.gcnn import (
    GROUP_SIZE,
    ChessLiftingConv,
    GBatchNorm,
    GConv,
    GLinear,
    GSELayer,
    LiftingConv,
    group_action_g_signal,
    regular_rep_perm,
)
from symmetry.actions import group_action_input
from symmetry.actions import group_action_output as spatial_action
from symmetry.group import COMPOSITION_TABLE, ELEMENTS, GChess


# ---------------------------------------------------------------------------
# Helper sanity: regular_rep_perm and group_action_g_signal
# ---------------------------------------------------------------------------


def test_regular_rep_perm_identity():
    perm = regular_rep_perm(GChess.E, num_logical_channels=3)
    assert torch.equal(perm, torch.arange(3 * GROUP_SIZE, dtype=torch.long))


@pytest.mark.parametrize("g", ELEMENTS)
def test_regular_rep_perm_self_inverse(g):
    perm = regular_rep_perm(g, num_logical_channels=2)
    twice = perm[perm]
    assert torch.equal(twice, torch.arange(2 * GROUP_SIZE, dtype=torch.long))


@pytest.mark.parametrize("g", ELEMENTS)
def test_group_action_g_signal_self_inverse(g):
    """Apply g twice should give back the original tensor exactly."""
    x = torch.randn(2, 3 * GROUP_SIZE, 8, 8)
    twice = group_action_g_signal(g, group_action_g_signal(g, x, 3), 3)
    assert torch.equal(twice, x)


# ---------------------------------------------------------------------------
# LiftingConv
# ---------------------------------------------------------------------------


def test_lifting_conv_shape():
    layer = LiftingConv(in_channels=5, out_channels=4, kernel_size=3, padding=1)
    x = torch.randn(2, 5, 8, 8)
    y = layer(x)
    assert y.shape == (2, 4 * GROUP_SIZE, 8, 8)


@pytest.mark.parametrize("g", [GChess.SIGMA_H, GChess.SIGMA_C, GChess.SIGMA_HC])
def test_lifting_conv_equivariance(g):
    """For a Z^2 input x, LiftConv(spatial_g . x) == g . LiftConv(x), where
    g acts on the output as the full G-signal action (regular rep + flip)."""
    torch.manual_seed(0)
    layer = LiftingConv(in_channels=5, out_channels=4, kernel_size=3, padding=1)
    layer.eval()

    x = torch.randn(2, 5, 8, 8)
    gx = spatial_action(g, x)  # input has no group dim, so input action = pure spatial flip

    with torch.no_grad():
        out_x = layer(x)
        out_gx = layer(gx)
        g_out_x = group_action_g_signal(g, out_x, num_logical_channels=4)

    diff = (out_gx - g_out_x).abs().max().item()
    assert diff < 1e-5, f"LiftingConv equivariance failed for {g.name}: max diff = {diff}"


# ---------------------------------------------------------------------------
# GConv
# ---------------------------------------------------------------------------


def test_gconv_shape():
    layer = GConv(in_channels=4, out_channels=6, kernel_size=3, padding=1)
    x = torch.randn(2, 4 * GROUP_SIZE, 8, 8)
    y = layer(x)
    assert y.shape == (2, 6 * GROUP_SIZE, 8, 8)


@pytest.mark.parametrize("g", [GChess.SIGMA_H, GChess.SIGMA_C, GChess.SIGMA_HC])
def test_gconv_equivariance(g):
    torch.manual_seed(0)
    layer = GConv(in_channels=4, out_channels=6, kernel_size=3, padding=1)
    layer.eval()

    x = torch.randn(2, 4 * GROUP_SIZE, 8, 8)
    gx = group_action_g_signal(g, x, num_logical_channels=4)

    with torch.no_grad():
        out_x = layer(x)
        out_gx = layer(gx)
        g_out_x = group_action_g_signal(g, out_x, num_logical_channels=6)

    diff = (out_gx - g_out_x).abs().max().item()
    assert diff < 1e-5, f"GConv equivariance failed for {g.name}: max diff = {diff}"


# ---------------------------------------------------------------------------
# GLinear (used inside GSELayer; also useful on its own)
# ---------------------------------------------------------------------------


def test_glinear_shape():
    layer = GLinear(in_channels=4, out_channels=6)
    x = torch.randn(2, 4 * GROUP_SIZE)
    y = layer(x)
    assert y.shape == (2, 6 * GROUP_SIZE)


@pytest.mark.parametrize("g", [GChess.SIGMA_H, GChess.SIGMA_C, GChess.SIGMA_HC])
def test_glinear_equivariance(g):
    """GLinear acts on a (B, C*|G|) "1x1 spatial" G-signal. Input action
    is just the regular-representation channel permutation (no spatial)."""
    torch.manual_seed(0)
    layer = GLinear(in_channels=4, out_channels=6)
    layer.eval()

    x = torch.randn(2, 4 * GROUP_SIZE)
    in_perm = regular_rep_perm(g, 4)
    out_perm = regular_rep_perm(g, 6)
    gx = x.index_select(dim=1, index=in_perm)

    with torch.no_grad():
        out_x = layer(x)
        out_gx = layer(gx)
        g_out_x = out_x.index_select(dim=1, index=out_perm)

    diff = (out_gx - g_out_x).abs().max().item()
    assert diff < 1e-5, f"GLinear equivariance failed for {g.name}: max diff = {diff}"


# ---------------------------------------------------------------------------
# GBatchNorm
# ---------------------------------------------------------------------------


def test_gbatchnorm_shape():
    layer = GBatchNorm(num_logical_channels=4)
    x = torch.randn(2, 4 * GROUP_SIZE, 8, 8)
    y = layer(x)
    assert y.shape == x.shape


@pytest.mark.parametrize("g", [GChess.SIGMA_H, GChess.SIGMA_C, GChess.SIGMA_HC])
def test_gbatchnorm_equivariance_at_init(g):
    """At init, BN running stats are defaults (mean=0, var=1) and gamma=1,
    beta=0 — so BN scales each channel uniformly. Equivariance must hold
    exactly. We test in eval mode so running stats are used."""
    layer = GBatchNorm(num_logical_channels=4)
    layer.eval()

    x = torch.randn(2, 4 * GROUP_SIZE, 8, 8)
    gx = group_action_g_signal(g, x, num_logical_channels=4)

    with torch.no_grad():
        out_x = layer(x)
        out_gx = layer(gx)
        g_out_x = group_action_g_signal(g, out_x, num_logical_channels=4)

    diff = (out_gx - g_out_x).abs().max().item()
    assert diff < 1e-5, f"GBatchNorm equivariance failed for {g.name}: max diff = {diff}"


@pytest.mark.parametrize("g", [GChess.SIGMA_H, GChess.SIGMA_C, GChess.SIGMA_HC])
def test_gbatchnorm_equivariance_after_training(g):
    """After SGD steps with non-trivial input, BN running stats are no
    longer at defaults. Equivariance MUST still hold because stats are
    shared across the |G| copies of each logical channel."""
    torch.manual_seed(1)
    layer = GBatchNorm(num_logical_channels=4)
    opt = torch.optim.SGD(layer.parameters(), lr=1e-2)

    layer.train()
    for _ in range(5):
        x_train = torch.randn(4, 4 * GROUP_SIZE, 8, 8)
        out = layer(x_train)
        loss = out.pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    layer.eval()
    x = torch.randn(2, 4 * GROUP_SIZE, 8, 8)
    gx = group_action_g_signal(g, x, num_logical_channels=4)

    with torch.no_grad():
        out_x = layer(x)
        out_gx = layer(gx)
        g_out_x = group_action_g_signal(g, out_x, num_logical_channels=4)

    diff = (out_gx - g_out_x).abs().max().item()
    assert diff < 1e-5, (
        f"GBatchNorm equivariance broke after training for {g.name}: "
        f"max diff = {diff}"
    )


# ---------------------------------------------------------------------------
# GSELayer
# ---------------------------------------------------------------------------


def test_gse_layer_shape():
    layer = GSELayer(channels=4, se_channels=2)
    x = torch.randn(2, 4 * GROUP_SIZE, 8, 8)
    y = layer(x)
    assert y.shape == x.shape


@pytest.mark.parametrize("g", [GChess.SIGMA_H, GChess.SIGMA_C, GChess.SIGMA_HC])
def test_gse_layer_equivariance(g):
    torch.manual_seed(0)
    layer = GSELayer(channels=4, se_channels=2)
    layer.eval()

    x = torch.randn(2, 4 * GROUP_SIZE, 8, 8)
    gx = group_action_g_signal(g, x, num_logical_channels=4)

    with torch.no_grad():
        out_x = layer(x)
        out_gx = layer(gx)
        g_out_x = group_action_g_signal(g, out_x, num_logical_channels=4)

    diff = (out_gx - g_out_x).abs().max().item()
    assert diff < 1e-5, f"GSELayer equivariance failed for {g.name}: max diff = {diff}"


# ---------------------------------------------------------------------------
# ChessLiftingConv: equivariance under the FULL chess input action
# (sigma_c includes channel permutation + plane-30 negation, not just
# spatial flip — plain LiftingConv is not equivariant under it).
# ---------------------------------------------------------------------------


def _sample_chess_input(batch=2, seed=0):
    """Sample a 34-plane chess-style input for ChessLiftingConv tests."""
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(batch, 34, 8, 8)
    x[:, 0:12] = (torch.rand(batch, 12, 8, 8, generator=g) > 0.85).float()
    x[:, 12] = (torch.rand(batch, 8, 8, generator=g) > 0.9).float()
    x[:, 13:25] = (torch.rand(batch, 12, 8, 8, generator=g) > 0.85).float()
    x[:, 25] = (torch.rand(batch, 8, 8, generator=g) > 0.9).float()
    x[:, 26:30] = (torch.rand(batch, 4, 8, 8, generator=g) > 0.5).float()
    side = (torch.rand(batch, generator=g) > 0.5).float()
    x[:, 30] = side.view(batch, 1, 1).expand(batch, 8, 8)
    x[:, 31] = torch.rand(batch, generator=g).view(batch, 1, 1).expand(batch, 8, 8)
    x[:, 32] = 0.0
    x[:, 33] = 1.0
    return x


def test_chess_lifting_conv_shape():
    layer = ChessLiftingConv(out_channels=4, kernel_size=3, padding=1)
    x = _sample_chess_input()
    y = layer(x)
    assert y.shape == (2, 4 * GROUP_SIZE, 8, 8)


def test_chess_lifting_conv_plane_30_kernel_is_zero():
    """The sigma_c-equivariance constraint zeros plane 30's kernel."""
    layer = ChessLiftingConv(out_channels=4, kernel_size=3, padding=1)
    kernel = layer._build_kernel()  # (4*|G|, 34, 3, 3)
    assert torch.equal(kernel[:, 30], torch.zeros_like(kernel[:, 30]))


def test_chess_lifting_conv_paired_kernels_equal():
    """Sanity: kernel slices for sigma_c-paired channels are equal."""
    from symmetry.actions import SIGMA_C_CHANNEL_PERM

    layer = ChessLiftingConv(out_channels=4, kernel_size=3, padding=1)
    kernel = layer._build_kernel()  # (4*|G|, 34, 3, 3) — for the e-group slice (g=0).
    # Take only the g=0 sub-kernel: kernel[c*|G|+0, :, :, :] for c in 0..3.
    # In our c-major folding that's kernel[0::|G|].
    e_kernel = kernel[0::GROUP_SIZE]  # (4, 34, 3, 3)
    for c in range(34):
        if c == 30:
            continue
        partner = SIGMA_C_CHANNEL_PERM[c]
        if partner == c:
            continue  # singleton orbit — trivially equal to self
        assert torch.equal(e_kernel[:, c], e_kernel[:, partner]), (
            f"kernel slices {c} and {partner} should be equal"
        )


@pytest.mark.parametrize("g", [GChess.SIGMA_H, GChess.SIGMA_C, GChess.SIGMA_HC])
def test_chess_lifting_conv_equivariance_full_input_action(g):
    """The DOD: ChessLiftingConv(group_action_input(g, x)) equals
    g . ChessLiftingConv(x), where the input action is the FULL chess
    sigma_c (rank flip + channel perm + plane-30 negate) and the output
    action is the standard G-signal action."""
    torch.manual_seed(0)
    layer = ChessLiftingConv(out_channels=4, kernel_size=3, padding=1)
    layer.eval()

    x = _sample_chess_input()
    gx = group_action_input(g, x)  # FULL chess action, not just spatial

    with torch.no_grad():
        out_x = layer(x)
        out_gx = layer(gx)
        g_out_x = group_action_g_signal(g, out_x, num_logical_channels=4)

    diff = (out_gx - g_out_x).abs().max().item()
    assert diff < 1e-5, (
        f"ChessLiftingConv {g.name}: max diff = {diff}"
    )


# ---------------------------------------------------------------------------
# Stack: LiftingConv -> GBatchNorm -> ReLU -> GConv (mini-pipeline) is also
# equivariant. This is a smoke check that primitives compose correctly.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("g", [GChess.SIGMA_H, GChess.SIGMA_C, GChess.SIGMA_HC])
def test_mini_pipeline_equivariance(g):
    torch.manual_seed(0)
    lift = LiftingConv(in_channels=3, out_channels=5, kernel_size=3, padding=1)
    bn = GBatchNorm(num_logical_channels=5)
    gconv = GConv(in_channels=5, out_channels=4, kernel_size=3, padding=1)

    lift.eval(); bn.eval(); gconv.eval()

    def fwd(z):
        z = lift(z)
        z = bn(z)
        z = torch.nn.functional.relu(z)
        z = gconv(z)
        return z

    x = torch.randn(2, 3, 8, 8)
    gx = spatial_action(g, x)  # input is a Z^2 signal

    with torch.no_grad():
        out_x = fwd(x)
        out_gx = fwd(gx)
        g_out_x = group_action_g_signal(g, out_x, num_logical_channels=4)

    diff = (out_gx - g_out_x).abs().max().item()
    assert diff < 1e-5, f"mini-pipeline equivariance failed for {g.name}: max diff = {diff}"
