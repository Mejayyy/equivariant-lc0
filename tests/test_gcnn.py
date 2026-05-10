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
    GBatchNorm,
    GConv,
    GLinear,
    GSELayer,
    LiftingConv,
    group_action_g_signal,
    regular_rep_perm,
)
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
