"""Phase 7: G-CNN primitives — shape + per-layer equivariance tests.

DOD (CLAUDE.md):
> per-layer shape tests + per-layer equivariance: each primitive in
> isolation has equivariance error < 1e-5

Equivariance is checked AT INIT (no training). With Kaiming-init kernels
and default BN running stats, every primitive in this file is equivariant
by construction; the test asserts this against the regular-representation
group action on G-signals.

Each primitive is tested twice: once with the default ``GCHESS_GROUP``
(|G|=4), and once with ``SGCHESS_GROUP`` (|G|=2, only sigma_c). The
SGChess case exercises the same construction with a different group
descriptor — same primitives, fewer fibers, still strictly equivariant.
"""

import pytest
import torch

from models.gcnn import (
    GCHESS_GROUP,
    GROUP_SIZE,
    GBatchNorm,
    GConv,
    GLinear,
    GSELayer,
    LiftingConv,
    SGCHESS_GROUP,
    group_action_g_signal,
    regular_rep_perm,
)
from symmetry.actions import group_action_input
from symmetry.group import COMPOSITION_TABLE, ELEMENTS, GChess, SGChess


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
    layer = LiftingConv(in_channels=34, out_channels=4, kernel_size=3, padding=1)
    x = torch.randn(2, 34, 8, 8)
    y = layer(x)
    assert y.shape == (2, 4 * GROUP_SIZE, 8, 8)


def test_lifting_conv_rejects_non_chess_input():
    """LiftingConv is chess-specific: in_channels MUST be 34.

    Both the constructor and the forward pass reject any other count —
    the chess-aware kernel transformation (sigma_c channel permutation +
    plane-30 sign flip) only makes sense on the canonical chess plane
    layout, so silently accepting a different channel count would be a
    quietly-incorrect equivariance bug.
    """
    with pytest.raises(ValueError):
        LiftingConv(in_channels=5, out_channels=4, kernel_size=3, padding=1)
    with pytest.raises(ValueError):
        LiftingConv(in_channels=33, out_channels=4, kernel_size=3, padding=1)


@pytest.mark.parametrize("g", [GChess.SIGMA_H, GChess.SIGMA_C, GChess.SIGMA_HC])
def test_lifting_conv_equivariance(g):
    """Equivariance under the FULL chess input action.

    Input: 34-plane chess input (signed plane 30, etc.).
    Input action: `group_action_input(g, x)` — spatial flip + sigma_c
        channel permutation (us<->opp, castling swap) + sigma_c sign
        flip on plane 30. NOT just a spatial flip.
    Output action: `group_action_g_signal(g, ...)` — regular
        representation on the group axis + spatial flip on (H, W).

    LiftingConv's construction (start from one base kernel K, build the
    4 output group slices by applying each g's chess action to K, stack)
    makes this equivariance hold by construction — verified here over
    multiple random samples per g for broader coverage than a single
    fixed seed would give.
    """
    for sample in range(3):
        torch.manual_seed(int(g) * 1000 + sample)
        layer = LiftingConv(in_channels=34, out_channels=4, kernel_size=3, padding=1)
        layer.eval()

        x = _sample_chess_input(seed=sample)
        gx = group_action_input(g, x)  # FULL chess action

        with torch.no_grad():
            out_x = layer(x)
            out_gx = layer(gx)
            g_out_x = group_action_g_signal(g, out_x, num_logical_channels=4)

        diff = (out_gx - g_out_x).abs().max().item()
        assert diff < 1e-5, (
            f"LiftingConv {g.name} sample={sample}: max diff = {diff}"
        )


def test_lifting_conv_parameter_count():
    """LiftingConv(34, C_out, k) holds exactly C_out * 34 * k * k weights
    plus C_out bias params. No per-group-slice copies — all four output
    group slices are built at forward time from a single (C_out, 34, k, k)
    base kernel by applying the chess action to it."""
    C_in, C_out, k = 34, 7, 3
    layer = LiftingConv(in_channels=C_in, out_channels=C_out, kernel_size=k, padding=1, bias=True)
    actual = sum(p.numel() for p in layer.parameters())
    expected = C_out * C_in * k * k + C_out
    assert actual == expected, f"expected {expected} params, got {actual}"

    # bias=False drops the bias term.
    layer_nb = LiftingConv(in_channels=C_in, out_channels=C_out, kernel_size=k, padding=1, bias=False)
    assert sum(p.numel() for p in layer_nb.parameters()) == C_out * C_in * k * k


def test_lifting_conv_nonzero_bias():
    """LiftingConv's bias is per LOGICAL output channel, broadcast across
    (a) all |G| group copies of that channel and (b) all spatial positions.

    Verified in two parts:
      (a) zero-kernel pass: output is entirely the bias contribution, so
          every (g, r, f) entry of logical channel c must equal bias[c].
      (b) center-only-kernel pass at an interior position: manual
          computation of the expected output for each g matches the
          actual output. We use the center of a 3x3 kernel (which is
          fixed by every spatial flip), but the chess action ALSO
          permutes input channels and sign-flips plane 30 — so the per-g
          expected values are NOT all equal; they reflect the chess
          channel transformation applied to the kernel.
    """
    # --- (a) bias broadcast with a zero kernel ---
    layer = LiftingConv(in_channels=34, out_channels=3, kernel_size=3, padding=1, bias=True)
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.copy_(torch.tensor([10.0, 20.0, 30.0]))
    layer.eval()

    x = torch.randn(2, 34, 8, 8)
    with torch.no_grad():
        out = layer(x)
    assert out.shape == (2, 3 * GROUP_SIZE, 8, 8)
    bias_vals = layer.bias.detach().tolist()
    for c in range(3):
        expected_val = bias_vals[c]
        for g_idx in range(GROUP_SIZE):
            channel_slice = out[:, c * GROUP_SIZE + g_idx, :, :]
            assert torch.allclose(
                channel_slice,
                torch.full_like(channel_slice, expected_val),
                atol=1e-6,
            ), f"c={c}, g={g_idx}: expected bias={expected_val} broadcast"

    # --- (b) manual computation at an interior position ---
    # Set kernel entries at the spatial CENTER (1, 1) of the 3x3 only:
    #   K[c, 0, 1, 1]  = +1   (our pawns)
    #   K[c, 6, 1, 1]  = +2   (opp pawns)
    #   K[c, 30, 1, 1] = +3   (plane 30, side-to-move)
    # Center positions are spatially-flip-invariant, so the per-g kernels
    # differ ONLY by sigma_c's channel permutation + plane-30 sign flip.
    layer2 = LiftingConv(in_channels=34, out_channels=3, kernel_size=3, padding=1, bias=True)
    with torch.no_grad():
        layer2.weight.zero_()
        for c in range(3):
            layer2.weight[c, 0, 1, 1] = 1.0
            layer2.weight[c, 6, 1, 1] = 2.0
            layer2.weight[c, 30, 1, 1] = 3.0
        layer2.bias.copy_(torch.tensor([10.0, 20.0, 30.0]))
    layer2.eval()

    # Build a simple input: known values at channels 0, 6, 30 at position (3, 4).
    x2 = torch.zeros(1, 34, 8, 8)
    x2[0, 0, 3, 4] = 5.0     # our pawn signal
    x2[0, 6, 3, 4] = 7.0     # opp pawn signal
    x2[0, 30, :, :] = 1.0    # black-to-move (signed +1)
    with torch.no_grad():
        out2 = layer2(x2)

    bias2_vals = layer2.bias.detach().tolist()
    # At position (3, 4), each g's kernel for the center pulls from a
    # different input-channel slice (after channel perm + plane-30 sign
    # flip — center is spatially-flip-invariant). Specifically:
    #   - e and sigma_h: kernel is K (no channel perm, no sign), so the
    #     contribution is K[c, 0, 1, 1] * 5 + K[c, 6, 1, 1] * 7
    #                   + K[c, 30, 1, 1] * 1
    #     = 1*5 + 2*7 + 3*1 = 5 + 14 + 3 = 22.
    #   - sigma_c and sigma_hc: kernel has input channels permuted by
    #     SIGMA_C_CHANNEL_PERM (us<->opp, so K[c, 0] swaps with K[c, 6]),
    #     AND plane-30 sign-flipped (K[c, 30] -> -K[c, 30]). So the
    #     contribution becomes K[c, 6, 1, 1] * 5 + K[c, 0, 1, 1] * 7
    #                          + (-K[c, 30, 1, 1]) * 1
    #     = 2*5 + 1*7 + (-3)*1 = 10 + 7 - 3 = 14.
    expected_eh = 1 * 5.0 + 2 * 7.0 + 3 * 1.0           # = 22
    expected_chc = 2 * 5.0 + 1 * 7.0 + (-3) * 1.0       # = 14
    for c in range(3):
        for g_idx, expected_kernel_term in [
            (int(GChess.E), expected_eh),
            (int(GChess.SIGMA_H), expected_eh),
            (int(GChess.SIGMA_C), expected_chc),
            (int(GChess.SIGMA_HC), expected_chc),
        ]:
            actual = out2[0, c * GROUP_SIZE + g_idx, 3, 4].item()
            expected = expected_kernel_term + bias2_vals[c]
            assert abs(actual - expected) < 1e-5, (
                f"c={c}, g_idx={g_idx}: actual={actual}, expected={expected}"
            )


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
    """Run three independent random initializations + random inputs per g
    so we don't only validate one specific weight/input configuration.
    Seeds are deterministic per (g, sample) for reproducibility."""
    for sample in range(3):
        torch.manual_seed(int(g) * 1000 + sample)
        layer = GConv(in_channels=4, out_channels=6, kernel_size=3, padding=1)
        layer.eval()

        x = torch.randn(2, 4 * GROUP_SIZE, 8, 8)
        gx = group_action_g_signal(g, x, num_logical_channels=4)

        with torch.no_grad():
            out_x = layer(x)
            out_gx = layer(gx)
            g_out_x = group_action_g_signal(g, out_x, num_logical_channels=6)

        diff = (out_gx - g_out_x).abs().max().item()
        assert diff < 1e-5, (
            f"GConv equivariance failed for {g.name}, sample={sample}: "
            f"max diff = {diff}"
        )


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
# Helpers used by chess-input equivariance tests above.
# ---------------------------------------------------------------------------


def _sample_chess_input(batch=2, seed=0):
    """Sample a 34-plane chess-style input. Plane 30 in {-1, +1}."""
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(batch, 34, 8, 8)
    x[:, 0:12] = (torch.rand(batch, 12, 8, 8, generator=g) > 0.85).float()
    x[:, 12] = (torch.rand(batch, 8, 8, generator=g) > 0.9).float()
    x[:, 13:25] = (torch.rand(batch, 12, 8, 8, generator=g) > 0.85).float()
    x[:, 25] = (torch.rand(batch, 8, 8, generator=g) > 0.9).float()
    x[:, 26:30] = (torch.rand(batch, 4, 8, 8, generator=g) > 0.5).float()
    side = 2.0 * (torch.rand(batch, generator=g) > 0.5).float() - 1.0
    x[:, 30] = side.view(batch, 1, 1).expand(batch, 8, 8)
    x[:, 31] = torch.rand(batch, generator=g).view(batch, 1, 1).expand(batch, 8, 8)
    x[:, 32] = 0.0
    x[:, 33] = 1.0
    return x


# ---------------------------------------------------------------------------
# Stack: LiftingConv -> GBatchNorm -> ReLU -> GConv (mini-pipeline) is also
# equivariant. This is a smoke check that primitives compose correctly.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("g", [GChess.SIGMA_H, GChess.SIGMA_C, GChess.SIGMA_HC])
def test_mini_pipeline_equivariance(g):
    """LiftingConv -> GBatchNorm -> ReLU -> GConv composes correctly under
    the FULL chess input action (group_action_input on the 34-plane input,
    group_action_g_signal on the G-signal output)."""
    torch.manual_seed(0)
    lift = LiftingConv(in_channels=34, out_channels=5, kernel_size=3, padding=1)
    bn = GBatchNorm(num_logical_channels=5)
    gconv = GConv(in_channels=5, out_channels=4, kernel_size=3, padding=1)

    lift.eval(); bn.eval(); gconv.eval()

    def fwd(z):
        z = lift(z)
        z = bn(z)
        z = torch.nn.functional.relu(z)
        z = gconv(z)
        return z

    x = _sample_chess_input(batch=2, seed=0)
    gx = group_action_input(g, x)  # FULL chess action on the 34-plane input

    with torch.no_grad():
        out_x = fwd(x)
        out_gx = fwd(gx)
        g_out_x = group_action_g_signal(g, out_x, num_logical_channels=4)

    diff = (out_gx - g_out_x).abs().max().item()
    assert diff < 1e-5, f"mini-pipeline equivariance failed for {g.name}: max diff = {diff}"


# ===========================================================================
# SG_chess subgroup ({e, sigma_c}, |G|=2): parallel tests
# ===========================================================================
#
# Each primitive is rebuilt with ``group=SGCHESS_GROUP`` and tested for
# strict equivariance under the single non-identity element sigma_c
# (which has local index 1 in the SGChess enumeration). Shape tests use
# |G|=2 instead of GROUP_SIZE.


SGCHESS_ORDER = 2  # for shape arithmetic; mirrors symmetry/group.SGCHESS_ORDER


# ---------------------------------------------------------------------------
# regular_rep_perm + group_action_g_signal (SGChess)
# ---------------------------------------------------------------------------


def test_regular_rep_perm_sgchess_identity():
    perm = regular_rep_perm(SGChess.E, num_logical_channels=3, group=SGCHESS_GROUP)
    assert torch.equal(perm, torch.arange(3 * SGCHESS_ORDER, dtype=torch.long))


def test_regular_rep_perm_sgchess_sigma_c_swaps_fibers():
    """SGChess sigma_c (local index 1) swaps the 2 fibers of each logical channel."""
    C = 3
    perm = regular_rep_perm(SGChess.SIGMA_C, num_logical_channels=C, group=SGCHESS_GROUP)
    expected = torch.empty(C * SGCHESS_ORDER, dtype=torch.long)
    for c in range(C):
        expected[c * 2 + 0] = c * 2 + 1
        expected[c * 2 + 1] = c * 2 + 0
    assert torch.equal(perm, expected)


def test_regular_rep_perm_sgchess_sigma_c_self_inverse():
    perm = regular_rep_perm(SGChess.SIGMA_C, num_logical_channels=2, group=SGCHESS_GROUP)
    assert torch.equal(perm[perm], torch.arange(2 * SGCHESS_ORDER, dtype=torch.long))


def test_group_action_g_signal_sgchess_self_inverse():
    x = torch.randn(2, 3 * SGCHESS_ORDER, 8, 8)
    twice = group_action_g_signal(
        SGChess.SIGMA_C,
        group_action_g_signal(SGChess.SIGMA_C, x, 3, group=SGCHESS_GROUP),
        3,
        group=SGCHESS_GROUP,
    )
    assert torch.equal(twice, x)


# ---------------------------------------------------------------------------
# LiftingConv (SGChess)
# ---------------------------------------------------------------------------


def test_lifting_conv_sgchess_shape():
    """SGChess LiftingConv outputs |G|=2 fibers per logical channel."""
    layer = LiftingConv(
        in_channels=34, out_channels=4, kernel_size=3, padding=1, group=SGCHESS_GROUP
    )
    x = torch.randn(2, 34, 8, 8)
    y = layer(x)
    assert y.shape == (2, 4 * SGCHESS_ORDER, 8, 8)


def test_lifting_conv_sgchess_parameter_count():
    """SGChess LiftingConv learns the same base kernel as GChess
    LiftingConv: one (C_out, 34, k, k) kernel + (C_out,) bias. The fiber
    count is a function of forward-time kernel construction, not of
    learned parameters."""
    layer = LiftingConv(
        in_channels=34, out_channels=7, kernel_size=3, padding=1, bias=True,
        group=SGCHESS_GROUP,
    )
    actual = sum(p.numel() for p in layer.parameters())
    expected = 7 * 34 * 3 * 3 + 7
    assert actual == expected


def test_lifting_conv_sgchess_equivariance():
    """LiftingConv with group=SGCHESS_GROUP must be strictly equivariant
    under the FULL chess input action of sigma_c."""
    for sample in range(3):
        torch.manual_seed(2026 + sample)
        layer = LiftingConv(
            in_channels=34, out_channels=4, kernel_size=3, padding=1,
            group=SGCHESS_GROUP,
        )
        layer.eval()

        x = _sample_chess_input(seed=sample)
        gx = group_action_input(GChess.SIGMA_C, x)

        with torch.no_grad():
            out_x = layer(x)
            out_gx = layer(gx)
            g_out_x = group_action_g_signal(
                SGChess.SIGMA_C, out_x, num_logical_channels=4, group=SGCHESS_GROUP,
            )

        diff = (out_gx - g_out_x).abs().max().item()
        assert diff < 1e-5, f"SGChess LiftingConv sample={sample}: max diff = {diff}"


def test_lifting_conv_sgchess_not_equivariant_under_sigma_h():
    """Sanity check: SGChess LiftingConv is NOT designed to be equivariant
    under sigma_h (file flip), only under sigma_c. The forward output
    under sigma_h-transformed input does NOT match the natural action on
    the SGChess-fiber output — confirming we built the right subgroup
    and didn't accidentally inherit full-group equivariance."""
    torch.manual_seed(0)
    layer = LiftingConv(
        in_channels=34, out_channels=4, kernel_size=3, padding=1, group=SGCHESS_GROUP,
    )
    layer.eval()
    x = _sample_chess_input(seed=0)
    gx = group_action_input(GChess.SIGMA_H, x)

    with torch.no_grad():
        out_x = layer(x)
        out_gx = layer(gx)
        # The "natural" output action of sigma_h on a SGChess G-signal
        # is just the spatial file flip (since SGChess has no sigma_h
        # in its regular rep — sigma_h is not even a group element).
        out_x_flipped = torch.flip(out_x, dims=[-1])

    diff = (out_gx - out_x_flipped).abs().max().item()
    # On a non-trivial random init, the discrepancy should be O(1).
    # Treat anything > 1e-3 as "definitely not equivariant under sigma_h".
    assert diff > 1e-3, (
        f"SGChess LiftingConv unexpectedly looks sigma_h-equivariant: "
        f"max diff = {diff}"
    )


# ---------------------------------------------------------------------------
# GConv (SGChess)
# ---------------------------------------------------------------------------


def test_gconv_sgchess_shape():
    layer = GConv(in_channels=4, out_channels=6, kernel_size=3, padding=1, group=SGCHESS_GROUP)
    x = torch.randn(2, 4 * SGCHESS_ORDER, 8, 8)
    y = layer(x)
    assert y.shape == (2, 6 * SGCHESS_ORDER, 8, 8)


def test_gconv_sgchess_parameter_count_is_half_of_gchess():
    """SGChess uses |G|=2 base kernels; GChess uses |G|=4. With matched
    (C_in, C_out, k), SGChess GConv has 2/4 = half the weight params."""
    layer_g = GConv(in_channels=4, out_channels=6, kernel_size=3, padding=1)
    layer_s = GConv(
        in_channels=4, out_channels=6, kernel_size=3, padding=1, group=SGCHESS_GROUP,
    )
    n_g = layer_g.weight.numel()
    n_s = layer_s.weight.numel()
    assert n_s * 2 == n_g, f"expected SGChess GConv to have half the params; got {n_s} vs {n_g}"


def test_gconv_sgchess_equivariance():
    """GConv with group=SGCHESS_GROUP must be strictly equivariant under
    sigma_c (the only non-identity element of SGChess)."""
    for sample in range(3):
        torch.manual_seed(100 + sample)
        layer = GConv(
            in_channels=4, out_channels=6, kernel_size=3, padding=1, group=SGCHESS_GROUP,
        )
        layer.eval()

        x = torch.randn(2, 4 * SGCHESS_ORDER, 8, 8)
        gx = group_action_g_signal(
            SGChess.SIGMA_C, x, num_logical_channels=4, group=SGCHESS_GROUP,
        )

        with torch.no_grad():
            out_x = layer(x)
            out_gx = layer(gx)
            g_out_x = group_action_g_signal(
                SGChess.SIGMA_C, out_x, num_logical_channels=6, group=SGCHESS_GROUP,
            )

        diff = (out_gx - g_out_x).abs().max().item()
        assert diff < 1e-5, f"SGChess GConv sample={sample}: max diff = {diff}"


# ---------------------------------------------------------------------------
# GLinear / GBatchNorm / GSELayer (SGChess)
# ---------------------------------------------------------------------------


def test_glinear_sgchess_shape_and_equivariance():
    torch.manual_seed(0)
    layer = GLinear(in_channels=4, out_channels=6, group=SGCHESS_GROUP)
    layer.eval()

    x = torch.randn(2, 4 * SGCHESS_ORDER)
    in_perm = regular_rep_perm(SGChess.SIGMA_C, 4, group=SGCHESS_GROUP)
    out_perm = regular_rep_perm(SGChess.SIGMA_C, 6, group=SGCHESS_GROUP)
    gx = x.index_select(dim=1, index=in_perm)

    with torch.no_grad():
        out_x = layer(x)
        out_gx = layer(gx)
        g_out_x = out_x.index_select(dim=1, index=out_perm)

    assert out_x.shape == (2, 6 * SGCHESS_ORDER)
    diff = (out_gx - g_out_x).abs().max().item()
    assert diff < 1e-5, f"SGChess GLinear sigma_c: max diff = {diff}"


def test_gbatchnorm_sgchess_shape_and_equivariance():
    layer = GBatchNorm(num_logical_channels=4, group=SGCHESS_GROUP)
    layer.eval()
    x = torch.randn(2, 4 * SGCHESS_ORDER, 8, 8)
    gx = group_action_g_signal(
        SGChess.SIGMA_C, x, num_logical_channels=4, group=SGCHESS_GROUP,
    )
    with torch.no_grad():
        out_x = layer(x)
        out_gx = layer(gx)
        g_out_x = group_action_g_signal(
            SGChess.SIGMA_C, out_x, num_logical_channels=4, group=SGCHESS_GROUP,
        )
    assert out_x.shape == x.shape
    diff = (out_gx - g_out_x).abs().max().item()
    assert diff < 1e-5, f"SGChess GBatchNorm sigma_c: max diff = {diff}"


def test_gbatchnorm_sgchess_equivariance_after_training():
    """Like the GChess version: after SGD steps the BN running stats
    are non-trivial; equivariance must still hold because stats are
    shared across the |G|=2 fibers."""
    torch.manual_seed(1)
    layer = GBatchNorm(num_logical_channels=4, group=SGCHESS_GROUP)
    opt = torch.optim.SGD(layer.parameters(), lr=1e-2)

    layer.train()
    for _ in range(5):
        x_train = torch.randn(4, 4 * SGCHESS_ORDER, 8, 8)
        out = layer(x_train)
        loss = out.pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    layer.eval()
    x = torch.randn(2, 4 * SGCHESS_ORDER, 8, 8)
    gx = group_action_g_signal(
        SGChess.SIGMA_C, x, num_logical_channels=4, group=SGCHESS_GROUP,
    )
    with torch.no_grad():
        out_x = layer(x)
        out_gx = layer(gx)
        g_out_x = group_action_g_signal(
            SGChess.SIGMA_C, out_x, num_logical_channels=4, group=SGCHESS_GROUP,
        )
    diff = (out_gx - g_out_x).abs().max().item()
    assert diff < 1e-5, (
        f"SGChess GBatchNorm equivariance broke after training: max diff = {diff}"
    )


def test_gse_layer_sgchess_shape_and_equivariance():
    torch.manual_seed(0)
    layer = GSELayer(channels=4, se_channels=2, group=SGCHESS_GROUP)
    layer.eval()
    x = torch.randn(2, 4 * SGCHESS_ORDER, 8, 8)
    gx = group_action_g_signal(
        SGChess.SIGMA_C, x, num_logical_channels=4, group=SGCHESS_GROUP,
    )
    with torch.no_grad():
        out_x = layer(x)
        out_gx = layer(gx)
        g_out_x = group_action_g_signal(
            SGChess.SIGMA_C, out_x, num_logical_channels=4, group=SGCHESS_GROUP,
        )
    assert out_x.shape == x.shape
    diff = (out_gx - g_out_x).abs().max().item()
    assert diff < 1e-5, f"SGChess GSELayer sigma_c: max diff = {diff}"


# ---------------------------------------------------------------------------
# Mini-pipeline (SGChess)
# ---------------------------------------------------------------------------


def test_mini_pipeline_sgchess_equivariance():
    """LiftingConv -> GBatchNorm -> ReLU -> GConv (all with
    group=SGCHESS_GROUP) is strictly equivariant under sigma_c."""
    torch.manual_seed(0)
    lift = LiftingConv(
        in_channels=34, out_channels=5, kernel_size=3, padding=1, group=SGCHESS_GROUP,
    )
    bn = GBatchNorm(num_logical_channels=5, group=SGCHESS_GROUP)
    gconv = GConv(
        in_channels=5, out_channels=4, kernel_size=3, padding=1, group=SGCHESS_GROUP,
    )

    lift.eval(); bn.eval(); gconv.eval()

    def fwd(z):
        z = lift(z)
        z = bn(z)
        z = torch.nn.functional.relu(z)
        z = gconv(z)
        return z

    x = _sample_chess_input(batch=2, seed=0)
    gx = group_action_input(GChess.SIGMA_C, x)

    with torch.no_grad():
        out_x = fwd(x)
        out_gx = fwd(gx)
        g_out_x = group_action_g_signal(
            SGChess.SIGMA_C, out_x, num_logical_channels=4, group=SGCHESS_GROUP,
        )

    diff = (out_gx - g_out_x).abs().max().item()
    assert diff < 1e-5, (
        f"SGChess mini-pipeline sigma_c equivariance failed: max diff = {diff}"
    )
