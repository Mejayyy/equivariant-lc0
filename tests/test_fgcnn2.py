"""FGCNN2: FlipGroup-equivariant model with the AZ-layout policy head.

fgcnn2 = ``GCNNChess(group=FLIPGROUP_GROUP, policy_head_style="az")`` —
identical to fgcnn except the policy head. Instead of the orbit-based
gather (group-sum + arbitrary cell re-housing, ~30/80 channels used),
``FlipGroupAZPolicyHead`` keeps the lc0 AZ 80-channel spatial layout
(cell = from-square, channel = move direction/type) and achieves exact
sigma_h-equivariance via the direction-mirror channel permutation
``m(c)`` plus a two-fiber readout ``z(c) = y[c, e] + y[m(c), h]``.

Coverage:
  - build_az_channel_mirror: permutation/involution structure +
    hand-checked direction pairs (queen, knight, promo, padding).
  - build_flipgroup_az_policy_gather: shape/range/fiber parity, AZ-cell
    consistency with the baseline gather, channel usage (73/80),
    gather-level equivariance hand-check against build_policy_permutation.
  - FlipGroupAZPolicyHead: shape, group validation, sigma_h equivariance,
    param parity with the orbit head (same conv stack).
  - GCNNChess(policy_head_style="az"): forward shapes, softmax/nonneg
    contracts, end-to-end sigma_h equivariance < 1e-5, NOT equivariant
    under sigma_c, param parity with fgcnn, style validation.
  - trainer: --model fgcnn2 wiring.

(The whole-model equivariance-after-SGD test for fgcnn2 lives in
tests/test_equivariance.py, parametrized alongside gcnn/sgcnn/fgcnn.)
"""

import pytest
import torch

from metrics.equivariance import equivariance_error, equivariance_error_flipgroup
from models import GCNNChess
from models.gcnn import FLIPGROUP_GROUP, GCHESS_GROUP, group_action_g_signal
from models.heads import EquivariantPolicyHead, FlipGroupAZPolicyHead
from models.policy_map import (
    POLICY_INDEX_SIZE,
    POLICY_NUM_CHANNELS,
    UCI_TO_POLICY_INDEX,
    build_az_channel_mirror,
    build_flipgroup_az_policy_gather,
    build_policy_gather_indices,
)
from symmetry.actions import build_policy_permutation, group_action_policy
from symmetry.group import FlipGroup, GChess
from training.trainer import MODEL_NAMES, build_model, parse_args


ORDER = 2  # |FlipGroup|
EQUIV_TOL = 1e-5
POLICY_PER_HEAD_TOL = 5e-5


# ---------------------------------------------------------------------------
# build_az_channel_mirror: structure + hand-checked direction pairs
# ---------------------------------------------------------------------------


def test_az_channel_mirror_is_involutive_permutation():
    m = build_az_channel_mirror()
    assert len(m) == POLICY_NUM_CHANNELS
    assert sorted(m) == list(range(POLICY_NUM_CHANNELS))  # permutation
    for c in range(POLICY_NUM_CHANNELS):
        assert m[m[c]] == c, f"not an involution at channel {c}"


def test_az_channel_mirror_queen_directions():
    """Queen planes: plane = dir*7 + (steps-1), dir order
    [N, NE, E, SE, S, SW, W, NW]. File mirror fixes N (0) and S (4),
    swaps NE(1)<->NW(7), E(2)<->W(6), SE(3)<->SW(5) — at EVERY distance."""
    m = build_az_channel_mirror()
    dir_mirror = {0: 0, 1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1}
    for d in range(8):
        for k in range(7):
            assert m[d * 7 + k] == dir_mirror[d] * 7 + k, (
                f"queen dir {d} distance {k+1}: m={m[d * 7 + k]}"
            )


def test_az_channel_mirror_knight_directions():
    """Knight planes 56..63 with vectors N(1,2), NE(2,1), E(2,-1),
    SE(1,-2), S(-1,-2), SW(-2,-1), W(-2,1), NW(-1,2). File mirror maps
    (dc, dr) -> (-dc, dr): N<->NW, NE<->W, E<->SW, SE<->S."""
    m = build_az_channel_mirror()
    pairs = {0: 7, 1: 6, 2: 5, 3: 4}  # dir index -> mirrored dir index
    for a, b in pairs.items():
        assert m[56 + a] == 56 + b
        assert m[56 + b] == 56 + a


def test_az_channel_mirror_promo_directions():
    """Promo planes 64..72: plane = 64 + dir*3 + piece, dirs [NW, N, NE],
    pieces [r, b, q]. File mirror swaps NW<->NE (same piece), fixes N."""
    m = build_az_channel_mirror()
    for piece in range(3):
        assert m[64 + 0 * 3 + piece] == 64 + 2 * 3 + piece  # NW <-> NE
        assert m[64 + 2 * 3 + piece] == 64 + 0 * 3 + piece
        assert m[64 + 1 * 3 + piece] == 64 + 1 * 3 + piece  # N fixed


def test_az_channel_mirror_padding_channels_fixed():
    m = build_az_channel_mirror()
    for c in range(73, POLICY_NUM_CHANNELS):
        assert m[c] == c


# ---------------------------------------------------------------------------
# build_flipgroup_az_policy_gather: structure
# ---------------------------------------------------------------------------


def test_az_gather_shape_dtype_range():
    g = build_flipgroup_az_policy_gather()
    assert g.shape == (POLICY_INDEX_SIZE, 2)
    assert g.dtype == torch.int64
    assert int(g.min()) >= 0
    assert int(g.max()) < POLICY_NUM_CHANNELS * ORDER * 64


def test_az_gather_fiber_parity_and_shared_square():
    """Column 0 reads the e-fiber (even folded channel), column 1 the
    h-fiber (odd folded channel); both columns read the SAME square."""
    g = build_flipgroup_az_policy_gather()
    folded_chan = g // 64
    assert ((folded_chan[:, 0] % ORDER) == 0).all(), "col 0 must be e-fiber"
    assert ((folded_chan[:, 1] % ORDER) == 1).all(), "col 1 must be h-fiber"
    assert (g[:, 0] % 64 == g[:, 1] % 64).all(), "columns must share the square"


def test_az_gather_column0_matches_baseline_az_cells():
    """Column 0 is exactly the baseline AZ gather re-indexed into the
    folded (80*2)-channel layout — the spatial semantics are identical
    to BaselinePolicyHead's."""
    g = build_flipgroup_az_policy_gather()
    az = build_policy_gather_indices()
    chan, sq = az // 64, az % 64
    expected = (chan * ORDER + 0) * 64 + sq
    assert torch.equal(g[:, 0], expected)


def test_az_gather_column1_uses_mirrored_channel():
    g = build_flipgroup_az_policy_gather()
    az = build_policy_gather_indices()
    m = build_az_channel_mirror()
    m_t = torch.tensor(m, dtype=torch.int64)
    chan, sq = az // 64, az % 64
    expected = (m_t[chan] * ORDER + 1) * 64 + sq
    assert torch.equal(g[:, 1], expected)


def test_az_gather_uses_73_channels():
    """All 73 real AZ channels are read (vs ~30 for the orbit gather);
    only the 7 padding channels are dead."""
    g = build_flipgroup_az_policy_gather()
    logical_used = torch.unique((g // 64) // ORDER)
    assert logical_used.numel() == 73
    assert int(logical_used.max()) == 72


def test_az_gather_hand_checked_e2e4():
    """e2e4: AZ flat cell 76 (chan 1 = queen N distance 2, sq 12). Its
    direction is N, which mirrors to itself, so:
        col 0 = (1*2 + 0)*64 + 12 = 140
        col 1 = (1*2 + 1)*64 + 12 = 204
    """
    g = build_flipgroup_az_policy_gather()
    i = UCI_TO_POLICY_INDEX["e2e4"]
    assert g[i, 0].item() == 140
    assert g[i, 1].item() == 204


def test_az_gather_hand_checked_g1f3():
    """g1f3: AZ flat cell 4038 (chan 63 = knight NW vector (-1, 2),
    sq 6). The mirrored knight direction is N (chan 56), so:
        col 0 = (63*2 + 0)*64 + 6 = 8070
        col 1 = (56*2 + 1)*64 + 6 = 7238
    """
    g = build_flipgroup_az_policy_gather()
    i = UCI_TO_POLICY_INDEX["g1f3"]
    assert g[i, 0].item() == 8070
    assert g[i, 1].item() == 7238


# ---------------------------------------------------------------------------
# Gather-level equivariance hand-check (no learned weights involved)
# ---------------------------------------------------------------------------


def test_az_gather_under_flipgroup_action_matches_policy_permutation():
    """Applying the FlipGroup sigma_h action (fiber swap + file flip) to
    a random folded (80*2, 8, 8) tensor and reading through the gather
    must permute the 1858 logits exactly by build_policy_permutation."""
    torch.manual_seed(0)
    z = torch.randn(1, POLICY_NUM_CHANNELS * ORDER, 8, 8)
    gather = build_flipgroup_az_policy_gather()

    def readout(t):
        flat = t.reshape(1, -1)
        picked = flat.index_select(1, gather.view(-1))
        return picked.view(1, POLICY_INDEX_SIZE, 2).sum(dim=2)

    base = readout(z)
    z_h = group_action_g_signal(
        FlipGroup.SIGMA_H, z,
        num_logical_channels=POLICY_NUM_CHANNELS, group=FLIPGROUP_GROUP,
    )
    natural = readout(z_h)
    canonical = group_action_policy(GChess.SIGMA_H, base)
    diff = (natural - canonical).abs().max().item()
    assert diff < EQUIV_TOL, f"AZ gather sigma_h disagreement: max diff = {diff}"


# ---------------------------------------------------------------------------
# FlipGroupAZPolicyHead
# ---------------------------------------------------------------------------


def test_az_policy_head_shape():
    F = 8
    head = FlipGroupAZPolicyHead(filters=F)
    x = torch.randn(2, F * ORDER, 8, 8)
    out = head(x)
    assert out.shape == (2, POLICY_INDEX_SIZE)
    assert out.dtype == torch.float32


def test_az_policy_head_rejects_non_flipgroup():
    with pytest.raises(ValueError):
        FlipGroupAZPolicyHead(filters=8, group=GCHESS_GROUP)


@pytest.mark.parametrize("seed", range(3))
def test_az_policy_head_equivariance(seed):
    """head(sigma_h . x) == build_policy_permutation(sigma_h) . head(x)."""
    torch.manual_seed(seed)
    F = 8
    head = FlipGroupAZPolicyHead(filters=F)
    head.eval()

    x = torch.randn(2, F * ORDER, 8, 8)
    gx = group_action_g_signal(
        FlipGroup.SIGMA_H, x, num_logical_channels=F, group=FLIPGROUP_GROUP,
    )
    with torch.no_grad():
        out_x = head(x)
        out_gx = head(gx)
        g_out_x = group_action_policy(GChess.SIGMA_H, out_x)

    diff = (out_gx - g_out_x).abs().max().item()
    assert diff < POLICY_PER_HEAD_TOL, (
        f"FlipGroupAZPolicyHead sigma_h: max diff = {diff}"
    )


def test_az_policy_head_param_parity_with_orbit_head():
    """The AZ head's conv stack is identical to the orbit head's — only
    the (non-trainable) gather buffer differs, so param counts match."""
    az = FlipGroupAZPolicyHead(filters=16)
    orbit = EquivariantPolicyHead(filters=16, group=FLIPGROUP_GROUP)
    n_az = sum(p.numel() for p in az.parameters())
    n_orbit = sum(p.numel() for p in orbit.parameters())
    assert n_az == n_orbit


# ---------------------------------------------------------------------------
# GCNNChess(policy_head_style="az") — the fgcnn2 model
# ---------------------------------------------------------------------------


def _fgcnn2(**overrides):
    kwargs = dict(
        blocks=2, filters=32, se_channels=8, mlh_channels=8, fc_size=64,
        group=FLIPGROUP_GROUP, policy_head_style="az",
    )
    kwargs.update(overrides)
    return GCNNChess(**kwargs)


def test_fgcnn2_uses_az_policy_head():
    model = _fgcnn2()
    assert isinstance(model.policy_head, FlipGroupAZPolicyHead)
    assert model.policy_head_style == "az"
    assert model.group is FLIPGROUP_GROUP


def test_fgcnn2_forward_shapes():
    model = _fgcnn2()
    x = torch.randn(2, 34, 8, 8)
    p, v, m = model(x)
    assert p.shape == (2, POLICY_INDEX_SIZE)
    assert v.shape == (2, 3)
    assert m.shape == (2, 1)


def test_fgcnn2_value_is_softmax_and_mlh_nonneg():
    from metrics.equivariance import sample_chess_input

    model = _fgcnn2()
    model.eval()
    x = sample_chess_input(batch_size=4)
    with torch.no_grad():
        _, v, m = model(x)
    assert torch.allclose(v.sum(dim=1), torch.ones(4), atol=1e-5)
    assert (v >= 0).all()
    assert (m >= 0).all()


def test_fgcnn2_equivariance_error_below_threshold():
    """Headline check: fgcnn2 end-to-end sigma_h equivariance < 1e-5."""
    torch.manual_seed(0)
    model = _fgcnn2()
    errors = equivariance_error_flipgroup(model, n_samples=8, device="cpu")
    assert errors["sigma_h"] < EQUIV_TOL, (
        f"fgcnn2 sigma_h equivariance error {errors['sigma_h']:.3e} >= "
        f"{EQUIV_TOL:.0e}"
    )
    assert errors["overall_mean"] < EQUIV_TOL


def test_fgcnn2_not_equivariant_under_sigma_c():
    """fgcnn2 is a FlipGroup model: sigma_h-equivariant only. The full
    GChess metric must show appreciable sigma_c / sigma_hc errors."""
    torch.manual_seed(0)
    model = _fgcnn2()
    errors = equivariance_error(model, n_samples=4, device="cpu")
    assert errors["sigma_h"] < EQUIV_TOL
    assert errors["sigma_c"] > 1e-3, (
        f"fgcnn2 looks accidentally sigma_c-equivariant: {errors['sigma_c']:.3e}"
    )
    assert errors["sigma_h_c"] > 1e-3


def test_fgcnn2_param_count_equals_fgcnn():
    """fgcnn2 differs from fgcnn only in the policy readout (a buffer),
    so the trainable parameter counts are identical."""
    kwargs = dict(blocks=2, filters=32, se_channels=8, mlh_channels=8, fc_size=64)
    fgcnn = GCNNChess(**kwargs, group=FLIPGROUP_GROUP)
    fgcnn2 = GCNNChess(**kwargs, group=FLIPGROUP_GROUP, policy_head_style="az")
    n1 = sum(p.numel() for p in fgcnn.parameters())
    n2 = sum(p.numel() for p in fgcnn2.parameters())
    assert n1 == n2


def test_gcnn_chess_rejects_unknown_policy_head_style():
    with pytest.raises(ValueError):
        GCNNChess(blocks=2, filters=16, se_channels=8,
                  group=FLIPGROUP_GROUP, policy_head_style="bogus")


def test_gcnn_chess_rejects_az_style_for_non_flipgroup():
    """The AZ head is only sigma_h-mirrorable; requesting it for a group
    containing sigma_c must fail loudly at construction."""
    with pytest.raises(ValueError):
        GCNNChess(blocks=2, filters=16, se_channels=8,
                  group=GCHESS_GROUP, policy_head_style="az")


def test_gcnn_chess_default_style_is_orbit():
    model = GCNNChess(blocks=2, filters=16, se_channels=8, group=FLIPGROUP_GROUP)
    assert isinstance(model.policy_head, EquivariantPolicyHead)
    assert model.policy_head_style == "orbit"


# ---------------------------------------------------------------------------
# Trainer wiring
# ---------------------------------------------------------------------------


def test_trainer_model_names_include_fgcnn2():
    assert "fgcnn2" in MODEL_NAMES


def test_trainer_cli_accepts_fgcnn2():
    args = parse_args([
        "--model", "fgcnn2",
        "--blocks", "2", "--filters", "16",
        "--max-steps", "1", "--device", "cpu",
    ])
    assert args.model == "fgcnn2"


def test_build_model_fgcnn2_returns_az_flipgroup_gcnn():
    model = build_model("fgcnn2", blocks=2, filters=16, se_channels=4,
                        mlh_channels=4, fc_size=16)
    assert isinstance(model, GCNNChess)
    assert model.group is FLIPGROUP_GROUP
    assert isinstance(model.policy_head, FlipGroupAZPolicyHead)
