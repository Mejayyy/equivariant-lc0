"""End-to-end equivariance of GCNNChess: equivariance_error < 1e-5 for
all three non-identity elements of G_chess.

Build a tiny GCNNChess (blocks=2, filters=32) and assert
equivariance_error gives per-element errors well under 1e-5 across all
three heads — i.e. the model is equivariant by construction at random
init. Training preserves this, because every tied parameter is a single
tensor and GBatchNorm shares its statistics across the group fiber;
test_whole_model_equivariance_survives_sgd checks it after real
optimiser steps.

Also covers the SGChess-equivariant variant (``GCNNChess(group=
SGCHESS_GROUP)``): the same model architecture parameterized by the
SG_chess subgroup. Equivariance is required ONLY under sigma_c (the
only non-identity element of SG_chess); sigma_h and sigma_hc are
allowed to be non-equivariant (and indeed are, since the body has
just 2 group fibers).
"""

import pytest
import torch

from metrics.equivariance import (
    equivariance_error,
    equivariance_error_flipgroup,
    equivariance_error_sgchess,
    sample_chess_input,
)
from models import FLIPGROUP_GROUP, GCHESS_GROUP, GCNNChess, SGCHESS_GROUP
from training.losses import combined_loss


EQUIV_TOL = 1e-5


# ---------------------------------------------------------------------------
# sample_chess_input sanity (used by the metric)
# ---------------------------------------------------------------------------


def test_sample_chess_input_shape_and_dtype():
    x = sample_chess_input(batch_size=4)
    assert x.shape == (4, 34, 8, 8)
    assert x.dtype == torch.float32


def test_sample_chess_input_constant_planes():
    x = sample_chess_input(batch_size=2)
    assert (x[:, 32] == 0.0).all()
    assert (x[:, 33] == 1.0).all()


def test_sample_chess_input_side_to_move_is_signed():
    """Plane 30 is signed: exactly -1 (white) or +1 (black).

    sigma_c on plane 30 is now a pure sign flip ``-x[30]``; bit-exact
    self-inversion ``-(-x) == x`` holds in IEEE-754, so equivariance
    tests stay clean.
    """
    x = sample_chess_input(batch_size=16)
    side = x[:, 30]
    unique = torch.unique(side)
    assert set(unique.tolist()).issubset({-1.0, 1.0}), (
        f"plane 30 has unexpected values: {unique.tolist()}"
    )


# ---------------------------------------------------------------------------
# Forward shape (defines the contract: same as BaselineLCZero)
# ---------------------------------------------------------------------------


def test_gcnn_chess_forward_shapes():
    model = GCNNChess(blocks=2, filters=32, se_channels=8, mlh_channels=8, fc_size=64)
    x = torch.randn(2, 34, 8, 8)
    p, v, m = model(x)
    assert p.shape == (2, 1858)
    assert v.shape == (2, 3)
    assert m.shape == (2, 1)


def test_gcnn_chess_value_is_softmax():
    model = GCNNChess(blocks=2, filters=32, se_channels=8, mlh_channels=8, fc_size=64)
    model.eval()
    x = sample_chess_input(batch_size=4)
    with torch.no_grad():
        _, v, _ = model(x)
    assert torch.allclose(v.sum(dim=1), torch.ones(4), atol=1e-5)
    assert (v >= 0).all()


def test_gcnn_chess_mlh_nonneg():
    model = GCNNChess(blocks=2, filters=32, se_channels=8, mlh_channels=8, fc_size=64)
    model.eval()
    x = sample_chess_input(batch_size=4)
    with torch.no_grad():
        _, _, m = model(x)
    assert (m >= 0).all()


def test_gcnn_chess_rejects_wrong_input_shape():
    model = GCNNChess(blocks=2, filters=32, se_channels=8, mlh_channels=8, fc_size=64)
    bad = torch.randn(2, 33, 8, 8)
    with pytest.raises(ValueError):
        model(bad)


# ---------------------------------------------------------------------------
# End-to-end equivariance error < 1e-5 for each non-identity g
# ---------------------------------------------------------------------------


def test_gcnn_chess_equivariance_error_below_threshold():
    """Headline check: the full model is G_chess-equivariant at init."""
    torch.manual_seed(0)
    model = GCNNChess(blocks=2, filters=32, se_channels=8, mlh_channels=8, fc_size=64)
    errors = equivariance_error(model, n_samples=8, device="cpu")

    for key in ("sigma_h", "sigma_c", "sigma_h_c"):
        assert errors[key] < EQUIV_TOL, (
            f"{key} equivariance error {errors[key]:.3e} >= {EQUIV_TOL:.0e}"
        )
    assert errors["overall_mean"] < EQUIV_TOL


def test_equivariance_error_returns_expected_keys():
    model = GCNNChess(blocks=2, filters=32, se_channels=8, mlh_channels=8, fc_size=64)
    errors = equivariance_error(model, n_samples=2)
    assert set(errors.keys()) == {"sigma_h", "sigma_c", "sigma_h_c", "overall_mean"}
    for v in errors.values():
        assert isinstance(v, float)


# ---------------------------------------------------------------------------
# SG_chess subgroup: end-to-end equivariance under sigma_c only
# ---------------------------------------------------------------------------


def test_sgcnn_chess_forward_shapes():
    """SGChess-parameterized GCNNChess has the same forward interface
    (B, 34, 8, 8) -> (policy, value, mlh) of shapes (B, 1858), (B, 3),
    (B, 1). Body activations have |G|=2 fibers instead of 4 internally."""
    model = GCNNChess(
        blocks=2, filters=32, se_channels=8, mlh_channels=8, fc_size=64,
        group=SGCHESS_GROUP,
    )
    x = torch.randn(2, 34, 8, 8)
    p, v, m = model(x)
    assert p.shape == (2, 1858)
    assert v.shape == (2, 3)
    assert m.shape == (2, 1)


def test_sgcnn_chess_value_is_softmax():
    model = GCNNChess(
        blocks=2, filters=32, se_channels=8, mlh_channels=8, fc_size=64,
        group=SGCHESS_GROUP,
    )
    model.eval()
    x = sample_chess_input(batch_size=4)
    with torch.no_grad():
        _, v, _ = model(x)
    assert torch.allclose(v.sum(dim=1), torch.ones(4), atol=1e-5)
    assert (v >= 0).all()


def test_sgcnn_chess_mlh_nonneg():
    model = GCNNChess(
        blocks=2, filters=32, se_channels=8, mlh_channels=8, fc_size=64,
        group=SGCHESS_GROUP,
    )
    model.eval()
    x = sample_chess_input(batch_size=4)
    with torch.no_grad():
        _, _, m = model(x)
    assert (m >= 0).all()


def test_sgcnn_chess_equivariance_error_below_threshold():
    """SGChess: sigma_c equivariance error < 1e-5 at random init."""
    torch.manual_seed(0)
    model = GCNNChess(
        blocks=2, filters=32, se_channels=8, mlh_channels=8, fc_size=64,
        group=SGCHESS_GROUP,
    )
    errors = equivariance_error_sgchess(model, n_samples=8, device="cpu")
    assert errors["sigma_c"] < EQUIV_TOL, (
        f"SGChess sigma_c equivariance error {errors['sigma_c']:.3e} >= "
        f"{EQUIV_TOL:.0e}"
    )
    assert errors["overall_mean"] < EQUIV_TOL


def test_equivariance_error_sgchess_returns_expected_keys():
    model = GCNNChess(
        blocks=2, filters=32, se_channels=8, mlh_channels=8, fc_size=64,
        group=SGCHESS_GROUP,
    )
    errors = equivariance_error_sgchess(model, n_samples=2)
    assert set(errors.keys()) == {"sigma_c", "overall_mean"}
    for v in errors.values():
        assert isinstance(v, float)


def test_sgcnn_chess_has_fewer_params_than_gcnn_chess():
    """At matched (blocks, filters), SGChess GCNNChess has fewer
    parameters than the full GChess GCNNChess — the body's GConv
    base-kernel count drops from |G|=4 to |G|=2, roughly halving body
    weights."""
    kwargs = dict(blocks=2, filters=32, se_channels=8, mlh_channels=8, fc_size=64)
    gcnn = GCNNChess(**kwargs, group=GCHESS_GROUP)
    sgcnn = GCNNChess(**kwargs, group=SGCHESS_GROUP)
    n_g = sum(p.numel() for p in gcnn.parameters())
    n_s = sum(p.numel() for p in sgcnn.parameters())
    assert n_s < n_g, (
        f"SGChess GCNNChess should have fewer params; got "
        f"sgcnn={n_s} >= gcnn={n_g}"
    )


def test_sgcnn_chess_not_equivariant_under_sigma_h():
    """Confirms the model is truly an SGChess (Z/2)-equivariant model,
    not a full G_chess one. Equivariance error under sigma_h should be
    appreciable (>> 1e-5)."""
    torch.manual_seed(0)
    model = GCNNChess(
        blocks=2, filters=32, se_channels=8, mlh_channels=8, fc_size=64,
        group=SGCHESS_GROUP,
    )
    # Reuse the GChess error computation: it tests sigma_h, sigma_c, sigma_hc.
    errors = equivariance_error(model, n_samples=4, device="cpu")
    # sigma_c must still be tiny (SGChess guarantees it).
    assert errors["sigma_c"] < EQUIV_TOL, (
        f"SGChess model unexpectedly not sigma_c-equivariant: "
        f"{errors['sigma_c']:.3e}"
    )
    # sigma_h and sigma_hc should be substantially non-zero.
    assert errors["sigma_h"] > 1e-3, (
        f"SGChess model looks accidentally sigma_h-equivariant: "
        f"{errors['sigma_h']:.3e}"
    )
    assert errors["sigma_h_c"] > 1e-3, (
        f"SGChess model looks accidentally sigma_hc-equivariant: "
        f"{errors['sigma_h_c']:.3e}"
    )


# ---------------------------------------------------------------------------
# Whole-model equivariance AFTER SGD steps (all three groups)
# ---------------------------------------------------------------------------
#
# The init-time tests above cannot catch the training-time equivariance
# hazards: parameters that must stay TIED during optimization (the value
# head's shared wl_linear, GBatchNorm's per-logical-channel stats shared
# across group fibers, the single LiftingConv/GConv base kernels). A
# refactor that replaced any of those with per-fiber copies initialized
# equal would pass every init-time test and silently lose equivariance
# after the first optimizer step. So: train the full model for a few
# AdamW steps with the REAL combined loss (all three heads receive
# gradients, BN running stats move off their defaults), then re-assert
# the group-appropriate equivariance error is still < 1e-5.


def _random_targets(batch_size: int, seed: int) -> dict:
    """Random supervised targets in the exact format the trainer feeds
    combined_loss: policy with -1 illegal sentinels + a random legal
    distribution, a WDL simplex, and a non-negative plies-left scalar."""
    g = torch.Generator().manual_seed(seed)
    policy = torch.full((batch_size, 1858), -1.0)
    for i in range(batch_size):
        legal = torch.randperm(1858, generator=g)[:40]
        probs = torch.rand(40, generator=g)
        policy[i, legal] = probs / probs.sum()
    value = torch.softmax(torch.randn(batch_size, 3, generator=g), dim=-1)
    mlh = torch.rand(batch_size, generator=g) * 100.0
    return {"policy": policy, "value": value, "mlh": mlh}


@pytest.mark.parametrize(
    "group, metric, error_keys",
    [
        pytest.param(
            GCHESS_GROUP, equivariance_error,
            ("sigma_h", "sigma_c", "sigma_h_c"), id="gcnn",
        ),
        pytest.param(
            SGCHESS_GROUP, equivariance_error_sgchess,
            ("sigma_c",), id="sgcnn",
        ),
        pytest.param(
            FLIPGROUP_GROUP, equivariance_error_flipgroup,
            ("sigma_h",), id="fgcnn",
        ),
    ],
)
def test_whole_model_equivariance_survives_sgd(group, metric, error_keys):
    """Equivariance must hold (< 1e-5) not only at init but also after
    training. Five AdamW steps with the trainer's combined_loss
    (weight decay on, BN in train mode so running stats update), then
    the model must still be equivariant under its own group."""
    torch.manual_seed(0)
    model = GCNNChess(
        blocks=2, filters=16, se_channels=8, mlh_channels=8, fc_size=32,
        group=group,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    model.train()
    for step in range(5):
        x = sample_chess_input(batch_size=4)
        targets = _random_targets(batch_size=4, seed=step)
        losses = combined_loss(model(x), targets)
        assert torch.isfinite(losses["total"]), f"loss NaN at step {step}"
        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        opt.step()

    # Every head must have actually received gradient during the steps
    # (otherwise this test silently stops guarding that head).
    for head_name in ("policy_head", "value_head", "moves_left_head"):
        head = getattr(model, head_name)
        got_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in head.parameters()
        )
        assert got_grad, f"{head_name} received no gradient — test is vacuous"

    errors = metric(model, n_samples=8, device="cpu")
    for key in error_keys:
        assert errors[key] < EQUIV_TOL, (
            f"{group.name}: equivariance broke after SGD — "
            f"{key} error {errors[key]:.3e} >= {EQUIV_TOL:.0e}"
        )
