"""Phase 8: end-to-end equivariance of GCNNChess.

DOD (CLAUDE.md):
> end-to-end equivariance_error(GCNNChess) < 1e-5 for all three
> non-identity elements

Build a tiny GCNNChess (blocks=2, filters=32) and assert
equivariance_error gives per-element errors well under 1e-5 across all
three heads — i.e. the model is equivariant by construction at random
init. Phase 11/12 will re-run the same metric AFTER training to verify
equivariance is preserved by SGD (it should be, given GBatchNorm shares
stats across the group fiber).
"""

import pytest
import torch

from metrics.equivariance import equivariance_error, sample_chess_input
from models import GCNNChess


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
# DOD: end-to-end equivariance error < 1e-5 for each non-identity g
# ---------------------------------------------------------------------------


def test_gcnn_chess_equivariance_error_below_threshold():
    """The headline Phase-8 DOD test."""
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
