"""BaselineLCZero: shape tests + BN-fold round-trip (folded ~= unfolded
output).

Tiny demo config (blocks=2, filters=32) keeps these tests fast.
"""

import pytest
import torch

from models.baseline import BaselineBody, BaselineLCZero, ResBlock, SELayer
from models.heads import BaselineMovesLeftHead, BaselinePolicyHead, BaselineValueHead


# ---------------------------------------------------------------------------
# Sub-component shape tests
# ---------------------------------------------------------------------------


def test_se_layer_shape():
    se = SELayer(channels=32, se_channels=8)
    x = torch.randn(2, 32, 8, 8)
    y = se(x)
    assert y.shape == x.shape
    assert y.dtype == torch.float32


def test_resblock_shape():
    block = ResBlock(channels=32, se_channels=8)
    x = torch.randn(2, 32, 8, 8)
    y = block(x)
    assert y.shape == (2, 32, 8, 8)


def test_body_shape():
    body = BaselineBody(blocks=2, filters=32, se_channels=8)
    x = torch.randn(2, 34, 8, 8)
    y = body(x)
    assert y.shape == (2, 32, 8, 8)


def test_policy_head_shape():
    head = BaselinePolicyHead(filters=32)
    h = torch.randn(2, 32, 8, 8)
    p = head(h)
    assert p.shape == (2, 1858)
    assert p.dtype == torch.float32


def test_value_head_shape_and_simplex():
    head = BaselineValueHead(filters=32)
    h = torch.randn(2, 32, 8, 8)
    v = head(h)
    assert v.shape == (2, 3)
    assert torch.allclose(v.sum(dim=1), torch.ones(2), atol=1e-5)
    assert (v >= 0).all() and (v <= 1).all()


def test_mlh_head_shape_and_nonneg():
    head = BaselineMovesLeftHead(filters=32, mlh_channels=8, fc_size=64)
    h = torch.randn(2, 32, 8, 8)
    m = head(h)
    assert m.shape == (2, 1)
    # Final ReLU ensures non-negative output.
    assert (m >= 0).all()


# ---------------------------------------------------------------------------
# Full-model interface
# ---------------------------------------------------------------------------


def _demo_model() -> BaselineLCZero:
    return BaselineLCZero(blocks=2, filters=32, se_channels=8, mlh_channels=8, fc_size=64)


def test_baseline_forward_shapes():
    model = _demo_model()
    x = torch.randn(2, 34, 8, 8)
    p, v, m = model(x)
    assert p.shape == (2, 1858)
    assert v.shape == (2, 3)
    assert m.shape == (2, 1)


def test_baseline_value_is_softmax():
    model = _demo_model()
    model.eval()
    x = torch.randn(4, 34, 8, 8)
    with torch.no_grad():
        _, v, _ = model(x)
    assert torch.allclose(v.sum(dim=1), torch.ones(4), atol=1e-5)
    assert (v >= 0).all()


def test_baseline_mlh_nonneg():
    model = _demo_model()
    model.eval()
    x = torch.randn(4, 34, 8, 8)
    with torch.no_grad():
        _, _, m = model(x)
    assert (m >= 0).all()


def test_baseline_rejects_wrong_input_shape():
    model = _demo_model()
    bad = torch.randn(2, 33, 8, 8)
    with pytest.raises(ValueError):
        model(bad)


def test_baseline_policy_uses_canonical_gather():
    """The policy head's policy_gather buffer is the canonical 1858-element
    tensor produced by build_policy_gather_indices()."""
    from models.policy_map import build_policy_gather_indices

    model = _demo_model()
    expected = build_policy_gather_indices()
    assert torch.equal(model.policy_head.policy_gather, expected)


# ---------------------------------------------------------------------------
# BN-fold round-trip
# ---------------------------------------------------------------------------


def _train_a_few_steps(model: BaselineLCZero, n_steps: int = 3) -> None:
    """Run a few forward+backward passes to populate BN running stats with
    non-trivial values. The fold is mathematically equivalent only when
    BN running stats != defaults (running_mean=0, running_var=1) AS IT
    IS for both before- and after-fold eval — the test would still pass
    with default stats, but exercising real stats catches subtle bugs."""
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(n_steps):
        x = torch.randn(4, 34, 8, 8)
        p, v, m = model(x)
        loss = p.sum() + v.sum() + m.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()


def test_baseline_fold_bn_round_trip():
    """model.eval() output before fold == output after fold,
    to within ~1e-5 (no approximation; just float-precision drift)."""
    torch.manual_seed(0)
    model = _demo_model()
    _train_a_few_steps(model, n_steps=3)

    model.eval()
    x = torch.randn(2, 34, 8, 8)

    with torch.no_grad():
        p1, v1, m1 = model(x)

    model.fold_bn()

    with torch.no_grad():
        p2, v2, m2 = model(x)

    assert torch.allclose(p1, p2, atol=1e-5), (
        f"policy mismatch: max diff = {(p1 - p2).abs().max().item()}"
    )
    assert torch.allclose(v1, v2, atol=1e-5), (
        f"value mismatch: max diff = {(v1 - v2).abs().max().item()}"
    )
    assert torch.allclose(m1, m2, atol=1e-5), (
        f"mlh mismatch: max diff = {(m1 - m2).abs().max().item()}"
    )


def test_fold_bn_replaces_bn_with_identity():
    """After fold_bn the model should contain NO BatchNorm2d modules."""
    model = _demo_model()
    model.eval()
    model.fold_bn()
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert bns == [], f"BN modules still present after fold: {len(bns)}"


def test_fold_bn_rejects_train_mode():
    model = _demo_model()
    model.train()
    with pytest.raises(RuntimeError):
        model.fold_bn()
