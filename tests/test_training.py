"""Phase 10: losses + trainer DOD checks.

DOD (CLAUDE.md):
> One training batch runs on each of the three models without NaN;
> checkpoint save/load round-trip preserves outputs

We only have BaselineLCZero and GCNNChess (Phase 9 SeparableGCNNChess
is skipped per user direction), so the DOD is checked on those two.
"""

import math

import pytest
import torch

from training.losses import (
    combined_loss,
    mlh_l1_loss,
    policy_cross_entropy,
    value_wdl_loss,
)
from training.trainer import (
    build_model,
    load_checkpoint,
    save_checkpoint,
)


# ---------------------------------------------------------------------------
# Loss-function unit tests
# ---------------------------------------------------------------------------


def test_policy_cross_entropy_handles_minus_one_sentinel():
    """Illegal-move slots (target == -1) must contribute 0 to the loss."""
    logits = torch.zeros(2, 1858)
    target = torch.full((2, 1858), -1.0)
    target[:, 0] = 1.0  # only slot 0 is legal, with prob 1
    loss = policy_cross_entropy(logits, target)
    # Softmax over only legal slot 0 gives prob 1.0 -> CE = 0.
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_policy_cross_entropy_uniform_over_legal_gives_log_n():
    """Uniform logits over n legal slots and uniform-target give CE = log n."""
    n_legal = 10
    target = torch.full((1, 1858), -1.0)
    target[0, :n_legal] = 1.0 / n_legal
    logits = torch.zeros(1, 1858)
    loss = policy_cross_entropy(logits, target)
    assert loss.item() == pytest.approx(math.log(n_legal), abs=1e-5)


def test_policy_cross_entropy_no_nan_on_random():
    torch.manual_seed(0)
    logits = torch.randn(3, 1858)
    target = torch.full((3, 1858), -1.0)
    for i in range(3):
        legal = torch.randperm(1858)[:50]
        probs = torch.rand(50)
        probs = probs / probs.sum()
        target[i, legal] = probs
    loss = policy_cross_entropy(logits, target)
    assert torch.isfinite(loss)


def test_policy_cross_entropy_gradient_flows():
    torch.manual_seed(0)
    logits = torch.randn(2, 1858, requires_grad=True)
    target = torch.full((2, 1858), -1.0)
    target[:, :5] = 0.2
    loss = policy_cross_entropy(logits, target)
    loss.backward()
    # Gradient should be non-zero at legal slots and zero at illegal slots.
    grad = logits.grad
    assert (grad[:, :5].abs().sum()) > 0
    assert torch.allclose(grad[:, 5:], torch.zeros_like(grad[:, 5:]))


def test_value_wdl_loss_perfect_prediction_equals_entropy():
    target = torch.tensor([[0.6, 0.3, 0.1]])
    pred = target.clone()
    loss = value_wdl_loss(pred, target)
    expected = -(target * target.log()).sum().item()
    assert loss.item() == pytest.approx(expected, abs=1e-5)


def test_mlh_l1_loss_zero_at_match():
    pred = torch.tensor([[5.0], [10.0]])
    target = torch.tensor([5.0, 10.0])
    loss = mlh_l1_loss(pred, target)
    assert loss.item() == 0.0


def test_combined_loss_keys_and_finite():
    logits = torch.randn(2, 1858)
    value = torch.softmax(torch.randn(2, 3), dim=-1)
    mlh = torch.relu(torch.randn(2, 1))
    target = {
        "policy": torch.softmax(torch.randn(2, 1858), dim=-1),
        "value": torch.softmax(torch.randn(2, 3), dim=-1),
        "mlh": torch.tensor([5.0, 10.0]),
    }
    out = combined_loss((logits, value, mlh), target)
    assert set(out.keys()) == {"total", "policy", "value", "mlh"}
    for v in out.values():
        assert torch.isfinite(v)


def test_combined_loss_weights_apply():
    """A zero weight must zero that head's contribution to total."""
    logits = torch.randn(1, 1858)
    value = torch.softmax(torch.randn(1, 3), dim=-1)
    mlh = torch.relu(torch.randn(1, 1))
    target = {
        "policy": torch.softmax(torch.randn(1, 1858), dim=-1),
        "value": torch.softmax(torch.randn(1, 3), dim=-1),
        "mlh": torch.tensor([5.0]),
    }
    out_full = combined_loss((logits, value, mlh), target)
    out_no_mlh = combined_loss(
        (logits, value, mlh), target, weights={"policy": 1.0, "value": 1.0, "mlh": 0.0}
    )
    expected = out_full["policy"] + out_full["value"]  # default weights both 1.0
    assert out_no_mlh["total"].item() == pytest.approx(expected.item(), abs=1e-5)


# ---------------------------------------------------------------------------
# Trainer DOD: one training step doesn't NaN; checkpoint round-trip preserves output
# ---------------------------------------------------------------------------


def _make_random_batch(batch_size: int = 2, seed: int = 0):
    torch.manual_seed(seed)
    planes = torch.zeros(batch_size, 34, 8, 8)
    planes[:, 0:12] = (torch.rand(batch_size, 12, 8, 8) > 0.85).float()
    planes[:, 12] = (torch.rand(batch_size, 8, 8) > 0.9).float()
    planes[:, 13:25] = (torch.rand(batch_size, 12, 8, 8) > 0.85).float()
    planes[:, 25] = (torch.rand(batch_size, 8, 8) > 0.9).float()
    planes[:, 26:30] = (torch.rand(batch_size, 4, 8, 8) > 0.5).float()
    side = (torch.rand(batch_size) > 0.5).float()
    planes[:, 30] = side.view(batch_size, 1, 1).expand(batch_size, 8, 8)
    planes[:, 31] = torch.rand(batch_size, 1, 1).expand(batch_size, 8, 8)
    planes[:, 33] = 1.0

    target = {
        "policy": torch.full((batch_size, 1858), -1.0),
        "value": torch.softmax(torch.randn(batch_size, 3), dim=-1),
        "mlh": torch.full((batch_size,), 20.0),
    }
    legal = torch.randperm(1858)[:50]
    probs = torch.rand(50)
    probs = probs / probs.sum()
    target["policy"][:, legal] = probs
    return planes, target


@pytest.mark.parametrize("model_name", ["baseline", "gcnn"])
def test_one_training_step_does_not_nan(model_name):
    """Phase 10 DOD #1: a forward+backward+step on the model produces
    finite gradients and finite parameters afterwards."""
    torch.manual_seed(0)
    model = build_model(
        model_name,
        blocks=2,
        filters=16,
        se_channels=4,
        mlh_channels=4,
        fc_size=16,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    planes, targets = _make_random_batch(batch_size=2)
    out = model(planes)
    losses = combined_loss(out, targets)
    assert torch.isfinite(losses["total"])

    optimizer.zero_grad(set_to_none=True)
    losses["total"].backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN/inf grad in {name}"
    optimizer.step()
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), f"NaN/inf param in {name} after step"


@pytest.mark.parametrize("model_name", ["baseline", "gcnn"])
def test_checkpoint_round_trip_preserves_outputs(tmp_path, model_name):
    """Phase 10 DOD #2: save -> load reproduces the same forward output."""
    torch.manual_seed(0)
    model = build_model(model_name, blocks=2, filters=16, se_channels=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Train one step to populate non-trivial state (BN running stats,
    # AdamW moment buffers, etc.).
    planes, targets = _make_random_batch(batch_size=2)
    out = model(planes)
    losses = combined_loss(out, targets)
    optimizer.zero_grad(set_to_none=True)
    losses["total"].backward()
    optimizer.step()

    # Capture eval-mode output.
    model.eval()
    x = torch.zeros(2, 34, 8, 8)
    x[:, 33] = 1.0
    x[:, 30] = 0.5  # arbitrary
    with torch.no_grad():
        out_before = model(x)

    ck = tmp_path / "ck.pt"
    save_checkpoint(ck, model, optimizer, step=1)

    model2 = build_model(model_name, blocks=2, filters=16, se_channels=4)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    step = load_checkpoint(ck, model2, optimizer2)
    assert step == 1

    model2.eval()
    with torch.no_grad():
        out_after = model2(x)

    for a, b, name in zip(out_before, out_after, ("policy", "value", "mlh")):
        assert torch.allclose(a, b, atol=1e-6), (
            f"{model_name}/{name} round-trip diff = {(a - b).abs().max().item()}"
        )
