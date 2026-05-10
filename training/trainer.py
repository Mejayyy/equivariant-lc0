"""Training loop for `BaselineLCZero` and `GCNNChess` on LCZero V6 data.

Usage (CLI)::

    python -m training.trainer \\
        --model {baseline,gcnn} \\
        --blocks 10 --filters 128 \\
        --batch-size 256 --lr 1e-3 \\
        --max-steps 10000 \\
        --device cuda \\
        --output-dir runs/my_run

Reads `DATA_SPLITS` from `data/config.py` for train / val folder lists.
Writes checkpoints (``ckpt_<step>.pt``) and a JSON-Lines metrics log
(``train_log.jsonl``) to ``--output-dir``. Stop with Ctrl-C; the most
recent checkpoint on disk is the resumption point (re-launch with
``--resume <ckpt_path>``).

Multi-worker dataloading: the IterableDataset doesn't shard across
workers by default — each worker would re-iterate the same files, giving
duplicate samples. ``--num-workers 0`` is the safe default; using a
positive value requires adding a ``worker_init_fn`` (TODO).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.config import DATA_SPLITS
from data.dataset import LCZeroV6Dataset
from models import BaselineLCZero, GCNNChess

from .losses import combined_loss


# ---------------------------------------------------------------------------
# Model + dataloader factories
# ---------------------------------------------------------------------------


def build_model(
    name: str,
    blocks: int,
    filters: int,
    se_channels: int = 32,
    mlh_channels: int = 8,
    fc_size: int = 128,
) -> nn.Module:
    if name == "baseline":
        return BaselineLCZero(
            blocks=blocks,
            filters=filters,
            se_channels=se_channels,
            mlh_channels=mlh_channels,
            fc_size=fc_size,
        )
    if name == "gcnn":
        return GCNNChess(
            blocks=blocks,
            filters=filters,
            se_channels=se_channels,
            mlh_channels=mlh_channels,
            fc_size=fc_size,
        )
    raise ValueError(f"unknown model: {name}; expected 'baseline' or 'gcnn'")


def make_loaders(
    batch_size: int,
    num_workers: int,
    shuffle_buffer: int,
    pin_memory: bool,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = LCZeroV6Dataset(
        folders=DATA_SPLITS["train"], shuffle_buffer=shuffle_buffer
    )
    val_ds = LCZeroV6Dataset(folders=DATA_SPLITS["val"], shuffle_buffer=0)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Train / eval helpers
# ---------------------------------------------------------------------------


def _to_device(planes, targets, device, non_blocking):
    planes = planes.to(device, non_blocking=non_blocking)
    targets = {
        k: v.to(device, non_blocking=non_blocking) for k, v in targets.items()
    }
    return planes, targets


def train_one_step(
    model: nn.Module,
    batch,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    grad_clip: Optional[float] = None,
) -> Dict[str, float]:
    """Forward + backward + step. Returns scalar losses."""
    model.train()
    planes, targets = _to_device(*batch, device, non_blocking=True)
    out = model(planes)
    losses = combined_loss(out, targets)
    optimizer.zero_grad(set_to_none=True)
    losses["total"].backward()
    if grad_clip is not None and grad_clip > 0:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return {k: v.item() for k, v in losses.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> Dict[str, float]:
    """Mean per-head losses over up to `max_batches` validation batches.
    Switches model to eval() and back to train() at exit."""
    was_training = model.training
    model.eval()
    sums = {"policy": 0.0, "value": 0.0, "mlh": 0.0, "total": 0.0}
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        planes, targets = _to_device(*batch, device, non_blocking=True)
        out = model(planes)
        losses = combined_loss(out, targets)
        for k in sums:
            sums[k] += losses[k].item()
        n += 1
    if was_training:
        model.train()
    if n == 0:
        return sums
    return {k: v / n for k, v in sums.items()}


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: Dict[str, Any] = None,
) -> None:
    """Save model + optimizer state + step + config dict to `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "config": config or {},
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> int:
    """Restore model (and optimizer, if provided) state from a checkpoint
    file. Returns the saved step counter."""
    ck = torch.load(path, map_location=map_location)
    model.load_state_dict(ck["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ck:
        optimizer.load_state_dict(ck["optimizer_state_dict"])
    return int(ck.get("step", 0))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def _next_batch(it: Iterator, loader: DataLoader) -> Tuple[Any, Iterator]:
    try:
        return next(it), it
    except StopIteration:
        it = iter(loader)
        return next(it), it


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    print(f"[init] device={device}")
    print(
        f"[init] model={args.model}  blocks={args.blocks}  "
        f"filters={args.filters}  se={args.se_channels}"
    )
    model = build_model(
        args.model,
        args.blocks,
        args.filters,
        args.se_channels,
        args.mlh_channels,
        args.fc_size,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[init] params: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    start_step = 0
    if args.resume:
        print(f"[init] resuming from {args.resume}")
        start_step = load_checkpoint(
            Path(args.resume), model, optimizer, map_location=str(device)
        )

    pin_memory = device.type == "cuda"
    train_loader, val_loader = make_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_buffer=args.shuffle_buffer,
        pin_memory=pin_memory,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    config_dict = vars(args).copy()
    config_dict["params"] = n_params
    (out_dir / "config.json").write_text(json.dumps(config_dict, indent=2))
    print(f"[init] output_dir={out_dir}  log={log_path.name}")

    log_file = open(log_path, "a")

    print(
        f"[train] starting at step {start_step + 1} -> {args.max_steps}  "
        f"(batch={args.batch_size}, lr={args.lr})"
    )
    t0 = time.time()
    train_iter = iter(train_loader)
    train_window = []

    for step in range(start_step + 1, args.max_steps + 1):
        batch, train_iter = _next_batch(train_iter, train_loader)
        losses = train_one_step(
            model, batch, device, optimizer, grad_clip=args.grad_clip
        )
        train_window.append(losses["total"])

        if step % args.log_every == 0:
            window = train_window[-args.log_every:]
            avg = sum(window) / len(window)
            elapsed = time.time() - t0
            samples = (step - start_step) * args.batch_size
            sps = samples / max(elapsed, 1e-6)
            print(
                f"[{step:>6d}/{args.max_steps}] train_loss={avg:.4f}  "
                f"p={losses['policy']:.3f} v={losses['value']:.3f} "
                f"m={losses['mlh']:.3f}  ({sps:.0f} samples/sec)"
            )
            log_file.write(
                json.dumps(
                    {
                        "step": step,
                        "phase": "train",
                        "loss_avg": avg,
                        "policy": losses["policy"],
                        "value": losses["value"],
                        "mlh": losses["mlh"],
                        "samples_per_sec": sps,
                    }
                )
                + "\n"
            )
            log_file.flush()

        if step % args.val_every == 0:
            val_losses = evaluate(
                model, val_loader, device, max_batches=args.val_batches
            )
            print(
                f"          val_loss={val_losses['total']:.4f}  "
                f"p={val_losses['policy']:.3f} v={val_losses['value']:.3f} "
                f"m={val_losses['mlh']:.3f}"
            )
            log_file.write(
                json.dumps({"step": step, "phase": "val", **val_losses}) + "\n"
            )
            log_file.flush()

        if step % args.checkpoint_every == 0 or step == args.max_steps:
            ck_path = out_dir / f"ckpt_{step}.pt"
            save_checkpoint(ck_path, model, optimizer, step, config_dict)
            print(f"          checkpoint -> {ck_path.name}")

    log_file.close()
    print("[train] done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train BaselineLCZero or GCNNChess on LCZero V6 data."
    )
    p.add_argument("--model", choices=["baseline", "gcnn"], required=True)
    p.add_argument("--blocks", type=int, default=10)
    p.add_argument("--filters", type=int, default=128)
    p.add_argument("--se-channels", type=int, default=32)
    p.add_argument("--mlh-channels", type=int, default=8)
    p.add_argument("--fc-size", type=int, default=128)

    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=5.0,
                   help="L2 grad-norm clip; 0 to disable")
    p.add_argument("--max-steps", type=int, default=10_000)

    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--val-every", type=int, default=1_000)
    p.add_argument("--val-batches", type=int, default=20)
    p.add_argument("--checkpoint-every", type=int, default=2_000)

    p.add_argument("--shuffle-buffer", type=int, default=8_192)
    p.add_argument("--num-workers", type=int, default=0,
                   help="WARNING: >0 requires worker-side file sharding "
                        "(not yet implemented); default 0 is safe.")

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    p.add_argument("--device", default=default_device)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--output-dir", default="runs/default")
    p.add_argument("--resume", default=None,
                   help="path to a ckpt_<step>.pt file to resume from")
    return p.parse_args(argv)


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
