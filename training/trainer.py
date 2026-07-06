"""Training loop for `BaselineLCZero` and the group-equivariant
`GCNNChess` variants on LCZero V6 data.

Usage (CLI)::

    python -m training.trainer \\
        --model {baseline,gcnn,sgcnn,fgcnn,fgcnn2} \\
        --blocks 10 --filters 128 \\
        --batch-size 256 --lr 1e-3 \\
        --max-steps 10000 \\
        --device cuda \\
        --output-dir runs/my_run

Model variants:

- ``baseline`` — ``BaselineLCZero`` (no equivariance; faithful LCZero
  SE-residual re-implementation).
- ``gcnn``     — ``GCNNChess(group=GCHESS_GROUP)``: full G_chess =
  {e, sigma_h, sigma_c, sigma_hc} equivariance. |G| = 4.
- ``sgcnn``    — ``GCNNChess(group=SGCHESS_GROUP)``: SG_chess =
  {e, sigma_c} subgroup equivariance only. |G| = 2; about half the
  body parameters of ``gcnn`` at matched (blocks, filters).
- ``fgcnn``    — ``GCNNChess(group=FLIPGROUP_GROUP)``: FlipGroup =
  {e, sigma_h} subgroup (file flip) equivariance only. |G| = 2.
- ``fgcnn2``   — ``fgcnn`` with the AZ-layout policy head
  (``policy_head_style="az"``); identical parameter count.

Pools all `.gz` games under `--data-root` (default `RAW_DATA_ROOT`) and
splits them disjointly into train/val at the game level (see
`data/splits.py`); draws a fixed budget of FILTERED plies per split.
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

from data.config import (
    DATA_SEED,
    RAW_DATA_ROOT,
    SPLIT_SEED,
    TRAIN_SAMPLES,
    VAL_SAMPLES,
)
from data.dataset import LCZeroV6Dataset
from data.splits import build_pooled_splits
from models import (
    BaselineLCZero,
    FLIPGROUP_GROUP,
    GCHESS_GROUP,
    GCNNChess,
    SGCHESS_GROUP,
)

from .losses import combined_loss


MODEL_NAMES = ("baseline", "gcnn", "sgcnn", "fgcnn", "fgcnn2")


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
    """Build one of: ``baseline``, ``gcnn`` (full G_chess), ``sgcnn``
    (SG_chess subgroup, sigma_c only), ``fgcnn`` (FlipGroup subgroup,
    sigma_h only, orbit-gather policy head), ``fgcnn2`` (FlipGroup like
    fgcnn but with the AZ-layout sigma_h-equivariant policy head —
    same body/value/mlh heads, only the policy readout differs)."""
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
            group=GCHESS_GROUP,
        )
    if name == "sgcnn":
        return GCNNChess(
            blocks=blocks,
            filters=filters,
            se_channels=se_channels,
            mlh_channels=mlh_channels,
            fc_size=fc_size,
            group=SGCHESS_GROUP,
        )
    if name == "fgcnn":
        return GCNNChess(
            blocks=blocks,
            filters=filters,
            se_channels=se_channels,
            mlh_channels=mlh_channels,
            fc_size=fc_size,
            group=FLIPGROUP_GROUP,
        )
    if name == "fgcnn2":
        return GCNNChess(
            blocks=blocks,
            filters=filters,
            se_channels=se_channels,
            mlh_channels=mlh_channels,
            fc_size=fc_size,
            group=FLIPGROUP_GROUP,
            policy_head_style="az",
        )
    raise ValueError(
        f"unknown model: {name}; expected one of {MODEL_NAMES}"
    )


def make_loaders(
    batch_size: int,
    num_workers: int,
    shuffle_buffer: int,
    pin_memory: bool,
    turn_start: int = 0,
    max_plies_left: Optional[float] = None,
    data_root: str = RAW_DATA_ROOT,
    train_samples: int = TRAIN_SAMPLES,
    val_samples: int = VAL_SAMPLES,
    split_seed: int = SPLIT_SEED,
    data_seed: int = DATA_SEED,
) -> Tuple[DataLoader, DataLoader]:
    """Build train/val loaders from a POOLED, game-level random split over
    all games under ``data_root``.

    - Games (whole `.gz` files) are pooled across every sub-folder and
      partitioned into disjoint train/val sets with ``split_seed`` (no
      per-game leakage; partition is filter-independent so the val games are
      stable across turn_start / max_plies_left).
    - Each loader draws a fixed budget of *filtered* plies — ``train_samples``
      / ``val_samples`` — via the dataset's ``max_samples`` cap, so the counts
      are constant regardless of the game-phase filters (given enough data).
    - The game-phase filters are applied to BOTH splits so val loss reflects
      the same game slice as training.
    - Training shuffles plies (shuffle_buffer) for SGD decorrelation; each
      sample is self-contained (current + 1-ply history), so this does not
      affect the history. Validation iterates deterministically.
    """
    train_files, val_files = build_pooled_splits(
        data_root, train_target=train_samples, val_target=val_samples,
        seed=split_seed,
    )
    train_ds = LCZeroV6Dataset(
        files=train_files,
        shuffle_buffer=shuffle_buffer,
        seed=data_seed,
        turn_start=turn_start,
        max_plies_left=max_plies_left,
        max_samples=train_samples,
    )
    val_ds = LCZeroV6Dataset(
        files=val_files,
        shuffle_buffer=0,
        seed=data_seed,
        turn_start=turn_start,
        max_plies_left=max_plies_left,
        max_samples=val_samples,
    )
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


def _total_grad_norm(parameters, norm_type: float = 2.0) -> float:
    """L2 norm of the gradients across ``parameters`` (PRE-clip), as a
    plain Python float. Mirrors what ``torch.nn.utils.clip_grad_norm_``
    returns internally but works when grad-clip is disabled."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    device = grads[0].device
    total = torch.norm(
        torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]),
        norm_type,
    )
    return float(total.item())


def train_one_step(
    model: nn.Module,
    batch,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    grad_clip: Optional[float] = None,
) -> Dict[str, float]:
    """Forward + backward + step. Returns scalar losses AND the L2
    gradient norm captured PRE-clip (so the logged value reflects the
    raw gradient magnitude, not what the optimizer actually saw)."""
    model.train()
    planes, targets = _to_device(*batch, device, non_blocking=True)
    out = model(planes)
    losses = combined_loss(out, targets)
    optimizer.zero_grad(set_to_none=True)
    losses["total"].backward()
    if grad_clip is not None and grad_clip > 0:
        # clip_grad_norm_ returns the pre-clip total norm as a Tensor.
        gn = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        grad_norm = float(gn.item() if isinstance(gn, torch.Tensor) else gn)
    else:
        grad_norm = _total_grad_norm(model.parameters())
    optimizer.step()
    out = {k: v.item() for k, v in losses.items()}
    out["grad_norm"] = grad_norm
    return out


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
    if args.turn_start > 0 or args.max_plies_left is not None:
        print(
            f"[init] game-phase filter: turn_start={args.turn_start} (plies) "
            f"max_plies_left={args.max_plies_left}"
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
        turn_start=args.turn_start,
        max_plies_left=args.max_plies_left,
        data_root=args.data_root,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        split_seed=args.split_seed,
        data_seed=args.data_seed,
    )
    n_train_files = len(train_loader.dataset.list_files())
    n_val_files = len(val_loader.dataset.list_files())
    print(
        f"[init] pooled split from {args.data_root!r}: "
        f"{n_train_files:,} train games / {n_val_files:,} val games "
        f"(split_seed={args.split_seed})"
    )
    print(
        f"[init] targets: {args.train_samples:,} train / {args.val_samples:,} "
        f"val filtered plies  (turn_start={args.turn_start}, "
        f"max_plies_left={args.max_plies_left})"
    )

    # Actual sizes + expected epochs. count_samples() honors the same filters
    # AND the max_samples cap, so it returns min(available, target) and stops
    # early once the cap is hit. With no plies filter this is near-instant
    # (gzip trailers); with a plies filter it decompresses until the cap.
    num_train_samples = None
    num_val_samples = None
    expected_epochs = None
    if args.count_samples:
        print("[init] counting samples (stops at the target cap; "
              "near-instant unless --max-plies-left is set)...")
        num_train_samples = train_loader.dataset.count_samples()
        num_val_samples = val_loader.dataset.count_samples()
        samples_seen = args.max_steps * args.batch_size
        expected_epochs = (
            samples_seen / num_train_samples if num_train_samples else None
        )
        print(f"[init] train samples: {num_train_samples:,}"
              + (f"  (target {args.train_samples:,})"
                 if num_train_samples < args.train_samples else "  (target met)"))
        print(f"[init] val samples:   {num_val_samples:,}"
              + (f"  (target {args.val_samples:,})"
                 if num_val_samples < args.val_samples else "  (target met)"))
        # Loud warning if a filter is too aggressive for the data pool.
        if num_train_samples < args.train_samples:
            print(f"[init] WARNING: train pool yields only "
                  f"{num_train_samples:,} < target {args.train_samples:,} "
                  f"filtered plies. The filter is too aggressive for the "
                  f"available data — add more data or relax the filter for a "
                  f"constant-size comparison.")
        if num_val_samples < args.val_samples:
            print(f"[init] WARNING: val pool yields only "
                  f"{num_val_samples:,} < target {args.val_samples:,}.")
        if expected_epochs is not None:
            print(
                f"[init] expected epochs: {expected_epochs:.3f}  "
                f"(= max_steps {args.max_steps} * batch {args.batch_size} "
                f"/ {num_train_samples:,} train samples)"
            )
        else:
            print("[init] expected epochs: n/a (0 train samples after filter)")
    else:
        print("[init] sample counting skipped (--no-count-samples)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    config_dict = vars(args).copy()
    config_dict["params"] = n_params
    config_dict["n_train_games"] = n_train_files
    config_dict["n_val_games"] = n_val_files
    config_dict["num_train_samples"] = num_train_samples
    config_dict["num_val_samples"] = num_val_samples
    config_dict["expected_epochs"] = expected_epochs
    (out_dir / "config.json").write_text(json.dumps(config_dict, indent=2))
    print(f"[init] output_dir={out_dir}  log={log_path.name}")

    log_file = open(log_path, "a")
    """
    # --- Added n_params in the run
    if start_step == 0:  # Only write this if we aren't resuming a previous run
        log_file.write(
            json.dumps({
                "phase": "init",
                "model": args.model,
                "total_parameters": n_params
            }) + "\n"
        )
        log_file.flush()
    # -------------------------------
    """
    print(
        f"[train] starting at step {start_step + 1} -> {args.max_steps}  "
        f"(batch={args.batch_size}, lr={args.lr})"
    )
    t0 = time.time()
    train_iter = iter(train_loader)
    # Window stores per-head losses + total over the last `log_every`
    # steps; we log both windowed means (`*_avg`) and the SINGLE
    # last-step values (`*_exact`) at each log_every checkpoint. Grad
    # norm is captured per-step and logged for the last step only.
    train_window = {"total": [], "policy": [], "value": [], "mlh": []}

    for step in range(start_step + 1, args.max_steps + 1):
        batch, train_iter = _next_batch(train_iter, train_loader)
        losses = train_one_step(
            model, batch, device, optimizer, grad_clip=args.grad_clip
        )
        for k in train_window:
            train_window[k].append(losses[k])

        if step % args.log_every == 0:
            window_means = {
                k: sum(v[-args.log_every:]) / min(len(v), args.log_every)
                for k, v in train_window.items()
            }
            elapsed = time.time() - t0
            samples = (step - start_step) * args.batch_size
            sps = samples / max(elapsed, 1e-6)
            grad_norm = losses["grad_norm"]
            print(
                f"[{step:>6d}/{args.max_steps}] "
                f"avg  L={window_means['total']:.4f}  "
                f"p={window_means['policy']:.3f} "
                f"v={window_means['value']:.3f} "
                f"m={window_means['mlh']:.3f}  "
                f"({sps:.0f} samples/sec)"
            )
            print(
                f"                  step L={losses['total']:.4f}  "
                f"p={losses['policy']:.3f} "
                f"v={losses['value']:.3f} "
                f"m={losses['mlh']:.3f}  "
                f"|grad|={grad_norm:.3f}"
            )
            log_file.write(
                json.dumps(
                    {
                        "step": step,
                        "phase": "train",
                        # Windowed means over the last log_every steps.
                        "total_avg": window_means["total"],
                        "policy_avg": window_means["policy"],
                        "value_avg": window_means["value"],
                        "mlh_avg": window_means["mlh"],
                        # Single last-step values.
                        "total_exact": losses["total"],
                        "policy_exact": losses["policy"],
                        "value_exact": losses["value"],
                        "mlh_exact": losses["mlh"],
                        # Last-step L2 gradient norm (pre-clip).
                        "grad_norm": grad_norm,
                        # Back-compat alias: older plot scripts read
                        # "loss_avg" / "policy" / "value" / "mlh" as
                        # the windowed mean. We keep them populated so
                        # downstream scripts (and old runs) continue
                        # to work; new code should prefer the *_avg /
                        # *_exact split above.
                        "loss_avg": window_means["total"],
                        "policy": window_means["policy"],
                        "value": window_means["value"],
                        "mlh": window_means["mlh"],
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


# --- TOTAL TIME IN JSON ---
    total_time = time.time() - t0
    total_time_str = f"{total_time / 60:.2f} minutes"

    print(f"[train] done. Total time: {total_time_str}.")

    # Write a final summary line to the JSONL file
    log_file.write(
        json.dumps({
            "phase": "summary",
            "total_time_seconds": total_time,
            "total_time_minutes": total_time / 60
        }) + "\n"
    )
    # -------------------------------
    log_file.close()

    # Render the 4-panel train/val loss curves (total + per-head). Failure
    # to plot must not abort the run — checkpoints + JSONL log are the
    # canonical artifacts; the PNG is a convenience and can also be
    # regenerated post-hoc via `python -m training.plot <output-dir>`.
    try:
        from .plot import plot_run
        plot_run(out_dir, save=True, show=False)
    except Exception as exc:
        print(f"[train] plot failed (non-fatal): {exc}")

    print("[train] done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train BaselineLCZero or a group-equivariant GCNNChess "
            "(gcnn / sgcnn / fgcnn / fgcnn2) on LCZero V6 data."
        )
    )
    p.add_argument("--model", choices=list(MODEL_NAMES), required=True)
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

    # Game-phase filters (focus training on a slice of the game, e.g.
    # endgames). Applied to BOTH the train and val streams. Each .gz is one
    # game in ply order, so the record index == ply-from-start.
    p.add_argument("--turn-start", type=int, default=0,
                   help="skip the first N records (plies / half-moves) of "
                        "every game; keep positions from ply N onward. "
                        "Default 0 keeps everything.")
    p.add_argument("--max-plies-left", type=float, default=None,
                   help="keep only positions with plies_left <= this value, "
                        "i.e. within this many plies of the game END (the "
                        "truer 'endgame' selector). Default None keeps "
                        "everything. Combine with --turn-start to keep a "
                        "mid-to-end window.")
    p.add_argument("--count-samples", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="at init, count train/val samples (honoring the "
                        "game-phase filters) and report expected epochs; "
                        "results are saved to config.json. Near-instant "
                        "without --max-plies-left; a full data pass with it. "
                        "Use --no-count-samples to skip.")

    # Pooled random split over ALL games under --data-root. A fixed budget of
    # FILTERED plies is drawn for each split, constant across turn_start /
    # max_plies_left (capped during iteration). Games are split disjointly
    # (no leakage) with --split-seed; --data-seed fixes the position order.
    p.add_argument("--data-root", default=RAW_DATA_ROOT,
                   help="root folder containing all .gz games (any sub-folder "
                        "names); pooled and split into train/val.")
    p.add_argument("--train-samples", type=int, default=TRAIN_SAMPLES,
                   help="target number of FILTERED training plies (cap).")
    p.add_argument("--val-samples", type=int, default=VAL_SAMPLES,
                   help="target number of FILTERED validation plies (cap).")
    p.add_argument("--split-seed", type=int, default=SPLIT_SEED,
                   help="seed for the game-level train/val partition (fixed "
                        "so val games + train pool are stable across runs and "
                        "filters).")
    p.add_argument("--data-seed", type=int, default=DATA_SEED,
                   help="seed for the per-position shuffle buffer (fixed so "
                        "the data order is identical across model runs).")

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

"""
python -m training.trainer `
    --model baseline `
    --blocks 10 `
    --filters 128 `
    --se-channels 32 `
    --mlh-channels 8 `
    --fc-size 128 `
    --batch-size 256 `
    --lr 5e-4 `
    --weight-decay 5e-4 `
    --max-steps 84000 `
    --log-every 4000 `
    --val-every 7000 `
    --val-batches 2000 `
    --checkpoint-every 42000 `
    --num-workers 0 `
    --turn-start 40 `
    --max-plies-left 160`
    --device cuda `
    --seed 0 `
    --output-dir runs/baseline_start_40_end_160
"""