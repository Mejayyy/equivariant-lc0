"""Plot training and validation loss curves from a JSONL training log.

Reads ``<output-dir>/train_log.jsonl`` produced by ``training/trainer.py``
and writes ``<output-dir>/loss_curves.png`` — a 2x2 grid of subplots
comparing train vs validation curves for:

    - Total loss        (weighted sum of the three head losses)
    - Policy loss       (cross-entropy over the 1858-move distribution)
    - Value loss        (cross-entropy over [W, D, L])
    - Moves-left loss   (L1 in plies)

Each subplot overlays the train curve (one point every ``--log-every``
steps, so it's smooth) and the val curve (one point every ``--val-every``
steps, so it's sparser; rendered with markers so individual val
checkpoints are visible).

Usable standalone after a run:

    python -m training.plot runs/gcnn_demo

or called automatically at the end of ``training.trainer.train``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


HEAD_KEYS = (
    ("total", "Total loss"),
    ("policy", "Policy loss (cross-entropy over 1858 moves)"),
    ("value", "Value loss (WDL cross-entropy)"),
    ("mlh", "Moves-left loss (L1, plies)"),
)


def _load_log(log_path: Path) -> Dict[str, Dict[str, List[float]]]:
    """Parse a JSONL trainer log into per-phase dict-of-lists.

    Train records use ``loss_avg`` for the windowed total; val records
    use ``total``. We normalise both into the ``total`` key.

    Records with ``phase`` other than ``train`` or ``val`` (e.g. the
    ``summary`` line at the end of a run) are ignored.
    """
    empty = lambda: {"step": [], "total": [], "policy": [], "value": [], "mlh": []}
    out = {"train": empty(), "val": empty()}
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            phase = rec.get("phase")
            if phase not in ("train", "val"):
                continue
            total = rec.get("total", rec.get("loss_avg"))
            if total is None:
                continue
            for key, val in (
                ("step", rec.get("step")),
                ("total", total),
                ("policy", rec.get("policy")),
                ("value", rec.get("value")),
                ("mlh", rec.get("mlh")),
            ):
                if val is None:
                    val = float("nan")
                out[phase][key].append(val)
    return out


def plot_run(
    output_dir,
    save: bool = True,
    show: bool = False,
    figsize=(12, 8),
) -> Optional[Path]:
    """Render the 2x2 grid of train/val curves and (by default) save it
    as ``<output_dir>/loss_curves.png``.

    Returns the path of the saved PNG (or ``None`` if ``save=False``).
    """
    import matplotlib  # local import — trainer.py shouldn't pay for it on every run

    if save and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    log_path = output_dir / "train_log.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(f"no train_log.jsonl in {output_dir}")

    data = _load_log(log_path)
    if not data["train"]["step"] and not data["val"]["step"]:
        raise ValueError(f"empty log at {log_path}")

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()
    for ax, (key, title) in zip(axes, HEAD_KEYS):
        if data["train"]["step"]:
            ax.plot(
                data["train"]["step"],
                data["train"][key],
                label="train",
                color="#1f77b4",
                alpha=0.7,
                linewidth=1.2,
            )
        if data["val"]["step"]:
            ax.plot(
                data["val"]["step"],
                data["val"][key],
                label="val",
                color="#d62728",
                marker="o",
                markersize=4,
                linewidth=1.5,
            )
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Training curves — {output_dir.name}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    saved = None
    if save:
        saved = output_dir / "loss_curves.png"
        fig.savefig(saved, dpi=120, bbox_inches="tight")
        print(f"[plot] saved -> {saved}")
    if show:
        plt.show()
    plt.close(fig)
    return saved


def main():
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Render train/val loss curves (total + policy + value + mlh) "
            "from a trainer run directory."
        )
    )
    p.add_argument(
        "output_dir",
        help="trainer --output-dir to read train_log.jsonl from",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="display the figure interactively in addition to saving it",
    )
    args = p.parse_args()
    plot_run(args.output_dir, save=True, show=args.show)


if __name__ == "__main__":
    main()
