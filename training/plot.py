"""Plot training / validation curves from a JSONL trainer log.

Reads ``<output-dir>/train_log.jsonl`` produced by ``training/trainer.py``
and writes three figures to ``<output-dir>``:

* ``loss_curves_avg.png``   — 2x2 grid (total, policy, value, mlh)
                              using the windowed-mean train losses
                              (``*_avg``) alongside the val curves.
* ``loss_curves_exact.png`` — 2x2 grid using the single-step train
                              losses (``*_exact``) at each log_every
                              checkpoint, alongside the val curves.
                              "Exact" means "not averaged over the
                              log_every window"; val is always a
                              batch-mean so its curve is identical
                              between the two figures.
* ``grad_norm.png``         — single-axis plot of the per-checkpoint
                              L2 gradient norm.

The legacy filename ``loss_curves.png`` is kept as a symlink-style alias
of ``loss_curves_avg.png`` for backwards compatibility with any external
tooling pinned to it.

Usable standalone after a run::

    python -m training.plot runs/gcnn_demo

or called automatically at the end of ``training.trainer.train``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


HEAD_KEYS: Tuple[Tuple[str, str], ...] = (
    ("total", "Total loss"),
    ("policy", "Policy loss (cross-entropy over 1858 moves)"),
    ("value", "Value loss (WDL cross-entropy)"),
    ("mlh", "Moves-left loss (L1, plies)"),
)


def _empty_train() -> Dict[str, List[float]]:
    return {
        "step": [],
        "total_avg": [],
        "policy_avg": [],
        "value_avg": [],
        "mlh_avg": [],
        "total_exact": [],
        "policy_exact": [],
        "value_exact": [],
        "mlh_exact": [],
        "grad_norm": [],
    }


def _empty_val() -> Dict[str, List[float]]:
    return {"step": [], "total": [], "policy": [], "value": [], "mlh": []}


def _train_record_fields(rec: dict) -> Dict[str, float]:
    """Pull the train-side fields out of one JSON record with backward
    compat for older logs.

    New schema keys: ``total_avg, policy_avg, value_avg, mlh_avg,
    total_exact, policy_exact, value_exact, mlh_exact, grad_norm``.
    Older logs only have ``loss_avg, policy, value, mlh`` — we treat
    those as the avg values and leave the exact / grad_norm fields as
    NaN (so plots that include them just gap-out for those steps).
    """
    NAN = float("nan")
    # Avg side
    total_avg = rec.get("total_avg")
    if total_avg is None:
        total_avg = rec.get("loss_avg", NAN)
    policy_avg = rec.get("policy_avg", rec.get("policy", NAN))
    value_avg = rec.get("value_avg", rec.get("value", NAN))
    mlh_avg = rec.get("mlh_avg", rec.get("mlh", NAN))
    # Exact side
    total_exact = rec.get("total_exact", NAN)
    policy_exact = rec.get("policy_exact", NAN)
    value_exact = rec.get("value_exact", NAN)
    mlh_exact = rec.get("mlh_exact", NAN)
    # Grad norm
    grad_norm = rec.get("grad_norm", NAN)
    return {
        "total_avg": total_avg,
        "policy_avg": policy_avg,
        "value_avg": value_avg,
        "mlh_avg": mlh_avg,
        "total_exact": total_exact,
        "policy_exact": policy_exact,
        "value_exact": value_exact,
        "mlh_exact": mlh_exact,
        "grad_norm": grad_norm,
    }


def _load_log(log_path: Path) -> Dict[str, Dict[str, List[float]]]:
    """Parse a JSONL trainer log into per-phase dict-of-lists.

    Train records: see ``_train_record_fields`` for the full set of
    parsed keys (avg + exact + grad_norm, with old-log fallback).
    Val records: ``step, total, policy, value, mlh``.

    Records with ``phase`` other than ``train`` or ``val`` (e.g. the
    ``summary`` line at the end of a run) are ignored.
    """
    out = {"train": _empty_train(), "val": _empty_val()}
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
            step = rec.get("step")
            if step is None:
                continue
            if phase == "train":
                fields = _train_record_fields(rec)
                out["train"]["step"].append(step)
                for k, v in fields.items():
                    out["train"][k].append(v)
            elif phase == "val":
                total = rec.get("total", rec.get("loss_avg"))
                if total is None:
                    continue
                out["val"]["step"].append(step)
                out["val"]["total"].append(total)
                out["val"]["policy"].append(rec.get("policy", float("nan")))
                out["val"]["value"].append(rec.get("value", float("nan")))
                out["val"]["mlh"].append(rec.get("mlh", float("nan")))
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _setup_matplotlib(save: bool, show: bool):
    import matplotlib
    if save and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _plot_2x2_train_vs_val(
    data: Dict[str, Dict[str, List[float]]],
    train_suffix: str,
    title: str,
    figsize=(12, 8),
):
    """Render a 2x2 grid comparing train vs val. ``train_suffix`` is
    either ``"_avg"`` (windowed-mean train losses) or ``"_exact"``
    (single-step train losses). Returns the matplotlib Figure.

    Val curves are identical between the two views (val is always a
    batch-mean over val_batches), so we plot them on both.
    """
    plt = _setup_matplotlib(save=True, show=False)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()
    for ax, (key, head_title) in zip(axes, HEAD_KEYS):
        train_key = f"{key}{train_suffix}"
        if data["train"]["step"]:
            ax.plot(
                data["train"]["step"],
                data["train"][train_key],
                label=f"train ({train_suffix.strip('_')})",
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
        ax.set_title(head_title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def _plot_grad_norm(
    data: Dict[str, Dict[str, List[float]]],
    title: str,
    figsize=(10, 5),
):
    plt = _setup_matplotlib(save=True, show=False)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if data["train"]["step"]:
        ax.plot(
            data["train"]["step"],
            data["train"]["grad_norm"],
            label="grad norm (pre-clip)",
            color="#2ca02c",
            linewidth=1.2,
        )
    ax.set_xlabel("step")
    ax.set_ylabel("||grad||_2")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_run(
    output_dir,
    save: bool = True,
    show: bool = False,
    figsize=(12, 8),
) -> Optional[Path]:
    """Render all three figures from a trainer run and (by default)
    save them under ``output_dir``.

    Returns the primary "avg" PNG path (``loss_curves_avg.png``) for
    back-compat with the previous single-figure API; the other two
    PNGs (``loss_curves_exact.png``, ``grad_norm.png``) are written
    alongside it but not returned. Pass ``save=False`` to skip writing
    entirely (e.g. for unit-testing the data plumbing).
    """
    plt = _setup_matplotlib(save=save, show=show)

    output_dir = Path(output_dir)
    log_path = output_dir / "train_log.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(f"no train_log.jsonl in {output_dir}")

    data = _load_log(log_path)
    if not data["train"]["step"] and not data["val"]["step"]:
        raise ValueError(f"empty log at {log_path}")

    figures = []

    # 1. Avg-loss 2x2 grid.
    fig_avg = _plot_2x2_train_vs_val(
        data,
        train_suffix="_avg",
        title=f"Training curves (window mean) — {output_dir.name}",
        figsize=figsize,
    )
    figures.append(("loss_curves_avg.png", fig_avg))

    # 2. Exact-loss 2x2 grid.
    fig_exact = _plot_2x2_train_vs_val(
        data,
        train_suffix="_exact",
        title=f"Training curves (single-step) — {output_dir.name}",
        figsize=figsize,
    )
    figures.append(("loss_curves_exact.png", fig_exact))

    # 3. Gradient norm.
    fig_grad = _plot_grad_norm(
        data,
        title=f"Gradient L2 norm (pre-clip) — {output_dir.name}",
    )
    figures.append(("grad_norm.png", fig_grad))

    saved_primary = None
    if save:
        for name, fig in figures:
            path = output_dir / name
            fig.savefig(path, dpi=120, bbox_inches="tight")
            print(f"[plot] saved -> {path}")
            if name == "loss_curves_avg.png":
                saved_primary = path

        # Back-compat alias: previous version of this script wrote
        # ``loss_curves.png``. Write that file too so any external
        # tooling pinned to the old name still finds it.
        legacy = output_dir / "loss_curves.png"
        fig_avg.savefig(legacy, dpi=120, bbox_inches="tight")

    if show:
        plt.show()
    for _, fig in figures:
        plt.close(fig)
    return saved_primary


def main():
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Render train/val loss curves (avg + exact) and the "
            "gradient-norm trajectory from a trainer run directory."
        )
    )
    p.add_argument(
        "output_dir",
        help="trainer --output-dir to read train_log.jsonl from",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="display the figures interactively in addition to saving them",
    )
    args = p.parse_args()
    plot_run(args.output_dir, save=True, show=args.show)


if __name__ == "__main__":
    main()
