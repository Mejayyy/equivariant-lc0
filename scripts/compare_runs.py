"""Compare training runs: validation-loss curves + a Markdown summary table.

Reads ``<run>/config.json`` and ``<run>/train_log.jsonl`` (as written by
``training/trainer.py``) for every run directory given.

    python scripts/compare_runs.py results/baseline_start_* results/fgcnn_start_* \\
        --metric policy --facet turn_start --out docs/figures/val_policy

prints a Markdown table of each run's validation losses and writes
``<out>-light.png`` and ``<out>-dark.png`` (one panel per value of
``--facet``, one curve per model). Colours are bound to the model name,
so a model keeps its colour in every figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

METRICS = {
    "policy": "validation policy loss (nats)",
    "value": "validation value loss (nats)",
    "mlh": "validation moves-left loss (plies)",
    "total": "validation total loss",
}

# Fixed model -> categorical slot (light, dark) and marker shape.
SERIES = {
    "baseline": ("#2a78d6", "#3987e5", "o"),
    "fgcnn": ("#eb6834", "#d95926", "s"),
    "fgcnn2": ("#1baf7a", "#199e70", "^"),
    "gcnn": ("#eda100", "#c98500", "D"),
    "sgcnn": ("#e87ba4", "#d55181", "v"),
}

THEMES = {
    "light": dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
                  muted="#898781", grid="#e1e0d9", axis="#c3c2b7"),
    "dark": dict(surface="#1a1a19", primary="#f0efec", secondary="#c3c2b7",
                 muted="#898781", grid="#2c2c2a", axis="#383835"),
}

# Figures are 8 in wide at 200 dpi (1600 px) and get fitted to a ~830 CSS-px
# README column, so 1 CSS px ~= 1.9 image px ~= 0.7 pt.
DPI = 200
PX = 0.7
WIDTH_IN = 8.0


def load_run(run_dir) -> Dict:
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    val: List[Dict] = []
    with open(run_dir / "train_log.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("phase") == "val" and "step" in rec:
                val.append(rec)
    if not val:
        raise ValueError(f"{run_dir}: no validation records in train_log.jsonl")
    return {"name": run_dir.name, "config": config, "val": val}


def val_at(run: Dict, max_step: Optional[int]) -> Dict:
    """Last validation record at or before ``max_step`` (or the last one)."""
    recs = [r for r in run["val"] if max_step is None or r["step"] <= max_step]
    if not recs:
        raise ValueError(f"{run['name']}: no validation record <= step {max_step}")
    return recs[-1]


def markdown_table(runs: List[Dict], max_step: Optional[int]) -> str:
    rows = [
        "| run | model | params | turn_start | step | policy | value | MLH | total |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        c, v = run["config"], val_at(run, max_step)
        rows.append(
            f"| {run['name']} | {c['model']} | {c['params']:,} | "
            f"{c.get('turn_start') or 0} | {v['step'] // 1000}k | "
            f"{v['policy']:.3f} | {v['value']:.3f} | {v['mlh']:.1f} | {v['total']:.3f} |"
        )
    return "\n".join(rows)


def plot(runs: List[Dict], metric: str, facet: Optional[str], theme: str,
         out: Path, title: Optional[str], max_step: Optional[int]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    t = THEMES[theme]
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 13 * PX,
        "axes.edgecolor": t["axis"],
        "axes.labelcolor": t["secondary"],
        "xtick.color": t["muted"],
        "ytick.color": t["muted"],
        "xtick.labelcolor": t["secondary"],
        "ytick.labelcolor": t["secondary"],
    })

    groups = [None]
    if facet:
        groups = sorted({run["config"].get(facet) or 0 for run in runs})
    n = len(groups)
    fig, axes = plt.subplots(1, n, figsize=(WIDTH_IN, 3.6 if n > 1 else 4.0),
                             sharey=True, squeeze=False, dpi=DPI)
    fig.patch.set_facecolor(t["surface"])

    models_present = []
    for ax, group in zip(axes[0], groups):
        ax.set_facecolor(t["surface"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(PX)
        ax.grid(True, axis="y", color=t["grid"], linewidth=PX, linestyle="-")
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=4)

        for run in runs:
            if facet and (run["config"].get(facet) or 0) != group:
                continue
            model = run["config"]["model"]
            light, dark, marker = SERIES.get(model, ("#6250d6", "#9085e9", "P"))
            color = light if theme == "light" else dark
            recs = [r for r in run["val"] if max_step is None or r["step"] <= max_step]
            xs = [r["step"] for r in recs]
            ys = [r[metric] for r in recs]
            # Solid 2px line; one shaped end-dot (with a surface ring) marks the
            # final evaluation and doubles as a non-colour identity cue.
            ax.plot(xs, ys, color=color, linewidth=2 * PX,
                    solid_joinstyle="round", solid_capstyle="round",
                    marker=marker, markevery=[len(xs) - 1], markersize=8 * PX,
                    markerfacecolor=color, markeredgecolor=t["surface"],
                    markeredgewidth=2 * PX, label=model, zorder=3)
            if model not in models_present:
                models_present.append(model)

        ax.set_xlim(left=0)
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _: "0" if x == 0 else f"{x / 1000:.0f}k"))
        if facet:
            ax.set_title(f"{facet} = {group}", color=t["primary"],
                         fontsize=14 * PX, pad=6)
        ax.set_xlabel("training step", color=t["secondary"])
    axes[0][0].set_ylabel(METRICS[metric], color=t["secondary"])

    from matplotlib.lines import Line2D

    handles, labels = [], []
    for model in models_present:
        light, dark, marker = SERIES.get(model, ("#6250d6", "#9085e9", "P"))
        color = light if theme == "light" else dark
        handles.append(Line2D([], [], color=color, linewidth=2 * PX, marker=marker,
                              markersize=8 * PX, markeredgewidth=0))
        labels.append(model)
    legend = fig.legend(handles, labels, loc="upper right", ncol=len(labels),
                        frameon=False, bbox_to_anchor=(0.995, 0.995),
                        handlelength=2.2, columnspacing=1.4)
    for text in legend.get_texts():
        text.set_color(t["secondary"])

    if title:
        fig.text(0.01, 0.985, title, ha="left", va="top", color=t["primary"],
                 fontsize=16 * PX, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, facecolor=t["surface"])
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("runs", nargs="+", help="run directories")
    p.add_argument("--metric", choices=list(METRICS), default="policy")
    p.add_argument("--facet", default=None,
                   help="config key to split into panels, e.g. turn_start")
    p.add_argument("--max-step", type=int, default=None,
                   help="ignore validation records after this step")
    p.add_argument("--title", default=None)
    p.add_argument("--out", default=None,
                   help="output path stem; writes <out>-light.png and <out>-dark.png")
    args = p.parse_args(argv)

    runs = [load_run(r) for r in args.runs]
    print(markdown_table(runs, args.max_step))
    if args.out:
        for theme in THEMES:
            path = Path(f"{args.out}-{theme}.png")
            plot(runs, args.metric, args.facet, theme, path, args.title, args.max_step)
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
