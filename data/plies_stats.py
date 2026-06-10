"""Empirical distributions of game length and `plies_left` over the pooled
training data, and the filtered-ply yield of the game-phase filters.

Why this exists: the `max_plies_left` filter keeps a record only if its
`plies_left` target is at or below the threshold. But LCZero self-play games
are frequently adjudicated (resignation / tablebase / agreed result) BEFORE
the hypothetical natural end, and the stored `plies_left` values count down
toward that *projected* end — so the final recorded position of a game often
still has a large `plies_left`. A small threshold therefore harvests far
fewer plies per game than `threshold` itself, and can even harvest zero from
most games. This script measures exactly that, so filter settings can be
chosen with eyes open.

Outputs (written to ``--out``, default ``runs/plies_stats``):

* ``game_length_hist.png``      — distribution of total plies per game.
* ``final_plies_left_hist.png`` — distribution of the LAST record's
                                  plies_left per game (the adjudication gap);
                                  only games left of the `max_plies_left`
                                  line contribute anything to that filter.
* ``plies_left_hist.png``       — distribution of plies_left across ALL
                                  sampled plies (the population the filter
                                  selects from).
* ``yield_curves.png``          — estimated filtered-ply totals for the full
                                  train/val pools vs `max_plies_left` and vs
                                  `turn_start`, with the budget target lines.
* ``summary.json``              — all numbers in machine-readable form.

Run::

    python -m data.plies_stats --sample 4000

The game sample is drawn (seeded) from the SAME pooled train split the
trainer uses (`build_pooled_splits` with `SPLIT_SEED`), so the estimates
extrapolate directly to the training pool.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import RAW_DATA_ROOT, SPLIT_SEED, TRAIN_SAMPLES, VAL_SAMPLES
from .splits import build_pooled_splits
from .v6_parser import iter_plies_left_from_file


# Filter grids reported in the yield tables / curves.
MAX_PLIES_LEFT_GRID = (10, 20, 30, 40, 60, 80, 100, 120, 160, 200, 240, 320)
TURN_START_GRID = (0, 10, 20, 40, 60, 80, 100, 120, 160)


def filtered_count(
    plies_left: Sequence[float],
    turn_start: int = 0,
    max_plies_left: Optional[float] = None,
) -> int:
    """Records of ONE game passing the game-phase filter — the same rule as
    ``LCZeroV6Dataset._keep_record``: keep iff in-game index >= turn_start
    AND (no plies filter or plies_left <= max_plies_left)."""
    n = 0
    for idx, pl in enumerate(plies_left):
        if idx < turn_start:
            continue
        if max_plies_left is not None and pl > max_plies_left:
            continue
        n += 1
    return n


def scan_games(files: Sequence[str]) -> List[List[float]]:
    """Read the plies_left sequence of every file (decompresses; cheap
    single-float parse per record)."""
    return [list(iter_plies_left_from_file(f)) for f in files]


def crosscheck_dataset(
    files: Sequence[str],
    games: Sequence[Sequence[float]],
    turn_start: int,
    max_plies_left: Optional[float],
) -> None:
    """Assert that the REAL dataset iteration path (full record parse via
    ``LCZeroV6Dataset``) yields exactly the count predicted from the fast
    plies_left stream — i.e. the two filter implementations agree and the
    fast path reads the right bytes."""
    from .dataset import LCZeroV6Dataset

    expected = sum(
        filtered_count(g, turn_start, max_plies_left) for g in games
    )
    ds = LCZeroV6Dataset(
        files=list(files),
        shuffle_buffer=0,
        turn_start=turn_start,
        max_plies_left=max_plies_left,
    )
    got = sum(1 for _ in ds)
    if got != expected:
        raise AssertionError(
            f"dataset iteration produced {got} samples but the plies_left "
            f"stream predicts {expected} (turn_start={turn_start}, "
            f"max_plies_left={max_plies_left})"
        )
    counted = ds.count_samples()
    if counted != expected:
        raise AssertionError(
            f"count_samples() returned {counted}, expected {expected}"
        )


def _mean_se(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return float(arr.mean()), se


def analyze(
    games: List[List[float]],
    n_train_games: int,
    n_val_games: int,
) -> Dict:
    """Compute all summary statistics + extrapolated yield tables."""
    lengths = np.array([len(g) for g in games])
    first_pl = np.array([g[0] for g in games])
    final_pl = np.array([g[-1] for g in games])

    # Monotonicity: plies_left should decrease by exactly 1.0 per record
    # (record index == ply-from-start). Count violating games.
    non_monotone = sum(
        1
        for g in games
        if any(abs((g[i] - g[i + 1]) - 1.0) > 1e-6 for i in range(len(g) - 1))
    )

    out: Dict = {
        "n_games_sampled": len(games),
        "n_train_games": n_train_games,
        "n_val_games": n_val_games,
        "game_length": {
            "mean": float(lengths.mean()),
            "median": float(np.median(lengths)),
            "min": int(lengths.min()),
            "max": int(lengths.max()),
        },
        "first_plies_left": {
            "mean": float(first_pl.mean()),
            "min": float(first_pl.min()),
            "max": float(first_pl.max()),
        },
        "final_plies_left": {
            "mean": float(final_pl.mean()),
            "median": float(np.median(final_pl)),
            "min": float(final_pl.min()),
            "max": float(final_pl.max()),
            "pct_games_le": {
                str(t): float((final_pl <= t).mean() * 100.0)
                for t in MAX_PLIES_LEFT_GRID
            },
        },
        "non_monotone_games": int(non_monotone),
    }

    # Yield tables. Per-game filtered counts -> mean +/- SE -> extrapolate
    # to the full train/val pools (game distributions are identical by the
    # random split, so one sample serves both).
    def yield_row(turn_start: int, mpl: Optional[float]) -> Dict:
        per_game = [filtered_count(g, turn_start, mpl) for g in games]
        mean, se = _mean_se(per_game)
        return {
            "turn_start": turn_start,
            "max_plies_left": mpl,
            "mean_per_game": mean,
            "se_per_game": se,
            "est_train_total": mean * n_train_games,
            "est_train_total_pm": 1.96 * se * n_train_games,
            "est_val_total": mean * n_val_games,
            "pct_games_contributing": float(
                np.mean([c > 0 for c in per_game]) * 100.0
            ),
        }

    out["yield_vs_max_plies_left"] = [
        yield_row(0, float(t)) for t in MAX_PLIES_LEFT_GRID
    ]
    out["yield_vs_turn_start"] = [
        yield_row(t, None) for t in TURN_START_GRID
    ]
    out["yield_unfiltered"] = yield_row(0, None)
    return out


def _print_report(stats: Dict, train_target: int, val_target: int) -> None:
    gl = stats["game_length"]
    fp = stats["final_plies_left"]
    print(f"\nSampled {stats['n_games_sampled']} games from the pooled train "
          f"split ({stats['n_train_games']:,} train / "
          f"{stats['n_val_games']:,} val games total).")
    print(f"Game length (plies/records per game): mean {gl['mean']:.1f}  "
          f"median {gl['median']:.0f}  min {gl['min']}  max {gl['max']}")
    print(f"First-record plies_left: mean {stats['first_plies_left']['mean']:.1f}  "
          f"range [{stats['first_plies_left']['min']:.0f}, "
          f"{stats['first_plies_left']['max']:.0f}]")
    print(f"FINAL-record plies_left (adjudication gap): mean {fp['mean']:.1f}  "
          f"median {fp['median']:.0f}  range [{fp['min']:.0f}, {fp['max']:.0f}]")
    print(f"Games with non-monotone plies_left (-1.0/record violated): "
          f"{stats['non_monotone_games']}")

    print(f"\nYield vs max_plies_left (turn_start=0); "
          f"targets: train {train_target:,} / val {val_target:,}")
    print(f"{'thresh':>7} {'plies/game':>11} {'%games>0':>9} "
          f"{'est train total':>18} {'est val total':>14}  fills?")
    for row in stats["yield_vs_max_plies_left"]:
        fills = ("YES" if row["est_train_total"] >= train_target
                 and row["est_val_total"] >= val_target else "no")
        print(f"{row['max_plies_left']:>7.0f} {row['mean_per_game']:>11.2f} "
              f"{row['pct_games_contributing']:>8.1f}% "
              f"{row['est_train_total']:>14,.0f} +/- "
              f"{row['est_train_total_pm']:>4,.0f} "
              f"{row['est_val_total']:>14,.0f}  {fills}")

    print(f"\nYield vs turn_start (max_plies_left=None)")
    print(f"{'start':>7} {'plies/game':>11} {'%games>0':>9} "
          f"{'est train total':>18} {'est val total':>14}  fills?")
    for row in stats["yield_vs_turn_start"]:
        fills = ("YES" if row["est_train_total"] >= train_target
                 and row["est_val_total"] >= val_target else "no")
        print(f"{row['turn_start']:>7d} {row['mean_per_game']:>11.2f} "
              f"{row['pct_games_contributing']:>8.1f}% "
              f"{row['est_train_total']:>14,.0f} +/- "
              f"{row['est_train_total_pm']:>4,.0f} "
              f"{row['est_val_total']:>14,.0f}  {fills}")


def make_plots(
    games: List[List[float]],
    stats: Dict,
    out_dir: Path,
    train_target: int,
    val_target: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lengths = np.array([len(g) for g in games])
    final_pl = np.array([g[-1] for g in games])
    all_pl = np.concatenate([np.asarray(g) for g in games])

    # 1. Game length distribution (total plies per game).
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lengths, bins=np.arange(0, lengths.max() + 10, 10),
            color="steelblue", edgecolor="white")
    ax.axvline(lengths.mean(), color="crimson", linestyle="--",
               label=f"mean = {lengths.mean():.1f}")
    ax.axvline(np.median(lengths), color="darkorange", linestyle=":",
               label=f"median = {np.median(lengths):.0f}")
    ax.set_xlabel("total plies per game (records per .gz)")
    ax.set_ylabel("games")
    ax.set_title(f"Game length distribution ({len(games)} sampled games)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "game_length_hist.png", dpi=150)
    plt.close(fig)

    # 2. FINAL plies_left per game (the adjudication gap).
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(final_pl, bins=np.arange(0, final_pl.max() + 5, 5),
            color="seagreen", edgecolor="white")
    ax.axvline(30, color="crimson", linestyle="--",
               label="max_plies_left = 30")
    pct30 = stats["final_plies_left"]["pct_games_le"]["30"]
    ax.set_xlabel("plies_left of the LAST record in the game")
    ax.set_ylabel("games")
    ax.set_title(
        "Final-record plies_left per game "
        f"(only the {pct30:.1f}% of games left of the line\n"
        "contribute ANY plies at max_plies_left=30)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "final_plies_left_hist.png", dpi=150)
    plt.close(fig)

    # 3. plies_left across all plies.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_pl, bins=np.arange(0, all_pl.max() + 10, 10),
            color="slategray", edgecolor="white")
    ax.axvline(30, color="crimson", linestyle="--",
               label="max_plies_left = 30")
    frac30 = float((all_pl <= 30).mean() * 100.0)
    ax.set_xlabel("plies_left")
    ax.set_ylabel("plies")
    ax.set_title(
        f"plies_left over all {len(all_pl):,} sampled plies "
        f"({frac30:.1f}% are <= 30)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "plies_left_hist.png", dpi=150)
    plt.close(fig)

    # 4. Yield curves vs the two filters, against the budget targets.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    rows = stats["yield_vs_max_plies_left"]
    x = [r["max_plies_left"] for r in rows]
    axes[0].plot(x, [r["est_train_total"] for r in rows], "o-",
                 color="steelblue", label="train pool (est.)")
    axes[0].plot(x, [r["est_val_total"] for r in rows], "s-",
                 color="seagreen", label="val pool (est.)")
    axes[0].axhline(train_target, color="steelblue", linestyle="--",
                    label=f"train target {train_target:,}")
    axes[0].axhline(val_target, color="seagreen", linestyle="--",
                    label=f"val target {val_target:,}")
    axes[0].set_xlabel("max_plies_left")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("filtered plies available (log)")
    axes[0].set_title("Yield vs max_plies_left (turn_start=0)")
    axes[0].legend(fontsize=8)

    rows = stats["yield_vs_turn_start"]
    x = [r["turn_start"] for r in rows]
    axes[1].plot(x, [r["est_train_total"] for r in rows], "o-",
                 color="steelblue", label="train pool (est.)")
    axes[1].plot(x, [r["est_val_total"] for r in rows], "s-",
                 color="seagreen", label="val pool (est.)")
    axes[1].axhline(train_target, color="steelblue", linestyle="--")
    axes[1].axhline(val_target, color="seagreen", linestyle="--")
    axes[1].set_xlabel("turn_start (plies)")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("filtered plies available (log)")
    axes[1].set_title("Yield vs turn_start (max_plies_left=None)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "yield_curves.png", dpi=150)
    plt.close(fig)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-root", default=RAW_DATA_ROOT)
    p.add_argument("--sample", type=int, default=4000,
                   help="number of train-split games to scan")
    p.add_argument("--seed", type=int, default=7,
                   help="seed for the game sample (NOT the split seed)")
    p.add_argument("--out", default="runs/plies_stats")
    p.add_argument("--train-samples", type=int, default=TRAIN_SAMPLES)
    p.add_argument("--val-samples", type=int, default=VAL_SAMPLES)
    p.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    args = p.parse_args(argv)

    train_files, val_files = build_pooled_splits(
        args.data_root, train_target=args.train_samples,
        val_target=args.val_samples, seed=args.split_seed,
    )
    rng = random.Random(args.seed)
    sample = rng.sample(train_files, min(args.sample, len(train_files)))
    print(f"Scanning plies_left of {len(sample)} sampled train games "
          f"(of {len(train_files):,})...")
    games = scan_games(sample)

    # Validate the real dataset path against the fast stream on a few games,
    # for every filter shape (none / turn_start / max_plies_left / both).
    print("Cross-checking LCZeroV6Dataset iteration vs the plies_left "
          "stream on 8 games...")
    for ts, mpl in ((0, None), (20, None), (0, 30.0), (20, 30.0)):
        crosscheck_dataset(sample[:8], games[:8], ts, mpl)
    print("Cross-check OK: dataset yield == count_samples == predicted, "
          "for all four filter shapes.")

    stats = analyze(games, len(train_files), len(val_files))
    _print_report(stats, args.train_samples, args.val_samples)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    make_plots(games, stats, out_dir, args.train_samples, args.val_samples)
    (out_dir / "summary.json").write_text(json.dumps(stats, indent=2))
    print(f"\nWrote plots + summary.json to {out_dir}")


if __name__ == "__main__":
    main()
