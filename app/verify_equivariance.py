"""Headless check of the app's equivariance verdicts on the shipped
checkpoints: the numeric metric (metrics/equivariance.py) for every model
and group element, the on-board residual + MATCH/MISMATCH verdict that the
app's badges use on a real position, and the sigma_c W<->L value swap.

Run from the repository root:  python app/verify_equivariance.py
"""

import os
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
for _path in (str(_APP_DIR), str(_APP_DIR.parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import metrics.equivariance as E  # noqa: E402
from viz.encoding import fen_to_planes, planes_to_fen  # noqa: E402
from viz.modelio import (  # noqa: E402
    GROUP_ELEMENTS, MATCH_THRESHOLD, MODEL_CLAIMS, discover_checkpoints,
    equivariance_compare, load_model, planes_to_tensor, raw_forward, wdl_probs,
)

CKPT_DIR = str(_APP_DIR / "checkpoints")

# A recognizable Black-to-move position (Sicilian, 2...) so we can check the
# sigma_c W<->L swap and that nothing is accidentally rank-flipped.
BLACK_FEN = "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"


def _ascii(text: str) -> str:
    """Console-safe version of a label (Windows consoles may not print σ)."""
    return text.replace("σ", "sigma").replace("∘", " o ")


def main() -> int:
    paths = discover_checkpoints(CKPT_DIR)
    names = [n for n in MODEL_CLAIMS if n in paths]
    if not names:
        print(f"no checkpoints found in {CKPT_DIR}")
        return 1
    models = {name: load_model(paths[name]) for name in names}
    print("Checkpoints:")
    for name, lm in models.items():
        print(f"  {name:9s} {os.path.basename(lm.ckpt_path):14s} {lm.arch_label}, "
              f"params={lm.n_params:,}, step={lm.step}")
    print()

    # (A) numeric metric over random inputs, all three group elements.
    print("(A) equivariance_error over 64 random inputs (metrics/equivariance.py)")
    print(f"{'model':9s} {'sigma_h':>10s} {'sigma_c':>10s} {'sigma_h_c':>10s}   equivariant under")
    for name, lm in models.items():
        r = E.equivariance_error(lm.model, n_samples=64)
        print(f"{name:9s} {r['sigma_h']:10.2e} {r['sigma_c']:10.2e} {r['sigma_h_c']:10.2e}   "
              f"{_ascii(MODEL_CLAIMS[name])}")
    print()

    # (B) the residual + verdict the app's badges show, on a real position.
    planes = fen_to_planes(BLACK_FEN, history="repeat")
    print(f"(B) on-board residual and verdict (threshold {MATCH_THRESHOLD:g})")
    print(f"    position: {BLACK_FEN}")
    print(f"    encode -> decode round trip: {planes_to_fen(planes)}")
    x = planes_to_tensor(planes)
    print(f"{'model':9s} {'g':10s} {'policy':>10s} {'value':>10s} {'mlh':>10s} "
          f"{'overall':>10s}  verdict")
    for name, lm in models.items():
        for g_key in GROUP_ELEMENTS:
            res = equivariance_compare(lm.model, x, g_key)
            print(f"{name:9s} {g_key:10s} {res.residual_policy:10.2e} "
                  f"{res.residual_value:10.2e} {res.residual_mlh:10.2e} "
                  f"{res.residual_overall:10.2e}  {'MATCH' if res.match else 'MISMATCH'}")
    print()

    # (C) sigma_c swaps W and L in the expected output g.f(x); a sigma_c-
    # equivariant model reproduces it on the transformed board f(g.x).
    print("(C) sigma_c value check, [W, D, L]")
    for name, lm in models.items():
        wdl_fx = wdl_probs(raw_forward(lm.model, x)[1][0])
        res = equivariance_compare(lm.model, x, "sigma_c")
        expected, actual = wdl_probs(res.g_f_x[1][0]), wdl_probs(res.f_gx[1][0])
        print(f"  {name:9s} f(x) {wdl_fx.round(3)}   g.f(x) {expected.round(3)}   "
              f"f(g.x) {actual.round(3)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
