"""Model loading + inference + equivariance comparison.

This is the compute layer the Streamlit app sits on top of. It is kept
free of any Streamlit imports so it can be exercised directly from a
script or a test.

Three responsibilities:

* **Loading** -- rebuild each model from its own checkpoint's ``config``
  (never hardcoded sizes) and load the weights.
* **Prediction** -- run a model on a position and return legal-move-masked
  top-k policy, WDL, and moves-left.
* **Equivariance** -- for a group element g, compute ``f(g.x)`` (what the
  model outputs on the transformed board) and ``g.f(x)`` (what an
  equivariant model *should* output), the per-head relative-Frobenius
  residual, and a MATCH/MISMATCH verdict. This uses the SAME repo
  functions as ``metrics/equivariance.py`` so the picture and the numbers
  agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import chess

from models.policy_map import POLICY_INDEX_TO_MOVE, POLICY_INDEX_TO_UCI, UCI_TO_POLICY_INDEX
from symmetry.group import GChess
from symmetry.actions import (
    group_action_input,
    group_action_policy,
    group_action_value,
    group_action_mlh,
    build_policy_permutation,
)
from training.trainer import build_model

# Match threshold for the equivariance verdict: well above the ~1e-6 the
# equivariant models hit, well below the O(0.1-1) a real mismatch shows.
MATCH_THRESHOLD = 1e-4

# Group elements we demo, in display order, with the metric's string keys.
GROUP_ELEMENTS: Dict[str, int] = {
    "sigma_h": int(GChess.SIGMA_H),
    "sigma_c": int(GChess.SIGMA_C),
    "sigma_h_c": int(GChess.SIGMA_HC),
}

GROUP_LABELS: Dict[str, str] = {
    "sigma_h": "σ_h  (file mirror)",
    "sigma_c": "σ_c  (color swap + rank flip)",
    "sigma_h_c": "σ_h∘σ_c  (180° rotation)",
}

# Which g's each model is equivariant under (for display). The keys are
# also the model names the app shows.
MODEL_CLAIMS: Dict[str, str] = {
    "baseline": "none (data-learned only)",
    "gcnn": "σ_h, σ_c, σ_h∘σ_c",
    "sgcnn": "σ_c",
    "fgcnn": "σ_h",
    "fgcnn2": "σ_h",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass
class LoadedModel:
    name: str
    model: torch.nn.Module
    config: dict
    step: int
    n_params: int
    ckpt_path: str

    @property
    def arch_label(self) -> str:
        c = self.config
        return f"blocks={c['blocks']}, filters={c['filters']}"


def load_model(ckpt_path: str) -> LoadedModel:
    """Load a checkpoint and rebuild + restore its model in eval mode.

    Works with full training checkpoints (``training/trainer.py``) and with
    the compact ones in ``app/checkpoints`` (model weights only, stored in
    float16; ``load_state_dict`` casts them back to float32).
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    c = ck["config"]
    if not isinstance(c, dict):
        c = vars(c)
    model = build_model(c["model"], c["blocks"], c["filters"],
                        c["se_channels"], c["mlh_channels"], c["fc_size"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    return LoadedModel(
        name=c["model"], model=model, config=c, step=int(ck.get("step", 0)),
        n_params=n_params, ckpt_path=ckpt_path,
    )


def discover_checkpoints(checkpoints_dir: str) -> Dict[str, str]:
    """Map ``model name -> checkpoint path`` by scanning ``*.pt`` files
    (recursively) and reading each checkpoint's ``config['model']``. When several checkpoints
    exist for the same model, the highest ``step`` wins."""
    import glob
    import os

    best: Dict[str, Tuple[int, str]] = {}
    patterns = ["*.pt", "**/*.pt"]
    seen = set()
    for pat in patterns:
        for path in glob.glob(os.path.join(checkpoints_dir, pat), recursive=True):
            if path in seen:
                continue
            seen.add(path)
            try:
                ck = torch.load(path, map_location="cpu", weights_only=False)
                c = ck.get("config")
                if c is None:
                    continue
                c = c if isinstance(c, dict) else vars(c)
                name = c.get("model")
                step = int(ck.get("step", 0))
            except Exception:
                continue
            if name in MODEL_CLAIMS:
                if name not in best or step > best[name][0]:
                    best[name] = (step, path)
    return {name: path for name, (step, path) in best.items()}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def planes_to_tensor(planes: np.ndarray) -> torch.Tensor:
    """(34, 8, 8) numpy -> (1, 34, 8, 8) float32 tensor."""
    return torch.from_numpy(np.ascontiguousarray(planes, dtype=np.float32)).unsqueeze(0)


def move_to_policy_index(board: chess.Board, move: chess.Move) -> Optional[int]:
    """Index of a legal move of ``board`` (canonical frame) in the 1858-move
    policy, following LCZero's encoding: castling is "king takes own rook"
    (e1h1, not e1g1) and a knight promotion is the plain pawn move (e7e8);
    queen, rook and bishop promotions keep their suffix."""
    uci = board.uci(move, chess960=True)
    if move.promotion == chess.KNIGHT:
        uci = uci[:4]
    return UCI_TO_POLICY_INDEX.get(uci)


@torch.no_grad()
def raw_forward(model: torch.nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(policy_logits[B,1858], value_wdl[B,3], moves_left[B,1])``."""
    model.eval()
    policy, value, mlh = model(x)
    return policy, value, mlh


@dataclass
class Prediction:
    wdl: np.ndarray            # softmax [W, D, L], side-to-move POV
    q: float                   # P(win) - P(loss) in [-1, 1], stm POV
    moves_left: float          # plies
    top_moves: List[dict]      # [{uci_canon, uci_abs, san, prob, index}]


def predict(model: torch.nn.Module, planes: np.ndarray, board: chess.Board,
            top_k: int = 5) -> Prediction:
    """Run a model and return legal-move-masked top-k + WDL + MLH.

    ``board`` is the ABSOLUTE position; legal moves are taken from its
    canonical frame (side to move at bottom) to line up with the policy's
    canonical indexing, then mapped back to absolute for SAN display.
    """
    x = planes_to_tensor(planes)
    policy, value, mlh = raw_forward(model, x)
    policy = policy[0]
    wdl = value[0].numpy()  # the value head already returns softmax [W, D, L]
    q = float(wdl[0] - wdl[2])
    moves_left = float(mlh[0].item())

    black_to_move = board.turn == chess.BLACK
    canonical = board if not black_to_move else board.mirror()

    # Gather logits for legal moves (canonical frame) and softmax over them.
    entries = []
    for mv in canonical.legal_moves:
        idx = move_to_policy_index(canonical, mv)
        if idx is None:
            continue
        entries.append((idx, mv.uci(), mv))

    top_moves: List[dict] = []
    if entries:
        idxs = torch.tensor([e[0] for e in entries], dtype=torch.long)
        legal_logits = policy.index_select(0, idxs)
        probs = torch.softmax(legal_logits, dim=-1).numpy()
        order = np.argsort(-probs)[:top_k]
        for rank in order:
            idx, uci_canon, mv_canon = entries[rank]
            mv_abs = _canon_move_to_absolute(mv_canon, black_to_move)
            san = _safe_san(board, mv_abs)
            top_moves.append({
                "index": int(idx),
                "uci_canon": uci_canon,
                "uci_abs": mv_abs.uci(),
                "san": san,
                "prob": float(probs[rank]),
                "from_canon": mv_canon.from_square,
                "to_canon": mv_canon.to_square,
                "from_abs": mv_abs.from_square,
                "to_abs": mv_abs.to_square,
            })

    return Prediction(wdl=wdl, q=q, moves_left=moves_left, top_moves=top_moves)


def _canon_move_to_absolute(mv: chess.Move, black_to_move: bool) -> chess.Move:
    """Map a canonical-frame move back to the absolute board (rank-mirror
    from/to squares when Black is to move)."""
    if not black_to_move:
        return mv
    return chess.Move(
        chess.square_mirror(mv.from_square),
        chess.square_mirror(mv.to_square),
        promotion=mv.promotion,
    )


def _safe_san(board: chess.Board, move: chess.Move) -> str:
    try:
        return board.san(move)
    except Exception:
        return move.uci()


# ---------------------------------------------------------------------------
# Equivariance comparison
# ---------------------------------------------------------------------------


def _rel_frob(a: torch.Tensor, b: torch.Tensor, ref: torch.Tensor, eps: float = 1e-9) -> float:
    """Relative-Frobenius residual ||a-b|| / (||ref|| + eps), matching
    metrics/equivariance.py."""
    return float((a - b).norm() / (ref.norm() + eps))


@dataclass
class EquivResult:
    g_key: str
    # model output on the transformed board: f(g.x)
    f_gx: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    # expected output if equivariant: g.f(x)
    g_f_x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    residual_policy: float
    residual_value: float
    residual_mlh: float
    residual_overall: float
    match: bool


@torch.no_grad()
def equivariance_compare(model: torch.nn.Module, x: torch.Tensor, g_key: str,
                         threshold: float = MATCH_THRESHOLD) -> EquivResult:
    """Compute f(g.x) vs g.f(x) for one model and one group element.

    Mirrors metrics/equivariance.py exactly: same group_action_input on the
    way in, same per-head output actions for the expected side, same
    relative-Frobenius residual per head, averaged for the overall.
    """
    g = GROUP_ELEMENTS[g_key]
    model.eval()

    fx = raw_forward(model, x)
    gx = group_action_input(g, x)
    fgx = raw_forward(model, gx)
    gfx = (
        group_action_policy(g, fx[0]),
        group_action_value(g, fx[1]),
        group_action_mlh(g, fx[2]),
    )

    r_pol = _rel_frob(fgx[0], gfx[0], fx[0])
    r_val = _rel_frob(fgx[1], gfx[1], fx[1])
    r_mlh = _rel_frob(fgx[2], gfx[2], fx[2])
    r_all = (r_pol + r_val + r_mlh) / 3.0

    return EquivResult(
        g_key=g_key, f_gx=fgx, g_f_x=gfx,
        residual_policy=r_pol, residual_value=r_val, residual_mlh=r_mlh,
        residual_overall=r_all, match=(r_all < threshold),
    )


def wdl_probs(value_row: torch.Tensor) -> np.ndarray:
    """[W, D, L] probabilities of one value-head output row. The value
    heads already end in a softmax, so this is a plain conversion."""
    return value_row.detach().cpu().numpy()


def legal_canonical_indices(board: chess.Board) -> List[int]:
    """Policy indices of the legal moves of ``board`` in the canonical frame
    (side to move at the bottom). These line up directly with the policy's
    1858-move indexing."""
    canonical = board if board.turn == chess.WHITE else board.mirror()
    out: List[int] = []
    for mv in canonical.legal_moves:
        idx = move_to_policy_index(canonical, mv)
        if idx is not None:
            out.append(idx)
    return out


def transform_indices(indices: List[int], g_key: str) -> List[int]:
    """Map a set of canonical move indices through the group action g. If the
    indices are the legal moves of x, the result is the legal moves of g·x
    (in canonical frame), since g permutes the move space."""
    perm = build_policy_permutation(GROUP_ELEMENTS[g_key])
    return [int(perm[i]) for i in indices]


def topk_over_indices(policy_row: torch.Tensor, indices: List[int],
                      k: int = 5) -> List[Tuple[int, float]]:
    """Top-k (index, softmax-prob) restricted to ``indices`` (e.g. the legal
    moves of the position), so arrows are legal rather than the raw 1858-wide
    argmax."""
    if not indices:
        return []
    idx_t = torch.tensor(indices, dtype=torch.long)
    logits = policy_row.index_select(0, idx_t)
    probs = torch.softmax(logits, dim=-1)
    kk = min(k, len(indices))
    order = torch.topk(probs, kk).indices.tolist()
    return [(indices[o], float(probs[o])) for o in order]


def canonical_index_to_squares(idx: int) -> Tuple[int, int]:
    """(from_sq, to_sq) of a policy index in canonical coordinates — draw
    directly on a board rendered with planes_to_board_canonical (no mirror)."""
    from_sq, to_sq, _promo = POLICY_INDEX_TO_MOVE[idx]
    return from_sq, to_sq


def policy_index_to_abs_squares(idx: int, black_to_move: bool) -> Tuple[int, int]:
    """Map a canonical-frame policy index to absolute (from_sq, to_sq) squares
    for drawing an arrow on a board rendered white-at-bottom. When Black is to
    move, the canonical frame is rank-mirrored relative to absolute."""
    from_sq, to_sq, _promo = POLICY_INDEX_TO_MOVE[idx]
    if black_to_move:
        from_sq = chess.square_mirror(from_sq)
        to_sq = chess.square_mirror(to_sq)
    return from_sq, to_sq


def top_indices(policy_row: torch.Tensor, k: int = 5) -> List[Tuple[int, float]]:
    """Top-k (index, softmax-prob) over the FULL 1858 policy (unmasked).

    Used by the equivariance demo, where we compare the model's raw policy
    response on the transformed board rather than the legal-move-masked set.
    """
    probs = torch.softmax(policy_row, dim=-1)
    vals, idxs = torch.topk(probs, k)
    return [(int(i), float(v)) for i, v in zip(idxs.tolist(), vals.tolist())]


__all__ = [
    "MATCH_THRESHOLD", "GROUP_ELEMENTS", "GROUP_LABELS", "MODEL_CLAIMS",
    "build_model", "load_model", "discover_checkpoints", "LoadedModel",
    "planes_to_tensor", "raw_forward", "predict", "Prediction",
    "equivariance_compare", "EquivResult", "wdl_probs", "top_indices",
    "move_to_policy_index",
    "policy_index_to_abs_squares",
    "legal_canonical_indices", "transform_indices", "topk_over_indices",
    "canonical_index_to_squares",
]
