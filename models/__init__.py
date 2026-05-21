from .baseline import BaselineBody, BaselineLCZero, ResBlock, SELayer
from .gcnn import (
    GROUP_SIZE,
    GBatchNorm,
    GCNNBody,
    GCNNChess,
    GConv,
    GLinear,
    GResBlock,
    GSELayer,
    LiftingConv,
    group_action_g_signal,
    regular_rep_perm,
)
from .heads import (
    BaselineMovesLeftHead,
    BaselinePolicyHead,
    BaselineValueHead,
    EquivariantPolicyHead,
    EquivariantValueHead,
    InvariantMovesLeftHead,
)
from .policy_map import (
    MOVE_TO_POLICY_INDEX,
    POLICY_INDEX_SIZE,
    POLICY_INDEX_TO_MOVE,
    POLICY_INDEX_TO_UCI,
    POLICY_NUM_CHANNELS,
    UCI_TO_POLICY_INDEX,
    build_flat_to_policy_index,
    build_orbit_policy_gather,
    build_policy_gather_indices,
    move_to_uci,
    square_to_uci,
)

__all__ = [
    # policy_map
    "POLICY_INDEX_SIZE",
    "POLICY_NUM_CHANNELS",
    "POLICY_INDEX_TO_MOVE",
    "MOVE_TO_POLICY_INDEX",
    "POLICY_INDEX_TO_UCI",
    "UCI_TO_POLICY_INDEX",
    "square_to_uci",
    "move_to_uci",
    "build_policy_gather_indices",
    "build_flat_to_policy_index",
    "build_orbit_policy_gather",
    # baseline
    "BaselineLCZero",
    "BaselineBody",
    "ResBlock",
    "SELayer",
    "BaselinePolicyHead",
    "BaselineValueHead",
    "BaselineMovesLeftHead",
    # gcnn primitives
    "GROUP_SIZE",
    "LiftingConv",
    "GConv",
    "GLinear",
    "GBatchNorm",
    "GSELayer",
    "regular_rep_perm",
    "group_action_g_signal",
    # gcnn model (Phase 8)
    "GResBlock",
    "GCNNBody",
    "GCNNChess",
    # equivariant heads (Phase 8)
    "EquivariantPolicyHead",
    "EquivariantValueHead",
    "InvariantMovesLeftHead",
]
