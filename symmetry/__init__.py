from .group import (
    COMPOSITION_TABLE,
    ELEMENTS,
    FLIPGROUP_COMPOSITION_TABLE,
    FLIPGROUP_ELEMENTS,
    FLIPGROUP_GROUP,
    FLIPGROUP_ORDER,
    FlipGroup,
    GCHESS_GROUP,
    GChess,
    GroupDef,
    ORDER,
    SGCHESS_COMPOSITION_TABLE,
    SGCHESS_ELEMENTS,
    SGCHESS_GROUP,
    SGCHESS_ORDER,
    SGChess,
    compose,
    inverse,
)
from .actions import (
    build_policy_permutation,
    group_action_input,
    group_action_mlh,
    group_action_output,
    group_action_policy,
    group_action_value,
)

__all__ = [
    # G_chess
    "GChess",
    "ELEMENTS",
    "ORDER",
    "COMPOSITION_TABLE",
    "compose",
    "inverse",
    # SG_chess subgroup
    "SGChess",
    "SGCHESS_ELEMENTS",
    "SGCHESS_ORDER",
    "SGCHESS_COMPOSITION_TABLE",
    # FlipGroup subgroup
    "FlipGroup",
    "FLIPGROUP_ELEMENTS",
    "FLIPGROUP_ORDER",
    "FLIPGROUP_COMPOSITION_TABLE",
    # generic group descriptor + presets
    "GroupDef",
    "GCHESS_GROUP",
    "SGCHESS_GROUP",
    "FLIPGROUP_GROUP",
    # chess-specific actions (always defined on G_chess elements)
    "group_action_input",
    "group_action_output",
    "group_action_policy",
    "group_action_value",
    "group_action_mlh",
    "build_policy_permutation",
]
