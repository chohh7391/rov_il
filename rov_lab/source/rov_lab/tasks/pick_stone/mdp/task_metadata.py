"""Pure-Python task metadata for the colored pick-stone task."""

from __future__ import annotations

TARGET_COLOR_NAMES: tuple[str, ...] = ("red", "green", "blue")
TARGET_INSTRUCTIONS: tuple[str, ...] = (
    "pick up the red cube",
    "pick up the green cube",
    "pick up the blue cube",
)
CUBE_ASSET_NAMES: tuple[str, ...] = ("red_cube", "green_cube", "blue_cube")


def validate_target_id(target_color_id: int) -> int:
    """Return a valid target id or raise a clear error."""
    if not 0 <= target_color_id < len(TARGET_COLOR_NAMES):
        raise ValueError(
            f"target_color_id must be in [0, {len(TARGET_COLOR_NAMES) - 1}], got {target_color_id}"
        )
    return target_color_id


def color_name_from_target_id(target_color_id: int) -> str:
    """Return the color name for a target id."""
    return TARGET_COLOR_NAMES[validate_target_id(target_color_id)]


def instruction_from_target_id(target_color_id: int) -> str:
    """Return the natural-language instruction for a target id."""
    return TARGET_INSTRUCTIONS[validate_target_id(target_color_id)]


def cube_asset_name_from_target_id(target_color_id: int) -> str:
    """Return the scene asset name for a target id."""
    return CUBE_ASSET_NAMES[validate_target_id(target_color_id)]
