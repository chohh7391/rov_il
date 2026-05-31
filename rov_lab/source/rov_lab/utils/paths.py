"""Path helpers for ROV Lab assets."""

from __future__ import annotations

import os
from pathlib import Path


def _resolve_project_root() -> Path:
    """Resolve the repository root from this source file location."""
    return Path(__file__).resolve().parents[4]


def resolve_assets_root() -> Path:
    """Return the ROV Lab assets root.

    `ROV_LAB_ASSETS_ROOT` can override the default when assets are stored outside
    the repository checkout.
    """
    env_root = os.environ.get("ROV_LAB_ASSETS_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return (_resolve_project_root() / "rov_lab" / "assets").resolve()


ASSETS_ROOT: Path = resolve_assets_root()

