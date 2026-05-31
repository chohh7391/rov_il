"""Configuration for underwater scene assets used by ROV Lab."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

from rov_lab.utils.paths import SCENE_DIR


UNDER_WATER_WITH_MHL_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(SCENE_DIR / "collected_MHL" / "mhl_water.usd"),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    ),
)
