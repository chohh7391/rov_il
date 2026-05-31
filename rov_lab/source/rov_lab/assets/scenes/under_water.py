"""Configuration for underwater scene assets used by ROV Lab."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

from rov_lab.utils.paths import ASSETS_ROOT


OCEAN_SIM_ASSETS_PATH: Path = ASSETS_ROOT / "OceanSim_assets"
UNDER_WATER_WITH_MHL_USD_PATH: str = str(OCEAN_SIM_ASSETS_PATH / "collected_MHL" / "mhl_water.usd")

UNDER_WATER_WITH_MHL_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=UNDER_WATER_WITH_MHL_USD_PATH,
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    ),
)

