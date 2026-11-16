"""Subsea intervention work sites (open-source scene assets, see ``rov_lab/assets_src/third_party``).

* ``shipwreck_site``: 180 m crop of Gazebo Fuel ``OpenRobotics/OceanFloorShipwreck`` (CC-BY 4.0) with a wreck
  15-25 m north-east of the work site, CC0 sand.
* ``portuguese_ledge``: 250 m patch of the MBARI Portuguese Ledge bathymetry (Gazebo Fuel, CC0), CC0 mud.

Both carry the same static layout (``work_site_layout``): DAVE skid frame, panel mounting plate, Ocean Rock
boulders (CC-BY 4.0) and UUV Simulator tools on the deck. The valve panel is a separate articulation
(:data:`rov_lab.assets.objects.rov_bop_panel.ROV_BOP_PANEL_CFG`).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

from rov_lab.assets.scenes.work_site_layout import WORK_SITE_SCENES
from rov_lab.utils.paths import SCENE_DIR


def work_site_cfg(scene: str = "shipwreck_site", per_env: bool = False) -> AssetBaseCfg:
    """Static work-site asset (terrain + structures).

    ``per_env=False``: the full site (180 m / 250 m terrain), spawned once at ``/World/WorkSite``; every env shares it,
    so use one env (or ``env_spacing=0``). ``per_env=True``: the compact site (+-25 m terrain crop) spawned in each env
    at ``{ENV_REGEX_NS}/WorkSite``, for parallel scenes (use ``env_spacing >= 60``)."""
    if scene not in WORK_SITE_SCENES:
        raise ValueError(f"Unknown work site {scene!r}; expected one of {WORK_SITE_SCENES}")
    name = f"{scene}_work_site_compact.usda" if per_env else f"{scene}_work_site.usda"
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WorkSite" if per_env else "/World/WorkSite",
        spawn=sim_utils.UsdFileCfg(usd_path=str(SCENE_DIR / scene / name)),
    )


SHIPWRECK_WORK_SITE_CFG: AssetBaseCfg = work_site_cfg("shipwreck_site")
PORTUGUESE_LEDGE_WORK_SITE_CFG: AssetBaseCfg = work_site_cfg("portuguese_ledge")
