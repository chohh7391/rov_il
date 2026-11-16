#!/usr/bin/env python3
"""Convert the I90-class ROV + Bravo 7-class arm URDF to one self-contained USD (Isaac Lab 3.0 URDF converter).

Same pipeline as ``convert_bravo7_class.py``, generalised to the vehicle articulation:
    1. ``isaaclab.sim.converters.UrdfConverter`` (Isaac Sim 6.2 URDF importer 3.0) writes its layered asset into a
       build directory.
    2. The composed stage (PhysX physics variant selected) is flattened into
       ``<ROV_LAB_ASSETS_ROOT>/robots/i90_bravo7/i90_bravo7.usd`` (gitignored, regenerate any time).

The base is floating (``fix_base=False``; root link ``rov_base``). Fixed joints are NOT merged so that the sensor
frames (``front_camera_link``, ``dvl_link``, ``pressure_sensor_link``), the arm ``base_link`` and ``jaw_base_link``
(behind ``ft_sensor_joint``) stay separate bodies. Arm drive gains come from ``I90_BRAVO7_CFG``.

Usage (from the repository root):
    python rov_lab/scripts/tools/make_i90_bravo7_urdf.py
    OMNI_KIT_ACCEPT_EULA=YES <isaaclab-3.0-python> rov_lab/scripts/tools/convert_i90_bravo7.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils.version import standalone_importers_available

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "rov_lab" / "source"))

from rov_lab.utils.paths import ROBOT_DIR  # noqa: E402  (pure-Python, honours ROV_LAB_ASSETS_ROOT)

DEFAULT_URDF: Path = REPO_ROOT / "rov_lab" / "assets_src" / "robots" / "i90_bravo7" / "i90_bravo7.urdf"
DEFAULT_USD: Path = ROBOT_DIR / "i90_bravo7" / "i90_bravo7.usd"

parser = argparse.ArgumentParser(description="Convert the I90 + Bravo 7-class URDF to USD (Isaac Lab 3.0).")
parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="Input URDF.")
parser.add_argument("--out", type=Path, default=DEFAULT_USD, help="Output (flattened) USD file.")
parser.add_argument(
    "--build-dir", type=Path, default=None, help="Importer output dir (default: <out dir>/_urdf_import)."
)
add_launcher_args(parser)
args_cli = parser.parse_args()
# Same launch decision as IsaacLab/scripts/tools/convert_urdf.py: the importer needs Kit unless the
# standalone importer wheel is installed.
args_cli.require_kit = not standalone_importers_available()
args_cli.physics = "isaacsim_physx" if args_cli.require_kit else "newton_mjwarp"


def gain_for(value: dict[str, float] | float | None, expr: str) -> float:
    """Resolve a scalar-or-dict actuator gain for one joint expression."""
    if isinstance(value, dict):
        return float(value[expr])
    return float(value or 0.0)


def drive_gains_from_cfg() -> tuple[dict[str, float], dict[str, float]]:
    """Per-joint stiffness/damping taken from ``I90_BRAVO7_CFG`` so the USD drives match the cfg."""
    from rov_lab.assets.robots.i90_bravo7 import I90_BRAVO7_CFG

    stiffness: dict[str, float] = {}
    damping: dict[str, float] = {}
    for actuator in I90_BRAVO7_CFG.actuators.values():
        for expr in actuator.joint_names_expr:
            stiffness[expr] = gain_for(actuator.stiffness, expr)
            damping[expr] = gain_for(actuator.damping, expr)
    return stiffness, damping


def flatten_to(src_usd: str, dst_usd: Path) -> None:
    """Flatten the composed importer output (selected variants included) into one USD file."""
    from pxr import Usd

    stage = Usd.Stage.Open(src_usd)
    if not stage:
        raise RuntimeError(f"Could not open converted stage: {src_usd}")
    dst_usd.parent.mkdir(parents=True, exist_ok=True)
    if not stage.Flatten().Export(str(dst_usd)):
        raise RuntimeError(f"Failed to export {dst_usd}")


def summarize(usd_path: Path) -> None:
    """Print the articulation root, bodies (mass, prim path) and joints found in the flattened USD."""
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    print(f"[convert] default prim: {stage.GetDefaultPrim().GetPath()}")
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            print(f"[convert] articulation root: {prim.GetPath()}")
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            mass_attr = UsdPhysics.MassAPI(prim).GetMassAttr()
            print(f"[convert] body  {prim.GetName():<22} mass={mass_attr.Get() if mass_attr else None}  {prim.GetPath()}")
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        lo = prim.GetAttribute("physics:lowerLimit").Get() if prim.HasAttribute("physics:lowerLimit") else None
        hi = prim.GetAttribute("physics:upperLimit").Get() if prim.HasAttribute("physics:upperLimit") else None
        print(f"[convert] joint {prim.GetName():<22} {prim.GetTypeName():<22} lim=({lo}, {hi})")


def main() -> None:
    urdf_path = args_cli.urdf.resolve()
    out_usd = args_cli.out.resolve()
    build_dir = (args_cli.build_dir or out_usd.parent / "_urdf_import").resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path} (run make_i90_bravo7_urdf.py first)")
    # The importer never overwrites (it writes <name>_1, <name>_2, ...), so start from an empty build dir.
    # Only a directory previously written by an Isaac Lab asset converter (it holds .asset_hash) is removed.
    if (build_dir / ".asset_hash").is_file():
        shutil.rmtree(build_dir)

    from isaaclab.physics import PhysicsCfg

    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli):
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

        stiffness, damping = drive_gains_from_cfg()
        cfg = UrdfConverterCfg(
            asset_path=str(urdf_path),
            usd_dir=str(build_dir),
            fix_base=False,  # floating vehicle
            merge_fixed_joints=False,  # keep sensor frames, base_link and jaw_base_link as bodies
            self_collision=False,
            collision_from_visuals=False,
            force_usd_conversion=True,
            physics_variant=UrdfConverterCfg.PhysicsVariant.PHYSX,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=stiffness, damping=damping),
            ),
        )
        converter = UrdfConverter(cfg)
        print(f"[convert] importer output: {converter.usd_path}")
        flatten_to(converter.usd_path, out_usd)
        print(f"[convert] flattened USD : {out_usd}")
        summarize(out_usd)


if __name__ == "__main__":
    main()
