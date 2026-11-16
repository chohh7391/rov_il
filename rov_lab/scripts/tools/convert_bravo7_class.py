#!/usr/bin/env python3
"""Convert the Bravo 7-class URDF to a single, self-contained USD with the Isaac Lab 3.0 URDF converter.

Pipeline:
    1. ``isaaclab.sim.converters.UrdfConverter`` (Isaac Sim 6.2 URDF importer 3.0) writes its layered asset
       (interface ``.usda`` + payloads + ``Physics`` variant set) into a build directory.
    2. The composed stage, with the PhysX physics variant selected, is flattened into one file at
       ``<ROV_LAB_ASSETS_ROOT>/robots/bravo7_class/bravo7_class.usd`` (gitignored, regenerate any time).

Fixed joints are NOT merged so that ``jaw_base_link`` stays a separate body behind ``ft_sensor_joint``
(wrist wrench readout). The base is left floating in the USD; ``BRAVO7_CLASS_CFG`` fixes it at spawn
(``fix_root_link``) and an ROV integration can instead attach ``base_link`` to the hull.

Usage (from the repository root):
    python rov_lab/scripts/tools/make_bravo7_class_urdf.py
    OMNI_KIT_ACCEPT_EULA=YES <isaaclab-3.0-python> \\
        rov_lab/scripts/tools/convert_bravo7_class.py
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

DEFAULT_URDF: Path = REPO_ROOT / "rov_lab" / "assets_src" / "robots" / "bravo7_class" / "bravo7_class.urdf"
DEFAULT_USD: Path = ROBOT_DIR / "bravo7_class" / "bravo7_class.usd"

parser = argparse.ArgumentParser(description="Convert the Bravo 7-class URDF to USD (Isaac Lab 3.0).")
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


def drive_gains_from_cfg() -> tuple[dict[str, float], dict[str, float]]:
    """Per-joint (regex) stiffness/damping taken from ``BRAVO7_CLASS_CFG`` so the USD matches the cfg."""
    from rov_lab.assets.robots.bravo7_class import BRAVO7_CLASS_CFG

    stiffness: dict[str, float] = {}
    damping: dict[str, float] = {}
    for actuator in BRAVO7_CLASS_CFG.actuators.values():
        for expr in actuator.joint_names_expr:
            stiffness[expr] = _gain_for(actuator.stiffness, expr)
            damping[expr] = _gain_for(actuator.damping, expr)
    return stiffness, damping


def _gain_for(value: dict[str, float] | float | None, expr: str) -> float:
    """Resolve a scalar-or-dict actuator gain for one joint expression."""
    if isinstance(value, dict):
        return float(value[expr])
    return float(value or 0.0)


def flatten_to(src_usd: str, dst_usd: Path) -> None:
    """Flatten the composed importer output (selected variants included) into one USD file."""
    from pxr import Usd

    stage = Usd.Stage.Open(src_usd)
    if not stage:
        raise RuntimeError(f"Could not open converted stage: {src_usd}")
    dst_usd.parent.mkdir(parents=True, exist_ok=True)
    flat = stage.Flatten()
    if not flat.Export(str(dst_usd)):
        raise RuntimeError(f"Failed to export {dst_usd}")


def summarize(usd_path: Path) -> None:
    """Print bodies (mass) and joints (type, limits, drive) found in the flattened USD."""
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    print(f"[convert] default prim: {stage.GetDefaultPrim().GetPath()}")
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            print(f"[convert] articulation root: {prim.GetPath()}")
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            mass_attr = UsdPhysics.MassAPI(prim).GetMassAttr()
            print(f"[convert] body  {prim.GetName():<20} mass={mass_attr.Get() if mass_attr else None}")
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        kind = prim.GetTypeName()
        lo = prim.GetAttribute("physics:lowerLimit").Get() if prim.HasAttribute("physics:lowerLimit") else None
        hi = prim.GetAttribute("physics:upperLimit").Get() if prim.HasAttribute("physics:upperLimit") else None
        drive = ""
        for inst in ("angular", "linear"):
            if prim.HasAPI(UsdPhysics.DriveAPI, inst):
                d = UsdPhysics.DriveAPI(prim, inst)
                drive = (f" drive[{inst}] k={d.GetStiffnessAttr().Get()} d={d.GetDampingAttr().Get()} "
                         f"maxF={d.GetMaxForceAttr().Get()}")
        vmax = prim.GetAttribute("physxJoint:maxJointVelocity")
        vtxt = f" vmax={vmax.Get()}" if vmax and vmax.HasAuthoredValue() else ""
        print(f"[convert] joint {prim.GetName():<22} {kind:<22} lim=({lo}, {hi}){drive}{vtxt}")


def main() -> None:
    urdf_path = args_cli.urdf.resolve()
    out_usd = args_cli.out.resolve()
    build_dir = (args_cli.build_dir or out_usd.parent / "_urdf_import").resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path} (run make_bravo7_class_urdf.py first)")
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
            fix_base=False,
            merge_fixed_joints=False,  # keep jaw_base_link behind ft_sensor_joint
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
