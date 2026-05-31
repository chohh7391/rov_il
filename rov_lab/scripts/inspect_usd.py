"""Inspect USD composition and physics metadata for ROV assets.

Run with the Isaac Sim Python environment, for example:

    ../.venv-rov_lab/bin/python scripts/inspect_usd.py assets/robots/blue_rov/blue_rov_single_arm.usd
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaacsim import SimulationApp


def _format_refs(refs) -> str:
    if not refs:
        return ""
    return ", ".join(str(ref) for ref in refs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect USD hierarchy, references, and physics metadata.")
    parser.add_argument("usd_path", type=Path, help="USD file to inspect.")
    parser.add_argument("--flatten_out", type=Path, default=None, help="Optional path to export a flattened USDA.")
    parser.add_argument("--max_depth", type=int, default=8, help="Maximum hierarchy depth to print.")
    args = parser.parse_args()

    app = SimulationApp({"headless": True})

    from pxr import Usd, UsdPhysics  # noqa: PLC0415

    usd_path = args.usd_path.expanduser().resolve()
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {usd_path}")

    print(f"[USD] {usd_path}")
    print(f"[DEFAULT_PRIM] {stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else '<none>'}")

    if args.flatten_out is not None:
        flatten_out = args.flatten_out.expanduser().resolve()
        flatten_out.parent.mkdir(parents=True, exist_ok=True)
        stage.Flatten().Export(str(flatten_out))
        print(f"[FLATTENED] {flatten_out}")

    print("\n[HIERARCHY]")
    for prim in stage.Traverse():
        path = prim.GetPath()
        depth = len(path.pathString.strip("/").split("/")) if path.pathString != "/" else 0
        if depth > args.max_depth:
            continue
        indent = "  " * max(depth - 1, 0)
        type_name = prim.GetTypeName() or "<untyped>"
        schemas = prim.GetAppliedSchemas()
        refs = prim.GetMetadata("references")
        refs_text = _format_refs(refs.GetAddedOrExplicitItems()) if refs else ""
        marker_parts: list[str] = []
        if UsdPhysics.RigidBodyAPI(prim):
            marker_parts.append("RigidBody")
        if UsdPhysics.MassAPI(prim):
            marker_parts.append("MassAPI")
        if UsdPhysics.ArticulationRootAPI(prim):
            marker_parts.append("ArticulationRoot")
        if prim.IsA(UsdPhysics.Joint):
            marker_parts.append("Joint")
        if refs_text:
            marker_parts.append(f"refs=[{refs_text}]")
        if schemas:
            marker_parts.append(f"schemas={schemas}")
        suffix = f"  ({'; '.join(marker_parts)})" if marker_parts else ""
        print(f"{indent}{path.name} [{type_name}]{suffix}")

    print("\n[RIGID_BODIES_AND_MASS]")
    for prim in stage.Traverse():
        if not UsdPhysics.RigidBodyAPI(prim):
            continue
        mass_api = UsdPhysics.MassAPI(prim)
        mass = mass_api.GetMassAttr().Get() if mass_api else None
        density = mass_api.GetDensityAttr().Get() if mass_api else None
        com = mass_api.GetCenterOfMassAttr().Get() if mass_api else None
        inertia = mass_api.GetDiagonalInertiaAttr().Get() if mass_api else None
        print(f"{prim.GetPath()} mass={mass} density={density} com={com} diagonal_inertia={inertia}")

    print("\n[JOINTS]")
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        local_pos0 = joint.GetLocalPos0Attr().Get()
        local_rot0 = joint.GetLocalRot0Attr().Get()
        local_pos1 = joint.GetLocalPos1Attr().Get()
        local_rot1 = joint.GetLocalRot1Attr().Get()
        print(
            f"{prim.GetPath()} type={prim.GetTypeName()} "
            f"body0={body0} body1={body1} "
            f"local0=({local_pos0}, {local_rot0}) local1=({local_pos1}, {local_rot1})"
        )

    app.close()


if __name__ == "__main__":
    main()
