"""Write the shared USD scene used to compare OceanSim (Isaac Sim 5.1) with the rov_lab port (Isaac Lab 3.0).

Both renderers open this exact file, so geometry, materials, lighting and the camera (pose and intrinsics) are
identical by construction. Run with any Python that has ``pxr`` (e.g. the Isaac Lab 3.0 env), no simulator needed:

    <isaaclab-3.0-python> rov_lab/scripts/tools/oceansim_compare/make_compare_scene.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

REPO = Path(__file__).resolve().parents[4]
SEABED = REPO / "rov_lab" / "assets" / "scenes" / "collected_MHL" / "mhl_scaled.usd"

# Camera: 640x480, 4:3 apertures, looking at three cubes ~0.85 m ahead (same view family as the uw3 experiments).
CAMERA = {
    "prim_path": "/World/uw_cam",
    "width": 640,
    "height": 480,
    "focal_length": 18.0,
    "horizontal_aperture": 20.955,
    "vertical_aperture": 20.955 * 3.0 / 4.0,
    "clipping_range": (0.05, 100.0),
    "eye": (0.2, 0.0, -1.2),
    "target": (1.0, 0.0, -1.45),
}
CUBES = {
    "red": ((1.0, -0.12, -1.46), (0.9, 0.05, 0.03)),
    "green": ((1.0, 0.0, -1.46), (0.05, 0.75, 0.10)),
    "blue": ((1.0, 0.12, -1.46), (0.05, 0.18, 0.9)),
}
CUBE_SIZE = 0.08
DOME = {"intensity": 2000.0, "color": (0.85, 0.9, 1.0)}


def look_at_matrix(eye: tuple[float, float, float], target: tuple[float, float, float]) -> Gf.Matrix4d:
    """USD camera transform (camera looks along local -Z with +Y up), world up = +Z."""
    e, t = Gf.Vec3d(*eye), Gf.Vec3d(*target)
    f = (t - e).GetNormalized()
    r = Gf.Cross(f, Gf.Vec3d(0, 0, 1)).GetNormalized()
    u = Gf.Cross(r, f)
    return Gf.Matrix4d(r[0], r[1], r[2], 0, u[0], u[1], u[2], 0, -f[0], -f[1], -f[2], 0, e[0], e[1], e[2], 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=REPO / "runs" / "oceansim_compare" / "compare_scene.usda")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(str(args.out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    seabed = stage.DefinePrim("/World/Seabed", "Xform")
    seabed.GetReferences().AddReference(str(SEABED))

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(DOME["intensity"])
    dome.CreateColorAttr(Gf.Vec3f(*DOME["color"]))

    UsdGeom.Scope.Define(stage, "/World/Looks")
    for name, (pos, color) in CUBES.items():
        mat = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
        shader = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        cube = UsdGeom.Cube.Define(stage, f"/World/Cubes/{name}")
        cube.CreateSizeAttr(CUBE_SIZE)
        cube.AddTranslateOp().Set(Gf.Vec3d(*pos))
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)

    cam = UsdGeom.Camera.Define(stage, CAMERA["prim_path"])
    cam.CreateFocalLengthAttr(CAMERA["focal_length"])
    cam.CreateHorizontalApertureAttr(CAMERA["horizontal_aperture"])
    cam.CreateVerticalApertureAttr(CAMERA["vertical_aperture"])
    cam.CreateClippingRangeAttr(Gf.Vec2f(*CAMERA["clipping_range"]))
    cam.AddTransformOp().Set(look_at_matrix(CAMERA["eye"], CAMERA["target"]))

    stage.GetRootLayer().Save()
    meta = {"scene": str(args.out), "seabed": str(SEABED), "camera": CAMERA, "cubes": CUBES, "cube_size": CUBE_SIZE, "dome": DOME}
    (args.out.parent / "compare_scene.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
