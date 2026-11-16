"""Build the subsea-intervention USD assets from the third-party sources (Kit-free: pxr + trimesh + h5py).

Inputs come from ``fetch_third_party_assets.py`` (``rov_lab/assets/third_party``); outputs go to
``<ROV_LAB_ASSETS_ROOT>/objects`` and ``<ROV_LAB_ASSETS_ROOT>/scenes`` (gitignored, published with the HF dataset).

Assets (all Z-up, meters):
  objects/rov_bop_panel/rov_bop_panel.usda   articulation: fixed panel + 5 revolute paddle valves (UUV Simulator)
  objects/hot_stab/hot_stab.usda             rigid tool, 4.4 kg (UUV Simulator)
  objects/torque_tool/torque_tool.usda       rigid tool, 3.0 kg (UUV Simulator)
  objects/subsea_frame/subsea_frame.usda     static skid frame with an electrical socket (DAVE electrical_panel)
  objects/wet_mate_plug/wet_mate_plug.usda   rigid plug (DAVE f_to_m_plug)
  objects/ocean_rock_0N/ocean_rock_0N.usda   static rocks normalised to 1 m (Gazebo Fuel, Cole)
  scenes/shipwreck_site/shipwreck_site.usda  180 m crop of OceanFloorShipwreck (Gazebo Fuel), work site at the origin
  scenes/portuguese_ledge/portuguese_ledge.usda  250 m patch of MBARI Portuguese Ledge bathymetry (Gazebo Fuel)

Textures under the texturelib license (UUV Simulator panel) are replaced with CC0 Poly Haven textures. Third-party
notices: ``rov_lab/assets_src/third_party/{NOTICE,ATTRIBUTION.md}``.

    <isaaclab-3.0-python> rov_lab/scripts/tools/build_subsea_assets.py [--only panel tools frame rocks shipwreck ledge]
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import trimesh
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

REPO: Path = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "rov_lab" / "source"))
from rov_lab.utils.paths import ASSETS_ROOT, OBJECT_DIR, SCENE_DIR  # noqa: E402  (pure Python)


def _load_layout():  # noqa: ANN202  (module object)
    """Import ``work_site_layout`` by path: ``rov_lab.assets.scenes`` itself imports Isaac Lab spawners."""
    import importlib.util

    path = REPO / "rov_lab" / "source" / "rov_lab" / "assets" / "scenes" / "work_site_layout.py"
    spec = importlib.util.spec_from_file_location("work_site_layout", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


L = _load_layout()

SRC: Path = ASSETS_ROOT / "third_party"
UUV: Path = SRC / "github" / "uuv_simulator" / "uuv_gazebo_worlds" / "models"
DAVE: Path = SRC / "github" / "dave" / "models" / "dave_object_models" / "models"
FUEL: Path = SRC / "fuel"
POLY: Path = SRC / "polyhaven"


# ----------------------------------------------------------------------------------------------------------------------
# mesh loading
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class Part:
    """One triangle mesh with its material, in the asset frame (meters)."""

    name: str
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray | None = None
    uv: np.ndarray | None = None
    color: tuple[float, float, float] = (0.7, 0.7, 0.7)
    roughness: float = 0.6
    metallic: float = 0.0
    textures: dict[str, Path] = field(default_factory=dict)
    image: Image.Image | None = None


def load_parts(path: Path, transform: np.ndarray | None = None) -> list[Part]:
    """Load a mesh file with its node transforms (and an extra 4x4 ``transform``) into :class:`Part` s."""
    scene = trimesh.load(str(path), force="scene")
    parts: list[Part] = []
    for node in scene.graph.nodes_geometry:
        node_tf, geom_name = scene.graph[node]
        g: trimesh.Trimesh = scene.geometry[geom_name]
        tf = node_tf if transform is None else transform @ node_tf
        verts = trimesh.transform_points(g.vertices, tf)
        rot = tf[:3, :3] / np.cbrt(abs(np.linalg.det(tf[:3, :3])))
        normals = g.vertex_normals @ rot.T
        visual = g.visual
        uv = getattr(visual, "uv", None)
        mat = getattr(visual, "material", None)
        color, rough, metal, image = (0.7, 0.7, 0.7), 0.6, 0.0, None
        if mat is not None:
            base = getattr(mat, "baseColorFactor", None)
            if base is not None:
                color = tuple(float(c) / 255.0 for c in np.asarray(base)[:3])
            rough = float(getattr(mat, "roughnessFactor", None) or rough)
            metal = float(getattr(mat, "metallicFactor", None) or 0.0)
            image = getattr(mat, "baseColorTexture", None)
        parts.append(Part(geom_name.replace("-", "_").replace(".", "_"), verts, np.asarray(g.faces), normals,
                          None if uv is None else np.asarray(uv), color, rough, metal, {}, image))
    return parts


def crop_part(part: Part, lo: np.ndarray, hi: np.ndarray) -> Part | None:
    """Keep faces whose centroid lies in the axis-aligned box [lo, hi] (xy only)."""
    c = part.vertices[part.faces].mean(axis=1)
    keep = np.all((c[:, :2] >= lo[:2]) & (c[:, :2] <= hi[:2]), axis=1)
    if not keep.any():
        return None
    faces = part.faces[keep]
    used, inv = np.unique(faces.reshape(-1), return_inverse=True)
    part.vertices = part.vertices[used]
    part.faces = inv.reshape(-1, 3)
    part.normals = None if part.normals is None else part.normals[used]
    part.uv = None if part.uv is None else part.uv[used]
    return part


def tile_part(part: Part, tiles: int) -> list[Part]:
    """Split a mesh into ``tiles`` x ``tiles`` pieces by face centroid (xy), each re-indexed."""
    c = part.vertices[part.faces].mean(axis=1)
    lo, hi = c[:, :2].min(0), c[:, :2].max(0) + 1e-6
    ij = np.floor((c[:, :2] - lo) / (hi - lo) * tiles).astype(int)
    out = []
    for i in range(tiles):
        for j in range(tiles):
            keep = (ij[:, 0] == i) & (ij[:, 1] == j)
            if not keep.any():
                continue
            faces = part.faces[keep]
            used, inv = np.unique(faces.reshape(-1), return_inverse=True)
            out.append(Part(f"{part.name}_tile_{i}_{j}", part.vertices[used], inv.reshape(-1, 3),
                            None if part.normals is None else part.normals[used],
                            None if part.uv is None else part.uv[used], part.color, part.roughness, part.metallic,
                            dict(part.textures), part.image))
    return out


def write_terrain(stage: Usd.Stage, mats: Materials, parent: str, part: Part, tiles: int = 6) -> None:
    """Large terrain mesh: one full mesh for collision / ray casting (purpose ``guide``, not rendered) at
    ``<parent>/<name>``, plus ``tiles`` x ``tiles`` render-only pieces under ``<parent>/<name>_visual``.

    RTX geometry streaming never reports completion for a single ~500k-triangle mesh, and Isaac Lab 3.0 then waits
    30 s per rendered frame; tiles of a few 10k triangles stream normally."""
    prim = write_mesh(stage, f"{parent}/{part.name}", Part(part.name, part.vertices, part.faces, part.normals), None)
    add_collision(prim, "none")
    UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.guide)
    write_parts(stage, mats, f"{parent}/{part.name}_visual", tile_part(part, tiles), None)


COMPACT_HALF: float = 25.0
"""Half size (m) of the compact terrain used for per-env work sites in parallel (multi-env) scenes."""


def write_terrain_usd(usd: Path, root: str, parts: list[Part], ground_name: str, tiles: int = 6) -> None:
    """Write terrain parts: the ground as guide collision mesh + render tiles, the rest as meshes (corals visual only)."""
    if usd.exists():
        usd.unlink()
    stage, _ = new_stage(usd, root, kind="assembly")
    mats = Materials(stage, usd, f"/{root}/Looks")
    for p in parts:
        if p.name == ground_name:
            write_terrain(stage, mats, f"/{root}/terrain", p, tiles=tiles)
        else:
            coll = None if p.name.startswith("Coral") else "none"
            write_parts(stage, mats, f"/{root}/terrain", [p], coll)
    stage.GetRootLayer().Save()


def compact_parts(parts: list[Part], half: float = COMPACT_HALF) -> list[Part]:
    """Copies of ``parts`` cropped to +-half around the origin (xy)."""
    import copy

    lo, hi = np.array([-half, -half, -np.inf]), np.array([half, half, np.inf])
    return [c for c in (crop_part(copy.deepcopy(p), lo, hi) for p in parts) if c is not None]


def box_uv(part: Part, tile: float) -> np.ndarray:
    """Per-vertex box-projection UVs (dominant normal axis), ``tile`` meters per texture repeat."""
    n = np.abs(part.normals if part.normals is not None else np.ones_like(part.vertices))
    axis = n.argmax(axis=1)
    v = part.vertices / tile
    uv = np.where((axis == 0)[:, None], v[:, [1, 2]], np.where((axis == 1)[:, None], v[:, [0, 2]], v[:, [0, 1]]))
    return uv


def subdivide(part: Part, levels: int) -> Part:
    """Midpoint-subdivide every triangle ``levels`` times (UVs/normals are recomputed by the caller)."""
    v, f = part.vertices, part.faces
    for _ in range(levels):
        v, f = trimesh.remesh.subdivide(v, f)
    part.vertices, part.faces, part.uv = v, f, None
    part.normals = trimesh.Trimesh(v, f, process=False).vertex_normals
    return part


def add_micro_relief(part: Part, amplitude: float, seed: int = 0) -> Part:
    """Add smooth sand-ripple style relief (sum of random plane waves, wavelengths 0.6-6 m) to a terrain mesh."""
    rng = np.random.default_rng(seed)
    xy = part.vertices[:, :2]
    dz = np.zeros(len(xy))
    for _ in range(24):
        lam = float(np.exp(rng.uniform(np.log(0.6), np.log(6.0))))
        th, ph = rng.uniform(0, 2 * np.pi, size=2)
        dz += (lam / 6.0) ** 0.7 * np.sin((xy @ np.array([np.cos(th), np.sin(th)])) * 2 * np.pi / lam + ph)
    part.vertices = part.vertices.copy()
    part.vertices[:, 2] += amplitude * dz / np.abs(dz).max()
    part.normals = trimesh.Trimesh(part.vertices, part.faces, process=False).vertex_normals
    return part


def level_pad(part: Part, r0: float, r1: float) -> Part:
    """Flatten the terrain to z = 0 within ``r0`` of the origin, blending back to the original at ``r1``."""
    r = np.hypot(part.vertices[:, 0], part.vertices[:, 1])
    t = np.clip((r - r0) / (r1 - r0), 0.0, 1.0)
    part.vertices = part.vertices.copy()
    part.vertices[:, 2] *= t * t * (3.0 - 2.0 * t)
    part.normals = trimesh.Trimesh(part.vertices, part.faces, process=False).vertex_normals
    return part


def tf_matrix(pos: tuple[float, float, float] = (0, 0, 0), rpy: tuple[float, float, float] = (0, 0, 0),
              scale: float = 1.0) -> np.ndarray:
    """4x4 transform from position, roll-pitch-yaw (rad, fixed XYZ as in SDF/URDF) and a uniform scale."""
    m = trimesh.transformations.euler_matrix(*rpy, axes="sxyz")
    m[:3, :3] *= scale
    m[:3, 3] = pos
    return m


def rpy_quat(rpy: tuple[float, float, float]) -> Gf.Quatf:
    q = trimesh.transformations.quaternion_from_euler(*rpy, axes="sxyz")  # wxyz
    return Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3]))


# ----------------------------------------------------------------------------------------------------------------------
# USD writing
# ----------------------------------------------------------------------------------------------------------------------
def new_stage(path: Path, root: str, kind: str = "component") -> tuple[Usd.Stage, Usd.Prim]:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    prim = UsdGeom.Xform.Define(stage, f"/{root}").GetPrim()
    stage.SetDefaultPrim(prim)
    Usd.ModelAPI(prim).SetKind(kind)
    return stage, prim


def set_pose(prim: Usd.Prim, pos: tuple[float, float, float] = (0, 0, 0),
             quat: Gf.Quatf | None = None, scale: float | None = None) -> None:
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    xf.AddOrientOp().Set(quat or Gf.Quatf(1, 0, 0, 0))
    if scale is not None:
        xf.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))


class Materials:
    """Creates UsdPreviewSurface materials and copies their textures next to the USD file."""

    def __init__(self, stage: Usd.Stage, usd_path: Path, scope: str) -> None:
        self.stage, self.dir, self.scope = stage, usd_path.parent, scope
        UsdGeom.Scope.Define(stage, scope)
        self.cache: dict[str, UsdShade.Material] = {}

    def _tex_file(self, src: Path | Image.Image, name: str) -> str:
        (self.dir / "textures").mkdir(parents=True, exist_ok=True)
        if isinstance(src, Image.Image):
            dst = self.dir / "textures" / f"{name}.png"
            src.convert("RGB").save(dst)
        else:
            dst = self.dir / "textures" / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
        return f"./textures/{dst.name}"

    def get(self, key: str, color: tuple[float, float, float] = (0.7, 0.7, 0.7), roughness: float = 0.6,
            metallic: float = 0.0, textures: dict[str, Path | Image.Image] | None = None) -> UsdShade.Material:
        if key in self.cache:
            return self.cache[key]
        st, path = self.stage, f"{self.scope}/{key}"
        mat = UsdShade.Material.Define(st, path)
        sh = UsdShade.Shader.Define(st, f"{path}/PreviewSurface")
        sh.CreateIdAttr("UsdPreviewSurface")
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        textures = textures or {}
        st_out = None
        if textures:
            reader = UsdShade.Shader.Define(st, f"{path}/stReader")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            st_out = reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        def tex(name: str, src: Path | Image.Image, srgb: bool, out: str, out_type: Sdf.ValueTypeName,
                scale_bias: bool = False) -> UsdShade.Output:
            t = UsdShade.Shader.Define(st, f"{path}/{name}Tex")
            t.CreateIdAttr("UsdUVTexture")
            t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(self._tex_file(src, f"{key}_{name}"))
            t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_out)
            t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB" if srgb else "raw")
            if scale_bias:
                t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
                t.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
            return t.CreateOutput(out, out_type)

        if "diffuse" in textures:
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                tex("diffuse", textures["diffuse"], True, "rgb", Sdf.ValueTypeNames.Float3))
        else:
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        if "normal" in textures:
            sh.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
                tex("normal", textures["normal"], False, "rgb", Sdf.ValueTypeNames.Float3, scale_bias=True))
        if "roughness" in textures:
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
                tex("roughness", textures["roughness"], False, "r", Sdf.ValueTypeNames.Float))
        else:
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
        if "metallic" in textures:
            sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).ConnectToSource(
                tex("metallic", textures["metallic"], False, "r", Sdf.ValueTypeNames.Float))
        else:
            sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
        if "occlusion" in textures:
            sh.CreateInput("occlusion", Sdf.ValueTypeNames.Float).ConnectToSource(
                tex("occlusion", textures["occlusion"], False, "r", Sdf.ValueTypeNames.Float))
        self.cache[key] = mat
        return mat


def write_mesh(stage: Usd.Stage, path: str, part: Part, material: UsdShade.Material | None) -> Usd.Prim:
    mesh = UsdGeom.Mesh.Define(stage, path)
    v = part.vertices.astype(np.float32)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(part.faces), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(part.faces.reshape(-1).astype(np.int32)))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*v.min(0).tolist()), Gf.Vec3f(*v.max(0).tolist())]))
    if part.normals is not None:
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(part.normals.astype(np.float32)))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if part.uv is not None:
        pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec2fArray.FromNumpy(part.uv.astype(np.float32)))
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh.GetPrim()


def add_collision(prim: Usd.Prim, approximation: str | None) -> None:
    """Collision on a mesh prim. ``approximation``: none (triangle mesh, static only), convexHull, convexDecomposition."""
    UsdPhysics.CollisionAPI.Apply(prim)
    if approximation is not None and prim.IsA(UsdGeom.Mesh):
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)


def collision_box(stage: Usd.Stage, path: str, center: tuple[float, float, float], size: tuple[float, float, float],
                  quat: Gf.Quatf | None = None) -> Usd.Prim:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreatePurposeAttr(UsdGeom.Tokens.guide)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddOrientOp().Set(quat or Gf.Quatf(1, 0, 0, 0))
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube.GetPrim()


def collision_cylinder(stage: Usd.Stage, path: str, center: tuple[float, float, float], radius: float,
                       height: float) -> Usd.Prim:
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    cyl.CreateAxisAttr(UsdGeom.Tokens.z)
    cyl.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdGeom.Xformable(cyl).AddTranslateOp().Set(Gf.Vec3d(*center))
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    return cyl.GetPrim()


def write_parts(stage: Usd.Stage, mats: Materials, parent: str, parts: list[Part], collision: str | None,
                tex_override: dict[str, dict[str, Path]] | None = None, prefix: str = "") -> list[Usd.Prim]:
    prims = []
    for i, p in enumerate(parts):
        name = f"{prefix}{p.name}" if p.name[0].isalpha() else f"{prefix}m{i}_{p.name}"
        textures: dict[str, Path | Image.Image] = dict(p.textures)
        if tex_override and p.name in tex_override:
            textures = dict(tex_override[p.name])
        elif not textures and p.image is not None and p.uv is not None:
            textures = {"diffuse": p.image}
        key = f"{name}_mat"
        mat = mats.get(key, p.color, p.roughness, p.metallic, textures if p.uv is not None else None)
        prim = write_mesh(stage, f"{parent}/{name}", p, mat)
        if collision is not None:
            add_collision(prim, collision)
        prims.append(prim)
    return prims


def polyhaven(asset: str) -> dict[str, Path]:
    d = POLY / asset
    return {"diffuse": d / f"{asset}_diff_2k.jpg", "normal": d / f"{asset}_nor_gl_2k.jpg",
            "roughness": d / f"{asset}_rough_2k.jpg", "occlusion": d / f"{asset}_ao_2k.jpg"}


def grid_rects(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Greedy cover of a boolean grid with axis-aligned rectangles (i0, i1, j0, j1), end-exclusive."""
    todo = mask.copy()
    rects = []
    for i in range(todo.shape[0]):
        j = 0
        while j < todo.shape[1]:
            if not todo[i, j]:
                j += 1
                continue
            j1 = j
            while j1 < todo.shape[1] and todo[i, j1]:
                j1 += 1
            i1 = i + 1
            while i1 < todo.shape[0] and todo[i1, j:j1].all():
                i1 += 1
            todo[i:i1, j:j1] = False
            rects.append((i, i1, j, j1))
            j = j1
    return rects


# ----------------------------------------------------------------------------------------------------------------------
# assets
# ----------------------------------------------------------------------------------------------------------------------
# rov_bop_panel valves from uuv_gazebo_worlds/models/rov_bop_panel/model.sdf (link pose, mesh)
BOP_VALVES: list[tuple[str, tuple[float, float, float], float, str]] = [
    ("valve_1", (-0.15, -0.2, 0.18), 1.57, "Valve"),
    ("valve_2", (0.15, -0.2, 0.18), 1.57, "Valve"),
    ("valve_3", (-0.15, 0.0, 0.18), 1.57, "Valve"),
    ("valve_4", (0.0, -0.2, 0.14), 0.0, "SmallValve"),
    ("valve_5", (0.0, 0.0, 0.14), 0.0, "SmallValve"),
]
# measured from the meshes: disk (radius, height) and paddle box (size x, y, z; bottom z)
VALVE_GEOM: dict[str, dict[str, tuple[float, ...]]] = {
    "Valve": {"disk": (0.0725, 0.01), "paddle": (0.015, 0.10, 0.04, 0.01)},
    "SmallValve": {"disk": (0.0375, 0.02), "paddle": (0.05, 0.01, 0.04, 0.02)},
}
PANEL_SIZE: tuple[float, float, float] = (0.5, 0.8, 0.2)  # x, y, face height (z) of the panel body
VALVE_MASS: float = 0.5
PANEL_MASS: float = 10.0
VALVE_LIMITS_DEG: tuple[float, float] = (-90.0, 0.0)
VALVE_DAMPING: float = 0.5  # N m s / rad (SDF)


def build_panel(out_dir: Path) -> dict:
    usd = out_dir / "rov_bop_panel" / "rov_bop_panel.usda"
    stage, root = new_stage(usd, "rov_bop_panel")
    mats = Materials(stage, usd, "/rov_bop_panel/Looks")
    mesh_dir = UUV / "rov_bop_panel" / "meshes"

    # panel link (articulation root; the env fixes it to the world)
    panel = UsdGeom.Xform.Define(stage, "/rov_bop_panel/panel").GetPrim()
    set_pose(panel)
    UsdPhysics.RigidBodyAPI.Apply(panel)
    UsdPhysics.MassAPI.Apply(panel).CreateMassAttr(PANEL_MASS)
    UsdPhysics.ArticulationRootAPI.Apply(panel)
    parts = load_parts(mesh_dir / "BOP_Panel.dae")
    metal = polyhaven("rusty_painted_metal")
    for p in parts:
        p.uv = box_uv(p, tile=0.5)
    write_parts(stage, mats, "/rov_bop_panel/panel/visuals", parts, None,
                tex_override={p.name: metal for p in parts})
    # collision: body slab with square pockets around each valve (so fingers fit around the paddles) + handles
    res = 0.01
    nx, ny = int(round(PANEL_SIZE[0] / res)), int(round(PANEL_SIZE[1] / res))
    xs = -PANEL_SIZE[0] / 2 + (np.arange(nx) + 0.5) * res
    ys = -PANEL_SIZE[1] / 2 + (np.arange(ny) + 0.5) * res
    height = np.full((nx, ny), PANEL_SIZE[2])
    for _, (vx, vy, vz), _, kind in BOP_VALVES:
        half = VALVE_GEOM[kind]["disk"][0] + 0.01
        sel = (np.abs(xs[:, None] - vx) <= half) & (np.abs(ys[None, :] - vy) <= half)
        height[sel] = np.minimum(height[sel], vz)
    levels = sorted(set(np.unique(height).tolist()))
    z0, n_boxes = 0.0, 0
    for lv in levels:
        for i0, i1, j0, j1 in grid_rects(height >= lv - 1e-9):
            cx, cy = (xs[i0] + xs[i1 - 1]) / 2, (ys[j0] + ys[j1 - 1]) / 2
            collision_box(stage, f"/rov_bop_panel/panel/collisions/box_{n_boxes}", (cx, cy, (z0 + lv) / 2),
                          ((i1 - i0) * res, (j1 - j0) * res, lv - z0))
            n_boxes += 1
        z0 = lv
    for k, sy in enumerate((-0.375, 0.375)):  # carrying handles (posts + bar)
        for sx in (-0.2, 0.2):
            collision_box(stage, f"/rov_bop_panel/panel/collisions/handle{k}_post{int(sx > 0)}", (sx, sy, 0.28),
                          (0.02, 0.02, 0.16))
        collision_box(stage, f"/rov_bop_panel/panel/collisions/handle{k}_bar", (0.0, sy, 0.35), (0.42, 0.02, 0.02))

    joints = UsdGeom.Scope.Define(stage, "/rov_bop_panel/joints")
    valve_meta = []
    for name, pos, yaw, kind in BOP_VALVES:
        link = UsdGeom.Xform.Define(stage, f"/rov_bop_panel/{name}").GetPrim()
        q = rpy_quat((0.0, 0.0, yaw))
        set_pose(link, pos, q)
        UsdPhysics.RigidBodyAPI.Apply(link)
        r_disk, h_disk = VALVE_GEOM[kind]["disk"]
        px, py, pz, pz0 = VALVE_GEOM[kind]["paddle"]
        mass = UsdPhysics.MassAPI.Apply(link)
        mass.CreateMassAttr(VALVE_MASS)
        i_zz = 0.5 * VALVE_MASS * r_disk**2
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(i_zz / 2 + 1e-5, i_zz / 2 + 1e-5, i_zz))
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, h_disk / 2))
        vparts = load_parts(mesh_dir / f"{kind}.dae")
        write_parts(stage, mats, f"/rov_bop_panel/{name}/visuals", vparts, None, prefix=kind)
        collision_cylinder(stage, f"/rov_bop_panel/{name}/collisions/disk", (0, 0, h_disk / 2), r_disk, h_disk)
        collision_box(stage, f"/rov_bop_panel/{name}/collisions/paddle", (0, 0, pz0 + pz / 2), (px, py, pz))
        j = UsdPhysics.RevoluteJoint.Define(stage, f"/rov_bop_panel/joints/{name}_joint")
        j.CreateBody0Rel().SetTargets([panel.GetPath()])
        j.CreateBody1Rel().SetTargets([link.GetPath()])
        j.CreateAxisAttr("Z")
        j.CreateLocalPos0Attr(Gf.Vec3f(*pos))
        j.CreateLocalRot0Attr(q)
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLowerLimitAttr(VALVE_LIMITS_DEG[0])
        j.CreateUpperLimitAttr(VALVE_LIMITS_DEG[1])
        drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(VALVE_DAMPING * math.pi / 180.0)  # USD angular drives are per degree
        valve_meta.append({"name": name, "joint": f"{name}_joint", "kind": kind, "pos_panel": list(pos),
                           "yaw_rad": yaw, "paddle_size": [px, py, pz],
                           "paddle_center_z": pos[2] + pz0 + pz / 2, "disk_radius": r_disk})
    stage.GetRootLayer().Save()
    return {"usd": str(usd), "collision_boxes": n_boxes, "valves": valve_meta,
            "limits_deg": list(VALVE_LIMITS_DEG), "panel_size": list(PANEL_SIZE)}


def build_rigid_tool(out_dir: Path, name: str, mesh: Path, mass: float, com: tuple[float, float, float]) -> dict:
    usd = out_dir / name / f"{name}.usda"
    stage, root = new_stage(usd, name)
    mats = Materials(stage, usd, f"/{name}/Looks")
    UsdPhysics.RigidBodyAPI.Apply(root)
    m = UsdPhysics.MassAPI.Apply(root)
    m.CreateMassAttr(mass)
    m.CreateCenterOfMassAttr(Gf.Vec3f(*com))
    parts = load_parts(mesh)
    write_parts(stage, mats, f"/{name}/visuals", parts, "convexHull")
    stage.GetRootLayer().Save()
    allv = np.concatenate([p.vertices for p in parts])
    return {"usd": str(usd), "mass": mass, "extent": (allv.max(0) - allv.min(0)).round(3).tolist()}


def build_frame(out_dir: Path) -> dict:
    """DAVE electrical_panel: skid frame (static, simplified collision mesh) + electrical socket fixture."""
    usd = out_dir / "subsea_frame" / "subsea_frame.usda"
    stage, root = new_stage(usd, "subsea_frame")
    mats = Materials(stage, usd, "/subsea_frame/Looks")
    d = DAVE / "electrical_panel"
    yaw = tf_matrix(rpy=(0, 0, math.pi / 2))  # SDF visual/collision pose
    vis = load_parts(d / "meshes" / "Underwater-Frame.dae", yaw)
    write_parts(stage, mats, "/subsea_frame/visuals", vis, None)
    col = load_parts(d / "meshes" / "COLLISION-Underwater-Frame.dae", yaw)
    for prim in write_parts(stage, mats, "/subsea_frame/collisions", col, "none"):
        UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.guide)
    # socket link pose from the SDF (m_to_f_socket in electrical_panel/model.sdf), mesh scaled 0.12, yaw -90 deg
    sock_tf = tf_matrix((0.475, 0.31, 1.315), (0, -math.pi / 4, 0)) @ tf_matrix(rpy=(0, 0, -math.pi / 2), scale=0.12)
    sock = load_parts(d / "meshes" / "Electrical-Socket.dae", sock_tf)
    write_parts(stage, mats, "/subsea_frame/socket", sock, "convexHull", prefix="socket_")
    stage.GetRootLayer().Save()
    allv = np.concatenate([p.vertices for p in vis])
    return {"usd": str(usd), "extent": (allv.max(0) - allv.min(0)).round(3).tolist(),
            "bounds_min": allv.min(0).round(3).tolist(), "bounds_max": allv.max(0).round(3).tolist()}


def build_plug(out_dir: Path) -> dict:
    usd = out_dir / "wet_mate_plug" / "wet_mate_plug.usda"
    stage, root = new_stage(usd, "wet_mate_plug")
    mats = Materials(stage, usd, "/wet_mate_plug/Looks")
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(2.0)
    tf = tf_matrix(rpy=(math.pi / 2, 0, math.pi / 2), scale=0.4)  # SDF visual pose + scale
    d = DAVE / "f_to_m_plug" / "meshes"
    write_parts(stage, mats, "/wet_mate_plug/visuals", load_parts(d / "Plug.dae", tf), None)
    col = load_parts(d / "COLLISION-Plug.dae", tf)
    for prim in write_parts(stage, mats, "/wet_mate_plug/collisions", col, "convexDecomposition"):
        UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.guide)
    stage.GetRootLayer().Save()
    return {"usd": str(usd)}


def build_rocks(out_dir: Path) -> dict:
    """Ocean Rock 01-07 normalised so the largest extent is 1 m (scale per instance in the scene)."""
    info = {}
    for k in range(1, 8):
        src = FUEL / f"ocean_rock_{k:02d}" / "model"
        dae = next((src / "meshes").glob("*.dae"))
        raw = load_parts(dae)
        allv = np.concatenate([p.vertices for p in raw])
        s = 1.0 / float((allv.max(0) - allv.min(0)).max())
        center = (allv.max(0) + allv.min(0)) / 2
        tf = tf_matrix(scale=s) @ tf_matrix(pos=tuple((-center * [1, 1, 0] - [0, 0, allv[:, 2].min()]).tolist()))
        parts = load_parts(dae, tf)
        tex = {"diffuse": src / "materials/textures/Rock_Albedo.png", "normal": src / "materials/textures/Rock_Normal.png",
               "roughness": src / "materials/textures/Rock_Roughness.png"}
        for p in parts:
            p.textures = dict(tex)
        name = f"ocean_rock_{k:02d}"
        usd = out_dir / name / f"{name}.usda"
        stage, _ = new_stage(usd, name)
        mats = Materials(stage, usd, f"/{name}/Looks")
        write_parts(stage, mats, f"/{name}/visuals", parts, "convexHull")
        stage.GetRootLayer().Save()
        v = np.concatenate([p.vertices for p in parts])
        info[name] = {"usd": str(usd), "extent": (v.max(0) - v.min(0)).round(3).tolist()}
    return info


SHIPWRECK_SITE_XY: tuple[float, float] = (-52.0, -166.0)  # flat sand ~15 m south-west of a wreck (DAE units x 0.01)
SHIPWRECK_CROP_HALF: float = 90.0


def ground_height(parts: list[Part], x: float, y: float) -> float:
    """Height of the highest triangle above (x, y) (vertical ray cast)."""
    best = -np.inf
    for p in parts:
        mesh = trimesh.Trimesh(p.vertices, p.faces, process=False)
        locs, _, _ = mesh.ray.intersects_location([[x, y, 1e4]], [[0, 0, -1]])
        if len(locs):
            best = max(best, float(locs[:, 2].max()))
    return best


def build_shipwreck(out_dir: Path) -> dict:
    src = FUEL / "ocean_floor_shipwreck" / "model"
    tx = src / "materials" / "textures"
    tex = {
        "Rock_lib": {"diffuse": tx / "Rock_Albedo.png", "normal": tx / "Rock_Normal.png", "roughness": tx / "Rock_Roughness.png"},
        "Coral03_lib": {"diffuse": tx / "Coral03_Albedo.png", "normal": tx / "Coral03_Normal.png", "roughness": tx / "Coral03_Roughness.png"},
        "Coral02_lib": {"diffuse": tx / "Coral02_Albedo.png", "normal": tx / "Coral02_Normal.png", "roughness": tx / "Coral02_Roughness.png"},
        "SunkenShip_lib": {"diffuse": tx / "SunkenShip_Albedo.png", "normal": tx / "SunkenShip_Normal.png",
                           "roughness": tx / "SunkenShip_Roughness.png", "metallic": tx / "SunkenShip_Metalness.png"},
    }
    parts = load_parts(src / "meshes" / "underwaterworld.dae", tf_matrix(scale=0.01))  # DAE unit: centimeter
    sx, sy = SHIPWRECK_SITE_XY
    lo = np.array([sx - SHIPWRECK_CROP_HALF, sy - SHIPWRECK_CROP_HALF, -np.inf])
    hi = np.array([sx + SHIPWRECK_CROP_HALF, sy + SHIPWRECK_CROP_HALF, np.inf])
    parts = [c for c in (crop_part(p, lo, hi) for p in parts) if c is not None]
    ground = [p for p in parts if p.name == "Ground_lib"]
    z_site = ground_height(ground, sx, sy)
    offset = np.array([sx, sy, z_site])
    for p in parts:
        p.vertices = p.vertices - offset
        p.textures = dict(tex.get(p.name, {}))
        if p.name == "Ground_lib":
            # 2 m cells and UVs spanning the whole 1 km world: subdivide to 0.5 m, add a few cm of ripples so the
            # close-up seabed is not faceted, and tile CC0 sand every 2.5 m
            subdivide(p, 2)
            add_micro_relief(p, amplitude=0.04, seed=7)
            level_pad(p, *L.PAD_RADII)
            p.uv = p.vertices[:, :2] / 2.5
            p.textures = polyhaven("coast_sand_01")
    usd = out_dir / "shipwreck_site" / "shipwreck_site.usda"
    write_terrain_usd(usd, "shipwreck_site", parts, "Ground_lib")
    write_terrain_usd(usd.with_name("shipwreck_site_compact.usda"), "shipwreck_site_compact", compact_parts(parts),
                      "Ground_lib", tiles=2)
    ship = next(p for p in parts if p.name == "SunkenShip_lib")
    return {"usd": str(usd), "site_source_xyz": offset.round(3).tolist(),
            "faces": {p.name: int(len(p.faces)) for p in parts},
            "wreck_bounds_min": ship.vertices.min(0).round(2).tolist(),
            "wreck_bounds_max": ship.vertices.max(0).round(2).tolist()}


LEDGE_HALF: int = 125  # cells (~1 m) each side of the tile centre


def build_ledge(out_dir: Path) -> dict:
    nc = FUEL / "portuguese_ledge" / "model" / "meshes" / "PortugueseLedgeTile1_DecDeg.nc"
    with h5py.File(nc, "r") as f:
        z = np.asarray(f["z"][:], dtype=np.float64)
        lon = np.asarray(f["_nc4_non_coord_lon"][:])
        lat = np.asarray(f["_nc4_non_coord_lat"][:])
    dx = (lon[1] - lon[0]) * 111320.0 * math.cos(math.radians(lat.mean()))
    dy = (lat[1] - lat[0]) * 110540.0
    ci, cj = z.shape[0] // 2, z.shape[1] // 2
    sub = z[ci - LEDGE_HALF: ci + LEDGE_HALF + 1, cj - LEDGE_HALF: cj + LEDGE_HALF + 1]  # [lat, lon]
    n = sub.shape[0]
    jj, ii = np.meshgrid(np.arange(n), np.arange(n))
    xs, ys = (jj - LEDGE_HALF) * dx, (ii - LEDGE_HALF) * dy
    verts = np.stack([xs.ravel(), ys.ravel(), (sub - sub[LEDGE_HALF, LEDGE_HALF]).ravel()], axis=1)
    idx = np.arange(n * n).reshape(n, n)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    faces = np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])
    part = Part("seabed", verts, faces)
    subdivide(part, 1)
    add_micro_relief(part, amplitude=0.03, seed=11)
    level_pad(part, *L.PAD_RADII)
    part.uv = part.vertices[:, :2] / 2.5
    part.textures = polyhaven("coral_mud_01")
    usd = out_dir / "portuguese_ledge" / "portuguese_ledge.usda"
    write_terrain_usd(usd, "portuguese_ledge", [part], "seabed")
    write_terrain_usd(usd.with_name("portuguese_ledge_compact.usda"), "portuguese_ledge_compact",
                      compact_parts([part]), "seabed", tiles=2)
    return {"usd": str(usd), "cell_m": [round(dx, 3), round(dy, 3)], "size_m": [round(n * dx, 1), round(n * dy, 1)],
            "relief_m": round(float(np.ptp(sub)), 2), "source_depth_m": round(float(sub[LEDGE_HALF, LEDGE_HALF]), 2)}


def usd_mesh(usd: Path, prim_path: str | None = None) -> trimesh.Trimesh:
    """Read every (or one) triangle mesh of a USD file into one trimesh, in the file's root frame."""
    stage = Usd.Stage.Open(str(usd))
    cache = UsdGeom.XformCache()
    meshes = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or (prim_path is not None and str(prim.GetPath()) != prim_path):
            continue
        m = UsdGeom.Mesh(prim)
        pts = np.asarray(m.GetPointsAttr().Get(), dtype=np.float64)
        idx = np.asarray(m.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 3)
        tf = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64).T
        meshes.append(trimesh.Trimesh(trimesh.transform_points(pts, tf), idx, process=False))
    return trimesh.util.concatenate(meshes)


def top_hit(mesh: trimesh.Trimesh, x: float, y: float) -> float:
    locs, _, _ = mesh.ray.intersects_location([[x, y, 100.0]], [[0, 0, -1]])
    return float(locs[:, 2].max()) if len(locs) else 0.0


def build_work_site(scene: str, compact: bool = False) -> dict:
    """Static work-site layout: terrain + skid frame + panel mounting plate + rocks + deck tools (references).

    ``compact``: the terrain is the +-25 m crop (``<scene>_compact.usda``), so the site can be instanced per env in
    parallel scenes; written as ``<scene>_work_site_compact.usda``."""
    sdir = SCENE_DIR / scene
    terrain = f"{scene}_compact" if compact else scene
    usd = sdir / (f"{scene}_work_site_compact.usda" if compact else f"{scene}_work_site.usda")
    if usd.exists():
        usd.unlink()
    stage, root = new_stage(usd, usd.stem, kind="assembly")
    rp = root.GetPath()
    mats = Materials(stage, usd, f"{rp}/Looks")
    ground_name = {"shipwreck_site": "Ground_lib", "portuguese_ledge": "seabed"}[scene]
    ground = usd_mesh(sdir / f"{terrain}.usda", f"/{terrain}/terrain/{ground_name}")
    stage.DefinePrim(f"{rp}/terrain").GetReferences().AddReference(f"./{terrain}.usda")

    frame = stage.DefinePrim(f"{rp}/subsea_frame")
    frame.GetReferences().AddReference("../../objects/subsea_frame/subsea_frame.usda")
    set_pose(frame, L.FRAME_POS)
    frame_col = usd_mesh(OBJECT_DIR / "subsea_frame" / "subsea_frame.usda")

    # panel mounting plate + two posts down to the seabed (simple steel, static)
    paint = mats.get("mount_paint", (0.62, 0.52, 0.18), 0.65, 0.2)
    px, py, pz = L.PANEL_POS
    sx, sy, sz = L.MOUNT_PLATE_SIZE
    mount = UsdGeom.Xform.Define(stage, f"{rp}/panel_mount").GetPrim()
    boxes = {"plate": ((px + sx / 2, py, pz), (sx, sy, sz))}
    for k, s in enumerate((-1, 1)):
        h = pz + sz / 2
        boxes[f"post_{k}"] = ((px + sx / 2, py + s * L.MOUNT_POST_Y_OFFSET, h / 2), (*L.MOUNT_POST_SIZE, h))
    for name, (c, size) in boxes.items():
        cube = UsdGeom.Cube.Define(stage, f"{mount.GetPath()}/{name}")
        cube.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(cube)
        xf.AddTranslateOp().Set(Gf.Vec3d(*c))
        xf.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(paint)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    rocks = []
    for k, (rid, x, y, yaw, size) in enumerate(L.ROCKS):
        name = f"ocean_rock_{rid:02d}"
        ext = usd_mesh(OBJECT_DIR / name / f"{name}.usda").extents
        z = top_hit(ground, x, y) - L.ROCK_EMBED * size * float(ext[2])
        prim = stage.DefinePrim(f"{rp}/rocks/rock_{k}")
        prim.GetReferences().AddReference(f"../../objects/{name}/{name}.usda")
        set_pose(prim, (x, y, z), rpy_quat((0, 0, math.radians(yaw))), scale=size)
        rocks.append({"rock": name, "pos": [x, y, round(z, 3)], "size": size})

    tools = []
    for name, x, y, yaw in L.DECK_TOOLS:
        mesh = usd_mesh(OBJECT_DIR / name / f"{name}.usda")
        z = top_hit(frame_col, x, y) - float(mesh.bounds[0, 2]) + 0.003
        prim = stage.DefinePrim(f"{rp}/tools/{name}")
        prim.GetReferences().AddReference(f"../../objects/{name}/{name}.usda")
        set_pose(prim, (x, y, z), rpy_quat((0, 0, math.radians(yaw))))
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(False)  # static props
        tools.append({"tool": name, "pos": [x, y, round(z, 3)]})
    stage.GetRootLayer().Save()
    corners = [top_hit(ground, cx, cy) for cx in (-0.95, 0.95) for cy in (-1.55, 1.38)]
    return {"usd": str(usd), "ground_z_at_frame_corners": [round(c, 3) for c in corners], "rocks": rocks,
            "tools": tools, "ground_z_under_rov": round(top_hit(ground, *L.ROV_STATION_POS[:2]), 3)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    choices = ["panel", "tools", "frame", "rocks", "shipwreck", "ledge", "site"]
    ap.add_argument("--only", nargs="*", default=None, choices=choices)
    args = ap.parse_args()
    todo = args.only or choices
    report_path = ASSETS_ROOT / "third_party" / "build_report.json"
    report: dict = json.loads(report_path.read_text()) if report_path.exists() else {}
    if "panel" in todo:
        report["rov_bop_panel"] = build_panel(OBJECT_DIR)
    if "tools" in todo:
        report["hot_stab"] = build_rigid_tool(OBJECT_DIR, "hot_stab", UUV / "hot_stab/meshes/HotStab.dae", 4.4,
                                              (0.160875, 0, 0))
        report["torque_tool"] = build_rigid_tool(OBJECT_DIR, "torque_tool", UUV / "torque_tool/meshes/TorqueTool.dae",
                                                 3.0, (0.12775, 0, 0))
        report["wet_mate_plug"] = build_plug(OBJECT_DIR)
    if "frame" in todo:
        report["subsea_frame"] = build_frame(OBJECT_DIR)
    if "rocks" in todo:
        report["rocks"] = build_rocks(OBJECT_DIR)
    if "shipwreck" in todo:
        report["shipwreck_site"] = build_shipwreck(SCENE_DIR)
    if "ledge" in todo:
        report["portuguese_ledge"] = build_ledge(SCENE_DIR)
    if "site" in todo:
        for scene in L.WORK_SITE_SCENES:
            report[f"{scene}_work_site"] = build_work_site(scene)
            report[f"{scene}_work_site_compact"] = build_work_site(scene, compact=True)
    report_path.write_text(json.dumps(report, indent=1))
    summary = {k: report[k] for k in report if any(k.startswith(t) for t in todo) and k != "rov_bop_panel"}
    print(json.dumps(summary, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
