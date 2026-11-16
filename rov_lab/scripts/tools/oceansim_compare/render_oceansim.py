"""Render the shared comparison scene with OceanSim's own ``UW_Camera`` (Isaac Sim 5.1).

Saves OceanSim's underwater image (written by ``UW_Camera`` itself) and the raw ``LdrColor`` / ``distance_to_camera``
annotator outputs it consumed. Run with the Isaac Sim 5.1 environment (``.venv-rov_lab``):

    OMNI_KIT_ACCEPT_EULA=YES .venv-rov_lab/bin/python rov_lab/scripts/tools/oceansim_compare/render_oceansim.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
OCEANSIM_ROOT = REPO / "external_dependencies" / "OceanSim"

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--scene", type=Path, default=REPO / "runs" / "oceansim_compare" / "compare_scene.usda")
parser.add_argument("--out", type=Path, default=REPO / "runs" / "oceansim_compare" / "oceansim")
parser.add_argument("--warmup", type=int, default=60)
parser.add_argument("--frames", type=int, default=3)
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": 640, "height": 480})

import json  # noqa: E402

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.sensors.camera")
app.update()

# make OceanSim's python package importable as ``isaacsim.oceansim`` without enabling its UI extension
import isaacsim  # noqa: E402

isaacsim.__path__.append(str(OCEANSIM_ROOT / "isaacsim"))

ctx = omni.usd.get_context()
ctx.open_stage(str(args.scene))
for _ in range(200):
    app.update()
    if not ctx.is_new_stage() and ctx.get_stage_loading_status()[2] == 0:
        break

from isaacsim.core.api import World  # noqa: E402
from isaacsim.oceansim.sensors.UW_Camera import UW_Camera  # noqa: E402

meta = json.loads((args.scene.parent / "compare_scene.json").read_text())
cam_meta = meta["camera"]
world = World(stage_units_in_meters=1.0)
cam = UW_Camera(prim_path=cam_meta["prim_path"], resolution=(cam_meta["width"], cam_meta["height"]))
world.reset()
args.out.mkdir(parents=True, exist_ok=True)
cam.initialize(viewport=False, writing_dir=str(args.out))  # OceanSim default water parameters

import omni.replicator.core as rep  # noqa: E402


def annot_ready() -> bool:
    d = cam._rgba_annot.get_data()
    return getattr(d, "size", 0) != 0 and getattr(cam._depth_annot.get_data(), "size", 0) != 0


# RTX needs a number of frames after startup before annotators deliver data (headless)
for i in range(600):
    world.step(render=True)
    if i >= args.warmup and annot_ready():
        break
    if i >= 200 and i % 50 == 0:
        rep.orchestrator.step(rt_subframes=4, pause_timeline=False)
print(f"[oceansim] annotators ready={annot_ready()} after {i + 1} steps", flush=True)
for k in range(args.frames):
    world.step(render=True)
    if k == args.frames - 1:
        # grab the raw inputs of the same frame that UW_Camera.render() processes (no app update in between)
        rgba = cam._rgba_annot.get_data()
        depth = cam._depth_annot.get_data()
        rgba = (rgba.numpy() if hasattr(rgba, "numpy") else np.asarray(rgba)).copy()
        depth = (depth.numpy() if hasattr(depth, "numpy") else np.asarray(depth)).copy()
    cam.render()
for _ in range(5):  # let the replicator backend flush the written images
    app.update()
K = cam.get_intrinsics_matrix()
np.savez(args.out / "oceansim_raw.npz", rgba=rgba, depth=depth, K=np.asarray(K))
print(f"[oceansim] rgba {rgba.shape} {rgba.dtype}, depth {depth.shape} range [{np.nanmin(depth):.3f}, {np.nanmax(depth):.3f}]")
print(f"[oceansim] K=\n{np.asarray(K)}")
print(f"[oceansim] wrote {args.out}")
app.close()
