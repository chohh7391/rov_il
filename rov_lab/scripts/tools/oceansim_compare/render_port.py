"""Render the shared comparison scene with the rov_lab port (Isaac Lab 3.0 / Isaac Sim 6.2).

Wraps the camera prim authored in the shared USD (no respawn, so pose and intrinsics are the file's) with
``UWCameraCfg`` and applies ``uw_render`` with the default (OceanSim) water parameters.

    PYTHONPATH=rov_lab/source OMNI_KIT_ACCEPT_EULA=YES <isaaclab-3.0-python> rov_lab/scripts/tools/oceansim_compare/render_port.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.app import add_launcher_args, launch_simulation

REPO = Path(__file__).resolve().parents[4]
parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--scene", type=Path, default=REPO / "runs" / "oceansim_compare" / "compare_scene.usda")
parser.add_argument("--out", type=Path, default=REPO / "runs" / "oceansim_compare" / "port")
parser.add_argument("--warmup", type=int, default=60)
parser.add_argument("--ambient_light_intensity", type=float, default=None,
                    help="override RTX ambient light (Isaac Lab's rendering experience sets 1.0; Isaac Sim default is 0)")
add_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

DEVICE = "cuda:0"
with launch_simulation(sim_utils.SimulationCfg(dt=1 / 60, device=DEVICE), args):
    import json

    import numpy as np
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils import configclass
    from PIL import Image

    from rov_lab.assets.sensors import UWCameraCfg, uw_render

    import torch
    import isaaclab.utils.math as mu

    meta = json.loads((args.scene.parent / "compare_scene.json").read_text())
    cam_meta = meta["camera"]

    # Isaac Lab 3.0's RTX renderer reads cfg.spawn.clipping_range even when wrapping an existing camera prim
    # (spawn=None crashes), so spawn a camera with exactly the authored intrinsics and pose instead.
    eye = torch.tensor(cam_meta["eye"], dtype=torch.float64)
    f = torch.tensor(cam_meta["target"], dtype=torch.float64) - eye
    f = f / f.norm()
    r = torch.linalg.cross(f, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
    r = r / r.norm()
    u = torch.linalg.cross(r, f)
    R = torch.stack([r, u, -f], dim=1)  # columns: camera local x, y, z (USD/OpenGL camera looks along -z)
    quat_xyzw = mu.quat_from_matrix(R[None].float())[0].tolist()

    renderer_kwargs = {}
    if args.ambient_light_intensity is not None:
        from isaaclab_physx.renderers import IsaacRtxRendererCfg
        from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import IsaacRtxRendererGlobalSettingsCfg

        renderer_kwargs["renderer_cfg"] = IsaacRtxRendererCfg(
            global_settings=IsaacRtxRendererGlobalSettingsCfg(ambient_light_intensity=args.ambient_light_intensity)
        )

    @configclass
    class CompareSceneCfg(InteractiveSceneCfg):
        compare = AssetBaseCfg(prim_path="/World/Compare", spawn=sim_utils.UsdFileCfg(usd_path=str(args.scene)))
        cam = UWCameraCfg(
            prim_path="/World/port_cam",
            width=cam_meta["width"],
            height=cam_meta["height"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=cam_meta["focal_length"],
                horizontal_aperture=cam_meta["horizontal_aperture"],
                vertical_aperture=cam_meta["vertical_aperture"],
                clipping_range=tuple(cam_meta["clipping_range"]),
            ),
            offset=UWCameraCfg.OffsetCfg(pos=tuple(cam_meta["eye"]), rot=tuple(quat_xyzw), convention="opengl"),
            **renderer_kwargs,
        )

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 60, device=DEVICE))
    scene = InteractiveScene(CompareSceneCfg(num_envs=1, env_spacing=1.0))
    sim.reset()
    cam = scene["cam"]
    for _ in range(args.warmup):
        sim.step()
        scene.update(sim.get_physics_dt())
    out = cam.data.output
    rgb = out["rgb"].torch[:1]
    dist = out["distance_to_camera"].torch[0]
    dist = dist[..., 0] if dist.dim() == 3 else dist
    uw = uw_render(rgb, dist[None], cam.cfg)[0]
    args.out.mkdir(parents=True, exist_ok=True)
    K = cam.data.intrinsic_matrices.torch[0].cpu().numpy()
    np.savez(args.out / "port_raw.npz", rgb=rgb[0].cpu().numpy(), depth=dist.cpu().numpy(), K=K)
    Image.fromarray(uw.cpu().numpy()).save(args.out / "UW_image_port.png")
    d = dist.cpu().numpy()
    print(f"[port] rgb {tuple(rgb.shape)} {rgb.dtype}, depth {d.shape} finite {np.isfinite(d).mean():.4f} "
          f"range [{np.nanmin(d[np.isfinite(d)]):.3f}, {np.nanmax(d[np.isfinite(d)]):.3f}]")
    print(f"[port] K=\n{K}\n[port] cam pos={cam.data.pos_w.torch[0].tolist()} quat_opengl={quat_xyzw} -> {args.out}")
