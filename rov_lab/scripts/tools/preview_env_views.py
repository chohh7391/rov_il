"""Render a rov_lab task from external viewpoints (plus its own cameras) after a reset, for visual checks.

    PYTHONPATH=rov_lab/source OMNI_KIT_ACCEPT_EULA=YES <isaaclab-3.0-python> rov_lab/scripts/tools/preview_env_views.py \\
        --task ROVLab-I90Bravo7-ValvePanel-v0 --out runs/work_site/env_views
"""

from __future__ import annotations

import argparse
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", type=str, default="ROVLab-I90Bravo7-ValvePanel-v0")
parser.add_argument("--out", type=Path, default=Path("runs/work_site/env_views"))
parser.add_argument("--steps", type=int, default=20)
add_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

device: str = getattr(args, "device", None) or "cuda:0"
with launch_simulation(sim_utils.SimulationCfg(device=device), args):
    import math

    import gymnasium as gym
    import numpy as np
    import torch
    from PIL import Image
    from isaaclab_tasks.utils import parse_env_cfg

    import rov_lab.tasks  # noqa: F401
    from rov_lab.assets.sensors import UWCameraCfg, uw_render

    def look_at_xyzw(eye, target) -> tuple[float, float, float, float]:
        f = np.asarray(target, float) - np.asarray(eye, float)
        f /= np.linalg.norm(f)
        left = np.cross([0.0, 0.0, 1.0], f)
        left /= np.linalg.norm(left)
        up = np.cross(f, left)
        r = np.stack([f, left, up], axis=1)
        w = math.sqrt(max(1e-12, 1.0 + np.trace(r))) / 2.0
        return ((r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w), (r[1, 0] - r[0, 1]) / (4 * w), w)

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=1)
    env_cfg.use_teleop_device("keyboard")
    rp = env_cfg.scene.robot.init_state.pos
    views = {
        "side": ((rp[0] + 0.5, rp[1] - 2.6, rp[2] + 0.5), (rp[0] + 0.6, rp[1], rp[2] - 0.1)),
        "rear_high": ((rp[0] - 2.2, rp[1] - 1.2, rp[2] + 1.6), (rp[0] + 1.0, rp[1], rp[2] - 0.2)),
        "front_low": ((rp[0] + 1.6, rp[1] + 1.6, rp[2] - 0.2), (rp[0] + 0.4, rp[1], rp[2] - 0.1)),
    }
    for name, (eye, tgt) in views.items():
        setattr(env_cfg.scene, f"view_{name}", UWCameraCfg(
            prim_path=f"{{ENV_REGEX_NS}}/view_{name}", update_period=0.0, width=800, height=600,
            offset=UWCameraCfg.OffsetCfg(pos=eye, rot=look_at_xyzw(eye, tgt), convention="world"),
            spawn=sim_utils.PinholeCameraCfg(focal_length=14.0, horizontal_aperture=20.955, clipping_range=(0.05, 200.0)),
        ))
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    obs, _ = env.reset()
    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    with torch.inference_mode():
        for _ in range(args.steps):
            obs, *_ = env.step(actions)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in views:
        cam = env.scene[f"view_{name}"]
        rgb = cam.data.output["rgb"].torch[..., :3]
        uw = uw_render(rgb, cam.data.output["distance_to_camera"].torch, cam.cfg)
        Image.fromarray(rgb[0].cpu().numpy().astype(np.uint8)).save(args.out / f"{name}_rgb.png")
        Image.fromarray(uw[0].cpu().numpy()).save(args.out / f"{name}_uw.png")
    for key, value in obs["policy"].items():
        if value.dtype == torch.uint8 and value.dim() == 4:
            Image.fromarray(value[0].cpu().numpy()).save(args.out / f"obs_{key}.png")
    print(f"[views] saved to {args.out}")
    env.close()
