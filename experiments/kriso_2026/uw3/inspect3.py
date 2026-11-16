"""Isaac Lab 3.0: spawn the underwater scene, print geometry, save camera images."""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="/home/home/rov_il/runs/kriso_2026/inspect3")
parser.add_argument("--rov", type=float, nargs=3, default=[-2.0, 0.0, -0.8])
add_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

import os
import sys
import time

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.utils import instantiate

sys.path.insert(0, os.path.dirname(__file__))

sim_cfg = sim_utils.SimulationCfg(dt=1 / 60, device="cuda:0")
with launch_simulation(sim_cfg, args):
    import uw_scene as U
    from PIL import Image

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    sim = sim_utils.SimulationContext(sim_cfg)
    scene = instantiate(U.make_scene_cfg(rov_start=tuple(args.rov))(num_envs=1, env_spacing=8.0))
    sim.reset()
    print(f"[i3] setup {time.time()-t0:.1f}s", flush=True)
    r = scene["robot"]
    print("[i3] joints", r.joint_names)
    print("[i3] bodies", r.body_names)
    print("[i3] num_base_dofs", r.num_base_dofs, "fixed", r.is_fixed_base)
    print("[i3] limits", np.round(r.data.soft_joint_pos_limits.torch[0].cpu().numpy(), 3).tolist())
    dt = sim.get_physics_dt()

    def step(n, jt=None):
        for _ in range(n):
            if jt is not None:
                r.set_joint_position_target_index(target=jt)
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)

    def dump(tag):
        bp = r.data.body_pos_w.torch[0].cpu().numpy()
        print(f"[i3] --- {tag}")
        for n, p in zip(r.body_names, bp):
            print(f"[i3]   {n:16s} {np.round(p, 3)}")
        print("[i3]   root quat", np.round(r.data.root_quat_w.torch[0].cpu().numpy(), 3))
        for c in U.CUBES:
            print(f"[i3]   {c:16s} {np.round(scene[c].data.root_pos_w.torch[0].cpu().numpy(), 3)}")

    step(10)
    dump("initial (q=0)")
    for cam in ("front", "wrist"):
        out = scene[cam].data.output
        rgb, dep = U.as_torch(out["rgb"]), U.as_torch(out["distance_to_camera"])
        print(f"[i3] {cam} rgb {tuple(rgb.shape)} {rgb.dtype} depth {tuple(dep.shape)} min/max {float(torch.nan_to_num(dep, posinf=-1).max()):.2f}")
        Image.fromarray(rgb[0, ..., :3].cpu().numpy()).save(f"{args.out}/{cam}_rgb.png")
        Image.fromarray(U.uw_render(rgb, dep)[0].cpu().numpy()).save(f"{args.out}/{cam}_uw.png")
    # arm probes
    q0 = r.data.joint_pos.torch.clone()
    for name, idx, val in [("shoulder_lift+0.8", 1, 0.8), ("elbow_flex+0.8", 2, 0.8), ("wrist_flex+0.8", 3, 0.8)]:
        q = q0.clone(); q[0, idx] = val
        step(90, q)
        dump(name)
    q = q0.clone(); q[0, 5] = 1.0
    step(60, q); print("[i3] gripper q after open target 1.0:", r.data.joint_pos.torch[0, 5].item())
    t = time.time(); step(120, q); print(f"[i3] {120/(time.time()-t):.1f} steps/s with 2 cameras")
    for cam in ("front", "wrist"):
        out = scene[cam].data.output
        Image.fromarray(U.cam_images(scene[cam])[0].cpu().numpy()).save(f"{args.out}/{cam}_uw_end.png")
    print("[i3] done", flush=True)
