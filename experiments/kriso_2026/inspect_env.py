"""Launch the env once and print geometry / action layout; save camera snapshots."""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="runs/kriso_2026/inspect")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(vars(args)).app

import time

import numpy as np
import torch
from PIL import Image

import sim_common as sc

os.makedirs(args.out, exist_ok=True)
t0 = time.time()
env = sc.make_env(seed=0)
obs, _ = env.reset()
print("[inspect] env ready in %.1fs" % (time.time() - t0))
print("[inspect] step_dt", env.step_dt, "physics_dt", env.physics_dt, "max_episode_length", env.max_episode_length)
am = env.action_manager
print("[inspect] action dim", am.total_action_dim, {n: t.action_dim for n, t in am._terms.items()})
print("[inspect] obs keys", {k: tuple(v.shape) for k, v in obs["policy"].items()})
robot = env.scene["robot"]
print("[inspect] joint names", robot.joint_names)
print("[inspect] joint pos", robot.data.joint_pos[0].cpu().numpy().round(3))
print("[inspect] joint limits", robot.data.soft_joint_pos_limits[0].cpu().numpy().round(3).tolist())
print("[inspect] body names", robot.body_names)
print("[inspect] root pos", robot.data.root_pos_w[0].cpu().numpy().round(3), "quat", robot.data.root_quat_w[0].cpu().numpy().round(3))
ee = env.scene["ee_frame"]
print("[inspect] ee source pos", ee.data.source_pos_w[0].cpu().numpy().round(3))
print("[inspect] gripper/jaw pos", ee.data.target_pos_w[0].cpu().numpy().round(3))
for c in sc.CUBES:
    print("[inspect]", c, env.scene[c].data.root_pos_w[0].cpu().numpy().round(3))
print("[inspect] target id", sc.target_id(env))
Image.fromarray(sc.to_uint8_img(obs["policy"]["front"])).save(f"{args.out}/front0.png")
Image.fromarray(sc.to_uint8_img(obs["policy"]["wrist"])).save(f"{args.out}/wrist0.png")

def jaw():
    return ee.data.target_pos_w[0, 1].cpu().numpy()

# timing + probes
probes = [
    ("rov +x", [1, 0, 0, 0, 0, 0] + [0] * 8, 60),
    ("hold", [0] * 14, 60),
    ("ik +z", [0] * 6 + [0, 0, 0.005, 0, 0, 0] + [0, 0], 40),
    ("ik +x", [0] * 6 + [0.005, 0, 0, 0, 0, 0] + [0, 0], 40),
    ("grip open", [0] * 12 + [0, 0.03], 40),
    ("pan +", [0] * 12 + [0.01, 0], 30),
]
for name, a, n in probes:
    t = time.time()
    p0, r0, q0 = jaw(), robot.data.root_pos_w[0].cpu().numpy(), robot.data.joint_pos[0].cpu().numpy()
    act = torch.tensor([a], dtype=torch.float32, device=env.device)
    for _ in range(n):
        obs, *_ = env.step(act)
    dt = (time.time() - t) / n
    print(f"[probe] {name:10s} {1/dt:5.1f} steps/s  d_jaw={np.round(jaw()-p0,3)}  d_root={np.round(robot.data.root_pos_w[0].cpu().numpy()-r0,3)}  d_q={np.round(robot.data.joint_pos[0].cpu().numpy()-q0,3)}")
Image.fromarray(sc.to_uint8_img(obs["policy"]["front"])).save(f"{args.out}/front1.png")
Image.fromarray(sc.to_uint8_img(obs["policy"]["wrist"])).save(f"{args.out}/wrist1.png")
print("[inspect] state vec", sc.policy_state(obs).round(3))
env.close()
app.close()
