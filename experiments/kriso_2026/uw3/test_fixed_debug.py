"""Debug trace of the scripted expert in fixed-base mode: phase, tracking error, gripper and cube state."""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--jitter", type=float, nargs=4, default=[0.0, 0.0, 0.0, 0.0])
parser.add_argument("--targets", type=int, nargs="+", default=[1, 2])
parser.add_argument("--seed", type=int, default=5)
add_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

import os
import sys

import torch

import isaaclab.sim as sim_utils

sys.path.insert(0, os.path.dirname(__file__))
sim_cfg = sim_utils.SimulationCfg(dt=1 / 60, device="cuda:0")

with launch_simulation(sim_cfg, args):
    import uw_env as E
    from uw_expert import UWExpert

    env = E.UWPickEnv(sim_cfg, seed=args.seed, fixed_base=True, station_jitter=tuple(args.jitter), episode_s=10.0)
    env.reset(target=0)
    ex = UWExpert(env)
    ex.calibrate()
    for t in args.targets:
        env.reset(target=t)
        ex.reset()
        rp, rq = env.root_pose()
        print(f"[dbg] target={t} root={rp.tolist()} quat={rq.tolist()} cube={env.cube_pos().tolist()}", flush=True)
        last = None
        while True:
            a = ex.act()
            _, succ, done = env.step(a)
            if ex.phase != last or env.k % 10 == 0:
                gp = ex.grasp_point()
                d = torch.linalg.vector_norm(env.cube_pos() - env.jaw_point()).item()
                rise = (env.cube_pos()[2] - env.spawn_z[env.target]).item()
                print(f"[dbg] k={env.k:3d} phase={ex.phase:8s} pk={ex.pk:3d} err={ex.err:.4f} grip={env.joint_pos()[5].item():.3f} "
                      f"jaw-cube={d:.3f} rise={rise:+.3f} q={[round(x, 2) for x in env.joint_pos().tolist()]} "
                      f"cmd={[round(float(x), 2) for x in a[4:10]]}", flush=True)
                last = ex.phase
            if done:
                print(f"[dbg] target={t} success={succ} steps={env.k}", flush=True)
                break
