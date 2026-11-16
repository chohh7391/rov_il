"""Collect underwater pick-and-place demonstrations with the scripted expert (Isaac Lab 3.0).

Saves only successful episodes as npz: front (90x160), wrist (120x160) underwater camera images,
state (10), action (10), task string. DART-style noise is injected into the *executed* action
while the clean expert action is recorded.
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=80)
parser.add_argument("--max_attempts", type=int, default=200)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="/home/home/rov_il/datasets/kriso_2026/demos")
parser.add_argument("--noise_rov", type=float, default=0.05)
parser.add_argument("--noise_arm", type=float, default=0.01)
parser.add_argument("--fixed_base", action="store_true", help="hull station-kept (fixed root); record 6-D arm actions")
parser.add_argument("--episode_s", type=float, default=20.0)
parser.add_argument("--max_ang_vel_deg", type=float, default=None, help="PhysX per-body angular velocity cap (deg/s)")
parser.add_argument("--station_jitter", type=float, nargs=4, default=[0.02, 0.02, 0.01, 3.0],
                    help="fixed-base station-keeping error: x y z (m) yaw (deg)")
add_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

import json
import os
import sys
import time

import numpy as np
import torch

import isaaclab.sim as sim_utils

sys.path.insert(0, os.path.dirname(__file__))
sim_cfg = sim_utils.SimulationCfg(dt=1 / 60, device="cuda:0")


def half(img: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(img).permute(2, 0, 1)[None].float()
    return torch.nn.functional.avg_pool2d(t, 2)[0].permute(1, 2, 0).round().clamp(0, 255).byte().numpy()


with launch_simulation(sim_cfg, args):
    import uw_env as E
    import uw_scene as U
    from uw_expert import UWExpert

    os.makedirs(args.out, exist_ok=True)
    env = E.UWPickEnv(sim_cfg, seed=args.seed, fixed_base=args.fixed_base, episode_s=args.episode_s,
                      max_ang_vel_deg=args.max_ang_vel_deg, station_jitter=tuple(args.station_jitter))
    act_slice = slice(4, 10) if args.fixed_base else slice(0, 10)
    env.reset(target=0)
    ex = UWExpert(env)
    ex.calibrate()
    import glob
    saved = len(glob.glob(os.path.join(args.out, "ep_*.npz")))  # resume
    attempts, log = 0, []
    prev_log = os.path.join(args.out, "collect_log.json")
    if os.path.exists(prev_log):
        log = json.load(open(prev_log))["log"]
        attempts = len(log)
    print(f"[collect] resume: saved={saved} attempts={attempts}", flush=True)
    rng = np.random.default_rng(args.seed + 7 + attempts)
    while saved < args.episodes and attempts < args.max_attempts:
        attempts += 1
        obs = env.reset()
        ex.reset()
        buf = {"front": [], "wrist": [], "state": [], "action": []}
        t0 = time.time()
        while True:
            a = ex.act()
            buf["front"].append(half(obs["front"]))
            buf["wrist"].append(half(obs["wrist"]))
            buf["state"].append(obs["state"])
            buf["action"].append(a[act_slice].copy())
            a_exec = a.copy()
            if ex.phase in ("approach", "reach"):
                a_exec[0:3] += rng.normal(0, args.noise_rov, 3).astype(np.float32)
                a_exec[4:9] += rng.normal(0, args.noise_arm, 5).astype(np.float32)
            obs, succ, done = env.step(a_exec)
            if done:
                break
            # early stop: missed grasp (gripper closed on nothing) or lift not succeeding
            if ex.phase == "lift" and ex.pk > 30:
                rise = (env.cube_pos()[2] - env.spawn_z[env.target]).item()
                if rise < 0.02 or ex.pk > 110:
                    break
        log.append({"attempt": attempts, "target": env.target, "success": bool(succ), "steps": env.k})
        if succ:
            np.savez_compressed(
                os.path.join(args.out, f"ep_{saved:04d}.npz"),
                front=np.stack(buf["front"]), wrist=np.stack(buf["wrist"]),
                state=np.stack(buf["state"]).astype(np.float32), action=np.stack(buf["action"]).astype(np.float32),
                task=U.INSTRUCTIONS[env.target], target=env.target,
            )
            saved += 1
        json.dump({"log": log}, open(prev_log, "w"))
        print(f"[collect] attempt {attempts} target={env.target} success={succ} steps={env.k} "
              f"({time.time()-t0:.1f}s) saved={saved}", flush=True)
    sr = float(np.mean([l["success"] for l in log]))
    json.dump({"saved": saved, "attempts": attempts, "expert_success_rate": sr, "log": log},
              open(os.path.join(args.out, "collect_log.json"), "w"), indent=1)
    print(f"[collect] DONE saved={saved} attempts={attempts} expert_success_rate={sr:.3f}", flush=True)
