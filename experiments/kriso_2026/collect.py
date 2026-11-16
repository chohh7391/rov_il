"""Generate pick-and-place demonstrations with a privileged scripted expert and save them as npz.

The expert uses simulator ground truth (cube and gripper poses) to produce actions in the same
14-D keyboard-teleop action space a human pilot would use. Only successful episodes are saved.
Recorded observations are limited to underwater-available sensors: front/wrist underwater
camera images, arm joint positions, DVL velocity, barometer.
Control/recording rate: 30 Hz (each action held for 2 env steps of 1/60 s).
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=60, help="number of successful demos to save")
parser.add_argument("--max_attempts", type=int, default=200)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="runs/kriso_2026/demos")
parser.add_argument("--max_s", type=float, default=20.0)
parser.add_argument("--debug", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(vars(args)).app

import time

import numpy as np
import torch

import sim_common as sc
from expert import ScriptedExpert

os.makedirs(args.out, exist_ok=True)
env = sc.make_env(seed=args.seed, episode_s=args.max_s)
expert = ScriptedExpert(env)
saved, attempts = 0, 0
stats = []
while saved < args.episodes and attempts < args.max_attempts:
    attempts += 1
    obs, _ = env.reset()
    expert.reset()
    tid = sc.target_id(env)
    buf = {"front": [], "wrist": [], "state": [], "action": []}
    steps, success, a = 0, False, None
    t0 = time.time()
    while True:
        if steps % 2 == 0:
            a = expert.act()
            buf["front"].append(sc.to_uint8_img(obs["policy"]["front"]))
            buf["wrist"].append(sc.to_uint8_img(obs["policy"]["wrist"]))
            buf["state"].append(sc.policy_state(obs))
            buf["action"].append(a.copy())
            if args.debug and steps % 30 == 0:
                print(f"  t={steps/60:5.2f}s phase={expert.phase:8s} err={expert.last_err} grip={expert.grip:.2f}", flush=True)
        obs, _, term, trunc, _ = env.step(torch.as_tensor(a[None], dtype=torch.float32, device=env.device))
        steps += 1
        if bool(term[0]) and not bool(trunc[0]):
            success = True
            break
        if bool(trunc[0]) or bool(term[0]):
            break
    stats.append((tid, success, steps, expert.phase))
    print(f"[collect] attempt {attempts} target={tid} success={success} steps={steps} last_phase={expert.phase} "
          f"({time.time()-t0:.1f}s) saved={saved + int(success)}", flush=True)
    if success:
        np.savez_compressed(
            os.path.join(args.out, f"ep_{saved:04d}.npz"),
            front=np.stack(buf["front"]), wrist=np.stack(buf["wrist"]),
            state=np.stack(buf["state"]), action=np.stack(buf["action"]).astype(np.float32),
            task=sc.INSTRUCTIONS[tid], target=tid,
        )
        saved += 1
sr = np.mean([s[1] for s in stats]) if stats else 0
print(f"[collect] DONE saved={saved} attempts={attempts} expert_success_rate={sr:.3f}", flush=True)
env.close()
app.close()
