"""Evaluate a LeRobot policy (served by policy_server.py) in the underwater pick-and-place sim.

The policy runs at 30 Hz: each predicted action is held for 2 env steps (env step = 1/60 s),
matching how demonstrations were recorded. Success = target cube grasped and lifted >= 10 cm.
"""

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--seed", type=int, default=10_000)
parser.add_argument("--port", type=int, default=50555)
parser.add_argument("--max_s", type=float, default=20.0)
parser.add_argument("--out", default="runs/kriso_2026/eval")
parser.add_argument("--save_frames", type=int, default=3, help="episodes for which to save image strips")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(vars(args)).app

import time
from multiprocessing.connection import Client

import numpy as np
import torch
from PIL import Image

import sim_common as sc

os.makedirs(args.out, exist_ok=True)
env = sc.make_env(seed=args.seed, episode_s=args.max_s)
conn = Client(("127.0.0.1", args.port), authkey=b"kriso")
results = []
for ep in range(args.episodes):
    obs, _ = env.reset()
    conn.send({"cmd": "reset"}); conn.recv()
    tid = sc.target_id(env)
    task = sc.INSTRUCTIONS[tid]
    success, steps, a = False, 0, None
    frames = []
    t0 = time.time()
    while True:
        if steps % 2 == 0:
            front = sc.to_uint8_img(obs["policy"]["front"])
            wrist = sc.to_uint8_img(obs["policy"]["wrist"])
            o = {"observation.images.front": front, "observation.images.wrist": wrist,
                 "observation.state": sc.policy_state(obs)}
            conn.send({"cmd": "act", "obs": o, "task": task})
            a = conn.recv()["action"].reshape(1, -1)
            if ep < args.save_frames and steps % 30 == 0:
                frames.append(np.concatenate([front, wrist[30:210]], axis=1))
        obs, _, term, trunc, _ = env.step(torch.as_tensor(a, dtype=torch.float32, device=env.device))
        steps += 1
        if bool(term[0]) and not bool(trunc[0]):
            success = True
            break
        if bool(trunc[0]) or bool(term[0]):
            break
    results.append({"episode": ep, "target": tid, "success": success, "steps": steps})
    if frames:
        Image.fromarray(np.concatenate(frames[:8], axis=0)).save(f"{args.out}/ep{ep:03d}_{'ok' if success else 'fail'}.png")
    rate = np.mean([r["success"] for r in results])
    print(f"[eval] ep {ep:3d} target={tid} success={success} steps={steps} ({time.time()-t0:.1f}s)  running={rate:.3f}", flush=True)

summary = {
    "episodes": len(results),
    "success_rate": float(np.mean([r["success"] for r in results])),
    "per_target": {sc.INSTRUCTIONS[k]: float(np.mean([r["success"] for r in results if r["target"] == k] or [np.nan])) for k in range(3)},
    "results": results,
}
json.dump(summary, open(f"{args.out}/summary.json", "w"), indent=1)
print("[eval] SUMMARY", json.dumps({k: v for k, v in summary.items() if k != "results"}), flush=True)
conn.send({"cmd": "close"}); conn.recv()
env.close()
app.close()
