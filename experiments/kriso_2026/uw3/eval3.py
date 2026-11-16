"""Evaluate a LeRobot policy (served by ../policy_server.py in the fire env) in the underwater pick env.

Episodes use seeds disjoint from data collection. Optional constant water current (m/s, random
horizontal direction per episode) tests robustness to a disturbance not present in the demos.
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=60)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--port", type=int, default=50555)
parser.add_argument("--current", type=float, default=0.0)
parser.add_argument("--max_s", type=float, default=20.0)
parser.add_argument("--out", default="/home/home/rov_il/runs/kriso_2026/eval_act")
parser.add_argument("--save_strips", type=int, default=4)
parser.add_argument("--fixed_base", action="store_true", help="hull station-kept (fixed root); policy outputs 6-D arm actions")
# Optional non-learned residual on the ROV velocity command (preliminary check before residual RL):
# PI feedback on the DVL-measured body velocity so the commanded velocity is actually achieved under current.
parser.add_argument("--residual", choices=["none", "dvl"], default="none")
parser.add_argument("--res_kp", type=float, default=1.0)
parser.add_argument("--res_ki", type=float, default=2.0)
parser.add_argument("--res_ilim", type=float, default=0.1)
parser.add_argument("--res_max", type=float, default=0.2)
add_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

import json
import os
import sys
import time
from multiprocessing.connection import Client

import numpy as np
import torch

import isaaclab.sim as sim_utils

sys.path.insert(0, os.path.dirname(__file__))
sim_cfg = sim_utils.SimulationCfg(dt=1 / 60, device="cuda:0")


def half(img: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(img).permute(2, 0, 1)[None].float()
    return torch.nn.functional.avg_pool2d(t, 2)[0].permute(1, 2, 0).round().clamp(0, 255).byte().numpy()


def pack(a: np.ndarray):
    a = np.ascontiguousarray(a)
    return (a.tobytes(), str(a.dtype), tuple(a.shape))


with launch_simulation(sim_cfg, args):
    from PIL import Image

    import uw_env as E
    import uw_scene as U

    os.makedirs(args.out, exist_ok=True)
    env = E.UWPickEnv(sim_cfg, seed=args.seed, episode_s=args.max_s, current=args.current, fixed_base=args.fixed_base)
    conn = Client(("127.0.0.1", args.port), authkey=b"kriso")
    results = []
    for ep in range(args.episodes):
        obs = env.reset()
        conn.send({"cmd": "reset"}); conn.recv()
        task = U.INSTRUCTIONS[env.target]
        strip, t0, succ = [], time.time(), False
        res_int = np.zeros(3, dtype=np.float32)
        lin_scale = np.asarray(U.ROVServo.LIN_SCALE, dtype=np.float32)
        drift = []
        while True:
            f, w = half(obs["front"]), half(obs["wrist"])
            o = {"observation.images.front": pack(f), "observation.images.wrist": pack(w),
                 "observation.state": pack(obs["state"].astype(np.float32)),
                 "observation.environment_state": pack(np.eye(3, dtype=np.float32)[env.target])}
            conn.send({"cmd": "act", "obs": o, "task": task})
            b, dt, sh = conn.recv()["action"]
            a = np.frombuffer(b, dtype=dt).reshape(sh).copy()
            v_dvl = obs["state"][6:9]
            if args.residual == "dvl" and a.shape[-1] == 10:
                err = np.clip(a[:3], -1, 1) * lin_scale - v_dvl
                res_int = np.clip(res_int + err / 30.0, -args.res_ilim, args.res_ilim)
                a[:3] += np.clip((args.res_kp * err + args.res_ki * res_int) / lin_scale, -args.res_max, args.res_max)
            if a.shape[-1] == 6 or np.abs(a[:3]).max() < 0.05:  # ROV holding station: record DVL speed as drift
                drift.append(float(np.linalg.norm(v_dvl[:2])))
            if ep < args.save_strips and env.k % 40 == 0:
                strip.append(np.concatenate([obs["front"], obs["wrist"][30:210]], axis=1))
            obs, succ, done = env.step(a)
            if done:
                break
        cube_dz = float((env.cube_pos()[2] - env.spawn_z[env.target]).item())
        results.append({"episode": ep, "target": env.target, "success": bool(succ), "steps": env.k, "cube_dz": cube_dz,
                        "hold_drift_mps": float(np.mean(drift)) if drift else None})
        if strip:
            Image.fromarray(np.concatenate(strip[:16], axis=0)).save(f"{args.out}/ep{ep:03d}_{'ok' if succ else 'fail'}.png")
        rate = np.mean([r["success"] for r in results])
        print(f"[eval] ep {ep:3d} target={env.target} success={succ} steps={env.k} dz={cube_dz:.3f} "
              f"({time.time()-t0:.1f}s) running={rate:.3f}", flush=True)
    sr = float(np.mean([r["success"] for r in results]))
    n = len(results)
    k = sum(r["success"] for r in results)
    # Wilson 95% interval
    z = 1.96
    c = (k / n + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(k / n * (1 - k / n) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    per = {U.INSTRUCTIONS[t]: [int(sum(r["success"] for r in results if r["target"] == t)),
                               int(sum(1 for r in results if r["target"] == t))] for t in range(3)}
    summary = {"episodes": n, "successes": int(k), "success_rate": sr, "wilson95": [c - h, c + h],
               "current_mps": args.current, "residual": args.residual, "fixed_base": args.fixed_base, "per_target": per,
               "mean_hold_drift_mps": float(np.mean([r["hold_drift_mps"] for r in results if r["hold_drift_mps"] is not None] or [np.nan])),
               "mean_steps_success": float(np.mean([r["steps"] for r in results if r["success"]] or [np.nan])),
               "results": results}
    json.dump(summary, open(f"{args.out}/summary.json", "w"), indent=1)
    print("[eval] SUMMARY", json.dumps({k_: v for k_, v in summary.items() if k_ != "results"}), flush=True)
    conn.send({"cmd": "close"}); conn.recv()
