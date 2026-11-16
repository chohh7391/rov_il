"""Evaluate a LeRobot policy (served by ../policy_server.py in the ``fire`` env) on the valve-panel task.

Runs the absolute-pose env (``ROVLab-I90Bravo7-ValvePanel-AbsPose-v0``). The policy's 10-D action (jaw position,
rotation 6D, gripper opening) becomes the env's 14-D action (zero ROV command + position + quaternion + gripper).
Targets cycle through the five valves; the env seed differs from data collection. Writes ``summary.json`` with the
success rate, Wilson 95% interval, per-valve counts and failure modes, plus frame strips of the first episodes.

    experiments/kriso_2026/valve/run_eval_valve.sh <pretrained_model dir> <out dir> [--episodes 60]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", type=str, default="ROVLab-I90Bravo7-ValvePanel-AbsPose-v0")
parser.add_argument("--episodes", type=int, default=60)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--port", type=int, default=50555)
parser.add_argument("--max_s", type=float, default=20.0)
parser.add_argument("--out", type=Path, default=Path("runs/kriso_2026/eval_act_valve"))
parser.add_argument("--save_strips", type=int, default=5)
parser.add_argument("--rollout_strip", action="store_true",
                    help="save rollout_strip.png from the first successful episode: front + wrist at approach, grasp, "
                         "turn and open (picked from the finger opening and the valve angle)")
add_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True


def pack(a):  # noqa: ANN001, ANN201
    import numpy as np

    a = np.ascontiguousarray(a)
    return (a.tobytes(), str(a.dtype), tuple(a.shape))


device: str = getattr(args, "device", None) or "cuda:0"
with launch_simulation(sim_utils.SimulationCfg(device=device), args):
    import json
    import time
    from multiprocessing.connection import Client

    import gymnasium as gym
    import numpy as np
    import torch
    from PIL import Image
    from isaaclab.managers import RecorderManagerBaseCfg
    from isaaclab_tasks.utils import parse_env_cfg

    import rov_lab.tasks  # noqa: F401
    import valve_io as V
    from rov_lab.tasks.valve_panel.mdp.valve import TARGET_ATTR
    from valve_sim import make_state_fn

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=1)
    env_cfg.seed = args.seed
    env_cfg.episode_length_s = args.max_s
    env_cfg.recorders = RecorderManagerBaseCfg()
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    state_vec = make_state_fn(env)
    panel = env.scene["panel"]
    conn = Client(("127.0.0.1", args.port), authkey=b"kriso")
    args.out.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    obs, _ = env.reset()
    with torch.inference_mode():
        for ep in range(args.episodes):
            target = ep % 5
            getattr(env, TARGET_ATTR)[:] = target
            conn.send({"cmd": "reset"})
            conn.recv()
            onehot = np.eye(5, dtype=np.float32)[target]
            strip, t0, steps, max_open = [], time.time(), 0, 0.0
            dump = {"front": [], "wrist": [], "opening": [], "finger": []}
            want_dump = args.rollout_strip and not (args.out / "rollout_strip.png").exists()
            while True:
                front = obs["policy"]["front"][0].cpu().numpy()
                wrist = obs["policy"]["wrist"][0].cpu().numpy()
                o = {"observation.images.front": pack(V.pool(front, V.FRONT_POOL)),
                     "observation.images.wrist": pack(V.pool(wrist, V.WRIST_POOL)),
                     "observation.state": pack(state_vec()),
                     "observation.environment_state": pack(onehot)}
                conn.send({"cmd": "act", "obs": o, "task": V.INSTRUCTIONS[target]})
                b, dt, sh = conn.recv()["action"]
                a = np.frombuffer(b, dtype=dt).reshape(sh).astype(np.float64)
                quat = V.rot6d_to_quat_xyzw(a[3:9])
                env_action = np.concatenate([np.zeros(6), a[:3], quat, [np.clip(a[9], -1.0, 1.0)]])
                if ep < args.save_strips and steps % 20 == 0:
                    strip.append(np.concatenate([front, wrist[60:420]], axis=1))
                opening = (-panel.data.joint_pos.torch[0] / (0.5 * np.pi)).clamp(0, 1).cpu().numpy()
                max_open = max(max_open, float(opening[target]))
                if want_dump:
                    dump["front"].append(front[::2, ::2].copy())
                    dump["wrist"].append(wrist[::2, ::2].copy())
                    dump["opening"].append(float(opening[target]))
                    dump["finger"].append(float(state_vec()[6]))
                obs, _, terminated, truncated, _ = env.step(
                    torch.as_tensor(env_action, dtype=torch.float32, device=env.device)[None])
                steps += 1
                if bool(terminated[0]) or bool(truncated[0]):
                    tm = env.termination_manager
                    res = {name: bool(tm.get_term(name)[0]) for name in tm.active_terms}
                    break
            results.append({"episode": ep, "valve": target + 1, "success": res.get("success", False),
                            "wrong_valve": res.get("wrong_valve", False), "time_out": res.get("time_out", False),
                            "steps": steps, "max_target_opening": round(max_open, 3),
                            "final_opening": [round(float(x), 3) for x in opening]})
            if want_dump and results[-1]["success"]:
                op, fi = np.asarray(dump["opening"]), np.asarray(dump["finger"])
                t_grasp = int(np.argmax(fi < 0.012)) if np.any(fi < 0.012) else len(fi) // 2
                t_turn = int(np.argmax(op >= 0.4)) if np.any(op >= 0.4) else (t_grasp + len(op)) // 2
                picks = [("approach", max(0, t_grasp // 2)), ("grasp", min(len(op) - 1, t_grasp + 5)),
                         ("turn", t_turn), ("open", len(op) - 1)]
                cols = [np.concatenate([dump["front"][t], dump["wrist"][t]], axis=0) for _, t in picks]
                Image.fromarray(np.concatenate(cols, axis=1)).save(args.out / "rollout_strip.png")
                (args.out / "rollout_strip.json").write_text(json.dumps(
                    {"episode": ep, "valve": target + 1, "frames": {k: t for k, t in picks},
                     "opening": {k: round(float(op[t]), 3) for k, t in picks},
                     "finger_opening_m": {k: round(float(fi[t]), 4) for k, t in picks},
                     "layout": "columns approach|grasp|turn|open; rows front (top) / wrist (bottom), 320 px wide each"}))
            if strip:
                tag = "ok" if results[-1]["success"] else "fail"
                Image.fromarray(np.concatenate(strip[:12], axis=0)).save(args.out / f"ep{ep:03d}_valve{target + 1}_{tag}.png")
            rate = np.mean([r["success"] for r in results])
            print(f"[eval] ep {ep:3d} valve={target + 1} {res} steps={steps} max_open={max_open:.2f} "
                  f"({time.time() - t0:.1f}s) running={rate:.3f}", flush=True)
    n = len(results)
    k = int(sum(r["success"] for r in results))
    z = 1.96
    c = (k / n + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(k / n * (1 - k / n) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    per = {f"valve_{v}": [int(sum(r["success"] for r in results if r["valve"] == v)),
                          int(sum(1 for r in results if r["valve"] == v))] for v in range(1, 6)}
    summary = {"episodes": n, "successes": k, "success_rate": k / n, "wilson95": [c - h, c + h], "per_valve": per,
               "wrong_valve": int(sum(r["wrong_valve"] for r in results)),
               "time_out": int(sum(r["time_out"] for r in results)),
               "partial_turn_ge_0.25": int(sum((not r["success"]) and r["max_target_opening"] >= 0.25 for r in results)),
               "mean_steps_success": float(np.mean([r["steps"] for r in results if r["success"]] or [np.nan])),
               "seed": args.seed, "max_s": args.max_s, "results": results}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("[eval] SUMMARY", json.dumps({k_: v for k_, v in summary.items() if k_ != "results"}), flush=True)
    conn.send({"cmd": "close"})
    conn.recv()
    env.close()
