"""Collect valve-panel demonstrations with the scripted expert (I90-class ROV + Bravo 7-class, Isaac Lab 3.0).

The expert drives the relative-IK env (``ROVLab-I90Bravo7-ValvePanel-v0``, 30 Hz, hull fixed with station jitter).
DART-style noise is added to the *executed* end-effector deltas while reaching and approaching; the recorded label is
the clean expert command as an absolute jaw pose + gripper opening (see ``valve_io``). Targets cycle through the five
valves so the dataset is balanced. Only successful episodes are saved (``ep_XXXX.npz``); the run resumes from the
files already in ``--out``.

    PYTHONPATH=rov_lab/source:experiments/kriso_2026/valve OMNI_KIT_ACCEPT_EULA=YES <isaaclab-3.0-python> \\
        experiments/kriso_2026/valve/collect_valve.py --episodes 150 --out datasets/kriso_2026/valve_demos
"""

from __future__ import annotations

import argparse
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", type=str, default="ROVLab-I90Bravo7-ValvePanel-v0")
parser.add_argument("--episodes", type=int, default=150)
parser.add_argument("--max_attempts", type=int, default=400)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=Path, default=Path("datasets/kriso_2026/valve_demos"))
parser.add_argument("--noise_pos", type=float, default=0.002, help="DART noise std on executed jaw deltas (m/step)")
parser.add_argument("--noise_rot", type=float, default=0.02, help="DART noise std on executed jaw rotation (rad/step)")
add_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

device: str = getattr(args, "device", None) or "cuda:0"
with launch_simulation(sim_utils.SimulationCfg(device=device), args):
    import json
    import time

    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab.managers import RecorderManagerBaseCfg
    from isaaclab_tasks.utils import parse_env_cfg

    import rov_lab.tasks  # noqa: F401
    import valve_io as V
    from rov_lab.tasks.valve_panel.expert import PHASES, ValveExpert
    from rov_lab.tasks.valve_panel.mdp.valve import TARGET_ATTR
    from valve_sim import make_state_fn

    torch.manual_seed(args.seed)
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=1)
    env_cfg.seed = args.seed
    env_cfg.use_teleop_device("keyboard")
    env_cfg.recorders = RecorderManagerBaseCfg()
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    state_vec = make_state_fn(env)
    expert = ValveExpert(env)

    args.out.mkdir(parents=True, exist_ok=True)
    saved = len(list(args.out.glob("ep_*.npz")))
    log_path = args.out / "collect_log.json"
    log: list[dict] = json.loads(log_path.read_text())["log"] if log_path.exists() else []
    attempts = len(log)
    rng = np.random.default_rng(args.seed + 7 + attempts)
    print(f"[collect] resume: saved={saved} attempts={attempts}", flush=True)
    obs, _ = env.reset()
    with torch.inference_mode():
        while saved < args.episodes and attempts < args.max_attempts:
            target = attempts % 5
            getattr(env, TARGET_ATTR)[:] = target
            expert.reset()
            attempts += 1
            buf: dict[str, list] = {"front": [], "wrist": [], "state": [], "action": []}
            t0, steps, success = time.time(), 0, False
            while True:
                action = expert.act()
                buf["front"].append(V.pool(obs["policy"]["front"][0].cpu().numpy(), V.FRONT_POOL))
                buf["wrist"].append(V.pool(obs["policy"]["wrist"][0].cpu().numpy(), V.WRIST_POOL))
                buf["state"].append(state_vec())
                lab = expert.label_pose_b[0].cpu().numpy()
                buf["action"].append(np.concatenate([lab[:3], V.quat_xyzw_to_rot6d(lab[3:]),
                                                     [float(expert.label_grip[0])]]).astype(np.float32))
                if PHASES[int(expert.phase[0])] in ("pregrasp", "approach"):
                    action[:, 6:9] += torch.from_numpy(rng.normal(0, args.noise_pos, 3)).float().to(env.device)
                    action[:, 9:12] += torch.from_numpy(rng.normal(0, args.noise_rot, 3)).float().to(env.device)
                obs, _, terminated, truncated, _ = env.step(action)
                steps += 1
                if bool(terminated[0]) or bool(truncated[0]):
                    success = bool(env.termination_manager.get_term("success")[0])
                    break
            log.append({"attempt": attempts, "target": target, "success": success, "steps": steps})
            if success:
                np.savez_compressed(
                    args.out / f"ep_{saved:04d}.npz",
                    front=np.stack(buf["front"]), wrist=np.stack(buf["wrist"]),
                    state=np.stack(buf["state"]), action=np.stack(buf["action"]),
                    task=V.INSTRUCTIONS[target], target=target)
                saved += 1
            log_path.write_text(json.dumps({"log": log}))
            print(f"[collect] attempt {attempts} valve={target + 1} success={success} steps={steps} "
                  f"({time.time() - t0:.1f}s) saved={saved}", flush=True)
    sr = float(np.mean([e["success"] for e in log]))
    log_path.write_text(json.dumps({"saved": saved, "attempts": attempts, "expert_success_rate": sr,
                                    "noise_pos": args.noise_pos, "noise_rot": args.noise_rot, "log": log}, indent=1))
    print(f"[collect] DONE saved={saved} attempts={attempts} expert_success_rate={sr:.3f}", flush=True)
    env.close()
