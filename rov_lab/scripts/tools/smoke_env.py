"""Smoke test: build a rov_lab task on Isaac Lab 3.0, reset it and step it with zero actions.

usage (from the repo root):
    PYTHONPATH=rov_lab/source OMNI_KIT_ACCEPT_EULA=YES <isaaclab-3.0-python> rov_lab/scripts/tools/smoke_env.py \
        --task ROVLab-SO101-PickStone-v0 --steps 60 [--no_sensors] [--num_envs 4]

``--no_sensors`` drops the custom underwater sensors (cameras, sonar, DVL, barometer) and their observation
terms so the robot, actions, hydrodynamics and terminations can be brought up before the sensors are ported.
"""

from __future__ import annotations

import argparse

import isaaclab.sim as sim_utils
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", type=str, default="ROVLab-SO101-PickStone-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--teleop_device", type=str, default="keyboard")
parser.add_argument("--no_sensors", action="store_true")
parser.add_argument("--save_images", type=str, default=None, help="directory to save the first camera frames")
add_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = not args.no_sensors

SENSOR_SCENE_ENTITIES: tuple[str, ...] = ("wrist", "front", "dvl")
SENSOR_OBS_TERMS: tuple[str, ...] = ("wrist", "front", "dvl_vel", "dvl_depth", "dvl_beam_hit")

device: str = getattr(args, "device", None) or "cuda:0"
with launch_simulation(sim_utils.SimulationCfg(device=device), args):
    import time

    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    import rov_lab.tasks  # noqa: F401  (registers the gym ids)

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env_cfg.use_teleop_device(args.teleop_device)
    if args.no_sensors:
        for name in SENSOR_SCENE_ENTITIES:
            if hasattr(env_cfg.scene, name):
                setattr(env_cfg.scene, name, None)
        for name in SENSOR_OBS_TERMS:
            if hasattr(env_cfg.observations.policy, name):
                setattr(env_cfg.observations.policy, name, None)

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    obs, _ = env.reset()
    for group, terms in obs.items():
        if isinstance(terms, dict):
            for key, value in terms.items():
                extra = f" = {[round(float(x), 4) for x in value[0].flatten()]}" if value[0].numel() <= 8 else ""
                print(f"[smoke] obs {group}/{key}: {tuple(value.shape)} {value.dtype}{extra}")
        else:
            print(f"[smoke] obs {group}: {tuple(terms.shape)} {terms.dtype}")
    if args.save_images:
        import os

        from PIL import Image

        os.makedirs(args.save_images, exist_ok=True)
        for key, value in obs["policy"].items():
            if value.dtype == torch.uint8 and value.dim() == 4:
                Image.fromarray(value[0].cpu().numpy()).save(os.path.join(args.save_images, f"{key}.png"))
        print(f"[smoke] saved camera images to {args.save_images}")
    print(f"[smoke] action dim: {env.action_manager.total_action_dim}")

    robot = env.scene["robot"]
    root0 = robot.data.root_pos_w.torch.clone()
    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    t0 = time.time()
    with torch.inference_mode():
        for _ in range(args.steps):
            env.step(actions)
    dt = time.time() - t0
    drift = (robot.data.root_pos_w.torch - root0).norm(dim=-1)
    print(f"[smoke] {args.steps} steps x {env.num_envs} envs in {dt:.2f}s "
          f"({args.steps * env.num_envs / dt:.1f} env-steps/s); root drift under zero action: {drift.tolist()}")
    env.close()
    print("[smoke] OK")
