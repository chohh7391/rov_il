"""Shared sim-side helpers for the KRISO 2026 underwater pick-and-place IL experiment.

Builds the ROVLab-SO101-PickStone-v0 environment with experiment overrides applied at runtime
(no edits to project source):
  * keyboard action space (14-D: ROV body velocity 6 + arm relative IK pose 6 + [shoulder_pan, gripper] deltas 2)
  * reduced camera resolution (front 320x180, wrist 320x240), sonar disabled for speed
  * per-episode randomization of the ROV start pose and cube positions
Must be imported AFTER the Isaac Sim app is launched.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import torch

TASK_ID = "ROVLab-SO101-PickStone-v0"
FRONT_HW = (180, 320)
WRIST_HW = (240, 320)
CUBES = ("red_cube", "green_cube", "blue_cube")
INSTRUCTIONS = ("pick up the red cube", "pick up the green cube", "pick up the blue cube")


def make_env(seed: int, episode_s: float = 20.0, rov_xy: float = 0.15, rov_yaw_deg: float = 5.0, cube_xy: float = 0.02):
    import isaaclab.envs.mdp as base_mdp
    from isaaclab.managers import EventTermCfg, SceneEntityCfg
    from isaaclab_tasks.utils import parse_env_cfg

    import rov_lab.tasks  # noqa: F401  (gym registration)

    cfg = parse_env_cfg(TASK_ID, device="cuda:0", num_envs=1)
    cfg.use_teleop_device("keyboard")
    cfg.seed = seed
    cfg.recorders = None
    cfg.episode_length_s = episode_s

    cfg.scene.front.height, cfg.scene.front.width = FRONT_HW
    cfg.scene.wrist.height, cfg.scene.wrist.width = WRIST_HW
    cfg.scene.sonar = None
    cfg.observations.policy.sonar = None

    yaw = math.radians(rov_yaw_deg)
    cfg.events.rand_rov = EventTermCfg(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-rov_xy, rov_xy), "y": (-rov_xy, rov_xy), "z": (-0.05, 0.05), "yaw": (-yaw, yaw)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    for name in CUBES:
        setattr(
            cfg.events,
            f"rand_{name}",
            EventTermCfg(
                func=base_mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (-cube_xy, cube_xy), "y": (-cube_xy / 2, cube_xy / 2)},
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg(name),
                },
            ),
        )

    env = gym.make(TASK_ID, cfg=cfg).unwrapped
    return env


def to_uint8_img(x: torch.Tensor) -> np.ndarray:
    """Observation image (1,H,W,C) float/uint8 -> (H,W,3) uint8."""
    a = x[0]
    if a.shape[-1] == 4:
        a = a[..., :3]
    a = a.detach()
    if a.dtype != torch.uint8:
        a = a.float()
        if a.max() <= 1.5:
            a = a * 255.0
        a = a.clamp(0, 255).to(torch.uint8)
    return a.cpu().numpy()


def policy_state(obs: dict) -> np.ndarray:
    """Underwater-available proprioceptive state: arm joints (6) + DVL velocity (3) + barometer (1)."""
    p = obs["policy"]
    parts = [p["joint_pos"][0, :6], p["dvl_vel"][0].reshape(-1)[:3], p["barometer"][0].reshape(-1)[:1]]
    return torch.cat([t.float() for t in parts]).cpu().numpy().astype(np.float32)


def target_id(env) -> int:
    return int(env.pick_stone_target_color_id[0].item())


def success_now(env) -> bool:
    from rov_lab.tasks.pick_stone.mdp.terminations import target_cube_grasped_and_lifted
    from isaaclab.managers import SceneEntityCfg

    ok = target_cube_grasped_and_lifted(env, [SceneEntityCfg(c) for c in CUBES], min_lift=0.10)
    return bool(ok[0].item())
