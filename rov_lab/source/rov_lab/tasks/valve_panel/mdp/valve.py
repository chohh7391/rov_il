"""MDP terms for the subsea valve-panel task (target valve selection, valve state, success)."""

from __future__ import annotations

from typing import Any

import torch

from isaaclab.managers import SceneEntityCfg

from rov_lab.utils.tensor import as_torch, env_ids_tensor

TARGET_ATTR: str = "valve_target_id"
"""Env attribute holding the target valve index (into ``valve_joint_names``) per environment."""


def _target(env: Any) -> torch.Tensor:
    if not hasattr(env, TARGET_ATTR):
        setattr(env, TARGET_ATTR, torch.zeros(env.num_envs, dtype=torch.long, device=env.device))
    return getattr(env, TARGET_ATTR)


def _valve_joint_ids(env: Any, panel_cfg: SceneEntityCfg) -> list[int]:
    panel = env.scene[panel_cfg.name]
    ids, _ = panel.find_joints(list(panel_cfg.joint_names), preserve_order=True)
    return ids


def reset_target_valve(env: Any, env_ids: torch.Tensor | slice | None, candidates: tuple[int, ...] = (0, 1, 2)) -> None:
    """Sample the target valve for each reset environment from ``candidates`` (indices into the valve list)."""
    env_ids = env_ids_tensor(env_ids, env.num_envs, env.device)
    cand = torch.tensor(candidates, dtype=torch.long, device=env.device)
    pick = torch.randint(0, len(candidates), (len(env_ids),), device=env.device)
    _target(env)[env_ids] = cand[pick]


def valve_angles(env: Any, panel_cfg: SceneEntityCfg) -> torch.Tensor:
    """Valve joint angles (rad, 0 = closed, -pi/2 = open), shape (N, num_valves)."""
    panel = env.scene[panel_cfg.name]
    return as_torch(panel.data.joint_pos)[:, _valve_joint_ids(env, panel_cfg)]


def valve_opening(env: Any, panel_cfg: SceneEntityCfg) -> torch.Tensor:
    """Valve opening fraction (0 closed .. 1 open), shape (N, num_valves)."""
    return (-valve_angles(env, panel_cfg) / (0.5 * torch.pi)).clamp(0.0, 1.0)


def target_valve_one_hot(env: Any, num_valves: int = 5) -> torch.Tensor:
    """One-hot target valve (ACT has no language input, so the target goes in as a state vector)."""
    return torch.nn.functional.one_hot(_target(env), num_valves).float()


def target_valve_opening(env: Any, panel_cfg: SceneEntityCfg) -> torch.Tensor:
    """Opening fraction of the target valve, shape (N, 1)."""
    return valve_opening(env, panel_cfg).gather(1, _target(env)[:, None])


def target_valve_opened(env: Any, panel_cfg: SceneEntityCfg, min_opening: float = 0.9) -> torch.Tensor:
    """Success: the target valve is turned at least ``min_opening`` of its quarter turn."""
    return target_valve_opening(env, panel_cfg)[:, 0] >= min_opening


def wrong_valve_turned(env: Any, panel_cfg: SceneEntityCfg, max_opening: float = 0.25) -> torch.Tensor:
    """Failure: a valve other than the target is turned more than ``max_opening``."""
    opening = valve_opening(env, panel_cfg)
    mask = torch.ones_like(opening, dtype=torch.bool)
    mask.scatter_(1, _target(env)[:, None], False)
    return (opening * mask).amax(dim=1) > max_opening


PADDLE_CENTER_Z: tuple[float, ...] = (0.03, 0.03, 0.03, 0.04, 0.04)
"""Paddle centre above each valve link origin along the valve axis (m): big valves 1-3, small valves 4-5."""


def target_paddle_center_w(env: Any, panel_cfg: SceneEntityCfg) -> torch.Tensor:
    """World position of the target valve's paddle centre, shape (N, 3)."""
    import isaaclab.utils.math as math_utils

    panel = env.scene[panel_cfg.name]
    names = [j.removesuffix("_joint") for j in panel_cfg.joint_names]
    body_ids = torch.tensor([panel.find_bodies(n)[0][0] for n in names], device=env.device)[_target(env)]
    ar = torch.arange(env.num_envs, device=env.device)
    pos = as_torch(panel.data.body_pos_w)[ar, body_ids]
    quat = as_torch(panel.data.body_quat_w)[ar, body_ids]
    cz = torch.tensor(PADDLE_CENTER_Z, device=env.device)[_target(env)]
    z_axis = math_utils.quat_apply(quat, torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(pos))
    return pos + z_axis * cz[:, None]


def paddle_grasped(env: Any, panel_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                   ee_body: str = "jaw_base_link", tcp_offset: tuple[float, float, float] = (0.11, 0.0, 0.0),
                   max_finger_opening: float = 0.012, max_dist: float = 0.045) -> torch.Tensor:
    """True when the jaws are closed (finger opening below ``max_finger_opening``) with the TCP within ``max_dist`` of
    the target paddle centre (subtask signal for Isaac Lab Mimic, like Isaac Lab's ``object_grasped``), shape (N,)."""
    import isaaclab.utils.math as math_utils

    robot = env.scene[robot_cfg.name]
    ee_id = robot.find_bodies(ee_body)[0][0]
    finger_ids = robot.find_joints("finger_.*_joint")[0]
    ee_pos = as_torch(robot.data.body_pos_w)[:, ee_id]
    ee_quat = as_torch(robot.data.body_quat_w)[:, ee_id]
    tcp = ee_pos + math_utils.quat_apply(ee_quat, torch.tensor(tcp_offset, device=env.device).expand_as(ee_pos))
    closed = as_torch(robot.data.joint_pos)[:, finger_ids].mean(dim=1) < max_finger_opening
    near = (tcp - target_paddle_center_w(env, panel_cfg)).norm(dim=-1) < max_dist
    return closed & near
