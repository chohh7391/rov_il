from __future__ import annotations

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from leisaac.utils.robot_utils import is_so101_at_rest_pose


def task_done(
    env: ManagerBasedRLEnv | DirectRLEnv,
    stones_cfg: list[SceneEntityCfg],
    plate_cfg: SceneEntityCfg,
    x_range: tuple[float, float] = (-0.10, 0.10),
    y_range: tuple[float, float] = (-0.10, 0.10),
    height_range: tuple[float, float] = (-0.07, 0.07),
) -> torch.Tensor:
    """Determine if the stone picking task is complete.

    This function checks whether all success conditions for the task have been met:
    1. stone is within the target x/y range
    2. stone is below a minimum height
    3. robot come back to the rest pose

    Args:
        env: The RL environment instance.
        stones_cfg: Configuration for the stone entities.
        plate_cfg: Configuration for the plate entity.
        x_range: Range of x positions of the object for task completion.
        y_range: Range of y positions of the object for task completion.
        height_range: Range of height (z position) of the object for task completion.
    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    plate: RigidObject = env.scene[plate_cfg.name]
    plate_x = plate.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    plate_y = plate.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    plate_height = plate.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]

    for stone_cfg in stones_cfg:
        stone: RigidObject = env.scene[stone_cfg.name]
        stone_x = stone.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
        stone_y = stone.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
        stone_height = stone.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]

        done = torch.logical_and(done, stone_x < plate_x + x_range[1])
        done = torch.logical_and(done, stone_x > plate_x + x_range[0])
        done = torch.logical_and(done, stone_y < plate_y + y_range[1])
        done = torch.logical_and(done, stone_y > plate_y + y_range[0])
        done = torch.logical_and(done, stone_height < plate_height + height_range[1])
        done = torch.logical_and(done, stone_height > plate_height + height_range[0])

    joint_pos = env.scene["robot"].data.joint_pos
    joint_names = env.scene["robot"].data.joint_names
    done = torch.logical_and(done, is_so101_at_rest_pose(joint_pos, joint_names))

    return done


def dummy_done(
    env: ManagerBasedRLEnv | DirectRLEnv,
) -> torch.Tensor:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)