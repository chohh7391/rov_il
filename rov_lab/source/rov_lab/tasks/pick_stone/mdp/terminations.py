from __future__ import annotations

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .observations import resolve_target_color_id, target_cube_grasped


def target_cube_grasped_and_lifted(
    env: ManagerBasedRLEnv | DirectRLEnv,
    cube_cfgs: list[SceneEntityCfg],
    min_lift: float = 0.10,
    diff_threshold: float = 0.05,
    grasp_threshold: float = 0.60,
) -> torch.Tensor:
    """Check whether the reset-selected target cube is grasped and lifted."""
    grasped = target_cube_grasped(
        env,
        cube_cfgs=cube_cfgs,
        diff_threshold=diff_threshold,
        grasp_threshold=grasp_threshold,
    )
    current_heights = []
    spawn_heights = []
    for cube_cfg in cube_cfgs:
        cube: RigidObject = env.scene[cube_cfg.name]
        current_heights.append(cube.data.root_pos_w[:, 2])
        # TODO: Store a per-episode spawn height when cube spawn randomization is introduced.
        spawn_heights.append(cube.data.default_root_state[:, 2])

    target_ids = resolve_target_color_id(env)
    cube_current_heights = torch.stack(current_heights, dim=1)
    cube_spawn_heights = torch.stack(spawn_heights, dim=1)
    target_current_height = torch.gather(cube_current_heights, dim=1, index=target_ids.unsqueeze(1)).squeeze(1)
    target_spawn_height = torch.gather(cube_spawn_heights, dim=1, index=target_ids.unsqueeze(1)).squeeze(1)
    lifted = target_current_height - target_spawn_height >= min_lift

    return torch.logical_and(grasped, lifted)
