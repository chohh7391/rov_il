from __future__ import annotations

from typing import Any

import torch

from rov_lab.utils.tensor import env_ids_tensor


def reset_target_color(env: Any, env_ids: torch.Tensor, num_colors: int = 3) -> None:
    """Sample and store the target cube color id for each reset environment."""
    env_ids = env_ids_tensor(env_ids, env.num_envs, env.device)
    if not hasattr(env, "pick_stone_target_color_id"):
        env.pick_stone_target_color_id = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env.pick_stone_target_color_id[env_ids] = torch.randint(
        low=0,
        high=num_colors,
        size=(len(env_ids),),
        dtype=torch.long,
        device=env.device,
    )
