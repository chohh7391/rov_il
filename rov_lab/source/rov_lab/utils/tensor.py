"""Small tensor helpers shared across rov_lab."""

from __future__ import annotations

from typing import Any

import torch


def env_ids_tensor(env_ids: Any, num_envs: int, device: str | torch.device) -> torch.Tensor:
    """Normalize manager ``env_ids`` to a long tensor.

    Isaac Lab 3.0 passes ``slice(None)`` (all environments) on full resets, besides index tensors, sequences and
    ``None``.
    """
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(num_envs, device=device, dtype=torch.long)[env_ids if isinstance(env_ids, slice) else slice(None)]
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=device, dtype=torch.long)
    return torch.as_tensor(list(env_ids), device=device, dtype=torch.long)


def as_torch(value: Any) -> torch.Tensor:
    """Return a torch tensor view of Isaac Lab asset/sensor data.

    Isaac Lab 3.0 exposes asset and sensor data as ``ProxyArray`` objects whose torch view is ``.torch``.
    Plain tensors are returned unchanged so helpers work on both kinds of input.
    """
    return value.torch if hasattr(value, "torch") else value
