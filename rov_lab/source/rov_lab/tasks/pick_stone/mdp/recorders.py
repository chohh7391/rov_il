"""Recorder terms for the colored pick-stone task."""

from __future__ import annotations

import torch
from isaaclab.managers.recorder_manager import RecorderTerm


class PreStepTargetColorRecorder(RecorderTerm):
    """Record the per-episode target cube color id at the beginning of each step.

    The reset event (`mdp.reset_target_color`) stores the sampled target on the env as
    `env.pick_stone_target_color_id` (shape ``[num_envs]``, dtype ``torch.long``). This term
    writes that id to HDF5 under ``task/target_color_id`` so the offline LeRobot converter can
    recover the per-episode natural-language instruction. The value is constant within an
    episode but recorded every step to match the recorder framework's per-step layout.
    """

    def record_pre_step(self) -> tuple[str | None, torch.Tensor | None]:
        target_color_id = getattr(self._env, "pick_stone_target_color_id", None)
        if target_color_id is None:
            return None, None
        # Shape [num_envs, 1]: the manager indexes value[env_id] -> [1] per episode, matching
        # the [T, 1] HDF5 layout the converter reads with read_constant_int().
        return "task/target_color_id", target_color_id.to(dtype=torch.long).unsqueeze(-1)
