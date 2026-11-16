"""Action-term implementations that subclass Isaac Lab's concrete actions (imported lazily by the cfgs).

``isaaclab.envs.mdp.actions.task_space_actions`` imports ``pxr``. Importing it while rov_lab's task configs are parsed
(before Kit starts, as Isaac Lab's Mimic and record scripts do) loads the pip ``usd-core`` libraries, which then clash
with Kit's own USD build. The cfgs in :mod:`rov_actions` therefore name these classes as ``"{DIR}.rov_actions_impl:..."``
strings, which Isaac Lab resolves only when the action manager builds the term.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

from rov_lab.utils.tensor import as_torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .rov_actions import ROVRelativeIKActionCfg, ROVRelativeJointPositionActionCfg


class ROVRelativeIKAction(DifferentialInverseKinematicsAction):
    """Differential IK action that idles as joint hold when no arm command is active."""

    cfg: ROVRelativeIKActionCfg

    def __init__(self, cfg: ROVRelativeIKActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._inactive_deadband = float(cfg.inactive_deadband)
        self._joint_pos_hold_target = as_torch(self._asset.data.joint_pos)[:, self._joint_ids].clone()

    def apply_actions(self):
        if torch.all(torch.abs(self._raw_actions) <= self._inactive_deadband):
            self._asset.set_joint_position_target_index(target=self._joint_pos_hold_target, joint_ids=self._joint_ids)
            return

        super().apply_actions()
        self._joint_pos_hold_target[:] = as_torch(self._asset.data.joint_pos_target)[:, self._joint_ids]

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._joint_pos_hold_target[:] = as_torch(self._asset.data.joint_pos)[:, self._joint_ids]
        else:
            self._joint_pos_hold_target[env_ids] = as_torch(self._asset.data.joint_pos)[env_ids][:, self._joint_ids]


class ROVRelativeJointPositionAction(RelativeJointPositionAction):
    """Relative joint action that holds the last target when no command is active."""

    cfg: ROVRelativeJointPositionActionCfg

    def __init__(self, cfg: ROVRelativeJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._inactive_deadband = float(cfg.inactive_deadband)
        self._joint_pos_hold_target = as_torch(self._asset.data.joint_pos)[:, self._joint_ids].clone()

    def apply_actions(self):
        if torch.all(torch.abs(self._raw_actions) <= self._inactive_deadband):
            self._asset.set_joint_position_target_index(target=self._joint_pos_hold_target, joint_ids=self._joint_ids)
            return

        current_actions = self.processed_actions + as_torch(self._asset.data.joint_pos)[:, self._joint_ids]
        self._asset.set_joint_position_target_index(target=current_actions, joint_ids=self._joint_ids)
        self._joint_pos_hold_target[:] = current_actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._joint_pos_hold_target[:] = as_torch(self._asset.data.joint_pos)[:, self._joint_ids]
        else:
            self._joint_pos_hold_target[env_ids] = as_torch(self._asset.data.joint_pos)[env_ids][:, self._joint_ids]
