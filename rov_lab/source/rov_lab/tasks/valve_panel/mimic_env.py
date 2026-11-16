"""Isaac Lab Mimic wrapper for the valve-panel task (parallel demonstration generation).

Mimic splits each source demonstration into object-centric subtasks (reach + grasp the paddle, then turn), moves the
segments to the valve pose of a new episode and replays them with the absolute-pose action, in many envs at once.

Frames: end-effector (jaw_base_link) and object (target valve link) poses are given in the ROV base frame, which is
also the frame of the absolute-pose action. The hull's station-keeping jitter therefore shows up as a change of the
valve pose, which Mimic compensates for.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv

from rov_lab.utils.tensor import as_torch

from .mdp.valve import TARGET_ATTR

ROV_ACTION_DIM: int = 6
"""Leading ROV-velocity slots of the 14-D absolute-pose action (zero while the hull is fixed)."""


class ValvePanelMimicEnv(ManagerBasedRLMimicEnv):
    """Mimic API for ``ROVLab-I90Bravo7-ValvePanel-Mimic-v0`` (one end effector: the Bravo jaws)."""

    def _base_frame_pose(self, pos_w: torch.Tensor, quat_w: torch.Tensor, env_ids) -> torch.Tensor:  # noqa: ANN001
        robot = self.scene["robot"]
        pos_b, quat_b = PoseUtils.subtract_frame_transforms(
            as_torch(robot.data.root_pos_w)[env_ids], as_torch(robot.data.root_quat_w)[env_ids], pos_w, quat_w)
        return PoseUtils.make_pose(pos_b, PoseUtils.matrix_from_quat(quat_b))

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = slice(None)
        robot = self.scene["robot"]
        ee_id = robot.find_bodies("jaw_base_link")[0][0]
        return self._base_frame_pose(as_torch(robot.data.body_pos_w)[env_ids, ee_id],
                                     as_torch(robot.data.body_quat_w)[env_ids, ee_id], env_ids)

    def target_eef_pose_to_action(self, target_eef_pose_dict: dict, gripper_action_dict: dict,
                                  action_noise_dict: dict | None = None, env_id: int = 0) -> torch.Tensor:
        (eef_name, target_eef_pose), = target_eef_pose_dict.items()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)
        pose_action = torch.cat([target_pos, PoseUtils.quat_from_matrix(target_rot)], dim=0)
        if action_noise_dict is not None and action_noise_dict.get(eef_name) is not None:
            pose_action = pose_action + action_noise_dict[eef_name] * torch.randn_like(pose_action)
        (gripper_action,) = gripper_action_dict.values()
        rov = torch.zeros(ROV_ACTION_DIM, device=pose_action.device)
        return torch.cat([rov, pose_action, gripper_action], dim=0).unsqueeze(0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        eef_name = list(self.cfg.subtask_configs.keys())[0]
        pos = action[:, ROV_ACTION_DIM:ROV_ACTION_DIM + 3]
        quat = action[:, ROV_ACTION_DIM + 3:ROV_ACTION_DIM + 7]
        quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return {eef_name: PoseUtils.make_pose(pos, PoseUtils.matrix_from_quat(quat)).clone()}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        return {list(self.cfg.subtask_configs.keys())[0]: actions[:, -1:]}

    def get_object_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Pose of the target valve link (paddle frame; rotates with the valve) in the ROV base frame."""
        if env_ids is None:
            env_ids = slice(None)
        panel = self.scene["panel"]
        body_ids = torch.tensor([panel.find_bodies(f"valve_{k}")[0][0] for k in range(1, 6)], device=self.device)
        tgt = getattr(self, TARGET_ATTR)[env_ids]
        ar = torch.arange(self.num_envs, device=self.device)[env_ids]
        pos = as_torch(panel.data.body_pos_w)[ar, body_ids[tgt]]
        quat = as_torch(panel.data.body_quat_w)[ar, body_ids[tgt]]
        return {"target_valve": self._base_frame_pose(pos, quat, env_ids)}

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        return {"grasped": self.obs_buf["subtask_terms"]["grasped"][env_ids]}

