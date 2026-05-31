"""ROV Lab teleoperation action preprocessing."""

from __future__ import annotations

from typing import Any

import torch
import isaaclab.envs.mdp as mdp
from leisaac.devices.action_process import init_action_cfg as init_leisaac_action_cfg
from leisaac.devices.action_process import preprocess_device_action as preprocess_leisaac_device_action

from rov_lab.tasks.template.mdp import ROVRelativeIKActionCfg, ROVRelativeJointPositionActionCfg


def init_action_cfg(action_cfg, device):
    """Initialize action configuration and use ROV-safe relative IK for keyboard-style arm control."""
    action_cfg = init_leisaac_action_cfg(action_cfg, device)
    if device == "keyboard":
        action_cfg.arm_action = ROVRelativeIKActionCfg(
            asset_name="robot",
            joint_names=["shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            body_name="gripper",
            controller=mdp.DifferentialIKControllerCfg(command_type="pose", ik_method="dls", use_relative_mode=True),
        )
        action_cfg.gripper_action = ROVRelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "gripper"],
            scale=1.0,
        )
    return action_cfg


def preprocess_device_action(action: dict[str, Any], teleop_device) -> torch.Tensor:
    """Convert teleoperation device output into the active ROV Lab action tensor."""
    if action.get("keyboard") is not None and action["joint_state"].shape[-1] == 14:
        processed_action = torch.zeros(teleop_device.env.num_envs, 14, device=teleop_device.env.device)
        processed_action[:, :] = action["joint_state"]
        return processed_action

    return preprocess_leisaac_device_action(action, teleop_device)
