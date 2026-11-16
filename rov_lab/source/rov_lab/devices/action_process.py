"""ROV Lab teleoperation action configuration and preprocessing.

Maps a teleoperation device to the env's arm/gripper action terms and converts the device output into the
action tensor. Only the keyboard device is supported for now; haptic devices will be added here.
"""

from __future__ import annotations

from dataclasses import MISSING, fields
from typing import Any

import torch
from isaaclab.controllers import DifferentialIKControllerCfg

from rov_lab.tasks.template.mdp import ROVRelativeIKActionCfg, ROVRelativeJointPositionActionCfg

# Action layout produced by ROVKeyboard: ROV velocity (6) + arm IK delta (pos 3 + axis-angle 3) + pan + gripper.
KEYBOARD_ACTION_DIM: int = 14


def init_action_cfg(action_cfg: Any, device: str) -> Any:
    """Set the arm and gripper action terms for ``device`` and check that every action term is configured."""
    if device == "keyboard":
        action_cfg.arm_action = ROVRelativeIKActionCfg(
            asset_name="robot",
            joint_names=["shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            body_name="gripper",
            controller=DifferentialIKControllerCfg(command_type="pose", ik_method="dls", use_relative_mode=True),
        )
        action_cfg.gripper_action = ROVRelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "gripper"],
            scale=1.0,
        )
    else:
        raise NotImplementedError(f"Unsupported teleoperation device for rov_lab: {device}")

    for field in fields(action_cfg):
        value = getattr(action_cfg, field.name, None)
        if value is None or value is MISSING:
            raise ValueError(f"Action configuration '{field.name}' for {device} is not set")
    return action_cfg


def preprocess_device_action(action: dict[str, Any], teleop_device: Any) -> torch.Tensor:
    """Convert the teleoperation device output into the env action tensor."""
    if action.get("keyboard") is not None:
        joint_state = action["joint_state"]
        if joint_state.shape[-1] != KEYBOARD_ACTION_DIM:
            raise ValueError(f"Keyboard action must be {KEYBOARD_ACTION_DIM}-D, got {tuple(joint_state.shape)}")
        processed = torch.zeros(teleop_device.env.num_envs, KEYBOARD_ACTION_DIM, device=teleop_device.env.device)
        processed[:, :] = joint_state
        return processed
    raise NotImplementedError(f"Unsupported teleoperation device for rov_lab: {teleop_device.device_type}")
