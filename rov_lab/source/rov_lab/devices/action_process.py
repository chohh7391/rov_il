"""ROV Lab teleoperation action configuration and preprocessing.

Maps a teleoperation device to the env's arm/gripper action terms and converts the device output into the
action tensor. The arm-specific parts (IK joints, end-effector body, gripper type) come from an
:class:`ArmTeleopSpec`, so the same device works with the SO-101 and the Bravo 7-class arm. Only the keyboard device
is supported for now; haptic devices will be added here.
"""

from __future__ import annotations

from dataclasses import MISSING, fields
from typing import Any

import torch
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.utils import configclass

from rov_lab.assets.robots.bravo7_class import (
    BRAVO7_CLASS_ARM_JOINT_NAMES,
    BRAVO7_CLASS_FINGER_JOINT_NAMES,
    JAW_BASE_LINK,
)

# Action layout produced by ROVKeyboard:
#   [0:6]  ROV body velocity command
#   [6:12] arm end-effector delta (position 3 + axis-angle 3)
#   [12]   auxiliary joint delta (e.g. SO-101 shoulder_pan; unused by arms whose IK covers every joint)
#   [13]   gripper command
KEYBOARD_ACTION_DIM: int = 14
KEYBOARD_AUX_SLOT: int = 12


@configclass
class ArmTeleopSpec:
    """Robot-specific mapping from the teleop layout to action terms."""

    ik_joint_names: list[str] = MISSING
    """Joints solved by the relative-pose differential IK."""

    ee_body_name: str = MISSING
    """Body whose pose the IK controls."""

    aux_joint_names: list[str] = []
    """Joints driven directly by the auxiliary slot (relative). Empty: the slot is dropped from the action."""

    gripper_joint_names: list[str] = MISSING
    """Gripper joint(s)."""

    gripper_type: str = "relative_joint"
    """``"relative_joint"``: one gripper joint, relative position (SO-101 jaw).
    ``"parallel"``: every finger joint gets the same opening (:class:`ROVParallelGripperAction`)."""

    gripper_scale: float = 1.0
    """Scale from the gripper slot to the gripper action term."""


SO101_TELEOP_SPEC = ArmTeleopSpec(
    ik_joint_names=["shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
    ee_body_name="gripper",
    aux_joint_names=["shoulder_pan"],
    gripper_joint_names=["gripper"],
    gripper_type="relative_joint",
)
"""SO-101 (5 joints + jaw): the 4-joint IK cannot yaw, so shoulder_pan is driven by the auxiliary slot."""

BRAVO7_TELEOP_SPEC = ArmTeleopSpec(
    ik_joint_names=list(BRAVO7_CLASS_ARM_JOINT_NAMES),
    ee_body_name=JAW_BASE_LINK,
    aux_joint_names=[],
    gripper_joint_names=list(BRAVO7_CLASS_FINGER_JOINT_NAMES),
    gripper_type="parallel",
    gripper_scale=0.01,  # keyboard slot +-0.15 per held step -> 1.5 mm finger travel per step
)
"""Bravo 7-class (6 rotary joints + parallel jaws): the 6-joint IK covers the full end-effector pose."""


def init_action_cfg(action_cfg: Any, device: str, spec: ArmTeleopSpec = SO101_TELEOP_SPEC) -> Any:
    """Set the arm and gripper action terms for ``device`` and check that every action term is configured."""
    if device != "keyboard":
        raise NotImplementedError(f"Unsupported teleoperation device for rov_lab: {device}")
    # imported here: rov_lab.tasks imports this module from its env configs (import cycle otherwise)
    from rov_lab.tasks.template.mdp import (
        ROVParallelGripperActionCfg,
        ROVRelativeIKActionCfg,
        ROVRelativeJointPositionActionCfg,
    )

    action_cfg.arm_action = ROVRelativeIKActionCfg(
        asset_name="robot",
        joint_names=list(spec.ik_joint_names),
        body_name=spec.ee_body_name,
        controller=DifferentialIKControllerCfg(command_type="pose", ik_method="dls", use_relative_mode=True),
    )
    if spec.gripper_type == "relative_joint":
        # aux joints and the gripper joint share one relative joint-position term (SO-101: pan + jaw)
        action_cfg.gripper_action = ROVRelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=list(spec.aux_joint_names) + list(spec.gripper_joint_names),
            scale=spec.gripper_scale,
        )
    elif spec.gripper_type == "parallel":
        if spec.aux_joint_names:
            raise NotImplementedError("aux joints are only supported with gripper_type='relative_joint'")
        action_cfg.gripper_action = ROVParallelGripperActionCfg(
            asset_name="robot",
            joint_names=list(spec.gripper_joint_names),
            mode="relative",
            scale=spec.gripper_scale,
        )
    else:
        raise ValueError(f"Unknown gripper_type: {spec.gripper_type}")

    for field in fields(action_cfg):
        value = getattr(action_cfg, field.name, None)
        if value is None or value is MISSING:
            raise ValueError(f"Action configuration '{field.name}' for {device} is not set")
    return action_cfg


def keyboard_to_env_action(joint_state: torch.Tensor, spec: ArmTeleopSpec) -> torch.Tensor:
    """Map the 14-D keyboard layout to the env action (drops the auxiliary slot when the arm has no aux joints)."""
    if joint_state.shape[-1] != KEYBOARD_ACTION_DIM:
        raise ValueError(f"Keyboard action must be {KEYBOARD_ACTION_DIM}-D, got {tuple(joint_state.shape)}")
    if spec.aux_joint_names:
        return joint_state
    keep = [i for i in range(KEYBOARD_ACTION_DIM) if i != KEYBOARD_AUX_SLOT]
    return joint_state[..., keep]


def preprocess_device_action(action: dict[str, Any], teleop_device: Any) -> torch.Tensor:
    """Convert the teleoperation device output into the env action tensor."""
    if action.get("keyboard") is not None:
        spec: ArmTeleopSpec = getattr(teleop_device.env.cfg, "arm_teleop_spec", SO101_TELEOP_SPEC)
        mapped = keyboard_to_env_action(action["joint_state"], spec)
        processed = torch.zeros(teleop_device.env.num_envs, mapped.shape[-1], device=teleop_device.env.device)
        processed[:, :] = mapped
        return processed
    raise NotImplementedError(f"Unsupported teleoperation device for rov_lab: {teleop_device.device_type}")
