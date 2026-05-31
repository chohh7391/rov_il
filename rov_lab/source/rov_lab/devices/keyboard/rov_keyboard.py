"""Keyboard teleoperation for a 6-DOF ROV plus one SO101 arm."""

from __future__ import annotations

import carb
import torch
import numpy as np
import isaaclab.utils.math as math_utils
from leisaac.devices.device_base import Device
from leisaac.utils.math_utils import rotvec_to_euler


class ROVKeyboard(Device):
    """Keyboard controller that emits a 14D action for ROV + single arm tasks."""

    def __init__(self, env, sensitivity: float = 1.0) -> None:
        super().__init__(env, "keyboard")

        self.rov_sensitivity = 1.0 * sensitivity
        self.pos_sensitivity = 0.01 * sensitivity
        self.joint_sensitivity = 0.15 * sensitivity
        self.rot_sensitivity = 0.15 * sensitivity

        self._create_key_bindings()
        self._delta_action = np.zeros(14)

        self.asset_name = "robot"
        self.robot_asset = self.env.scene[self.asset_name]

        self.target_frame = "gripper"
        body_idxs, _ = self.robot_asset.find_bodies(self.target_frame)
        self.target_frame_idx = body_idxs[0]
        self.base_frame = "rov_base"
        base_body_idxs, _ = self.robot_asset.find_bodies(self.base_frame)
        self.base_frame_idx = base_body_idxs[0]

    def _add_device_control_description(self) -> None:
        self._display_controls_table.add_row(["UP", "rov_forward"])
        self._display_controls_table.add_row(["DOWN", "rov_backward"])
        self._display_controls_table.add_row(["LEFT", "rov_left"])
        self._display_controls_table.add_row(["RIGHT", "rov_right"])
        self._display_controls_table.add_row(["W", "forward"])
        self._display_controls_table.add_row(["S", "backward"])
        self._display_controls_table.add_row(["A", "left"])
        self._display_controls_table.add_row(["D", "right"])
        self._display_controls_table.add_row(["Q", "up"])
        self._display_controls_table.add_row(["E", "down"])
        self._display_controls_table.add_row(["J", "rotate_left"])
        self._display_controls_table.add_row(["L", "rotate_right"])
        self._display_controls_table.add_row(["K", "rotate_up"])
        self._display_controls_table.add_row(["I", "rotate_down"])
        self._display_controls_table.add_row(["U", "gripper_open"])
        self._display_controls_table.add_row(["O", "gripper_close"])

    def _convert_delta_from_frame(self, delta_action: np.ndarray) -> np.ndarray:
        """
        Convert delta action from target frame to ROV base frame.
        target_frame -> rov_base frame
        Args:
            delta_action: Delta action in target frame.
        Returns:
            Delta action in ROV base frame.
        """
        if np.allclose(delta_action[:3], 0.0) and np.allclose(delta_action[3:6], 0.0):
            return delta_action
        is_delta_rot = not np.allclose(delta_action[3:6], 0.0)

        torch_delta_action = torch.tensor(delta_action, device=self.env.device, dtype=torch.float32)

        delta_pos_f = torch_delta_action[:3].repeat(self.env.num_envs, 1)
        delta_rot_f = torch_delta_action[3:6].repeat(self.env.num_envs, 1)
        delta_quat_f = math_utils.quat_from_euler_xyz(delta_rot_f[:, 0], delta_rot_f[:, 1], delta_rot_f[:, 2])
        delta_rotvec_f = math_utils.axis_angle_from_quat(delta_quat_f)

        # The arm teleoperation command is expressed in the current gripper frame.
        # Convert it to the physical ROV hull frame, not the articulation root frame.
        # This keeps arm commands tied to the moving ROV base.
        base_pos_w = self.robot_asset.data.body_pos_w[:, self.base_frame_idx]
        base_quat_w = self.robot_asset.data.body_quat_w[:, self.base_frame_idx]
        target_quat_w = self.robot_asset.data.body_quat_w[:, self.target_frame_idx]
        _, target_to_base_quat = math_utils.subtract_frame_transforms(
            base_pos_w,
            base_quat_w,
            base_pos_w,
            target_quat_w,
        )
        target_to_base_quat = math_utils.quat_unique(target_to_base_quat)

        delta_pos_b = math_utils.quat_apply(target_to_base_quat, delta_pos_f)
        delta_rotvec_b = math_utils.quat_apply(target_to_base_quat, delta_rotvec_f)
        delta_rot_b = (
            rotvec_to_euler(delta_rotvec_b)
            if is_delta_rot
            else torch.zeros(self.env.num_envs, 3, device=self.env.device)
        )

        delta_action_b = torch.cat([delta_pos_b[0], delta_rot_b[0], torch_delta_action[6:]], dim=0)

        return delta_action_b.cpu().numpy()

    def get_device_state(self) -> np.ndarray:
        rov_action = self._delta_action[:6]
        arm_action = self._delta_action[6:]
        converted_arm_action = self._convert_delta_from_frame(arm_action)
        return np.concatenate((rov_action, converted_arm_action))

    def reset(self) -> None:
        self._delta_action[:] = 0.0

    def _on_keyboard_event(self, event, *args, **kwargs) -> None:
        super()._on_keyboard_event(event, *args, **kwargs)
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._INPUT_KEY_MAPPING:
                self._delta_action += self._ACTION_DELTA_MAPPING[self._INPUT_KEY_MAPPING[event.input.name]]
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._INPUT_KEY_MAPPING:
                self._delta_action -= self._ACTION_DELTA_MAPPING[self._INPUT_KEY_MAPPING[event.input.name]]

    def _create_key_bindings(self) -> None:
        self._ACTION_DELTA_MAPPING = {
            "rov_forward": np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            * self.rov_sensitivity,
            "rov_backward": np.asarray(
                [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            )
            * self.rov_sensitivity,
            "rov_left": np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            * self.rov_sensitivity,
            "rov_right": np.asarray(
                [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            )
            * self.rov_sensitivity,
            "forward": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            * self.pos_sensitivity,
            "backward": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            * self.pos_sensitivity,
            "left": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0])
            * self.joint_sensitivity,
            "right": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            * self.joint_sensitivity,
            "up": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            * self.pos_sensitivity,
            "down": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            * self.pos_sensitivity,
            "rotate_up": np.asarray(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0]
            )
            * self.rot_sensitivity,
            "rotate_down": np.asarray(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            )
            * self.rot_sensitivity,
            "rotate_left": np.asarray(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            )
            * self.rot_sensitivity,
            "rotate_right": np.asarray(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0]
            )
            * self.rot_sensitivity,
            "gripper_open": np.asarray(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            )
            * self.joint_sensitivity,
            "gripper_close": np.asarray(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
            )
            * self.joint_sensitivity,
        }
        self._INPUT_KEY_MAPPING = {
            "UP": "rov_forward",
            "DOWN": "rov_backward",
            "LEFT": "rov_left",
            "RIGHT": "rov_right",
            "W": "forward",
            "S": "backward",
            "A": "left",
            "D": "right",
            "Q": "up",
            "E": "down",
            "K": "rotate_up",
            "I": "rotate_down",
            "J": "rotate_left",
            "L": "rotate_right",
            "U": "gripper_open",
            "O": "gripper_close",
        }
