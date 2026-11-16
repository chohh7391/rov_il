"""Simulator-side helpers for the valve-panel IL experiment (import inside the Isaac Lab app context)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

import isaaclab.utils.math as math_utils

import valve_io as V


def make_state_fn(env: Any) -> Callable[[], np.ndarray]:
    """Return a function giving the 16-D policy state of env 0 (see ``valve_io.STATE_NAMES``)."""
    robot = env.scene["robot"]
    arm_ids = robot.find_joints(list(V.ARM_JOINTS), preserve_order=True)[0]
    finger_ids = robot.find_joints("finger_.*_joint")[0]
    ee_id = robot.find_bodies("jaw_base_link")[0][0]

    def state_vec() -> np.ndarray:
        jp = robot.data.joint_pos.torch[0]
        pos_b, quat_b = math_utils.subtract_frame_transforms(
            robot.data.root_pos_w.torch, robot.data.root_quat_w.torch,
            robot.data.body_pos_w.torch[:, ee_id], robot.data.body_quat_w.torch[:, ee_id])
        return np.concatenate([jp[arm_ids].cpu().numpy(), [float(jp[finger_ids].mean())], pos_b[0].cpu().numpy(),
                               V.quat_xyzw_to_rot6d(quat_b[0].cpu().numpy())]).astype(np.float32)

    return state_vec
