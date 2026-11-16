"""Shared observation / action schema for the valve-panel IL experiment (numpy only; imported by the collector and
the evaluator in the Isaac Lab env, and by the LeRobot converter in the ``fire`` env).

* images: front 90x160, wrist 120x160 (4x average-pooled underwater images)
* state (16): arm joints 6, finger opening 1 (m), jaw position in the ROV base frame 3, jaw rotation 6D 6
* environment_state (5): target valve one-hot (ACT has no language input)
* action (10): next jaw position in the ROV base frame 3, jaw rotation 6D 6, gripper opening in [-1, 1] 1.
  Executed with the absolute-pose env ``ROVLab-I90Bravo7-ValvePanel-AbsPose-v0`` (rotation 6D -> quaternion).
"""

from __future__ import annotations

import numpy as np

ARM_JOINTS: tuple[str, ...] = ("base_yaw_joint", "shoulder_pitch_joint", "elbow_pitch_joint", "forearm_roll_joint",
                               "wrist_pitch_joint", "ee_roll_joint")
ROT6D: tuple[str, ...] = ("r00", "r10", "r20", "r01", "r11", "r21")
STATE_NAMES: list[str] = [*ARM_JOINTS, "finger_opening", "jaw_x", "jaw_y", "jaw_z", *(f"jaw_{n}" for n in ROT6D)]
TARGET_NAMES: list[str] = [f"target_valve_{k}" for k in range(1, 6)]
ACTION_NAMES: list[str] = ["cmd_jaw_x", "cmd_jaw_y", "cmd_jaw_z", *(f"cmd_jaw_{n}" for n in ROT6D), "cmd_gripper"]
INSTRUCTIONS: list[str] = [f"Open valve {k} on the subsea panel with a quarter turn." for k in range(1, 6)]
FRONT_POOL: int = 4
WRIST_POOL: int = 4


def quat_xyzw_to_rot6d(q: np.ndarray) -> np.ndarray:
    """(..., 4) xyzw quaternion -> (..., 6) first two rotation-matrix columns."""
    x, y, z, w = (q[..., i] for i in range(4))
    c0 = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], -1)
    c1 = np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], -1)
    return np.concatenate([c0, c1], -1)


def rot6d_to_matrix(r: np.ndarray) -> np.ndarray:
    """(..., 6) -> (..., 3, 3) rotation matrix (Gram-Schmidt on the two columns)."""
    a, b = r[..., :3], r[..., 3:6]
    c0 = a / np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-9)
    b = b - (c0 * b).sum(-1, keepdims=True) * c0
    c1 = b / np.linalg.norm(b, axis=-1, keepdims=True).clip(1e-9)
    return np.stack([c0, c1, np.cross(c0, c1)], -1)


def _matrix_to_quat_xyzw_single(m: np.ndarray) -> np.ndarray:
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4]
    else:
        k = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
        j, l = (k + 1) % 3, (k + 2) % 3
        s = np.sqrt(1.0 + m[k, k] - m[j, j] - m[l, l]) * 2
        q = [0.0, 0.0, 0.0, (m[l, j] - m[j, l]) / s]
        q[k] = s / 4
        q[j] = (m[j, k] + m[k, j]) / s
        q[l] = (m[l, k] + m[k, l]) / s
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def rot6d_to_quat_xyzw(r: np.ndarray) -> np.ndarray:
    """(..., 6) -> (..., 4) xyzw quaternion."""
    m = rot6d_to_matrix(np.asarray(r, dtype=np.float64))
    flat = m.reshape(-1, 3, 3)
    q = np.stack([_matrix_to_quat_xyzw_single(x) for x in flat])
    return q.reshape(m.shape[:-2] + (4,))


def pool(img: np.ndarray, k: int) -> np.ndarray:
    """(H, W, 3) uint8 -> (H/k, W/k, 3) uint8 average pooling."""
    h, w, c = img.shape
    return img.reshape(h // k, k, w // k, k, c).mean(axis=(1, 3)).round().clip(0, 255).astype(np.uint8)
