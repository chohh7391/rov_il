"""Configuration for converting ROV Lab pick-stone demos to LeRobot v3."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_TASK_DESCRIPTION = "Pick three stones and put them into the plate, then reset the arm to rest state."

DEFAULT_ACTION_NAMES = [
    "rov.vx",
    "rov.vy",
    "rov.vz",
    "rov.wx",
    "rov.wy",
    "rov.wz",
    "arm.dx",
    "arm.dy",
    "arm.dz",
    "arm.droll",
    "arm.dpitch",
    "arm.dyaw",
    "arm.shoulder_pan",
    "arm.gripper",
]

DEFAULT_STATE_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


@dataclass(frozen=True)
class PickStoneDatasetConfig:
    """Settings for a pick-stone LeRobot Dataset v3 conversion."""

    repo_id: str
    output_root: Path
    hdf5_files: tuple[Path, ...]
    fps: int = 30
    robot_type: str = "fixed_rov_single_arm"
    task: str = DEFAULT_TASK_DESCRIPTION
    action_key: str = "actions"
    state_key: str = "obs/joint_pos"
    camera_keys: tuple[str, ...] = ("front", "wrist", "sonar")
    skip_first_frames: int = 5
    min_episode_frames: int = 10
    require_success: bool = False
    overwrite: bool = False
    use_videos: bool = True
    vcodec: str = "auto"
    streaming_encoding: bool = False
    action_names: tuple[str, ...] = field(default_factory=lambda: tuple(DEFAULT_ACTION_NAMES))
    state_names: tuple[str, ...] = field(default_factory=lambda: tuple(DEFAULT_STATE_NAMES))

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.skip_first_frames < 0:
            raise ValueError(f"skip_first_frames must be non-negative, got {self.skip_first_frames}")
        if self.min_episode_frames <= 0:
            raise ValueError(f"min_episode_frames must be positive, got {self.min_episode_frames}")
        if not self.hdf5_files:
            raise ValueError("At least one HDF5 file is required.")

