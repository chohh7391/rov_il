from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from rov_lerobot.lerobot_datasets.pick_stone.config import PickStoneDatasetConfig
from rov_lerobot.lerobot_datasets.pick_stone.converter import _add_episode_frames
from rov_lerobot.lerobot_datasets.pick_stone.hdf5_reader import Hdf5Episode


class StubDataset:
    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []

    def add_frame(self, frame: dict[str, Any]) -> None:
        self.frames.append(dict(frame))


def test_add_episode_frames_uses_per_episode_task_strings(tmp_path: Path) -> None:
    file_path = tmp_path / "demo.hdf5"
    _write_synthetic_hdf5(file_path, target_ids=(0, 2))
    dataset = StubDataset()
    cfg = _cfg(tmp_path, hdf5_file=file_path)

    with h5py.File(file_path, "r") as h5_file:
        red_episode = Hdf5Episode(file_path=file_path, name="demo_0", group=h5_file["data/demo_0"])
        blue_episode = Hdf5Episode(file_path=file_path, name="demo_1", group=h5_file["data/demo_1"])

        assert _add_episode_frames(dataset, red_episode, cfg) == 3
        assert _add_episode_frames(dataset, blue_episode, cfg) == 3

    assert [frame["task"] for frame in dataset.frames[:3]] == ["pick up the red cube"] * 3
    assert [frame["task"] for frame in dataset.frames[3:]] == ["pick up the blue cube"] * 3


def test_missing_target_metadata_raises_without_fallback(tmp_path: Path) -> None:
    file_path = tmp_path / "demo.hdf5"
    _write_synthetic_hdf5(file_path, target_ids=(None,))
    dataset = StubDataset()
    cfg = _cfg(tmp_path, hdf5_file=file_path)

    with h5py.File(file_path, "r") as h5_file:
        episode = Hdf5Episode(file_path=file_path, name="demo_0", group=h5_file["data/demo_0"])
        with pytest.raises(KeyError, match="missing target color metadata"):
            _add_episode_frames(dataset, episode, cfg)


def test_missing_target_metadata_can_use_fixed_task_fallback(tmp_path: Path) -> None:
    file_path = tmp_path / "demo.hdf5"
    _write_synthetic_hdf5(file_path, target_ids=(None,))
    dataset = StubDataset()
    cfg = _cfg(
        tmp_path,
        hdf5_file=file_path,
        task="legacy fixed task",
        allow_fixed_task_fallback=True,
    )

    with h5py.File(file_path, "r") as h5_file:
        episode = Hdf5Episode(file_path=file_path, name="demo_0", group=h5_file["data/demo_0"])
        assert _add_episode_frames(dataset, episode, cfg) == 3

    assert [frame["task"] for frame in dataset.frames] == ["legacy fixed task"] * 3


def _cfg(
    tmp_path: Path,
    hdf5_file: Path,
    task: str = "unused fixed task",
    allow_fixed_task_fallback: bool = False,
) -> PickStoneDatasetConfig:
    return PickStoneDatasetConfig(
        repo_id="rov_il/test_pick_stone",
        output_root=tmp_path / "out",
        hdf5_files=(hdf5_file,),
        task=task,
        action_key="actions",
        state_key="obs/joint_pos",
        camera_keys=("front",),
        skip_first_frames=0,
        min_episode_frames=1,
        allow_fixed_task_fallback=allow_fixed_task_fallback,
    )


def _write_synthetic_hdf5(file_path: Path, target_ids: tuple[int | None, ...]) -> None:
    with h5py.File(file_path, "w") as h5_file:
        data_group = h5_file.create_group("data")
        for episode_idx, target_id in enumerate(target_ids):
            group = data_group.create_group(f"demo_{episode_idx}")
            group.create_dataset("actions", data=np.ones((3, 2), dtype=np.float32))
            obs_group = group.create_group("obs")
            obs_group.create_dataset("joint_pos", data=np.ones((3, 2), dtype=np.float32))
            obs_group.create_dataset("front", data=np.ones((3, 2, 2, 3), dtype=np.uint8))
            if target_id is not None:
                task_group = group.create_group("task")
                task_group.create_dataset("target_color_id", data=np.full((3,), target_id, dtype=np.int64))
