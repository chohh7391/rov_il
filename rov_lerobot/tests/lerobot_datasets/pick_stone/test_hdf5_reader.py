from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from rov_lerobot.lerobot_datasets.pick_stone.hdf5_reader import Hdf5Episode


def _episode(tmp_path: Path, values: list[int]) -> Hdf5Episode:
    file_path = tmp_path / "demo.hdf5"
    h5_file = h5py.File(file_path, "w")
    data_group = h5_file.create_group("data")
    episode_group = data_group.create_group("demo_0")
    episode_group.create_dataset("task/target_color_id", data=values)
    return Hdf5Episode(file_path=file_path, name="demo_0", group=episode_group)


def test_read_constant_int_returns_constant_value(tmp_path: Path) -> None:
    episode = _episode(tmp_path, [2, 2, 2])

    assert episode.read_constant_int("task/target_color_id") == 2
    episode.group.file.close()


def test_read_constant_int_rejects_non_constant_values(tmp_path: Path) -> None:
    episode = _episode(tmp_path, [1, 2])

    with pytest.raises(ValueError, match="must be constant"):
        episode.read_constant_int("task/target_color_id")
    episode.group.file.close()


def test_read_constant_int_rejects_empty_values(tmp_path: Path) -> None:
    episode = _episode(tmp_path, [])

    with pytest.raises(ValueError, match="is empty"):
        episode.read_constant_int("task/target_color_id")
    episode.group.file.close()
