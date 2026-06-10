"""Small HDF5 reader for Isaac Lab style ROV Lab demonstration files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np


@dataclass(frozen=True)
class Hdf5Episode:
    """A lightweight view over one HDF5 episode group."""

    file_path: Path
    name: str
    group: h5py.Group

    @property
    def success(self) -> bool | None:
        if "success" not in self.group.attrs:
            return None
        return bool(self.group.attrs["success"])

    @property
    def num_samples(self) -> int | None:
        if "num_samples" not in self.group.attrs:
            return None
        return int(self.group.attrs["num_samples"])

    def read(self, key: str) -> np.ndarray:
        """Read a slash-delimited dataset key from this episode."""
        if key not in self.group:
            raise KeyError(f"{self.file_path}:{self.name} does not contain dataset '{key}'")
        value = self.group[key]
        if not isinstance(value, h5py.Dataset):
            raise KeyError(f"{self.file_path}:{self.name}:{key} is a group, not a dataset")
        return np.asarray(value)

    def has(self, key: str) -> bool:
        return key in self.group and isinstance(self.group[key], h5py.Dataset)

    def read_constant_int(self, key: str) -> int:
        """Read an episode metadata dataset that must contain one constant integer value."""
        values = np.ravel(self.read(key))
        if values.size == 0:
            raise ValueError(f"{self.file_path}:{self.name}:{key} is empty")
        first_value = values[0]
        if not np.all(values == first_value):
            raise ValueError(f"{self.file_path}:{self.name}:{key} must be constant within an episode")
        return int(first_value)


def iter_episodes(hdf5_files: tuple[Path, ...]) -> Iterator[tuple[h5py.File, Hdf5Episode]]:
    """Yield open HDF5 episode groups in deterministic file/name order.

    The yielded file handle must stay alive while the episode is read, so callers
    should consume the generator sequentially and not store episodes past an
    iteration.
    """
    for file_path in hdf5_files:
        with h5py.File(file_path, "r") as h5_file:
            if "data" not in h5_file:
                raise KeyError(f"{file_path} does not contain a top-level 'data' group")
            data_group = h5_file["data"]
            for episode_name in sorted(data_group.keys(), key=_episode_sort_key):
                group = data_group[episode_name]
                if isinstance(group, h5py.Group):
                    yield h5_file, Hdf5Episode(file_path=file_path, name=episode_name, group=group)


def _episode_sort_key(name: str) -> tuple[int, str]:
    prefix = "demo_"
    if name.startswith(prefix):
        suffix = name[len(prefix) :]
        if suffix.isdigit():
            return int(suffix), name
    return 1_000_000_000, name
