"""Convert ROV Lab pick-stone HDF5 demonstrations to LeRobot Dataset v3."""

from __future__ import annotations

import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterator

import numpy as np

from .config import PickStoneDatasetConfig
from .features import infer_features, normalize_image_channels
from .hdf5_reader import Hdf5Episode, iter_episodes


@dataclass(frozen=True)
class ConversionSummary:
    """Result counts from a pick-stone conversion."""

    output_root: Path
    episodes_written: int
    episodes_skipped: int
    frames_written: int


def convert_pick_stone_hdf5(cfg: PickStoneDatasetConfig) -> ConversionSummary:
    """Convert configured HDF5 files into one LeRobot Dataset v3 directory."""
    _validate_inputs(cfg)
    _validate_success_availability(cfg)
    _prepare_output_root(cfg)

    features = _infer_features_from_first_usable_episode(cfg)

    try:
        with _without_repo_root_shadowing():
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise ImportError(
            "lerobot and its learning dependencies are required for conversion. Install lerobot==0.4.4 "
            "or add external_dependencies/lerobot/src to PYTHONPATH, and make sure the Hugging Face "
            "'datasets' package is installed."
        ) from exc

    dataset = LeRobotDataset.create(
        repo_id=cfg.repo_id,
        root=cfg.output_root,
        fps=cfg.fps,
        robot_type=cfg.robot_type,
        features=features,
        use_videos=cfg.use_videos,
        vcodec=cfg.vcodec,
        streaming_encoding=cfg.streaming_encoding,
    )

    episodes_written = 0
    episodes_skipped = 0
    frames_written = 0

    for _, episode in iter_episodes(cfg.hdf5_files):
        if _should_skip_episode(episode, cfg):
            episodes_skipped += 1
            continue

        try:
            episode_frames = _add_episode_frames(dataset, episode, cfg)
        except Exception:
            dataset.clear_episode_buffer()
            raise

        if episode_frames == 0:
            dataset.clear_episode_buffer()
            episodes_skipped += 1
            continue

        dataset.save_episode(parallel_encoding=False)
        episodes_written += 1
        frames_written += episode_frames

    dataset.finalize()

    return ConversionSummary(
        output_root=cfg.output_root,
        episodes_written=episodes_written,
        episodes_skipped=episodes_skipped,
        frames_written=frames_written,
    )


def _validate_inputs(cfg: PickStoneDatasetConfig) -> None:
    for hdf5_file in cfg.hdf5_files:
        if not hdf5_file.is_file():
            raise FileNotFoundError(f"HDF5 file does not exist: {hdf5_file}")


def _validate_success_availability(cfg: PickStoneDatasetConfig) -> None:
    """Fail fast with an actionable message when success filtering is impossible.

    ROV Lab recordings do not always carry a per-episode ``success`` attribute. If the
    user asked to keep only successful episodes but none of the episodes expose that flag,
    every episode would silently be skipped and conversion would fail later with an opaque
    "no usable episodes" error. Detect that case up front instead.
    """
    if not cfg.require_success:
        return
    for _, episode in iter_episodes(cfg.hdf5_files):
        if episode.success is not None:
            return
    raise ValueError(
        "--require-success was set, but none of the episodes in the provided HDF5 files "
        "carry a 'success' attribute. Re-record demonstrations with success flags or drop "
        "--require-success."
    )


def _prepare_output_root(cfg: PickStoneDatasetConfig) -> None:
    if cfg.output_root.exists():
        if not cfg.overwrite:
            raise FileExistsError(
                f"Output root already exists: {cfg.output_root}. Use overwrite=True or --overwrite."
            )
        shutil.rmtree(cfg.output_root)


def _infer_features_from_first_usable_episode(cfg: PickStoneDatasetConfig) -> dict:
    for _, episode in iter_episodes(cfg.hdf5_files):
        if not _should_skip_episode(episode, cfg):
            return infer_features(episode, cfg)
    raise ValueError("No usable episodes found in the provided HDF5 files.")


def _should_skip_episode(episode: Hdf5Episode, cfg: PickStoneDatasetConfig) -> bool:
    if cfg.require_success and episode.success is not True:
        return True
    if not episode.has(cfg.action_key) or not episode.has(cfg.state_key):
        return True
    action_len = int(episode.read(cfg.action_key).shape[0])
    state_len = int(episode.read(cfg.state_key).shape[0])
    usable_len = min(action_len, state_len)
    for camera_key in cfg.camera_keys:
        hdf5_key = f"obs/{camera_key}"
        if not episode.has(hdf5_key):
            return True
        usable_len = min(usable_len, int(episode.read(hdf5_key).shape[0]))
    usable_len -= cfg.skip_first_frames
    return usable_len < cfg.min_episode_frames


def _add_episode_frames(dataset: object, episode: Hdf5Episode, cfg: PickStoneDatasetConfig) -> int:
    actions = _as_float32(episode.read(cfg.action_key))
    states = _as_float32(episode.read(cfg.state_key))
    images = {camera_key: episode.read(f"obs/{camera_key}") for camera_key in cfg.camera_keys}
    episode_task = _resolve_episode_task(episode, cfg)

    episode_len = min([len(actions), len(states), *[len(value) for value in images.values()]])
    start = min(cfg.skip_first_frames, episode_len)
    written = 0

    for frame_idx in range(start, episode_len):
        frame: dict[str, object] = {
            "action": actions[frame_idx],
            "observation.state": states[frame_idx],
            "task": episode_task,
        }
        for camera_key, frames in images.items():
            frame[f"observation.images.{camera_key}"] = normalize_image_channels(frames[frame_idx])
        dataset.add_frame(frame=frame)
        written += 1

    return written


def _resolve_episode_task(episode: Hdf5Episode, cfg: PickStoneDatasetConfig) -> str:
    try:
        target_color_id = episode.read_constant_int(cfg.target_color_key)
    except KeyError as exc:
        if cfg.allow_fixed_task_fallback:
            return cfg.task
        raise KeyError(
            f"{episode.file_path}:{episode.name} is missing target color metadata "
            f"'{cfg.target_color_key}'. Re-record with task metadata or set "
            "allow_fixed_task_fallback=True to use cfg.task."
        ) from exc

    if not 0 <= target_color_id < len(cfg.instruction_templates):
        raise ValueError(
            f"{episode.file_path}:{episode.name}:{cfg.target_color_key} has invalid target id "
            f"{target_color_id}; expected 0 <= id < {len(cfg.instruction_templates)}"
        )
    return cfg.instruction_templates[target_color_id]


def _as_float32(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.float32:
        return np.ascontiguousarray(array)
    return np.ascontiguousarray(array.astype(np.float32))


@contextmanager
def _without_repo_root_shadowing() -> Iterator[None]:
    """Avoid local ./datasets shadowing the Hugging Face datasets package."""
    repo_root = Path(__file__).resolve().parents[3]
    original_sys_path = list(sys.path)
    bad_datasets_module = _local_datasets_module(sys.modules.get("datasets"), repo_root)
    if bad_datasets_module:
        sys.modules.pop("datasets", None)
    sys.path = [
        entry
        for entry in sys.path
        if entry not in ("", str(repo_root)) and Path(entry or ".").resolve() != repo_root
    ]
    try:
        yield
    finally:
        sys.path = original_sys_path


def _local_datasets_module(module: ModuleType | None, repo_root: Path) -> bool:
    if module is None:
        return False
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        module_paths = getattr(module, "__path__", [])
        return any(Path(path).resolve().is_relative_to(repo_root / "datasets") for path in module_paths)
    return Path(module_file).resolve().is_relative_to(repo_root / "datasets")
