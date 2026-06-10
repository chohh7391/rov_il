"""LeRobot v3 feature inference for pick-stone demonstrations."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import PickStoneDatasetConfig
from .hdf5_reader import Hdf5Episode


def infer_features(episode: Hdf5Episode, cfg: PickStoneDatasetConfig) -> dict[str, dict[str, Any]]:
    """Infer LeRobot feature metadata from one representative HDF5 episode."""
    action = episode.read(cfg.action_key)
    state = episode.read(cfg.state_key)

    if action.ndim != 2:
        raise ValueError(f"Expected action array with shape [T, A], got {action.shape}")
    if state.ndim != 2:
        raise ValueError(f"Expected state array with shape [T, S], got {state.shape}")

    action_dim = int(action.shape[-1])
    state_dim = int(state.shape[-1])

    features: dict[str, dict[str, Any]] = {
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": _names_or_dims(tuple(cfg.action_names), action_dim),
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": _names_or_dims(tuple(cfg.state_names), state_dim),
        },
    }

    for camera_key in cfg.camera_keys:
        hdf5_key = f"obs/{camera_key}"
        if not episode.has(hdf5_key):
            raise KeyError(f"Camera '{camera_key}' was requested, but '{hdf5_key}' is missing")
        image = episode.read(hdf5_key)
        if image.ndim != 4:
            raise ValueError(f"Expected camera '{camera_key}' shape [T, H, W, C], got {image.shape}")
        height, width, channels = (int(image.shape[1]), int(image.shape[2]), int(image.shape[3]))
        if channels not in (1, 3, 4):
            raise ValueError(f"Camera '{camera_key}' has unsupported channel count {channels}")
        stored_channels = 3 if channels == 4 else channels
        # LeRobot populates the per-video "info" (codec, pix_fmt, fps, ...) itself after the
        # actual encoding, so we only declare dtype/shape/names here. Pre-filling video info
        # would be ignored and could disagree with the auto-selected codec.
        features[f"observation.images.{camera_key}"] = {
            "dtype": "video" if cfg.use_videos else "image",
            "shape": [height, width, stored_channels],
            "names": ["height", "width", "channels"],
        }

    return features


def normalize_image_channels(image: np.ndarray) -> np.ndarray:
    """Convert Isaac/RGX images into video-encodable uint8 HWC frames."""
    if image.ndim != 3:
        raise ValueError(f"Expected image frame shape [H, W, C], got {image.shape}")
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] == 1:
        image = np.repeat(image, repeats=3, axis=-1)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _names_or_dims(names: tuple[str, ...], dim: int) -> list[str]:
    if len(names) == dim:
        return list(names)
    return [f"dim_{idx}" for idx in range(dim)]
