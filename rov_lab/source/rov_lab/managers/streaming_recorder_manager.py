# Copyright (c) 2025, LeIsaac Project (Apache-2.0); adapted for rov_lab.
# SPDX-License-Identifier: Apache-2.0

"""Recorder manager that streams episodes to HDF5 while they are recorded.

Ported from ``leisaac.enhance.managers.recorder_manager`` for Isaac Lab 3.0. Episodes are flushed every
``flush_steps`` environment steps and finalized at reset. Only the "export all" modes are supported.
"""

from __future__ import annotations

import enum
import os
from collections.abc import Sequence
from typing import Any

import torch
from isaaclab.managers import DatasetExportMode, RecorderManager

from rov_lab.datasets import StreamingHDF5DatasetFileHandler, StreamWriteMode


class EnhanceDatasetExportMode(enum.IntEnum):
    """Export modes beyond :class:`DatasetExportMode`.

    ``EXPORT_ALL_RESUME`` reuses the value of ``DatasetExportMode.EXPORT_NONE``; exporting nothing is not
    supported while recording.
    """

    EXPORT_ALL_RESUME = 0
    EXPORT_SUCCEEDED_ONLY_RESUME = 4


def _env_id_list(env_ids: Sequence[int] | slice | torch.Tensor | None, num_envs: int) -> list[int]:
    if env_ids is None:
        return list(range(num_envs))
    if isinstance(env_ids, slice):
        return list(range(num_envs))[env_ids]
    if isinstance(env_ids, torch.Tensor):
        return env_ids.tolist()
    return list(env_ids)


class StreamingRecorderManager(RecorderManager):
    def __init__(self, cfg: Any, env: Any) -> None:
        cfg.dataset_file_handler_class_type = StreamingHDF5DatasetFileHandler
        super().__init__(cfg, env)
        if cfg.dataset_export_mode not in (DatasetExportMode.EXPORT_ALL, EnhanceDatasetExportMode.EXPORT_ALL_RESUME):
            raise ValueError("StreamingRecorderManager supports only EXPORT_ALL and EXPORT_ALL_RESUME")
        if cfg.dataset_export_mode == EnhanceDatasetExportMode.EXPORT_ALL_RESUME:
            # the base class creates no handler for this value; open the file in append mode here
            self._dataset_file_handler = StreamingHDF5DatasetFileHandler()
            self._dataset_file_handler.create(
                os.path.join(cfg.dataset_export_dir_path, cfg.dataset_filename), resume=True
            )
        self._env_steps_record = torch.zeros(self._env.num_envs)
        self._flush_steps: int = 100
        self._compression: str | None = None
        if self._dataset_file_handler is not None:
            self._dataset_file_handler.chunks_length = self._flush_steps
            self._dataset_file_handler.compression = self._compression

    @property
    def flush_steps(self) -> int:
        return self._flush_steps

    @flush_steps.setter
    def flush_steps(self, flush_steps: int) -> None:
        self._flush_steps = flush_steps
        if self._dataset_file_handler is not None:
            self._dataset_file_handler.chunks_length = flush_steps

    @property
    def compression(self) -> str | None:
        return self._compression

    @compression.setter
    def compression(self, compression: str | None) -> None:
        self._compression = compression
        if self._dataset_file_handler is not None:
            self._dataset_file_handler.compression = compression

    def __str__(self) -> str:
        return "[rov_lab] StreamingRecorderManager.\n" + super().__str__()

    def record_pre_step(self) -> None:
        self._env_steps_record += 1
        super().record_pre_step()
        self.export_episodes(from_step=True)

    def export_episodes(  # type: ignore[override]
        self,
        env_ids: Sequence[int] | slice | None = None,
        demo_ids: Sequence[int] | None = None,
        from_step: bool = False,
    ) -> None:
        if len(self.active_terms) == 0 or self._dataset_file_handler is None:
            return
        for env_id in _env_id_list(env_ids, self._env.num_envs):
            episode = self._episodes.get(env_id)
            if episode is None or episode.is_empty():
                continue
            if from_step and self._env_steps_record[env_id] < self._flush_steps:
                continue
            episode.pre_export()
            if self._env.cfg.seed is not None:
                episode.seed = self._env.cfg.seed
            write_mode = StreamWriteMode.APPEND if from_step else StreamWriteMode.LAST
            self._dataset_file_handler.write_episode(episode, write_mode)
            self._clear_episode_cache([env_id])
            if not from_step:
                counts = (
                    self._exported_successful_episode_count if episode.success else self._exported_failed_episode_count
                )
                counts[env_id] = counts.get(env_id, 0) + 1

    def _clear_episode_cache(self, env_ids: Sequence[int]) -> None:
        for env_id in env_ids:
            self._episodes[env_id].data = dict()
            self._env_steps_record[env_id] = 0

    def record_pre_reset(self, env_ids: Sequence[int] | slice | None, force_export_or_skip: bool | None = None) -> None:
        if len(self.active_terms) == 0:
            return
        ids = _env_id_list(env_ids, self._env.num_envs)
        for term in self._terms.values():
            key, value = term.record_pre_reset(ids)
            self.add_to_episodes(key, value, ids)
        success = torch.zeros(len(ids), dtype=torch.bool, device=self._env.device)
        if hasattr(self._env, "termination_manager") and "success" in self._env.termination_manager.active_terms:
            success |= self._env.termination_manager.get_term("success")[ids]
        self.set_success_to_episodes(ids, success)
        if force_export_or_skip or (force_export_or_skip is None and self.cfg.export_in_record_pre_reset):
            self.export_episodes(ids)
