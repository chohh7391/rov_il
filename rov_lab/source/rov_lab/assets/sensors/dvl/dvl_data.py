# Copyright (c) 2022-2026, The Isaac Lab Project Developers ...
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import torch

from isaaclab.sensors.ray_caster.ray_caster_data import RayCasterData


@dataclass
class DVLData(RayCasterData):
    """Data container for the DVL sensor.

    Inherits ``pos_w``, ``quat_w``, and ``ray_hits_w`` from :class:`RayCasterData`.
    """

    lin_vel_b: torch.Tensor = None
    """Linear velocity in the DVL (body) frame. Zeroed-out on dropout.

    Shape is (N, 3), where ``N`` is the number of environments.
    """

    lin_vel_fd_b: torch.Tensor = None
    """Frequency-dependent (latched) linear velocity in the DVL (body) frame.

    Holds the last latched value; only refreshed when elapsed time exceeds the per-env sensor period.
    Shape is (N, 3).
    """

    depth: torch.Tensor = None
    """Per-beam range in meters. NaN for beams with no hit.

    Shape is (N, 4).
    """

    depth_fd: torch.Tensor = None
    """Frequency-dependent (latched) per-beam range.

    Shape is (N, 4).
    """

    beam_hit: torch.Tensor = None
    """Per-beam hit flag (``bool``).

    Shape is (N, 4).
    """

    dt: torch.Tensor = None
    """Current sensor period (s) under the configured operating mode.

    Shape is (N, 1).
    """