# Copyright (c) 2022-2026, The Isaac Lab Project Developers ...
# SPDX-License-Identifier: BSD-3-Clause
"""4-beam Janus pattern for the DVL sensor."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg
from isaaclab.utils import configclass


def dvl_janus_pattern(cfg: "DVLJanusPatternCfg", device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate 4 Janus-configured beam rays.

    All beams originate at the sensor origin. Each beam's nominal forward axis is -Z (down),
    rotated by Euler angles ``(rx, ry, rz)`` (deg, XYZ) matching the original Isaac Sim DVL:

        beam 0:  (+elevation,            0,  rotation)
        beam 1:  (         0,  +elevation,  rotation)
        beam 2:  (-elevation,            0,  rotation)
        beam 3:  (         0,  -elevation,  rotation)

    Returns:
        ray_starts:     (4, 3) all zeros (origin)
        ray_directions: (4, 3) unit vectors in the sensor body frame
    """
    elev = math.radians(cfg.elevation)
    rot = math.radians(cfg.rotation)

    sE, cE = math.sin(elev), math.cos(elev)
    sR, cR = math.sin(rot), math.cos(rot)

    # -Z rotated by Rx(±elev) or Ry(±elev):
    #   Rx(+e) @ -Z = (0,  +sE, -cE)
    #   Ry(+e) @ -Z = (-sE,  0, -cE)
    #   Rx(-e) @ -Z = (0,  -sE, -cE)
    #   Ry(-e) @ -Z = (+sE,  0, -cE)
    pre = torch.tensor(
        [
            [0.0,   sE, -cE],
            [-sE,  0.0, -cE],
            [0.0,  -sE, -cE],
            [ sE,  0.0, -cE],
        ],
        device=device,
    )
    Rz = torch.tensor(
        [
            [cR, -sR, 0.0],
            [sR,  cR, 0.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
    )
    ray_directions = (Rz @ pre.T).T.contiguous()   # (4, 3)
    ray_starts = torch.zeros_like(ray_directions)  # (4, 3)
    return ray_starts, ray_directions


@configclass
class DVLJanusPatternCfg(PatternBaseCfg):
    """Pattern config for the DVL 4-beam Janus configuration."""

    func = dvl_janus_pattern

    elevation: float = 22.5
    """Beam elevation angle from horizontal in degrees."""

    rotation: float = 45.0
    """Beam rotation about Z-axis in degrees."""