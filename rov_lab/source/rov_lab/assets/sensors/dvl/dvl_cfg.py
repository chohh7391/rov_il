# Copyright (c) 2022-2026, The Isaac Lab Project Developers ...
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.sensors.ray_caster.ray_caster_cfg import RayCasterCfg
from isaaclab.utils import configclass

from .dvl import DVL
from .dvl_pattern_cfg import DVLJanusPatternCfg


@configclass
class DVLCfg(RayCasterCfg):
    """Configuration for a Doppler Velocity Log (DVL) sensor.

    Inherits ``mesh_prim_paths``, ``offset``, ``ray_alignment``, ``drift_range``, and
    ``visualizer_cfg`` from :class:`RayCasterCfg`. The ray ``pattern_cfg`` defaults to the
    DVL Janus 4-beam pattern and is kept in sync with :attr:`elevation` / :attr:`rotation` via
    ``__post_init__``.
    """

    class_type: type = DVL

    # Auto-populated; users normally don't override this directly. Override elevation/rotation instead.
    pattern_cfg: DVLJanusPatternCfg = DVLJanusPatternCfg()

    # ------------------------------------------------------------------
    # Beam geometry
    # ------------------------------------------------------------------
    elevation: float = 22.5
    """Beam elevation angle from horizontal in degrees. Defaults to 22.5."""

    rotation: float = 45.0
    """Beam rotation about Z-axis in degrees. Defaults to 45 (Janus configuration)."""

    # ------------------------------------------------------------------
    # Range / dropout
    # ------------------------------------------------------------------
    min_range: float = 0.1
    """Minimum valid range in meters. Defaults to 0.1."""

    max_range: float = 100.0
    """Maximum valid range in meters. Defaults to 100.0.

    Forwarded to :attr:`RayCasterCfg.max_distance` in ``__post_init__``.
    """

    num_beams_out_range_threshold: int = 2
    """Number of lost beams before declaring dropout. Defaults to 2."""

    # ------------------------------------------------------------------
    # Noise (per-beam covariance, matching the original Isaac Sim implementation)
    # ------------------------------------------------------------------
    vel_cov: float = 0.0
    """Velocity measurement noise covariance (per-beam). Defaults to 0.0."""

    depth_cov: float = 0.0
    """Depth measurement noise covariance (per-beam). Defaults to 0.0."""

    # ------------------------------------------------------------------
    # Operating frequency
    # ------------------------------------------------------------------
    freq: float | None = None
    """Fixed operating frequency in Hz. If None, adaptive frequency is used. Defaults to None."""

    freq_bound: tuple[float, float] = (5.0, 100.0)
    """``(min_freq, max_freq)`` for adaptive operation in Hz."""

    freq_dependent_range_bound: tuple[float, float] = (7.5, 50.0)
    """``(min_range, max_range)`` for frequency adaptation in meters."""

    sound_speed: float = 1500.0
    """Speed of sound in water in m/s. Defaults to 1500.0."""

    def __post_init__(self):
        super().__post_init__()
        # Keep the underlying RayCaster pattern + max distance in sync with DVL fields.
        self.pattern_cfg = DVLJanusPatternCfg(elevation=self.elevation, rotation=self.rotation)
        self.max_distance = self.max_range
        # DVL beams are attached to the body, so full orientation matters.
        self.ray_alignment = "base"