# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from isaaclab.sensors.sensor_base_cfg import SensorBaseCfg
from .barometer import Barometer


@configclass
class BarometerCfg(SensorBaseCfg):
    """Configuration for an Barometer sensor."""

    class_type: type = Barometer

    @configclass
    class OffsetCfg:
        """The offset pose of the sensor's frame from the sensor's parent frame."""

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""

        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation (w, x, y, z) w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""

    offset: OffsetCfg = OffsetCfg()
    """The offset pose of the sensor's frame from the sensor's parent frame. Defaults to identity."""

    water_density: float = 1000.0
    """The density of water in kg/m^3. Defaults to 1000.0."""

    gravity: float = 9.81
    """The acceleration due to gravity in m/s^2. Defaults to 9.81."""

    noise_cov: float = 0.0
    """The noise of the sensor. Defaults to 0.0."""

    water_surface_z: float = 0.0
    """The z-coordinate of the water surface. Defaults to 0.0."""

    atmosphere_pressure: float  = 101325.0
    """The atmospheric pressure in Pascals. Defaults to 101325.0"""
