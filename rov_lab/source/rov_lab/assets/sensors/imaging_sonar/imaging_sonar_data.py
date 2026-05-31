# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass

import torch

from isaaclab.sensors.camera.camera_data import CameraData


@dataclass
class ImagingSonarSensorData(CameraData):
    """Data container for the imaging sonar sensor.

    Extends :class:`CameraData` with sonar-specific outputs. The standard ``output`` dict
    inherited from :class:`CameraData` will contain the following keys (all keyed on env index):

    - ``"sonar_map"``: Polar sonar return in cartesian xy form. Shape (N, R, A, 3) torch.float32
      where R is number of range bins, A is number of azimuth bins. Channel 0 = x, 1 = y, 2 = intensity.
    - ``"sonar_image"``: Visualizable grayscale (RGBA) image of the polar sonar return.
      Shape (N, R, A, 4) torch.uint8.
    - ``"intensity"``: Raw per-bin normalized intensity after noise and normalization.
      Shape (N, R, A) torch.float32.

    Sonar geometry parameters are exposed below for convenience.
    """

    ##
    # Sonar geometry (static; mirrored from cfg for convenience)
    ##

    sonar_shape: tuple[int, int] = None
    """A tuple containing (num_range_bins, num_azimuth_bins) of the sonar polar map."""

    min_range: float = None
    """Minimum detection range in meters."""

    max_range: float = None
    """Maximum detection range in meters."""

    range_res: float = None
    """Range bin resolution in meters."""

    hori_fov: float = None
    """Horizontal field of view in degrees."""

    vert_fov: float = None
    """Vertical field of view in degrees."""

    angular_res: float = None
    """Azimuth bin resolution in degrees."""