# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.sensors.camera.camera_cfg import CameraCfg
from isaaclab.utils import configclass

from .uw_camera import UWCamera


@configclass
class UWCameraCfg(CameraCfg):
    """Configuration for an underwater camera sensor.

    The underwater camera is a USD pinhole camera with a per-pixel post-processing pass
    (the Oceansim ``UW_render`` Warp kernel) that applies depth-dependent backscatter
    and attenuation to the raw LDR color output. One instance is spawned per environment.

    .. note::
        - ``data_types`` from :class:`CameraCfg` is ignored. The UW camera uses its own
          annotator set (``LdrColor`` and ``distance_to_camera``) attached internally so
          that the kernel can consume the raw linear color and the depth in a single tick.
        - Default underwater parameters approximate the typical coastal-water values
          baked into the original Oceansim sensor.
    """

    class_type: type = UWCamera

    # the parent's data_types pipeline is unused; the UW camera owns its annotators.
    data_types: list[str] = []

    ##
    # Underwater rendering parameters.
    # All triples are in linear (R, G, B) order, matching the original Oceansim sensor.
    ##

    backscatter_value: tuple[float, float, float] = (0.0, 0.31, 0.24)
    """Ambient backscatter color (R, G, B) added uniformly to the scene."""

    atten_coeff: tuple[float, float, float] = (0.05, 0.05, 0.05)
    """Per-channel attenuation coefficients (R, G, B).

    Applied exponentially with depth: ``radiance *= exp(-atten_coeff * depth)``.
    Larger values → faster light loss (more turbid water).
    """

    backscatter_coeff: tuple[float, float, float] = (0.05, 0.05, 0.2)
    """Per-channel backscatter coefficients (R, G, B).

    Controls how strongly the ambient ``backscatter_value`` builds up with depth.
    The blue channel is typically the largest, reflecting how blue-shifted scattered
    light dominates in water.
    """