# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Literal

from isaaclab.sensors.camera.camera_cfg import CameraCfg
from isaaclab.utils import configclass

from .imaging_sonar import ImagingSonarSensor


@configclass
class ImagingSonarSensorCfg(CameraCfg):
    """Configuration for an imaging sonar sensor.

    The imaging sonar is implemented on top of a USD pinhole camera (one per environment) whose
    rendered frustum matches the sonar's horizontal / vertical FOV. Per-pixel raytraced returns
    are converted to a polar sonar map via Warp kernels. Each environment in the scene receives
    its own sonar instance, mirroring the standard Isaac Lab :class:`Camera` semantics.

    Defaults match the Oculus M370s / MT370s / MD370s family.

    .. note::
        - ``width`` and ``height`` from :class:`CameraCfg` are overridden by ``hori_res`` /
          ``vert_fov`` / ``hori_fov``. They are computed automatically in
          :meth:`ImagingSonarSensor.__init__` and the values you set on ``width`` / ``height``
          are ignored.
        - ``data_types`` from :class:`CameraCfg` is ignored. The sonar uses its own annotator
          set (``pointcloud``, ``CameraParams``, ``semantic_segmentation``) attached internally.
        - The semantic_segmentation annotator is set with ``colorize=False`` so the raw label
          ids are available for material-reflectivity lookup. ``semantic_filter`` from the
          parent cfg still applies.
    """

    class_type: type = ImagingSonarSensor

    # -- override parent defaults: sonar does not use the standard data_types pipeline
    data_types: list[str] = []

    # -- override parent defaults: width / height are auto-computed from sonar FOV / res
    width: int = 1  # placeholder, overridden in ImagingSonarSensor.__init__
    height: int = 1  # placeholder, overridden in ImagingSonarSensor.__init__

    ##
    # Sonar physical parameters
    ##

    min_range: float = 0.2
    """Minimum detection range in meters. Default 0.2 (Oculus M370s datasheet)."""

    max_range: float = 3.0
    """Maximum detection range in meters. Default 3.0."""

    range_res: float = 0.008
    """Range resolution in meters. Default 0.008 (Oculus M370s datasheet)."""

    hori_fov: float = 130.0
    """Horizontal field of view in degrees. Default 130.0."""

    vert_fov: float = 20.0
    """Vertical field of view in degrees. Default 20.0."""

    angular_res: float = 0.5
    """Azimuth bin resolution in degrees. Default 0.5."""

    hori_res: int = 3000
    """Horizontal raytrace resolution (pixels).

    Vertical resolution is automatically derived as ``int(hori_res / (hori_fov / vert_fov))``
    so that the pinhole render matches the sonar's FOV aspect ratio under Isaac Sim's
    square-pixel constraint.
    """

    include_unlabelled: bool = False
    """Whether to include unlabelled prims in the pointcloud annotator. Default False.

    Unlabelled returns are dropped by the sonar's intensity computation (background and
    unlabelled ids map to zero reflectivity).
    """

    ##
    # Sonar processing parameters
    ##

    binning_method: Literal["sum", "mean"] = "sum"
    """How to collapse per-ray intensities falling into the same (range, azimuth) bin.

    - ``"sum"``: accumulate intensity (default, follows the original ImagingSonarSensor).
    - ``"mean"``: average over ray count.

    Noise scales should be re-tuned when switching between these.
    """

    normalizing_method: Literal["all", "range"] = "range"
    """How to normalize binned intensity before image generation.

    - ``"all"``: divide by the global max intensity across the whole sonar frame.
    - ``"range"``: divide by the per-range-bin max (default; emphasizes far returns).
    """

    attenuation: float = 0.1
    """Distance attenuation coefficient applied when computing per-ray intensity."""

    gau_noise_param: float = 0.2
    """Multiplicative Gaussian noise standard deviation."""

    ray_noise_param: float = 0.05
    """Additive range-dependent Rayleigh noise scale."""

    intensity_offset: float = 0.0
    """Constant offset applied to intensity after normalization."""

    intensity_gain: float = 1.0
    """Multiplicative gain applied to intensity after normalization."""

    central_peak: float = 2.0
    """Peak strength of the central-beam streak artifact."""

    central_std: float = 0.001
    """Spread (std) of the central-beam streak artifact."""