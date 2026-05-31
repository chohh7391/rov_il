# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass

from isaaclab.sensors.camera.camera_data import CameraData


@dataclass
class UWCameraData(CameraData):
    """Data container for the underwater camera sensor.

    Extends :class:`CameraData`. The ``output`` dict inherited from :class:`CameraData`
    is populated with the following keys (all keyed on env index):

    - ``"uw_image"``: Underwater-rendered image after backscatter / attenuation are applied
      to the raw RGBA. Shape ``(N, H, W, 4)`` torch.uint8.
    - ``"raw_rgba"``: Raw LdrColor output before underwater post-processing.
      Shape ``(N, H, W, 4)`` torch.uint8.
    - ``"depth"``: Distance-to-camera depth in meters. Shape ``(N, H, W, 1)`` torch.float32.
      Values beyond ``cfg.spawn.clipping_range[1]`` are ``inf`` unless
      ``cfg.depth_clipping_behavior`` is set to ``"max"`` or ``"zero"``.

    where ``N`` is the number of envs, ``H``, ``W`` are the camera resolution.
    """

    pass