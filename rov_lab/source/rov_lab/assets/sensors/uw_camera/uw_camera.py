# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab port of the Oceansim underwater camera sensor.

This module ports ``isaacsim.oceansim.sensors.UW_Camera`` to the Isaac Lab sensor framework.
The original sensor is a single-environment :class:`isaacsim.sensors.camera.Camera` subclass
driven by an explicit ``render()`` call; here we re-express it so that:

- one underwater camera is spawned per environment;
- the underwater post-process kernel runs once per env on every sensor tick;
- outputs (``uw_image``, ``raw_rgba``, ``depth``) live in ``self.data.output``;
- no UI / viewport code runs (headless-friendly for RL).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from isaaclab.sensors.camera.camera import Camera
from isaaclab.utils.array import convert_to_torch

# The underwater post-process kernel lives in the Oceansim extension.
from ..utils.uw_renderer_utils import UW_render

from .uw_camera_data import UWCameraData

if TYPE_CHECKING:
    from .uw_camera_cfg import UWCameraCfg


class UWCamera(Camera):
    """Underwater camera sensor implemented on top of a pinhole camera render product.

    Each environment owns one USD camera prim. We attach two Replicator annotators
    (``LdrColor`` for raw linear color, ``distance_to_camera`` for depth) to each render
    product, and on every sensor tick we run the Oceansim ``UW_render`` Warp kernel per
    environment to produce a depth-modulated underwater image.

    Outputs in :attr:`UWCameraData.output`:

    - ``"uw_image"``: ``(N, H, W, 4)`` torch.uint8 — underwater-rendered image.
    - ``"raw_rgba"``: ``(N, H, W, 4)`` torch.uint8 — pre-effect linear color from LdrColor.
    - ``"depth"``: ``(N, H, W, 1)`` torch.float32 — distance to camera (meters).
    """

    cfg: UWCameraCfg

    def __init__(self, cfg: UWCameraCfg):
        """Initialize the underwater camera sensor.

        Args:
            cfg: The configuration parameters.
        """
        # data_types is forced empty so the parent does not attach standard annotators;
        # we own the annotator set ourselves (see _initialize_impl).
        cfg.data_types = []

        super().__init__(cfg)

        # swap parent's CameraData for the UW-extended container.
        self._data = UWCameraData()

    """
    Properties
    """

    @property
    def data(self) -> UWCameraData:
        self._update_outdated_buffers()
        return self._data

    """
    Implementation -- Isaac Lab sensor lifecycle.
    """

    def _initialize_impl(self):
        """Initialize the UW camera: parent Camera setup + UW annotators + warp params."""
        import omni.replicator.core as rep

        # Parent _initialize_impl creates view, render products, frame buffer, etc.
        # data_types=[] means the parent attaches nothing to the render products; we do.
        super()._initialize_impl()

        # device-name string accepted by Replicator (e.g. "cuda" rather than "cuda:0")
        if "cuda" in self._device:
            self._rep_device_name = self._device.split(":")[0]
        else:
            self._rep_device_name = "cpu"

        # per-env annotators. LdrColor gives raw linear RGBA before tone-mapping; that's
        # what the UW physical model needs (attenuation is linear in radiance).
        self._rgba_annots: list = []
        self._depth_annots: list = []
        for render_prod_path in self._render_product_paths:
            rgba = rep.AnnotatorRegistry.get_annotator("LdrColor", device=self._rep_device_name)
            depth = rep.AnnotatorRegistry.get_annotator("distance_to_camera", device=self._rep_device_name)
            rgba.attach(render_prod_path)
            depth.attach(render_prod_path)
            self._rgba_annots.append(rgba)
            self._depth_annots.append(depth)

        # static UW kernel inputs — build once on this device.
        self._wp_backscatter_value = wp.vec3f(*self.cfg.backscatter_value)
        self._wp_atten_coeff = wp.vec3f(*self.cfg.atten_coeff)
        self._wp_backscatter_coeff = wp.vec3f(*self.cfg.backscatter_coeff)

        # pre-allocate persistent torch output buffers. Shapes are known a priori.
        H, W = self.cfg.height, self.cfg.width
        self._data.output = {
            "uw_image": torch.zeros((self._view.count, H, W, 4), device=self._device, dtype=torch.uint8),
            "raw_rgba": torch.zeros((self._view.count, H, W, 4), device=self._device, dtype=torch.uint8),
            "depth": torch.zeros((self._view.count, H, W, 1), device=self._device, dtype=torch.float32),
        }
        self._data.info = [
            {"uw_image": None, "raw_rgba": None, "depth": None} for _ in range(self._view.count)
        ]

        # per-env scratch buffers for the UW kernel output. Allocating once avoids
        # reallocation every tick.
        self._wp_uw_images: list[wp.array] = []
        for _ in range(self._view.count):
            # shape (H, W, 4) uint8 — matches LdrColor layout.
            self._wp_uw_images.append(
                wp.zeros(shape=(H, W, 4), dtype=wp.uint8, device=self._device)
            )

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Per-env UW camera tick. Renders the underwater image for each env in ``env_ids``."""
        self._frame[env_ids] += 1
        if self.cfg.update_latest_camera_pose:
            self._update_poses(env_ids)

        for env_id in env_ids:
            idx = int(env_id)
            self._process_env(idx)

    """
    Per-environment processing
    """

    def _process_env(self, env_id: int):
        """Run the UW render kernel for a single environment.

        Mirrors the original :meth:`render` flow: fetch raw_rgba + depth, run UW_render,
        write outputs.
        """
        raw_rgba: wp.array = self._rgba_annots[env_id].get_data()
        depth: wp.array = self._depth_annots[env_id].get_data()

        # During CUDA warmup the first few ticks may return empty arrays; leave previous
        # outputs untouched (matches original sensor's behavior).
        if raw_rgba.size == 0 or depth.size == 0:
            return

        uw_image = self._wp_uw_images[env_id]
        wp.launch(
            kernel=UW_render,
            dim=(self.cfg.height, self.cfg.width),  # (H, W) — kernel uses 2D thread layout
            inputs=[
                raw_rgba,
                depth,
                self._wp_backscatter_value,
                self._wp_atten_coeff,
                self._wp_backscatter_coeff,
            ],
            outputs=[uw_image],
            device=self._device,
        )

        # publish to torch buffers via zero-copy warp->torch bridge.
        self._data.output["uw_image"][env_id] = wp.to_torch(uw_image)
        self._data.output["raw_rgba"][env_id] = wp.to_torch(raw_rgba)
        # depth from the annotator is (H, W) float32; we keep an explicit channel axis
        # to match the IsaacLab Camera convention of (H, W, C).
        depth_torch = convert_to_torch(depth, device=self._device).view(self.cfg.height, self.cfg.width, 1)
        # apply parent's depth clipping behavior to stay consistent with stock Camera outputs.
        if self.cfg.depth_clipping_behavior != "none":
            far = self.cfg.spawn.clipping_range[1]
            depth_torch = torch.where(
                torch.isinf(depth_torch),
                torch.full_like(depth_torch, 0.0 if self.cfg.depth_clipping_behavior == "zero" else far),
                depth_torch,
            )
        self._data.output["depth"][env_id] = depth_torch

    """
    Lifecycle
    """

    def __del__(self):
        """Detach UW annotators on teardown, then defer to parent for the rest."""
        if hasattr(self, "_rgba_annots"):
            for rgba, depth, rpp in zip(
                self._rgba_annots, self._depth_annots, self._render_product_paths
            ):
                try:
                    rgba.detach([rpp])
                    depth.detach([rpp])
                except Exception:
                    # teardown-time detachment failures are non-fatal
                    pass
        super().__del__()

    def _invalidate_initialize_callback(self, event):
        """Invalidate views and clear UW annotator lists on stage close."""
        super()._invalidate_initialize_callback(event)
        self._rgba_annots = []
        self._depth_annots = []