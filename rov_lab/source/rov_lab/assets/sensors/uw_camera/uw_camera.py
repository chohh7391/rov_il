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
import math
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

        if self.cfg.debug_print:
            print(
                "[UWCamera] initialized:"
                f" envs={self._view.count}, device={self._device}, rep_device={self._rep_device_name},"
                f" resolution=({self.cfg.width}, {self.cfg.height}),"
                f" render_products={len(self._render_product_paths)},"
                f" backscatter_value={self.cfg.backscatter_value},"
                f" atten_coeff={self.cfg.atten_coeff},"
                f" backscatter_coeff={self.cfg.backscatter_coeff},"
                f" depth_clipping_behavior={self.cfg.depth_clipping_behavior},"
                f" clipping_range={self.cfg.spawn.clipping_range}"
            )
            for env_id, sensor_prim in enumerate(self._sensor_prims):
                focal_length = float(sensor_prim.GetFocalLengthAttr().Get())
                horizontal_aperture = float(sensor_prim.GetHorizontalApertureAttr().Get())
                vertical_aperture = float(sensor_prim.GetVerticalApertureAttr().Get())
                horizontal_fov = 2.0 * math.degrees(math.atan(horizontal_aperture / (2.0 * focal_length)))
                vertical_fov = 2.0 * math.degrees(math.atan(vertical_aperture / (2.0 * focal_length)))
                print(
                    f"[UWCamera][env={env_id}] usd_camera:"
                    f" focal_length={focal_length:.6g},"
                    f" horizontal_aperture={horizontal_aperture:.6g},"
                    f" vertical_aperture={vertical_aperture:.6g},"
                    f" horizontal_fov_deg={horizontal_fov:.3f},"
                    f" vertical_fov_deg={vertical_fov:.3f}"
                )

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
        frame = int(self._frame[env_id].item())
        should_print = self._should_debug_print(frame)

        # During CUDA warmup the first few ticks may return empty arrays; leave previous
        # outputs untouched (matches original sensor's behavior).
        if raw_rgba.size == 0 or depth.size == 0:
            if should_print:
                print(
                    f"[UWCamera][env={env_id}][frame={frame}] empty annotator data:"
                    f" raw_size={raw_rgba.size}, depth_size={depth.size}"
                )
            return

        if should_print:
            raw_torch = wp.to_torch(raw_rgba)
            depth_torch_raw = convert_to_torch(depth, device=self._device)
            print(
                f"[UWCamera][env={env_id}][frame={frame}] annotators:"
                f" pose={self._pose_summary(env_id)},"
                f" raw={self._tensor_summary(raw_torch)},"
                f" raw_channels={self._image_channel_summary(raw_torch)},"
                f" depth_raw={self._tensor_summary(depth_torch_raw)},"
                f" depth_distribution={self._depth_distribution_summary(depth_torch_raw)},"
                f" uw_model_at_depth={self._uw_model_depth_summary(depth_torch_raw)}"
            )

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
        uw_torch = wp.to_torch(uw_image)
        raw_torch = wp.to_torch(raw_rgba)
        self._data.output["uw_image"][env_id] = uw_torch
        self._data.output["raw_rgba"][env_id] = raw_torch
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

        if should_print:
            print(
                f"[UWCamera][env={env_id}][frame={frame}] outputs:"
                f" uw_image={self._tensor_summary(uw_torch)},"
                f" uw_channels={self._image_channel_summary(uw_torch)},"
                f" raw_to_uw_delta={self._image_delta_summary(raw_torch, uw_torch)},"
                f" samples={self._image_sample_summary(raw_torch, uw_torch, depth_torch)},"
                f" raw_buffer={self._tensor_summary(self._data.output['raw_rgba'][env_id])},"
                f" depth_buffer={self._tensor_summary(self._data.output['depth'][env_id])}"
            )

    def _should_debug_print(self, frame: int) -> bool:
        """Return whether debug diagnostics should be printed for a sensor frame."""
        if not self.cfg.debug_print:
            return False
        if frame <= self.cfg.debug_print_first_n_frames:
            return True
        return self.cfg.debug_print_interval > 0 and frame % self.cfg.debug_print_interval == 0

    def _tensor_summary(self, tensor: torch.Tensor) -> str:
        """Create a compact tensor diagnostic string for camera debug prints."""
        tensor = tensor.detach()
        shape = tuple(tensor.shape)
        if tensor.numel() == 0:
            return f"shape={shape}, dtype={tensor.dtype}, empty=True"

        if torch.is_floating_point(tensor):
            finite = torch.isfinite(tensor)
            finite_count = int(finite.sum().item())
            total = tensor.numel()
            if finite_count > 0:
                values = tensor[finite]
                min_value = float(values.min().item())
                max_value = float(values.max().item())
                mean_value = float(values.mean().item())
                return (
                    f"shape={shape}, dtype={tensor.dtype}, finite={finite_count}/{total},"
                    f" min={min_value:.6g}, max={max_value:.6g}, mean={mean_value:.6g}"
                )
            return f"shape={shape}, dtype={tensor.dtype}, finite=0/{total}"

        min_value = int(tensor.min().item())
        max_value = int(tensor.max().item())
        mean_value = float(tensor.float().mean().item())
        return (
            f"shape={shape}, dtype={tensor.dtype}, min={min_value},"
            f" max={max_value}, mean={mean_value:.3f}"
        )

    def _image_channel_summary(self, image: torch.Tensor) -> str:
        """Summarize RGB(A) image channels independently."""
        if image.numel() == 0 or image.ndim != 3:
            return "unavailable"

        channel_names = ("R", "G", "B", "A")
        summaries = []
        for channel_index in range(min(image.shape[-1], len(channel_names))):
            channel = image[..., channel_index].float()
            summaries.append(
                f"{channel_names[channel_index]}(min={int(channel.min().item())},"
                f" max={int(channel.max().item())}, mean={float(channel.mean().item()):.3f})"
            )
        return "[" + ", ".join(summaries) + "]"

    def _image_delta_summary(self, raw_image: torch.Tensor, uw_image: torch.Tensor) -> str:
        """Summarize RGB differences introduced by the UW renderer."""
        if raw_image.numel() == 0 or uw_image.numel() == 0:
            return "unavailable"

        raw_rgb = raw_image[..., :3].float()
        uw_rgb = uw_image[..., :3].float()
        delta = uw_rgb - raw_rgb
        abs_delta = delta.abs()
        return (
            f"rgb_mean_delta={delta.mean(dim=(0, 1)).detach().cpu().numpy()},"
            f" rgb_abs_delta_mean={abs_delta.mean(dim=(0, 1)).detach().cpu().numpy()},"
            f" rgb_abs_delta_max={abs_delta.amax(dim=(0, 1)).detach().cpu().numpy()}"
        )

    def _image_sample_summary(
        self, raw_image: torch.Tensor, uw_image: torch.Tensor, depth_image: torch.Tensor
    ) -> str:
        """Print a few representative pixels to catch channel/order/orientation issues."""
        if raw_image.numel() == 0 or uw_image.numel() == 0 or depth_image.numel() == 0:
            return "unavailable"

        height, width = raw_image.shape[:2]
        sample_points = {
            "top_left": (0, 0),
            "center": (height // 2, width // 2),
            "bottom_right": (height - 1, width - 1),
        }
        parts = []
        for name, (row, col) in sample_points.items():
            raw_rgb = raw_image[row, col, :3].detach().cpu().tolist()
            uw_rgb = uw_image[row, col, :3].detach().cpu().tolist()
            depth = float(depth_image[row, col, 0].detach().cpu().item())
            parts.append(f"{name}(depth={depth:.4g}, raw_rgb={raw_rgb}, uw_rgb={uw_rgb})")
        return "[" + ", ".join(parts) + "]"

    def _depth_distribution_summary(self, depth_image: torch.Tensor) -> str:
        """Summarize depth distribution to catch too-near or clipped depth frames."""
        if depth_image.numel() == 0:
            return "empty=True"

        depth = depth_image.detach().flatten()
        finite = torch.isfinite(depth)
        finite_count = int(finite.sum().item())
        if finite_count == 0:
            return f"finite=0/{depth.numel()}"

        depth = depth[finite].float()
        quantiles = torch.quantile(
            depth,
            torch.tensor([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99], device=depth.device),
        )
        near_1m = float((depth < 1.0).float().mean().item()) * 100.0
        near_3m = float((depth < 3.0).float().mean().item()) * 100.0
        far_10m = float((depth > 10.0).float().mean().item()) * 100.0
        q = quantiles.detach().cpu().numpy()
        return (
            f"p01={q[0]:.3g}, p05={q[1]:.3g}, p25={q[2]:.3g},"
            f" p50={q[3]:.3g}, p75={q[4]:.3g}, p95={q[5]:.3g}, p99={q[6]:.3g},"
            f" pct_lt_1m={near_1m:.2f}, pct_lt_3m={near_3m:.2f}, pct_gt_10m={far_10m:.2f}"
        )

    def _uw_model_depth_summary(self, depth_image: torch.Tensor) -> str:
        """Show the UW model coefficients implied by the current depth distribution."""
        if depth_image.numel() == 0:
            return "unavailable"

        depth = depth_image.detach().flatten()
        finite = torch.isfinite(depth)
        if int(finite.sum().item()) == 0:
            return "unavailable"

        depth = depth[finite].float()
        sample_depths = torch.stack(
            [
                torch.quantile(depth, 0.25),
                torch.quantile(depth, 0.50),
                torch.quantile(depth, 0.75),
            ]
        )
        atten = torch.tensor(self.cfg.atten_coeff, device=depth.device, dtype=torch.float32)
        back = torch.tensor(self.cfg.backscatter_coeff, device=depth.device, dtype=torch.float32)
        backscatter_value = torch.tensor(self.cfg.backscatter_value, device=depth.device, dtype=torch.float32)
        atten_factor = torch.exp(-sample_depths[:, None] * atten[None, :])
        backscatter_add = backscatter_value[None, :] * 255.0 * (1.0 - torch.exp(-sample_depths[:, None] * back[None, :]))

        parts = []
        for label, depth_value, atten_value, backscatter_value_at_depth in zip(
            ("p25", "p50", "p75"), sample_depths, atten_factor, backscatter_add
        ):
            parts.append(
                f"{label}(d={float(depth_value.item()):.3g},"
                f" atten_rgb={atten_value.detach().cpu().numpy()},"
                f" backscatter_add_rgb={backscatter_value_at_depth.detach().cpu().numpy()})"
            )
        return "[" + ", ".join(parts) + "]"

    def _pose_summary(self, env_id: int) -> str:
        """Summarize camera pose in the conventions Isaac Lab exposes."""
        pos_w = self._data.pos_w[env_id].detach().cpu().numpy()
        quat_world = self._data.quat_w_world[env_id].detach().cpu().numpy()
        quat_ros = self._data.quat_w_ros[env_id].detach().cpu().numpy()
        quat_opengl = self._data.quat_w_opengl[env_id].detach().cpu().numpy()
        return (
            f"pos_w={pos_w},"
            f" quat_world={quat_world},"
            f" quat_ros={quat_ros},"
            f" quat_opengl={quat_opengl}"
        )

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
