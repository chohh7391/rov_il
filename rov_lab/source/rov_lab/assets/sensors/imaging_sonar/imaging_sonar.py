# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab port of the Oceansim imaging sonar sensor.

This module ports ``isaacsim.oceansim.sensors.ImagingSonarSensor`` to the Isaac Lab sensor
framework. The original sensor inherits from the single-environment ``isaacsim.sensors.camera.Camera``
and is driven by explicit ``scan()`` / ``make_sonar_data()`` calls. Here we re-express it as an
Isaac Lab :class:`Camera` subclass so that:

- one sonar instance is spawned per environment (e.g. ``/World/envs/env_.*/Robot/Sonar``);
- all envs are stepped in parallel through :meth:`_update_buffers_impl`;
- outputs live in ``self.data.output`` like any other Isaac Lab camera sensor;
- no UI / viewport code runs (headless-friendly for RL).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors.camera.camera import Camera

# Warp kernels live in the Oceansim extension; importing them keeps the kernel definitions
# identical to the upstream implementation.
from ..utils.imaging_sonar_kernels import *

from .imaging_sonar_data import ImagingSonarSensorData

if TYPE_CHECKING:
    from .imaging_sonar_cfg import ImagingSonarSensorCfg


class ImagingSonarSensor(Camera):
    """Imaging sonar sensor implemented on top of a pinhole camera render product.

    Each environment owns one USD camera prim configured to match the sonar's horizontal and
    vertical FOV. We attach three Replicator annotators (``pointcloud``, ``CameraParams``,
    ``semantic_segmentation``) to each render product, and on every sensor tick we run the
    original Oceansim Warp kernels (intensity computation, world→local, polar binning, noise
    injection, normalization, image generation) per environment.

    The processed outputs are written into :attr:`ImagingSonarSensorData.output`:

    - ``"sonar_map"``: ``(N, R, A, 3)`` torch.float32 cartesian polar map (x, y, intensity).
    - ``"sonar_image"``: ``(N, R, A, 4)`` torch.uint8 visualizable RGBA image.
    - ``"intensity"``: ``(N, R, A)`` torch.float32 raw normalized intensity.

    where ``N`` is the number of envs, ``R`` is the number of range bins, and ``A`` is the
    number of azimuth bins.
    """

    cfg: ImagingSonarSensorCfg

    def __init__(self, cfg: ImagingSonarSensorCfg):
        """Initialize the imaging sonar sensor.

        Args:
            cfg: The configuration parameters.
        """
        # -- derive render resolution from sonar FOV before the parent spawns the prim.
        #    Isaac Sim only supports square pixels, so we fix horizontal resolution and
        #    scale vertical resolution by the FOV aspect ratio. The horizontal aperture is
        #    re-set after init() to actually match hori_fov (see end of __init__).
        #    The aspect ratio must be the ratio of FOV *tangents*, not of the raw angles: with
        #    square pixels and the horizontal aperture matched to hori_fov, the vertical FOV is
        #    determined by height/width = tan(vert_fov/2)/tan(hori_fov/2). Using the angle ratio
        #    only approximates this for small angles and, at hori_fov=130 deg, renders a far too
        #    tall vertical FOV (e.g. ~36 deg instead of 20 deg).
        self._aspect_ratio = float(
            np.tan(np.deg2rad(cfg.hori_fov) / 2.0) / np.tan(np.deg2rad(cfg.vert_fov) / 2.0)
        )
        cfg.width = int(cfg.hori_res)
        cfg.height = int(cfg.hori_res / self._aspect_ratio)

        # -- precompute the polar (range, azimuth) meshgrid.
        #    These are env-independent and shared across all sonars.
        self._min_azi_rad = np.deg2rad(90.0 - cfg.hori_fov / 2.0)
        self._max_azi_rad = np.deg2rad(90.0 + cfg.hori_fov / 2.0)
        self._angular_res_rad = np.deg2rad(cfg.angular_res)

        r_np, azi_np = np.meshgrid(
            np.arange(cfg.min_range, cfg.max_range, cfg.range_res),
            np.arange(self._min_azi_rad, self._max_azi_rad, self._angular_res_rad),
            indexing="ij",
        )
        self._sonar_shape: tuple[int, int] = r_np.shape  # (num_range_bins, num_azimuth_bins)

        # We delay the actual creation of warp buffers until _initialize_impl, when we know
        # the device. But we keep the numpy meshgrid around to upload per-env copies.
        self._r_np = r_np.astype(np.float32)
        self._azi_np = azi_np.astype(np.float32)
        self._debug_line_endpoints_b_np = self._make_debug_line_endpoints_b(cfg).astype(np.float32)

        # -- override the parent ``depth_clipping_behavior`` semantics: the sonar uses
        #    pointcloud annotator directly, so the parent's depth post-processing does not
        #    apply. Force ``data_types`` to be empty so the parent does not allocate buffers
        #    for unused annotators.
        cfg.data_types = []

        # -- initialize parent (Camera). This spawns the USD camera prim, creates the view,
        #    and would normally build annotators -- but with data_types=[] no annotators are
        #    attached. We attach our own in _initialize_impl below.
        super().__init__(cfg)

        # replace the parent's CameraData with the sonar-extended container.
        self._data = ImagingSonarSensorData()

    """
    Properties
    """

    @property
    def data(self) -> ImagingSonarSensorData:
        # update sensors if needed
        if not self._is_initialized:
            return self._data
        self._update_outdated_buffers()
        return self._data

    @property
    def sonar_shape(self) -> tuple[int, int]:
        """Tuple of (num_range_bins, num_azimuth_bins)."""
        return self._sonar_shape

    """
    Implementation -- Isaac Lab sensor lifecycle.
    """

    def _initialize_impl(self):
        """Initialize the sonar sensor: parent Camera setup + sonar annotators + warp buffers."""
        import omni.replicator.core as rep

        # -- Parent _initialize_impl creates the view, render products, frame buffer, etc.
        #    Because cfg.data_types is empty, no Replicator annotators are attached to the
        #    render products inside the parent. We attach pointcloud / CameraParams /
        #    semantic_segmentation ourselves below.
        super()._initialize_impl()

        # device-name string accepted by Replicator (e.g. "cuda" rather than "cuda:0")
        if "cuda" in self._device:
            self._rep_device_name = self._device.split(":")[0]
        else:
            self._rep_device_name = "cpu"

        # -- per-env annotators. Each env has its own render product, so each annotator
        #    triple is indexed by env id.
        self._pointcloud_annots: list = []
        self._cam_params_annots: list = []
        self._sem_seg_annots: list = []

        for render_prod_path in self._render_product_paths:
            pc = rep.AnnotatorRegistry.get_annotator(
                name="pointcloud",
                init_params={"includeUnlabelled": self.cfg.include_unlabelled},
                device=self._rep_device_name,
            )
            cp = rep.AnnotatorRegistry.get_annotator(
                name="CameraParams",
                device=self._rep_device_name,
            )
            ss = rep.AnnotatorRegistry.get_annotator(
                name="semantic_segmentation",
                init_params={"colorize": False},
                device=self._rep_device_name,
            )
            pc.attach(render_prod_path)
            cp.attach(render_prod_path)
            ss.attach(render_prod_path)
            self._pointcloud_annots.append(pc)
            self._cam_params_annots.append(cp)
            self._sem_seg_annots.append(ss)

        # -- correct the horizontal aperture so the rendered frustum actually matches
        #    hori_fov. Isaac Sim's pinhole camera fixes vertical aperture given aspect
        #    ratio (for square pixels), so we recompute horizontal aperture from focal
        #    length here. Reference:
        #    https://forums.developer.nvidia.com/t/how-to-modify-the-cameras-field-of-view/278427/5
        from pxr import UsdGeom  # local import to avoid module-load issues
        import omni.usd

        for sensor_prim in self._sensor_prims:
            focal_length = sensor_prim.GetFocalLengthAttr().Get()
            horizontal_aper = 2.0 * focal_length * float(np.tan(np.deg2rad(self.cfg.hori_fov) / 2.0))
            omni.usd.set_prop_val(sensor_prim.GetHorizontalApertureAttr(), float(horizontal_aper))

        # -- warp buffers (per env). Each env keeps its own (R, A)-shaped working buffers
        #    so kernels can run independently without inter-env interference.
        self._wp_r: list[wp.array] = []
        self._wp_azi: list[wp.array] = []
        self._wp_bin_sum: list[wp.array] = []
        self._wp_bin_count: list[wp.array] = []
        self._wp_binned_intensity: list[wp.array] = []
        self._wp_gau_noise: list[wp.array] = []
        self._wp_range_dep_ray_noise: list[wp.array] = []
        self._wp_sonar_map: list[wp.array] = []
        self._wp_sonar_image: list[wp.array] = []

        for _ in range(self._view.count):
            self._wp_r.append(wp.array(self._r_np, dtype=wp.float32, device=self._device))
            self._wp_azi.append(wp.array(self._azi_np, dtype=wp.float32, device=self._device))
            self._wp_bin_sum.append(wp.zeros(shape=self._sonar_shape, dtype=wp.float32, device=self._device))
            self._wp_bin_count.append(wp.zeros(shape=self._sonar_shape, dtype=wp.int32, device=self._device))
            self._wp_binned_intensity.append(
                wp.zeros(shape=self._sonar_shape, dtype=wp.float32, device=self._device)
            )
            self._wp_gau_noise.append(wp.zeros(shape=self._sonar_shape, dtype=wp.float32, device=self._device))
            self._wp_range_dep_ray_noise.append(
                wp.zeros(shape=self._sonar_shape, dtype=wp.float32, device=self._device)
            )
            self._wp_sonar_map.append(wp.zeros(shape=self._sonar_shape, dtype=wp.vec3, device=self._device))
            self._wp_sonar_image.append(
                wp.zeros(shape=(self._sonar_shape[0], self._sonar_shape[1], 4), dtype=wp.uint8, device=self._device)
            )

        # -- allocate the persistent torch output buffers. Unlike the parent Camera, we
        #    know our output shapes a priori, so we can pre-allocate instead of lazy-alloc.
        R, A = self._sonar_shape
        self._data.output = {
            "sonar_map": torch.zeros((self._view.count, R, A, 3), device=self._device, dtype=torch.float32),
            "sonar_image": torch.zeros((self._view.count, R, A, 4), device=self._device, dtype=torch.uint8),
            "intensity": torch.zeros((self._view.count, R, A), device=self._device, dtype=torch.float32),
        }
        self._data.info = [
            {"sonar_map": None, "sonar_image": None, "intensity": None} for _ in range(self._view.count)
        ]

        # -- mirror static sonar geometry into data for downstream consumers.
        self._data.sonar_shape = self._sonar_shape
        self._data.min_range = self.cfg.min_range
        self._data.max_range = self.cfg.max_range
        self._data.range_res = self.cfg.range_res
        self._data.hori_fov = self.cfg.hori_fov
        self._data.vert_fov = self.cfg.vert_fov
        self._data.angular_res = self.cfg.angular_res
        self._debug_line_endpoints_b = torch.tensor(
            self._debug_line_endpoints_b_np, device=self._device, dtype=torch.float32
        )
        self._debug_line_orientations_b = self._make_debug_line_orientations_b(self._debug_line_endpoints_b)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        # annotator가 아직 attach되지 않은 경우 skip
        if not self._is_initialized or not self._pointcloud_annots:
            return

        super()._update_buffers_impl(env_ids)

        for env_id in env_ids:
            idx = int(env_id)
            self._process_env(idx)

    """
    Per-environment processing
    """

    def _process_env(self, env_id: int):
        """Run the sonar pipeline for a single environment.

        Mirrors the original :meth:`make_sonar_data` flow: scan -> intensity -> world2local
        -> binning -> noise -> normalization -> sonar_map -> sonar_image.
        """
        # -- (1) fetch raw annotator data; bail out cleanly when there is no return yet.
        try:
            sem_data = self._sem_seg_annots[env_id].get_data()
        except (KeyError, Exception):
            return
        id_to_labels = sem_data.get("info", {}).get("idToLabels", {})
        if len(id_to_labels) == 0:
            return
        
        try:
            pc_data = self._pointcloud_annots[env_id].get_data(device=self._device)
        except (KeyError, Exception):
            return
        pcl: wp.array = pc_data["data"]
        normals: wp.array = pc_data["info"]["pointNormals"]
        semantics: wp.array = pc_data["info"]["pointSemantic"]

        cam_params = self._cam_params_annots[env_id].get_data()
        view_transform_np = np.asarray(cam_params["cameraViewTransform"]).reshape(4, 4).T
        view_transform = wp.mat44(view_transform_np)

        num_points = pcl.shape[0]
        if num_points == 0:
            return

        # -- (2) build indexToReflectivity lookup from idToLabels (matches original helper).
        index_to_refl_np = self._make_index_to_prop_array(id_to_labels, query_property="reflectivity")
        index_to_refl = wp.array(index_to_refl_np, dtype=wp.float32, device=self._device)

        # -- (3) per-ray intensity
        intensity = wp.empty(shape=(num_points,), dtype=wp.float32, device=self._device)
        wp.launch(
            kernel=compute_intensity,  # noqa: F405 (from kernels star-import)
            dim=num_points,
            inputs=[pcl, normals, view_transform, semantics, index_to_refl, self.cfg.attenuation],
            outputs=[intensity],
            device=self._device,
        )

        # -- (4) world -> sonar-local + spherical coordinates
        pcl_local = wp.empty(shape=(num_points,), dtype=wp.vec3, device=self._device)
        pcl_spher = wp.empty(shape=(num_points,), dtype=wp.vec3, device=self._device)
        wp.launch(
            kernel=world2local,  # noqa: F405
            dim=num_points,
            inputs=[view_transform, pcl],
            outputs=[pcl_local, pcl_spher],
            device=self._device,
        )

        # -- (5) bin into (range, azimuth) accumulator
        bin_sum = self._wp_bin_sum[env_id]
        bin_count = self._wp_bin_count[env_id]
        binned_intensity = self._wp_binned_intensity[env_id]
        bin_sum.zero_()
        bin_count.zero_()
        binned_intensity.zero_()

        wp.launch(
            kernel=bin_intensity,  # noqa: F405
            dim=num_points,
            inputs=[
                pcl_spher,
                intensity,
                self.cfg.min_range,
                self._min_azi_rad,
                self.cfg.range_res,
                self._angular_res_rad,
            ],
            outputs=[bin_sum, bin_count],
            device=self._device,
        )

        if self.cfg.binning_method == "mean":
            wp.launch(
                kernel=average,  # noqa: F405
                dim=bin_sum.shape,
                inputs=[bin_sum, bin_count],
                outputs=[binned_intensity],
                device=self._device,
            )
        else:  # "sum"
            # mimic original: just reuse the running sum buffer
            binned_intensity = bin_sum

        # -- (6) noise generation
        gau_noise = self._wp_gau_noise[env_id]
        range_dep_ray_noise = self._wp_range_dep_ray_noise[env_id]
        gau_noise.zero_()
        range_dep_ray_noise.zero_()

        frame_seed = int(self._frame[env_id].item())

        wp.launch(
            kernel=normal_2d,  # noqa: F405
            dim=bin_sum.shape,
            inputs=[frame_seed, 0.0, self.cfg.gau_noise_param],
            outputs=[gau_noise],
            device=self._device,
        )
        wp.launch(
            kernel=range_dependent_rayleigh_2d,  # noqa: F405
            dim=bin_sum.shape,
            inputs=[
                frame_seed,
                self._wp_r[env_id],
                self._wp_azi[env_id],
                self.cfg.max_range,
                self.cfg.ray_noise_param,
                self.cfg.central_peak,
                self.cfg.central_std,
            ],
            outputs=[range_dep_ray_noise],
            device=self._device,
        )

        # -- (7) normalization + polar->cartesian sonar map
        sonar_map = self._wp_sonar_map[env_id]
        sonar_map.zero_()

        if self.cfg.normalizing_method == "all":
            maximum = wp.zeros(shape=(1,), dtype=wp.float32, device=self._device)
            wp.launch(
                dim=bin_sum.shape,
                kernel=all_max,  # noqa: F405
                inputs=[binned_intensity],
                outputs=[maximum],
                device=self._device,
            )
            wp.launch(
                kernel=make_sonar_map_all,  # noqa: F405
                dim=sonar_map.shape,
                inputs=[
                    self._wp_r[env_id],
                    self._wp_azi[env_id],
                    binned_intensity,
                    maximum,
                    gau_noise,
                    range_dep_ray_noise,
                    self.cfg.intensity_offset,
                    self.cfg.intensity_gain,
                ],
                outputs=[sonar_map],
                device=self._device,
            )
        else:  # "range"
            maximum = wp.zeros(shape=(self._sonar_shape[0],), dtype=wp.float32, device=self._device)
            wp.launch(
                dim=bin_sum.shape,
                kernel=range_max,  # noqa: F405
                inputs=[binned_intensity],
                outputs=[maximum],
                device=self._device,
            )
            wp.launch(
                kernel=make_sonar_map_range,  # noqa: F405
                dim=sonar_map.shape,
                inputs=[
                    self._wp_r[env_id],
                    self._wp_azi[env_id],
                    binned_intensity,
                    maximum,
                    gau_noise,
                    range_dep_ray_noise,
                    self.cfg.intensity_offset,
                    self.cfg.intensity_gain,
                ],
                outputs=[sonar_map],
                device=self._device,
            )

        # -- (8) sonar image (uint8 RGBA)
        sonar_image = self._wp_sonar_image[env_id]
        sonar_image.zero_()
        wp.launch(
            dim=sonar_map.shape,
            kernel=make_sonar_image,  # noqa: F405
            inputs=[sonar_map],
            outputs=[sonar_image],
            device=self._device,
        )

        # -- (9) publish to torch buffers via zero-copy warp->torch bridge
        self._data.output["sonar_map"][env_id] = wp.to_torch(sonar_map)
        self._data.output["sonar_image"][env_id] = wp.to_torch(sonar_image)
        # intensity is the 3rd channel of sonar_map (post-normalization, post-noise)
        self._data.output["intensity"][env_id] = self._data.output["sonar_map"][env_id, ..., 2]

    """
    Debug visualization
    """

    @staticmethod
    def _make_debug_line_endpoints_b(cfg: "ImagingSonarSensorCfg") -> np.ndarray:
        """Create OpenGL camera-frame endpoints for the four sonar FOV corner rays."""
        half_h = np.deg2rad(cfg.hori_fov * 0.5)
        half_v = np.deg2rad(cfg.vert_fov * 0.5)
        debug_range = 1.0
        endpoints: list[np.ndarray] = []
        for h_sign, v_sign in (
            (1.0, 1.0),
            (1.0, -1.0),
            (-1.0, 1.0),
            (-1.0, -1.0),
        ):
            direction = np.array(
                [
                    h_sign * np.tan(half_h),
                    v_sign * np.tan(half_v),
                    -1.0,
                ],
                dtype=np.float32,
            )
            direction = direction / np.linalg.norm(direction)
            endpoints.append(debug_range * direction)
        return np.stack(endpoints, axis=0)

    @staticmethod
    def _make_debug_line_orientations_b(line_endpoints_b: torch.Tensor) -> torch.Tensor:
        """Return local quaternions rotating a +X cylinder into each line direction."""
        x_axis = torch.zeros_like(line_endpoints_b)
        x_axis[:, 0] = 1.0
        directions = math_utils.normalize(line_endpoints_b)
        axes = torch.linalg.cross(x_axis, directions, dim=1)
        axis_norm = torch.linalg.norm(axes, dim=1, keepdim=True)
        axes = torch.where(axis_norm > 1.0e-6, axes / axis_norm, torch.tensor([0.0, 0.0, 1.0], device=axes.device))
        angles = torch.acos(torch.clamp(torch.sum(x_axis * directions, dim=1), -1.0, 1.0))
        return math_utils.quat_from_angle_axis(angles, axes)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "sonar_visualizer"):
                marker_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/ROVLab/SonarFOV",
                    markers={
                        "line": sim_utils.CylinderCfg(
                            radius=0.01,
                            height=1.0,
                            axis="X",
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(1.0, 1.0, 1.0),
                                emissive_color=(0.4, 0.4, 0.4),
                            ),
                        )
                    },
                )
                self.sonar_visualizer = VisualizationMarkers(marker_cfg)
            self.sonar_visualizer.set_visibility(True)
        elif hasattr(self, "sonar_visualizer"):
            self.sonar_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self._is_initialized or self._data.pos_w is None or self._data.quat_w_world is None:
            return

        # Do not call _update_poses() here. This callback is executed from the
        # post-update event stream, and querying XformPrimView world poses can
        # trigger a Fabric/USD sync that recursively emits another update event.
        # The sensor update path already refreshes pos_w / quat_w_world when
        # debug visualization is enabled.
        num_lines = self._debug_line_endpoints_b.shape[0]
        endpoints_b = self._debug_line_endpoints_b.unsqueeze(0).repeat(self._view.count, 1, 1)
        quat_w = self._data.quat_w_opengl.unsqueeze(1).repeat(1, num_lines, 1)
        midpoints_w = math_utils.quat_apply(quat_w, 0.5 * endpoints_b) + self._data.pos_w.unsqueeze(1)

        line_orientations_b = self._debug_line_orientations_b.unsqueeze(0).repeat(self._view.count, 1, 1)
        line_orientations_w = math_utils.quat_mul(quat_w.reshape(-1, 4), line_orientations_b.reshape(-1, 4))
        line_lengths = torch.linalg.norm(endpoints_b.reshape(-1, 3), dim=1)
        line_scales = torch.ones((self._view.count * num_lines, 3), device=self._device)
        line_scales[:, 0] = line_lengths

        self.sonar_visualizer.visualize(
            midpoints_w.reshape(-1, 3),
            line_orientations_w,
            line_scales,
        )

    """
    Helpers
    """

    @staticmethod
    def _make_index_to_prop_array(id_to_labels: dict, query_property: str) -> np.ndarray:
        """Build the index->property lookup array consumed by the intensity kernel.

        Mirrors the inner helper in the original ``make_sonar_data``: returns a 1D float32
        array where ``arr[id] = value`` if ``id_to_labels[id]`` carries ``query_property``,
        otherwise ``arr[id] = 1.0``. Entries for BACKGROUND (id 0) and UNLABELLED (id 1)
        are kept at 1.0 then implicitly zeroed downstream by the kernel masking.
        """
        if len(id_to_labels) == 0:
            return np.ones((1,), dtype=np.float32)
        max_id = max(int(k) for k in id_to_labels.keys())
        arr = np.ones((max_id + 1,), dtype=np.float32)
        for raw_id, label_dict in id_to_labels.items():
            i = int(raw_id)
            if query_property in label_dict:
                arr[i] = float(label_dict[query_property])
        return arr

    """
    Lifecycle
    """

    def __del__(self):
        """Detach sonar annotators on teardown, then defer to parent for the rest."""
        # detach sonar-owned annotators
        if hasattr(self, "_pointcloud_annots"):
            for pc, cp, ss, rpp in zip(
                self._pointcloud_annots,
                self._cam_params_annots,
                self._sem_seg_annots,
                self._render_product_paths,
            ):
                try:
                    pc.detach([rpp])
                    cp.detach([rpp])
                    ss.detach([rpp])
                except Exception:
                    # detachment failures during teardown are non-fatal
                    pass
        # parent handles _rep_registry (empty in our case) and the rest
        super().__del__()

    def _invalidate_initialize_callback(self, event):
        """Invalidate views and clear sonar annotator lists on stage close."""
        super()._invalidate_initialize_callback(event)
        self._pointcloud_annots = []
        self._cam_params_annots = []
        self._sem_seg_annots = []
