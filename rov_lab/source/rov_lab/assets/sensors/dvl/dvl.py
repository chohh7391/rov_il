# Copyright (c) 2022-2026, The Isaac Lab Project Developers ...
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaacsim.core.simulation_manager import SimulationManager
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.sensors.ray_caster.ray_caster import RayCaster
from isaaclab.utils.warp import raycast_mesh

from .dvl_data import DVLData

if TYPE_CHECKING:
    from .dvl_cfg import DVLCfg


class DVL(RayCaster):
    """Doppler Velocity Log sensor (4-beam Janus configuration).

    Extends Isaac Lab's :class:`RayCaster` to add:

    - Linear velocity readout in the DVL body frame (via a separate ``RigidBodyView`` over the
      ray-caster's parent rigid body).
    - Per-beam range with NaN-on-miss semantics and a dropout flag.
    - Per-beam additive Gaussian noise on depth and Janus-mapped noise on velocity.
    - Frequency-dependent (latched) outputs ``depth_fd`` / ``lin_vel_fd_b`` with an adaptive
      per-env sensor period.
    """

    cfg: DVLCfg

    def __init__(self, cfg: DVLCfg):
        # RayCaster's __init__ sets up _data = RayCasterData(); we overwrite below.
        super().__init__(cfg)
        self._data = DVLData()

        # Will be created in _initialize_impl after RayCaster's setup.
        self._rb_view = None
        self._rigid_parent_expr: str | None = None

    def __str__(self) -> str:
        return (
            f"DVL sensor @ '{self.cfg.prim_path}': \n"
            f"\tview type            : {self._view.__class__}\n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tnumber of sensors    : {self._view.count}\n"
            f"\televation (deg)      : {self.cfg.elevation}\n"
            f"\trotation (deg)       : {self.cfg.rotation}\n"
        )

    """
    Properties
    """

    @property
    def data(self) -> DVLData:
        self._update_outdated_buffers()
        return self._data

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)

        self._data.lin_vel_b[env_ids] = 0.0
        self._data.lin_vel_fd_b[env_ids] = 0.0
        self._data.depth[env_ids] = float("nan")
        self._data.depth_fd[env_ids] = float("nan")
        self._data.beam_hit[env_ids] = False
        self._data.dt[env_ids] = 0.0

        self._elapsed_vel[env_ids] = 0.0
        self._elapsed_depth[env_ids] = 0.0

    def update(self, dt: float, force_recompute: bool = False):
        # save timestamp
        self._dt = dt
        # execute updating
        super().update(dt, force_recompute)

    """
    Implementation
    """

    def _initialize_impl(self):
        """Initialize RayCaster, then ensure a RigidBodyView is available for velocity."""
        super()._initialize_impl()
        # RayCaster._view may be RigidBodyView (most common when prim is on a body) or
        # XformPrimView / ArticulationView. We need linear velocities -> ensure RigidBodyView.
        import omni.physics.tensors.impl.api as physx

        if isinstance(self._view, physx.RigidBodyView):
            # Reuse the existing view -- avoids duplicate physics view allocation.
            self._rb_view = self._view
        else:
            # Fall back: locate the closest rigid-body ancestor and build a dedicated view.
            self._physics_sim_view = SimulationManager.get_physics_sim_view()
            prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
            if prim is None:
                raise RuntimeError(f"Failed to find a prim at path: {self.cfg.prim_path}")
            ancestor_prim = sim_utils.get_first_matching_ancestor_prim(
                prim.GetPath(),
                predicate=lambda _prim: _prim.HasAPI(UsdPhysics.RigidBodyAPI),
            )
            if ancestor_prim is None:
                raise RuntimeError(f"Failed to find a rigid body ancestor prim at path expression: {self.cfg.prim_path}")

            if ancestor_prim == prim:
                rb_expr = self.cfg.prim_path
            else:
                rel = prim.GetPath().MakeRelativePath(ancestor_prim.GetPath()).pathString
                rb_expr = self.cfg.prim_path.replace(rel, "")
            self._rb_view = self._physics_sim_view.create_rigid_body_view(
                rb_expr.replace(".*", "*")
            )

        if self._rb_view.count != self._view.count:
            raise RuntimeError(
                f"DVL rigid-body view count ({self._rb_view.count}) does not match raycaster "
                f"view count ({self._view.count})."
            )

        # Constants
        self._noise_std_vel = math.sqrt(self.cfg.vel_cov) if self.cfg.vel_cov > 0.0 else 0.0
        self._noise_std_dep = math.sqrt(self.cfg.depth_cov) if self.cfg.depth_cov > 0.0 else 0.0
        self._janus_transform = self._compute_janus_transform()

        self._adaptive_freq = self.cfg.freq is None
        if not self._adaptive_freq:
            self._fixed_dt = 1.0 / float(self.cfg.freq)

    def _initialize_rays_impl(self):
        # Let RayCaster do its standard ray/buffer setup (this will also create self._data.pos_w /
        # quat_w / ray_hits_w because DVLData inherits RayCasterData).
        super()._initialize_rays_impl()

        # DVL-specific buffers.
        N = self._view.count
        dev = self._device

        self._data.lin_vel_b = torch.zeros(N, 3, device=dev)
        self._data.lin_vel_fd_b = torch.zeros(N, 3, device=dev)
        self._data.depth = torch.full((N, 4), float("nan"), device=dev)
        self._data.depth_fd = torch.full((N, 4), float("nan"), device=dev)
        self._data.beam_hit = torch.zeros(N, 4, device=dev, dtype=torch.bool)
        self._data.dt = torch.zeros(N, 1, device=dev)

        self._elapsed_vel = torch.zeros(N, device=dev)
        self._elapsed_depth = torch.zeros(N, device=dev)

    def _compute_janus_transform(self) -> torch.Tensor:
        """3x4 matrix mapping 4 beam-space noise samples to body-frame velocity noise."""
        sE = math.sin(math.radians(self.cfg.elevation))
        cE = math.cos(math.radians(self.cfg.elevation))
        return torch.tensor(
            [
                [1.0 / (2 * sE),  0.0,            -1.0 / (2 * sE),  0.0           ],
                [0.0,             1.0 / (2 * sE),  0.0,            -1.0 / (2 * sE)],
                [1.0 / (4 * cE),  1.0 / (4 * cE),  1.0 / (4 * cE),  1.0 / (4 * cE)],
            ],
            device=self._device,
        )

    """
    Main update
    """

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        # RayCaster populates pos_w / quat_w / ray_hits_w using `env_ids`.
        super()._update_buffers_impl(env_ids)

        # Unify indexing. Use slice(None) only when env_ids covers all envs.
        if isinstance(env_ids, slice):
            idx = env_ids
        elif isinstance(env_ids, (list, tuple, torch.Tensor)) and len(env_ids) == self._num_envs:
            idx = slice(None)
        else:
            idx = env_ids

        sensor_pos_w = self._data.pos_w[idx]
        sensor_quat_w = self._data.quat_w[idx]
        hits_w = self._data.ray_hits_w[idx]
        N = sensor_pos_w.shape[0]

        # ---- 1) Per-beam range + hit mask ----
        vec_to_hit = hits_w - sensor_pos_w.unsqueeze(1)
        depths = torch.linalg.vector_norm(vec_to_hit, dim=-1)

        # Assuming warp leaves missed hits at (0,0,0)
        missed_rays = torch.norm(hits_w, dim=-1) < 1e-5 

        finite = torch.isfinite(depths) & ~missed_rays
        hits = finite & (depths >= self.cfg.min_range) & (depths <= self.cfg.max_range)

        # ---- 2) Depth noise ----
        if self._noise_std_dep > 0.0:
            depths = depths + torch.randn_like(depths) * self._noise_std_dep

        # ---- 3) Linear velocity in body frame ----
        # Get linear and angular velocities
        body_vels = self._rb_view.get_velocities()[idx]
        world_lin_vel = body_vels[:, :3]
        world_ang_vel = body_vels[:, 3:6]

        # Calculate offset in world frame
        # You need the body's world position to calculate the offset vector
        body_transforms = self._rb_view.get_transforms()[idx]
        body_pos_w = body_transforms[:, :3]
        sensor_offset_w = sensor_pos_w - body_pos_w

        # v_sensor = v_body + (omega x r)
        sensor_world_vel = world_lin_vel + torch.linalg.cross(world_ang_vel, sensor_offset_w, dim=-1)

        # Rotate to sensor body frame
        vel_b = math_utils.quat_apply(math_utils.quat_conjugate(sensor_quat_w), sensor_world_vel)

        if self._noise_std_vel > 0.0:
            beam_noise = torch.randn(N, 4, device=self._device) * self._noise_std_vel
            vel_noise = (self._janus_transform @ beam_noise.unsqueeze(-1)).squeeze(-1)
            vel_b = vel_b + vel_noise

        # ---- 4) Dropout + NaN-out ----
        n_miss = (~hits).sum(dim=-1)
        dropout = n_miss >= self.cfg.num_beams_out_range_threshold
        vel_b = torch.where(dropout.unsqueeze(-1), torch.zeros_like(vel_b), vel_b)
        depths = torch.where(hits, depths, torch.full_like(depths, float("nan")))

        # ---- 5) Per-env sensor period ----
        safe = torch.where(hits, depths, torch.full_like(depths, self.cfg.max_range))
        min_range_per_env = safe.min(dim=-1).values

        if self._adaptive_freq:
            dt_per_env = self._compute_adaptive_dt(min_range_per_env)
        else:
            dt_per_env = torch.full_like(min_range_per_env, self._fixed_dt)

        # ---- 6) Frequency-dependent latched outputs ----
        self._elapsed_vel[idx] = self._elapsed_vel[idx] + self._dt
        self._elapsed_depth[idx] = self._elapsed_depth[idx] + self._dt

        vel_due = self._elapsed_vel[idx] >= dt_per_env
        dep_due = self._elapsed_depth[idx] >= dt_per_env

        lin_vel_fd = self._data.lin_vel_fd_b[idx].clone()
        lin_vel_fd[vel_due] = vel_b[vel_due]
        depth_fd = self._data.depth_fd[idx].clone()
        depth_fd[dep_due] = depths[dep_due]

        if isinstance(idx, slice):
            elapsed_vel_local = self._elapsed_vel[idx]
            elapsed_vel_local = torch.where(vel_due, torch.zeros_like(elapsed_vel_local), elapsed_vel_local)
            self._elapsed_vel[idx] = elapsed_vel_local

            elapsed_dep_local = self._elapsed_depth[idx]
            elapsed_dep_local = torch.where(dep_due, torch.zeros_like(elapsed_dep_local), elapsed_dep_local)
            self._elapsed_depth[idx] = elapsed_dep_local
        else:
            idx_t = idx if isinstance(idx, torch.Tensor) else torch.as_tensor(
                idx, device=self._device, dtype=torch.long
            )
            self._elapsed_vel[idx_t[vel_due]] = 0.0
            self._elapsed_depth[idx_t[dep_due]] = 0.0

        # ---- 7) Commit ----
        self._data.lin_vel_b[idx] = vel_b
        self._data.lin_vel_fd_b[idx] = lin_vel_fd
        self._data.depth[idx] = depths
        self._data.depth_fd[idx] = depth_fd
        self._data.beam_hit[idx] = hits
        self._data.dt[idx] = dt_per_env.unsqueeze(-1)

    def _compute_adaptive_dt(self, min_range_per_env: torch.Tensor) -> torch.Tensor:
        """Vectorized adaptive-frequency rule matching the original Isaac Sim DVL."""
        f_min, f_max = self.cfg.freq_bound
        r_min, r_max = self.cfg.freq_dependent_range_bound
        c = self.cfg.sound_speed

        freq = torch.full_like(min_range_per_env, float(f_min))

        short = min_range_per_env <= r_min
        freq = torch.where(short, torch.full_like(freq, float(f_max)), freq)

        ramp = (min_range_per_env > r_min) & (min_range_per_env < r_max)
        h = min_range_per_env
        f_ramp = f_max - (f_max - c / (2.0 * torch.clamp(h, min=1e-6))) / (r_max - r_min) * (h - r_min)
        freq = torch.where(ramp, f_ramp, freq)

        return 1.0 / torch.clamp(freq, min=1e-6)
