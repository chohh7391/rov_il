# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaacsim.core.simulation_manager import SimulationManager
from pxr import UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers

from isaaclab.sensors.sensor_base import SensorBase
from .barometer_data import BarometerData

if TYPE_CHECKING:
    from .barometer_cfg import BarometerCfg


class Barometer(SensorBase):

    cfg: BarometerCfg
    """The configuration parameters."""

    def __init__(self, cfg: BarometerCfg):
        """Initializes the Imu sensor.

        Args:
            cfg: The configuration parameters.
        """
        # initialize base class
        super().__init__(cfg)
        # Create empty variables for storing output data
        self._data = BarometerData()

        # Internal: expression used to build the rigid body view (may be different from cfg.prim_path)
        self._rigid_parent_expr: str | None = None

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"barometer sensor @ '{self.cfg.prim_path}': \n"
            f"\tview type         : {self._view.__class__}\n"
            f"\tupdate period (s) : {self.cfg.update_period}\n"
            f"\tnumber of sensors : {self._view.count}\n"
        )

    """
    Properties
    """

    @property
    def data(self) -> BarometerData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    @property
    def num_instances(self) -> int:
        return self._view.count

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timestamps
        super().reset(env_ids)
        # resolve None
        if env_ids is None:
            env_ids = slice(None)
        # reset accumulative data buffers
        self._data.pos_w[env_ids] = 0.0
        self._data.quat_w[env_ids] = 0.0
        self._data.quat_w[env_ids, 0] = 1.0
        self._data.pressure[env_ids] = 0.0

    def update(self, dt: float, force_recompute: bool = False):
        # save timestamp
        self._dt = dt
        # execute updating
        super().update(dt, force_recompute)

    """
    Implementation.
    """

    def _initialize_impl(self):
        """Initializes the sensor handles and internal buffers.

        - If the target prim path is a rigid body, build the view directly on it.
        - Otherwise find the closest rigid-body ancestor, cache the fixed transform from that ancestor
          to the target prim, and build the view on the ancestor expression.
        """
        # Initialize parent class
        super()._initialize_impl()
        # obtain global simulation view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        # check if the prim at path is a rigid prim
        prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if prim is None:
            raise RuntimeError(f"Failed to find a prim at path expression: {self.cfg.prim_path}")

        # Find the first matching ancestor prim that implements rigid body API.
        # The sensor prim may be a child Xform below the ROV body, so walking
        # upward is required here; child traversal only works when cfg.prim_path
        # already targets the rigid body itself.
        ancestor_prim = sim_utils.get_first_matching_ancestor_prim(
            prim.GetPath(), predicate=lambda _prim: _prim.HasAPI(UsdPhysics.RigidBodyAPI)
        )
        if ancestor_prim is None:
            raise RuntimeError(f"Failed to find a rigid body ancestor prim at path expression: {self.cfg.prim_path}")
        # Convert ancestor prim path to expression
        if ancestor_prim == prim:
            self._rigid_parent_expr = self.cfg.prim_path
            fixed_pos_b, fixed_quat_b = None, None
        else:
            # Convert ancestor prim path to expression
            relative_path = prim.GetPath().MakeRelativePath(ancestor_prim.GetPath()).pathString
            self._rigid_parent_expr = self.cfg.prim_path.replace(relative_path, "")
            # Resolve the relative pose between the target prim and the ancestor prim
            fixed_pos_b, fixed_quat_b = sim_utils.resolve_prim_pose(prim, ancestor_prim)

        # Create the rigid body view on the ancestor
        self._view = self._physics_sim_view.create_rigid_body_view(self._rigid_parent_expr.replace(".*", "*"))

        # Create internal buffers
        self._initialize_buffers_impl()

        # Compose the configured offset with the fixed ancestor->target transform (done once)
        # new_offset = fixed * cfg.offset
        # where composition is: p = p_fixed + R_fixed * p_cfg, q = q_fixed * q_cfg
        if fixed_pos_b is not None and fixed_quat_b is not None:
            # Broadcast fixed transform across instances
            fixed_p = torch.tensor(fixed_pos_b, device=self._device).repeat(self._view.count, 1)
            fixed_q = torch.tensor(fixed_quat_b, device=self._device).repeat(self._view.count, 1)

            cfg_p = self._offset_pos_b.clone()
            cfg_q = self._offset_quat_b.clone()

            composed_p = fixed_p + math_utils.quat_apply(fixed_q, cfg_p)
            composed_q = math_utils.quat_mul(fixed_q, cfg_q)

            self._offset_pos_b = composed_p
            self._offset_quat_b = composed_q

        # noise std
        self._noise_std = self.cfg.noise_cov ** 0.5

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""

        # default to all sensors
        if len(env_ids) == self._num_envs:
            env_ids = slice(None)
        # world pose of the rigid source (ancestor) from the PhysX view
        pos_w, quat_w = self._view.get_transforms()[env_ids].split([3, 4], dim=-1)
        quat_w = quat_w.roll(1, dims=-1)

        # sensor pose in world: apply composed offset
        self._data.pos_w[env_ids] = pos_w + math_utils.quat_apply(quat_w, self._offset_pos_b[env_ids])
        self._data.quat_w[env_ids] = math_utils.quat_mul(quat_w, self._offset_quat_b[env_ids])

        depth = torch.clamp(self.cfg.water_surface_z - self._data.pos_w[env_ids, 2], min=0.0).unsqueeze(-1)

        # Compute hydrostatic pressure.
        pressure = self.cfg.atmosphere_pressure + self.cfg.water_density * self.cfg.gravity * depth

        if self._noise_std > 0.0:
            noise = torch.randn_like(pressure) * self._noise_std
            pressure = pressure + noise

        self._data.pressure[env_ids] = pressure

    def _initialize_buffers_impl(self):
        """Create buffers for storing data."""
        # data buffers
        self._data.pos_w = torch.zeros(self._view.count, 3, device=self._device)
        self._data.quat_w = torch.zeros(self._view.count, 4, device=self._device)
        self._data.quat_w[:, 0] = 1.0

        # store sensor offset (applied relative to rigid source).
        # This may be composed later with a fixed ancestor->target transform.
        self._offset_pos_b = torch.tensor(list(self.cfg.offset.pos), device=self._device).repeat(self._view.count, 1)
        self._offset_quat_b = torch.tensor(list(self.cfg.offset.rot), device=self._device).repeat(self._view.count, 1)

        self._data.pressure = torch.zeros(self._view.count, 1, device=self._device)
