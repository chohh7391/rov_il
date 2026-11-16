"""Observation terms for ROV tasks: arm state and underwater sensors (Isaac Lab 3.0)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from rov_lab.assets.sensors import DVLCfg, UWCameraCfg, uw_render
from rov_lab.assets.sensors.barometer_model import barometer_pressure_model
from rov_lab.assets.sensors.dvl import (
    bottom_lock,
    dvl_body_velocity,
    dvl_ranges_from_hits,
    dvl_ray_origins_w,
    janus_noise_transform,
)
from rov_lab.utils.tensor import as_torch

if TYPE_CHECKING:  # runtime classes pull in pxr; keep this module importable before Kit starts
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sensors import Camera, FrameTransformer, RayCaster


# ---------------------------------------------------------------------------------------------------- arm
def ee_frame_state(
    env: ManagerBasedEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """End-effector pose in the robot root frame: position (3) + quaternion (4, xyzw in Isaac Lab 3.0).

    Ported from leisaac's ``ee_frame_state``.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    pos_b, quat_b = math_utils.subtract_frame_transforms(
        as_torch(robot.data.root_pos_w),
        as_torch(robot.data.root_quat_w),
        as_torch(ee_frame.data.target_pos_w)[:, 0, :],
        as_torch(ee_frame.data.target_quat_w)[:, 0, :],
    )
    return torch.cat([pos_b, quat_b], dim=1)


def joint_pos_target(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint position targets of the configured joints. Ported from leisaac's ``joint_pos_target``."""
    asset: Articulation = env.scene[asset_cfg.name]
    return as_torch(asset.data.joint_pos_target)[:, asset_cfg.joint_ids]


# ---------------------------------------------------------------------------------------------------- camera
def uw_image(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist")) -> torch.Tensor:
    """Underwater camera image, shape ``(N, H, W, 3)`` uint8, using the camera's OceanSim water parameters."""
    camera: Camera = env.scene[sensor_cfg.name]
    cfg: UWCameraCfg = camera.cfg
    output = camera.data.output
    return uw_render(as_torch(output["rgb"]), as_torch(output["distance_to_camera"]), cfg)


# ---------------------------------------------------------------------------------------------------- barometer
def barometer_pressure(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rov_base"),
    water_surface_z: float = 0.0,
    water_density: float = 1000.0,
    gravity: float = 9.81,
    atmosphere_pressure: float = 101325.0,
    noise_cov: float = 0.0,
) -> torch.Tensor:
    """Absolute pressure at the hull (Pa), shape ``(N, 1)``. OceanSim-compatible (see :func:`barometer_pressure_model`)."""
    robot: Articulation = env.scene[asset_cfg.name]
    z = as_torch(robot.data.body_link_pos_w)[:, int(asset_cfg.body_ids[0]), 2].unsqueeze(-1)
    return barometer_pressure_model(z, water_surface_z, water_density, gravity, atmosphere_pressure, noise_cov)


def barometer_depth(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rov_base"),
    water_surface_z: float = 0.0,
    water_density: float = 1000.0,
    gravity: float = 9.81,
    atmosphere_pressure: float = 101325.0,
    noise_cov: float = 0.0,
) -> torch.Tensor:
    """Depth (m) inverted from the OceanSim-compatible pressure: ``(p - p_atm) / (rho * g)``, shape ``(N, 1)``.

    Same measurement and noise as :func:`barometer_pressure`, scaled for use as a policy input (the 2.x
    ``Barometer`` returned raw Pa, ~1.4e5).
    """
    p = barometer_pressure(env, asset_cfg, water_surface_z, water_density, gravity, atmosphere_pressure, noise_cov)
    return (p - atmosphere_pressure) / (water_density * gravity)


# ---------------------------------------------------------------------------------------------------- DVL
def _dvl_ranges(dvl: RayCaster) -> tuple[torch.Tensor, torch.Tensor]:
    cfg: DVLCfg = dvl.cfg
    origin_w = dvl_ray_origins_w(as_torch(dvl.data.pos_w), as_torch(dvl.data.quat_w), cfg.offset.pos)
    return dvl_ranges_from_hits(as_torch(dvl.data.ray_hits_w), origin_w, cfg.min_range, cfg.max_range)


def dvl_depth(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("dvl")) -> torch.Tensor:
    """Per-beam range to the seabed (m), shape ``(N, 4)``, with per-beam noise of variance ``depth_cov``.

    Beams with no return report ``cfg.miss_value`` (OceanSim: NaN) or ``max_range`` when it is ``None``.
    """
    dvl: RayCaster = env.scene[sensor_cfg.name]
    cfg: DVLCfg = dvl.cfg
    rng, hit = _dvl_ranges(dvl)
    if cfg.depth_cov > 0.0:
        rng = rng + cfg.depth_cov**0.5 * torch.randn_like(rng)
    miss = cfg.max_range if cfg.miss_value is None else cfg.miss_value
    return torch.where(hit, rng, torch.full_like(rng, miss))


def dvl_beam_hit(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("dvl")) -> torch.Tensor:
    """Per-beam valid flag (1.0 = bottom return), shape ``(N, 4)``."""
    dvl: RayCaster = env.scene[sensor_cfg.name]
    return _dvl_ranges(dvl)[1].float()


def dvl_linear_velocity(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("dvl"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Velocity over ground in the vehicle body frame (m/s), shape ``(N, 3)``. OceanSim measurement model:

    * reference point: body center of mass (``velocity_point="com"``, as OceanSim) or the DVL mounting point
      (``"sensor"``, adds the lever-arm term),
    * noise: per-beam samples of variance ``vel_cov`` mapped by the Janus transform,
    * dropout: zero when ``num_beams_out_range_threshold`` or more beams have no return.
    """
    dvl: RayCaster = env.scene[sensor_cfg.name]
    cfg: DVLCfg = dvl.cfg
    robot: Articulation = env.scene[robot_cfg.name]
    body_id = robot.find_bodies(cfg.body_name)[0][0]
    quat_w = as_torch(robot.data.body_link_quat_w)[:, body_id]
    if cfg.velocity_point == "com":
        vel_b = dvl_body_velocity(as_torch(robot.data.body_com_lin_vel_w)[:, body_id], quat_w)
    else:
        offset_b = torch.tensor(cfg.offset.pos, device=env.device, dtype=torch.float32)
        vel_b = dvl_body_velocity(
            as_torch(robot.data.body_link_lin_vel_w)[:, body_id],
            quat_w,
            as_torch(robot.data.body_link_ang_vel_w)[:, body_id],
            offset_b,
        )
    if cfg.vel_cov > 0.0:
        beam_noise = cfg.vel_cov**0.5 * torch.randn(vel_b.shape[0], 4, device=vel_b.device)
        vel_b = vel_b + beam_noise @ janus_noise_transform(cfg.elevation, vel_b.device).T
    lock = bottom_lock(_dvl_ranges(dvl)[1], cfg.num_beams_out_range_threshold)
    return torch.where(lock, vel_b, torch.zeros_like(vel_b))
