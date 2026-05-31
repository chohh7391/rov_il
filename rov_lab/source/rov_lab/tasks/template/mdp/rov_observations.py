import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from rov_lab.assets.sensors import Barometer, DVL


def brometer_pressure(
    env: ManagerBasedRLEnv | DirectRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("barometer"),
) -> torch.Tensor:
    barometer: Barometer = env.scene[sensor_cfg.name]
    return barometer.data.pressure

def dvl_linear_velocity(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("dvl")
) -> torch.Tensor:
    """Returns DVL linear velocity in body frame. Shape: (num_envs, 3)"""
    dvl: DVL = env.scene[sensor_cfg.name]
    return dvl.data.lin_vel_b

def dvl_depth(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("dvl")
) -> torch.Tensor:
    """Returns per-beam depth in meters. NaN on miss. Shape: (num_envs, 4)"""
    dvl: DVL = env.scene[sensor_cfg.name]
    return dvl.data.depth

def dvl_beam_hit(env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("dvl")
) -> torch.Tensor:
    """Returns per-beam hit flag. Shape: (num_envs, 4)"""
    dvl: DVL = env.scene[sensor_cfg.name]
    return dvl.data.beam_hit.float()
