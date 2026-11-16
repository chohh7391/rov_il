"""Isaac Lab 3.0 port of the rov_lab underwater pick-and-place scene (KRISO 2026 experiment).

Re-uses rov_lab assets (BlueROV + SO-101 arm USD, MHL seabed scene) and ports, in simplified form:
  * ROVVelocityAction  -> ROVServo (4-DOF body-velocity PI servo, rov_lab gains/limits)
  * apply_hydro_wrench -> Hydro (BlueROV linear+quadratic damping, buoyancy restoring moment)
  * UWCamera           -> uw_render() (OceanSim attenuation + backscatter on RGB + depth)
  * DVL / barometer    -> body-frame velocity / depth-from-surface with noise
Quaternions follow Isaac Lab 3.0 (x, y, z, w).
Import only after the simulation app is launched.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as mu
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

ASSETS = "/home/home/rov_il/rov_lab/assets"
CUBES = ("red_cube", "green_cube", "blue_cube")
INSTRUCTIONS = ("pick up the red cube", "pick up the green cube", "pick up the blue cube")
CUBE_COLORS = ((0.9, 0.05, 0.03), (0.05, 0.75, 0.10), (0.05, 0.18, 0.9))
WATER_SURFACE = 3.43389

# experiment geometry (world frame)
CUBE_X = 1.0
CUBE_Z = -1.5
CUBE_YS = (-0.05, 0.05, 0.15)
ROV_START = (-2.0, 0.0, -0.8)  # overridden by experiment


def wxyz_to_xyzw(q):
    return (q[1], q[2], q[3], q[0])


def _cube(name, pos, color):
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.03, 0.03, 0.03),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, linear_damping=3.0, angular_damping=3.0,
                max_linear_velocity=5.0, max_angular_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.5, dynamic_friction=1.5,
                                                            friction_combine_mode="max"),
        ),
    )


ROBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSETS}/robots/blue_rov_single_arm.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True, linear_damping=10.0, angular_damping=10.0,
            max_linear_velocity=5.0, max_angular_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4,
            solver_velocity_iteration_count=4, fix_root_link=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=ROV_START, rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 0.0,
                   "wrist_flex": 0.0, "wrist_roll": 0.0, "gripper": 0.0},
    ),
    actuators={
        "gripper": ImplicitActuatorCfg(joint_names_expr=["gripper"], effort_limit_sim=10,
                                       velocity_limit_sim=10, stiffness=17.8, damping=2.0),
        "arm": ImplicitActuatorCfg(joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex",
                                                     "wrist_flex", "wrist_roll"],
                                   effort_limit_sim=10, velocity_limit_sim=10, stiffness=60.0, damping=4.0),
    },
    soft_joint_pos_limit_factor=1.0,
)


def make_scene_cfg(cam_hw_front=(180, 320), cam_hw_wrist=(240, 320), rov_start=ROV_START, probe=False, debug_cam=False,
                   fixed_base: bool = False, max_ang_vel_deg: float = 5.0):
    """``fixed_base`` locks the ROV hull (PhysX fixed root link): manipulation with the vehicle station-kept.
    ``max_ang_vel_deg`` is PhysX's per-body angular velocity cap in deg/s. The legacy default 5 deg/s (inherited
    from rov_lab) also throttles the arm links; fixed-base runs use a realistic value."""
    robot_cfg = ROBOT_CFG.replace(
        init_state=ROBOT_CFG.init_state.replace(pos=rov_start),
        spawn=ROBOT_CFG.spawn.replace(
            rigid_props=ROBOT_CFG.spawn.rigid_props.replace(max_angular_velocity=max_ang_vel_deg),
            articulation_props=ROBOT_CFG.spawn.articulation_props.replace(fix_root_link=fixed_base),
        ),
    )

    @configclass
    class UWPickSceneCfg(InteractiveSceneCfg):
        seabed = AssetBaseCfg(prim_path="/World/Scene",
                              spawn=sim_utils.UsdFileCfg(usd_path=f"{ASSETS}/scenes/collected_MHL/mhl_scaled.usd"))
        light = AssetBaseCfg(prim_path="/World/Light",
                             spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.85, 0.9, 1.0)))
        robot = robot_cfg
        red_cube = _cube(CUBES[0], (CUBE_X, CUBE_YS[0], CUBE_Z), CUBE_COLORS[0])
        green_cube = _cube(CUBES[1], (CUBE_X, CUBE_YS[1], CUBE_Z), CUBE_COLORS[1])
        blue_cube = _cube(CUBES[2], (CUBE_X, CUBE_YS[2], CUBE_Z), CUBE_COLORS[2])
        front = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/rov_base/front_cam",
            height=cam_hw_front[0], width=cam_hw_front[1],
            data_types=["rgb", "distance_to_camera"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=2.1, horizontal_aperture=4.2, clipping_range=(0.05, 100.0)),
            offset=CameraCfg.OffsetCfg(pos=(0.3, 0.0, 0.1), rot=wxyz_to_xyzw((0.5, -0.5, 0.5, -0.5)), convention="ros"),
        )
        wrist = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Arm/gripper/wrist_cam",
            height=cam_hw_wrist[0], width=cam_hw_wrist[1],
            data_types=["rgb", "distance_to_camera"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=36.5, focus_distance=400.0, horizontal_aperture=36.83,
                                             clipping_range=(0.01, 50.0)),
            offset=CameraCfg.OffsetCfg(pos=(-0.001, 0.1, -0.04),
                                       rot=wxyz_to_xyzw((-0.404379, -0.912179, -0.0451242, 0.0486914)),
                                       convention="ros"),
        )

    if debug_cam:
        # attached to the ROV body (offsets on robot-attached cameras are applied); body ~ world-aligned (yaw ~ 0)
        eye = torch.tensor([[0.55, -0.85, 0.25]])
        tgt = torch.tensor([[0.30, 0.0, -0.05]])
        R = mu.create_rotation_matrix_from_view(eye, tgt, "Z")
        q = mu.quat_from_matrix(R)[0].tolist()
        UWPickSceneCfg.debug = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/rov_base/debug_cam", height=540, width=720,
            data_types=["rgb", "distance_to_camera"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=14.0, horizontal_aperture=20.955, clipping_range=(0.01, 50.0)),
            offset=CameraCfg.OffsetCfg(pos=tuple(eye[0].tolist()), rot=tuple(q), convention="opengl"),
        )
    if probe:
        UWPickSceneCfg.probe = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/probe",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(CUBE_X, 0.35, -1.0)),
            spawn=sim_utils.SphereCfg(radius=0.02, rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                                      mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                                      collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True)),
        )
    return UWPickSceneCfg


# ----------------------------------------------------------------------------- underwater image formation
BACKSCATTER = torch.tensor([0.0, 0.31, 0.24])
ATTEN = torch.tensor([0.05, 0.05, 0.05])
BACK_COEFF = torch.tensor([0.05, 0.05, 0.2])


def as_torch(x) -> torch.Tensor:
    """Camera outputs may be ProxyArray / warp arrays in Isaac Lab 3.0."""
    if isinstance(x, torch.Tensor):
        return x
    if hasattr(x, "torch"):
        return x.torch
    import warp as wp

    return wp.to_torch(x)


def cam_images(cam, uw: bool = True) -> torch.Tensor:
    out = cam.data.output
    rgb = as_torch(out["rgb"])
    if not uw:
        return rgb[..., :3]
    return uw_render(rgb, as_torch(out["distance_to_camera"]))


def uw_render(rgb: torch.Tensor, depth: torch.Tensor, far: float = 50.0) -> torch.Tensor:
    """OceanSim UW_render: I = J*exp(-d*b_att) + 255*B_inf*(1-exp(-d*b_back)). rgb (N,H,W,3) uint8, depth (N,H,W,1)."""
    d = torch.nan_to_num(depth.float(), nan=far, posinf=far, neginf=far).clamp(0, far)
    dev = rgb.device
    j = rgb[..., :3].float()
    out = j * torch.exp(-d * ATTEN.to(dev)) + 255.0 * BACKSCATTER.to(dev) * (1.0 - torch.exp(-d * BACK_COEFF.to(dev)))
    return out.clamp(0, 255).to(torch.uint8)


# ----------------------------------------------------------------------------- ROV servo + hydrodynamics
class ROVServo:
    """4-DOF body-velocity PI servo (rov_lab ROVVelocityAction): cmd = [vx, vy, vz, yaw_rate] in [-1, 1]."""

    LIN_SCALE = (1.5, 1.0, 0.7)
    YAW_SCALE = 0.6
    KP_LIN = (25.0, 25.0, 30.0)
    KI_LIN = (3.0, 3.0, 5.0)
    I_LIM = 0.3
    I_LEAK = 0.98
    KP_ANG = (12.0, 12.0, 18.0)
    F_UP = (588.0, 329.0, 304.0)
    F_LO = (294.0, 329.0, 304.0)
    T_LIM = 100.0

    def __init__(self, robot, body_id: int, dt: float, num_envs: int, device):
        self.r, self.b, self.dt = robot, body_id, dt
        t = lambda v: torch.tensor(v, device=device, dtype=torch.float32)  # noqa: E731
        self.lin_scale, self.kp_lin, self.ki_lin = t(self.LIN_SCALE), t(self.KP_LIN), t(self.KI_LIN)
        self.kp_ang, self.f_up, self.f_lo = t(self.KP_ANG), t(self.F_UP), t(self.F_LO)
        self.integ = torch.zeros(num_envs, 3, device=device)

    def reset(self):
        self.integ.zero_()

    def body_vel(self):
        q = self.r.data.body_quat_w.torch[:, self.b]
        qi = mu.quat_conjugate(q)
        lin = mu.quat_apply(qi, self.r.data.body_lin_vel_w.torch[:, self.b])
        ang = mu.quat_apply(qi, self.r.data.body_ang_vel_w.torch[:, self.b])
        return q, lin, ang

    def wrench_b(self, cmd: torch.Tensor):
        q, lin, ang = self.body_vel()
        tgt_lin = cmd[:, 0:3].clamp(-1, 1) * self.lin_scale
        err = tgt_lin - lin
        active = (cmd[:, 0:3].abs() > 1e-4).any(dim=1, keepdim=True).float()
        self.integ = ((self.I_LEAK * self.integ + err * self.dt).clamp(-self.I_LIM, self.I_LIM)) * active
        f = self.kp_lin * err + self.ki_lin * self.integ
        tgt_ang = torch.zeros_like(ang)
        tgt_ang[:, 2] = cmd[:, 3].clamp(-1, 1) * self.YAW_SCALE
        tq = self.kp_ang * (tgt_ang - ang)
        f = torch.maximum(torch.minimum(f, self.f_up), -self.f_lo)
        tq = tq.clamp(-self.T_LIM, self.T_LIM)
        return f, tq


class Hydro:
    """BlueROV Fossen damping + buoyancy restoring moment (rov_lab HydroParamsCfg defaults), body frame."""

    LIN_D = (4.03, 6.22, 5.18, 0.07, 0.07, 0.07)
    QUAD_D = (18.18, 21.66, 36.99, 1.55, 1.55, 1.55)
    RHO, G, VOL, CO_BM = 997.0, 9.8, 0.0113459, 0.01

    def __init__(self, device):
        self.ld = torch.tensor(self.LIN_D, device=device)
        self.qd = torch.tensor(self.QUAD_D, device=device)

    def wrench_b(self, q, lin_b, ang_b, current_b=None):
        v = torch.cat([lin_b - (current_b if current_b is not None else 0.0), ang_b], dim=-1)
        w = -(self.ld + self.qd * v.abs()) * v
        roll, pitch, _ = mu.euler_xyz_from_quat(q)
        bm = self.RHO * self.G * self.VOL * self.CO_BM
        w[:, 3] += -bm * torch.sin(roll) * torch.cos(pitch)
        w[:, 4] += -bm * torch.sin(pitch)
        return w[:, :3], w[:, 3:]
