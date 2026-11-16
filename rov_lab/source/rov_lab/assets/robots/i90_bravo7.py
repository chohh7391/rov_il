"""Configuration for the "I90-class" ROV with the Bravo 7-class arm: one floating articulation (Isaac Lab 3.0).

The vehicle is our own simplified box/cylinder design sized to the public DWTEK Investigator 90 catalogue
(observation class: 1100 x 700 x 490 mm, 120 kg in air, 27 kg payload, 3 kn, 500 m; 4 horizontal + 2 vertical
DWT6537 thrusters; forward/reverse 60/30 kgf, lateral 33.5 kgf, vertical 31 kgf). The arm is the parametric
Bravo 7-class (``bravo7_class.py``). Geometry, masses and the values below are produced by
``rov_lab/scripts/tools/make_i90_bravo7_urdf.py`` (design table in
``rov_lab/assets_src/robots/i90_bravo7/i90_bravo7_params.json``); the USD by
``rov_lab/scripts/tools/convert_i90_bravo7.py``; checks in ``rov_lab/scripts/tools/test_i90_bravo7.py``.

Difference to the catalogue: the I90's optional manipulator is "up to 5 axis". We mount the 6-DOF Bravo 7-class
(9.5 kg in air, 4.5 kg in water), which fits the 27 kg payload. The catalogue length (1100 mm) excludes the arm;
stowed, the arm reaches 0.22 m ahead of the bow.

Frames: ``rov_base`` is FLU (+x bow, +y port, +z up) with its origin at the centre of the 1100 x 700 x 490 mm
envelope (hull bottom z = -0.245, bow x = +0.55). Quaternions are Isaac Lab 3.0 (x, y, z, w).

Physics convention (rov_lab): gravity is disabled on every body (neutral buoyancy assumed). Hydrodynamics are
applied to ``rov_base`` as external wrenches by ``mdp.apply_hydro_wrench`` with ``HydroParamsCfg(**I90_HYDRO_PARAMS)``.
PhysX rigid-body damping is explicitly zero (the old BlueROV asset double-counted damping). With gravity off the
arm's wet weight (4.5 kg) and the hull's +5 kg reserve are not simulated; the restoring moment comes only from
the ``co_bm`` term.

USD prim paths after conversion (spawned at ``{ENV_REGEX_NS}/Robot``; the importer nests links under
``Geometry``): see ``I90_BRAVO7_PRIM_PATHS``.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from rov_lab.utils.paths import ROBOT_DIR

from .bravo7_class import (
    BASE_LINK,
    BASE_YAW_JOINT,
    BRAVO7_CLASS_CFG,
    BRAVO7_CLASS_JOINT_LIMITS,
    BRAVO7_CLASS_JOINT_NAMES,
    BRAVO7_CLASS_LINK_NAMES,
    EE_ROLL_JOINT,
    ELBOW_PITCH_JOINT,
    FINGER_LEFT_JOINT,
    FINGER_RIGHT_JOINT,
    FOREARM_ROLL_JOINT,
    GRASP_POINT_OFFSET_M,
    JAW_BASE_LINK,
    JAW_TIP_OFFSET_M,
    LINK_DISPLACED_VOLUME_M3,
    SHOULDER_PITCH_JOINT,
    WRIST_PITCH_JOINT,
)

##
# Asset path (honours ROV_LAB_ASSETS_ROOT through rov_lab.utils.paths)
##

I90_BRAVO7_USD_PATH: Path = ROBOT_DIR / "i90_bravo7" / "i90_bravo7.usd"

##
# Names
##

ROV_BASE_LINK: str = "rov_base"
"""Vehicle hull body; rov_lab's servo, hydrodynamics, DVL and barometer terms use ``body_name="rov_base"``."""
ARM_BASE_LINK: str = BASE_LINK  # "base_link", Bravo mounting flange
EE_LINK: str = JAW_BASE_LINK  # "jaw_base_link", TCP at GRASP_POINT_OFFSET_M, jaw tip at JAW_TIP_OFFSET_M
FRONT_CAMERA_LINK: str = "front_camera_link"
DVL_LINK: str = "dvl_link"
PRESSURE_SENSOR_LINK: str = "pressure_sensor_link"

ARM_MOUNT_JOINT: str = "arm_mount_joint"
FRONT_CAMERA_JOINT: str = "front_camera_joint"
DVL_JOINT: str = "dvl_joint"
PRESSURE_SENSOR_JOINT: str = "pressure_sensor_joint"

I90_VEHICLE_LINK_NAMES: tuple[str, ...] = (ROV_BASE_LINK, FRONT_CAMERA_LINK, DVL_LINK, PRESSURE_SENSOR_LINK)
I90_BRAVO7_LINK_NAMES: tuple[str, ...] = I90_VEHICLE_LINK_NAMES + BRAVO7_CLASS_LINK_NAMES
I90_BRAVO7_JOINT_NAMES: tuple[str, ...] = BRAVO7_CLASS_JOINT_NAMES  # actuated joints (fixed joints not listed)

_GEOM: str = "{ENV_REGEX_NS}/Robot/Geometry"
_ARM: str = f"{_GEOM}/rov_base/base_link"
_WRIST_ROLL: str = f"{_ARM}/shoulder_link/upper_arm_link/elbow_link/forearm_link/wrist_link/wrist_roll_link"
I90_BRAVO7_PRIM_PATHS: dict[str, str] = {
    ROV_BASE_LINK: f"{_GEOM}/rov_base",
    FRONT_CAMERA_LINK: f"{_GEOM}/rov_base/front_camera_link",
    DVL_LINK: f"{_GEOM}/rov_base/dvl_link",
    PRESSURE_SENSOR_LINK: f"{_GEOM}/rov_base/pressure_sensor_link",
    ARM_BASE_LINK: _ARM,
    "wrist_roll_link": _WRIST_ROLL,
    JAW_BASE_LINK: f"{_WRIST_ROLL}/jaw_base_link",
}
"""Body prim paths when the asset is spawned at ``{ENV_REGEX_NS}/Robot`` (verified by test_i90_bravo7.py)."""

##
# Geometry and frames (rov_base frame, from i90_bravo7_params.json)
##

I90_DIMENSIONS_M: tuple[float, float, float] = (1.100, 0.700, 0.490)
I90_BOW_X_M: float = 0.550
I90_HULL_BOTTOM_Z_M: float = -0.245

I90_ARM_MOUNT_POS: tuple[float, float, float] = (0.310, 0.0, -0.170)
I90_ARM_MOUNT_ROT: tuple[float, float, float, float] = (0.0, 0.7071068, 0.0, 0.7071068)
"""``base_link`` pose in ``rov_base`` (fixed joint): flange face at the lower front bay, base-rotate axis (+z of
``base_link``) pointing forward, arm zero direction (+x of ``base_link``) pointing straight down (rpy 0, 90 deg, 0).
Consequences: base yaw 0 -> the arm works in the vertical centre plane and +shoulder/+elbow/+wrist swing the arm
from down toward forward; base yaw +90 deg -> the arm folds in the horizontal plane (stow)."""
I90_SHOULDER_AXIS_POS: tuple[float, float, float] = (0.500, 0.0, -0.170)
"""Shoulder pitch axis (5 cm behind the bow, 75 mm above the hull bottom) at base yaw 0."""

I90_FRONT_CAMERA_POS: tuple[float, float, float] = (0.545, 0.0, 0.100)
I90_FRONT_CAMERA_ROT: tuple[float, float, float, float] = (0.0, 0.173648, 0.0, 0.984808)
"""``front_camera_link``: optical centre in front of the foam, +x = optical axis tilted 20 deg down, +z = image up.
Attach a camera under the link with ``OffsetCfg(pos=(0, 0, 0), rot=(0, 0, 0, 1), convention="world")``, or under
``rov_base`` with ``OffsetCfg(pos=I90_FRONT_CAMERA_POS, rot=I90_FRONT_CAMERA_ROT, convention="world")``."""
I90_DVL_POS: tuple[float, float, float] = (0.0, 0.0, -0.225)
I90_DVL_ROT: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
"""``dvl_link``: transducer face centre at the bottom centre, 20 mm above the skid line; FRD (z down, x forward)."""
I90_DVL_OFFSET_FOR_DVLCFG: tuple[tuple[float, float, float], tuple[float, float, float, float]] = (
    I90_DVL_POS,
    (0.0, 0.0, 0.0, 1.0),
)
"""rov_lab's ``DVLCfg`` (prim_path = rov_base body) casts its beams along -z of an un-rotated offset frame, so use
this (pos, identity rot) offset there rather than the FRD ``dvl_link`` orientation."""
I90_PRESSURE_SENSOR_POS: tuple[float, float, float] = (-0.545, -0.200, 0.10133)
I90_PRESSURE_SENSOR_ROT: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
"""``pressure_sensor_link``: sensing port at the stern, under the foam; depth = water_surface_z - z_world(link)."""

##
# Mass properties and hydrostatics (design values; vehicle = rov_base + sensor pods, arm = Bravo links)
##

WATER_DENSITY_KG_M3: float = 997.0
I90_VEHICLE_MASS_KG: float = 120.0
I90_ROV_BASE_MASS_KG: float = 116.1  # 120 kg minus camera (1.2), DVL (2.5) and pressure sensor (0.2) pods
I90_SENSOR_LINK_MASS_KG: dict[str, float] = {FRONT_CAMERA_LINK: 1.2, DVL_LINK: 2.5, PRESSURE_SENSOR_LINK: 0.2}
I90_VEHICLE_DISPLACED_VOLUME_M3: float = 0.12537613  # vehicle alone: +5.0 kg reserve buoyancy
I90_ARM_DISPLACED_VOLUME_M3: float = sum(LINK_DISPLACED_VOLUME_M3.values())  # 5.015 L
I90_SYSTEM_MASS_KG: float = 129.5
I90_SYSTEM_NET_BUOYANCY_KG: float = 0.5
"""Vehicle + arm: 997 * (0.125376 + 0.005015) - 129.5 = +0.50 kg (+4.9 N), slightly positive."""
I90_VEHICLE_COM_M: tuple[float, float, float] = (-0.075477, -0.000333, 0.015088)
I90_VEHICLE_COB_M: tuple[float, float, float] = (-0.051435, -0.000045, 0.101518)
I90_ROV_BASE_COM_M: tuple[float, float, float] = (-0.082256, 0.0, 0.017978)
I90_ROV_BASE_PRINCIPAL_INERTIA_KG_M2: tuple[float, float, float] = (7.388383, 10.975614, 13.650398)
I90_SYSTEM_STOW_COM_M: tuple[float, float, float] = (-0.027182, 0.000576, 0.001510)
I90_SYSTEM_STOW_COB_M: tuple[float, float, float] = (-0.027182, 0.001646, 0.091075)
I90_SYSTEM_HOME_COM_M: tuple[float, float, float] = (-0.022297, -0.000309, -0.000385)
"""Combined (vehicle + arm) centre of mass in rov_base at the stow / home poses. In stow the combined CoM lies
directly below the combined CoB (level trim, solved via the lead ballast position): CoB - CoM = 89.6 mm."""

##
# Hydrodynamics: exactly the physical fields of rov_lab's HydroParamsCfg (use HydroParamsCfg(**I90_HYDRO_PARAMS))
##
# Same form, units and sign convention as the BlueROV defaults in hydrodynamics.py: diagonal coefficients in the
# order [surge, sway, heave, roll, pitch, yaw], all positive, force = -(D_lin + D_quad * |v|) * v (N, N*m).
#
# * Surge (calibrated): the 60 kgf (588.4 N) forward thrust balances drag at the 3 kn (1.5433 m/s) top speed,
#   F = D_lin * U + D_quad * U^2. The linear/quadratic ratio is kept from the BlueROV defaults
#   (4.03 / 18.18 = 0.2217 m/s crossover), giving D_quad = 216.0, D_lin = 47.9. Equivalent Cd = 1.26 on the
#   0.343 m^2 frontal area (typical open frame).
# * Sway / heave: surge coefficients scaled by projected area (same Cd): A_x = W*H = 0.343, A_y = L*H = 0.539,
#   A_z = L*W = 0.770 m^2 -> x1.571 and x2.245.
# * Rotation (strip/plate): each envelope face is cut into strips; a strip at lever arm r moving at omega*r
#   carries the per-area surge coefficients q = D_quad_x / A_x = 629.8 N*s^2/m^4 and c = D_lin_x / A_x =
#   139.6 N*s/m^3. Integrating over the faces whose normal is perpendicular to the rotation axis:
#     quadratic: q/32 * sum(face_width * span^4), linear: c/12 * sum(face_width * span^3), e.g. yaw =
#     q/32 * (H*L^4 + H*W^4) = 16.43 N*m*s^2, c/12 * (H*L^3 + H*W^3) = 9.54 N*m*s.
# * Added mass: Lamb's coefficients for the ellipsoid inscribed in the envelope (semi-axes 0.55, 0.35, 0.245 m,
#   rho*V = 197 kg; alpha0, beta0, gamma0 = 0.367, 0.655, 0.977): m_ii = k_i * rho * V, rotational terms from the
#   Lamb/Imlay ellipsoid formulas. Only used when HydroParamsCfg.enable_added_mass / enable_coriolis are set.
# * volume: vehicle (hull) displaced volume only; the arm's 5.015 L belongs to its own links.
# * co_bm: effective metacentric arm so that co_bm * rho * g * volume equals the righting stiffness of the whole
#   vehicle + stowed-arm system (114.2 N*m/rad; system CoB - CoM = 89.6 mm). The hull alone has 86.4 mm.
# * g: 9.81 (physical value; the BlueROV default keeps MarineGym's 9.8).
# Steady-state speeds with these values: surge 1.543 m/s at 588 N (1.062 m/s at -294 N), sway 0.879 m/s at
# 329 N, heave 0.689 m/s at 304 N; yaw 2.19 rad/s at the servo's 100 N*m clamp.
I90_HYDRO_PARAMS: dict[str, float | tuple[float, ...]] = {
    "volume": 0.1253761,
    "co_bm": 0.09315,
    "rho": 997.0,
    "g": 9.81,
    "added_mass": (44.288, 95.993, 188.293, 0.749, 5.361, 1.549),
    "linear_damping": (47.883, 75.244, 107.491, 5.895, 11.797, 9.542),
    "quadratic_damping": (216.006, 339.438, 484.912, 6.446, 20.963, 16.434),
}

##
# Thrusters (metadata for a later thrust-allocation model; the current servo caps per-axis force)
##


@dataclass(frozen=True)
class ThrusterSpec:
    """One thruster in ``rov_base``: position [m], unit force direction for forward spin, limits [N] (>= 0)."""

    name: str
    pos: tuple[float, float, float]
    direction: tuple[float, float, float]
    max_forward_n: float
    max_reverse_n: float


# Fit (make_i90_bravo7_urdf.thruster_fit): all six pods are DWT6537 with one bollard rating.
# * Horizontal, vectored X (diagonal pairs parallel), alpha = atan(33.5 / 30) = 48.15 deg from +x:
#   forward 4*T_fwd*cos(alpha) = 60 kgf, reverse 4*T_rev*cos(alpha) = 30 kgf (-> T_rev/T_fwd = 0.5),
#   lateral (zero surge/yaw, all four at the reverse limit) 4*T_rev*sin(alpha) = 33.5 kgf.
#   -> T_fwd = 220.5 N (22.5 kgf), T_rev = 110.25 N.
# * Vertical: the same rating would give 45 kgf up / 22.5 kgf down; the catalogue lists 31 kgf, read as the strong
#   (up) direction and attributed to an installation (thrust-deduction) factor 0.689 of the pods in the foam wells.
# Allocation (Fx, Fy, Fz, Mz about I90_THRUST_ALLOCATION_REF_M, other actuated components held at zero) vs the
# catalogue: +x 60.0, -x 30.0, +-y 33.5, +z 31.0 kgf (residual < 1e-4). Not in the catalogue: -z = 15.5 kgf, i.e.
# half of the servo's symmetric 304 N heave cap; yaw torque 180.8 N*m (servo clamp 100 N*m).
# The pods are symmetric about the combined stowed CoM (horizontal ones at its height), so pure surge/sway/heave
# produce no yaw or pitch moment there.
I90_THRUST_ALLOCATION_REF_M: tuple[float, float, float] = (-0.027182, 0.0, 0.001510)
"""Combined stowed CoM projected on the centre line (moment reference for the allocation)."""
I90_THRUSTERS: tuple[ThrusterSpec, ...] = (
    ThrusterSpec("thr_fl", (0.312818, 0.235, 0.00151), (0.66712, -0.74495, 0.0), 220.5, 110.25),
    ThrusterSpec("thr_fr", (0.312818, -0.235, 0.00151), (0.66712, 0.74495, 0.0), 220.5, 110.25),
    ThrusterSpec("thr_rl", (-0.367182, 0.235, 0.00151), (0.66712, 0.74495, 0.0), 220.5, 110.25),
    ThrusterSpec("thr_rr", (-0.367182, -0.235, 0.00151), (0.66712, -0.74495, 0.0), 220.5, 110.25),
    ThrusterSpec("thr_vl", (-0.027182, 0.170, 0.183165), (0.0, 0.0, 1.0), 152.003, 76.002),
    ThrusterSpec("thr_vr", (-0.027182, -0.170, 0.183165), (0.0, 0.0, 1.0), 152.003, 76.002),
)

##
# Joint poses
##

_FINGERS_OPEN: dict[str, float] = {FINGER_LEFT_JOINT: 0.03, FINGER_RIGHT_JOINT: 0.03}

I90_BRAVO7_HOME_JOINT_POS: dict[str, float] = {
    BASE_YAW_JOINT: 0.0,
    SHOULDER_PITCH_JOINT: 0.66454,
    ELBOW_PITCH_JOINT: 2.37321,
    FOREARM_ROLL_JOINT: 0.0,
    WRIST_PITCH_JOINT: -1.72876,
    EE_ROLL_JOINT: 0.0,
    **_FINGERS_OPEN,
}
"""Ready pose: TCP 0.40 m ahead of the bow at z = -0.15 (about 12 deg below the camera axis), jaws pointing forward
15 deg down, 60 mm open. Elbow hangs 0.16 m below the skid line, so keep the hull >= 0.2 m off the seabed."""

I90_BRAVO7_STOW_JOINT_POS: dict[str, float] = {
    BASE_YAW_JOINT: 0.5 * math.pi,
    SHOULDER_PITCH_JOINT: 0.40,
    ELBOW_PITCH_JOINT: 2.40,
    FOREARM_ROLL_JOINT: 0.0,
    WRIST_PITCH_JOINT: 0.74,
    EE_ROLL_JOINT: 0.0,
    FINGER_LEFT_JOINT: 0.005,
    FINGER_RIGHT_JOINT: 0.005,
}
"""Arm folded flat in the horizontal plane at z = -0.17 ahead of the bow: x in [0.31, 0.77], y in [-0.29, 0.32],
z in [-0.23, -0.11] (above the skid line, clear of the frame and camera view)."""

I90_BRAVO7_REACH_POSES: dict[str, dict[str, float]] = {
    "seabed_0.30": {SHOULDER_PITCH_JOINT: -0.06119, ELBOW_PITCH_JOINT: 2.15844, WRIST_PITCH_JOINT: -1.92272},
    "seabed_0.45": {SHOULDER_PITCH_JOINT: 1.68771, ELBOW_PITCH_JOINT: -1.68131, WRIST_PITCH_JOINT: -0.00640},
    "seabed_0.60": {SHOULDER_PITCH_JOINT: 1.17433, ELBOW_PITCH_JOINT: -1.10155, WRIST_PITCH_JOINT: -0.07278},
    "forward_0.50": {SHOULDER_PITCH_JOINT: 0.24827, ELBOW_PITCH_JOINT: 2.06266, WRIST_PITCH_JOINT: -0.74013},
}
"""Planar IK solutions (other arm joints 0, fingers 0.03) for the TCP at: the seabed 0.30 / 0.45 / 0.60 m below the
hull bottom, 0.25 m ahead of the bow (jaws 10 / 0 / 0 deg from vertical), and hull-bottom height 0.50 m ahead of
the bow (jaws horizontal). The 0.30 m and forward targets use the elbow-down branch (the elbow-up branch would
need the shoulder beyond its 100 deg limit)."""
I90_BRAVO7_REACH_TCP_TARGETS_M: dict[str, tuple[float, float, float]] = {
    "seabed_0.30": (0.80, 0.0, -0.545),
    "seabed_0.45": (0.80, 0.0, -0.695),
    "seabed_0.60": (0.80, 0.0, -0.845),
    "forward_0.50": (1.05, 0.0, -0.245),
}

##
# Configuration
##

I90_MAX_LINEAR_VELOCITY_M_S: float = 5.0
"""PhysX per-body clamp [m/s]: ~3x the 1.54 m/s top speed, above arm-tip speeds during fast yaw."""
I90_MAX_ANGULAR_VELOCITY_DEG_S: float = 500.0
"""PhysX per-body clamp [deg/s] (PhysX unit). It applies to every link: the hull's terminal yaw rate at the full
thruster yaw torque is 174 deg/s (3.04 rad/s) and the arm's revolute joint speeds add up to 268 deg/s, so 500 deg/s
never binds physically but still catches numerical blow-ups."""

I90_BRAVO7_CFG: ArticulationCfg = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(I90_BRAVO7_USD_PATH),
        activate_contact_sensors=False,
        fix_root_link=False,
        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=True,  # neutral buoyancy assumed (rov_lab convention)
            linear_damping=0.0,  # no PhysX damping: hydrodynamic damping comes from apply_hydro_wrench only
            angular_damping=0.0,
            max_linear_velocity=I90_MAX_LINEAR_VELOCITY_M_S,
            max_angular_velocity=I90_MAX_ANGULAR_VELOCITY_DEG_S,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=PhysxArticulationCfg(
            enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos=I90_BRAVO7_HOME_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    # same actuator groups / gains / limits as the standalone arm (bravo7_large, bravo7_wrist, bravo7_jaws)
    actuators={name: copy.deepcopy(actuator) for name, actuator in BRAVO7_CLASS_CFG.actuators.items()},
    soft_joint_pos_limit_factor=1.0,
)
"""I90-class ROV + Bravo 7-class arm, floating base, gravity off, implicit PD arm drives."""

__all__ = [
    "ARM_BASE_LINK",
    "ARM_MOUNT_JOINT",
    "BRAVO7_CLASS_JOINT_LIMITS",
    "DVL_LINK",
    "EE_LINK",
    "FRONT_CAMERA_LINK",
    "GRASP_POINT_OFFSET_M",
    "I90_BRAVO7_CFG",
    "I90_BRAVO7_HOME_JOINT_POS",
    "I90_BRAVO7_PRIM_PATHS",
    "I90_BRAVO7_STOW_JOINT_POS",
    "I90_HYDRO_PARAMS",
    "I90_THRUSTERS",
    "JAW_TIP_OFFSET_M",
    "PRESSURE_SENSOR_LINK",
    "ROV_BASE_LINK",
    "ThrusterSpec",
]
