#!/usr/bin/env python3
"""Generate the parametric "Bravo 7-class" underwater manipulator URDF.

The model is designed from public specifications only (Reach Bravo 7 tech sheet values listed
under "Spec targets" below). No vendor URDF, mesh or SDK file was used: every link is built from
cylinders and boxes whose dimensions are the parameters at the top of this file.

Functional layout, base to tip (zero configuration = arm fully extended along +x of base_link)::

    base_link (mount flange at z = 0, flange normal -z)
      base_yaw_joint       revolute  axis +z   (base rotate)
    shoulder_link
      shoulder_pitch_joint revolute  axis -y   (shoulder bend, + raises the arm)
    upper_arm_link
      elbow_pitch_joint    revolute  axis -y   (elbow bend, + raises the forearm)
    elbow_link
      forearm_roll_joint   revolute  axis +x   (forearm rotate)
    forearm_link
      wrist_pitch_joint    revolute  axis -y   (wrist bend, + raises the jaws)
    wrist_link
      ee_roll_joint        revolute  axis +x   (end-effector rotate, +-720 deg = +-2 turns)
    wrist_roll_link
      ft_sensor_joint      fixed               (6-axis wrench readout location)
    jaw_base_link
      finger_left_joint    prismatic axis +y   (jaws; opening = 2 * q)
      finger_right_joint   prismatic axis -y   (same target as the left finger, no mimic joint)

Reach is defined as the distance from the shoulder pitch axis to the jaw tip at full extension.

Outputs (repository-relative):
    rov_lab/assets_src/robots/bravo7_class/bravo7_class.urdf
    rov_lab/assets_src/robots/bravo7_class/bravo7_class_params.json   (derived table, for review/tests)

Usage:
    python rov_lab/scripts/tools/make_bravo7_class_urdf.py [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

# =============================================================================================
# Spec targets (public Reach Bravo 7 tech sheet)
# =============================================================================================
SPEC_REACH_M: float = 0.900  # shoulder pitch axis -> jaw tip at full extension
SPEC_OUTER_DIAMETER_M: float = 0.092
SPEC_MASS_AIR_KG: float = 9.5
SPEC_MASS_WATER_KG: float = 4.5
SPEC_WRIST_ROLL_TORQUE_NM: float = 20.0
SPEC_WRIST_ROLL_SPEED_DEG_S: float = 68.0
SPEC_GRAB_FORCE_N: float = 80.0 * 9.80665  # 80 kgf ~= 785 N squeeze between the jaws
SPEC_LIFT_KG: float = 10.0
WATER_DENSITY_KG_M3: float = 997.0
GRAVITY_M_S2: float = 9.81

# =============================================================================================
# Geometry parameters [m] (our own design; chain lengths sum to SPEC_REACH_M)
# =============================================================================================
R_MODULE: float = SPEC_OUTER_DIAMETER_M / 2.0  # 92 mm actuator modules (base/shoulder/elbow/wrist pitch)
R_WRIST_MODULE: float = 0.040  # 80 mm distal modules (wrist/end-effector roll, F/T sensor)
R_UPPER_TUBE: float = 0.030  # 60 mm upper-arm tube between modules
R_FOREARM_TUBE: float = 0.030  # 60 mm forearm tube between modules

FLANGE_DIAMETER: float = 0.120
FLANGE_THICKNESS: float = 0.010
BASE_HEIGHT: float = 0.090  # mount face -> base_yaw_joint
SHOULDER_HEIGHT: float = 0.100  # base_yaw_joint -> shoulder pitch axis (along +z)
SHOULDER_HOUSING_LEN: float = 0.090  # pitch-module housing length along its axis
ELBOW_HOUSING_LEN: float = 0.090
WRIST_HOUSING_LEN: float = 0.090

UPPER_ARM_LENGTH: float = 0.300  # shoulder axis -> elbow axis
ELBOW_LENGTH: float = 0.100  # elbow axis -> forearm roll joint
FOREARM_LENGTH: float = 0.220  # forearm roll joint -> wrist pitch axis
WRIST_LENGTH: float = 0.070  # wrist pitch axis -> ee roll joint
WRIST_ROLL_LENGTH: float = 0.060  # ee roll joint -> ft_sensor_joint
JAW_BASE_LENGTH: float = 0.050  # ft_sensor_joint -> palm (finger joint origin)
FINGER_LENGTH: float = 0.100  # palm -> jaw tip
FT_SENSOR_THICKNESS: float = 0.015
JAW_HOUSING_SIZE_YZ: tuple[float, float] = (0.120, 0.050)
FINGER_SIZE_YZ: tuple[float, float] = (0.020, 0.040)  # finger thickness (closing dir) x height
FINGER_STROKE: float = 0.050  # per finger; max jaw opening = 2 * stroke = 100 mm
GRASP_POINT_FROM_PALM: float = 0.060  # nominal grasp centre along the fingers (for TCP / payload)

# =============================================================================================
# Mass budget [kg] (sums to SPEC_MASS_AIR_KG; heavier where actuators sit)
# =============================================================================================
LINK_MASS_KG: dict[str, float] = {
    "base_link": 1.30,  # flange + base-rotate actuator stator
    "shoulder_link": 1.60,  # shoulder pitch actuator (largest torque)
    "upper_arm_link": 1.55,  # tube + elbow actuator
    "elbow_link": 0.80,  # forearm-roll actuator stator
    "forearm_link": 1.50,  # tube + wrist pitch actuator
    "wrist_link": 0.55,  # end-effector roll actuator stator
    "wrist_roll_link": 0.55,  # roll output + wiring
    "jaw_base_link": 1.15,  # F/T sensor + jaw drive
    "finger_left_link": 0.25,
    "finger_right_link": 0.25,
}

# =============================================================================================
# Joint limits and drive parameters
# =============================================================================================
DEG: float = math.pi / 180.0
EE_ROLL_LIMIT_RAD: float = 4.0 * math.pi  # +-2 turns ("continuous" while avoiding angle wrap)


@dataclass
class JointLimitSpec:
    """Limits of one actuated joint (units: rad, rad/s, N*m or m, m/s, N)."""

    lower: float
    upper: float
    velocity: float
    effort: float
    damping: float  # passive viscous joint damping (URDF <dynamics damping>)
    friction: float  # static (Coulomb) joint friction (URDF <dynamics friction>)
    rationale: str


JOINT_LIMITS: dict[str, JointLimitSpec] = {
    "base_yaw_joint": JointLimitSpec(
        -175 * DEG, 175 * DEG, 30 * DEG, 80.0, 2.0, 1.0,
        "350 deg cable-wrap range; 80 N*m = 1.5x the 52 N*m needed at mid reach (10 kg wet) when the base axis "
        "is horizontal (side/forward mount or pitched ROV); 30 deg/s keeps tau*omega ~42 W like the large modules.",
    ),
    "shoulder_pitch_joint": JointLimitSpec(
        -100 * DEG, 100 * DEG, 20 * DEG, 120.0, 3.0, 1.5,
        "+-100 deg from horizontal; 120 N*m = 2.3x the 52 N*m mid-reach need and still 1.16x the 103 N*m "
        "full-extension need (10 kg wet); 20 deg/s -> tau*omega ~42 W, 0.31 m/s tip speed at full reach.",
    ),
    "elbow_pitch_joint": JointLimitSpec(
        -150 * DEG, 150 * DEG, 30 * DEG, 80.0, 2.0, 1.0,
        "+-150 deg fold; 80 N*m = 1.9x the 42 N*m mid-reach need, 1.2x the 65 N*m full-extension need; "
        "30 deg/s -> tau*omega ~42 W.",
    ),
    "forearm_roll_joint": JointLimitSpec(
        -175 * DEG, 175 * DEG, 60 * DEG, 40.0, 0.5, 0.5,
        "350 deg cable-wrap range; 40 N*m = 1.5x the 26 N*m needed with the jaws held sideways "
        "(wrist bent 90 deg, 10 kg wet); 60 deg/s -> tau*omega ~42 W.",
    ),
    "wrist_pitch_joint": JointLimitSpec(
        -120 * DEG, 120 * DEG, 60 * DEG, 40.0, 0.5, 0.5,
        "+-120 deg; 40 N*m = 1.5x the 26 N*m needed to hold 10 kg wet at the grasp point (0.24 m); "
        "60 deg/s -> tau*omega ~42 W.",
    ),
    "ee_roll_joint": JointLimitSpec(
        -EE_ROLL_LIMIT_RAD, EE_ROLL_LIMIT_RAD, SPEC_WRIST_ROLL_SPEED_DEG_S * DEG, SPEC_WRIST_ROLL_TORQUE_NM, 0.2, 0.25,
        "Spec: 20 N*m, 68 deg/s (smallest module, tau*omega ~24 W). +-4 pi stands in for continuous rotation "
        "so position targets never wrap.",
    ),
    "finger_left_joint": JointLimitSpec(
        0.0, FINGER_STROKE, 0.040, SPEC_GRAB_FORCE_N, 20.0, 5.0,
        "Spec grab force 80 kgf (785 N squeeze = per-finger force); 0-50 mm stroke (100 mm opening); "
        "40 mm/s per finger (785 N * 0.04 m/s ~31 W).",
    ),
    "finger_right_joint": JointLimitSpec(
        0.0, FINGER_STROKE, 0.040, SPEC_GRAB_FORCE_N, 20.0, 5.0,
        "Mirror of finger_left_joint; driven with the same target (no mimic joint).",
    ),
}

# =============================================================================================
# Visual colours (RGBA)
# =============================================================================================
MATERIALS: dict[str, tuple[float, float, float, float]] = {
    "housing_grey": (0.22, 0.23, 0.25, 1.0),
    "tube_black": (0.08, 0.08, 0.09, 1.0),
    "flange_silver": (0.70, 0.71, 0.73, 1.0),
    "sensor_blue": (0.15, 0.30, 0.60, 1.0),
    "jaw_yellow": (0.90, 0.72, 0.10, 1.0),
}

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR: Path = REPO_ROOT / "rov_lab" / "assets_src" / "robots" / "bravo7_class"
ROBOT_NAME: str = "bravo7_class"

HALF_PI: float = math.pi / 2.0
ROT_Z_TO_X: tuple[float, float, float] = (0.0, HALF_PI, 0.0)  # cylinder axis along +x
ROT_Z_TO_Y: tuple[float, float, float] = (HALF_PI, 0.0, 0.0)  # cylinder axis along y

Vec3 = tuple[float, float, float]
Mat3 = list[list[float]]


# =============================================================================================
# Small linear-algebra helpers (pure Python so the generator runs with any Python 3.10+)
# =============================================================================================
def mat_mul(a: Mat3, b: Mat3) -> Mat3:
    """Return the 3x3 matrix product a @ b."""
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def mat_t(a: Mat3) -> Mat3:
    """Return the transpose of a 3x3 matrix."""
    return [[a[j][i] for j in range(3)] for i in range(3)]


def mat_vec(a: Mat3, v: Vec3) -> Vec3:
    """Return a @ v for a 3x3 matrix and a 3-vector."""
    return (
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    )


def rpy_to_mat(rpy: Vec3) -> Mat3:
    """URDF fixed-axis roll-pitch-yaw to rotation matrix (R = Rz(yaw) Ry(pitch) Rx(roll))."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def axis_angle_to_mat(axis: Vec3, angle: float) -> Mat3:
    """Rodrigues rotation matrix for a unit axis and an angle [rad]."""
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    return [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]


def cross(a: Vec3, b: Vec3) -> Vec3:
    """Cross product of two 3-vectors."""
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def dot(a: Vec3, b: Vec3) -> float:
    """Dot product of two 3-vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vadd(a: Vec3, b: Vec3) -> Vec3:
    """Vector sum."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vsub(a: Vec3, b: Vec3) -> Vec3:
    """Vector difference."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vscale(a: Vec3, s: float) -> Vec3:
    """Scale a vector."""
    return (a[0] * s, a[1] * s, a[2] * s)


# =============================================================================================
# Model description
# =============================================================================================
@dataclass
class Primitive:
    """A collision/visual primitive expressed in its link frame (URDF cylinders run along local z)."""

    kind: str  # "cylinder" | "box"
    size: tuple[float, ...]  # cylinder: (radius, length); box: (sx, sy, sz)
    xyz: Vec3
    rpy: Vec3 = (0.0, 0.0, 0.0)
    material: str = "housing_grey"

    def volume(self) -> float:
        """Solid volume [m^3]."""
        if self.kind == "cylinder":
            r, h = self.size
            return math.pi * r * r * h
        sx, sy, sz = self.size
        return sx * sy * sz

    def local_inertia_per_kg(self) -> Vec3:
        """Principal inertia per unit mass about the primitive centroid, in primitive axes."""
        if self.kind == "cylinder":
            r, h = self.size
            ixx = (3.0 * r * r + h * h) / 12.0
            return (ixx, ixx, r * r / 2.0)
        sx, sy, sz = self.size
        return ((sy * sy + sz * sz) / 12.0, (sx * sx + sz * sz) / 12.0, (sx * sx + sy * sy) / 12.0)


@dataclass
class LinkSpec:
    """One URDF link with primitives (used for visual, collision, inertia and displaced volume)."""

    name: str
    primitives: list[Primitive]
    mass: float = 0.0
    com: Vec3 = (0.0, 0.0, 0.0)
    inertia: Mat3 = field(default_factory=lambda: [[0.0] * 3 for _ in range(3)])
    geometric_volume: float = 0.0
    displaced_volume: float = 0.0


@dataclass
class JointSpec:
    """One URDF joint."""

    name: str
    kind: str  # "revolute" | "prismatic" | "fixed"
    parent: str
    child: str
    xyz: Vec3
    axis: Vec3 = (0.0, 0.0, 1.0)


def cyl_x(r: float, x0: float, x1: float, material: str) -> Primitive:
    """Cylinder of radius r spanning x0..x1 along the link x axis."""
    return Primitive("cylinder", (r, x1 - x0), (0.5 * (x0 + x1), 0.0, 0.0), ROT_Z_TO_X, material)


def cyl_z(r: float, z0: float, z1: float, material: str) -> Primitive:
    """Cylinder of radius r spanning z0..z1 along the link z axis."""
    return Primitive("cylinder", (r, z1 - z0), (0.0, 0.0, 0.5 * (z0 + z1)), (0.0, 0.0, 0.0), material)


def cyl_y(r: float, length: float, center: Vec3, material: str) -> Primitive:
    """Cylinder of radius r and given length along y (a pitch-actuator housing) centred at `center`."""
    return Primitive("cylinder", (r, length), center, ROT_Z_TO_Y, material)


def build_links() -> list[LinkSpec]:
    """Create the link primitives from the geometry parameters (non-overlapping within each link)."""
    jy, jz = JAW_HOUSING_SIZE_YZ
    fy, fz = FINGER_SIZE_YZ
    jaw_box_len = JAW_BASE_LENGTH - FT_SENSOR_THICKNESS
    finger_x = 0.5 * FINGER_LENGTH
    return [
        LinkSpec("base_link", [
            cyl_z(0.5 * FLANGE_DIAMETER, 0.0, FLANGE_THICKNESS, "flange_silver"),
            cyl_z(R_MODULE, FLANGE_THICKNESS, BASE_HEIGHT, "housing_grey"),
        ]),
        LinkSpec("shoulder_link", [
            cyl_z(R_MODULE, 0.0, SHOULDER_HEIGHT - R_MODULE, "housing_grey"),
            cyl_y(R_MODULE, SHOULDER_HOUSING_LEN, (0.0, 0.0, SHOULDER_HEIGHT), "housing_grey"),
        ]),
        LinkSpec("upper_arm_link", [
            cyl_x(R_UPPER_TUBE, R_MODULE, UPPER_ARM_LENGTH - R_MODULE, "tube_black"),
            cyl_y(R_MODULE, ELBOW_HOUSING_LEN, (UPPER_ARM_LENGTH, 0.0, 0.0), "housing_grey"),
        ]),
        LinkSpec("elbow_link", [
            cyl_x(R_MODULE, R_MODULE, ELBOW_LENGTH, "housing_grey"),
        ]),
        LinkSpec("forearm_link", [
            cyl_x(R_FOREARM_TUBE, 0.0, FOREARM_LENGTH - R_MODULE, "tube_black"),
            cyl_y(R_MODULE, WRIST_HOUSING_LEN, (FOREARM_LENGTH, 0.0, 0.0), "housing_grey"),
        ]),
        LinkSpec("wrist_link", [
            cyl_x(R_WRIST_MODULE, R_MODULE, WRIST_LENGTH, "housing_grey"),
        ]),
        LinkSpec("wrist_roll_link", [
            cyl_x(R_WRIST_MODULE, 0.0, WRIST_ROLL_LENGTH, "housing_grey"),
        ]),
        LinkSpec("jaw_base_link", [
            cyl_x(R_WRIST_MODULE, 0.0, FT_SENSOR_THICKNESS, "sensor_blue"),
            Primitive("box", (jaw_box_len, jy, jz), (FT_SENSOR_THICKNESS + 0.5 * jaw_box_len, 0.0, 0.0),
                      material="housing_grey"),
        ]),
        # Finger frames sit on the palm; the inner (gripping) face is the plane y = 0 of the finger frame,
        # so the jaw opening equals 2 * q.
        LinkSpec("finger_left_link", [
            Primitive("box", (FINGER_LENGTH, fy, fz), (finger_x, 0.5 * fy, 0.0), material="jaw_yellow"),
        ]),
        LinkSpec("finger_right_link", [
            Primitive("box", (FINGER_LENGTH, fy, fz), (finger_x, -0.5 * fy, 0.0), material="jaw_yellow"),
        ]),
    ]


def build_joints() -> list[JointSpec]:
    """Create the kinematic chain (all joint frames are axis-aligned with base_link at q = 0)."""
    neg_y: Vec3 = (0.0, -1.0, 0.0)
    return [
        JointSpec("base_yaw_joint", "revolute", "base_link", "shoulder_link", (0.0, 0.0, BASE_HEIGHT), (0.0, 0.0, 1.0)),
        JointSpec("shoulder_pitch_joint", "revolute", "shoulder_link", "upper_arm_link", (0.0, 0.0, SHOULDER_HEIGHT),
                  neg_y),
        JointSpec("elbow_pitch_joint", "revolute", "upper_arm_link", "elbow_link", (UPPER_ARM_LENGTH, 0.0, 0.0), neg_y),
        JointSpec("forearm_roll_joint", "revolute", "elbow_link", "forearm_link", (ELBOW_LENGTH, 0.0, 0.0),
                  (1.0, 0.0, 0.0)),
        JointSpec("wrist_pitch_joint", "revolute", "forearm_link", "wrist_link", (FOREARM_LENGTH, 0.0, 0.0), neg_y),
        JointSpec("ee_roll_joint", "revolute", "wrist_link", "wrist_roll_link", (WRIST_LENGTH, 0.0, 0.0),
                  (1.0, 0.0, 0.0)),
        JointSpec("ft_sensor_joint", "fixed", "wrist_roll_link", "jaw_base_link", (WRIST_ROLL_LENGTH, 0.0, 0.0)),
        JointSpec("finger_left_joint", "prismatic", "jaw_base_link", "finger_left_link", (JAW_BASE_LENGTH, 0.0, 0.0),
                  (0.0, 1.0, 0.0)),
        JointSpec("finger_right_joint", "prismatic", "jaw_base_link", "finger_right_link",
                  (JAW_BASE_LENGTH, 0.0, 0.0), (0.0, -1.0, 0.0)),
    ]


def compute_mass_properties(links: list[LinkSpec]) -> float:
    """Fill mass, COM, inertia and displaced volume of every link; return the volume scale factor.

    Each link's mass is spread uniformly over its primitives, so the centre of mass coincides with
    the centre of buoyancy. Displaced volumes are the primitive volumes scaled by one common factor
    so that the whole arm displaces (m_air - m_water) / rho_water, i.e. weighs SPEC_MASS_WATER_KG in water.
    """
    total_geo = sum(p.volume() for link in links for p in link.primitives)
    target_volume = (SPEC_MASS_AIR_KG - SPEC_MASS_WATER_KG) / WATER_DENSITY_KG_M3
    scale = target_volume / total_geo
    for link in links:
        link.mass = LINK_MASS_KG[link.name]
        vols = [p.volume() for p in link.primitives]
        v_link = sum(vols)
        link.geometric_volume = v_link
        link.displaced_volume = v_link * scale
        com = (0.0, 0.0, 0.0)
        for p, v in zip(link.primitives, vols):
            com = vadd(com, vscale(p.xyz, v / v_link))
        link.com = com
        inertia: Mat3 = [[0.0] * 3 for _ in range(3)]
        for p, v in zip(link.primitives, vols):
            m_i = link.mass * v / v_link
            principal = p.local_inertia_per_kg()
            rot = rpy_to_mat(p.rpy)
            local: Mat3 = [[principal[i] * m_i if i == j else 0.0 for j in range(3)] for i in range(3)]
            rotated = mat_mul(mat_mul(rot, local), mat_t(rot))
            d = vsub(p.xyz, com)
            dd = dot(d, d)
            for i in range(3):
                for j in range(3):
                    inertia[i][j] += rotated[i][j] + m_i * ((dd if i == j else 0.0) - d[i] * d[j])
        link.inertia = [[0.0 if abs(v) < 1e-12 else v for v in row] for row in inertia]
    return scale


# =============================================================================================
# Forward kinematics + static torque check (rationale for the effort limits)
# =============================================================================================
def forward_kinematics(
    joints: list[JointSpec], q: dict[str, float]
) -> tuple[dict[str, tuple[Vec3, Mat3]], dict[str, tuple[Vec3, Vec3]]]:
    """Return link poses {link: (pos, R)} and joint axes {joint: (pos, axis_world)} in base_link."""
    poses: dict[str, tuple[Vec3, Mat3]] = {"base_link": ((0.0, 0.0, 0.0), [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])}
    axes: dict[str, tuple[Vec3, Vec3]] = {}
    for j in joints:  # joints are listed parent-before-child
        p_par, r_par = poses[j.parent]
        p_j = vadd(p_par, mat_vec(r_par, j.xyz))
        axis_w = mat_vec(r_par, j.axis)
        qj = q.get(j.name, 0.0)
        if j.kind == "revolute":
            r_child = mat_mul(r_par, axis_angle_to_mat(j.axis, qj))
            p_child = p_j
        elif j.kind == "prismatic":
            r_child = r_par
            p_child = vadd(p_j, vscale(axis_w, qj))
        else:
            r_child, p_child = r_par, p_j
        poses[j.child] = (p_child, r_child)
        axes[j.name] = (p_j, axis_w)
    return poses, axes


def subtree(joints: list[JointSpec], link: str) -> list[str]:
    """All links at and beyond `link`."""
    out = [link]
    for j in joints:
        if j.parent in out and j.child not in out:
            out.append(j.child)
    return out


def static_joint_torques(
    links: list[LinkSpec], joints: list[JointSpec], q: dict[str, float], gravity: Vec3, payload_kg: float
) -> dict[str, float]:
    """Joint torques [N*m] needed to hold pose q in water (wet link weights + net payload at the grasp point)."""
    poses, axes = forward_kinematics(joints, q)
    by_name = {link.name: link for link in links}
    g_norm = math.sqrt(dot(gravity, gravity))
    g_dir = vscale(gravity, 1.0 / g_norm)
    jaw_p, jaw_r = poses["jaw_base_link"]
    grasp_w = vadd(jaw_p, mat_vec(jaw_r, (JAW_BASE_LENGTH + GRASP_POINT_FROM_PALM, 0.0, 0.0)))
    out: dict[str, float] = {}
    for j in joints:
        if j.kind != "revolute":
            continue
        p_j, a_w = axes[j.name]
        tau = 0.0
        for name in subtree(joints, j.child):
            link = by_name[name]
            p_l, r_l = poses[name]
            com_w = vadd(p_l, mat_vec(r_l, link.com))
            wet_force = vscale(g_dir, (link.mass - WATER_DENSITY_KG_M3 * link.displaced_volume) * g_norm)
            tau += dot(cross(vsub(com_w, p_j), wet_force), a_w)
        tau += dot(cross(vsub(grasp_w, p_j), vscale(g_dir, payload_kg * g_norm)), a_w)
        out[j.name] = abs(tau)
    return out


def mid_reach_pose() -> dict[str, float]:
    """Elbow-up pose with the jaws horizontal and the grasp point at half reach, level with the shoulder axis."""
    distal = WRIST_LENGTH + WRIST_ROLL_LENGTH + JAW_BASE_LENGTH + GRASP_POINT_FROM_PALM
    wx, wz = 0.5 * SPEC_REACH_M - distal, 0.0  # wrist pitch axis in the shoulder (x, z) plane
    a, b = UPPER_ARM_LENGTH, ELBOW_LENGTH + FOREARM_LENGTH
    d2 = wx * wx + wz * wz
    # elbow-up solution of the planar 2-link problem
    cos_e = (d2 - a * a - b * b) / (2 * a * b)
    elbow = -math.acos(max(-1.0, min(1.0, cos_e)))  # negative = forearm folds down (axis -y)
    shoulder = math.atan2(wz, wx) - math.atan2(b * math.sin(elbow), a + b * math.cos(elbow))
    wrist = -(shoulder + elbow)  # keep the jaws horizontal
    return {"shoulder_pitch_joint": shoulder, "elbow_pitch_joint": elbow, "wrist_pitch_joint": wrist}


def torque_check(links: list[LinkSpec], joints: list[JointSpec]) -> dict[str, dict[str, float]]:
    """Static torque requirements for the effort-limit rationale (10 kg net payload, in water)."""
    down: Vec3 = (0.0, 0.0, -GRAVITY_M_S2)
    sideways: Vec3 = (0.0, -GRAVITY_M_S2, 0.0)  # base axis horizontal (side / forward mount)
    mid = mid_reach_pose()
    cases = {
        "mid_reach_upright": static_joint_torques(links, joints, mid, down, SPEC_LIFT_KG),
        "full_extension_upright": static_joint_torques(links, joints, {}, down, SPEC_LIFT_KG),
        "mid_reach_base_axis_horizontal": static_joint_torques(links, joints, mid, sideways, SPEC_LIFT_KG),
    }
    # forearm horizontal, wrist bent 90 deg and rolled 90 deg: jaws stick out sideways, so the payload
    # moment acts about the forearm roll axis (worst case for that joint)
    roll_case = {"wrist_pitch_joint": math.pi / 2, "forearm_roll_joint": math.pi / 2}
    cases["forearm_horizontal_jaws_sideways"] = static_joint_torques(links, joints, roll_case, down, SPEC_LIFT_KG)
    cases["mid_reach_pose_rad"] = mid
    return cases


# =============================================================================================
# URDF writer
# =============================================================================================
def fmt(v: float) -> str:
    """Compact float formatting for URDF attributes."""
    s = f"{v:.9g}"
    return "0" if s in ("-0", "0") else s


def fvec(v: tuple[float, ...]) -> str:
    """Space-separated vector."""
    return " ".join(fmt(x) for x in v)


def add_geometry(parent: ET.Element, prim: Primitive) -> None:
    """Append <origin> and <geometry> for a primitive."""
    ET.SubElement(parent, "origin", xyz=fvec(prim.xyz), rpy=fvec(prim.rpy))
    geom = ET.SubElement(parent, "geometry")
    if prim.kind == "cylinder":
        ET.SubElement(geom, "cylinder", radius=fmt(prim.size[0]), length=fmt(prim.size[1]))
    else:
        ET.SubElement(geom, "box", size=fvec(prim.size))


def build_urdf(links: list[LinkSpec], joints: list[JointSpec]) -> ET.Element:
    """Build the URDF XML tree."""
    robot = ET.Element("robot", name=ROBOT_NAME)
    robot.append(ET.Comment(
        " Parametric 'Bravo 7-class' underwater manipulator built from public specs only. "
        "Generated by rov_lab/scripts/tools/make_bravo7_class_urdf.py - edit the generator, not this file. "
    ))
    for name, rgba in MATERIALS.items():
        mat = ET.SubElement(robot, "material", name=name)
        ET.SubElement(mat, "color", rgba=fvec(rgba))
    for link in links:
        el = ET.SubElement(robot, "link", name=link.name)
        el.append(ET.Comment(
            f" displaced_volume_m3={link.displaced_volume:.6e} (rho={WATER_DENSITY_KG_M3:g}) "
            f"centre_of_buoyancy=centre_of_mass "
        ))
        inertial = ET.SubElement(el, "inertial")
        ET.SubElement(inertial, "origin", xyz=fvec(link.com), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=fmt(link.mass))
        i = link.inertia
        ET.SubElement(inertial, "inertia", ixx=fmt(i[0][0]), ixy=fmt(i[0][1]), ixz=fmt(i[0][2]),
                      iyy=fmt(i[1][1]), iyz=fmt(i[1][2]), izz=fmt(i[2][2]))
        for k, prim in enumerate(link.primitives):
            vis = ET.SubElement(el, "visual", name=f"{link.name}_visual_{k}")
            add_geometry(vis, prim)
            ET.SubElement(vis, "material", name=prim.material)
        for k, prim in enumerate(link.primitives):
            col = ET.SubElement(el, "collision", name=f"{link.name}_collision_{k}")
            add_geometry(col, prim)
    for j in joints:
        el = ET.SubElement(robot, "joint", name=j.name, type=j.kind)
        ET.SubElement(el, "origin", xyz=fvec(j.xyz), rpy="0 0 0")
        ET.SubElement(el, "parent", link=j.parent)
        ET.SubElement(el, "child", link=j.child)
        if j.kind != "fixed":
            lim = JOINT_LIMITS[j.name]
            ET.SubElement(el, "axis", xyz=fvec(j.axis))
            ET.SubElement(el, "limit", lower=fmt(lim.lower), upper=fmt(lim.upper),
                          effort=fmt(lim.effort), velocity=fmt(lim.velocity))
            ET.SubElement(el, "dynamics", damping=fmt(lim.damping), friction=fmt(lim.friction))
    ET.indent(robot, space="  ")
    return robot


def reach_from_chain(joints: list[JointSpec]) -> float:
    """Shoulder pitch axis -> jaw tip distance at q = 0, from forward kinematics."""
    poses, axes = forward_kinematics(joints, {})
    shoulder = axes["shoulder_pitch_joint"][0]
    jaw_p, jaw_r = poses["jaw_base_link"]
    tip = vadd(jaw_p, mat_vec(jaw_r, (JAW_BASE_LENGTH + FINGER_LENGTH, 0.0, 0.0)))
    return math.sqrt(dot(vsub(tip, shoulder), vsub(tip, shoulder)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    args = parser.parse_args()

    links = build_links()
    joints = build_joints()
    scale = compute_mass_properties(links)
    reach = reach_from_chain(joints)
    total_mass = sum(link.mass for link in links)
    total_disp = sum(link.displaced_volume for link in links)
    if abs(reach - SPEC_REACH_M) > 1e-9:
        raise ValueError(f"Chain lengths give reach {reach:.4f} m, expected {SPEC_REACH_M} m")
    if abs(total_mass - SPEC_MASS_AIR_KG) > 1e-9:
        raise ValueError(f"Link masses sum to {total_mass:.3f} kg, expected {SPEC_MASS_AIR_KG} kg")
    for link in links:
        if link.mass <= WATER_DENSITY_KG_M3 * link.displaced_volume:
            raise ValueError(f"{link.name} would be positively buoyant")
    torques = torque_check(links, joints)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    urdf_path = args.out_dir / f"{ROBOT_NAME}.urdf"
    tree = ET.ElementTree(build_urdf(links, joints))
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)

    params = {
        "robot": ROBOT_NAME,
        "generator": "rov_lab/scripts/tools/make_bravo7_class_urdf.py",
        "spec": {
            "reach_m": SPEC_REACH_M, "outer_diameter_m": SPEC_OUTER_DIAMETER_M, "mass_air_kg": SPEC_MASS_AIR_KG,
            "mass_water_kg": SPEC_MASS_WATER_KG, "water_density_kg_m3": WATER_DENSITY_KG_M3,
            "wrist_roll_torque_nm": SPEC_WRIST_ROLL_TORQUE_NM, "wrist_roll_speed_deg_s": SPEC_WRIST_ROLL_SPEED_DEG_S,
            "grab_force_n": SPEC_GRAB_FORCE_N, "lift_kg": SPEC_LIFT_KG,
        },
        "derived": {
            "reach_m": reach, "total_mass_kg": total_mass, "total_displaced_volume_m3": total_disp,
            "wet_weight_kg": total_mass - WATER_DENSITY_KG_M3 * total_disp,
            "displaced_volume_scale_vs_primitives": scale,
            "jaw_tip_offset_in_jaw_base_m": [JAW_BASE_LENGTH + FINGER_LENGTH, 0.0, 0.0],
            "grasp_point_offset_in_jaw_base_m": [JAW_BASE_LENGTH + GRASP_POINT_FROM_PALM, 0.0, 0.0],
        },
        "segment_lengths_m": {
            "flange_to_base_yaw": BASE_HEIGHT, "base_yaw_to_shoulder": SHOULDER_HEIGHT,
            "shoulder_to_elbow": UPPER_ARM_LENGTH, "elbow_to_forearm_roll": ELBOW_LENGTH,
            "forearm_roll_to_wrist_pitch": FOREARM_LENGTH, "wrist_pitch_to_ee_roll": WRIST_LENGTH,
            "ee_roll_to_ft_sensor": WRIST_ROLL_LENGTH, "ft_sensor_to_palm": JAW_BASE_LENGTH,
            "palm_to_jaw_tip": FINGER_LENGTH,
        },
        "links": {
            link.name: {
                "mass_kg": link.mass, "geometric_volume_m3": link.geometric_volume,
                "displaced_volume_m3": link.displaced_volume,
                "wet_mass_kg": link.mass - WATER_DENSITY_KG_M3 * link.displaced_volume,
                "com_m": list(link.com), "center_of_buoyancy_m": list(link.com),
                "inertia_kg_m2": link.inertia,
            }
            for link in links
        },
        "joints": {name: asdict(lim) for name, lim in JOINT_LIMITS.items()},
        "static_torque_check_nm": torques,
    }
    params_path = args.out_dir / f"{ROBOT_NAME}_params.json"
    params_path.write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")

    print(f"[bravo7_class] wrote {urdf_path}")
    print(f"[bravo7_class] wrote {params_path}")
    print(f"[bravo7_class] reach {reach:.4f} m | mass {total_mass:.2f} kg | displaced {total_disp * 1e3:.3f} L "
          f"| wet {params['derived']['wet_weight_kg']:.3f} kg | volume scale vs primitives {scale:.3f}")
    print(f"{'link':<18}{'mass kg':>9}{'V_disp L':>10}{'wet kg':>8}  com [m]")
    for link in links:
        wet = link.mass - WATER_DENSITY_KG_M3 * link.displaced_volume
        print(f"{link.name:<18}{link.mass:>9.3f}{link.displaced_volume * 1e3:>10.4f}{wet:>8.3f}  "
              f"({link.com[0]:+.4f}, {link.com[1]:+.4f}, {link.com[2]:+.4f})")
    print("static torque check [N*m] (10 kg net payload, in water):")
    for case, values in torques.items():
        print(f"  {case}: " + ", ".join(f"{k.replace('_joint', '')}={v:.1f}" for k, v in values.items()))
    print("effort limits: " + ", ".join(f"{k.replace('_joint', '')}={v.effort:.0f}" for k, v in JOINT_LIMITS.items()))


if __name__ == "__main__":
    main()
