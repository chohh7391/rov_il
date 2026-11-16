#!/usr/bin/env python3
"""Generate the "I90-class" ROV with the Bravo 7-class arm as ONE floating articulation (URDF + params JSON).

The vehicle is our own simplified design sized to the public DWTEK Investigator 90 (I90, observation class)
catalogue values listed under "Spec targets". No vendor CAD, mesh or OceanSim asset was used: the hull is a set
of boxes and cylinders (frame plates, skids, cross bars, buoyancy foam block, two pressure housings, six thruster
pods, arm bracket, trim ballast) plus three sensor pods on fixed joints (front camera, DVL, pressure sensor).
The arm links come unchanged from ``make_bravo7_class_urdf.py`` (imported, not copied).

Frames (vehicle ``rov_base``, FLU): origin at the centre of the 1100 x 700 x 490 mm envelope, +x forward (bow),
+y port, +z up. Hull bottom (skid underside) is z = -0.245 m, bow x = +0.55 m.

Design decisions (details in the params JSON):

* Mass budget: frame, housings, thrusters and lead trim ballast are low; the syntactic foam (400 kg/m^3) is the
  only light part and sits on top. The foam thickness and the ballast mass are solved so that the vehicle weighs
  exactly 120 kg in air and the vehicle + arm system is slightly positive (TARGET_NET_BUOYANCY_KG). The ballast
  position is solved so that vehicle + stowed arm floats level (combined CoM directly below the combined CoB).
* Thrusters: four horizontal DWT6537 pods in a vectored X layout (diagonal pairs parallel) and two vertical pods
  in foam wells. They are placed symmetric about the combined (stowed) CoM, horizontal pods at CoM height, so pure
  surge/sway/heave commands produce no yaw/pitch moment. Limits are fitted to the catalogue axis thrusts.
* Arm mount: forward-facing base. The Bravo ``base_link`` flange sits on a vertical bracket plate in the lower
  front bay with the base-rotate axis pointing forward (+x) at z = ARM_AXIS_Z; the shoulder pitch axis is 5 cm
  behind the bow. At base yaw 0 the arm works in the vertical centre plane (zero direction = straight down,
  +shoulder swings the arm forward); at base yaw +90 deg the same joints fold the arm flat in the horizontal
  plane for stowing ahead of the bow.

Outputs (repository-relative):
    rov_lab/assets_src/robots/i90_bravo7/i90_bravo7.urdf
    rov_lab/assets_src/robots/i90_bravo7/i90_bravo7_params.json

Usage:
    python rov_lab/scripts/tools/make_i90_bravo7_urdf.py [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_bravo7_class_urdf as bravo  # noqa: E402  (the arm generator; its links/joints are reused as-is)
from make_bravo7_class_urdf import (  # noqa: E402
    LinkSpec,
    Mat3,
    Primitive,
    Vec3,
    add_geometry,
    cyl_x,
    cyl_z,
    dot,
    fmt,
    fvec,
    mat_mul,
    mat_t,
    mat_vec,
    rpy_to_mat,
    vadd,
    vscale,
    vsub,
)

# =============================================================================================
# Spec targets (DWTEK Investigator 90 catalogue, observation class)
# =============================================================================================
SPEC_LENGTH_M: float = 1.100
SPEC_WIDTH_M: float = 0.700
SPEC_HEIGHT_M: float = 0.490
SPEC_MASS_AIR_KG: float = 120.0
SPEC_PAYLOAD_KG: float = 27.0
SPEC_SPEED_KN: float = 3.0
SPEC_DEPTH_M: float = 500.0
KGF_N: float = 9.80665
KNOT_M_S: float = 1852.0 / 3600.0
SPEC_THRUST_FWD_KGF: float = 60.0
SPEC_THRUST_REV_KGF: float = 30.0
SPEC_THRUST_LAT_KGF: float = 33.5
SPEC_THRUST_VERT_KGF: float = 31.0
SPEC_CATALOGUE_MANIPULATOR: str = "up to 5-axis, optional"

RHO_WATER: float = bravo.WATER_DENSITY_KG_M3  # 997 kg/m^3, same as the arm model
GRAVITY: float = 9.81
TARGET_NET_BUOYANCY_KG: float = 0.5
"""Vehicle + arm net buoyancy (+ = floats up). Slightly positive so a dead vehicle surfaces."""

# Material densities [kg/m^3]
RHO_FOAM: float = 400.0  # syntactic foam, 500-1000 m rating class
RHO_HDPE: float = 950.0  # frame plates, skids, cross bars
RHO_ALU: float = 2700.0  # arm bracket
RHO_LEAD: float = 11340.0  # trim ballast

# =============================================================================================
# Geometry parameters [m] (our own simplified layout inside the catalogue envelope)
# =============================================================================================
X_BOW: float = 0.5 * SPEC_LENGTH_M
Y_SIDE: float = 0.5 * SPEC_WIDTH_M
Z_TOP: float = 0.5 * SPEC_HEIGHT_M
Z_BOTTOM: float = -0.5 * SPEC_HEIGHT_M

SIDE_PLATE_T: float = 0.020
SIDE_PLATE_SOLID_FRACTION: float = 0.60  # lightening cut-outs (visual/collision stay solid boxes)
SKID_W: float = 0.070
SKID_H: float = 0.030
CROSSBAR_SECTION: float = 0.040
FOAM_X: tuple[float, float] = (-0.530, 0.440)  # foam ends 11 cm behind the bow, the camera pod sits in front
WELL_R: float = 0.065  # vertical-thruster wells through the foam
HOUSING_MCC: dict[str, float] = {"r": 0.090, "x0": -0.470, "x1": 0.000, "z": -0.025, "mass": 25.0}
HOUSING_TLC: dict[str, float] = {"r": 0.065, "x0": 0.020, "x1": 0.280, "z": -0.030, "mass": 8.0}
UPPER_CROSSBAR_X: tuple[float, ...] = (0.420, -0.500)
LOWER_CROSSBAR_X: tuple[float, ...] = (-0.480, -0.150)

# Thruster pods (DWT6537 class): duct envelope cylinders
THRUSTER_R: float = 0.060
H_THRUSTER_LEN: float = 0.140
V_THRUSTER_LEN: float = 0.110
THRUSTER_MASS_KG: float = 3.5
THRUSTER_DISPLACED_M3: float = 1.4e-3
H_THRUSTER_DX: float = 0.340  # +- x from the allocation reference (combined CoM)
H_THRUSTER_Y: float = 0.235
V_THRUSTER_Y: float = 0.170

# Trim ballast: two lead bars on the skids
BALLAST_W: float = 0.040
BALLAST_H: float = 0.025
BALLAST_Y: float = 0.300

# Arm mount (forward-facing base on a vertical bracket plate in the lower front bay)
ARM_FLANGE_X: float = 0.310  # flange face (base_link origin) x
ARM_AXIS_Z: float = -0.170  # base-rotate axis height; shoulder housing stays 29 mm above the skid line
ARM_MOUNT_RPY: Vec3 = (0.0, 0.5 * math.pi, 0.0)  # base_link z -> rov_base +x, base_link x -> rov_base -z
ARM_BRACKET: dict[str, float] = {"t": 0.015, "w": 0.160, "h": 0.130}

# Sensor frames
CAMERA_POS: Vec3 = (0.545, 0.0, 0.100)  # optical centre, in front of the foam
CAMERA_TILT_DEG: float = 20.0  # looking forward, 20 deg down
CAMERA_POD: dict[str, float] = {"r": 0.040, "len": 0.100, "mass": 1.2}
DVL_FACE_POS: Vec3 = (0.0, 0.0, Z_BOTTOM + 0.020)  # transducer face 20 mm above the skid line
DVL_POD: dict[str, float] = {"r": 0.065, "len": 0.100, "mass": 2.5}
PRESSURE_POS: Vec3 = (-0.545, -0.200, 0.0)  # z replaced by the rear upper cross-bar height at build time
PRESSURE_POD: dict[str, float] = {"r": 0.015, "len": 0.040, "mass": 0.2}

# Visual colours (RGBA); the arm keeps its own palette
VEHICLE_MATERIALS: dict[str, tuple[float, float, float, float]] = {
    "foam_yellow": (0.98, 0.78, 0.10, 1.0),
    "frame_white": (0.92, 0.92, 0.90, 1.0),
    "thruster_blue": (0.10, 0.32, 0.78, 1.0),
    "housing_alu": (0.62, 0.64, 0.66, 1.0),
    "ballast_dark": (0.20, 0.20, 0.22, 1.0),
    "camera_black": (0.05, 0.05, 0.05, 1.0),
    "dvl_orange": (0.90, 0.45, 0.10, 1.0),
}

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR: Path = REPO_ROOT / "rov_lab" / "assets_src" / "robots" / "i90_bravo7"
ROBOT_NAME: str = "i90_bravo7"

ROV_BASE: str = "rov_base"
CAMERA_LINK: str = "front_camera_link"
DVL_LINK: str = "dvl_link"
PRESSURE_LINK: str = "pressure_sensor_link"
ARM_BASE_LINK: str = "base_link"
JAW_BASE_LINK: str = "jaw_base_link"
VEHICLE_LINKS: tuple[str, ...] = (ROV_BASE, CAMERA_LINK, DVL_LINK, PRESSURE_LINK)

DEG: float = math.pi / 180.0
IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

# Planar arm lengths [m] (from the Bravo generator)
ARM_A: float = bravo.UPPER_ARM_LENGTH  # shoulder axis -> elbow axis
ARM_B: float = bravo.ELBOW_LENGTH + bravo.FOREARM_LENGTH  # elbow axis -> wrist pitch axis
ARM_TCP: float = bravo.WRIST_LENGTH + bravo.WRIST_ROLL_LENGTH + bravo.JAW_BASE_LENGTH + bravo.GRASP_POINT_FROM_PALM
ARM_TIP: float = bravo.WRIST_LENGTH + bravo.WRIST_ROLL_LENGTH + bravo.JAW_BASE_LENGTH + bravo.FINGER_LENGTH
GRASP_OFFSET_IN_JAW: Vec3 = (bravo.JAW_BASE_LENGTH + bravo.GRASP_POINT_FROM_PALM, 0.0, 0.0)
SHOULDER_X: float = ARM_FLANGE_X + bravo.BASE_HEIGHT + bravo.SHOULDER_HEIGHT  # shoulder pitch axis x (0.50 m)

# Reach requirements
SEABED_DEPTHS_BELOW_HULL_M: tuple[float, ...] = (0.30, 0.45, 0.60)
SEABED_TARGET_AHEAD_OF_BOW_M: float = 0.25
FORWARD_TARGET_AHEAD_OF_BOW_M: float = 0.50


# =============================================================================================
# Small helpers
# =============================================================================================
def mat_to_quat_xyzw(r: Mat3) -> tuple[float, float, float, float]:
    """Rotation matrix to a unit quaternion (x, y, z, w), w >= 0."""
    tr = r[0][0] + r[1][1] + r[2][2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (r[2][1] - r[1][2]) / s, (r[0][2] - r[2][0]) / s, (r[1][0] - r[0][1]) / s
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2.0
        w, x, y, z = (r[2][1] - r[1][2]) / s, 0.25 * s, (r[0][1] + r[1][0]) / s, (r[0][2] + r[2][0]) / s
    elif r[1][1] > r[2][2]:
        s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2.0
        w, x, y, z = (r[0][2] - r[2][0]) / s, (r[0][1] + r[1][0]) / s, 0.25 * s, (r[1][2] + r[2][1]) / s
    else:
        s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2.0
        w, x, y, z = (r[1][0] - r[0][1]) / s, (r[0][2] + r[2][0]) / s, (r[1][2] + r[2][1]) / s, 0.25 * s
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    n = math.sqrt(x * x + y * y + z * z + w * w)
    return (x / n, y / n, z / n, w / n)


def rnd(v: float, n: int = 6) -> float:
    """Round for the JSON output."""
    return round(float(v), n)


def rvec(v: tuple[float, ...] | list[float], n: int = 6) -> list[float]:
    """Round a vector for the JSON output."""
    return [rnd(x, n) for x in v]


def eig_sym3(m: Mat3) -> list[float]:
    """Eigenvalues of a symmetric 3x3 matrix (closed form, ascending)."""
    p1 = m[0][1] ** 2 + m[0][2] ** 2 + m[1][2] ** 2
    q = (m[0][0] + m[1][1] + m[2][2]) / 3.0
    if p1 < 1e-18:
        return sorted([m[0][0], m[1][1], m[2][2]])
    p2 = (m[0][0] - q) ** 2 + (m[1][1] - q) ** 2 + (m[2][2] - q) ** 2 + 2.0 * p1
    p = math.sqrt(p2 / 6.0)
    b = [[(m[i][j] - (q if i == j else 0.0)) / p for j in range(3)] for i in range(3)]
    det_b = (b[0][0] * (b[1][1] * b[2][2] - b[1][2] * b[2][1]) - b[0][1] * (b[1][0] * b[2][2] - b[1][2] * b[2][0])
             + b[0][2] * (b[1][0] * b[2][1] - b[1][1] * b[2][0]))
    r = max(-1.0, min(1.0, det_b / 2.0))
    phi = math.acos(r) / 3.0
    e1 = q + 2.0 * p * math.cos(phi)
    e3 = q + 2.0 * p * math.cos(phi + 2.0 * math.pi / 3.0)
    return sorted([e1, 3.0 * q - e1 - e3, e3])


# =============================================================================================
# Vehicle parts and mass properties
# =============================================================================================
@dataclass
class Part:
    """One vehicle primitive with its own mass and displaced volume (holes have negative mass/volume)."""

    name: str
    link: str
    prim: Primitive
    mass: float
    volume: float
    visual: bool = True


@dataclass
class TrimState:
    """Quantities solved by the trim iteration."""

    x_ref: float = 0.0  # allocation reference x (combined stowed CoM): thrusters symmetric about it
    z_ref: float = 0.0  # horizontal thruster height (combined stowed CoM height)
    foam_t: float = 0.12  # foam block thickness
    ballast_kg: float = 8.0  # total lead
    ballast_x: float = 0.0  # lead bar centre x


def thruster_dirs(alpha: float) -> dict[str, Vec3]:
    """Unit force directions of the four horizontal pods (vectored X: diagonal pairs parallel)."""
    c, s = math.cos(alpha), math.sin(alpha)
    return {"thr_fl": (c, -s, 0.0), "thr_fr": (c, s, 0.0), "thr_rl": (c, s, 0.0), "thr_rr": (c, -s, 0.0)}


def horizontal_alpha() -> float:
    """Vector angle from +x that reproduces the reverse/lateral thrust ratio (lateral is reverse-limited)."""
    return math.atan2(SPEC_THRUST_LAT_KGF, SPEC_THRUST_REV_KGF)


def foam_bottom(s: TrimState) -> float:
    """Underside of the foam block."""
    return Z_TOP - s.foam_t


def sensor_link_poses(s: TrimState) -> dict[str, tuple[Vec3, Vec3]]:
    """Fixed sensor frames {link: (xyz, rpy)} in rov_base."""
    p_z = foam_bottom(s) - 0.5 * CROSSBAR_SECTION
    return {
        CAMERA_LINK: (CAMERA_POS, (0.0, CAMERA_TILT_DEG * DEG, 0.0)),  # x = optical axis, z = image up
        DVL_LINK: (DVL_FACE_POS, (math.pi, 0.0, 0.0)),  # FRD: z down (beam axis), x forward
        PRESSURE_LINK: ((PRESSURE_POS[0], PRESSURE_POS[1], p_z), (0.0, 0.0, 0.0)),
    }


def build_parts(s: TrimState) -> list[Part]:
    """All vehicle primitives for one trim state (rov_base parts in rov_base, sensor pods in their link)."""
    parts: list[Part] = []
    fb = foam_bottom(s)

    def box(name: str, size: Vec3, xyz: Vec3, material: str, density: float, fraction: float = 1.0,
            link: str = ROV_BASE) -> None:
        vol = size[0] * size[1] * size[2] * fraction
        parts.append(Part(name, link, Primitive("box", size, xyz, material=material), density * vol, vol))

    # buoyancy foam with two vertical-thruster wells (holes only carry negative mass/volume, no geometry)
    fx0, fx1 = FOAM_X
    box("foam", (fx1 - fx0, SPEC_WIDTH_M, s.foam_t), (0.5 * (fx0 + fx1), 0.0, Z_TOP - 0.5 * s.foam_t),
        "foam_yellow", RHO_FOAM)
    for sy, tag in ((1.0, "port"), (-1.0, "stbd")):
        well = Primitive("cylinder", (WELL_R, s.foam_t), (s.x_ref, sy * V_THRUSTER_Y, Z_TOP - 0.5 * s.foam_t))
        v = -well.volume()
        parts.append(Part(f"foam_well_{tag}", ROV_BASE, well, RHO_FOAM * v, v, visual=False))
    # frame: side plates (with cut-outs), skids, cross bars
    plate_z0 = Z_BOTTOM + SKID_H
    for sy, tag in ((1.0, "port"), (-1.0, "stbd")):
        box(f"side_plate_{tag}", (SPEC_LENGTH_M, SIDE_PLATE_T, fb - plate_z0),
            (0.0, sy * (Y_SIDE - 0.5 * SIDE_PLATE_T), 0.5 * (plate_z0 + fb)), "frame_white", RHO_HDPE,
            SIDE_PLATE_SOLID_FRACTION)
        box(f"skid_{tag}", (SPEC_LENGTH_M, SKID_W, SKID_H), (0.0, sy * (Y_SIDE - 0.5 * SKID_W), Z_BOTTOM + 0.5 * SKID_H),
            "frame_white", RHO_HDPE)
    inner_w = SPEC_WIDTH_M - 2.0 * SIDE_PLATE_T
    for k, x in enumerate(UPPER_CROSSBAR_X):
        box(f"upper_crossbar_{k}", (CROSSBAR_SECTION, inner_w, CROSSBAR_SECTION),
            (x, 0.0, fb - 0.5 * CROSSBAR_SECTION), "frame_white", RHO_HDPE)
    skid_inner_w = SPEC_WIDTH_M - 2.0 * SKID_W
    lower_x = list(LOWER_CROSSBAR_X) + [ARM_FLANGE_X - ARM_BRACKET["t"] - 0.5 * CROSSBAR_SECTION]
    for k, x in enumerate(lower_x):
        box(f"lower_crossbar_{k}", (CROSSBAR_SECTION, skid_inner_w, SKID_H), (x, 0.0, Z_BOTTOM + 0.5 * SKID_H),
            "frame_white", RHO_HDPE)
    # arm bracket: vertical plate behind the Bravo flange, standing on the front lower cross bar
    bt, bw, bh = ARM_BRACKET["t"], ARM_BRACKET["w"], ARM_BRACKET["h"]
    box("arm_bracket", (bt, bw, bh), (ARM_FLANGE_X - 0.5 * bt, 0.0, ARM_AXIS_Z), "flange_silver", RHO_ALU)
    # pressure housings (main control chamber + telemetry chamber), axis along x
    for name, h in (("mcc_housing", HOUSING_MCC), ("tlc_housing", HOUSING_TLC)):
        prim = Primitive("cylinder", (h["r"], h["x1"] - h["x0"]), (0.5 * (h["x0"] + h["x1"]), 0.0, h["z"]),
                         bravo.ROT_Z_TO_X, "housing_alu")
        parts.append(Part(name, ROV_BASE, prim, h["mass"], prim.volume()))
    # thrusters
    for name, d in thruster_dirs(horizontal_alpha()).items():
        sx = 1.0 if name in ("thr_fl", "thr_fr") else -1.0
        sy = 1.0 if name in ("thr_fl", "thr_rl") else -1.0
        prim = Primitive("cylinder", (THRUSTER_R, H_THRUSTER_LEN),
                         (s.x_ref + sx * H_THRUSTER_DX, sy * H_THRUSTER_Y, s.z_ref),
                         (0.0, 0.5 * math.pi, math.atan2(d[1], d[0])), "thruster_blue")
        parts.append(Part(name, ROV_BASE, prim, THRUSTER_MASS_KG, THRUSTER_DISPLACED_M3))
    for sy, name in ((1.0, "thr_vl"), (-1.0, "thr_vr")):
        prim = Primitive("cylinder", (THRUSTER_R, V_THRUSTER_LEN), (s.x_ref, sy * V_THRUSTER_Y, Z_TOP - 0.5 * s.foam_t),
                         material="thruster_blue")
        parts.append(Part(name, ROV_BASE, prim, THRUSTER_MASS_KG, THRUSTER_DISPLACED_M3))
    # lead trim ballast on the skids
    bar_kg = 0.5 * s.ballast_kg
    bar_len = bar_kg / (RHO_LEAD * BALLAST_W * BALLAST_H)
    for sy, tag in ((1.0, "port"), (-1.0, "stbd")):
        box(f"ballast_{tag}", (bar_len, BALLAST_W, BALLAST_H),
            (s.ballast_x, sy * BALLAST_Y, Z_BOTTOM + SKID_H + 0.5 * BALLAST_H), "ballast_dark", RHO_LEAD)
    # sensor pods (in their own link frames)
    cam = cyl_x(CAMERA_POD["r"], -CAMERA_POD["len"], 0.0, "camera_black")
    parts.append(Part("camera_pod", CAMERA_LINK, cam, CAMERA_POD["mass"], cam.volume()))
    dvl = cyl_z(DVL_POD["r"], -DVL_POD["len"], 0.0, "dvl_orange")  # link z points down: pod extends up from the face
    parts.append(Part("dvl_pod", DVL_LINK, dvl, DVL_POD["mass"], dvl.volume()))
    prs = cyl_x(PRESSURE_POD["r"], 0.0, PRESSURE_POD["len"], "housing_alu")  # sensing port (aft face) at the origin
    parts.append(Part("pressure_pod", PRESSURE_LINK, prs, PRESSURE_POD["mass"], prs.volume()))
    return parts


def link_to_base(s: TrimState) -> dict[str, tuple[Vec3, Mat3]]:
    """Vehicle link frames {link: (pos, R)} in rov_base."""
    out: dict[str, tuple[Vec3, Mat3]] = {ROV_BASE: ((0.0, 0.0, 0.0), IDENTITY)}
    for link, (xyz, rpy) in sensor_link_poses(s).items():
        out[link] = (xyz, rpy_to_mat(rpy))
    return out


def mass_props(items: list[tuple[float, Primitive]]) -> tuple[float, Vec3, Mat3]:
    """Total mass, CoM and inertia about the CoM (in the items' frame) of uniformly filled primitives."""
    m_tot = sum(m for m, _ in items)
    com = (0.0, 0.0, 0.0)
    for m, p in items:
        com = vadd(com, vscale(p.xyz, m / m_tot))
    inertia: Mat3 = [[0.0] * 3 for _ in range(3)]
    for m, p in items:
        principal = p.local_inertia_per_kg()
        rot = rpy_to_mat(p.rpy)
        local: Mat3 = [[principal[i] * m if i == j else 0.0 for j in range(3)] for i in range(3)]
        rotated = mat_mul(mat_mul(rot, local), mat_t(rot))
        d = vsub(p.xyz, com)
        dd = dot(d, d)
        for i in range(3):
            for j in range(3):
                inertia[i][j] += rotated[i][j] + m * ((dd if i == j else 0.0) - d[i] * d[j])
    inertia = [[0.0 if abs(v) < 1e-12 else v for v in row] for row in inertia]
    return m_tot, com, inertia


def transform_inertia(inertia: Mat3, rot: Mat3) -> Mat3:
    """Rotate an inertia tensor into the parent frame."""
    return mat_mul(mat_mul(rot, inertia), mat_t(rot))


def system_props(
    bodies: list[tuple[float, Vec3, Mat3]]
) -> tuple[float, Vec3, Mat3]:
    """Combine (mass, com, inertia_about_com) bodies expressed in one frame."""
    m_tot = sum(b[0] for b in bodies)
    com = (0.0, 0.0, 0.0)
    for m, c, _ in bodies:
        com = vadd(com, vscale(c, m / m_tot))
    inertia: Mat3 = [[0.0] * 3 for _ in range(3)]
    for m, c, i_c in bodies:
        d = vsub(c, com)
        dd = dot(d, d)
        for i in range(3):
            for j in range(3):
                inertia[i][j] += i_c[i][j] + m * ((dd if i == j else 0.0) - d[i] * d[j])
    return m_tot, com, inertia


# =============================================================================================
# Arm: kinematics in rov_base
# =============================================================================================
ARM_LINKS: list[LinkSpec] = bravo.build_links()
ARM_JOINTS: list[bravo.JointSpec] = bravo.build_joints()
ARM_VOLUME_SCALE: float = bravo.compute_mass_properties(ARM_LINKS)
ARM_LINK_BY_NAME: dict[str, LinkSpec] = {link.name: link for link in ARM_LINKS}
MOUNT_POS: Vec3 = (ARM_FLANGE_X, 0.0, ARM_AXIS_Z)
MOUNT_ROT: Mat3 = rpy_to_mat(ARM_MOUNT_RPY)


def arm_poses(q: dict[str, float]) -> dict[str, tuple[Vec3, Mat3]]:
    """Arm link frames {link: (pos, R)} in rov_base for joint positions q (Bravo forward kinematics + mount)."""
    poses_b, _ = bravo.forward_kinematics(ARM_JOINTS, q)
    out: dict[str, tuple[Vec3, Mat3]] = {}
    for name, (p, r) in poses_b.items():
        out[name] = (vadd(MOUNT_POS, mat_vec(MOUNT_ROT, p)), mat_mul(MOUNT_ROT, r))
    return out


def point_in(pose: tuple[Vec3, Mat3], offset: Vec3) -> Vec3:
    """Point given in a link frame -> rov_base."""
    return vadd(pose[0], mat_vec(pose[1], offset))


def arm_mass_items(q: dict[str, float]) -> tuple[list[tuple[float, Vec3, Mat3]], float, Vec3]:
    """Arm bodies (mass, com, inertia) in rov_base, total displaced volume and volume centroid."""
    poses = arm_poses(q)
    bodies: list[tuple[float, Vec3, Mat3]] = []
    v_tot = 0.0
    v_mom = (0.0, 0.0, 0.0)
    for link in ARM_LINKS:
        p, r = poses[link.name]
        com = point_in((p, r), link.com)
        bodies.append((link.mass, com, transform_inertia(link.inertia, r)))
        v_tot += link.displaced_volume
        v_mom = vadd(v_mom, vscale(com, link.displaced_volume))  # CoB = CoM per arm link
    return bodies, v_tot, vscale(v_mom, 1.0 / v_tot)


def tcp(q: dict[str, float]) -> tuple[Vec3, Vec3, Vec3]:
    """(grasp point, jaw tip, jaw axis) in rov_base."""
    jaw = arm_poses(q)[JAW_BASE_LINK]
    return (point_in(jaw, GRASP_OFFSET_IN_JAW), point_in(jaw, (bravo.JAW_BASE_LENGTH + bravo.FINGER_LENGTH, 0.0, 0.0)),
            mat_vec(jaw[1], (1.0, 0.0, 0.0)))


def planar_ik(target_xz: tuple[float, float], phi3: float, elbow_up: bool) -> dict[str, float] | None:
    """Joint angles (base yaw 0, arm in the vertical centre plane) that put the TCP at target_xz.

    Planar angles are measured from straight down toward forward (+x): upper arm phi1 = q_shoulder,
    forearm phi2 = phi1 + q_elbow, jaws phi3 = phi2 + q_wrist. ``elbow_up`` puts the elbow above the
    shoulder-wrist line (q_elbow < 0). Returns None when the target is out of reach.
    """
    sx, sz = SHOULDER_X, ARM_AXIS_Z
    wx = target_xz[0] - ARM_TCP * math.sin(phi3)
    wz = target_xz[1] + ARM_TCP * math.cos(phi3)
    dx, dz = wx - sx, wz - sz
    r2 = dx * dx + dz * dz
    c_e = (r2 - ARM_A * ARM_A - ARM_B * ARM_B) / (2.0 * ARM_A * ARM_B)
    if abs(c_e) > 1.0:
        return None
    q_e = -math.acos(c_e) if elbow_up else math.acos(c_e)
    psi = math.atan2(dx, -dz)
    phi1 = psi - math.atan2(ARM_B * math.sin(q_e), ARM_A + ARM_B * math.cos(q_e))
    phi1 = math.atan2(math.sin(phi1), math.cos(phi1))
    q_w = phi3 - phi1 - q_e
    q_w = math.atan2(math.sin(q_w), math.cos(q_w))
    return {"base_yaw_joint": 0.0, "shoulder_pitch_joint": phi1, "elbow_pitch_joint": q_e,
            "forearm_roll_joint": 0.0, "wrist_pitch_joint": q_w, "ee_roll_joint": 0.0}


def within_limits(q: dict[str, float], margin: float = 0.0) -> bool:
    """All joint values inside the Bravo limits (minus a margin)."""
    for name, v in q.items():
        lim = bravo.JOINT_LIMITS[name]
        if not (lim.lower + margin <= v <= lim.upper - margin):
            return False
    return True


# =============================================================================================
# Geometric overlap check (sampled points inside primitives)
# =============================================================================================
@dataclass
class WorldPrim:
    """A primitive placed in rov_base."""

    link: str
    name: str
    prim: Primitive
    pos: Vec3
    rot: Mat3
    radius: float


def bounding_radius(p: Primitive) -> float:
    """Bounding-sphere radius about the primitive centre."""
    if p.kind == "cylinder":
        r, h = p.size
        return math.sqrt(r * r + 0.25 * h * h)
    return 0.5 * math.sqrt(sum(x * x for x in p.size))


def sample_local(p: Primitive, margin: float) -> list[Vec3]:
    """Grid of points strictly inside a primitive (inset by margin), in primitive axes."""
    pts: list[Vec3] = []
    n = 7
    if p.kind == "cylinder":
        r, h = p.size
        r_in, h_in = max(r - margin, 0.0), max(0.5 * h - margin, 0.0)
        for iz in range(n):
            z = -h_in + 2.0 * h_in * iz / (n - 1)
            pts.append((0.0, 0.0, z))
            for fr in (0.34, 0.67, 1.0):
                for k in range(16):
                    a = 2.0 * math.pi * k / 16
                    pts.append((fr * r_in * math.cos(a), fr * r_in * math.sin(a), z))
        return pts
    hx, hy, hz = (max(0.5 * s - margin, 0.0) for s in p.size)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                pts.append((-hx + 2 * hx * i / (n - 1), -hy + 2 * hy * j / (n - 1), -hz + 2 * hz * k / (n - 1)))
    return pts


def contains(wp: WorldPrim, pt: Vec3, inflate: float) -> bool:
    """Is a rov_base point inside the (inflated) world primitive?"""
    d = vsub(pt, wp.pos)
    lx = wp.rot[0][0] * d[0] + wp.rot[1][0] * d[1] + wp.rot[2][0] * d[2]
    ly = wp.rot[0][1] * d[0] + wp.rot[1][1] * d[1] + wp.rot[2][1] * d[2]
    lz = wp.rot[0][2] * d[0] + wp.rot[1][2] * d[1] + wp.rot[2][2] * d[2]
    if wp.prim.kind == "cylinder":
        r, h = wp.prim.size
        return abs(lz) <= 0.5 * h + inflate and lx * lx + ly * ly <= (r + inflate) ** 2
    sx, sy, sz = wp.prim.size
    return abs(lx) <= 0.5 * sx + inflate and abs(ly) <= 0.5 * sy + inflate and abs(lz) <= 0.5 * sz + inflate


def world_prims(link_poses: dict[str, tuple[Vec3, Mat3]], link_prims: dict[str, list[Primitive]]) -> list[WorldPrim]:
    """Place every collision primitive in rov_base."""
    out: list[WorldPrim] = []
    for link, prims in link_prims.items():
        p_l, r_l = link_poses[link]
        for k, p in enumerate(prims):
            out.append(WorldPrim(link, f"{link}[{k}]", p, point_in((p_l, r_l), p.xyz), mat_mul(r_l, rpy_to_mat(p.rpy)),
                                 bounding_radius(p)))
    return out


def overlaps(
    link_poses: dict[str, tuple[Vec3, Mat3]],
    link_prims: dict[str, list[Primitive]],
    adjacent: set[frozenset[str]],
    inflate: float = 0.0,
    margin: float = 0.001,
    only_links: set[str] | None = None,
) -> dict[str, int]:
    """Count sampled points of one primitive inside another, for every non-adjacent link pair.

    Returns {"linkA|linkB": n_points}; empty means no overlap (with ``inflate`` = clearance tested).
    ``only_links`` restricts the check to pairs involving at least one of these links.
    """
    wps = world_prims(link_poses, link_prims)
    samples = {id(w): [point_in((w.pos, w.rot), q) for q in sample_local(w.prim, margin)] for w in wps}
    hits: dict[str, int] = {}
    for i, a in enumerate(wps):
        for b in wps[i + 1:]:
            if a.link == b.link or frozenset((a.link, b.link)) in adjacent:
                continue
            if only_links is not None and a.link not in only_links and b.link not in only_links:
                continue
            gap = math.sqrt(dot(vsub(a.pos, b.pos), vsub(a.pos, b.pos)))
            if gap > a.radius + b.radius + inflate:
                continue
            n = sum(contains(b, pt, inflate) for pt in samples[id(a)]) + sum(contains(a, pt, inflate)
                                                                          for pt in samples[id(b)])
            if n:
                key = "|".join(sorted((a.link, b.link)))
                hits[key] = hits.get(key, 0) + n
    return hits


# =============================================================================================
# Hydrodynamic coefficients (Fossen diagonal model, same form/sign convention as HydroParamsCfg)
# =============================================================================================
BLUEROV_LIN_OVER_QUAD_SURGE: float = 4.03 / 18.18
"""HydroParamsCfg BlueROV defaults: linear/quadratic surge damping ratio (crossover speed, m/s)."""


def lamb_integrals(a: float, b: float, c: float, n: int = 20000) -> tuple[float, float, float]:
    """Lamb's ellipsoid coefficients alpha0, beta0, gamma0 (semi-axes a, b, c along x, y, z)."""

    def integral(k2: float) -> float:
        h = 1.0 / n
        total = 0.0
        for i in range(n + 1):
            t = i * h
            if t >= 1.0:
                f = 0.0
            else:
                lam = t / (1.0 - t)
                f = 1.0 / ((k2 + lam) * math.sqrt((a * a + lam) * (b * b + lam) * (c * c + lam))) / (1.0 - t) ** 2
            w = 1.0 if i in (0, n) else (4.0 if i % 2 else 2.0)
            total += w * f
        return a * b * c * total * h / 3.0

    return integral(a * a), integral(b * b), integral(c * c)


def hydro_coefficients(volume: float, co_bm: float) -> tuple[dict, dict]:
    """I90 hydrodynamic coefficients and the derivation table.

    surge   : 60 kgf forward thrust balances drag at the 3 kn catalogue top speed,
              F = D_lin * U + D_quad * U^2, with D_lin / D_quad kept at the BlueROV ratio (0.222 m/s).
    sway/heave: surge coefficients scaled by projected area (same drag coefficient):
              A_x = W*H, A_y = L*H, A_z = L*W.
    rotation: strip/plate integration over the envelope faces with the per-area surge coefficients
              (quadratic ~ face * arm^4 / 32, linear ~ face * arm^3 / 12).
    added mass: Lamb's coefficients of the ellipsoid inscribed in the L x W x H envelope.
    """
    L, W, H = SPEC_LENGTH_M, SPEC_WIDTH_M, SPEC_HEIGHT_M
    u_top = SPEC_SPEED_KN * KNOT_M_S
    f_fwd = SPEC_THRUST_FWD_KGF * KGF_N
    v_c = BLUEROV_LIN_OVER_QUAD_SURGE
    dq_x = f_fwd / (u_top * u_top + v_c * u_top)
    dl_x = v_c * dq_x
    a_x, a_y, a_z = W * H, L * H, L * W
    q_area, c_area = dq_x / a_x, dl_x / a_x  # per unit projected area
    dq = [dq_x, dq_x * a_y / a_x, dq_x * a_z / a_x]
    dl = [dl_x, dl_x * a_y / a_x, dl_x * a_z / a_x]
    # strips: faces normal to y (L*H) spread along x, normal to x (W*H) along y, normal to z (L*W) along x/y
    roll_q = q_area * (L * W ** 4 + L * H ** 4) / 32.0
    roll_l = c_area * (L * W ** 3 + L * H ** 3) / 12.0
    pitch_q = q_area * (W * L ** 4 + W * H ** 4) / 32.0
    pitch_l = c_area * (W * L ** 3 + W * H ** 3) / 12.0
    yaw_q = q_area * (H * L ** 4 + H * W ** 4) / 32.0
    yaw_l = c_area * (H * L ** 3 + H * W ** 3) / 12.0
    dq += [roll_q, pitch_q, yaw_q]
    dl += [roll_l, pitch_l, yaw_l]
    # added mass, inscribed ellipsoid
    a, b, c = 0.5 * L, 0.5 * W, 0.5 * H
    al, be, ga = lamb_integrals(a, b, c)
    m_ell = RHO_WATER * 4.0 / 3.0 * math.pi * a * b * c
    added = [al / (2.0 - al) * m_ell, be / (2.0 - be) * m_ell, ga / (2.0 - ga) * m_ell]
    added.append(0.2 * m_ell * (b * b - c * c) ** 2 * (ga - be) / (2.0 * (b * b - c * c) + (b * b + c * c) * (be - ga)))
    added.append(0.2 * m_ell * (c * c - a * a) ** 2 * (al - ga) / (2.0 * (c * c - a * a) + (c * c + a * a) * (ga - al)))
    added.append(0.2 * m_ell * (a * a - b * b) ** 2 * (be - al) / (2.0 * (a * a - b * b) + (a * a + b * b) * (al - be)))
    params = {
        "volume": rnd(volume, 7),
        "co_bm": rnd(co_bm, 5),
        "rho": RHO_WATER,
        "g": GRAVITY,
        "added_mass": tuple(rnd(x, 3) for x in added),
        "linear_damping": tuple(rnd(x, 3) for x in dl),
        "quadratic_damping": tuple(rnd(x, 3) for x in dq),
    }
    table = {
        "top_speed_m_s": u_top, "forward_thrust_n": f_fwd, "lin_over_quad_m_s": v_c,
        "projected_area_m2": {"x": a_x, "y": a_y, "z": a_z},
        "drag_coefficient_quadratic_part": 2.0 * dq_x / (RHO_WATER * a_x),
        "per_area_quadratic_n_s2_m4": q_area, "per_area_linear_n_s_m3": c_area,
        "lamb_alpha_beta_gamma": [al, be, ga], "lamb_sum_check": al + be + ga,
        "inscribed_ellipsoid_mass_kg": m_ell,
    }
    return params, table


def steady_speed(d_lin: float, d_quad: float, force: float) -> float:
    """Speed where D_lin*v + D_quad*v^2 balances a constant force."""
    return (-d_lin + math.sqrt(d_lin * d_lin + 4.0 * d_quad * force)) / (2.0 * d_quad)


# =============================================================================================
# Thrusters (fit to the catalogue axis thrusts)
# =============================================================================================
@dataclass
class Thruster:
    """One thruster: position [m] in rov_base, unit force direction for forward spin, limits [N] (>= 0)."""

    name: str
    pos: Vec3
    direction: Vec3
    max_forward_n: float
    max_reverse_n: float


def thruster_fit() -> dict[str, float]:
    """Common DWT6537 rating from the horizontal specs + vertical installation factor.

    All four horizontals contribute equally to forward and reverse surge, so forward/reverse = 60/30 fixes the
    reverse/forward ratio of the propeller (kappa = 0.5). Lateral thrust with zero surge/yaw uses all four pods
    at the reverse limit: F_lat = 4 * T_rev * sin(alpha), F_rev = 4 * T_rev * cos(alpha) -> tan(alpha) =
    33.5 / 30. The same bollard rating in the two vertical pods would give 45 kgf up / 22.5 kgf down, but the
    catalogue lists 31 kgf: we attribute the gap to an installation (thrust-deduction) factor of the pods in the
    foam wells, eta_v = 31 / 45, and read the single catalogue value as the strong (up) direction.
    """
    alpha = horizontal_alpha()
    t_fwd = SPEC_THRUST_FWD_KGF * KGF_N / (4.0 * math.cos(alpha))
    t_rev = SPEC_THRUST_REV_KGF * KGF_N / (4.0 * math.cos(alpha))
    eta_v = SPEC_THRUST_VERT_KGF * KGF_N / (2.0 * t_fwd)
    return {"alpha_rad": alpha, "t_fwd_n": t_fwd, "t_rev_n": t_rev, "kappa": t_rev / t_fwd, "eta_vertical": eta_v}


def build_thrusters(parts: list[Part]) -> list[Thruster]:
    """Thruster metadata from the placed pods."""
    fit = thruster_fit()
    dirs = thruster_dirs(fit["alpha_rad"])
    by = {p.name: p for p in parts}
    out: list[Thruster] = []
    for name in ("thr_fl", "thr_fr", "thr_rl", "thr_rr"):
        out.append(Thruster(name, by[name].prim.xyz, dirs[name], fit["t_fwd_n"], fit["t_rev_n"]))
    for name in ("thr_vl", "thr_vr"):
        out.append(Thruster(name, by[name].prim.xyz, (0.0, 0.0, 1.0), fit["eta_vertical"] * fit["t_fwd_n"],
                            fit["eta_vertical"] * fit["t_rev_n"]))
    return out


def axis_capability(thrusters: list[Thruster], ref: Vec3) -> dict[str, float]:
    """Max force per axis direction with the other actuated components (Fx, Fy, Fz, Mz about ref) zero.

    Closed form for this symmetric layout (verified with an LP in test_i90_bravo7.py): surge uses all four
    horizontals at one limit, sway all four at the reverse limit (diagonal pairs opposite), heave both verticals,
    yaw all four at the reverse limit.
    """
    h = [t for t in thrusters if t.direction[2] == 0.0]
    v = [t for t in thrusters if t.direction[2] != 0.0]
    t_f, t_r = h[0].max_forward_n, h[0].max_reverse_n
    c, s = abs(h[0].direction[0]), abs(h[0].direction[1])
    arm = 0.0
    for t in h:  # |moment arm| of each pod about the reference for its own direction
        r = vsub(t.pos, ref)
        arm += abs(r[0] * t.direction[1] - r[1] * t.direction[0])
    return {
        "+x": 4 * t_f * c, "-x": 4 * t_r * c, "+y": 4 * t_r * s, "-y": 4 * t_r * s,
        "+z": sum(t.max_forward_n for t in v), "-z": sum(t.max_reverse_n for t in v),
        "yaw": t_r * arm,
    }


# =============================================================================================
# Poses
# =============================================================================================
FINGER_OPEN: float = 0.03
FINGER_STOW: float = 0.005  # 10 mm opening (fingers not touching each other)


def stow_pose() -> dict[str, float]:
    """Arm folded flat in the horizontal plane ahead of the bow (base yaw +90 deg).

    In that plane, angles run from +y (port) toward +x (forward): the upper arm points 23 deg forward of port, the
    forearm folds back 137.5 deg toward the centre line and the jaws point to starboard, slightly aft.
    """
    return {"base_yaw_joint": 0.5 * math.pi, "shoulder_pitch_joint": 0.40, "elbow_pitch_joint": 2.40,
            "forearm_roll_joint": 0.0, "wrist_pitch_joint": 0.74, "ee_roll_joint": 0.0,
            "finger_left_joint": FINGER_STOW, "finger_right_joint": FINGER_STOW}


HOME_TCP_AHEAD_OF_BOW_M: float = 0.40
HOME_TCP_Z: float = -0.150  # 9.5 cm above the skid line, ~12 deg below the camera axis
HOME_JAW_ANGLE_FROM_DOWN_DEG: float = 75.0  # jaws 15 deg below horizontal; keeps the wrist >10 deg off its limit


# =============================================================================================
# Trim solve
# =============================================================================================
def vehicle_bodies(s: TrimState, parts: list[Part]) -> tuple[list[tuple[float, Vec3, Mat3]], float, Vec3]:
    """Vehicle link bodies in rov_base plus the vehicle displaced volume and its centroid."""
    l2b = link_to_base(s)
    bodies: list[tuple[float, Vec3, Mat3]] = []
    for link in VEHICLE_LINKS:
        items = [(p.mass, p.prim) for p in parts if p.link == link]
        m, com, inertia = mass_props(items)
        pos, rot = l2b[link]
        bodies.append((m, point_in((pos, rot), com), transform_inertia(inertia, rot)))
    v_tot = sum(p.volume for p in parts)
    v_mom = (0.0, 0.0, 0.0)
    for p in parts:
        pos, rot = l2b[p.link]
        v_mom = vadd(v_mom, vscale(point_in((pos, rot), p.prim.xyz), p.volume))
    return bodies, v_tot, vscale(v_mom, 1.0 / v_tot)


def solve_trim(q_stow: dict[str, float]) -> tuple[TrimState, list[Part]]:
    """Solve foam thickness, ballast mass/position and thruster reference for mass, buoyancy and level trim."""
    arm_bodies, v_arm, cob_arm = arm_mass_items(q_stow)
    m_arm = sum(b[0] for b in arm_bodies)
    v_target = (SPEC_MASS_AIR_KG + m_arm + TARGET_NET_BUOYANCY_KG) / RHO_WATER - v_arm
    fx0, fx1 = FOAM_X
    a_eff = (fx1 - fx0) * SPEC_WIDTH_M - 2.0 * math.pi * WELL_R ** 2  # foam area net of the wells
    s = TrimState()
    for _ in range(200):
        parts = build_parts(s)
        others = [p for p in parts if not p.name.startswith(("foam", "ballast"))]
        m_o = sum(p.mass for p in others)
        v_o = sum(p.volume for p in others)
        # m_o + RHO_FOAM*a_eff*t + m_b = M ; v_o + a_eff*t + m_b/RHO_LEAD = V
        a11, a12, b1 = RHO_FOAM * a_eff, 1.0, SPEC_MASS_AIR_KG - m_o
        a21, a22, b2 = a_eff, 1.0 / RHO_LEAD, v_target - v_o
        det = a11 * a22 - a12 * a21
        t_new = (b1 * a22 - a12 * b2) / det
        mb_new = (a11 * b2 - a21 * b1) / det
        # level trim: combined CoM x == combined CoB x, solved for the ballast x (linear in it)
        s_tmp = TrimState(s.x_ref, s.z_ref, t_new, mb_new, 0.0)
        parts0 = build_parts(s_tmp)
        bodies0, v_veh0, cob0 = vehicle_bodies(s_tmp, parts0)
        m_tot = sum(b[0] for b in bodies0) + m_arm
        v_tot = v_veh0 + v_arm
        mx0 = sum(b[0] * b[1][0] for b in bodies0) + sum(b[0] * b[1][0] for b in arm_bodies)
        vx0 = v_veh0 * cob0[0] + v_arm * cob_arm[0]
        v_b = mb_new / RHO_LEAD
        xb_new = (vx0 / v_tot - mx0 / m_tot) / (mb_new / m_tot - v_b / v_tot)
        s_new = TrimState(s.x_ref, s.z_ref, t_new, mb_new, xb_new)
        bodies1, _, _ = vehicle_bodies(s_new, build_parts(s_new))
        _, com_sys, _ = system_props(bodies1 + arm_bodies)
        s_new.x_ref, s_new.z_ref = com_sys[0], com_sys[2]
        delta = max(abs(s_new.x_ref - s.x_ref), abs(s_new.z_ref - s.z_ref), abs(s_new.foam_t - s.foam_t),
                    abs(s_new.ballast_kg - s.ballast_kg), abs(s_new.ballast_x - s.ballast_x))
        s = s_new
        if delta < 1e-10:
            break
    return s, build_parts(s)


# =============================================================================================
# URDF writer
# =============================================================================================
@dataclass
class OutLink:
    """URDF link."""

    name: str
    mass: float
    com: Vec3
    inertia: Mat3
    primitives: list[Primitive]
    comment: str


@dataclass
class OutJoint:
    """URDF joint (fixed joints may carry an rpy)."""

    name: str
    kind: str
    parent: str
    child: str
    xyz: Vec3
    rpy: Vec3 = (0.0, 0.0, 0.0)
    axis: Vec3 = (0.0, 0.0, 1.0)


def build_urdf(links: list[OutLink], joints: list[OutJoint]) -> ET.Element:
    """Build the URDF XML tree (arm joint limits from the Bravo generator)."""
    robot = ET.Element("robot", name=ROBOT_NAME)
    robot.append(ET.Comment(
        " 'I90-class' ROV + 'Bravo 7-class' arm, one floating articulation, built from public catalogue specs only. "
        "Generated by rov_lab/scripts/tools/make_i90_bravo7_urdf.py - edit the generator, not this file. "
    ))
    for name, rgba in {**bravo.MATERIALS, **VEHICLE_MATERIALS}.items():
        mat = ET.SubElement(robot, "material", name=name)
        ET.SubElement(mat, "color", rgba=fvec(rgba))
    for link in links:
        el = ET.SubElement(robot, "link", name=link.name)
        el.append(ET.Comment(link.comment))
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
        ET.SubElement(el, "origin", xyz=fvec(j.xyz), rpy=fvec(j.rpy))
        ET.SubElement(el, "parent", link=j.parent)
        ET.SubElement(el, "child", link=j.child)
        if j.kind != "fixed":
            lim = bravo.JOINT_LIMITS[j.name]
            ET.SubElement(el, "axis", xyz=fvec(j.axis))
            ET.SubElement(el, "limit", lower=fmt(lim.lower), upper=fmt(lim.upper),
                          effort=fmt(lim.effort), velocity=fmt(lim.velocity))
            ET.SubElement(el, "dynamics", damping=fmt(lim.damping), friction=fmt(lim.friction))
    ET.indent(robot, space="  ")
    return robot


# =============================================================================================
# Model assembly (shared with the test)
# =============================================================================================
def link_primitives(parts: list[Part]) -> dict[str, list[Primitive]]:
    """Visual/collision primitives per link (vehicle + arm)."""
    out: dict[str, list[Primitive]] = {link: [p.prim for p in parts if p.link == link and p.visual]
                                       for link in VEHICLE_LINKS}
    for link in ARM_LINKS:
        out[link.name] = list(link.primitives)
    return out


def adjacent_pairs() -> set[frozenset[str]]:
    """Parent/child link pairs (joint-connected; PhysX never collides them)."""
    pairs = {frozenset((ROV_BASE, child)) for child in (CAMERA_LINK, DVL_LINK, PRESSURE_LINK, ARM_BASE_LINK)}
    pairs |= {frozenset((j.parent, j.child)) for j in ARM_JOINTS}
    return pairs


def full_link_poses(s: TrimState, q: dict[str, float]) -> dict[str, tuple[Vec3, Mat3]]:
    """All link frames in rov_base."""
    return {**link_to_base(s), **arm_poses(q)}


def solve_reach(
    s: TrimState,
    link_prims: dict[str, list[Primitive]],
    target: tuple[float, float],
    phi3_list: list[float],
    limit_margin: float = 0.0,
) -> dict:
    """First feasible IK (joint limits minus a margin, no overlap) over the preferred jaw angles, elbow-up first."""
    adj = adjacent_pairs()
    tried = 0
    for phi3 in phi3_list:
        for elbow_up in (True, False):
            q = planar_ik(target, phi3, elbow_up)
            tried += 1
            if q is None or not within_limits(q, limit_margin):
                continue
            q_full = {**q, "finger_left_joint": FINGER_OPEN, "finger_right_joint": FINGER_OPEN}
            hits = overlaps(full_link_poses(s, q_full), link_prims, adj)
            if hits:
                continue
            grasp, tip, axis = tcp(q_full)
            elbow = arm_poses(q_full)["elbow_link"][0]
            return {"reachable": True, "q": {k: rnd(v, 5) for k, v in q_full.items()},
                    "jaw_angle_from_down_deg": rnd(phi3 / DEG, 2), "elbow_up": elbow_up,
                    "tcp_m": rvec(grasp, 5), "tcp_err_m": rnd(math.dist(grasp, (target[0], 0.0, target[1])), 7),
                    "jaw_tip_m": rvec(tip, 5), "jaw_axis": rvec(axis, 4), "elbow_axis_m": rvec(elbow, 4),
                    "candidates_tried": tried}
    return {"reachable": False, "candidates_tried": tried}


def build_model() -> dict:
    """Assemble the full design (parts, trim, links, joints, poses, checks, hydro, thrusters)."""
    q_stow = stow_pose()
    s, parts = solve_trim(q_stow)
    l2b = link_to_base(s)
    link_prims = link_primitives(parts)
    adj = adjacent_pairs()

    # --- per-link mass properties
    vehicle_links: dict[str, tuple[float, Vec3, Mat3]] = {}
    for link in VEHICLE_LINKS:
        vehicle_links[link] = mass_props([(p.mass, p.prim) for p in parts if p.link == link])
    veh_bodies, v_veh, cob_veh = vehicle_bodies(s, parts)
    m_veh, com_veh, i_veh = system_props(veh_bodies)

    # --- poses (home from IK so that the TCP sits in the camera view ahead of the bow)
    home_target = (X_BOW + HOME_TCP_AHEAD_OF_BOW_M, HOME_TCP_Z)
    home_ik = solve_reach(s, link_prims, home_target, [HOME_JAW_ANGLE_FROM_DOWN_DEG * DEG], limit_margin=10.0 * DEG)
    if not home_ik["reachable"]:
        raise ValueError("home pose not reachable")
    q_home = dict(home_ik["q"])

    def system_at(q: dict[str, float]) -> dict:
        arm_bodies, v_arm, cob_arm = arm_mass_items(q)
        m_sys, com_sys, i_sys = system_props(veh_bodies + arm_bodies)
        v_sys = v_veh + v_arm
        cob_sys = vscale(vadd(vscale(cob_veh, v_veh), vscale(cob_arm, v_arm)), 1.0 / v_sys)
        b_n = RHO_WATER * v_sys * GRAVITY
        return {"mass_kg": m_sys, "com_m": com_sys, "inertia_about_com": i_sys, "displaced_volume_m3": v_sys,
                "cob_m": cob_sys, "net_buoyancy_kg": RHO_WATER * v_sys - m_sys,
                "cob_minus_com_m": vsub(cob_sys, com_sys),
                "righting_stiffness_nm_per_rad": b_n * (cob_sys[2] - com_sys[2]),
                "arm_mass_kg": sum(b[0] for b in arm_bodies), "arm_displaced_volume_m3": v_arm}

    sys_stow = system_at(q_stow)
    sys_home = system_at(q_home)

    # --- reach requirements
    seabed: dict[str, dict] = {}
    phi_seabed = [k * DEG for k in (0, 10, -10, 20, -20, 30, -30, 45, -45, 60)]
    for depth in SEABED_DEPTHS_BELOW_HULL_M:
        target = (X_BOW + SEABED_TARGET_AHEAD_OF_BOW_M, Z_BOTTOM - depth)
        seabed[f"{depth:.2f}"] = {"target_tcp_m": rvec((target[0], 0.0, target[1]), 4),
                                  **solve_reach(s, link_prims, target, phi_seabed)}
    fwd_target = (X_BOW + FORWARD_TARGET_AHEAD_OF_BOW_M, Z_BOTTOM)
    forward = {"target_tcp_m": rvec((fwd_target[0], 0.0, fwd_target[1]), 4),
               **solve_reach(s, link_prims, fwd_target, [k * DEG for k in (90, 80, 70, 60, 100, 50, 45)])}
    # seabed reach envelope along the centre line (jaws within +-30 deg of vertical), informative
    envelope: dict[str, list[float]] = {}
    for depth in SEABED_DEPTHS_BELOW_HULL_M:
        xs = []
        for i in range(0, 131):  # scan from 0.30 m behind to 1.0 m ahead of the bow
            x = X_BOW - 0.30 + 0.01 * i
            if solve_reach(s, link_prims, (x, Z_BOTTOM - depth), [k * DEG for k in (0, 15, -15, 30, -30)])["reachable"]:
                xs.append(x)
        envelope[f"{depth:.2f}"] = [rnd(min(xs) - X_BOW, 3), rnd(max(xs) - X_BOW, 3)] if xs else []

    # --- overlap checks at home and stow (arm self, arm vs hull, inflated = clearance)
    pose_checks: dict[str, dict] = {}
    for name, q in (("home", q_home), ("stow", q_stow)):
        poses = full_link_poses(s, q)
        pose_checks[name] = {
            "overlaps": overlaps(poses, link_prims, adj),
            "within_5mm": overlaps(poses, link_prims, adj, inflate=0.005),
            "jaw_tip_m": rvec(tcp(q)[1], 4),
            "arm_bbox_m": arm_bbox(poses, link_prims),
        }

    # --- hydro + thrusters (moments about the combined stowed CoM)
    co_bm_hull = cob_veh[2] - com_veh[2]
    # effective arm for the single-body restoring term: system righting stiffness / (rho g V_hull)
    co_bm_eff = sys_stow["righting_stiffness_nm_per_rad"] / (RHO_WATER * GRAVITY * v_veh)
    hydro, hydro_table = hydro_coefficients(v_veh, co_bm_eff)
    thrusters = build_thrusters(parts)
    # allocation reference: combined stowed CoM on the centre line (its 0.6 mm lateral offset from the stowed arm is
    # ignored so that the symmetric layout stays exactly decoupled)
    ref = (sys_stow["com_m"][0], 0.0, sys_stow["com_m"][2])
    cap = axis_capability(thrusters, ref)
    speeds = {
        "surge_fwd_588N": steady_speed(hydro["linear_damping"][0], hydro["quadratic_damping"][0],
                                       SPEC_THRUST_FWD_KGF * KGF_N),
        "surge_rev_294N": steady_speed(hydro["linear_damping"][0], hydro["quadratic_damping"][0],
                                       SPEC_THRUST_REV_KGF * KGF_N),
        "sway_329N": steady_speed(hydro["linear_damping"][1], hydro["quadratic_damping"][1],
                                  SPEC_THRUST_LAT_KGF * KGF_N),
        "heave_304N": steady_speed(hydro["linear_damping"][2], hydro["quadratic_damping"][2],
                                   SPEC_THRUST_VERT_KGF * KGF_N),
        "yaw_rate_at_thruster_yaw_torque_rad_s": steady_speed(hydro["linear_damping"][5],
                                                               hydro["quadratic_damping"][5], cap["yaw"]),
        "yaw_rate_at_100Nm_rad_s": steady_speed(hydro["linear_damping"][5], hydro["quadratic_damping"][5], 100.0),
    }

    return {"state": s, "parts": parts, "l2b": l2b, "link_prims": link_prims, "vehicle_links": vehicle_links,
            "vehicle": {"mass_kg": m_veh, "com_m": com_veh, "inertia_about_com": i_veh,
                        "displaced_volume_m3": v_veh, "cob_m": cob_veh, "cob_minus_com_m": vsub(cob_veh, com_veh),
                        "co_bm_hull_only_m": co_bm_hull},
            "system_stow": sys_stow, "system_home": sys_home, "q_home": q_home, "q_stow": q_stow,
            "home_ik": home_ik, "seabed": seabed, "forward": forward, "seabed_envelope": envelope,
            "pose_checks": pose_checks, "hydro": hydro, "hydro_table": hydro_table, "thrusters": thrusters,
            "thruster_fit": thruster_fit(), "axis_capability_n": cap, "allocation_ref_m": ref, "speeds": speeds}


def arm_bbox(poses: dict[str, tuple[Vec3, Mat3]], link_prims: dict[str, list[Primitive]]) -> dict[str, list[float]]:
    """Axis-aligned bounds of the arm's sampled primitive points in rov_base."""
    pts: list[Vec3] = []
    for wp in world_prims(poses, {k: v for k, v in link_prims.items() if k not in VEHICLE_LINKS}):
        pts += [point_in((wp.pos, wp.rot), q) for q in sample_local(wp.prim, 0.0)]
    return {"min": rvec([min(p[i] for p in pts) for i in range(3)], 3),
            "max": rvec([max(p[i] for p in pts) for i in range(3)], 3)}


def build_links_and_joints(model: dict) -> tuple[list[OutLink], list[OutJoint]]:
    """URDF links/joints from the assembled model."""
    s: TrimState = model["state"]
    links: list[OutLink] = []
    for link in VEHICLE_LINKS:
        m, com, inertia = model["vehicle_links"][link]
        vol = sum(p.volume for p in model["parts"] if p.link == link)
        links.append(OutLink(link, m, com, inertia, model["link_prims"][link],
                             f" displaced_volume_m3={vol:.6e} (rho={RHO_WATER:g}) "))
    for link in ARM_LINKS:
        links.append(OutLink(link.name, link.mass, link.com, link.inertia, link.primitives,
                             f" displaced_volume_m3={link.displaced_volume:.6e} centre_of_buoyancy=centre_of_mass "))
    joints: list[OutJoint] = []
    for link, (xyz, rpy) in sensor_link_poses(s).items():
        joints.append(OutJoint(link.replace("_link", "_joint"), "fixed", ROV_BASE, link, xyz, rpy))
    joints.append(OutJoint("arm_mount_joint", "fixed", ROV_BASE, ARM_BASE_LINK, MOUNT_POS, ARM_MOUNT_RPY))
    for j in ARM_JOINTS:
        joints.append(OutJoint(j.name, j.kind, j.parent, j.child, j.xyz, (0.0, 0.0, 0.0), j.axis))
    return links, joints


def sym_list(m: Mat3, n: int = 6) -> list[list[float]]:
    """Rounded 3x3 for JSON."""
    return [[rnd(v, n) for v in row] for row in m]


def params_json(model: dict) -> dict:
    """Serialisable design table."""
    s: TrimState = model["state"]
    veh = model["vehicle"]
    out_links = {}
    for link in VEHICLE_LINKS:
        m, com, inertia = model["vehicle_links"][link]
        vol = sum(p.volume for p in model["parts"] if p.link == link)
        pos, rot = model["l2b"][link]
        out_links[link] = {"mass_kg": rnd(m), "com_in_link_m": rvec(com), "inertia_about_com_kg_m2": sym_list(inertia),
                           "principal_inertia_kg_m2": rvec(eig_sym3(inertia)), "displaced_volume_m3": rnd(vol, 8),
                           "frame_pos_in_rov_base_m": rvec(pos), "frame_quat_xyzw_in_rov_base": rvec(mat_to_quat_xyzw(rot))}
    parts = [{"name": p.name, "link": p.link, "kind": p.prim.kind, "size": rvec(p.prim.size), "xyz": rvec(p.prim.xyz),
              "rpy": rvec(p.prim.rpy), "mass_kg": rnd(p.mass, 4), "displaced_volume_m3": rnd(p.volume, 7),
              "visual": p.visual} for p in model["parts"]]

    def sys_dict(d: dict) -> dict:
        return {"mass_kg": rnd(d["mass_kg"], 4), "com_m": rvec(d["com_m"]), "cob_m": rvec(d["cob_m"]),
                "cob_minus_com_m": rvec(d["cob_minus_com_m"]), "displaced_volume_m3": rnd(d["displaced_volume_m3"], 7),
                "net_buoyancy_kg": rnd(d["net_buoyancy_kg"], 4), "net_buoyancy_n": rnd(d["net_buoyancy_kg"] * GRAVITY, 3),
                "righting_stiffness_nm_per_rad": rnd(d["righting_stiffness_nm_per_rad"], 3),
                "inertia_about_com_kg_m2": sym_list(d["inertia_about_com"]),
                "principal_inertia_kg_m2": rvec(eig_sym3(d["inertia_about_com"]))}

    sensor_frames = {}
    for link in (CAMERA_LINK, DVL_LINK, PRESSURE_LINK):
        pos, rot = model["l2b"][link]
        sensor_frames[link] = {"pos_m": rvec(pos), "quat_xyzw": rvec(mat_to_quat_xyzw(rot)),
                               "rpy_rad": rvec(sensor_link_poses(s)[link][1])}
    fit = model["thruster_fit"]
    cap = model["axis_capability_n"]
    spec = {"+x": SPEC_THRUST_FWD_KGF, "-x": SPEC_THRUST_REV_KGF, "+y": SPEC_THRUST_LAT_KGF, "-y": SPEC_THRUST_LAT_KGF,
            "+z": SPEC_THRUST_VERT_KGF}
    match = {k: {"model_kgf": rnd(cap[k] / KGF_N, 4), "catalogue_kgf": v, "rel_err": rnd(cap[k] / KGF_N / v - 1.0, 6)}
             for k, v in spec.items()}
    match["-z"] = {"model_kgf": rnd(cap["-z"] / KGF_N, 4), "catalogue_kgf": None,
                   "servo_cap_kgf": SPEC_THRUST_VERT_KGF, "rel_err_vs_servo_cap": rnd(cap["-z"] / KGF_N / 31.0 - 1.0, 6)}
    return {
        "robot": ROBOT_NAME,
        "generator": "rov_lab/scripts/tools/make_i90_bravo7_urdf.py",
        "spec": {"length_m": SPEC_LENGTH_M, "width_m": SPEC_WIDTH_M, "height_m": SPEC_HEIGHT_M,
                 "mass_air_kg": SPEC_MASS_AIR_KG, "payload_kg": SPEC_PAYLOAD_KG, "speed_kn": SPEC_SPEED_KN,
                 "depth_m": SPEC_DEPTH_M, "thrust_kgf": {"forward": SPEC_THRUST_FWD_KGF, "reverse": SPEC_THRUST_REV_KGF,
                                                         "lateral": SPEC_THRUST_LAT_KGF, "vertical": SPEC_THRUST_VERT_KGF},
                 "catalogue_manipulator": SPEC_CATALOGUE_MANIPULATOR,
                 "mounted_manipulator": "Bravo 7-class, 6 DOF + jaws, 9.5 kg air / 4.5 kg water (inside the 27 kg payload)",
                 "rho_water": RHO_WATER, "gravity": GRAVITY, "target_net_buoyancy_kg": TARGET_NET_BUOYANCY_KG},
        "trim_solution": {"foam_thickness_m": rnd(s.foam_t), "foam_density_kg_m3": RHO_FOAM,
                          "ballast_lead_kg": rnd(s.ballast_kg, 4), "ballast_x_m": rnd(s.ballast_x, 5),
                          "allocation_ref_x_m": rnd(s.x_ref), "horizontal_thruster_z_m": rnd(s.z_ref)},
        "vehicle": {"mass_kg": rnd(veh["mass_kg"], 5), "com_m": rvec(veh["com_m"]), "cob_m": rvec(veh["cob_m"]),
                    "cob_minus_com_m": rvec(veh["cob_minus_com_m"]), "displaced_volume_m3": rnd(veh["displaced_volume_m3"], 8),
                    "net_buoyancy_kg": rnd(RHO_WATER * veh["displaced_volume_m3"] - veh["mass_kg"], 4),
                    "inertia_about_com_kg_m2": sym_list(veh["inertia_about_com"]),
                    "principal_inertia_kg_m2": rvec(eig_sym3(veh["inertia_about_com"]))},
        "system_stow": sys_dict(model["system_stow"]),
        "system_home": sys_dict(model["system_home"]),
        "links": out_links,
        "arm_links": {link.name: {"mass_kg": link.mass, "displaced_volume_m3": link.displaced_volume} for link in ARM_LINKS},
        "parts": parts,
        "link_primitives": {link: [{"kind": p.kind, "size": list(p.size), "xyz": list(p.xyz), "rpy": list(p.rpy)}
                                   for p in prims] for link, prims in model["link_prims"].items()},
        "arm_mount": {"joint": "arm_mount_joint", "parent": ROV_BASE, "child": ARM_BASE_LINK, "pos_m": rvec(MOUNT_POS),
                      "rpy_rad": rvec(ARM_MOUNT_RPY), "quat_xyzw": rvec(mat_to_quat_xyzw(MOUNT_ROT)),
                      "shoulder_axis_m": rvec((SHOULDER_X, 0.0, ARM_AXIS_Z))},
        "sensor_frames": sensor_frames,
        "poses": {"home": model["q_home"], "stow": {k: rnd(v, 5) for k, v in model["q_stow"].items()},
                  "home_ik": model["home_ik"]},
        "reach": {"seabed_ahead_of_bow_m": SEABED_TARGET_AHEAD_OF_BOW_M, "seabed": model["seabed"],
                  "forward_ahead_of_bow_m": FORWARD_TARGET_AHEAD_OF_BOW_M, "forward": model["forward"],
                  "seabed_envelope_ahead_of_bow_m": model["seabed_envelope"]},
        "pose_checks": model["pose_checks"],
        "hydro_params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in model["hydro"].items()},
        "hydro_derivation": model["hydro_table"],
        "hydro_co_bm_hull_only_m": rnd(veh["co_bm_hull_only_m"], 5),
        "steady_state": {k: rnd(v, 4) for k, v in model["speeds"].items()},
        "thrusters": [{"name": t.name, "pos_m": rvec(t.pos), "direction": rvec(t.direction),
                       "max_forward_n": rnd(t.max_forward_n, 3), "max_reverse_n": rnd(t.max_reverse_n, 3)}
                      for t in model["thrusters"]],
        "thruster_fit": {k: rnd(v, 6) for k, v in fit.items()},
        "allocation_ref_m": rvec(model["allocation_ref_m"]),
        "axis_capability_n": {k: rnd(v, 3) for k, v in cap.items()},
        "axis_thrust_match": match,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    args = parser.parse_args()

    model = build_model()
    veh = model["vehicle"]
    if abs(veh["mass_kg"] - SPEC_MASS_AIR_KG) > 1e-6:
        raise ValueError(f"vehicle mass {veh['mass_kg']:.4f} kg != {SPEC_MASS_AIR_KG}")
    s: TrimState = model["state"]
    if not (0.05 < s.foam_t < 0.25) or s.ballast_kg < 0.0:
        raise ValueError(f"trim solution out of range: {s}")
    for name, chk in model["pose_checks"].items():
        if chk["overlaps"]:
            raise ValueError(f"{name} pose overlaps: {chk['overlaps']}")

    links, joints = build_links_and_joints(model)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    urdf_path = args.out_dir / f"{ROBOT_NAME}.urdf"
    ET.ElementTree(build_urdf(links, joints)).write(urdf_path, encoding="utf-8", xml_declaration=True)
    params = params_json(model)
    params_path = args.out_dir / f"{ROBOT_NAME}_params.json"
    params_path.write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")

    st, hm = params["system_stow"], params["system_home"]
    print(f"[i90_bravo7] wrote {urdf_path}\n[i90_bravo7] wrote {params_path}")
    print(f"[i90_bravo7] trim: foam {s.foam_t * 1e3:.1f} mm, lead {s.ballast_kg:.2f} kg at x={s.ballast_x:+.3f}, "
          f"thruster ref x={s.x_ref:+.4f} z={s.z_ref:+.4f}")
    v = params["vehicle"]
    print(f"[i90_bravo7] vehicle: m={v['mass_kg']} kg V={v['displaced_volume_m3'] * 1e3:.2f} L net={v['net_buoyancy_kg']} kg "
          f"CoM={v['com_m']} CoB={v['cob_m']} I_principal={v['principal_inertia_kg_m2']}")
    print(f"[i90_bravo7] system stow: m={st['mass_kg']} net={st['net_buoyancy_kg']} kg CoM={st['com_m']} CoB={st['cob_m']} "
          f"CoB-CoM={st['cob_minus_com_m']}")
    print(f"[i90_bravo7] system home: CoM={hm['com_m']} CoB-CoM={hm['cob_minus_com_m']}")
    for depth, r in params["reach"]["seabed"].items():
        print(f"[i90_bravo7] seabed {depth} m: {r.get('reachable')} jaw {r.get('jaw_angle_from_down_deg')} deg "
              f"elbow_up={r.get('elbow_up')} q={r.get('q')}")
    f = params["reach"]["forward"]
    print(f"[i90_bravo7] forward: {f.get('reachable')} jaw {f.get('jaw_angle_from_down_deg')} q={f.get('q')}")
    print(f"[i90_bravo7] seabed envelope (ahead of bow): {params['reach']['seabed_envelope_ahead_of_bow_m']}")
    for name, chk in params["pose_checks"].items():
        print(f"[i90_bravo7] {name}: overlaps={chk['overlaps']} within5mm={chk['within_5mm']} bbox={chk['arm_bbox_m']}")
    print(f"[i90_bravo7] hydro: {params['hydro_params']}")
    print(f"[i90_bravo7] steady: {params['steady_state']}")
    print(f"[i90_bravo7] thrust match: {params['axis_thrust_match']}")


if __name__ == "__main__":
    main()
