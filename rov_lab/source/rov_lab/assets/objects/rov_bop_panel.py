"""Subsea valve panel: UUV Simulator ``rov_bop_panel`` (Apache-2.0) rebuilt as a USD articulation.

Panel body (0.5 x 0.8 x 0.2 m, 10 kg, fixed to the world) with five quarter-turn paddle valves on revolute joints:

* ``valve_1``-``valve_3``: 145 mm disk, paddle 15 x 100 x 40 mm (big valves)
* ``valve_4``, ``valve_5``: 75 mm disk, paddle 50 x 10 x 40 mm (small valves, recessed 6 cm)

Joint range -90 deg (open) .. 0 deg (closed). Gazebo used damping 0.5 N*m*s/rad and an ODE friction of 10 N*m; here
the resistance is an Isaac Lab joint friction (static breakaway / dynamic effort in N*m) plus viscous damping, set per
task through :data:`VALVE_FRICTION`. Build with ``rov_lab/scripts/tools/build_subsea_assets.py --only panel``.
"""

from __future__ import annotations

from pathlib import Path

from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from rov_lab.assets.scenes.work_site_layout import PANEL_POS, PANEL_QUAT_WXYZ
from rov_lab.utils.paths import OBJECT_DIR

ROV_BOP_PANEL_USD_PATH: Path = OBJECT_DIR / "rov_bop_panel" / "rov_bop_panel.usda"

VALVE_NAMES: tuple[str, ...] = ("valve_1", "valve_2", "valve_3", "valve_4", "valve_5")
VALVE_JOINT_NAMES: tuple[str, ...] = tuple(f"{v}_joint" for v in VALVE_NAMES)
VALVE_OPEN_POS: float = -1.5708
VALVE_CLOSED_POS: float = 0.0

VALVE_FRICTION: dict[str, float] = {"static": 4.0, "dynamic": 3.0, "viscous": 0.5}
"""Valve resistance: breakaway and running friction (N*m) and viscous coefficient (N*m*s/rad).

Kept below the Bravo 7-class end-effector roll limit (20 N*m) with margin for the IK-driven wrist."""

VALVE_ARMATURE: float = 0.01
"""Reflected stem/gear inertia (kg*m^2). Also required numerically: without it PhysX's joint friction acts far above
its set value on the light small valves (3.5e-4 kg*m^2 link inertia; 8 N*m could not break a 4 N*m static friction),
while the big valves behaved. With >= 0.002 both kinds hold at 0.6x and open at 1.1x the static friction."""

# panel-frame geometry of the valves (from the source SDF and meshes): name -> (x, y, z of the paddle centre)
VALVE_PADDLE_CENTERS_PANEL: dict[str, tuple[float, float, float]] = {
    "valve_1": (-0.15, -0.2, 0.21),
    "valve_2": (0.15, -0.2, 0.21),
    "valve_3": (-0.15, 0.0, 0.21),
    "valve_4": (0.0, -0.2, 0.18),
    "valve_5": (0.0, 0.0, 0.18),
}

_xyzw = (PANEL_QUAT_WXYZ[1], PANEL_QUAT_WXYZ[2], PANEL_QUAT_WXYZ[3], PANEL_QUAT_WXYZ[0])

ROV_BOP_PANEL_CFG: ArticulationCfg = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ROV_BOP_PANEL_USD_PATH),
        activate_contact_sensors=False,
        fix_root_link=True,
        rigid_props=PhysxRigidBodyCfg(disable_gravity=True, max_depenetration_velocity=1.0),
        articulation_props=PhysxArticulationCfg(
            enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=PANEL_POS,
        rot=_xyzw,
        joint_pos={".*": VALVE_CLOSED_POS},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "valves": ImplicitActuatorCfg(
            joint_names_expr=["valve_.*_joint"],
            stiffness=0.0,
            damping=0.0,
            friction=VALVE_FRICTION["static"],
            dynamic_friction=VALVE_FRICTION["dynamic"],
            viscous_friction=VALVE_FRICTION["viscous"],
            armature=VALVE_ARMATURE,
            joint_effort_limit=100.0,
            joint_velocity_limit=10.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Valve panel fixed at the work-site mount (``work_site_layout.PANEL_POS``), all valves closed."""
