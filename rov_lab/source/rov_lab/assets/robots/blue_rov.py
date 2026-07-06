"""Configuration for the BlueROV robot assets used by ROV Lab."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from rov_lab.utils.paths import ROBOT_DIR


BLUE_ROV_SINGLE_ARM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ROBOT_DIR / "blue_rov_single_arm.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            linear_damping=10.0,
            angular_damping=10.0,
            max_linear_velocity=5.0,
            max_angular_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            fix_root_link=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        },
    ),
    actuators={
        "sts3215-gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=10,
            velocity_limit_sim=10,
            stiffness=17.8,
            # Raised from 0.60 to damp the arm-joint resonance (zeta ~0.2 -> ~0.7). The lightly
            # damped joints were excited by continuous base yaw (centripetal/gyroscopic load),
            # feeding an oscillation back into the hull. See docs/rov-base-control-design.md.
            # Starting target for typical link inertia; fine-tune in sim.
            damping=2.0,
        ),
        "sts3215-arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            effort_limit_sim=10,
            velocity_limit_sim=10,
            stiffness=17.8,
            # Raised from 0.60 to damp the arm-joint resonance (zeta ~0.2 -> ~0.7). The lightly
            # damped joints were excited by continuous base yaw (centripetal/gyroscopic load),
            # feeding an oscillation back into the hull. See docs/rov-base-control-design.md.
            # Starting target for typical link inertia; fine-tune in sim.
            damping=2.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
