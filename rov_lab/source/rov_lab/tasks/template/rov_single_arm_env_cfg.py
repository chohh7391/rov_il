from dataclasses import MISSING
from typing import Any

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import (
    ActionStateRecorderManagerCfg as RecordTerm,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg

from rov_lab.devices.action_process import SO101_TELEOP_SPEC, ArmTeleopSpec, init_action_cfg, preprocess_device_action
from . import mdp
from rov_lab.assets.robots import BLUE_ROV_SINGLE_ARM_CFG
from isaaclab_physx.physics import PhysxCfg

from rov_lab.assets.sensors import DVLCfg, UWCameraCfg


WATER_SURFACE = 3.43389
"""World z (m) of the water surface used for barometric depth. Chosen to sit above the MHL seabed scene."""

@configclass
class ROVSingleArmTaskSceneCfg(InteractiveSceneCfg):
    """Scene configuration for the single arm task."""

    scene: AssetBaseCfg = MISSING

    robot: ArticulationCfg = BLUE_ROV_SINGLE_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (-2.0, 0.0, -0.8)

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Arm/base",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/Arm/gripper", name="gripper"
            ),  # no offset for ik convert
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/Arm/jaw", name="jaw", offset=OffsetCfg(pos=(-0.021, -0.070, 0.02))
            ),  # set offset for obj detection
        ],
    )

    wrist: UWCameraCfg = UWCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Arm/gripper/wrist_camera",
        offset=UWCameraCfg.OffsetCfg(
            pos=(-0.001, 0.1, -0.04), rot=(-0.912179, -0.0451242, 0.0486914, -0.404379), convention="ros"
        ),  # xyzw (Isaac Lab 3.0)
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            focus_distance=400.0,
            horizontal_aperture=36.83,  # For a 75° FOV (assuming square image)
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,  # 30FPS
        update_latest_camera_pose=True,
        # OceanSim default underwater camera parameters.
        backscatter_value=(0.0, 0.31, 0.24),
        atten_coeff=(0.05, 0.05, 0.05),
        backscatter_coeff=(0.05, 0.05, 0.2),
    )

    front: UWCameraCfg = UWCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rov_base/uw_camera",
        offset=UWCameraCfg.OffsetCfg(pos=(0.3, 0.0, 0.1), rot=(-0.5, 0.5, -0.5, 0.5), convention="ros"),  # xyzw
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            horizontal_aperture=4.2,
            clipping_range=(0.1, 100.0),
            lock_camera=True,
        ),
        # 16:9 at 640x360 (was 1920x1080): the full-HD stream at 60 Hz made HDF5 demos unmanageably large.
        width=640,
        height=360,
        update_period=1/30.0,
        update_latest_camera_pose=True,
        # OceanSim default underwater camera parameters.
        backscatter_value=(0.0, 0.31, 0.24),
        atten_coeff=(0.05, 0.05, 0.05),
        backscatter_coeff=(0.05, 0.05, 0.2),
    )

    # ROV sensors (barometric depth is an observation term on the hull body; see mdp.barometer_depth)
    dvl: DVLCfg = DVLCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rov_base",
        body_name="rov_base",
        max_range=10.0,
        vel_cov=1.0e-4,  # per-beam velocity noise std 1 cm/s (OceanSim vel_cov)
        mesh_prim_paths=["/World/Scene/mhl_scaled/Mesh/mesh"],
        offset=DVLCfg.OffsetCfg(pos=(0.0, 0.0, -0.1), rot=(0.0, 0.0, 0.0, 1.0)),
    )

    # TODO(isaaclab3): imaging sonar not ported yet (see rov_lab/assets/sensors/imaging_sonar).

    


@configclass
class SingleArmActionsCfg:
    """Configuration for the actions."""

    rov_action: mdp.ROVVelocityActionCfg = mdp.ROVVelocityActionCfg(
        asset_name="robot",
        body_name="rov_base",
    )
    arm_action: mdp.ActionTermCfg = MISSING
    gripper_action: mdp.ActionTermCfg = MISSING


@configclass
class SingleArmEventCfg:
    """Configuration for the events."""

    # reset to default scene
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # MarineGym Fossen hydrodynamics on the ROV base, applied every step as an external wrench
    # (instantaneous composer, auto-summed with the action-term servo's permanent wrench). Default
    # term set = damping + buoyancy restoring moment (see mdp.HydroParamsCfg / docs).
    apply_hydro = EventTerm(
        func=mdp.apply_hydro_wrench,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        is_global_time=True,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="rov_base"),
            "hydro_cfg": mdp.HydroParamsCfg(),
        },
    )


@configclass
class ROVSingleArmObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        wrist = ObsTerm(func=mdp.uw_image, params={"sensor_cfg": SceneEntityCfg("wrist")})
        ee_frame_state = ObsTerm(
            func=mdp.ee_frame_state,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame"), "robot_cfg": SceneEntityCfg("robot")},
        )
        joint_pos_target = ObsTerm(func=mdp.joint_pos_target, params={"asset_cfg": SceneEntityCfg("robot")})

        # ROV observations
        depth = ObsTerm(
            func=mdp.barometer_depth,
            # pressure noise 49 Pa std (~5 mm of depth), OceanSim-compatible pressure model inverted to depth
            params={"asset_cfg": SceneEntityCfg("robot", body_names="rov_base"), "water_surface_z": WATER_SURFACE,
                    "noise_cov": 2400.0},
        )
        dvl_vel = ObsTerm(
            func=mdp.dvl_linear_velocity,
            params={"sensor_cfg": SceneEntityCfg("dvl")},
        )
        dvl_depth = ObsTerm(
            func=mdp.dvl_depth,
            params={"sensor_cfg": SceneEntityCfg("dvl")},
        )
        dvl_beam_hit = ObsTerm(
            func=mdp.dvl_beam_hit,
            params={"sensor_cfg": SceneEntityCfg("dvl")},
        )
        front = ObsTerm(func=mdp.uw_image, params={"sensor_cfg": SceneEntityCfg("front")})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class SingleArmRewardsCfg:
    """Configuration for the rewards"""


@configclass
class ROVSingleArmTerminationsCfg:
    """Configuration for the termination"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class ROVSingleArmTaskEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the single arm task template environment."""

    scene: ROVSingleArmTaskSceneCfg = MISSING

    observations: ROVSingleArmObservationsCfg = MISSING
    actions: SingleArmActionsCfg = SingleArmActionsCfg()
    events: SingleArmEventCfg = SingleArmEventCfg()

    rewards: SingleArmRewardsCfg = SingleArmRewardsCfg()
    terminations: ROVSingleArmTerminationsCfg = MISSING

    recorders: RecordTerm = RecordTerm()

    task_description: str = MISSING
    """Task description stored with recorded demonstrations."""

    arm_teleop_spec: ArmTeleopSpec = SO101_TELEOP_SPEC
    """Arm-specific teleop mapping (IK joints, end-effector body, gripper type)."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.decimation = 1
        self.episode_length_s = 25.0
        self.viewer.eye = (-5.0, 0.0, 0.4)
        self.viewer.lookat = (-2.0, 0.0, -0.8)

        # Isaac Lab 3.0: physics backend settings live on sim.physics (was sim.physx). Translucency (sim.render)
        # moved to the RTX renderer settings and is left at its default.
        self.sim.physics = PhysxCfg(bounce_threshold_velocity=0.01, friction_correlation_distance=0.00625)

        self.scene.ee_frame.visualizer_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)

    def use_teleop_device(self, teleop_device) -> None:
        self.task_type = teleop_device
        self.actions = init_action_cfg(self.actions, device=teleop_device, spec=self.arm_teleop_spec)
        if teleop_device in ["keyboard", "gamepad", "so101_state_machine"]:
            self.scene.robot.spawn.rigid_props.disable_gravity = True

    def preprocess_device_action(self, action: dict[str, Any], teleop_device) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)
