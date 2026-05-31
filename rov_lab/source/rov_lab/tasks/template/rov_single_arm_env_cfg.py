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
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets.episode_data import EpisodeData
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg

from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg
from leisaac.utils.constant import SINGLE_ARM_JOINT_NAMES
from leisaac.utils.robot_utils import convert_leisaac_action_to_lerobot

from rov_lab.devices.action_process import init_action_cfg, preprocess_device_action
from . import mdp
from rov_lab.assets.robots import BLUE_ROV_SINGLE_ARM_CFG
from rov_lab.assets.sensors import BarometerCfg, DVLCfg, ImagingSonarSensorCfg, UWCameraCfg


WATER_SURFACE = 3.43389  # Arbitary

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

    wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Arm/gripper/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.001, 0.1, -0.04), rot=(-0.404379, -0.912179, -0.0451242, 0.0486914), convention="ros"
        ),  # wxyz
        data_types=["rgb"],
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
    )

    front: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Arm/base/front_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, -0.5, 0.6), rot=(0.1650476, -0.9862856, 0.0, 0.0), convention="ros"
        ),  # wxyz
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.7,
            focus_distance=400.0,
            horizontal_aperture=38.11,  # For a 78° FOV (assuming square image)
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,  # 30FPS
    )

    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # ROV sensors
    barometer: BarometerCfg = BarometerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rov_base",
        offset=BarometerCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        water_surface_z=WATER_SURFACE,
    )
    
    dvl: DVLCfg = DVLCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rov_base",
        max_range=10.0,
        mesh_prim_paths=["/World/Scene/mhl_scaled/Mesh/mesh"],
        offset=DVLCfg.OffsetCfg(pos=(0.0, 0.0, -0.1), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    sonar: ImagingSonarSensorCfg = ImagingSonarSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rov_base/sonar",
        offset=ImagingSonarSensorCfg.OffsetCfg(
            pos=(0.0, 0.0, -0.3),
            rot=(0.0, 1.0, 0.0, 0.0),
            convention="ros",
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            clipping_range=(0.2, 3.0),
        ),
        # Oceansim defaults the user has been running with
        min_range=0.2,
        max_range=3.0,
        range_res=0.005,
        hori_fov=130.0,
        vert_fov=20.0,
        angular_res=0.25,
        hori_res=4000,
        include_unlabelled=True,
        update_period=1/10.0,
        update_latest_camera_pose=True,
        debug_vis=True,
    )

    uw_camera: UWCameraCfg = UWCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rov_base/uw_camera",
        offset=UWCameraCfg.OffsetCfg(pos=(0.3, 0.0, 0.1), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            clipping_range=(0.1, 100.0),
            lock_camera=True,
        ),
        width=1920,
        height=1080,
        update_period=1/30.0,
        update_latest_camera_pose=True,
        # coastal-water defaults; tune per scene
        backscatter_value=(0.0, 0.31, 0.24),
        atten_coeff=(0.05, 0.05, 0.05),
        backscatter_coeff=(0.05, 0.05, 0.20),
        depth_clipping_behavior="max",
    )


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
        wrist = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist"), "data_type": "rgb", "normalize": False}
        )
        front = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("front"), "data_type": "rgb", "normalize": False}
        )
        ee_frame_state = ObsTerm(
            func=mdp.ee_frame_state,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame"), "robot_cfg": SceneEntityCfg("robot")},
        )
        joint_pos_target = ObsTerm(func=mdp.joint_pos_target, params={"asset_cfg": SceneEntityCfg("robot")})

        # ROV observations
        barometer = ObsTerm(
            func=mdp.brometer_pressure,
            params={"sensor_cfg": SceneEntityCfg("barometer")}
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
        uw_image = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("uw_camera"), "data_type": "uw_image", "normalize": False},
        )
        sonar_image = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("sonar"), "data_type": "sonar_image", "normalize": False},
        )

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

    dynamic_reset_gripper_effort_limit: bool = True
    """Whether to dynamically reset the gripper effort limit."""

    robot_name: str = "so101_follower"
    """Robot name for lerobot dataset export."""
    default_feature_joint_names: list[str] = MISSING
    """Default feature joint names for lerobot dataset export."""
    task_description: str = MISSING
    """Task description for lerobot dataset export."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.decimation = 1
        self.episode_length_s = 25.0
        self.viewer.eye = (5.0, 0.6, 0.4)
        self.viewer.lookat = (-2.0, 0.0, -0.8)

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.render.enable_translucency = True

        self.scene.ee_frame.visualizer_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)

        self.default_feature_joint_names = [f"{joint_name}.pos" for joint_name in SINGLE_ARM_JOINT_NAMES]

    def use_teleop_device(self, teleop_device) -> None:
        self.task_type = teleop_device
        self.actions = init_action_cfg(self.actions, device=teleop_device)
        if teleop_device in ["keyboard", "gamepad", "so101_state_machine"]:
            self.scene.robot.spawn.rigid_props.disable_gravity = True

    def preprocess_device_action(self, action: dict[str, Any], teleop_device) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)

    def build_lerobot_frame(self, episode_data: EpisodeData, dataset_cfg: LeRobotDatasetCfg) -> dict:
        obs_data = episode_data._data["obs"]
        action = episode_data._data["actions"][-1]
        if dataset_cfg.action_align:
            processed_action = convert_leisaac_action_to_lerobot(action.unsqueeze(0)).squeeze(0)
        else:
            processed_action = action.cpu().numpy()
        frame = {
            "action": processed_action,
            "observation.state": convert_leisaac_action_to_lerobot(obs_data["joint_pos"][-1].unsqueeze(0)).squeeze(0),
            "task": self.task_description,
        }
        for frame_key in dataset_cfg.features.keys():
            if not frame_key.startswith("observation.images"):
                continue
            camera_key = frame_key.split(".")[-1]
            frame[frame_key] = obs_data[camera_key][-1].cpu().numpy()

        return frame
