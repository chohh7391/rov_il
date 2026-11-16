"""Valve-panel intervention task: I90-class ROV with a Bravo 7-class arm turns one paddle valve a quarter turn.

Scene (``rov_lab.assets.scenes.work_site_layout``): the ROV holds station 1 m in front of a UUV Simulator
``rov_bop_panel`` bolted to a DAVE skid frame on an open-source seabed (Gazebo Fuel shipwreck site or MBARI Portuguese
Ledge). Each episode one target valve is drawn; success = that valve opened >= 90 % of its quarter turn, failure = a
different valve turned more than 25 %. As in the 2026 fixed-base IL experiment the ROV base is fixed while the arm
works (``fix_rov_base``); set it to False for free-floating station keeping with the I90 hydrodynamics.

Control runs at 30 Hz (physics 60 Hz, decimation 2), matching the 30 Hz cameras. Each reset jitters the hull pose
(station-keeping error, +-2 cm / +-1 cm / +-3 deg) and the arm's start pose (+-0.05 rad), and draws the target among
all five valves.

Actions:
* ``ROVLab-I90Bravo7-ValvePanel-v0`` (keyboard, scripted expert): ROV body velocity 6 + arm end-effector delta 6 (relative
  IK in the ROV base frame) + parallel gripper rate 1 = 13-D.
* ``ROVLab-I90Bravo7-ValvePanel-AbsPose-v0`` (learned policies): ROV velocity 6 + absolute jaw pose in the ROV base frame
  7 (position + quaternion xyzw, absolute IK) + gripper opening 1 (absolute, [-1, 1] -> [closed, open]) = 14-D.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass

from rov_lab.assets.objects.rov_bop_panel import ROV_BOP_PANEL_CFG, VALVE_JOINT_NAMES
from rov_lab.assets.robots.i90_bravo7 import (
    ARM_BASE_LINK,
    EE_LINK,
    FRONT_CAMERA_LINK,
    GRASP_POINT_OFFSET_M,
    I90_BRAVO7_CFG,
    I90_BRAVO7_PRIM_PATHS,
    I90_DVL_OFFSET_FOR_DVLCFG,
    I90_HYDRO_PARAMS,
    ROV_BASE_LINK,
)
from rov_lab.assets.scenes import work_site_cfg
from rov_lab.assets.scenes.work_site_layout import PANEL_POS, ROV_STATION_POS
from rov_lab.assets.sensors import DVLCfg, UWCameraCfg
from rov_lab.devices.action_process import BRAVO7_TELEOP_SPEC, ArmTeleopSpec

from ..template import ROVSingleArmObservationsCfg, ROVSingleArmTaskEnvCfg, ROVSingleArmTerminationsCfg, SingleArmEventCfg
from . import mdp

VALVE_TARGET_CANDIDATES: tuple[int, ...] = (0, 1, 2, 3, 4)
"""Targets drawn each episode: all five valves (valves 4 and 5 are the small ones, 6 cm deep in their pockets)."""

STATION_JITTER: dict[str, tuple[float, float]] = {
    "x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.01, 0.01), "yaw": (-0.0524, 0.0524)}
"""Hull pose error at each reset (m, rad), as in the 2026 fixed-base experiment (station-keeping accuracy)."""
ARM_START_JITTER: tuple[float, float] = (-0.05, 0.05)
"""Offset (rad) added to each arm joint's home position at reset."""

SITE_DEPTH_M: dict[str, float] = {"shipwreck_site": 30.0, "portuguese_ledge": 97.0}
"""Water depth at the work-site seabed (barometric depth reference); the Portuguese Ledge tile is at 97 m."""

SEABED_MESH: dict[str, str] = {
    "shipwreck_site": "/World/WorkSite/terrain/terrain/Ground_lib",
    "portuguese_ledge": "/World/WorkSite/terrain/terrain/seabed",
}

I90_HYDRO_CFG = mdp.HydroParamsCfg(**I90_HYDRO_PARAMS)
"""MarineGym/Fossen coefficients sized for the I90 hull (see ``i90_bravo7_params.json`` hydro_derivation)."""

_P = I90_BRAVO7_PRIM_PATHS
_UW_WATER = dict(backscatter_value=(0.0, 0.31, 0.24), atten_coeff=(0.05, 0.05, 0.05), backscatter_coeff=(0.05, 0.05, 0.2))


@configclass
class ValvePanelSceneCfg(InteractiveSceneCfg):
    """I90 + Bravo 7-class in front of the valve panel at the work site."""

    site: AssetBaseCfg = work_site_cfg("shipwreck_site")

    sun: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/sun", spawn=sim_utils.DistantLightCfg(intensity=2500.0, angle=8.0, color=(0.85, 0.95, 1.0))
    )
    dome: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/dome", spawn=sim_utils.DomeLightCfg(intensity=400.0, color=(0.55, 0.75, 0.9))
    )

    robot: ArticulationCfg = I90_BRAVO7_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=I90_BRAVO7_CFG.init_state.replace(pos=ROV_STATION_POS, rot=(0.0, 0.0, 0.0, 1.0)),
    )
    panel: ArticulationCfg = ROV_BOP_PANEL_CFG.replace(prim_path="{ENV_REGEX_NS}/Panel")

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path=_P[ARM_BASE_LINK],
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path=_P[EE_LINK], name="jaw_base"),  # IK body, no offset
            FrameTransformerCfg.FrameCfg(prim_path=_P[EE_LINK], name="tcp", offset=OffsetCfg(pos=GRASP_POINT_OFFSET_M)),
        ],
    )

    # wrist camera on the jaw base, 6 cm off the jaw axis, looking along the jaws 22 deg toward the grasp point
    wrist: UWCameraCfg = UWCameraCfg(
        prim_path=f"{_P[EE_LINK]}/wrist_camera",
        offset=UWCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.06), rot=(0.0, 0.190809, 0.0, 0.981627), convention="world"),
        spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, horizontal_aperture=20.955, clipping_range=(0.02, 50.0)),
        width=640,
        height=480,
        update_period=1 / 30.0,
        update_latest_camera_pose=True,
        **_UW_WATER,
    )

    # pilot camera in the front_camera_link pod (optical axis = link +x, 20 deg down), 90 deg HFOV
    front: UWCameraCfg = UWCameraCfg(
        prim_path=f"{_P[FRONT_CAMERA_LINK]}/front_camera",
        offset=UWCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
        spawn=sim_utils.PinholeCameraCfg(focal_length=14.0, horizontal_aperture=28.0, clipping_range=(0.05, 100.0)),
        width=640,
        height=360,
        update_period=1 / 30.0,
        update_latest_camera_pose=True,
        **_UW_WATER,
    )

    dvl: DVLCfg = DVLCfg(
        prim_path=_P[ROV_BASE_LINK],
        body_name=ROV_BASE_LINK,
        max_range=30.0,
        vel_cov=1.0e-4,
        mesh_prim_paths=[SEABED_MESH["shipwreck_site"]],
        offset=DVLCfg.OffsetCfg(pos=I90_DVL_OFFSET_FOR_DVLCFG[0], rot=I90_DVL_OFFSET_FOR_DVLCFG[1]),
    )


_PANEL = SceneEntityCfg("panel", joint_names=list(VALVE_JOINT_NAMES))


@configclass
class ValvePanelEventCfg(SingleArmEventCfg):
    """Reset to default, draw the target valve, I90 hydrodynamics (free-floating only)."""

    reset_target = EventTerm(func=mdp.reset_target_valve, mode="reset",
                             params={"candidates": VALVE_TARGET_CANDIDATES})
    jitter_station = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": STATION_JITTER, "velocity_range": {}, "asset_cfg": SceneEntityCfg("robot")},
    )
    jitter_arm = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": ARM_START_JITTER, "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=list(BRAVO7_TELEOP_SPEC.ik_joint_names))},
    )

    def __post_init__(self) -> None:
        self.apply_hydro.params["hydro_cfg"] = I90_HYDRO_CFG


@configclass
class ValvePanelObservationsCfg(ROVSingleArmObservationsCfg):
    @configclass
    class PolicyCfg(ROVSingleArmObservationsCfg.PolicyCfg):
        target_valve = ObsTerm(func=mdp.target_valve_one_hot, params={"num_valves": len(VALVE_JOINT_NAMES)})

    @configclass
    class SubtaskCfg(ObsGroup):
        valve_opening = ObsTerm(func=mdp.valve_opening, params={"panel_cfg": _PANEL})
        target_valve_opening = ObsTerm(func=mdp.target_valve_opening, params={"panel_cfg": _PANEL})
        grasped = ObsTerm(func=mdp.paddle_grasped, params={"panel_cfg": _PANEL})
        """Jaws closed on the target paddle (Isaac Lab Mimic subtask signal)."""

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class ValvePanelTerminationsCfg(ROVSingleArmTerminationsCfg):
    success = DoneTerm(func=mdp.target_valve_opened, params={"panel_cfg": _PANEL, "min_opening": 0.9})
    wrong_valve = DoneTerm(func=mdp.wrong_valve_turned, params={"panel_cfg": _PANEL, "max_opening": 0.25})


@configclass
class ValvePanelEnvCfg(ROVSingleArmTaskEnvCfg):
    """Shipwreck work site."""

    scene: ValvePanelSceneCfg = ValvePanelSceneCfg(num_envs=1, env_spacing=0.0)
    observations: ValvePanelObservationsCfg = ValvePanelObservationsCfg()
    events: ValvePanelEventCfg = ValvePanelEventCfg()
    terminations: ValvePanelTerminationsCfg = ValvePanelTerminationsCfg()

    task_description: str = "Turn the target valve on the subsea panel a quarter turn to open it."
    arm_teleop_spec: ArmTeleopSpec = BRAVO7_TELEOP_SPEC

    work_site: str = "shipwreck_site"
    fix_rov_base: bool = True
    """Fix the hull while the arm works (the 2026 fixed-base protocol); False = free-floating with hydrodynamics."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.decimation = 2  # 30 Hz control on 60 Hz physics
        self.episode_length_s = 40.0
        self.viewer.eye = (-4.6, -2.2, 2.1)
        self.viewer.lookat = (PANEL_POS[0], PANEL_POS[1], PANEL_POS[2] - 0.1)
        self.scene.site = work_site_cfg(self.work_site)
        self.scene.dvl.mesh_prim_paths = [SEABED_MESH[self.work_site]]
        self.observations.policy.depth.params["water_surface_z"] = SITE_DEPTH_M[self.work_site]
        self.scene.ee_frame.visualizer_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        if self.fix_rov_base:
            self.scene.robot.spawn.fix_root_link = True
            self.events.apply_hydro = None


@configclass
class ValvePanelLedgeEnvCfg(ValvePanelEnvCfg):
    """Portuguese Ledge work site (same layout, CC0 bathymetry and mud)."""

    work_site: str = "portuguese_ledge"


@configclass
class ValvePanelAbsPoseEnvCfg(ValvePanelEnvCfg):
    """Shipwreck site with absolute end-effector pose actions, for learned policies (14-D action)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        spec = self.arm_teleop_spec
        self.actions.arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=list(spec.ik_joint_names),
            body_name=spec.ee_body_name,
            controller=DifferentialIKControllerCfg(command_type="pose", ik_method="dls", use_relative_mode=False),
        )
        self.actions.gripper_action = mdp.ROVParallelGripperActionCfg(
            asset_name="robot", joint_names=list(spec.gripper_joint_names), mode="absolute")

    def use_teleop_device(self, teleop_device) -> None:  # noqa: ANN001
        raise RuntimeError("ValvePanelAbsPoseEnvCfg takes policy actions; use ROVLab-I90Bravo7-ValvePanel-v0 for teleop")
