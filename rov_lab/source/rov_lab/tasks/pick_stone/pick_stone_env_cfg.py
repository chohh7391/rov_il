import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

from rov_lab.assets.scenes.under_water import UNDER_WATER_WITH_MHL_CFG
from rov_lab.utils.paths import OBJECT_DIR

from ..template import (
    ROVSingleArmObservationsCfg,
    SingleArmEventCfg,
    ROVSingleArmTaskEnvCfg,
    ROVSingleArmTaskSceneCfg,
    ROVSingleArmTerminationsCfg,
)
from . import mdp
from .mdp.recorders_cfg import PickStoneRecorderManagerCfg
from .mdp.task_metadata import CUBE_ASSET_NAMES, TARGET_INSTRUCTIONS


def _colored_cube_cfg(
    name: str,
    position: tuple[float, float, float],
    color: tuple[float, float, float],
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=position, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.03, 0.03, 0.03),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                linear_damping=10.0,
                angular_damping=10.0,
                max_linear_velocity=5.0,
                max_angular_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
    )


def _cube_cfgs() -> list[SceneEntityCfg]:
    return [SceneEntityCfg(name) for name in CUBE_ASSET_NAMES]


@configclass
class PickStoneSceneCfg(ROVSingleArmTaskSceneCfg):
    """Scene configuration for the pick stone task."""

    scene: AssetBaseCfg = UNDER_WATER_WITH_MHL_CFG.replace(prim_path="/World/Scene")

    red_cube: RigidObjectCfg = _colored_cube_cfg(CUBE_ASSET_NAMES[0], (1.0, -0.05, -1.5), (0.9, 0.05, 0.03))
    green_cube: RigidObjectCfg = _colored_cube_cfg(CUBE_ASSET_NAMES[1], (1.0, 0.05, -1.5), (0.05, 0.75, 0.10))
    blue_cube: RigidObjectCfg = _colored_cube_cfg(CUBE_ASSET_NAMES[2], (1.0, 0.15, -1.5), (0.05, 0.18, 0.9))

    # Decorative rock placed behind the cubes (farther along +x, away from the ROV) purely as a
    # sonar target for inspecting sonar returns. It is intentionally NOT referenced by the
    # observations, subtask terms, or the success condition. reset_scene_to_default restores this
    # pose every episode.
    rock: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/rock",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.5, 0.05, -1.5),
            rot=(0.205415, -0.502620, -0.743210, 0.390915),
        ),
        spawn=UsdFileCfg(
            usd_path=str(OBJECT_DIR / "collected_rock" / "rock.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                linear_damping=10.0,
                angular_damping=10.0,
                max_linear_velocity=5.0,
                max_angular_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
    )


@configclass
class EventCfg(SingleArmEventCfg):
    """Event configuration for the pick stone task."""

    reset_target = EventTerm(func=mdp.reset_target_color, mode="reset")


@configclass
class ObservationsCfg(ROVSingleArmObservationsCfg):

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        red_cube_grasped = ObsTerm(
            func=mdp.object_grasped, params={"object_cfg": SceneEntityCfg(CUBE_ASSET_NAMES[0])}
        )
        green_cube_grasped = ObsTerm(
            func=mdp.object_grasped, params={"object_cfg": SceneEntityCfg(CUBE_ASSET_NAMES[1])}
        )
        blue_cube_grasped = ObsTerm(
            func=mdp.object_grasped, params={"object_cfg": SceneEntityCfg(CUBE_ASSET_NAMES[2])}
        )
        target_cube_grasped = ObsTerm(func=mdp.target_cube_grasped, params={"cube_cfgs": _cube_cfgs()})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg(ROVSingleArmTerminationsCfg):

    success = DoneTerm(
        func=mdp.target_cube_grasped_and_lifted,
        params={"cube_cfgs": _cube_cfgs(), "min_lift": 0.10},
    )



@configclass
class PickStoneEnvCfg(ROVSingleArmTaskEnvCfg):
    """Configuration for the pick stone environment."""

    scene: PickStoneSceneCfg = PickStoneSceneCfg(env_spacing=8.0)

    observations: ObservationsCfg = ObservationsCfg()

    events: EventCfg = EventCfg()

    terminations: TerminationsCfg = TerminationsCfg()

    recorders: PickStoneRecorderManagerCfg = PickStoneRecorderManagerCfg()

    task_description: str = "Pick up the target colored cube and lift it off the seabed."

    # Per-color natural-language instructions, indexed by target_color_id. The template's
    # build_lerobot_frame() reads this (via getattr) to emit color-specific tasks on the
    # in-recording LeRobot route; the offline converter resolves instructions independently.
    task_instructions: tuple[str, ...] = TARGET_INSTRUCTIONS

    def __post_init__(self) -> None:
        super().__post_init__()
