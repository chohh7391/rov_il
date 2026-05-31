import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

from rov_lab.assets.scenes.under_water import (
    OCEAN_SIM_ASSETS_PATH,
    UNDER_WATER_WITH_MHL_CFG,
)

from ..template import (
    ROVSingleArmObservationsCfg,
    ROVSingleArmTaskEnvCfg,
    ROVSingleArmTaskSceneCfg,
    ROVSingleArmTerminationsCfg,
)
from . import mdp


@configclass
class PickStoneSceneCfg(ROVSingleArmTaskSceneCfg):
    """Scene configuration for the pick stone task."""

    scene: AssetBaseCfg = UNDER_WATER_WITH_MHL_CFG.replace(prim_path="/World/Scene")

    rock: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/rock",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.0, 0.1, -1.5),
            rot=(0.205415, -0.502620, -0.743210, 0.390915)
        ),
        spawn=UsdFileCfg(
            usd_path=str(OCEAN_SIM_ASSETS_PATH / "collected_rock" / "rock.usd"),
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
class ObservationsCfg(ROVSingleArmObservationsCfg):

    @configclass
    class DummyCfg(ObsGroup):
        """Observations for subtask group."""
        dummy = ObsTerm(func=mdp.dummy_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: DummyCfg = DummyCfg()


@configclass
class TerminationsCfg(ROVSingleArmTerminationsCfg):

    success = DoneTerm(
        func=mdp.dummy_done
    )



@configclass
class PickStoneEnvCfg(ROVSingleArmTaskEnvCfg):
    """Configuration for the pick stone environment."""

    scene: PickStoneSceneCfg = PickStoneSceneCfg(env_spacing=8.0)

    observations: ObservationsCfg = ObservationsCfg()

    terminations: TerminationsCfg = TerminationsCfg()

    task_description: str = "Pick three stones and put them into the plate, then reset the arm to rest state."

    def __post_init__(self) -> None:
        super().__post_init__()
