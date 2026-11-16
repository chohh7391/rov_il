"""Isaac Lab Mimic configuration for the valve-panel task: parallel envs, each with its own compact work site.

Two subtasks per demonstration, both relative to the target valve (link frame, in the ROV base frame):
  1. reach, approach and close the jaws on the paddle  (ends a few steps after the ``grasped`` signal)
  2. turn the valve open                               (to the end of the demonstration)
The target valve is fixed per Mimic task (``MIMIC_TARGET_VALVE``): reset_to() does not restore the env's target id, so
source demos, annotation and generation must share one target.
"""

from __future__ import annotations

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from rov_lab.assets.scenes import work_site_cfg

from .valve_panel_env_cfg import ValvePanelAbsPoseEnvCfg

MIMIC_TARGET_VALVE: int = 0
"""Valve index (0 = valve_1, a big valve) used for Mimic source demos and generation."""

MIMIC_ENV_SPACING: float = 60.0
"""Env spacing (m); the compact work-site terrain is 50 m wide."""


@configclass
class ValvePanelMimicEnvCfg(ValvePanelAbsPoseEnvCfg, MimicEnvCfg):
    """Absolute-pose valve task with per-env work sites and the Mimic data-generation settings."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.site = work_site_cfg(self.work_site, per_env=True)
        self.scene.env_spacing = MIMIC_ENV_SPACING
        # state-only datasets: the DVL ray caster needs one global seabed mesh, and image terms would be recorded
        # into the HDF5 files. The front camera stays (not observed) for tiled previews of the parallel envs.
        self.scene.dvl = None
        self.scene.wrist = None
        for name in ("dvl_vel", "dvl_depth", "dvl_beam_hit", "wrist", "front"):
            setattr(self.observations.policy, name, None)
        self.events.reset_target.params["candidates"] = (MIMIC_TARGET_VALVE,)

        self.datagen_config.name = "valve_panel_i90_bravo7"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 100
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 300
        self.datagen_config.seed = 1
        self.subtask_configs["bravo"] = [
            SubTaskConfig(
                object_ref="target_valve",
                subtask_term_signal="grasped",
                subtask_term_offset_range=(6, 10),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.002,
                num_interpolation_steps=8,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Reach the target valve, approach the paddle and close the jaws on it",
            ),
            SubTaskConfig(
                object_ref="target_valve",
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.001,
                num_interpolation_steps=3,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Turn the valve a quarter turn",
            ),
        ]
