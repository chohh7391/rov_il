"""Subsea valve-panel intervention task: I90-class ROV + Bravo 7-class arm turning a paddle valve."""

import gymnasium as gym

gym.register(
    id="ROVLab-I90Bravo7-ValvePanel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.valve_panel_env_cfg:ValvePanelEnvCfg"},
)

gym.register(
    id="ROVLab-I90Bravo7-ValvePanel-Ledge-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.valve_panel_env_cfg:ValvePanelLedgeEnvCfg"},
)

gym.register(
    id="ROVLab-I90Bravo7-ValvePanel-AbsPose-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.valve_panel_env_cfg:ValvePanelAbsPoseEnvCfg"},
)

gym.register(
    id="ROVLab-I90Bravo7-ValvePanel-Mimic-v0",
    entry_point="rov_lab.tasks.valve_panel.mimic_env:ValvePanelMimicEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.mimic_env_cfg:ValvePanelMimicEnvCfg"},
)


def register_tasks() -> list[str]:
    """External callback for Isaac Lab's Mimic scripts (``--external_callback rov_lab.tasks.valve_panel.register_tasks``).

    The gym ids are registered by importing this package; the callback consumes no CLI arguments. It lives here, not in
    ``mimic_env``, because the scripts import the callback's module before Kit starts and the Mimic env class pulls in
    ``pxr`` (whose pip build then clashes with Kit's USD libraries)."""
    return []
