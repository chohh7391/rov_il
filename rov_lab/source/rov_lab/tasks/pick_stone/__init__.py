import gymnasium as gym

gym.register(
    id="ROVLab-SO101-PickStone-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_stone_env_cfg:PickStoneEnvCfg",
    },
)
