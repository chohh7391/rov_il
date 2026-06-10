import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer


def resolve_target_color_id(env: ManagerBasedRLEnv | DirectRLEnv) -> torch.Tensor:
    """Return the per-episode target color id, defaulting to zeros before the first reset.

    The reset event (`mdp.reset_target_color`) sets `env.pick_stone_target_color_id`, but the
    observation/termination managers probe their term functions once at load time -- before any
    reset -- so the attribute may not exist yet. Return a zeros `[num_envs]` long tensor in that
    case (used only for shape inference; real values are set on reset).
    """
    target_ids = getattr(env, "pick_stone_target_color_id", None)
    if target_ids is None:
        return torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    return target_ids.to(device=env.device, dtype=torch.long)


def object_grasped(
    env: ManagerBasedRLEnv | DirectRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("red_cube"),
    diff_threshold: float = 0.05,
    grasp_threshold: float = 0.60,
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    rigid_object: RigidObject = env.scene[object_cfg.name]

    object_pos = rigid_object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 1, :]
    pos_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    grasped = torch.logical_and(pos_diff < diff_threshold, robot.data.joint_pos[:, -1] < grasp_threshold)

    return grasped


def target_cube_grasped(
    env: ManagerBasedRLEnv | DirectRLEnv,
    cube_cfgs: list[SceneEntityCfg],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    diff_threshold: float = 0.05,
    grasp_threshold: float = 0.60,
) -> torch.Tensor:
    """Check whether the reset-selected target cube is grasped."""
    grasped = torch.stack(
        [
            object_grasped(
                env,
                robot_cfg=robot_cfg,
                ee_frame_cfg=ee_frame_cfg,
                object_cfg=cube_cfg,
                diff_threshold=diff_threshold,
                grasp_threshold=grasp_threshold,
            )
            for cube_cfg in cube_cfgs
        ],
        dim=1,
    )
    target_ids = resolve_target_color_id(env)
    return torch.gather(grasped, dim=1, index=target_ids.unsqueeze(1)).squeeze(1)
