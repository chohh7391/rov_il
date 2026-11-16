"""Record Isaac Lab Mimic source demonstrations for the valve-panel task with the scripted expert.

Writes an Isaac Lab HDF5 dataset (initial state, actions, states, observations) through the env's recorder manager,
exporting successful episodes only, exactly like ``IsaacLab/scripts/tools/record_demos.py`` does for teleoperation.
The expert drives the Mimic env (``ROVLab-I90Bravo7-ValvePanel-Mimic-v0``, absolute jaw-pose actions) on the Mimic
target valve. Next steps: ``annotate_demos.py --auto`` and ``generate_mimic.py``.

    PYTHONPATH=rov_lab/source OMNI_KIT_ACCEPT_EULA=YES <isaaclab-3.0-python> \\
        experiments/kriso_2026/valve/record_mimic_src.py --num_demos 10 --output_file datasets/kriso_2026/valve_mimic/source.hdf5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", type=str, default="ROVLab-I90Bravo7-ValvePanel-Mimic-v0")
parser.add_argument("--num_demos", type=int, default=10)
parser.add_argument("--max_attempts", type=int, default=20)
parser.add_argument("--max_steps", type=int, default=600)
parser.add_argument("--num_success_steps", type=int, default=5)
parser.add_argument("--output_file", type=Path, default=Path("datasets/kriso_2026/valve_mimic/source.hdf5"))
add_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

device: str = getattr(args, "device", None) or "cuda:0"
with launch_simulation(sim_utils.SimulationCfg(device=device), args):
    import gymnasium as gym
    import torch
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab.envs.utils._manual_success import _prepare_success_term
    from isaaclab.managers import DatasetExportMode
    from isaaclab_tasks.utils import parse_env_cfg

    import rov_lab.tasks  # noqa: F401
    from rov_lab.tasks.valve_panel.expert import ValveExpert

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=1)
    env_cfg.env_name = args.task
    success_term = _prepare_success_term(env_cfg)
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = str(args.output_file.parent.resolve())
    env_cfg.recorders.dataset_filename = args.output_file.stem
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    expert = ValveExpert(env)

    def new_episode() -> None:
        env.sim.reset()
        env.recorder_manager.reset()
        env.reset()
        expert.reset()

    new_episode()
    attempts = 0
    with torch.inference_mode():
        while env.recorder_manager.exported_successful_episode_count < args.num_demos and attempts < args.max_attempts:
            attempts += 1
            ok_steps, recorded = 0, False
            for _ in range(args.max_steps):
                expert.act()
                action = torch.cat([torch.zeros(1, 6, device=env.device), expert.label_pose_b,
                                    expert.label_grip[:, None]], dim=-1)
                env.step(action)
                if bool(success_term.func(env, **success_term.params)[0]):
                    ok_steps += 1
                    if ok_steps >= args.num_success_steps:
                        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                        env.recorder_manager.set_success_to_episodes(
                            [0], torch.tensor([[True]], dtype=torch.bool, device=env.device))
                        env.recorder_manager.export_episodes([0])
                        recorded = True
                        break
                else:
                    ok_steps = 0
            print(f"[mimic-src] attempt {attempts}: success={recorded} "
                  f"exported={env.recorder_manager.exported_successful_episode_count}", flush=True)
            new_episode()
    print(f"[mimic-src] DONE {env.recorder_manager.exported_successful_episode_count} demos in {attempts} attempts "
          f"-> {args.output_file}", flush=True)
    env.close()
