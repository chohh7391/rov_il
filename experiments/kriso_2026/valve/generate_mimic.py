"""Generate valve-panel demonstrations in parallel with Isaac Lab Mimic, and render the parallel envs as a tiled image.

Same flow as ``IsaacLab/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py`` (``setup_env_config`` ->
``setup_async_generation`` -> ``env_loop``), plus:
  * snapshots of every env's front underwater camera at ``--grid_steps`` -> ``parallel_grid*.png`` (tiled)
  * ``summary.json``: num_envs, source demos, generation attempts / successes, wall time

    PYTHONPATH=rov_lab/source OMNI_KIT_ACCEPT_EULA=YES <isaaclab-3.0-python> experiments/kriso_2026/valve/generate_mimic.py \\
        --num_envs 16 --generation_num_trials 128 --input_file datasets/kriso_2026/valve_mimic/annotated.hdf5 \\
        --output_file datasets/kriso_2026/valve_mimic/generated.hdf5 --out runs/kriso_2026/valve_mimic
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import rov_lab.tasks.valve_panel  # noqa: F401  (registers the gym ids; pxr-free, safe before Kit starts)
from isaaclab.app import add_launcher_args, launch_simulation, scan
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", type=str, default="ROVLab-I90Bravo7-ValvePanel-Mimic-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--generation_num_trials", type=int, default=128)
parser.add_argument("--input_file", type=Path, default=Path("datasets/kriso_2026/valve_mimic/annotated.hdf5"))
parser.add_argument("--output_file", type=Path, default=Path("datasets/kriso_2026/valve_mimic/generated.hdf5"))
parser.add_argument("--out", type=Path, default=Path("runs/kriso_2026/valve_mimic"))
parser.add_argument("--grid_steps", type=int, nargs="+", default=[90, 160, 230])
parser.add_argument("--grid_cols", type=int, default=4)
parser.add_argument("--tile_width", type=int, default=480)
add_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

with launch_simulation(parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs), args):
    import asyncio
    import json
    import random

    import gymnasium as gym
    import numpy as np
    import torch
    from PIL import Image

    import isaaclab_mimic.datagen.generation as G
    from isaaclab.utils.datasets import HDF5DatasetFileHandler
    from isaaclab_mimic.datagen.utils import setup_output_paths

    from rov_lab.assets.sensors import uw_render

    output_dir, output_name = setup_output_paths(str(args.output_file))
    env_cfg, success_term = G.setup_env_config(
        env_name=args.task, output_dir=output_dir, output_file_name=output_name, num_envs=args.num_envs,
        device=args.device, generation_num_trials=args.generation_num_trials)
    scan(env_cfg, args)
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    random.seed(env.cfg.datagen_config.seed)
    np.random.seed(env.cfg.datagen_config.seed)
    torch.manual_seed(env.cfg.datagen_config.seed)
    env.reset()

    args.out.mkdir(parents=True, exist_ok=True)
    step_count = {"n": 0}
    grids: list[str] = []
    orig_step = env.step

    def save_grid(tag: str) -> str:
        cam = env.scene["front"]
        rgb = cam.data.output["rgb"].torch[..., :3]
        uw = uw_render(rgb, cam.data.output["distance_to_camera"].torch, cam.cfg).cpu().numpy()
        w = args.tile_width
        h = int(round(w * uw.shape[1] / uw.shape[2]))
        tiles = [np.asarray(Image.fromarray(im).resize((w, h), Image.LANCZOS)) for im in uw]
        cols = args.grid_cols
        rows = (len(tiles) + cols - 1) // cols
        canvas = np.zeros((rows * h + (rows - 1) * 4, cols * w + (cols - 1) * 4, 3), np.uint8)
        canvas[:] = 255
        for k, t in enumerate(tiles):
            r, c = divmod(k, cols)
            canvas[r * (h + 4): r * (h + 4) + h, c * (w + 4): c * (w + 4) + w] = t
        path = args.out / f"parallel_grid_step{step_count['n']:04d}.png"
        Image.fromarray(canvas).save(path)
        return str(path)

    def step_with_snapshots(action):  # noqa: ANN001, ANN202
        out = orig_step(action)
        step_count["n"] += 1
        if step_count["n"] in args.grid_steps:
            grids.append(save_grid(str(step_count["n"])))
        return out

    env.step = step_with_snapshots
    t0 = time.time()
    comps = G.setup_async_generation(env=env, num_envs=args.num_envs, input_file=str(args.input_file),
                                     success_term=success_term, pause_subtask=False, motion_planners=None)
    tasks = asyncio.ensure_future(asyncio.gather(*comps["tasks"]))
    try:
        G.env_loop(env, comps["reset_queue"], comps["action_queue"], comps["info_pool"], comps["event_loop"],
                   data_gen_tasks=tasks)
    finally:
        tasks.cancel()
        try:
            comps["event_loop"].run_until_complete(tasks)
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
    wall = time.time() - t0
    src = HDF5DatasetFileHandler()
    src.open(str(args.input_file))
    n_src = src.get_num_episodes()
    gen_path = Path(output_dir) / f"{output_name}.hdf5"
    n_gen = None
    if gen_path.exists():
        g = HDF5DatasetFileHandler()
        g.open(str(gen_path))
        n_gen = g.get_num_episodes()
    if grids:
        import shutil

        shutil.copy(grids[min(1, len(grids) - 1)], args.out / "parallel_grid.png")
    summary = {
        "task": args.task, "framework": "Isaac Lab Mimic (isaaclab_mimic.datagen, Isaac Lab 3.0 / Isaac Sim 6.2)",
        "num_envs": args.num_envs, "env_spacing_m": float(env.cfg.scene.env_spacing),
        "source_demos": int(n_src), "source_file": str(args.input_file),
        "generation_attempts": int(G.num_attempts), "generation_successes": int(G.num_success),
        "generation_failures": int(G.num_failures),
        "generation_success_rate": float(G.num_success / G.num_attempts) if G.num_attempts else None,
        "generated_episodes_in_file": n_gen, "generated_file": str(gen_path),
        "wall_time_s": round(wall, 1), "env_steps": step_count["n"],
        "env_steps_per_s": round(step_count["n"] / wall, 2), "control_hz": round(1.0 / env.step_dt, 1),
        "subtasks": [c.description for c in env.cfg.subtask_configs["bravo"]],
        "target_valve": "valve_1", "grid_images": grids,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("[mimic] SUMMARY", json.dumps(summary), flush=True)
    env.close()
