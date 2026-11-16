"""Convert valve-panel npz demos into a LeRobot v3 dataset (run in the ``fire`` conda env, lerobot 0.4.4).

    python experiments/kriso_2026/valve/to_lerobot_valve.py --src datasets/kriso_2026/valve_demos \\
        --root datasets/kriso_2026/lerobot_v3/kriso_valve --fps 30
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import valve_io as V  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--root", required=True)
ap.add_argument("--repo_id", default="rov_il/kriso_valve")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--max_episodes", type=int, default=None)
args = ap.parse_args()

files = sorted(glob.glob(os.path.join(args.src, "ep_*.npz")))[: args.max_episodes]
assert files, "no episodes"
first = np.load(files[0])
assert first["state"].shape[-1] == len(V.STATE_NAMES) and first["action"].shape[-1] == len(V.ACTION_NAMES)
features = {
    "observation.images.front": {"dtype": "image", "shape": first["front"].shape[1:], "names": ["height", "width", "channels"]},
    "observation.images.wrist": {"dtype": "image", "shape": first["wrist"].shape[1:], "names": ["height", "width", "channels"]},
    "observation.state": {"dtype": "float32", "shape": (len(V.STATE_NAMES),), "names": V.STATE_NAMES},
    "observation.environment_state": {"dtype": "float32", "shape": (len(V.TARGET_NAMES),), "names": V.TARGET_NAMES},
    "action": {"dtype": "float32", "shape": (len(V.ACTION_NAMES),), "names": V.ACTION_NAMES},
}
if os.path.exists(args.root):
    shutil.rmtree(args.root)
ds = LeRobotDataset.create(repo_id=args.repo_id, fps=args.fps, root=args.root, features=features,
                           robot_type="i90_bravo7", use_videos=False, image_writer_processes=4, image_writer_threads=4)
n_frames, per_target = 0, [0] * len(V.TARGET_NAMES)
for f in files:
    z = np.load(f)
    d = {k: z[k] for k in ("front", "wrist", "state", "action", "task", "target")}  # decompress each array once
    onehot = np.eye(len(V.TARGET_NAMES), dtype=np.float32)[int(d["target"])]
    for t in range(len(d["action"])):
        ds.add_frame({
            "observation.images.front": d["front"][t],
            "observation.images.wrist": d["wrist"][t],
            "observation.state": d["state"][t].astype(np.float32),
            "observation.environment_state": onehot,
            "action": d["action"][t].astype(np.float32),
            "task": str(d["task"]),
        })
    ds.save_episode()
    n_frames += len(d["action"])
    per_target[int(d["target"])] += 1
ds.finalize()
print(f"converted {len(files)} episodes ({per_target} per valve), {n_frames} frames -> {args.root}")
