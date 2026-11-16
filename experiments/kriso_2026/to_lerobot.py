"""Convert recorded npz episodes into a LeRobot v3 dataset (run in the `fire` conda env, lerobot 0.4.4).

usage: python to_lerobot.py --src runs/kriso_2026/demos --root datasets/lerobot_v3/kriso_pick --fps 30
"""

import argparse
import glob
import os
import shutil

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

STATE_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper",
               "dvl_vx", "dvl_vy", "dvl_vz", "depth"]
# Target-object conditioning: ACT has no language encoder, so the instruction ("pick up the <color> cube")
# is given as a one-hot over the three target cubes (red, green, blue) in observation.environment_state.
TARGET_NAMES = ["target_red", "target_green", "target_blue"]
ACTION_NAMES = ["rov_vx", "rov_vy", "rov_vz", "rov_yaw_rate",
                "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--root", required=True)
ap.add_argument("--repo_id", default="rov_il/kriso_pick")
ap.add_argument("--fps", type=int, default=30)
args = ap.parse_args()

files = sorted(glob.glob(os.path.join(args.src, "ep_*.npz")))
assert files, "no episodes"
first = np.load(files[0])
# 6-D actions come from fixed-base (station-kept hull) collection: arm joint targets only
action_names = ACTION_NAMES if first["action"].shape[-1] == len(ACTION_NAMES) else ACTION_NAMES[4:]
assert first["action"].shape[-1] == len(action_names), f"unexpected action dim {first['action'].shape}"
features = {
    "observation.images.front": {"dtype": "image", "shape": first["front"].shape[1:], "names": ["height", "width", "channels"]},
    "observation.images.wrist": {"dtype": "image", "shape": first["wrist"].shape[1:], "names": ["height", "width", "channels"]},
    "observation.state": {"dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES},
    "observation.environment_state": {"dtype": "float32", "shape": (len(TARGET_NAMES),), "names": TARGET_NAMES},
    "action": {"dtype": "float32", "shape": (len(action_names),), "names": action_names},
}
if os.path.exists(args.root):
    shutil.rmtree(args.root)
ds = LeRobotDataset.create(repo_id=args.repo_id, fps=args.fps, root=args.root, features=features,
                           robot_type="bluerov_so101", use_videos=False,
                           image_writer_processes=4, image_writer_threads=4)
n_frames = 0
for f in files:
    z = np.load(f)
    # materialize each array once: indexing an NpzFile member re-reads and decompresses it on every access
    d = {k: z[k] for k in ("front", "wrist", "state", "action", "task", "target")}
    task = str(d["task"])
    onehot = np.eye(len(TARGET_NAMES), dtype=np.float32)[int(d["target"])]
    for t in range(len(d["action"])):
        ds.add_frame({
            "observation.images.front": d["front"][t],
            "observation.images.wrist": d["wrist"][t],
            "observation.state": d["state"][t].astype(np.float32),
            "observation.environment_state": onehot,
            "action": d["action"][t].astype(np.float32),
            "task": task,
        })
    ds.save_episode()
    n_frames += len(d["action"])
ds.finalize()
print(f"converted {len(files)} episodes, {n_frames} frames -> {args.root}")
