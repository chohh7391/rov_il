"""Grid-search the fixed-base grasp station (hull pose relative to the target cube) for the scripted expert.

Runs the expert with a locked hull at each candidate offset (no station-keeping error) for every target cube and
reports the success rate per offset. One Isaac Sim session for the whole grid.
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--dx", type=float, nargs="+", default=[-0.41, -0.39, -0.37, -0.35, -0.33])
parser.add_argument("--dy", type=float, nargs="+", default=[0.02])
parser.add_argument("--dz", type=float, nargs="+", default=[-0.03, -0.015, 0.0])
parser.add_argument("--trials", type=int, default=2)
parser.add_argument("--seed", type=int, default=5)
add_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

import itertools
import json
import os
import sys

import numpy as np

import isaaclab.sim as sim_utils

sys.path.insert(0, os.path.dirname(__file__))
sim_cfg = sim_utils.SimulationCfg(dt=1 / 60, device="cuda:0")


def run_episode(env, ex, target: int) -> bool:
    env.reset(target=target)
    ex.reset()
    while True:
        _, succ, done = env.step(ex.act())
        if done:
            return bool(succ)
        if ex.phase == "lift" and ex.pk > 30:
            rise = (env.cube_pos()[2] - env.spawn_z[env.target]).item()
            if rise < 0.02 or ex.pk > 110:
                return False


with launch_simulation(sim_cfg, args):
    import uw_env as E
    from uw_expert import UWExpert

    env = E.UWPickEnv(sim_cfg, seed=args.seed, fixed_base=True, station_jitter=(0.0, 0.0, 0.0, 0.0), episode_s=15.0)
    env.reset(target=0)
    ex = UWExpert(env)
    ex.calibrate()
    results = []
    for dx, dy, dz in itertools.product(args.dx, args.dy, args.dz):
        env.station_offset = np.array([dx, dy, dz])
        per = [sum(run_episode(env, ex, t) for _ in range(args.trials)) for t in range(3)]
        rate = sum(per) / (3 * args.trials)
        results.append({"offset": [dx, dy, dz], "per_target": per, "rate": rate})
        print(f"[search] offset=({dx:+.3f},{dy:+.3f},{dz:+.3f}) per_target={per} rate={rate:.2f}", flush=True)
    best = max(results, key=lambda r: r["rate"])
    print(f"[search] BEST {best}", flush=True)
    json.dump(results, open("/home/home/rov_il/runs/kriso_2026/station_search.json", "w"), indent=1)
