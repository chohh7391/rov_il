"""Run the scripted expert for a few episodes and log progress (Isaac Lab 3.0)."""
import argparse
from isaaclab.app import add_launcher_args, launch_simulation
parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--out", default="/home/home/rov_il/runs/kriso_2026/test_expert")
parser.add_argument("--probe", action="store_true")
add_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
import os, sys, time
import numpy as np
import isaaclab.sim as sim_utils
sys.path.insert(0, os.path.dirname(__file__))
sim_cfg = sim_utils.SimulationCfg(dt=1 / 60, device="cuda:0")
with launch_simulation(sim_cfg, args):
    from PIL import Image
    import uw_env as E
    from uw_expert import UWExpert
    os.makedirs(args.out, exist_ok=True)
    env = E.UWPickEnv(sim_cfg, seed=1, probe=args.probe)
    env.reset(target=0)
    ex = UWExpert(env)
    ex.calibrate()
    if args.probe:
        for _ in range(240):
            env._physics_step()
        print("[t] probe settled z =", env.scene["probe"].data.root_pos_w.torch[0].cpu().numpy().round(3), flush=True)
    for ep in range(args.episodes):
        obs = env.reset()
        ex.reset()
        frames, t0 = [], time.time()
        while True:
            a = ex.act()
            last_a = a
            obs, succ, done = env.step(a)
            if env.k % 10 == 0:
                rp, _ = env.root_pose()
                import isaaclab.utils.math as mu
                print(f"[t] ep{ep} k={env.k:3d} {ex.phase:8s} err={ex.err:.3f} rov={rp.cpu().numpy().round(2)} q={env.joint_pos().cpu().numpy().round(2)} q_cmd={last_a[4:].round(2)}", flush=True)
            if env.k % 15 == 0:
                frames.append(np.concatenate([obs["front"], obs["wrist"][30:210]], axis=1))
            if done:
                break
        print(f"[t] ep{ep} target={env.target} success={succ} steps={env.k} phase={ex.phase} ({time.time()-t0:.1f}s)", flush=True)
        Image.fromarray(np.concatenate(frames[-10:], axis=0)).save(f"{args.out}/ep{ep}.png")
