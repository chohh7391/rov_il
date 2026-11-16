"""Render report figures: third-person view of the expert episode + onboard underwater camera views."""
import argparse
from isaaclab.app import add_launcher_args, launch_simulation
parser = argparse.ArgumentParser(); add_launcher_args(parser); args = parser.parse_args(); args.enable_cameras = True
import os, sys
import numpy as np
import isaaclab.sim as sim_utils
sys.path.insert(0, os.path.dirname(__file__))
out = "/home/home/rov_il/runs/kriso_2026/figs"; os.makedirs(out, exist_ok=True)
sim_cfg = sim_utils.SimulationCfg(dt=1 / 60, device="cuda:0")
with launch_simulation(sim_cfg, args):
    from PIL import Image
    import uw_env as E, uw_scene as U
    from uw_expert import UWExpert
    env = E.UWPickEnv(sim_cfg, seed=5, debug_cam=True)
    env.reset(target=1)
    ex = UWExpert(env); ex.calibrate()
    cam = env.scene["debug"]
    obs = env.reset(target=1); ex.reset()
    shots = {}
    want = {"approach": 40, "reach": 8, "close": 10, "lift": 45}
    while True:
        a = ex.act()
        obs, succ, done = env.step(a)
        ph = ex.phase
        if ph in want and ex.pk == want[ph] and ph not in shots:
            shots[ph] = (U.cam_images(cam)[0].cpu().numpy(), obs["front"], obs["wrist"])
            Image.fromarray(shots[ph][0]).save(f"{out}/third_{ph}.png")
            Image.fromarray(obs["front"]).save(f"{out}/front_{ph}.png")
            Image.fromarray(obs["wrist"]).save(f"{out}/wrist_{ph}.png")
            print("[fig] shot", ph, env.k, flush=True)
        if done:
            if "lift" not in shots:
                shots["lift"] = (U.cam_images(cam)[0].cpu().numpy(), obs["front"], obs["wrist"])
                Image.fromarray(shots["lift"][0]).save(f"{out}/third_lift.png")
                Image.fromarray(obs["wrist"]).save(f"{out}/wrist_lift.png")
                Image.fromarray(obs["front"]).save(f"{out}/front_lift.png")
            break
    print("[fig] success", succ, "steps", env.k, flush=True)
