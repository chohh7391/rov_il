import argparse
from isaaclab.app import add_launcher_args, launch_simulation
parser = argparse.ArgumentParser(); add_launcher_args(parser); args = parser.parse_args(); args.enable_cameras = True
import os, sys
import numpy as np, torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as mu
sys.path.insert(0, os.path.dirname(__file__))
out = "/home/home/rov_il/runs/kriso_2026/debug"; os.makedirs(out, exist_ok=True)
sim_cfg = sim_utils.SimulationCfg(dt=1 / 60, device="cuda:0")
with launch_simulation(sim_cfg, args):
    from PIL import Image
    import uw_env as E, uw_scene as U
    from uw_expert import UWExpert
    env = E.UWPickEnv(sim_cfg, seed=1, debug_cam=True)
    TGT = int(os.environ.get('TGT', '1'))
    env.reset(target=TGT)
    ex = UWExpert(env); ex.calibrate()
    cam = env.scene["debug"]
    obs = env.reset(target=TGT); ex.reset()
    c0 = env.cube_spawn[TGT]
    cam.set_world_poses_from_view(eyes=(c0 + torch.tensor([0.05, -0.45, 0.10], device=env.dev))[None],
                                  targets=(c0 + torch.tensor([-0.05, 0.0, 0.02], device=env.dev))[None])
    shots, last = [], None
    while True:
        a = ex.act()
        obs, succ, done = env.step(a)
        if ex.phase in ("descend", "close") or (ex.phase == "lift" and ex.pk % 8 == 0) or (ex.phase == "reach" and ex.pk % 10 == 0):
            rp, rq = env.root_pose()
            rel = lambda w: mu.quat_apply(mu.quat_conjugate(rq[None]), (w - rp)[None])[0].cpu().numpy().round(3)
            eul = [round(float(x), 3) for x in mu.euler_xyz_from_quat(rq[None])]
            ht = rel(ex.hold_tgt) if ex.hold_tgt is not None else None
            print(f"[d] k={env.k} {ex.phase:8s} pk={ex.pk} err={ex.err:.3f} hold_r={ht} rpy={eul} cube_r={rel(env.cube_pos())} grasp_r={rel(ex.grasp_point())} jaw_r={rel(env.jaw_point())} q={env.joint_pos().cpu().numpy().round(2)} qcmd={a[4:].round(2)}", flush=True)
        if ex.phase != last or (ex.phase in ("descend", "close") and env.k % 6 == 0):
            img = U.as_torch(cam.data.output["rgb"])[0, ..., :3].cpu().numpy()
            shots.append(np.concatenate([img, np.zeros((360, 8, 3), np.uint8)], 1))
            last = ex.phase
        if done or (ex.phase == "lift" and ex.pk > 100):
            break
    rows = [np.concatenate(shots[i:i + 4], axis=1) for i in range(0, len(shots) - len(shots) % 4, 4)]
    Image.fromarray(np.concatenate(rows[:4], axis=0)).save(f"{out}/debug_sheet.png")
    print("[d] success", succ, "shots", len(shots))
