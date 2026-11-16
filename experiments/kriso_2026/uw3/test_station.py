import argparse
from isaaclab.app import add_launcher_args, launch_simulation
parser = argparse.ArgumentParser(); add_launcher_args(parser); args = parser.parse_args(); args.enable_cameras = True
import os, sys
import numpy as np, torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as mu
sys.path.insert(0, os.path.dirname(__file__))
sim_cfg = sim_utils.SimulationCfg(dt=1 / 60, device="cuda:0")
with launch_simulation(sim_cfg, args):
    import uw_env as E
    env = E.UWPickEnv(sim_cfg, seed=1, rov_jitter=(0, 0, 0, 0), cube_jitter=0)
    env.reset(target=1)
    rp, rq = env.root_pose()
    hold = rp.clone()
    print("[s] root body", env.robot.body_names[0], "root_pos", rp.cpu().numpy().round(3), "base body pos", env.robot.data.body_pos_w.torch[0, env.base_id].cpu().numpy().round(3))
    def run(n, arm, tag, cmd_override=None):
        for k in range(n):
            rp, rq = env.root_pose()
            e_b = mu.quat_apply(mu.quat_conjugate(rq[None]), (hold - rp)[None])[0].cpu().numpy()
            a = np.zeros(10, np.float32)
            a[0:3] = np.clip(0.8 * e_b / np.array(env.servo.LIN_SCALE), -1, 1) if cmd_override is None else cmd_override
            a[4:] = arm
            env.step(a)
            if k % 20 == 0:
                q, lin, ang = env.servo.body_vel()
                f, t = env.servo.wrench_b(env.cmd)
                print(f"[s] {tag} k={k:3d} pos={rp.cpu().numpy().round(3)} vel_b={lin[0].cpu().numpy().round(3)} cmd={a[0:3].round(3)} f_servo={f[0].cpu().numpy().round(1)}", flush=True)
    run(60, list(E.STOW), "stow")
    run(120, [0.0, 1.2, 0.3, 0.0, 0.0, 1.0], "arm-down")
    run(60, list(E.STOW), "stow2")
    run(40, list(E.STOW), "cmd+x", cmd_override=np.array([0.2, 0, 0], np.float32))
    run(40, list(E.STOW), "cmd0", cmd_override=np.array([0, 0, 0], np.float32))
