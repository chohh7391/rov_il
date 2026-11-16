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
    from uw_expert import UWExpert
    env = E.UWPickEnv(sim_cfg, seed=1, rov_jitter=(0, 0, 0, 0), cube_jitter=0)
    env.reset(target=1)
    ex = UWExpert(env); ex.calibrate()
    rp, rq = env.root_pose()
    for tgt_r in [(0.38, 0.0, 0.0), (0.38, 0.0, 0.07), (0.33, 0.0, -0.03), (0.42, 0.03, 0.05)]:
        tw = rp + mu.quat_apply(rq[None], torch.tensor([tgt_r], device=env.dev))[0]
        q = ex._ik(tw).cpu().numpy()
        env.set_arm(list(q) + [0.0], n=120)
        gp = ex.grasp_point()
        rp2, rq2 = env.root_pose()
        g_r = mu.quat_apply(mu.quat_conjugate(rq2[None]), (gp - rp2)[None])[0].cpu().numpy()
        gq = env.robot.data.body_quat_w.torch[0, env.grip_body_id]
        pointing = mu.quat_apply(gq[None], ex.g_off / torch.linalg.vector_norm(ex.g_off))[0].cpu().numpy()
        print(f"[ik] tgt={tgt_r} q_cmd={q.round(3)} q_act={env.joint_pos().cpu().numpy()[:5].round(3)} grasp_r={g_r.round(3)} pointing={pointing.round(2)} rov_moved={(rp2-rp).cpu().numpy().round(3)}", flush=True)
