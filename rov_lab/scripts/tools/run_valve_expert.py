"""Run the scripted valve-panel expert and report per-valve success (layout / task feasibility check).

    PYTHONPATH=rov_lab/source OMNI_KIT_ACCEPT_EULA=YES <isaaclab-3.0-python> rov_lab/scripts/tools/run_valve_expert.py \\
        --task ROVLab-I90Bravo7-ValvePanel-v0 --episodes 10 --targets 0 1 2 3 4 [--cameras --out runs/valve_expert]

Targets cycle through ``--targets`` (valve indices 0-4). Without ``--cameras`` the cameras and DVL are dropped for
speed. Writes ``<out>/expert_report.json`` (and phase snapshots of the first episode with ``--cameras``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", type=str, default="ROVLab-I90Bravo7-ValvePanel-v0")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--targets", type=int, nargs="+", default=[0, 1, 2, 3, 4])
parser.add_argument("--cameras", action="store_true")
parser.add_argument("--abs", action="store_true",
                    help="drive the AbsPose env with the expert's absolute labels (checks the IL action representation)")
parser.add_argument("--out", type=Path, default=Path("runs/valve_expert"))
add_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = args.cameras

device: str = getattr(args, "device", None) or "cuda:0"
with launch_simulation(sim_utils.SimulationCfg(device=device), args):
    import math

    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    import rov_lab.tasks  # noqa: F401
    from rov_lab.tasks.valve_panel.expert import PHASES, ValveExpert
    from rov_lab.tasks.valve_panel.mdp.valve import TARGET_ATTR

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=1)
    if not args.abs:
        env_cfg.use_teleop_device("keyboard")
    from isaaclab.managers import RecorderManagerBaseCfg

    env_cfg.recorders = RecorderManagerBaseCfg()  # no in-memory episode recording (camera frames fill RAM)
    if not args.cameras:
        for name in ("wrist", "front", "dvl"):
            setattr(env_cfg.scene, name, None)
        for name in ("wrist", "front", "dvl_vel", "dvl_depth", "dvl_beam_hit"):
            setattr(env_cfg.observations.policy, name, None)
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    expert = ValveExpert(env)
    args.out.mkdir(parents=True, exist_ok=True)

    def set_target(k: int) -> None:
        getattr(env, TARGET_ATTR)[:] = k
        expert.reset()

    env.reset()
    episodes: list[dict] = []
    ep, target = 0, args.targets[0]
    set_target(target)
    steps, phase_log, min_err, snaps, trace = 0, [], {"tcp_err": 9.0, "rot_err": 9.0}, set(), []
    panel = env.scene["panel"]
    last_obs = None
    with torch.inference_mode():
        while ep < args.episodes:
            action = expert.act()
            if args.abs:
                action = torch.cat([torch.zeros(env.num_envs, 6, device=env.device), expert.label_pose_b,
                                    expert.label_grip[:, None]], dim=-1)
            ph = PHASES[int(expert.phase[0])]
            if not phase_log or phase_log[-1][0] != ph:
                phase_log.append((ph, steps))
                if args.cameras and ep == 0 and ph not in snaps:
                    from PIL import Image

                    obs = env.observation_manager.compute()
                    for key in ("front", "wrist"):
                        Image.fromarray(obs["policy"][key][0].cpu().numpy()).save(args.out / f"ep0_{steps:04d}_{ph}_{key}.png")
                    snaps.add(ph)
            for k in min_err:
                min_err[k] = min(min_err[k], float(expert.last_errors[k][0]))
            if ep == 0 and steps % 25 == 0:
                q_t = float(panel.data.joint_pos.torch[0, expert.valve_joint_ids[target]])
                trace.append((steps, ph, round(float(expert.last_errors["tcp_err"][0]), 4),
                              round(float(expert.last_errors["rot_err"][0]), 4), round(math.degrees(q_t), 1)))
            openings_before = (-panel.data.joint_pos.torch[0] / (0.5 * math.pi)).clamp(0, 1).tolist()
            obs, _, terminated, truncated, _ = env.step(action)
            steps += 1
            if bool(terminated[0]) or bool(truncated[0]):
                if args.cameras and ep == 0 and last_obs is not None:  # the returned obs is already post-reset
                    from PIL import Image

                    for key in ("front", "wrist"):
                        Image.fromarray(last_obs["policy"][key][0].cpu().numpy()).save(args.out / f"ep0_{steps:04d}_end_{key}.png")
                tm = env.termination_manager
                result = {name: bool(tm.get_term(name)[0]) for name in tm.active_terms}
                episodes.append({
                    "episode": ep, "target": target, "steps": steps, "seconds": round(steps * env.step_dt, 2),
                    "result": result, "success": result.get("success", False),
                    "phases": phase_log, "min_tcp_err_m": round(min_err["tcp_err"], 4),
                    "min_rot_err_rad": round(min_err["rot_err"], 4),
                    "valve_opening_before_end": [round(o, 3) for o in openings_before],
                })
                print(f"[expert] ep {ep} valve {target}: {result} in {steps} steps, phases {phase_log}", flush=True)
                ep += 1
                target = args.targets[ep % len(args.targets)]
                set_target(target)
                steps, phase_log, min_err = 0, [], {"tcp_err": 9.0, "rot_err": 9.0}
            last_obs = obs if args.cameras and ep == 0 else None

    per_valve: dict[str, dict] = {}
    for e in episodes:
        d = per_valve.setdefault(f"valve_{e['target'] + 1}", {"n": 0, "success": 0, "seconds": []})
        d["n"] += 1
        d["success"] += int(e["success"])
        d["seconds"].append(e["seconds"])
    report = {"task": args.task, "episodes": len(episodes), "ep0_trace(step,phase,tcp_err,rot_err,valve_deg)": trace,
              "success_rate": sum(e["success"] for e in episodes) / max(1, len(episodes)),
              "per_valve": per_valve, "episodes_detail": episodes}
    (args.out / "expert_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("task", "episodes", "success_rate", "per_valve")}, indent=1))
    env.close()
