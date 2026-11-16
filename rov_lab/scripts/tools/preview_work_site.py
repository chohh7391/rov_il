"""Render the subsea work sites and check the valve panel physics (Isaac Lab 3.0).

Saves clean RGB and underwater (``uw_render``) images from a few viewpoints and runs a valve breakaway check:
a torque below the static friction must leave the valve closed, a torque above it must open it to the -90 deg stop.
Results go to ``--out`` (``preview_report.json`` + PNGs).

    PYTHONPATH=rov_lab/source OMNI_KIT_ACCEPT_EULA=YES <isaaclab-3.0-python> \\
        rov_lab/scripts/tools/preview_work_site.py --scene shipwreck_site --out runs/work_site/shipwreck_site
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--scene", type=str, default="shipwreck_site")
parser.add_argument("--out", type=Path, default=None)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
add_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
out_dir: Path = args.out or Path("runs/work_site") / args.scene

device: str = getattr(args, "device", None) or "cuda:0"
sim_cfg = sim_utils.SimulationCfg(device=device, dt=1.0 / 120.0)
with launch_simulation(sim_cfg, args):
    import math

    import numpy as np
    import torch
    from PIL import Image

    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils import configclass

    from rov_lab.assets.objects.rov_bop_panel import ROV_BOP_PANEL_CFG, VALVE_FRICTION, VALVE_JOINT_NAMES
    from rov_lab.assets.scenes import work_site_cfg
    from rov_lab.assets.scenes.work_site_layout import PANEL_POS, ROV_STATION_POS
    from rov_lab.assets.sensors import UWCameraCfg, uw_render

    def look_at_xyzw(eye: tuple[float, float, float], target: tuple[float, float, float]) -> tuple[float, ...]:
        """World-convention camera orientation (x forward, z up) looking from ``eye`` at ``target``, xyzw."""
        f = np.asarray(target, float) - np.asarray(eye, float)
        f /= np.linalg.norm(f)
        left = np.cross([0.0, 0.0, 1.0], f)
        left /= np.linalg.norm(left)
        up = np.cross(f, left)
        r = np.stack([f, left, up], axis=1)
        w = math.sqrt(max(1e-12, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2.0
        x = (r[2, 1] - r[1, 2]) / (4 * w)
        y = (r[0, 2] - r[2, 0]) / (4 * w)
        z = (r[1, 0] - r[0, 1]) / (4 * w)
        return (x, y, z, w)

    panel_face = (PANEL_POS[0] - 0.2, PANEL_POS[1], PANEL_POS[2])
    views: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
        "rov_front": ((ROV_STATION_POS[0] + 0.55, ROV_STATION_POS[1], ROV_STATION_POS[2] + 0.1), panel_face),
        "overview": ((-6.5, -4.5, 3.2), (0.0, 0.3, 0.6)),
        "wreck": ((-3.0, -1.0, 2.5), (14.0, 14.0, 1.5)),
        "panel_close": ((PANEL_POS[0] - 0.9, PANEL_POS[1] - 0.25, PANEL_POS[2] + 0.15), panel_face),
    }

    def camera(eye, target) -> UWCameraCfg:
        return UWCameraCfg(
            prim_path="{ENV_REGEX_NS}/cam_%d" % len(cams),
            update_period=0.0,
            height=args.height,
            width=args.width,
            offset=UWCameraCfg.OffsetCfg(pos=eye, rot=look_at_xyzw(eye, target), convention="world"),
            spawn=sim_utils.PinholeCameraCfg(focal_length=14.0, horizontal_aperture=20.955,
                                             clipping_range=(0.05, 200.0)),
        )

    cams: dict[str, UWCameraCfg] = {}
    for name, (eye, tgt) in views.items():
        cams[name] = camera(eye, tgt)

    @configclass
    class PreviewSceneCfg(InteractiveSceneCfg):
        site = work_site_cfg(args.scene)
        panel = ROV_BOP_PANEL_CFG.replace(prim_path="{ENV_REGEX_NS}/Panel")
        sun = AssetBaseCfg(prim_path="/World/sun", spawn=sim_utils.DistantLightCfg(
            intensity=2500.0, angle=8.0, color=(0.85, 0.95, 1.0)))
        dome = AssetBaseCfg(prim_path="/World/dome", spawn=sim_utils.DomeLightCfg(
            intensity=400.0, color=(0.55, 0.75, 0.9)))

    scene_cfg = PreviewSceneCfg(num_envs=1, env_spacing=0.0)
    for name, cfg in cams.items():
        setattr(scene_cfg, f"cam_{name}", cfg)

    sim = sim_utils.SimulationContext(sim_cfg)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    for _ in range(30):
        sim.step(render=True)
        scene.update(sim.get_physics_dt())
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"scene": args.scene, "views": {}}
    for name in views:
        cam = scene[f"cam_{name}"]
        rgb = cam.data.output["rgb"].torch[..., :3]
        dist = cam.data.output["distance_to_camera"].torch
        uw = uw_render(rgb, dist, cam.cfg)
        Image.fromarray(rgb[0].cpu().numpy().astype(np.uint8)).save(out_dir / f"{name}_rgb.png")
        Image.fromarray(uw[0].cpu().numpy()).save(out_dir / f"{name}_uw.png")
        d = dist[0].cpu().numpy()
        report["views"][name] = {"mean_rgb": rgb[0].float().mean((0, 1)).tolist(),
                                 "depth_valid": float(np.isfinite(d).mean()),
                                 "depth_median": float(np.nanmedian(np.where(np.isfinite(d), d, np.nan)))}

    # valve breakaway check on every valve: torque below static friction -> stays shut; above -> opens to the stop
    panel = scene["panel"]
    results = {}
    all_pass = True
    for jname in VALVE_JOINT_NAMES:
        jid = panel.find_joints(jname)[0][0]
        res = {}
        for label, tau in (("below_static", -0.6 * VALVE_FRICTION["static"]),
                           ("above_static", -2.0 * VALVE_FRICTION["static"])):
            panel.write_joint_state_to_sim(torch.zeros_like(panel.data.joint_pos.torch),
                                           torch.zeros_like(panel.data.joint_vel.torch))
            effort = torch.zeros_like(panel.data.joint_pos.torch)
            effort[:, jid] = tau
            for _ in range(240):  # 2 s
                panel.set_joint_effort_target(effort)
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(sim.get_physics_dt())
            res[label] = round(math.degrees(float(panel.data.joint_pos.torch[0, jid])), 2)
        others = panel.data.joint_pos.torch[0].clone()
        others[jid] = 0.0
        res["others_max_abs_deg"] = round(math.degrees(float(others.abs().max())), 3)
        res["pass"] = abs(res["below_static"]) < 1.0 and res["above_static"] < -85.0 and res["others_max_abs_deg"] < 1.0
        all_pass &= res["pass"]
        results[jname] = res
    results["torques_Nm"] = [-0.6 * VALVE_FRICTION["static"], -2.0 * VALVE_FRICTION["static"]]
    results["pass"] = all_pass
    report["valve_breakaway"] = results
    report["panel_joint_names"] = list(panel.joint_names)
    report["panel_body_names"] = list(panel.body_names)
    (out_dir / "preview_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
