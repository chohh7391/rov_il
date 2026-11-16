"""In-simulator known-answer validation of rov_lab's underwater sensors on Isaac Lab 3.0.

Scene: a flat seabed (mesh, top face at z = 0), a kinematic probe body carrying the DVL ray caster, and a downward
camera at a known height. The sensors are the real Isaac Lab 3.0 sensors used by rov_lab (``DVLCfg``,
``UWCameraCfg``), so this checks what the unit tests cannot: mounting, ray pattern orientation, frames and the
renderer's depth semantics.

Checks:
  * DVL: at several probe heights and attitudes (level, rolled, pitched, yawed), each beam's range equals
    ``h_sensor / (-d_w.z)`` (``d_w`` = beam direction in world), all beams hit; above ``max_range`` all beams miss.
  * Camera: ``distance_to_camera`` over the flat floor equals ``d * sqrt(1 + x^2 + y^2)`` with normalized image
    coordinates from the camera intrinsics (Euclidean distance, as OceanSim's depth input); the underwater image of
    the floor matches the OceanSim model applied to the rendered colour and distance.

usage:
    PYTHONPATH=rov_lab/source OMNI_KIT_ACCEPT_EULA=YES <isaaclab-3.0-python> rov_lab/scripts/tools/validate_sensors_sim.py
"""

from __future__ import annotations

import argparse

import isaaclab.sim as sim_utils
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Known-answer validation of rov_lab underwater sensors")
parser.add_argument("--out", type=str, default="runs/isaaclab3_port/validate_sensors.json")
add_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

DEVICE = "cuda:0"
CAM_HEIGHT = 3.0
DVL_OFFSET = (0.10, -0.05, -0.10)  # sensor position on the probe body (body frame)
DVL_MAX_RANGE = 10.0

with launch_simulation(sim_utils.SimulationCfg(dt=1 / 60, device=DEVICE), args):
    import json
    import math
    import os

    import torch
    import isaaclab.utils.math as mu
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils import configclass
    from pxr import UsdGeom

    from rov_lab.assets.sensors import DVLCfg, UWCameraCfg, uw_render
    from rov_lab.assets.sensors.dvl import DVLJanusPatternCfg, dvl_janus_pattern, dvl_ranges_from_hits, dvl_ray_origins_w

    @configclass
    class ValidationSceneCfg(InteractiveSceneCfg):
        seabed = AssetBaseCfg(
            prim_path="/World/seabed",
            spawn=sim_utils.MeshCuboidCfg(
                size=(60.0, 60.0, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.5, 0.35)),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.5)),
        )
        light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))
        probe = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/probe",
            spawn=sim_utils.CuboidCfg(
                size=(0.3, 0.2, 0.15),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
        )
        dvl = DVLCfg(
            prim_path="{ENV_REGEX_NS}/probe",
            mesh_prim_paths=["/World/seabed"],
            max_range=DVL_MAX_RANGE,
            offset=DVLCfg.OffsetCfg(pos=DVL_OFFSET, rot=(0.0, 0.0, 0.0, 1.0)),
        )
        cam = UWCameraCfg(
            prim_path="/World/down_cam",
            width=160,
            height=120,
            spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, horizontal_aperture=20.955, clipping_range=(0.05, 100.0)),
            # world convention: camera looks along +X; rotate +90 deg about Y so it looks down (-Z)
            offset=UWCameraCfg.OffsetCfg(
                pos=(5.0, 5.0, CAM_HEIGHT), rot=(0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)), convention="world"
            ),
        )

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 60, device=DEVICE))
    scene = InteractiveScene(ValidationSceneCfg(num_envs=1, env_spacing=1.0))
    # the ray caster needs the actual Mesh prim under the spawned cuboid
    stage = sim.stage
    mesh_paths = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and str(p.GetPath()).startswith("/World/seabed")]
    print(f"[validate] seabed mesh prims: {mesh_paths}", flush=True)
    sim.reset()
    probe, dvl, cam = scene["probe"], scene["dvl"], scene["cam"]
    _, beam_dirs_s = dvl_janus_pattern(DVLJanusPatternCfg(elevation=dvl.cfg.elevation, rotation=dvl.cfg.rotation), DEVICE)
    offset_b = torch.tensor(DVL_OFFSET, device=DEVICE)

    def settle(n: int = 3) -> None:
        for _ in range(n):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())

    def place_probe(pos: tuple[float, float, float], rpy_deg: tuple[float, float, float]) -> torch.Tensor:
        r, p, y = (math.radians(a) for a in rpy_deg)
        q = mu.quat_from_euler_xyz(torch.tensor([r]), torch.tensor([p]), torch.tensor([y])).to(DEVICE)
        pose = torch.cat([torch.tensor([pos], device=DEVICE) + scene.env_origins, q], dim=1)
        probe.write_root_pose_to_sim_index(root_pose=pose)
        probe.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=DEVICE))
        settle()
        return q

    results: dict = {"dvl": [], "camera": {}}
    worst_dvl = 0.0
    cases = [
        ((0.0, 0.0, 2.0), (0.0, 0.0, 0.0)),
        ((1.0, -2.0, 4.5), (0.0, 0.0, 0.0)),
        ((0.0, 0.0, 3.0), (10.0, 0.0, 0.0)),
        ((0.0, 0.0, 3.0), (0.0, -12.0, 0.0)),
        ((0.0, 0.0, 3.0), (0.0, 0.0, 60.0)),
        ((2.0, 1.0, 6.0), (8.0, -5.0, 30.0)),
    ]
    for pos, rpy in cases:
        q = place_probe(pos, rpy)
        sensor_w = scene.env_origins[0] + torch.tensor(pos, device=DEVICE) + mu.quat_apply(q, offset_b[None])[0]
        dirs_w = mu.quat_apply(q.expand(4, 4), beam_dirs_s)  # sensor offset rotation is identity
        expected = sensor_w[2] / (-dirs_w[:, 2])
        origin_w = dvl_ray_origins_w(dvl.data.pos_w.torch, dvl.data.quat_w.torch, dvl.cfg.offset.pos)
        rng, hit = dvl_ranges_from_hits(dvl.data.ray_hits_w.torch, origin_w, dvl.cfg.min_range, dvl.cfg.max_range)
        err = (rng[0] - expected).abs().max().item()
        worst_dvl = max(worst_dvl, err)
        pos_err = (origin_w[0] - sensor_w).norm().item()
        results["dvl"].append({"pos": pos, "rpy_deg": rpy, "expected_m": expected.tolist(), "measured_m": rng[0].tolist(),
                               "max_abs_err_m": err, "sensor_pos_err_m": pos_err, "all_hit": bool(hit.all())})
        print(f"[validate] DVL pos={pos} rpy={rpy}: expected={[round(x, 4) for x in expected.tolist()]} "
              f"measured={[round(x, 4) for x in rng[0].tolist()]} max_err={err:.2e} m sensor_pos_err={pos_err:.2e} m", flush=True)

    place_probe((0.0, 0.0, DVL_MAX_RANGE + 5.0), (0.0, 0.0, 0.0))
    _, hit_far = dvl_ranges_from_hits(dvl.data.ray_hits_w.torch, dvl_ray_origins_w(dvl.data.pos_w.torch, dvl.data.quat_w.torch, dvl.cfg.offset.pos),
                                      dvl.cfg.min_range, dvl.cfg.max_range)
    results["dvl_out_of_range_all_miss"] = bool((~hit_far).all())
    print(f"[validate] DVL above max_range: all beams miss = {results['dvl_out_of_range_all_miss']}", flush=True)

    # ---------------------------------------------------------------- camera over the flat seabed
    settle(10)
    out = cam.data.output
    dist = out["distance_to_camera"].torch[0, ..., 0] if out["distance_to_camera"].torch.dim() == 4 else out["distance_to_camera"].torch[0]
    rgb = out["rgb"].torch
    K = cam.data.intrinsic_matrices.torch[0]
    h, w = dist.shape
    v, u = torch.meshgrid(torch.arange(h, device=DEVICE, dtype=torch.float32) + 0.5,
                          torch.arange(w, device=DEVICE, dtype=torch.float32) + 0.5, indexing="ij")
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    cam_h = cam.data.pos_w.torch[0, 2].item()
    expected_dist = cam_h * torch.sqrt(1 + x * x + y * y)
    rel_err = ((dist - expected_dist).abs() / expected_dist)
    uw = uw_render(rgb[:1], dist[None], cam.cfg)[0].float()
    uw_ref = rgb[0, ..., :3].float() * torch.exp(-dist[..., None] * torch.tensor(cam.cfg.atten_coeff, device=DEVICE)) + \
        255.0 * torch.tensor(cam.cfg.backscatter_value, device=DEVICE) * (1 - torch.exp(-dist[..., None] * torch.tensor(cam.cfg.backscatter_coeff, device=DEVICE)))
    uw_err = (uw - uw_ref.clamp(0, 255).floor()).abs().max().item()
    results["camera"] = {
        "camera_height_m": cam_h, "rgb_dtype": str(rgb.dtype), "rgb_shape": list(rgb.shape),
        "center_distance_m": dist[h // 2, w // 2].item(), "corner_distance_m": dist[0, 0].item(),
        "expected_corner_m": expected_dist[0, 0].item(),
        "distance_rel_err_median": rel_err.median().item(), "distance_rel_err_p99": rel_err.flatten().quantile(0.99).item(),
        "uw_image_max_abs_err": uw_err,
    }
    print(f"[validate] camera: {results['camera']}", flush=True)

    results["pass"] = {
        "dvl_ranges": worst_dvl < 2e-3 and all(c["all_hit"] for c in results["dvl"]),
        "dvl_sensor_position": all(c["sensor_pos_err_m"] < 1e-3 for c in results["dvl"]),
        "dvl_out_of_range": results["dvl_out_of_range_all_miss"],
        "camera_distance_semantics": results["camera"]["distance_rel_err_p99"] < 5e-3,
        "camera_uw_model": uw_err <= 1.0,
        "camera_rgb_uint8": results["camera"]["rgb_dtype"] == "torch.uint8",
    }
    results["all_pass"] = all(results["pass"].values())
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=1)
    print(f"[validate] PASS={results['pass']} ALL={results['all_pass']} -> {args.out}", flush=True)
