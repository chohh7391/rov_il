#!/usr/bin/env python3
"""Standalone Isaac Lab 3.0 check of the I90-class ROV + Bravo 7-class arm (one floating articulation).

Scene: one env, no cameras, gravity off on every body (rov_lab convention). Two copies of the asset: ``robot``
(I90_BRAVO7_CFG as shipped) and ``robot_sc`` (same, with articulation self-collisions enabled) 4 m apart.

Checks:
    prim_paths        I90_BRAVO7_PRIM_PATHS exist and are rigid bodies; rigid-body props (gravity off, zero PhysX
                      damping, velocity clamps) authored on the spawned bodies
    cfg_vs_design     cfg literals (hydro, thrusters, poses, masses, frames) equal the generator's params JSON
    mass_com          body masses, total (129.5 kg) / vehicle (120 kg), rov_base CoM + principal inertia, combined
                      CoM at home and stow vs the design
    joint_limits      sim position / velocity / effort limits equal BRAVO7_CLASS_JOINT_LIMITS
    reach             seabed targets 0.30 / 0.45 / 0.60 m below the hull and the forward target (hull-bottom height,
                      0.5 m ahead of the bow): FK after teleport, and closed-loop with joint targets from home
                      (hull floating, hydrodynamic wrench applied), TCP measured in the rov_base frame
    no_overlap        home and stow: geometric overlap of the simulated link poses (sampled primitives, arm-arm and
                      arm-hull, plus 5 mm clearance), and a dynamic check on robot_sc (self-collisions on): holding
                      the pose for 1 s must not push joints or the hull
    hydro_analytic    steady-state speeds from I90_HYDRO_PARAMS (surge 588 N ~ 1.54 m/s, sway 329 N, heave 304 N)
    hydro_sim         the same forces applied in sim (servo-style COM couple + rov_lab's compute_hydro_wrench)
    thrust_allocation LP over I90_THRUSTERS: max axis forces with the other actuated components (Fx, Fy, Fz, Mz,
                      plus roll Mx) zero, vs the catalogue axis thrusts

Physics runs on the CPU unless ``--device`` is given (shared GPU). Output JSON: runs/i90_model/test_i90_bravo7.json.

Run (from the repository root):
    OMNI_KIT_ACCEPT_EULA=YES <isaaclab-3.0-python> rov_lab/scripts/tools/test_i90_bravo7.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
TOOLS_DIR: Path = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description="I90-class ROV + Bravo 7-class arm checks (Isaac Lab 3.0).")
parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "i90_model", help="Output directory.")
parser.add_argument("--dt", type=float, default=1.0 / 120.0, help="Physics time step [s].")
add_launcher_args(parser)
args_cli = parser.parse_args()
if "--device" not in sys.argv:
    args_cli.device = "cpu"  # small articulations: CPU PhysX, minimal GPU memory

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "rov_lab" / "source"))
sys.path.insert(0, str(TOOLS_DIR))

PARAMS_JSON: Path = REPO_ROOT / "rov_lab" / "assets_src" / "robots" / "i90_bravo7" / "i90_bravo7_params.json"
HYDRO_PY: Path = REPO_ROOT / "rov_lab" / "source" / "rov_lab" / "tasks" / "template" / "mdp" / "hydrodynamics.py"
KGF: float = 9.80665
SC_OFFSET: tuple[float, float, float] = (0.0, 4.0, 0.0)

sim_cfg = sim_utils.SimulationCfg(dt=args_cli.dt, device=args_cli.device)


def rnd(x: float, n: int = 4) -> float:
    """Round for the JSON report."""
    return round(float(x), n)


def load_hydro_module() -> types.ModuleType:
    """Import rov_lab's hydrodynamics.py directly (skips the task package, which registers every gym env)."""
    spec = importlib.util.spec_from_file_location("rov_hydrodynamics", HYDRO_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve string annotations through sys.modules
    spec.loader.exec_module(module)
    return module


with launch_simulation(sim_cfg, args_cli):
    import isaaclab.sim.utils.stage as stage_utils
    import isaaclab.utils.math as mu
    from isaaclab.assets import Articulation, AssetBaseCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils import configclass, instantiate
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg
    from pxr import Usd, UsdPhysics

    import make_i90_bravo7_urdf as G
    from rov_lab.assets.robots import bravo7_class as B
    from rov_lab.assets.robots import i90_bravo7 as I

    params = json.loads(PARAMS_JSON.read_text())
    hydro_mod = load_hydro_module()

    sc_spawn = I.I90_BRAVO7_CFG.spawn.replace(
        articulation_props=PhysxArticulationCfg(
            enabled_self_collisions=True, solver_position_iteration_count=16, solver_velocity_iteration_count=1
        )
    )

    @configclass
    class I90TestSceneCfg(InteractiveSceneCfg):
        light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))
        robot = I.I90_BRAVO7_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        robot_sc = I.I90_BRAVO7_CFG.replace(
            prim_path="{ENV_REGEX_NS}/RobotSC",
            spawn=sc_spawn,
            init_state=I.I90_BRAVO7_CFG.init_state.replace(pos=SC_OFFSET),
        )

    sim = sim_utils.SimulationContext(sim_cfg)
    scene = instantiate(I90TestSceneCfg(num_envs=1, env_spacing=10.0))
    sim.reset()
    dt = sim.get_physics_dt()
    t_wall = time.time()

    robot = scene["robot"]
    robot_sc = scene["robot_sc"]
    dev = robot.device
    jnames: list[str] = list(robot.joint_names)
    bnames: list[str] = list(robot.body_names)
    jid = {n: i for i, n in enumerate(jnames)}
    bid = {n: i for i, n in enumerate(bnames)}
    base_i = bid[I.ROV_BASE_LINK]
    arm_joint_names = list(B.BRAVO7_CLASS_ARM_JOINT_NAMES)
    print(f"[test] joints: {jnames}\n[test] bodies: {bnames}", flush=True)
    report: dict = {"dt": dt, "device": str(dev), "joints": jnames, "bodies": bnames, "checks": {}}

    # ------------------------------------------------------------------------------------------ helpers
    q_tgt = {robot: robot.data.default_joint_pos.torch.clone(), robot_sc: robot_sc.data.default_joint_pos.torch.clone()}
    hydro_cfg = hydro_mod.HydroParamsCfg(**I.I90_HYDRO_PARAMS)
    hydro_coeffs = hydro_mod.build_hydro_coeffs(hydro_cfg, 1, dev)
    SERVO_KP_ANGULAR = torch.tensor([[12.0, 12.0, 18.0]], device=dev)
    masses = robot.data.body_mass.torch[0].clone()
    m_total = float(masses.sum())

    def base_pose(art: Articulation) -> tuple[torch.Tensor, torch.Tensor]:
        return art.data.body_link_pos_w.torch[0, base_i], art.data.body_link_quat_w.torch[0, base_i]

    def to_base(art: Articulation, p_w: torch.Tensor) -> torch.Tensor:
        """World point(s) (..., 3) -> rov_base frame."""
        p0, q0 = base_pose(art)
        flat = p_w.reshape(-1, 3)
        out = mu.quat_apply(mu.quat_conjugate(q0)[None].expand(flat.shape[0], 4), flat - p0)
        return out.reshape(p_w.shape)

    def link_poses_in_base(art: Articulation) -> dict[str, tuple[list[float], list[list[float]]]]:
        """Simulated link frames in rov_base, as (pos, R) python lists for the generator's overlap check."""
        pos_b = to_base(art, art.data.body_link_pos_w.torch[0])
        q0 = base_pose(art)[1]
        r0 = mu.matrix_from_quat(q0[None])[0]
        rl = mu.matrix_from_quat(art.data.body_link_quat_w.torch[0])
        r_rel = torch.matmul(r0.T[None], rl)
        return {n: (pos_b[i].tolist(), r_rel[i].tolist()) for i, n in enumerate(bnames)}

    def tcp_in_base(art: Articulation) -> torch.Tensor:
        jaw = bid[I.EE_LINK]
        off = torch.tensor([B.GRASP_POINT_OFFSET_M], device=dev)
        p_w = art.data.body_link_pos_w.torch[0, jaw] + mu.quat_apply(art.data.body_link_quat_w.torch[0, jaw][None], off)[0]
        return to_base(art, p_w)

    def combined_com_in_base(art: Articulation) -> torch.Tensor:
        com_w = art.data.body_com_pos_w.torch[0]
        c_w = (masses[:, None] * com_w).sum(0) / m_total
        return to_base(art, c_w)

    def set_pose(art: Articulation, pose: dict[str, float]) -> None:
        """Teleport the joints, reset the root to its default pose at rest, and hold the pose."""
        jp = art.data.default_joint_pos.torch.clone()
        for n, v in pose.items():
            jp[0, jid[n]] = v
        art.write_joint_position_to_sim_index(position=jp)
        art.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(jp))
        root = art.data.default_root_pose.torch.clone()
        root[0, :3] += scene.env_origins[0]
        art.write_root_pose_to_sim_index(root_pose=root)
        art.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=dev))
        q_tgt[art][:] = jp

    def set_target(art: Articulation, pose: dict[str, float]) -> None:
        for n, v in pose.items():
            q_tgt[art][0, jid[n]] = v

    def hydro_and_thrust(art: Articulation, thrust_b: tuple[float, float, float] | None, com_couple: bool = False) -> None:
        """Hydrodynamic wrench (rov_lab compute_hydro_wrench) + optional body-frame thrust, applied at the rov_base CoM
        through the permanent wrench composer (world frame). ``com_couple`` adds the servo's couple that moves the
        thrust line through the combined CoM."""
        q0 = base_pose(art)[1][None]
        q_inv = mu.quat_conjugate(q0)
        lin_b = mu.quat_apply(q_inv, art.data.body_lin_vel_w.torch[0, base_i][None])
        ang_b = mu.quat_apply(q_inv, art.data.body_ang_vel_w.torch[0, base_i][None])
        roll, pitch, _ = mu.euler_xyz_from_quat(q0)
        f_b, t_b = hydro_mod.compute_hydro_wrench(lin_b, ang_b, roll, pitch, None, hydro_coeffs, hydro_cfg)
        f_w = mu.quat_apply(q0, f_b)
        t_w = mu.quat_apply(q0, t_b)
        if thrust_b is not None:
            # servo-style angular-rate damping (ROVVelocityActionCfg.kp_angular, torque clamp 100 N*m): roll/pitch/yaw
            # rate targets are zero while translating
            tau_b = torch.clamp(-SERVO_KP_ANGULAR * ang_b, -100.0, 100.0)
            t_w = t_w + mu.quat_apply(q0, tau_b)
            ft_w = mu.quat_apply(q0, torch.tensor([thrust_b], device=dev))
            com_w = art.data.body_com_pos_w.torch[0]
            c_w = (masses[:, None] * com_w).sum(0) / m_total
            f_w = f_w + ft_w
            if com_couple:
                t_w = t_w + torch.linalg.cross((c_w - com_w[base_i])[None], ft_w)  # thrust through combined CoM
        art.permanent_wrench_composer.set_forces_and_torques_index(
            forces=f_w[:, None], torques=t_w[:, None], body_ids=[base_i], is_global=True
        )

    def step(n: int, hydro: bool = False, thrust_b: tuple[float, float, float] | None = None,
             arts: tuple[Articulation, ...] = (), com_couple: bool = False) -> None:
        arts = arts or (robot, robot_sc)
        for _ in range(n):
            for art in arts:
                art.set_joint_position_target_index(target=q_tgt[art])
            if hydro:
                hydro_and_thrust(robot, thrust_b, com_couple)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(dt)

    def clear_wrench(art: Articulation) -> None:
        z = torch.zeros(1, 1, 3, device=dev)
        art.permanent_wrench_composer.set_forces_and_torques_index(forces=z, torques=z, body_ids=[base_i], is_global=True)

    def full_pose(partial: dict[str, float], fingers: float = 0.03) -> dict[str, float]:
        pose = {n: 0.0 for n in arm_joint_names}
        pose.update({B.FINGER_LEFT_JOINT: fingers, B.FINGER_RIGHT_JOINT: fingers})
        pose.update(partial)
        return pose

    # ------------------------------------------------------------------------------------------ 0. prim paths + props
    stage = stage_utils.get_current_stage()
    env0 = "/World/envs/env_0"
    prim_rows = {}
    for link, path in I.I90_BRAVO7_PRIM_PATHS.items():
        p = stage.GetPrimAtPath(path.replace("{ENV_REGEX_NS}", env0))
        prim_rows[link] = {"path": path, "exists": bool(p and p.IsValid()),
                           "rigid_body": bool(p and p.IsValid() and p.HasAPI(UsdPhysics.RigidBodyAPI))}
    base_prim = stage.GetPrimAtPath(I.I90_BRAVO7_PRIM_PATHS[I.ROV_BASE_LINK].replace("{ENV_REGEX_NS}", env0))
    jaw_prim = stage.GetPrimAtPath(I.I90_BRAVO7_PRIM_PATHS[I.EE_LINK].replace("{ENV_REGEX_NS}", env0))

    def attr(prim: Usd.Prim, name: str) -> float | bool | None:
        a = prim.GetAttribute(name)
        return a.Get() if a and a.HasAuthoredValue() else None

    props = {}
    for tag, prim in (("rov_base", base_prim), ("jaw_base_link", jaw_prim)):
        props[tag] = {k: attr(prim, f"physxRigidBody:{k}") for k in
                      ("disableGravity", "linearDamping", "angularDamping", "maxLinearVelocity", "maxAngularVelocity")}
    props_ok = all(v["disableGravity"] is True and v["linearDamping"] == 0.0 and v["angularDamping"] == 0.0
                   and abs(v["maxAngularVelocity"] - I.I90_MAX_ANGULAR_VELOCITY_DEG_S) < 1e-3
                   and abs(v["maxLinearVelocity"] - I.I90_MAX_LINEAR_VELOCITY_M_S) < 1e-3 for v in props.values())
    report["checks"]["prim_paths_and_props"] = {
        "pass": all(r["exists"] and r["rigid_body"] for r in prim_rows.values()) and props_ok,
        "prim_paths": prim_rows, "rigid_body_props": props,
    }
    print(f"[test] prim paths/props pass={report['checks']['prim_paths_and_props']['pass']} {props}", flush=True)

    # ------------------------------------------------------------------------------------------ 0b. cfg vs design JSON
    mism: list[str] = []

    def close(a: float, b: float, tol: float = 1e-5) -> bool:
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))

    for k, v in I.I90_HYDRO_PARAMS.items():
        ref = params["hydro_params"][k]
        vals = v if isinstance(v, tuple) else (v,)
        refs = ref if isinstance(ref, list) else [ref]
        if len(vals) != len(refs) or not all(close(a, b) for a, b in zip(vals, refs)):
            mism.append(f"hydro.{k}")
    hydro_fields = set(hydro_mod.HydroParamsCfg.__dataclass_fields__) if hasattr(
        hydro_mod.HydroParamsCfg, "__dataclass_fields__") else set()
    if hydro_fields and not set(I.I90_HYDRO_PARAMS) <= hydro_fields:
        mism.append("hydro keys not in HydroParamsCfg")
    for t, ref in zip(I.I90_THRUSTERS, params["thrusters"]):
        if t.name != ref["name"] or not all(close(a, b) for a, b in zip(t.pos + t.direction, ref["pos_m"] + ref["direction"])) \
                or not close(t.max_forward_n, ref["max_forward_n"]) or not close(t.max_reverse_n, ref["max_reverse_n"]):
            mism.append(f"thruster.{t.name}")
    for n, v in I.I90_BRAVO7_HOME_JOINT_POS.items():
        if not close(v, params["poses"]["home"][n]):
            mism.append(f"home.{n}")
    for n, v in I.I90_BRAVO7_STOW_JOINT_POS.items():
        if not close(v, params["poses"]["stow"][n], 1e-4):
            mism.append(f"stow.{n}")
    for name, pose in I.I90_BRAVO7_REACH_POSES.items():
        key = name.split("_")[1]
        ref = params["reach"]["seabed"][key]["q"] if name.startswith("seabed") else params["reach"]["forward"]["q"]
        if not all(close(v, ref[n]) for n, v in pose.items()):
            mism.append(f"reach.{name}")
    frames = {I.FRONT_CAMERA_LINK: (I.I90_FRONT_CAMERA_POS, I.I90_FRONT_CAMERA_ROT),
              I.DVL_LINK: (I.I90_DVL_POS, I.I90_DVL_ROT),
              I.PRESSURE_SENSOR_LINK: (I.I90_PRESSURE_SENSOR_POS, I.I90_PRESSURE_SENSOR_ROT)}
    for link, (pos, rot) in frames.items():
        ref = params["sensor_frames"][link]
        if not all(close(a, b) for a, b in zip(pos + rot, ref["pos_m"] + ref["quat_xyzw"])):
            mism.append(f"frame.{link}")
    if not all(close(a, b) for a, b in zip(I.I90_ARM_MOUNT_POS + I.I90_ARM_MOUNT_ROT,
                                           params["arm_mount"]["pos_m"] + params["arm_mount"]["quat_xyzw"])):
        mism.append("arm_mount")
    for name, val, ref in (("rov_base_mass", I.I90_ROV_BASE_MASS_KG, params["links"]["rov_base"]["mass_kg"]),
                           ("vehicle_volume", I.I90_VEHICLE_DISPLACED_VOLUME_M3, params["vehicle"]["displaced_volume_m3"]),
                           ("system_net_buoyancy", I.I90_SYSTEM_NET_BUOYANCY_KG, params["system_stow"]["net_buoyancy_kg"])):
        if not close(val, ref, 1e-4):
            mism.append(name)
    report["checks"]["cfg_vs_design"] = {"pass": not mism, "mismatches": mism}
    print(f"[test] cfg vs design JSON mismatches: {mism}", flush=True)

    # ------------------------------------------------------------------------------------------ 1. mass + CoM
    vehicle_mass = sum(float(masses[bid[n]]) for n in I.I90_VEHICLE_LINK_NAMES)
    mass_err = max(abs(float(masses[bid[n]]) - (params["links"][n]["mass_kg"] if n in params["links"]
                                                else params["arm_links"][n]["mass_kg"])) for n in bnames)
    com_b_sim = robot.data.body_com_pos_b.torch[0, base_i]
    inertia_sim = robot.data.body_inertia.torch[0, base_i].reshape(3, 3)
    eig_sim = torch.linalg.eigvalsh(inertia_sim.double()).tolist()
    eig_ref = sorted(I.I90_ROV_BASE_PRINCIPAL_INERTIA_KG_M2)
    com_rows = {}
    for name, pose, ref in (("home", I.I90_BRAVO7_HOME_JOINT_POS, params["system_home"]["com_m"]),
                            ("stow", I.I90_BRAVO7_STOW_JOINT_POS, params["system_stow"]["com_m"])):
        set_pose(robot, pose)
        set_pose(robot_sc, pose)
        step(2)
        c = combined_com_in_base(robot)
        com_rows[name] = {"sim_m": [rnd(x, 5) for x in c.tolist()], "design_m": ref,
                          "err_m": rnd(math.dist(c.tolist(), ref), 6)}
    report["checks"]["mass_com"] = {
        "pass": abs(m_total - I.I90_SYSTEM_MASS_KG) < 1e-3 and abs(vehicle_mass - I.I90_VEHICLE_MASS_KG) < 1e-3
        and mass_err < 1e-4 and math.dist(com_b_sim.tolist(), I.I90_ROV_BASE_COM_M) < 1e-4
        and max(abs(a - b) for a, b in zip(eig_sim, eig_ref)) < 1e-3 and all(r["err_m"] < 1e-3 for r in com_rows.values()),
        "total_mass_kg": rnd(m_total, 4), "vehicle_mass_kg": rnd(vehicle_mass, 4),
        "rov_base_mass_kg": rnd(float(masses[base_i]), 4), "max_body_mass_err_kg": mass_err,
        "rov_base_com_sim_m": [rnd(x, 6) for x in com_b_sim.tolist()], "rov_base_com_design_m": list(I.I90_ROV_BASE_COM_M),
        "rov_base_principal_inertia_sim": [rnd(x, 4) for x in eig_sim], "rov_base_principal_inertia_design": eig_ref,
        "combined_com_in_rov_base": com_rows,
    }
    print(f"[test] mass total {m_total:.4f} vehicle {vehicle_mass:.4f}; rov_base CoM {com_b_sim.tolist()} "
          f"I {eig_sim}; combined CoM {com_rows}", flush=True)

    # ------------------------------------------------------------------------------------------ 2. joint limits
    pos_lim = robot.data.joint_pos_limits.torch[0]
    vel_lim = robot.data.joint_vel_limits.torch[0]
    eff_lim = robot.data.joint_effort_limits.torch[0]
    lim_rows = {}
    lim_ok = True
    for n in jnames:
        lo, hi, v, e = B.BRAVO7_CLASS_JOINT_LIMITS[n]
        i = jid[n]
        sim_vals = [float(pos_lim[i, 0]), float(pos_lim[i, 1]), float(vel_lim[i]), float(eff_lim[i])]
        ok = all(abs(a - b) <= 1e-3 * max(1.0, abs(b)) for a, b in zip(sim_vals, (lo, hi, v, e)))
        lim_ok &= ok
        lim_rows[n] = {"sim": [rnd(x, 5) for x in sim_vals], "bravo_cfg": [rnd(x, 5) for x in (lo, hi, v, e)], "ok": ok}
    report["checks"]["joint_limits"] = {"pass": lim_ok, "per_joint": lim_rows}
    print(f"[test] joint limits match Bravo cfg: {lim_ok}", flush=True)

    # ------------------------------------------------------------------------------------------ 3. reach
    reach_rows = {}
    reach_ok = True
    for name, partial in I.I90_BRAVO7_REACH_POSES.items():
        target = torch.tensor(I.I90_BRAVO7_REACH_TCP_TARGETS_M[name], device=dev)
        pose = full_pose(partial)
        # (a) FK: teleport, two steps
        set_pose(robot, pose)
        step(2, arts=(robot,))
        fk_err = float(torch.linalg.vector_norm(tcp_in_base(robot) - target))
        # (b) closed loop from home with joint targets, floating hull + hydrodynamic wrench
        set_pose(robot, I.I90_BRAVO7_HOME_JOINT_POS)
        step(5, arts=(robot,))
        set_target(robot, pose)
        hull0 = base_pose(robot)[0].clone()
        t_settle = None
        for k in range(int(14.0 / dt)):
            step(1, hydro=True, arts=(robot,))
            if t_settle is None and float(torch.linalg.vector_norm(tcp_in_base(robot) - target)) < 0.005 and \
                    float(robot.data.joint_vel.torch[0].abs().max()) < 0.01:
                t_settle = (k + 1) * dt
                step(int(0.5 / dt), hydro=True, arts=(robot,))
                break
        drive_err = float(torch.linalg.vector_norm(tcp_in_base(robot) - target))
        hull_drift = float(torch.linalg.vector_norm(base_pose(robot)[0] - hull0))
        q_err = max(abs(float(robot.data.joint_pos.torch[0, jid[n]]) - v) for n, v in pose.items()
                    if n in arm_joint_names)
        ok = fk_err < 1e-3 and drive_err < 5e-3
        reach_ok &= ok
        reach_rows[name] = {"target_tcp_m": list(I.I90_BRAVO7_REACH_TCP_TARGETS_M[name]),
                            "fk_tcp_err_m": rnd(fk_err, 6), "driven_tcp_err_m": rnd(drive_err, 5),
                            "driven_settle_time_s": rnd(t_settle, 3) if t_settle else None,
                            "driven_max_joint_err_rad": rnd(q_err, 5), "hull_drift_during_move_m": rnd(hull_drift, 4),
                            "ok": ok}
        clear_wrench(robot)
        print(f"[test] reach {name}: FK err {fk_err * 1e3:.3f} mm, driven err {drive_err * 1e3:.2f} mm "
              f"(settle {t_settle}), hull drift {hull_drift:.3f} m", flush=True)
    report["checks"]["reach"] = {"pass": reach_ok, "targets": reach_rows,
                                 "criterion": "FK < 1 mm, driven < 5 mm (TCP = grasp point, rov_base frame)"}

    # ------------------------------------------------------------------------------------------ 4. overlap / self-collision
    link_prims = {link: [G.Primitive(p["kind"], tuple(p["size"]), tuple(p["xyz"]), tuple(p["rpy"])) for p in prims]
                  for link, prims in params["link_primitives"].items()}
    adj = G.adjacent_pairs()
    ov_rows = {}
    ov_ok = True
    for name, pose in (("home", I.I90_BRAVO7_HOME_JOINT_POS), ("stow", I.I90_BRAVO7_STOW_JOINT_POS)):
        set_pose(robot, pose)
        set_pose(robot_sc, pose)
        step(2)
        poses = link_poses_in_base(robot)
        hits = G.overlaps(poses, link_prims, adj)
        near = G.overlaps(poses, link_prims, adj, inflate=0.005)
        # dynamic: robot_sc (self-collisions on) must stay put while holding the pose
        sc_q0 = robot_sc.data.joint_pos.torch[0].clone()
        sc_p0 = base_pose(robot_sc)[0].clone()
        max_qdev, max_v = 0.0, 0.0
        for _ in range(int(1.0 / dt)):
            step(1)
            max_qdev = max(max_qdev, float((robot_sc.data.joint_pos.torch[0] - sc_q0).abs().max()))
            max_v = max(max_v, float(robot_sc.data.root_lin_vel_w.torch[0].abs().max()))
        hull_move = float(torch.linalg.vector_norm(base_pose(robot_sc)[0] - sc_p0))
        ref_qdev = float((robot.data.joint_pos.torch[0] - q_tgt[robot][0]).abs().max())
        ok = not hits and max_qdev < 2e-3 and hull_move < 1e-3
        ov_ok &= ok
        ov_rows[name] = {"geometric_overlaps": hits, "pairs_within_5mm": near,
                         "self_collision_robot_max_joint_dev_rad": rnd(max_qdev, 6),
                         "self_collision_robot_max_hull_speed_m_s": rnd(max_v, 6),
                         "self_collision_robot_hull_move_m": rnd(hull_move, 6),
                         "reference_robot_max_joint_err_rad": rnd(ref_qdev, 6), "ok": ok}
        print(f"[test] {name}: overlaps {hits}, within 5 mm {near}, self-collision robot joint dev {max_qdev:.2e} rad, "
              f"hull move {hull_move:.2e} m", flush=True)
    # positive control: a pose that drives the upper arm into the hull must be geometrically flagged and pushed by
    # PhysX on robot_sc (proves that the dynamic check can detect contact)
    sc_root = stage.GetPrimAtPath(I.I90_BRAVO7_PRIM_PATHS[I.ROV_BASE_LINK].replace("{ENV_REGEX_NS}", env0)
                                  .replace("/Robot/", "/RobotSC/"))
    sc_flag = attr(sc_root, "physxArticulation:enabledSelfCollisions")
    bad_pose = {**I.I90_BRAVO7_HOME_JOINT_POS, B.SHOULDER_PITCH_JOINT: -1.2, B.ELBOW_PITCH_JOINT: 0.0,
                B.WRIST_PITCH_JOINT: 0.0}
    set_pose(robot, bad_pose)
    set_pose(robot_sc, bad_pose)
    step(2)
    bad_hits = G.overlaps(link_poses_in_base(robot), link_prims, adj)
    sc_q0 = robot_sc.data.joint_pos.torch[0].clone()
    sc_p0 = base_pose(robot_sc)[0].clone()
    step(int(0.5 / dt))
    bad_qdev = float((robot_sc.data.joint_pos.torch[0] - sc_q0).abs().max())
    bad_move = float(torch.linalg.vector_norm(base_pose(robot_sc)[0] - sc_p0))
    control_ok = bool(bad_hits) and (bad_qdev > 2e-3 or bad_move > 1e-3)
    ov_rows["positive_control_arm_into_hull"] = {
        "pose": {k: rnd(v, 4) for k, v in bad_pose.items()}, "geometric_overlaps": bad_hits,
        "self_collision_robot_max_joint_dev_rad": rnd(bad_qdev, 5), "self_collision_robot_hull_move_m": rnd(bad_move, 5),
        "ok": control_ok}
    print(f"[test] positive control: enabledSelfCollisions={sc_flag}, overlaps {bad_hits}, "
          f"joint dev {bad_qdev:.3e}, hull move {bad_move:.3e}", flush=True)
    report["checks"]["no_overlap_home_stow"] = {"pass": ov_ok and control_ok, "poses": ov_rows,
                                                "robot_sc_enabled_self_collisions": sc_flag}

    # ------------------------------------------------------------------------------------------ 5. hydro, analytic
    hp = I.I90_HYDRO_PARAMS

    def v_ss(axis: int, force: float) -> float:
        dl, dq = hp["linear_damping"][axis], hp["quadratic_damping"][axis]
        return (-dl + math.sqrt(dl * dl + 4.0 * dq * force)) / (2.0 * dq)

    fwd_n, rev_n, lat_n, vert_n = 60.0 * KGF, 30.0 * KGF, 33.5 * KGF, 31.0 * KGF
    analytic = {"surge_fwd_588N_m_s": v_ss(0, fwd_n), "surge_rev_294N_m_s": v_ss(0, rev_n),
                "sway_329N_m_s": v_ss(1, lat_n), "heave_304N_m_s": v_ss(2, vert_n),
                "yaw_rate_100Nm_rad_s": v_ss(5, 100.0)}
    target_speed = 3.0 * 1852.0 / 3600.0
    report["checks"]["hydro_analytic"] = {
        "pass": abs(analytic["surge_fwd_588N_m_s"] - target_speed) < 0.005 * target_speed,
        "catalogue_top_speed_m_s": rnd(target_speed, 4), **{k: rnd(v, 4) for k, v in analytic.items()},
        "servo_lin_vel_scale_m_s": [1.5, 1.0, 0.7],
    }
    print(f"[test] analytic steady state: {report['checks']['hydro_analytic']}", flush=True)

    # ------------------------------------------------------------------------------------------ 6. hydro, in sim
    def run_axis(axis: int, force: float, com_couple: bool) -> dict:
        set_pose(robot, I.I90_BRAVO7_STOW_JOINT_POS)
        step(2, arts=(robot,))
        thrust = [0.0, 0.0, 0.0]
        thrust[axis] = force
        vs, tilt, yaw_rates = [], 0.0, []
        for k in range(int(8.0 / dt)):
            step(1, hydro=True, thrust_b=tuple(thrust), arts=(robot,), com_couple=com_couple)
            q0 = base_pose(robot)[1][None]
            q_inv = mu.quat_conjugate(q0)
            v_b = mu.quat_apply(q_inv, robot.data.body_lin_vel_w.torch[0, base_i][None])[0]
            w_b = mu.quat_apply(q_inv, robot.data.body_ang_vel_w.torch[0, base_i][None])[0]
            if k * dt >= 7.0:
                vs.append(float(v_b[axis]))
                yaw_rates.append(float(w_b[2]))
            r, p, _ = mu.euler_xyz_from_quat(q0)
            tilt = max(tilt, abs(float(mu.wrap_to_pi(r))), abs(float(mu.wrap_to_pi(p))))
        clear_wrench(robot)
        v_mean = sum(vs) / len(vs)
        return {"speed_sim_m_s": rnd(v_mean, 4), "speed_analytic_m_s": rnd(v_ss(axis, force), 4),
                "rel_err": rnd(v_mean / v_ss(axis, force) - 1.0, 4), "max_roll_pitch_deg": rnd(math.degrees(tilt), 2),
                "mean_yaw_rate_rad_s": rnd(sum(yaw_rates) / len(yaw_rates), 4)}

    sim_rows, couple_rows = {}, {}
    for name, axis, force in (("surge_588N", 0, fwd_n), ("sway_329N", 1, lat_n), ("heave_304N", 2, vert_n)):
        sim_rows[name] = run_axis(axis, force, com_couple=False)
        couple_rows[name] = run_axis(axis, force, com_couple=True)
        print(f"[test] sim {name}: thrust at rov_base CoM {sim_rows[name]} | servo COM couple {couple_rows[name]}",
              flush=True)
    report["checks"]["hydro_sim"] = {
        "pass": all(abs(r["rel_err"]) < 0.02 for r in sim_rows.values()), "axes": sim_rows,
        "note": "constant body-frame force at the rov_base CoM (collinear with the hydrodynamic force, which rov_lab "
                "applies there) + servo-style angular-rate damping + rov_lab compute_hydro_wrench (damping + buoyancy "
                "moment); stowed arm, 8 s, speed averaged over the last second",
        "informative_with_servo_com_couple": couple_rows,
    }

    # ------------------------------------------------------------------------------------------ 7. thrust allocation LP
    from scipy.optimize import linprog

    ref = I.I90_THRUST_ALLOCATION_REF_M
    cols = []
    for t in I.I90_THRUSTERS:
        r = [t.pos[i] - ref[i] for i in range(3)]
        d = t.direction
        m = [r[1] * d[2] - r[2] * d[1], r[2] * d[0] - r[0] * d[2], r[0] * d[1] - r[1] * d[0]]
        cols.append(list(d) + m)
    bmat = [[cols[j][i] for j in range(len(cols))] for i in range(6)]  # 6 x n
    bounds = [(-t.max_reverse_n, t.max_forward_n) for t in I.I90_THRUSTERS]
    actuated = [0, 1, 2, 3, 5]  # Fx, Fy, Fz, Mz (4-DOF) + Mx, which the vertical pair can null (removes LP degeneracy)
    lp_rows = {}
    spec = {"+x": 60.0, "-x": 30.0, "+y": 33.5, "-y": 33.5, "+z": 31.0, "-z": None, "+yaw": None}
    for key in spec:
        axis = {"x": 0, "y": 1, "z": 2, "yaw": 5}[key[1:]]
        sign = 1.0 if key[0] == "+" else -1.0
        a_eq = [bmat[i] for i in actuated if i != axis]
        res = linprog([-sign * c for c in bmat[axis]], A_eq=a_eq, b_eq=[0.0] * len(a_eq), bounds=bounds, method="highs")
        w = [sum(bmat[i][j] * res.x[j] for j in range(len(res.x))) for i in range(6)]
        val = sign * w[axis]
        row = {"max_n_or_nm": rnd(val, 3), "parasitic_mx_nm": rnd(w[3], 3), "parasitic_my_nm": rnd(w[4], 3),
               "thrust_n": [rnd(x, 2) for x in res.x]}
        if spec[key] is not None:
            row.update({"model_kgf": rnd(val / KGF, 4), "catalogue_kgf": spec[key],
                        "rel_err": rnd(val / KGF / spec[key] - 1.0, 6)})
        lp_rows[key] = row
    report["checks"]["thrust_allocation"] = {
        "pass": all(abs(r["rel_err"]) < 1e-3 for r in lp_rows.values() if "rel_err" in r),
        "reference_point_m": list(ref), "axes": lp_rows,
        "servo_caps_n": {"force_upper": [588.0, 329.0, 304.0], "force_lower": [294.0, 329.0, 304.0], "torque": 100.0},
    }
    print("[test] allocation " + ", ".join(f"{k}: {r['max_n_or_nm']}" for k, r in lp_rows.items()), flush=True)

    # ------------------------------------------------------------------------------------------ report
    args_cli.out.mkdir(parents=True, exist_ok=True)
    report["design"] = {k: params[k] for k in ("vehicle", "system_stow", "system_home", "trim_solution", "arm_mount",
                                               "sensor_frames", "steady_state", "axis_thrust_match")}
    report["wall_time_s"] = rnd(time.time() - t_wall, 2)
    report["all_pass"] = all(c["pass"] for c in report["checks"].values())
    out_json = args_cli.out / "test_i90_bravo7.json"
    out_json.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"[test] all_pass={report['all_pass']} -> {out_json}", flush=True)
    for k, c in report["checks"].items():
        print(f"[test]   {k:<26} {'PASS' if c['pass'] else 'FAIL'}", flush=True)
