#!/usr/bin/env python3
"""Standalone Isaac Lab 3.0 check of the Bravo 7-class arm (fixed base, empty scene, one env, no cameras).

The arm is simulated "in water": gravity acts on the link masses and a per-link buoyancy force
rho * g * LINK_DISPLACED_VOLUME_M3 is applied at each link's centre of buoyancy (= centre of mass)
through the articulation's permanent wrench composer. Test payloads are plain gravity bodies whose mass
stands for their net (in-water) weight.

Checks:
    reach        shoulder-axis -> jaw-tip distance at full extension ~= 0.9 m
    signs        +pitch raises the next link, +ee roll lifts the +y (left) finger
    limits       sim-side position/velocity/effort limits equal the table; joints driven past their
                 position limits stay inside them; large steps saturate at (not above) the velocity limit
    grasp        jaws close on a 40 mm cylinder (gap ~= 40 mm) and hold 2 kg and 10 kg payloads
    wrench       ft_sensor_joint (jaw_base_link incoming joint wrench) changes by the held load
    lift         10 kg net payload at mid reach: holding torques vs the static model, shoulder lift
    ee torque    end-effector roll drive holds an opposing roll load up to, and breaks away at, ~20 N*m

Physics runs on the CPU unless ``--device`` is given, to keep the footprint on a shared GPU small.

Run (from the repository root):
    OMNI_KIT_ACCEPT_EULA=YES <isaaclab-3.0-python> rov_lab/scripts/tools/test_bravo7_class.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab.app import add_launcher_args, launch_simulation

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject, RigidObjectCfg

REPO_ROOT: Path = Path(__file__).resolve().parents[3]

parser = argparse.ArgumentParser(description="Bravo 7-class arm checks (Isaac Lab 3.0).")
parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "arm_model", help="Output directory.")
parser.add_argument("--dt", type=float, default=1.0 / 120.0, help="Physics time step [s].")
add_launcher_args(parser)
args_cli = parser.parse_args()
if "--device" not in sys.argv:
    args_cli.device = "cpu"  # single small articulation: CPU PhysX, minimal GPU memory

import json  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "rov_lab" / "source"))

PARAMS_JSON: Path = REPO_ROOT / "rov_lab" / "assets_src" / "robots" / "bravo7_class" / "bravo7_class_params.json"
GRAVITY: float = 9.81
PAYLOAD_RADIUS: float = 0.020
PAYLOAD_HEIGHT: float = 0.120
SHELF_POS: tuple[float, float, float] = (3.0, 0.0, -0.025)  # static shelf where payloads rest when unused
PAYLOAD_PARK: dict[str, tuple[float, float, float]] = {
    "payload_2kg": (3.0, -0.15, 0.5 * PAYLOAD_HEIGHT + 0.002),
    "payload_10kg": (3.0, 0.15, 0.5 * PAYLOAD_HEIGHT + 0.002),
}
PAYLOAD_MASS: dict[str, float] = {"payload_2kg": 2.0, "payload_10kg": 10.0}

sim_cfg = sim_utils.SimulationCfg(dt=args_cli.dt, device=args_cli.device)


def as_float(x: torch.Tensor | float) -> float:
    """Tensor/number to Python float."""
    return float(x.item()) if isinstance(x, torch.Tensor) else float(x)


def rnd(x: float, n: int = 4) -> float:
    """Round for the JSON report."""
    return round(float(x), n)


def payload_cfg(name: str) -> RigidObjectCfg:
    """40 mm test cylinder (axis z) with gravity on; mass = net weight it represents."""
    from isaaclab.assets import RigidObjectCfg

    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=PAYLOAD_PARK[name]),
        # convex-hull mesh cylinder: robust box-vs-cylinder contact on both CPU and GPU PhysX
        spawn=sim_utils.MeshCylinderCfg(
            radius=PAYLOAD_RADIUS,
            height=PAYLOAD_HEIGHT,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=0.5),
            mass_props=sim_utils.MassPropertiesCfg(mass=PAYLOAD_MASS[name]),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.2, 0.1)),
        ),
    )


with launch_simulation(sim_cfg, args_cli):
    import warp as wp

    import isaaclab.utils.math as mu
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import JointWrenchSensorCfg
    from isaaclab.utils import configclass, instantiate

    from rov_lab.assets.robots import bravo7_class as B

    params = json.loads(PARAMS_JSON.read_text())

    @configclass
    class Bravo7TestSceneCfg(InteractiveSceneCfg):
        light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))
        robot = B.BRAVO7_CLASS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")
        shelf = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Shelf",
            init_state=AssetBaseCfg.InitialStateCfg(pos=SHELF_POS),
            spawn=sim_utils.CuboidCfg(
                size=(0.6, 0.6, 0.05), collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True)
            ),
        )
        payload_2kg = payload_cfg("payload_2kg")
        payload_10kg = payload_cfg("payload_10kg")

    sim = sim_utils.SimulationContext(sim_cfg)
    scene = instantiate(Bravo7TestSceneCfg(num_envs=1, env_spacing=8.0))
    sim.reset()
    dt = sim.get_physics_dt()

    robot = scene["robot"]
    wrench = scene["wrench"]
    dev = robot.device
    origin = scene.env_origins[0]

    jnames: list[str] = list(robot.joint_names)
    bnames: list[str] = list(robot.body_names)
    jid = {n: i for i, n in enumerate(jnames)}
    bid = {n: i for i, n in enumerate(bnames)}
    arm_ids = [jid[n] for n in B.BRAVO7_CLASS_ARM_JOINT_NAMES]
    finger_ids = [jid[n] for n in B.BRAVO7_CLASS_FINGER_JOINT_NAMES]
    backend_dof = [list(robot.backend_joint_names).index(n) for n in jnames]
    wrench_bid = {n: i for i, n in enumerate(wrench.body_names)}
    print(f"[test] joints: {jnames}\n[test] bodies: {bnames}", flush=True)

    report: dict = {"dt": dt, "device": str(dev), "joints": jnames, "bodies": bnames, "checks": {}}

    # ---------------------------------------------------------------- buoyancy (global +z at each CoM)
    rho = B.WATER_DENSITY_KG_M3
    buoy = torch.zeros(1, len(bnames), 3, device=dev)
    for name, vol in B.LINK_DISPLACED_VOLUME_M3.items():
        buoy[0, bid[name], 2] = rho * GRAVITY * vol

    def set_external_wrench(jaw_torque_w: tuple[float, float, float] | None = None) -> None:
        """Buoyancy on every link plus an optional world-frame torque on jaw_base_link (one global wrench set)."""
        torques = torch.zeros_like(buoy)
        if jaw_torque_w is not None:
            torques[0, bid[B.JAW_BASE_LINK]] = torch.tensor(jaw_torque_w, device=dev)
        robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=buoy, torques=torques, body_ids=list(range(len(bnames))), is_global=True
        )

    set_external_wrench()

    # ---------------------------------------------------------------- helpers
    q_tgt = robot.data.default_joint_pos.torch.clone()

    def T(name: str) -> torch.Tensor:
        return robot.data.body_link_pos_w.torch[0, bid[name]]

    def Q(name: str) -> torch.Tensor:
        return robot.data.body_link_quat_w.torch[0, bid[name]]

    def jaw_point(offset: tuple[float, float, float]) -> torch.Tensor:
        off = torch.tensor([offset], device=dev)
        return T(B.JAW_BASE_LINK) + mu.quat_apply(Q(B.JAW_BASE_LINK)[None], off)[0]

    def to_jaw(p_w: torch.Tensor) -> torch.Tensor:
        """World point -> jaw_base_link frame."""
        return mu.quat_apply(mu.quat_conjugate(Q(B.JAW_BASE_LINK)[None]), (p_w - T(B.JAW_BASE_LINK))[None])[0]

    def q() -> torch.Tensor:
        return robot.data.joint_pos.torch[0]

    def qd() -> torch.Tensor:
        return robot.data.joint_vel.torch[0]

    def projected_torques() -> torch.Tensor:
        """Solver joint forces projected on each DOF (drive + friction + limit), in Isaac Lab joint order."""
        raw = robot.root_view.get_dof_projected_joint_forces()
        raw = raw if isinstance(raw, torch.Tensor) else wp.to_torch(raw)
        return raw.reshape(-1)[backend_dof].to(dev)

    def ft_wrench() -> tuple[torch.Tensor, torch.Tensor]:
        """Incoming joint wrench of jaw_base_link (= ft_sensor_joint) in its child joint frame."""
        i = wrench_bid[B.JAW_BASE_LINK]
        return wrench.data.force.torch[0, i].clone(), wrench.data.torque.torch[0, i].clone()

    def set_pose(pose: dict[str, float]) -> None:
        """Teleport to a joint configuration and hold it."""
        jp = robot.data.joint_pos.torch.clone()
        for n, v in pose.items():
            jp[0, jid[n]] = v
        robot.write_joint_position_to_sim_index(position=jp)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(jp))
        q_tgt[:] = jp

    def set_target(pose: dict[str, float]) -> None:
        for n, v in pose.items():
            q_tgt[0, jid[n]] = v

    def place(obj: RigidObject, pos_env: torch.Tensor | tuple[float, float, float]) -> None:
        """Teleport a rigid object (env-local position, identity orientation) and zero its velocity."""
        pose = obj.data.default_root_pose.torch.clone()
        pose[0, :3] = torch.as_tensor(pos_env, device=dev, dtype=torch.float32) + origin
        pose[0, 3:] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev)
        obj.write_root_pose_to_sim_index(root_pose=pose)
        obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=dev))

    def step(n: int, monitor: dict | None = None) -> None:
        for _ in range(n):
            robot.set_joint_position_target_index(target=q_tgt)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(dt)
            if monitor is not None:
                v = qd().abs()
                monitor["qd_max"] = torch.maximum(monitor["qd_max"], v)
                monitor["qd_hist"].append(v.clone())
                monitor["q_max"] = torch.maximum(monitor["q_max"], q())
                monitor["q_min"] = torch.minimum(monitor["q_min"], q())

    def new_monitor() -> dict:
        return {
            "qd_max": torch.zeros(len(jnames), device=dev),
            "qd_hist": [],
            "q_max": torch.full((len(jnames),), -1e9, device=dev),
            "q_min": torch.full((len(jnames),), 1e9, device=dev),
        }

    t_wall = time.time()
    lim_tab = B.BRAVO7_CLASS_JOINT_LIMITS
    zero_pose = {n: 0.0 for n in jnames}

    # ---------------------------------------------------------------- 0. sim-side limits == table
    pos_lim = robot.data.joint_pos_limits.torch[0]
    vel_lim = robot.data.joint_vel_limits.torch[0]
    eff_lim = robot.data.joint_effort_limits.torch[0]
    table_ok = True
    table = {}
    for n in jnames:
        lo, hi, v, e = lim_tab[n]
        i = jid[n]
        sim_vals = [as_float(pos_lim[i, 0]), as_float(pos_lim[i, 1]), as_float(vel_lim[i]), as_float(eff_lim[i])]
        ok = all(abs(a - b) <= 1e-3 * max(1.0, abs(b)) for a, b in zip(sim_vals, (lo, hi, v, e)))
        table_ok &= ok
        table[n] = {"sim": [rnd(x, 5) for x in sim_vals], "table": [rnd(x, 5) for x in (lo, hi, v, e)], "ok": ok}
    masses = robot.data.body_mass.torch[0]
    mass_err = max(abs(as_float(masses[bid[n]]) - m) for n, m in B.LINK_MASS_KG.items())
    vol_err = max(abs(params["links"][n]["displaced_volume_m3"] - v) for n, v in B.LINK_DISPLACED_VOLUME_M3.items())
    total_vol = sum(B.LINK_DISPLACED_VOLUME_M3.values())
    report["checks"]["sim_limits_match_table"] = {"pass": table_ok, "per_joint": table}
    report["checks"]["masses_and_volumes"] = {
        "pass": mass_err < 1e-4 and vol_err < 1e-9,
        "total_mass_kg": rnd(as_float(masses.sum())),
        "total_displaced_volume_m3": rnd(total_vol, 7),
        "wet_weight_kg": rnd(as_float(masses.sum()) - rho * total_vol),
        "max_mass_err_kg": mass_err,
        "max_cfg_vs_generator_volume_err_m3": vol_err,
    }
    print(f"[test] limits match table: {table_ok}; mass err {mass_err:.2e}; volume err {vol_err:.2e}", flush=True)

    # ---------------------------------------------------------------- 1. reach at full extension
    set_pose(zero_pose)
    step(240)
    reach = as_float(torch.linalg.vector_norm(jaw_point(B.JAW_TIP_OFFSET_M) - T("upper_arm_link")))
    sag = (q() - q_tgt[0])[arm_ids]
    tau = projected_torques()
    report["checks"]["reach"] = {
        "pass": abs(reach - B.BRAVO7_CLASS_REACH_M) < 0.005,
        "shoulder_axis_to_jaw_tip_m": rnd(reach, 5),
        "joint_sag_rad": [rnd(x, 5) for x in sag.tolist()],
        "holding_torque_nm": {n: rnd(as_float(tau[jid[n]]), 2) for n in B.BRAVO7_CLASS_ARM_JOINT_NAMES},
    }
    # static wet-weight moment about the shoulder axis (validates mass + buoyancy application)
    com_w = robot.data.body_com_pos_w.torch[0]
    axis_w = mu.quat_apply(Q("shoulder_link")[None], torch.tensor([[0.0, -1.0, 0.0]], device=dev))[0]
    tau_model = 0.0
    for n in B.BRAVO7_CLASS_LINK_NAMES[2:]:
        f_wet = torch.tensor([0.0, 0.0, -(B.LINK_MASS_KG[n] - rho * B.LINK_DISPLACED_VOLUME_M3[n]) * GRAVITY],
                             device=dev)
        tau_model += as_float(torch.dot(torch.linalg.cross(com_w[bid[n]] - T("upper_arm_link"), f_wet), axis_w))
    report["checks"]["reach"]["shoulder_holding_torque_wet_model_nm"] = rnd(-tau_model, 2)
    print(f"[test] reach {reach:.4f} m, sag {sag.cpu().numpy().round(4)}", flush=True)

    # ---------------------------------------------------------------- 1b. joint sign conventions
    def tip_after(pose: dict[str, float]) -> torch.Tensor:
        set_pose({**zero_pose, **pose})
        step(2)
        return jaw_point(B.JAW_TIP_OFFSET_M)

    tip0 = tip_after({})
    signs = {
        "shoulder_+0.3_raises_tip_m": as_float(tip_after({B.SHOULDER_PITCH_JOINT: 0.3})[2] - tip0[2]),
        "elbow_+0.3_raises_tip_m": as_float(tip_after({B.ELBOW_PITCH_JOINT: 0.3})[2] - tip0[2]),
        "wrist_+0.3_raises_tip_m": as_float(tip_after({B.WRIST_PITCH_JOINT: 0.3})[2] - tip0[2]),
        "base_yaw_+0.3_moves_tip_+y_m": as_float(tip_after({B.BASE_YAW_JOINT: 0.3})[1] - tip0[1]),
    }
    set_pose({**zero_pose, B.FINGER_LEFT_JOINT: 0.02, B.FINGER_RIGHT_JOINT: 0.02})
    step(2)
    fl0 = T("finger_left_link").clone()
    set_pose({**zero_pose, B.FINGER_LEFT_JOINT: 0.02, B.FINGER_RIGHT_JOINT: 0.02, B.EE_ROLL_JOINT: 0.3})
    step(2)
    signs["ee_roll_+0.3_lifts_left_finger_m"] = as_float(T("finger_left_link")[2] - fl0[2])
    set_pose({**zero_pose, B.FINGER_LEFT_JOINT: 0.03})
    step(2)
    signs["finger_left_+q_moves_+y_m"] = as_float(to_jaw(T("finger_left_link"))[1])
    report["checks"]["joint_signs"] = {"pass": all(v > 0 for v in signs.values()),
                                       **{k: rnd(v, 4) for k, v in signs.items()}}
    print(f"[test] signs {report['checks']['joint_signs']}", flush=True)

    # ---------------------------------------------------------------- 2a. velocity limits on large steps
    home = dict(B.BRAVO7_CLASS_HOME_JOINT_POS)
    home.update({B.FINGER_LEFT_JOINT: 0.005, B.FINGER_RIGHT_JOINT: 0.005})
    vel_rows: dict = {}
    vel_ok = True
    for group, names, delta in (
        ("arm", B.BRAVO7_CLASS_ARM_JOINT_NAMES, 0.6),
        ("jaws", B.BRAVO7_CLASS_FINGER_JOINT_NAMES, 0.04),
    ):
        set_pose(home)
        step(60)
        mon = new_monitor()
        set_target({n: home[n] + delta for n in names})
        step(int(2.5 / dt), mon)
        set_target({n: home[n] - (delta if group == "arm" else 0.0) for n in names})
        step(int(4.0 / dt), mon)
        hist = torch.stack(mon["qd_hist"])
        for n in names:
            lim = lim_tab[n][2]
            col = hist[:, jid[n]]
            vmax = as_float(col.max())
            frac_over = as_float((col > 1.02 * lim).float().mean())
            ok = 0.9 * lim <= vmax <= 1.05 * lim  # PhysX max joint velocity is a solver constraint, not exact
            vel_ok &= ok
            vel_rows[n] = {"max_abs_vel": rnd(vmax, 4), "limit": rnd(lim, 4), "peak_ratio": rnd(vmax / lim, 3),
                           "fraction_of_steps_above_1.02x": rnd(frac_over, 4), "ok": ok}
    report["checks"]["velocity_limits"] = {"pass": vel_ok, "criterion": "0.9 <= peak/limit <= 1.05",
                                           "per_joint": vel_rows}
    print("[test] velocity peak ratio " + ", ".join(f"{n}:{r['peak_ratio']}" for n, r in vel_rows.items()), flush=True)

    # ---------------------------------------------------------------- 2b. position limits (drive past them)
    lim_rows: dict = {}
    lim_ok = True
    for side in ("upper", "lower"):
        sgn = 1.0 if side == "upper" else -1.0
        start, beyond = {}, {}
        for n in jnames:
            lo, hi, _, _ = lim_tab[n]
            edge = hi if side == "upper" else lo
            finger = n in B.BRAVO7_CLASS_FINGER_JOINT_NAMES
            start[n] = edge - sgn * (0.01 if finger else 0.15)
            beyond[n] = edge + sgn * (0.02 if finger else 0.5)
        set_pose(start)
        mon = new_monitor()
        set_target(beyond)
        step(int(1.5 / dt), mon)
        for n in jnames:
            lo, hi, _, _ = lim_tab[n]
            viol = as_float(mon["q_max"][jid[n]]) - hi if side == "upper" else lo - as_float(mon["q_min"][jid[n]])
            reached = abs(as_float(q()[jid[n]]) - (hi if side == "upper" else lo))
            ok = viol < (1e-3 if n in B.BRAVO7_CLASS_FINGER_JOINT_NAMES else 1e-2)
            lim_ok &= ok
            lim_rows.setdefault(n, {})[side] = {"max_violation": rnd(viol, 5), "final_dist_to_limit": rnd(reached, 5),
                                                "ok": ok}
    report["checks"]["position_limits"] = {"pass": lim_ok, "per_joint": lim_rows}
    print(f"[test] position limits respected: {lim_ok}", flush=True)

    # ---------------------------------------------------------------- 3. grasp 40 mm cylinders at mid reach
    mid = dict(params["static_torque_check_nm"]["mid_reach_pose_rad"])
    grasp_pose = dict(zero_pose)
    grasp_pose.update(mid)
    grasp_pose.update({B.FINGER_LEFT_JOINT: 0.035, B.FINGER_RIGHT_JOINT: 0.035})
    open_jaws = {B.FINGER_LEFT_JOINT: 0.035, B.FINGER_RIGHT_JOINT: 0.035}
    closed_jaws = {B.FINGER_LEFT_JOINT: 0.0, B.FINGER_RIGHT_JOINT: 0.0}

    def grasp(name: str) -> dict:
        """Close on the payload while pinning it at the grasp point, release it, and measure the hold."""
        obj = scene[name]
        mass = PAYLOAD_MASS[name]
        set_pose(grasp_pose)
        step(120)
        f0, t0 = ft_wrench()  # empty jaws
        gp_env = jaw_point(B.GRASP_POINT_OFFSET_M) - origin
        set_target(closed_jaws)
        for _ in range(int(0.6 / dt)):  # pin the payload (zero velocity) until the fingers have closed on it
            place(obj, gp_env)
            step(1)
        p_release = to_jaw(obj.data.root_pos_w.torch[0])
        step(int(1.5 / dt))
        f1, t1 = ft_wrench()
        p_held = to_jaw(obj.data.root_pos_w.torch[0])
        slip = as_float(torch.linalg.vector_norm(p_held - p_release))
        w_jaw = mu.quat_apply(mu.quat_conjugate(Q(B.JAW_BASE_LINK)[None]),
                              torch.tensor([[0.0, 0.0, -mass * GRAVITY]], device=dev))[0]
        exp_df = -w_jaw  # the parent (wrist_roll_link) must additionally carry the payload weight
        exp_dt = -torch.linalg.cross(p_held, w_jaw)
        df, dtq = f1 - f0, t1 - t0
        rel_f = as_float(torch.linalg.vector_norm(df - exp_df) / torch.linalg.vector_norm(exp_df))
        rel_t = as_float(torch.linalg.vector_norm(dtq - exp_dt) / torch.linalg.vector_norm(exp_dt))
        tau = projected_torques()
        row = {
            "payload_net_weight_n": rnd(mass * GRAVITY, 2),
            "closed_jaw_gap_m": rnd(as_float(q()[finger_ids].sum()), 5),
            "finger_drive_force_n": [rnd(x, 1) for x in tau[finger_ids].tolist()],
            "payload_slip_after_release_m": rnd(slip, 5),
            "payload_in_jaw_m": [rnd(x, 4) for x in p_held.tolist()],
            "held": slip < 0.005,
            "ft_force_empty_n": [rnd(x, 2) for x in f0.tolist()],
            "ft_force_delta_n": [rnd(x, 2) for x in df.tolist()],
            "ft_force_delta_expected_n": [rnd(x, 2) for x in exp_df.tolist()],
            "ft_torque_delta_nm": [rnd(x, 3) for x in dtq.tolist()],
            "ft_torque_delta_expected_nm": [rnd(x, 3) for x in exp_dt.tolist()],
            "force_rel_err": rnd(rel_f, 4),
            "torque_rel_err": rnd(rel_t, 4),
            "wrench_ok": rel_f < 0.1 and rel_t < 0.15,
            "arm_tracking_err_rad": [rnd(x, 4) for x in (q() - q_tgt[0])[arm_ids].tolist()],
            "joint_torque_nm": {n: rnd(as_float(tau[jid[n]]), 2) for n in B.BRAVO7_CLASS_ARM_JOINT_NAMES},
        }
        print(f"[test] grasp {name}: gap {row['closed_jaw_gap_m']} m, slip {slip * 1e3:.2f} mm, "
              f"dF {row['ft_force_delta_n']} (exp {row['ft_force_delta_expected_n']}), "
              f"dT {row['ft_torque_delta_nm']} (exp {row['ft_torque_delta_expected_nm']})", flush=True)
        return row

    def release(name: str) -> None:
        set_target(open_jaws)
        step(60)
        place(scene[name], PAYLOAD_PARK[name])
        step(30)

    grasp_rows = {"payload_2kg": grasp("payload_2kg")}
    release("payload_2kg")
    grasp_rows["payload_10kg"] = grasp("payload_10kg")

    # lift the 10 kg net payload: raise the shoulder by 0.15 rad at mid reach
    heavy = scene["payload_10kg"]
    p_before = to_jaw(heavy.data.root_pos_w.torch[0])
    z0 = as_float(heavy.data.root_pos_w.torch[0, 2])
    set_target({B.SHOULDER_PITCH_JOINT: grasp_pose[B.SHOULDER_PITCH_JOINT] + 0.15})
    step(int(1.5 / dt))
    lift_dz = as_float(heavy.data.root_pos_w.torch[0, 2]) - z0
    lift_err = as_float((q() - q_tgt[0])[jid[B.SHOULDER_PITCH_JOINT]])
    lift_slip = as_float(torch.linalg.vector_norm(to_jaw(heavy.data.root_pos_w.torch[0]) - p_before))
    release("payload_10kg")

    static_pred = params["static_torque_check_nm"]["mid_reach_upright"]
    hold = grasp_rows["payload_10kg"]["joint_torque_nm"]
    hydro_rows = {
        n: {"measured": hold[n], "static_model": rnd(static_pred[n], 2),
            "rel_err": rnd(abs(abs(hold[n]) - static_pred[n]) / max(static_pred[n], 1e-6), 3)}
        for n in (B.SHOULDER_PITCH_JOINT, B.ELBOW_PITCH_JOINT, B.WRIST_PITCH_JOINT)
    }
    report["checks"]["grasp_and_wrench"] = {
        "pass": all(r["held"] and r["wrench_ok"] and abs(r["closed_jaw_gap_m"] - 2 * PAYLOAD_RADIUS) < 0.005
                    for r in grasp_rows.values()),
        "loads": grasp_rows,
    }
    report["checks"]["lift_10kg_wet_mid_reach"] = {
        "pass": lift_dz > 0.03 and lift_slip < 0.005 and abs(lift_err) < 0.05
        and all(r["rel_err"] < 0.1 for r in hydro_rows.values()),
        "payload_rise_m": rnd(lift_dz, 4),
        "shoulder_tracking_err_rad": rnd(lift_err, 4),
        "payload_slip_m": rnd(lift_slip, 5),
        "holding_torque_vs_static_model": hydro_rows,
    }
    print(f"[test] lift: rise {lift_dz:.3f} m, shoulder err {lift_err:.4f} rad, slip {lift_slip * 1e3:.2f} mm; "
          f"hold vs model {hydro_rows}", flush=True)

    # ---------------------------------------------------------------- 4. end-effector roll torque saturation
    # 4a. Breakaway against a known blocking load: the roll joint is commanded 0.15 rad ahead (PD demand 22.5 N*m,
    #     above the 20 N*m limit) while an opposing torque about the roll axis (world x at this pose) loads the jaws.
    #     Below the effort limit the joint holds and advances; above it the drive saturates and is back-driven.
    ee = jid[B.EE_ROLL_JOINT]
    tau_lim = lim_tab[B.EE_ROLL_JOINT][3]
    k_ee = as_float(robot.data.joint_stiffness.torch[0, ee])
    lead = 0.15
    roll_pose = dict(grasp_pose)
    roll_pose.update({B.FINGER_LEFT_JOINT: 0.02, B.FINGER_RIGHT_JOINT: 0.02})
    trials = {}
    for tau_ext in (15.0, 18.0, 19.0, 19.5, 20.5, 21.0, 22.0, 25.0):
        set_pose(roll_pose)
        step(30)
        q0 = as_float(q()[ee])
        set_external_wrench((-tau_ext, 0.0, 0.0))
        set_target({B.EE_ROLL_JOINT: q0 + lead})
        step(int(0.5 / dt))
        dq = as_float(q()[ee]) - q0
        trials[tau_ext] = {"dq_rad": rnd(dq, 4), "held": dq > 0.0, "solver_joint_torque_nm": rnd(as_float(projected_torques()[ee]), 3)}
        set_external_wrench()
    held = [t for t, r in trials.items() if r["held"]]
    lost = [t for t, r in trials.items() if not r["held"]]
    breakaway = 0.5 * (max(held) + min(lost)) if held and lost else float("nan")
    # 4b. Informational: deep saturation against a gripped, world-pinned 40 mm bar (target 1 rad ahead).
    bar_name = "payload_10kg"
    bar = scene[bar_name]
    set_pose(grasp_pose)
    step(60)
    bar_env = jaw_point(B.GRASP_POINT_OFFSET_M) - origin
    set_target(closed_jaws)
    for _ in range(int(0.6 / dt)):
        place(bar, bar_env)
        step(1)
    q_ee0 = as_float(q()[ee])
    set_target({B.EE_ROLL_JOINT: q_ee0 + 1.0})
    tau_hist = []
    for _ in range(int(1.5 / dt)):
        place(bar, bar_env)
        step(1)
        tau_hist.append(as_float(projected_torques()[ee]))
    deep = {
        "stall_angle_rad": rnd(as_float(q()[ee]) - q_ee0, 4),
        "solver_joint_torque_nm": rnd(sum(tau_hist[-30:]) / 30.0, 3),
        "isaaclab_applied_torque_estimate_nm": rnd(as_float(robot.data.applied_torque.torch[0, ee]), 3),
        "note": "PhysX implicit drive + joint velocity limit: a target far beyond a blocked joint delivers well below "
                "the effort limit; keep |q_target - q| below ~1.3 * effort / stiffness (rate-limit targets).",
    }
    release(bar_name)
    report["checks"]["ee_roll_torque_saturation"] = {
        "pass": abs(breakaway - tau_lim) <= 1.0 and trials[19.0]["held"] and not trials[21.0]["held"],
        "effort_limit_nm": tau_lim,
        "stiffness_nm_per_rad": rnd(k_ee, 2),
        "target_lead_rad": lead,
        "breakaway_torque_nm": rnd(breakaway, 3),
        "trials_opposing_torque_nm": {str(k): v for k, v in trials.items()},
        "deep_saturation_gripped_bar_informational": deep,
    }
    print(f"[test] ee roll: breakaway {breakaway:.2f} N*m (limit {tau_lim}); trials "
          + ", ".join(f"{k}:{'hold' if v['held'] else 'slip'}" for k, v in trials.items())
          + f"; deep-saturation bar torque {deep['solver_joint_torque_nm']} N*m", flush=True)

    # ---------------------------------------------------------------- report
    args_cli.out.mkdir(parents=True, exist_ok=True)
    report["wall_time_s"] = rnd(time.time() - t_wall, 2)
    report["all_pass"] = all(c["pass"] for c in report["checks"].values())
    out_json = args_cli.out / "test_bravo7_class.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[test] all_pass={report['all_pass']} -> {out_json}", flush=True)
    for k, c in report["checks"].items():
        print(f"[test]   {k:<28} {'PASS' if c['pass'] else 'FAIL'}", flush=True)
