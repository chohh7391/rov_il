# Copyright (c) 2022-2026, leisaac Project.
# SPDX-License-Identifier: BSD-3-Clause

"""ROV sensor real-time visualizer using rerun-sdk (no isaaclab_visualizers dependency).

설치
----
    pip install rerun-sdk

사용법
------
teleop 스크립트에 아래처럼 추가:

    # 1) env 생성 직후 (한 번만)
    from rov_rerun_visualizer import init_rerun, log_rov_sensors
    init_rerun()

    # 2) 메인 루프 안, env.step() 직후
    step_count = 0
    ...
    env.step(actions)
    log_rov_sensors(env, step=step_count)
    step_count += 1

브라우저에서 http://localhost:9090 접속하면 실시간으로 확인 가능.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _import_rerun_sdk():
    import rerun as rr

    if not hasattr(rr, "init"):
        raise RuntimeError(
            "Installed package 'rerun' is not the Rerun robotics SDK. "
            "Uninstall it and install 'rerun-sdk>=0.22,<0.23' for the Isaac Lab environment."
        )
    return rr


def init_rerun(app_id: str = "rov-sensor-monitor", web_port: int = 9090) -> None:
    rr = _import_rerun_sdk()

    rr.init(app_id)
    rr.spawn()  # 로컬 viewer 실행
    # 또는 웹 뷰어: rr.serve_web_viewer(web_port=web_port)
    print(f"[ROV Visualizer] Rerun started (app_id={app_id})")


def log_rov_sensors(env, step, env_idx=0, image_every=5):
    try:
        rr = _import_rerun_sdk()
    except ImportError:
        return

    i = env_idx
    rr.set_time_sequence("step", step)

    def _scalar(path, value):
        rr.log(path, rr.Scalar(float(value)))

    # ── 1. Barometer ──────────────────────────────────────────────────
    try:
        baro = env.scene["barometer"]
        pressure = float(baro.data.pressure[i, 0].cpu())
        _scalar("sensors/barometer/pressure_Pa", pressure)
        depth_m = (pressure - baro.cfg.atmosphere_pressure) / (baro.cfg.water_density * baro.cfg.gravity)
        _scalar("sensors/barometer/depth_m", depth_m)
    except (KeyError, AttributeError):
        pass

    # ── 2. DVL velocity ───────────────────────────────────────────────
    try:
        dvl = env.scene["dvl"]
        vel = dvl.data.lin_vel_b[i].cpu().numpy()
        _scalar("sensors/dvl/velocity/vx", vel[0])
        _scalar("sensors/dvl/velocity/vy", vel[1])
        _scalar("sensors/dvl/velocity/vz", vel[2])
        _scalar("sensors/dvl/speed", float(np.linalg.norm(vel)))
    except (KeyError, AttributeError):
        pass

    # ── 3. DVL beam depth ─────────────────────────────────────────────
    try:
        dvl = env.scene["dvl"]
        depths = dvl.data.depth[i].cpu().numpy()
        hits   = dvl.data.beam_hit[i].cpu().numpy()
        for b in range(4):
            d = float(depths[b])
            if np.isfinite(d):
                _scalar(f"sensors/dvl/beam{b}/depth_m", d)
            _scalar(f"sensors/dvl/beam{b}/hit", float(hits[b]))
    except (KeyError, AttributeError):
        pass

    # ── 4, 5. Images (throttled) ──────────────────────────────────────
    if step % image_every == 0:
        def _image(path, img):
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            rr.log(path, rr.Image(img))

        try:
            img = env.scene["uw_camera"].data.output["uw_image"][i].cpu().numpy()
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            _image("sensors/uw_camera", img[:, :, :3])
        except (KeyError, AttributeError):
            pass

        try:
            img = env.scene["sonar"].data.output["sonar_image"][i].cpu().numpy()
            if img.ndim == 3 and img.shape[-1] > 1:
                img = img[:, :, 0]
            _image("sensors/sonar", img)
        except (KeyError, AttributeError):
            pass


def print_robot_dynamics_summary(env: "ManagerBasedRLEnv", env_idx: int = 0) -> None:
    """Print robot body mass / COM diagnostics for comparing USD physics setup."""
    try:
        robot = env.scene["robot"]
        data = robot.data
        body_names = list(data.body_names)
        masses = robot.root_physx_view.get_masses()[env_idx].detach().cpu().numpy()
        total_mass = float(np.sum(masses))
        print(f"[ROBOT_DYNAMICS][env={env_idx}]")
        print(f"  total_body_mass_kg={total_mass:.4f}")
        for name, mass in zip(body_names, masses):
            print(f"  body_mass_kg[{name}]={float(mass):.4f}")

        try:
            com_pos_b = data.body_com_pos_b[env_idx].detach().cpu().numpy()
            for name, com_pos in zip(body_names, com_pos_b):
                print(f"  body_com_pos_b[{name}]={com_pos}")
        except (AttributeError, IndexError, RuntimeError) as exc:
            print(f"  body_com_pos_b: unavailable ({exc})")

        try:
            root_com_pos_w = data.root_com_pos_w[env_idx].detach().cpu().numpy()
            root_link_pos_w = data.root_link_pos_w[env_idx].detach().cpu().numpy()
            print(f"  root_link_pos_w={root_link_pos_w}")
            print(f"  root_com_pos_w={root_com_pos_w}")
            print(f"  root_com_offset_w={root_com_pos_w - root_link_pos_w}")
        except (AttributeError, IndexError, RuntimeError) as exc:
            print(f"  root_com: unavailable ({exc})")

        try:
            if "rov_base" in body_names and "base" in body_names:
                base_idx = body_names.index("rov_base")
                arm_base_idx = body_names.index("base")
                pos_w = data.body_link_pos_w[env_idx]
                quat_w = data.body_link_quat_w[env_idx]
                rel_pos, rel_quat = math_utils.subtract_frame_transforms(
                    pos_w[base_idx].unsqueeze(0),
                    quat_w[base_idx].unsqueeze(0),
                    pos_w[arm_base_idx].unsqueeze(0),
                    quat_w[arm_base_idx].unsqueeze(0),
                )
                rel_roll, rel_pitch, rel_yaw = math_utils.euler_xyz_from_quat(rel_quat)
                rel_rpy = torch.stack((rel_roll, rel_pitch, rel_yaw), dim=-1)[0].detach().cpu().numpy()
                print(f"  rov_base.pos_w={pos_w[base_idx].detach().cpu().numpy()}")
                print(f"  rov_base.quat_w={quat_w[base_idx].detach().cpu().numpy()}")
                print(f"  arm_base.pos_w={pos_w[arm_base_idx].detach().cpu().numpy()}")
                print(f"  arm_base.quat_w={quat_w[arm_base_idx].detach().cpu().numpy()}")
                print(f"  rov_base_to_arm_base.pos={rel_pos[0].detach().cpu().numpy()}")
                print(f"  rov_base_to_arm_base.quat={rel_quat[0].detach().cpu().numpy()}")
                print(f"  rov_base_to_arm_base.rpy_rad={rel_rpy}")
        except (AttributeError, IndexError, RuntimeError, ValueError) as exc:
            print(f"  rov_base/arm_base frame comparison: unavailable ({exc})")
    except (KeyError, AttributeError, IndexError, RuntimeError) as exc:
        print(f"[ROBOT_DYNAMICS] unavailable ({exc})")


def print_rov_sensor_summary(env, step: int, env_idx: int = 0) -> None:
    """Print compact sensor values for quick runtime validation."""

    def _tensor_summary(name: str, value) -> str:
        arr = value.detach().cpu().numpy()
        finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
        if finite.size == 0:
            return f"{name}: shape={arr.shape}, dtype={arr.dtype}, all_non_finite"
        if arr.dtype == np.bool_:
            return f"{name}: shape={arr.shape}, dtype={arr.dtype}, true_count={int(arr.sum())}"
        if np.issubdtype(arr.dtype, np.number):
            return (
                f"{name}: shape={arr.shape}, dtype={arr.dtype}, "
                f"min={float(finite.min()):.4g}, max={float(finite.max()):.4g}"
            )
        return f"{name}: shape={arr.shape}, dtype={arr.dtype}"

    print(f"[SENSOR][step={step}][env={env_idx}]")

    try:
        robot = env.scene["robot"]
        print(f"  robot.root_pos_w={robot.data.root_pos_w[env_idx].detach().cpu().numpy()}")
        print(f"  robot.root_quat_w={robot.data.root_quat_w[env_idx].detach().cpu().numpy()}")
    except (KeyError, AttributeError, IndexError, RuntimeError) as exc:
        print(f"  robot: unavailable ({exc})")

    try:
        baro = env.scene["barometer"]
        pressure = baro.data.pressure[env_idx]
        depth_m = (pressure - baro.cfg.atmosphere_pressure) / (baro.cfg.water_density * baro.cfg.gravity)
        print(f"  barometer.pressure_Pa={float(pressure[0].detach().cpu()):.3f}, depth_m={float(depth_m[0].detach().cpu()):.4f}")
    except (KeyError, AttributeError, IndexError, RuntimeError) as exc:
        print(f"  barometer: unavailable ({exc})")

    try:
        dvl = env.scene["dvl"]
        print(f"  dvl.lin_vel_b={dvl.data.lin_vel_b[env_idx].detach().cpu().numpy()}")
        print(f"  dvl.depth={dvl.data.depth[env_idx].detach().cpu().numpy()}")
        print(f"  dvl.beam_hit={dvl.data.beam_hit[env_idx].detach().cpu().numpy()}")
    except (KeyError, AttributeError, IndexError, RuntimeError) as exc:
        print(f"  dvl: unavailable ({exc})")

    try:
        uw_camera = env.scene["uw_camera"]
        uw_data = uw_camera.data
        print(f"  uw_camera.pos_w={uw_data.pos_w[env_idx].detach().cpu().numpy()}")
        print(f"  uw_camera.quat_w_world={uw_data.quat_w_world[env_idx].detach().cpu().numpy()}")
        uw = uw_data.output
        print("  " + _tensor_summary("uw_camera.uw_image", uw["uw_image"][env_idx]))
        print("  " + _tensor_summary("uw_camera.depth", uw["depth"][env_idx]))
    except (KeyError, AttributeError, IndexError, RuntimeError) as exc:
        print(f"  uw_camera: unavailable ({exc})")

    try:
        sonar_sensor = env.scene["sonar"]
        sonar_data = sonar_sensor.data
        print(f"  sonar.pos_w={sonar_data.pos_w[env_idx].detach().cpu().numpy()}")
        print(f"  sonar.quat_w_world={sonar_data.quat_w_world[env_idx].detach().cpu().numpy()}")
        sonar = sonar_data.output
        print("  " + _tensor_summary("sonar.sonar_image", sonar["sonar_image"][env_idx]))
        print("  " + _tensor_summary("sonar.intensity", sonar["intensity"][env_idx]))
    except (KeyError, AttributeError, IndexError, RuntimeError) as exc:
        print(f"  sonar: unavailable ({exc})")


def print_arm_frame_summary(env: "ManagerBasedRLEnv", step: int, env_idx: int = 0) -> None:
    """Print arm pose/joint diagnostics relative to the moving ROV base."""

    def _print_relative_body_pose(label: str, body_names: list[str], data, base_idx: int, target_name: str) -> None:
        if target_name not in body_names:
            print(f"  {label}: unavailable (body '{target_name}' not found)")
            return

        target_idx = body_names.index(target_name)
        pos_w = data.body_link_pos_w[env_idx]
        quat_w = data.body_link_quat_w[env_idx]
        rel_pos, rel_quat = math_utils.subtract_frame_transforms(
            pos_w[base_idx].unsqueeze(0),
            quat_w[base_idx].unsqueeze(0),
            pos_w[target_idx].unsqueeze(0),
            quat_w[target_idx].unsqueeze(0),
        )
        rel_roll, rel_pitch, rel_yaw = math_utils.euler_xyz_from_quat(rel_quat)
        rel_rpy = torch.stack((rel_roll, rel_pitch, rel_yaw), dim=-1)[0]
        print(f"  {label}.pos_w={pos_w[target_idx].detach().cpu().numpy()}")
        print(f"  {label}.quat_w={quat_w[target_idx].detach().cpu().numpy()}")
        print(f"  rov_base_to_{label}.pos={rel_pos[0].detach().cpu().numpy()}")
        print(f"  rov_base_to_{label}.rpy_rad={rel_rpy.detach().cpu().numpy()}")

    print(f"[ARM_FRAME][step={step}][env={env_idx}]")

    try:
        robot = env.scene["robot"]
        data = robot.data
        body_names = list(data.body_names)
        if "rov_base" not in body_names:
            print("  rov_base: unavailable (body 'rov_base' not found)")
            return

        base_idx = body_names.index("rov_base")
        print(f"  rov_base.pos_w={data.body_link_pos_w[env_idx, base_idx].detach().cpu().numpy()}")
        print(f"  rov_base.quat_w={data.body_link_quat_w[env_idx, base_idx].detach().cpu().numpy()}")
        _print_relative_body_pose("arm_base", body_names, data, base_idx, "base")
        _print_relative_body_pose("gripper", body_names, data, base_idx, "gripper")

        joint_names = list(data.joint_names)
        arm_joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        joint_indices = [joint_names.index(name) for name in arm_joint_names if name in joint_names]
        printed_joint_names = [joint_names[index] for index in joint_indices]
        if joint_indices:
            joint_pos = data.joint_pos[env_idx, joint_indices].detach().cpu().numpy()
            joint_vel = data.joint_vel[env_idx, joint_indices].detach().cpu().numpy()
            print(f"  joint_names={printed_joint_names}")
            print(f"  joint_pos={joint_pos}")
            print(f"  joint_vel={joint_vel}")
            try:
                joint_pos_target = data.joint_pos_target[env_idx, joint_indices].detach().cpu().numpy()
                print(f"  joint_pos_target={joint_pos_target}")
                print(f"  joint_target_error={joint_pos_target - joint_pos}")
            except (AttributeError, IndexError, RuntimeError) as exc:
                print(f"  joint_pos_target: unavailable ({exc})")
    except (KeyError, AttributeError, IndexError, RuntimeError, ValueError) as exc:
        print(f"  robot arm frame: unavailable ({exc})")

    try:
        action_manager = env.action_manager
        for term_name in ["rov_action", "arm_action", "gripper_action"]:
            if term_name not in action_manager._terms:
                continue
            term = action_manager._terms[term_name]
            raw = term.raw_actions[env_idx].detach().cpu().numpy()
            processed = term.processed_actions[env_idx].detach().cpu().numpy()
            print(f"  action.{term_name}.raw={raw}")
            print(f"  action.{term_name}.processed={processed}")
    except (AttributeError, KeyError, IndexError, RuntimeError) as exc:
        print(f"  action terms: unavailable ({exc})")
