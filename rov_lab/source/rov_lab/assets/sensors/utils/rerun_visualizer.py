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

import numpy as np


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

        # Underwater RGB cameras. Both wrist and front are UW cameras whose
        # post-processed frame is exposed under the "uw_image" output key.
        for cam_name in ("wrist", "front"):
            try:
                img = env.scene[cam_name].data.output["uw_image"][i].cpu().numpy()
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=-1)
                _image(f"sensors/{cam_name}", img[:, :, :3])
            except (KeyError, AttributeError):
                pass

        # Imaging sonar.
        try:
            img = env.scene["sonar"].data.output["sonar_image"][i].cpu().numpy()
            if img.ndim == 3 and img.shape[-1] > 1:
                img = img[:, :, 0]
            _image("sensors/sonar", img)
        except (KeyError, AttributeError):
            pass
