"""Minimal underwater pick-and-place environment on Isaac Lab 3.0 (single env, 30 Hz control).

Action (10-D): [rov_vx, rov_vy, rov_vz, rov_yaw_rate] in [-1, 1]  +  arm joint position targets (6, incl. gripper)
Observation (underwater-available sensors only):
  front, wrist : underwater camera images (OceanSim attenuation/backscatter), uint8 HxWx3
  state (10)   : arm joint positions (6) + DVL body velocity (3, noisy) + depth from barometer (1, noisy)
Success (as rov_lab pick_stone): target cube grasped (jaw point < 5 cm, gripper < 0.6 rad) and lifted >= 10 cm.
"""

from __future__ import annotations

import math

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as mu
from isaaclab.utils import instantiate

import uw_scene as U

JAW_OFFSET = (-0.021, -0.070, 0.02)  # rov_lab ee_frame "jaw" offset (grasp point)
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
STOW = (0.0, -0.3, 0.6, 0.6, 0.0, 0.0)


class UWPickEnv:
    def __init__(self, sim_cfg, seed=0, front_hw=(180, 320), wrist_hw=(240, 320), episode_s=20.0,
                 rov_jitter=(0.10, 0.10, 0.05, 5.0), cube_jitter=0.02, current=0.0, decimation=2, probe=False, debug_cam=False,
                 fixed_base=False, station_jitter=(0.02, 0.02, 0.01, 3.0), max_ang_vel_deg=None):
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        # fixed_base: the hull is locked at the grasp station (station-keeping assumed) and only the arm acts
        self.fixed_base = fixed_base
        self.station_jitter = station_jitter
        if max_ang_vel_deg is None:
            max_ang_vel_deg = 180.0 if fixed_base else 5.0
        self.scene = instantiate(U.make_scene_cfg(front_hw, wrist_hw, probe=probe, debug_cam=debug_cam,
                                                  fixed_base=fixed_base, max_ang_vel_deg=max_ang_vel_deg)(
            num_envs=1, env_spacing=8.0))
        self.sim.reset()
        self.dt = self.sim.get_physics_dt()
        self.dec = decimation
        self.max_steps = int(episode_s / (self.dt * self.dec))
        self.rov_jitter, self.cube_jitter, self.current_mag = rov_jitter, cube_jitter, current
        self.robot = self.scene["robot"]
        self.dev = self.robot.device
        self.base_id = self.robot.body_names.index("rov_base")
        self.jaw_id = self.robot.body_names.index("jaw")
        self.grip_body_id = self.robot.body_names.index("gripper")
        self.joint_ids = [self.robot.joint_names.index(j) for j in ARM_JOINTS]
        self.servo = U.ROVServo(self.robot, self.base_id, self.dt, 1, self.dev)
        self.hydro = U.Hydro(self.dev)
        self.jaw_off = torch.tensor([JAW_OFFSET], device=self.dev)
        self.target = 0
        self.spawn_z = torch.zeros(3, device=self.dev)
        self.cube_spawn = torch.zeros(3, 3, device=self.dev)
        self.current_w = torch.zeros(1, 3, device=self.dev)
        self.cmd = torch.zeros(1, 4, device=self.dev)
        self.q_tgt = torch.tensor([STOW], device=self.dev)
        # nominal start: 0.8 m behind and 0.3 m above the grasp station
        self.station_offset = np.array([-0.37, 0.02, -0.03])  # rov_base relative to target cube at grasp station
        self.start_offset = np.array([-0.90, 0.0, 0.25])

    # ------------------------------------------------------------------ helpers
    def cube_pos(self, i=None):
        i = self.target if i is None else i
        return self.scene[U.CUBES[i]].data.root_pos_w.torch[0]

    def jaw_point(self):
        p = self.robot.data.body_pos_w.torch[:, self.jaw_id]
        q = self.robot.data.body_quat_w.torch[:, self.jaw_id]
        return (p + mu.quat_apply(q, self.jaw_off))[0]

    def root_pose(self):
        return self.robot.data.root_pos_w.torch[0], self.robot.data.root_quat_w.torch[0]

    def joint_pos(self):
        return self.robot.data.joint_pos.torch[0, self.joint_ids]

    def grasped(self):
        d = torch.linalg.vector_norm(self.cube_pos() - self.jaw_point()).item()
        return d < 0.05 and self.joint_pos()[5].item() < 0.6

    def success(self):
        lifted = (self.cube_pos()[2] - self.spawn_z[self.target]).item() >= 0.10
        return lifted and self.grasped()

    # ------------------------------------------------------------------ reset / step
    def reset(self, target: int | None = None):
        sc, r = self.scene, self.robot
        self.target = int(self.rng.integers(0, 3)) if target is None else target
        origin = sc.env_origins
        for i, name in enumerate(U.CUBES):
            obj = sc[name]
            pose = obj.data.default_root_pose.torch.clone()
            pose[:, :3] += origin
            pose[:, 0] += float(self.rng.uniform(-self.cube_jitter, self.cube_jitter))
            pose[:, 1] += float(self.rng.uniform(-self.cube_jitter / 2, self.cube_jitter / 2))
            obj.write_root_pose_to_sim_index(root_pose=pose)
            obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=self.dev))
            self.spawn_z[i] = pose[0, 2]
            self.cube_spawn[i] = pose[0, :3]
        # ROV start: behind/above the target cube's nominal grasp station
        cube_nom = torch.tensor([U.CUBE_X, U.CUBE_YS[self.target], U.CUBE_Z], device=self.dev)
        if self.fixed_base:
            # hull station-kept at the grasp station of the target cube, with a small station-keeping error
            jx, jy, jz, jyaw = self.station_jitter
            start = self.cube_spawn[self.target] + torch.tensor(self.station_offset, device=self.dev, dtype=torch.float32)
            start = start - origin[0]
        else:
            jx, jy, jz, jyaw = self.rov_jitter
            start = cube_nom + torch.tensor(self.start_offset, device=self.dev, dtype=torch.float32)
            start[1] = float(np.mean(U.CUBE_YS))  # start centered on the cube row, not on the target
        start = start + torch.tensor([self.rng.uniform(-jx, jx), self.rng.uniform(-jy, jy), self.rng.uniform(-jz, jz)],
                                     device=self.dev, dtype=torch.float32)
        yaw = math.radians(float(self.rng.uniform(-jyaw, jyaw)))
        quat = torch.tensor([[0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]], device=self.dev)
        r.write_root_pose_to_sim_index(root_pose=torch.cat([start[None] + origin, quat], dim=1))
        r.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=self.dev))
        jp = r.data.default_joint_pos.torch.clone()
        jp[0, self.joint_ids] = torch.tensor(STOW, device=self.dev)
        r.write_joint_position_to_sim_index(position=jp)
        r.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(jp))
        self.q_tgt = torch.tensor([STOW], device=self.dev)
        self.cmd.zero_()
        self.servo.reset()
        if self.current_mag > 0:
            ang = float(self.rng.uniform(0, 2 * math.pi))
            self.current_w = torch.tensor([[self.current_mag * math.cos(ang), self.current_mag * math.sin(ang), 0.0]],
                                          device=self.dev)
        sc.reset()
        self.k = 0
        for _ in range(4):
            self._physics_step()
        return self.observe()

    def set_arm(self, q, n=60):
        """Drive the arm to joint configuration q (6) with the ROV holding still (used for calibration)."""
        jp = self.robot.data.joint_pos.torch.clone()
        jp[0, self.joint_ids] = torch.tensor(q, device=self.dev, dtype=torch.float32)
        self.robot.write_joint_position_to_sim_index(position=jp)
        self.q_tgt = torch.tensor([q], device=self.dev, dtype=torch.float32)
        self.cmd.zero_()
        for _ in range(n):
            self._physics_step()

    def _physics_step(self):
        r = self.robot
        if self.fixed_base:
            r.set_joint_position_target_index(target=self.q_tgt, joint_ids=self.joint_ids)
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.dt)
            return
        q, lin_b, ang_b = self.servo.body_vel()
        f_s, t_s = self.servo.wrench_b(self.cmd)
        cur_b = mu.quat_apply(mu.quat_conjugate(q), self.current_w)
        f_h, t_h = self.hydro.wrench_b(q, lin_b, ang_b, cur_b)
        r.permanent_wrench_composer.set_forces_and_torques_index(
            forces=(f_s + f_h).unsqueeze(1), torques=(t_s + t_h).unsqueeze(1), body_ids=[self.base_id], is_global=False)
        r.set_joint_position_target_index(target=self.q_tgt, joint_ids=self.joint_ids)
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.dt)

    def step(self, action: np.ndarray):
        """action: 10-D [rov_vx, rov_vy, rov_vz, rov_yaw_rate, arm(6)], or 6-D arm-only when ``fixed_base``."""
        a = torch.as_tensor(np.asarray(action, dtype=np.float32), device=self.dev).reshape(1, -1)
        if a.shape[1] == 6:
            a = torch.cat([torch.zeros(1, 4, device=self.dev), a], dim=1)
        self.cmd = a[:, :4].clamp(-1, 1)
        lim = self.robot.data.soft_joint_pos_limits.torch[0, self.joint_ids]
        self.q_tgt = torch.max(torch.min(a[:, 4:], lim[:, 1]), lim[:, 0])
        for _ in range(self.dec):
            self._physics_step()
        self.k += 1
        succ = self.success()
        done = succ or self.k >= self.max_steps
        return self.observe(), succ, done

    def observe(self):
        q, lin_b, _ = self.servo.body_vel()
        dvl = lin_b[0] + torch.randn(3, device=self.dev) * 0.01
        depth = torch.tensor([U.WATER_SURFACE - self.robot.data.root_pos_w.torch[0, 2].item()], device=self.dev)
        depth = depth + torch.randn(1, device=self.dev) * 0.005
        state = torch.cat([self.joint_pos(), dvl, depth]).cpu().numpy().astype(np.float32)
        return {
            "front": U.cam_images(self.scene["front"])[0].cpu().numpy(),
            "wrist": U.cam_images(self.scene["wrist"])[0].cpu().numpy(),
            "state": state,
        }
