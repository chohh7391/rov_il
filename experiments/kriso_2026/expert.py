"""Privileged scripted expert for underwater pick-and-place (BlueROV + SO-101).

Phases (gains/offsets tuned in sim):
  approach : ROV body-velocity servo moves the vehicle so the jaw sits at a pre-grasp point near the cube
  align    : arm relative-IK brings the jaw to the pre-grasp point, gripper opens
  descend  : jaw moves onto the cube
  close    : gripper closes
  lift     : arm lifts (and ROV heaves up) until the success condition fires
Output: 14-D action = [rov_v(6), ee_delta_pose(6), d_shoulder_pan, d_gripper].
"""

from __future__ import annotations

import numpy as np
import torch
import isaaclab.utils.math as mu

CUBES = ("red_cube", "green_cube", "blue_cube")


class ScriptedExpert:
    # geometry / gains (overridable)
    PRE_OFFSET = np.array([0.0, 0.0, 0.07])     # jaw pre-grasp point relative to cube (world)
    GRASP_OFFSET = np.array([0.0, 0.0, 0.0])    # jaw grasp point relative to cube (world)
    APPROACH_TOL = 0.05
    ALIGN_TOL = 0.012
    GRASP_TOL = 0.010
    K_ROV = 1.0          # (m/s) per m
    K_EE = 0.25          # fraction of error applied per control step
    EE_STEP = 0.006      # max ee translation per step (m)
    GRIP_OPEN = 1.0      # rad
    GRIP_RATE = 0.04     # rad per step
    PAN_K = 0.0          # shoulder-pan correction gain (0 = disabled)
    LIFT_STEP = 0.004
    LIFT_ROV = 0.25      # heave command fraction during lift

    def __init__(self, env, noise: float = 0.0, rng: np.random.Generator | None = None):
        self.env = env
        self.robot = env.scene["robot"]
        self.ee = env.scene["ee_frame"]
        term = env.action_manager.get_term("rov_action")
        self.vscale = np.array(term.cfg.lin_vel_scale, dtype=np.float32)
        self.grip_idx = self.robot.joint_names.index("gripper")
        self.noise = noise
        self.rng = rng or np.random.default_rng(0)
        self.reset()

    def reset(self):
        self.phase = "approach"
        self.k = 0
        self.phase_k = 0
        self.last_err = None
        self.grip = 0.0

    def _set(self, phase):
        self.phase, self.phase_k = phase, 0

    def act(self) -> np.ndarray:
        env = self.env
        tid = int(env.pick_stone_target_color_id[0].item())
        p_cube = env.scene[CUBES[tid]].data.root_pos_w[0]
        p_jaw = self.ee.data.target_pos_w[0, 1]
        q_root = self.robot.data.root_quat_w[0]
        q_inv = mu.quat_conjugate(q_root)
        self.grip = float(self.robot.data.joint_pos[0, self.grip_idx].item())
        dev = p_cube.device

        def to_root(v_w):
            return mu.quat_apply(q_inv, torch.as_tensor(v_w, dtype=torch.float32, device=dev)).cpu().numpy()

        a = np.zeros(14, dtype=np.float32)
        pre = (p_cube + torch.as_tensor(self.PRE_OFFSET, dtype=torch.float32, device=dev)) - p_jaw
        grasp = (p_cube + torch.as_tensor(self.GRASP_OFFSET, dtype=torch.float32, device=dev)) - p_jaw
        pre_r, grasp_r = to_root(pre), to_root(grasp)

        def ee_move(err_r):
            d = self.K_EE * err_r
            n = np.linalg.norm(d)
            if n > self.EE_STEP:
                d = d * self.EE_STEP / n
            a[6:9] = d

        if self.phase == "approach":
            v = self.K_ROV * pre_r
            a[0:3] = np.clip(v / self.vscale, -1.0, 1.0)
            self.last_err = np.round(np.linalg.norm(pre_r), 3)
            a[13] = self.GRIP_RATE if self.grip < self.GRIP_OPEN else 0.0
            if np.linalg.norm(pre_r) < self.APPROACH_TOL:
                self._set("align")
        elif self.phase == "align":
            ee_move(pre_r)
            a[13] = self.GRIP_RATE if self.grip < self.GRIP_OPEN else 0.0
            self.last_err = np.round(np.linalg.norm(pre_r), 3)
            if np.linalg.norm(pre_r) < self.ALIGN_TOL and self.grip >= self.GRIP_OPEN - 0.05:
                self._set("descend")
            elif self.phase_k > 150 and np.linalg.norm(pre_r) > 3 * self.APPROACH_TOL:
                self._set("approach")
        elif self.phase == "descend":
            ee_move(grasp_r)
            self.last_err = np.round(np.linalg.norm(grasp_r), 3)
            if np.linalg.norm(grasp_r) < self.GRASP_TOL or self.phase_k > 90:
                self._set("close")
        elif self.phase == "close":
            a[13] = -self.GRIP_RATE
            self.last_err = round(self.grip, 3)
            if self.phase_k > 30:
                self._set("lift")
        elif self.phase == "lift":
            a[13] = -0.01
            a[8] = self.LIFT_STEP
            a[2] = self.LIFT_ROV
            self.last_err = np.round(float(p_cube[2].item()), 3)

        if self.PAN_K and self.phase in ("align", "descend"):
            a[12] = float(np.clip(self.PAN_K * pre_r[1], -0.01, 0.01))
        if self.noise > 0 and self.phase in ("approach", "align"):
            a[0:3] += self.rng.normal(0, self.noise, 3).astype(np.float32)
        self.k += 1
        self.phase_k += 1
        return a
