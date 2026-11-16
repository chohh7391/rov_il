"""Privileged scripted expert (simulator ground truth) producing 10-D actions for UWPickEnv.

approach : ROV P-control to the grasp station in front of the target cube (arm stowed, gripper opens)
reach    : ROV station-keeping, arm DLS-IK to a pre-grasp pose 6 cm above the cube, gripper pointing down
descend  : arm IK onto the cube
close    : gripper closes
lift     : arm IK lifts 15 cm
"""

from __future__ import annotations

import math

import numpy as np
import torch

import isaaclab.utils.math as mu
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

import uw_env as E


def _yaw(q):
    return mu.euler_xyz_from_quat(q.unsqueeze(0))[2][0].item()


class UWExpert:
    K_ROV = 1.2
    K_YAW = 1.5
    EE_MAX = 0.02            # max IK position step per control step (m)
    PRE_H = 0.07
    LIFT_H = 0.16
    GRIP_OPEN = 1.0
    GRIP_CLOSE = 0.0
    LIFT_RATE = 0.002
    # fixed-base (station-kept hull) mode: the arm is no longer slowed by the hull's motion, so approach the
    # cube gradually and close the gripper over several steps instead of slamming it shut on the light cube
    DESCEND_STEP = 0.005     # max grasp-target step per control step during descend (m)
    CLOSE_STEPS = 20         # gripper close ramp length (control steps)

    def __init__(self, env: "E.UWPickEnv", noise: float = 0.0, seed: int = 0):
        self.env, self.r = env, env.robot
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls",
                                        ik_params={"lambda_val": 0.05}),
            num_envs=1, device=env.dev)
        self.ik_ids = env.joint_ids[:5]
        self.jac_cols = [j + self.r.num_base_dofs for j in self.ik_ids]
        self.gb = env.grip_body_id
        self.lin_scale = np.array(env.servo.LIN_SCALE)
        self.noise, self.rng = noise, np.random.default_rng(seed)
        self.q_down_b = None
        self.g_off = None
        self.err = 0.0
        self.reset()

    # --------------------------------------------------------------- frames
    def _grip_pose_b(self):
        rp, rq = self.env.root_pose()
        gp = self.r.data.body_pos_w.torch[0, self.gb]
        gq = self.r.data.body_quat_w.torch[0, self.gb]
        pb, qb = mu.subtract_frame_transforms(rp[None], rq[None], gp[None], gq[None])
        return pb, qb, gp, gq

    def reset(self):
        # with a fixed (station-kept) hull there is no approach phase: start by reaching with the arm
        self.gentle = bool(getattr(self.env, "fixed_base", False))
        start_phase = "reach" if self.gentle else "approach"
        self.phase, self.pk, self.lift_start, self.err = start_phase, 0, None, 0.0
        self.hold_tgt = None
        self.corr = np.zeros(3)
        self.ik.reset()

    def calibrate(self):
        """Call once after env.reset(): measures arm geometry at q=0 (root frame) for the planar IK.
        SO-101: shoulder_lift / elbow_flex / wrist_flex rotate about parallel axes (pitch plane);
        positive angles pitch the chain forward-down. Joint axes ~ child body origins."""
        env = self.env
        env.set_arm([0.0] * 6, n=90)
        rp, rq = env.root_pose()
        qi = mu.quat_conjugate(rq[None])
        bp = env.robot.data.body_pos_w.torch[0]
        rel = lambda w: mu.quat_apply(qi, (w - rp)[None])[0].cpu().numpy()  # noqa: E731
        names = env.robot.body_names
        S = rel(bp[names.index("shoulder")])
        L = rel(bp[names.index("upper_arm")])
        Eb = rel(bp[names.index("lower_arm")])
        W = rel(bp[names.index("wrist")])
        G = rel(env.jaw_point())  # grasp point at q=0
        gp, gq = env.robot.data.body_pos_w.torch[0, self.gb], env.robot.data.body_quat_w.torch[0, self.gb]
        self.g_off = mu.quat_apply(mu.quat_conjugate(gq[None]), (env.jaw_point() - gp)[None])  # closed-gripper grasp pt
        self.pan_axis = S
        self.L = L
        v1, v2, v3 = Eb - L, W - Eb, G - W
        self.l1, self.a1 = float(np.hypot(v1[0], v1[2])), math.atan2(v1[2], v1[0])
        self.l2, self.a2 = float(np.hypot(v2[0], v2[2])), math.atan2(v2[2], v2[0])
        self.l3, self.a3 = float(np.hypot(v3[0], v3[2])), math.atan2(v3[2], v3[0])
        self.lat = float(G[1] - S[1])  # lateral offset of the grasp point from the pan axis at q=0
        # direction from the closed grasp point (fixed finger pad) toward the opening jaw, gripper frame
        jaw_c = env.jaw_point()
        env.set_arm([0.0, 0, 0, 0, 0, 1.0], n=60)
        gp2, gq2 = env.robot.data.body_pos_w.torch[0, self.gb], env.robot.data.body_quat_w.torch[0, self.gb]
        d_open = mu.quat_apply(mu.quat_conjugate(gq2[None]), (env.jaw_point() - jaw_c)[None])
        self.n_open_g = d_open / torch.linalg.vector_norm(d_open)
        # pan sign check
        env.set_arm([0.3, 0, 0, 0, 0, 0], n=60)
        self.pan_sign = 1.0 if rel(env.jaw_point())[1] > G[1] else -1.0
        env.set_arm([0.0] * 6, n=30)
        self.corr = np.zeros(3)
        print(f"[expert] l1={self.l1:.3f} a1={math.degrees(self.a1):.1f} l2={self.l2:.3f} a2={math.degrees(self.a2):.1f} "
              f"l3={self.l3:.3f} a3={math.degrees(self.a3):.1f} lat={self.lat:.3f} pan_sign={self.pan_sign}", flush=True)

    GRASP_SHIFT = 0.030
    GRASP_DZ = -0.012

    def grasp_target(self, cube: torch.Tensor) -> torch.Tensor:
        """Place the fixed finger pad beside the cube: shift opposite to the jaw-opening direction (horizontal)."""
        gq = self.r.data.body_quat_w.torch[0, self.gb]
        n = mu.quat_apply(gq[None], self.n_open_g)[0]
        n[2] = 0.0
        n = n / torch.linalg.vector_norm(n).clamp_min(1e-6)
        return cube - self.GRASP_SHIFT * n + torch.tensor([0.0, 0.0, self.GRASP_DZ], device=cube.device)

    def grasp_point(self) -> torch.Tensor:
        """Grasp point fixed to the gripper body (independent of jaw opening)."""
        gp = self.r.data.body_pos_w.torch[0, self.gb]
        gq = self.r.data.body_quat_w.torch[0, self.gb]
        return gp + mu.quat_apply(gq[None], self.g_off)[0]

    def _ik(self, grasp_target_w: torch.Tensor) -> torch.Tensor:
        """Analytic IK: pan toward target, then 2-link planar IK with the gripper pointing straight down."""
        env = self.env
        rp, rq = env.root_pose()
        t = mu.quat_apply(mu.quat_conjugate(rq[None]), (grasp_target_w - rp)[None])[0].cpu().numpy() + self.corr
        dx, dy = t[0] - self.pan_axis[0], t[1] - self.pan_axis[1]
        r_h = math.sqrt(max(dx * dx + dy * dy - self.lat ** 2, 1e-6))
        pan = math.atan2(dy, dx) - math.atan2(self.lat, r_h)
        # in-plane coordinates relative to the lift axis (distance along the pan direction, height)
        x = r_h - (self.L[0] - self.pan_axis[0])
        z = t[2] - self.L[2]
        phi3 = -math.pi / 2  # gripper pointing down
        wx, wz = x - self.l3 * math.cos(phi3), z - self.l3 * math.sin(phi3)
        d = min(math.hypot(wx, wz), self.l1 + self.l2 - 1e-4)
        c = (self.l1 ** 2 + d ** 2 - self.l2 ** 2) / (2 * self.l1 * d)
        beta = math.acos(max(-1.0, min(1.0, c)))
        gam = math.atan2(wz, wx)
        th1 = gam + beta                      # elbow-up
        e2x, e2z = wx - self.l1 * math.cos(th1), wz - self.l1 * math.sin(th1)
        th2 = math.atan2(e2z, e2x)
        q1 = self.a1 - th1
        q2 = (self.a2 - th2) - q1
        q3 = (self.a3 - phi3) - q1 - q2
        return torch.tensor([self.pan_sign * pan, q1, q2, q3, 0.0], device=env.dev, dtype=torch.float32)

    # --------------------------------------------------------------- policy
    def act(self) -> np.ndarray:
        env = self.env
        cube = env.cube_pos()
        rp, rq = env.root_pose()
        yaw = _yaw(rq)
        a = np.zeros(10, dtype=np.float32)
        q_now = env.joint_pos()
        arm = q_now[:5].clone()
        grip = float(q_now[5].item())
        # ROV station keeping toward the grasp station (world -> body)
        station = env.cube_spawn[env.target] + torch.tensor(env.station_offset, device=env.dev, dtype=torch.float32)
        e_w = station - rp
        e_b = mu.quat_apply(mu.quat_conjugate(rq[None]), e_w[None])[0].cpu().numpy()
        k_rov = self.K_ROV if self.phase == "approach" else 0.8
        a[0:3] = np.clip(k_rov * e_b / self.lin_scale, -1, 1)
        a[3] = float(np.clip(-self.K_YAW * yaw / 0.6, -1, 1))
        grasp_pt = self.grasp_point()
        if self.phase == "approach":
            arm = torch.tensor(E.STOW[:5], device=env.dev)
            g = self.GRIP_OPEN
            self.err = float(np.linalg.norm(e_b))
            if self.err < 0.04 and self.pk > 10:
                self._go("reach")
        elif self.phase == "reach":
            tgt = self.grasp_target(cube) + torch.tensor([0, 0, self.PRE_H], device=env.dev)
            arm = self._ik(tgt)
            g = self.GRIP_OPEN
            self.err = torch.linalg.vector_norm(tgt - grasp_pt).item()
            if (self.err < 0.015 and self.pk > 15) or self.pk > 120:
                self._go("descend")
        elif self.phase == "descend":
            if self.hold_tgt is None:
                self.hold_tgt = self.grasp_target(cube)
            if self.gentle:
                # interpolate from the pre-grasp height down to the grasp target
                pre = self.hold_tgt + torch.tensor([0, 0, self.PRE_H], device=env.dev)
                frac = min(1.0, self.pk * self.DESCEND_STEP / self.PRE_H)
                arm = self._ik(pre + (self.hold_tgt - pre) * frac)
            else:
                arm = self._ik(self.hold_tgt)
            g = self.GRIP_OPEN
            self.err = torch.linalg.vector_norm(self.hold_tgt - grasp_pt).item()
            done_interp = (not self.gentle) or self.pk * self.DESCEND_STEP >= self.PRE_H
            if (self.err < 0.012 and self.pk > 8 and done_interp) or self.pk > 60:
                self._go("close")
        elif self.phase == "close":
            arm = self._ik(self.hold_tgt)
            if self.gentle:
                g = max(self.GRIP_CLOSE, self.GRIP_OPEN - (self.GRIP_OPEN - self.GRIP_CLOSE) * self.pk / self.CLOSE_STEPS)
            else:
                g = self.GRIP_CLOSE
            self.err = grip
            if self.pk > (self.CLOSE_STEPS + 8 if self.gentle else 18):
                self._go("lift")
                self.lift_start = cube.clone()
        else:  # lift
            tgt = self.hold_tgt + torch.tensor([0, 0, min(self.LIFT_H, self.LIFT_RATE * self.pk)], device=env.dev)
            arm = self._ik(tgt)
            g = self.GRIP_CLOSE
            self.err = (cube[2] - self.lift_start[2]).item()
        a[4:9] = arm.cpu().numpy()
        a[9] = g
        if self.noise > 0 and self.phase in ("approach",):
            a[0:3] += self.rng.normal(0, self.noise, 3).astype(np.float32)
        self.pk += 1
        return a

    def _go(self, p):
        self.phase, self.pk = p, 0
        self.ik.reset()
