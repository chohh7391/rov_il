"""Scripted privileged expert for the valve-panel task (reach -> approach -> grasp -> quarter turn -> release).

Reads the true valve and end-effector poses and emits the env's 13-D action (ROV velocity 6, arm end-effector delta 6
in the ROV base frame, parallel-gripper delta 1). It validates the layout (every valve reachable and turnable) and is
the demonstration source for imitation learning, like the pick-stone expert of the 2026 experiment.

Frames: each valve link has its z axis on the valve axis pointing out of the panel (towards the ROV). The jaws grasp
the paddle with the jaw axis (jaw_base_link +x) along -z_valve and the fingers (jaw_base_link +-y) across the paddle's
thin axis. Opening the valve turns it from 0 to -90 deg about +z_valve; the expert tracks the paddle frame of a
desired angle that it ramps down, so the wrist rolls with the valve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

import isaaclab.utils.math as math_utils

from rov_lab.assets.objects.rov_bop_panel import VALVE_NAMES
from rov_lab.assets.robots.i90_bravo7 import EE_LINK, GRASP_POINT_OFFSET_M
from rov_lab.utils.tensor import as_torch

from .mdp.valve import TARGET_ATTR

# paddle centre in the valve link frame and the paddle's thin axis (from the rov_bop_panel meshes)
PADDLE_CENTER_Z: dict[str, float] = {"big": 0.03, "small": 0.04}
VALVE_KIND: dict[str, str] = {"valve_1": "big", "valve_2": "big", "valve_3": "big", "valve_4": "small", "valve_5": "small"}
THIN_AXIS: dict[str, int] = {"big": 0, "small": 1}  # link x for the big paddles, link y for the small ones

PHASES: tuple[str, ...] = ("pregrasp", "approach", "close", "turn", "release", "done")


@dataclass
class ValveExpertCfg:
    pregrasp_standoff: float = 0.10
    """TCP distance in front of the paddle centre (along +z_valve) before the straight approach (m)."""
    grasp_offset: float = 0.025
    """TCP stop distance in front of the paddle centre (along +z_valve, m). The jaw tips are 40 mm past the TCP, so
    this leaves them 5 mm above the valve disk (big and small valves alike); the pads still cover 35 of the paddle's
    40 mm height."""
    lin_speed: float = 0.24
    """Max end-effector speed while reaching (m/s); the straight approach uses half of it."""
    ang_speed: float = 2.4
    """Max end-effector angular speed (rad/s)."""
    pos_tol: float = 0.008
    rot_tol: float = 0.06
    close_time: float = 0.42
    turn_speed: float = 0.72
    """Desired-valve-angle ramp (rad/s)."""
    turn_target: float = -1.62
    """Desired valve angle at the end of the turn (slightly past the -pi/2 stop to keep the valve pressed on it)."""
    release_time: float = 0.67
    gripper_open: dict[str, float] = None  # type: ignore[assignment]
    """Finger opening per valve kind (m): the small valves sit in 95 mm pockets, so the jaws open only 40 mm there."""
    phase_timeout: float = 6.7
    """Give up if the reach or approach takes longer than this (s)."""
    finger_limits: tuple[float, float] = (0.0, 0.05)
    """Finger joint range (m), for the absolute gripper label."""

    def __post_init__(self) -> None:
        if self.gripper_open is None:
            self.gripper_open = {"big": 0.03, "small": 0.018}


class ValveExpert:
    """Batched state machine; one instance per env (all envs advance in lock step but keep their own phase)."""

    def __init__(self, env: Any, cfg: ValveExpertCfg | None = None) -> None:
        self.env = env
        self.cfg = cfg or ValveExpertCfg()
        self.robot = env.scene["robot"]
        self.panel = env.scene["panel"]
        self.ee_id = self.robot.find_bodies(EE_LINK)[0][0]
        self.valve_body_ids = [self.panel.find_bodies(v)[0][0] for v in VALVE_NAMES]
        self.valve_joint_ids = [self.panel.find_joints(f"{v}_joint")[0][0] for v in VALVE_NAMES]
        finger_ids = self.robot.find_joints("finger_.*_joint")[0]
        self.finger_ids = finger_ids
        n, dev = env.num_envs, env.device
        self.tcp_offset = torch.tensor(GRASP_POINT_OFFSET_M, device=dev).expand(n, 3)
        self.phase = torch.zeros(n, dtype=torch.long, device=dev)
        self.phase_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.finger_sign = torch.ones(n, device=dev)
        self.q_des = torch.zeros(n, device=dev)
        dt = float(env.step_dt)
        c = self.cfg
        self.max_step_pos = c.lin_speed * dt
        self.max_step_rot = c.ang_speed * dt
        self.turn_step = c.turn_speed * dt
        self.close_steps = max(1, round(c.close_time / dt))
        self.release_steps = max(1, round(c.release_time / dt))
        self.timeout_steps = max(1, round(c.phase_timeout / dt))
        self.label_pose_b = torch.zeros(n, 7, device=dev)
        """Clean expert command as an absolute jaw pose in the ROV base frame (pos 3 + quat xyzw 4)."""
        self.label_grip = torch.zeros(n, device=dev)
        """Clean gripper command as an absolute opening in [-1, 1] (closed .. open)."""

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.phase[ids] = 0
        self.phase_steps[ids] = 0
        self.finger_sign[ids] = 0.0  # chosen on the first step
        self.q_des[ids] = 0.0

    # ------------------------------------------------------------------------------------------------------------------
    def _target_valve(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
        tgt = getattr(self.env, TARGET_ATTR)
        body_ids = torch.tensor(self.valve_body_ids, device=self.env.device)[tgt]
        joint_ids = torch.tensor(self.valve_joint_ids, device=self.env.device)[tgt]
        ar = torch.arange(self.env.num_envs, device=self.env.device)
        pos = as_torch(self.panel.data.body_pos_w)[ar, body_ids]
        quat = as_torch(self.panel.data.body_quat_w)[ar, body_ids]
        q = as_torch(self.panel.data.joint_pos)[ar, joint_ids]
        kinds = [VALVE_KIND[VALVE_NAMES[int(t)]] for t in tgt]
        return pos, quat, q, tgt, kinds

    def _grasp_frame(self, valve_pos: torch.Tensor, valve_quat: torch.Tensor, kinds: list[str],
                     q_offset: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Paddle centre, jaw orientation and valve axis (world) for the paddle rotated by ``q_offset`` about z."""
        n, dev = valve_pos.shape[0], valve_pos.device
        rot_z = math_utils.quat_from_angle_axis(q_offset, torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3))
        quat = math_utils.quat_mul(valve_quat, rot_z)
        r = math_utils.matrix_from_quat(quat)  # columns = link axes in world
        z_v = r[:, :, 2]
        thin = torch.stack([r[i, :, THIN_AXIS[k]] for i, k in enumerate(kinds)])
        cz = torch.tensor([PADDLE_CENTER_Z[k] for k in kinds], device=dev)
        center = valve_pos + z_v * cz[:, None]
        x_j = -z_v
        y_j = thin * torch.where(self.finger_sign == 0, torch.ones_like(self.finger_sign), self.finger_sign)[:, None]
        z_j = torch.cross(x_j, y_j, dim=-1)
        jaw_quat = math_utils.quat_from_matrix(torch.stack([x_j, y_j, z_j], dim=-1))
        return center, jaw_quat, z_v

    def act(self) -> torch.Tensor:
        env, c = self.env, self.cfg
        n, dev = env.num_envs, env.device
        root_pos = as_torch(self.robot.data.root_pos_w)
        root_quat = as_torch(self.robot.data.root_quat_w)
        ee_pos = as_torch(self.robot.data.body_pos_w)[:, self.ee_id]
        ee_quat = as_torch(self.robot.data.body_quat_w)[:, self.ee_id]
        valve_pos, valve_quat, q, _, kinds = self._target_valve()

        # choose the finger direction (+-thin axis) closest to the current jaw orientation, once per episode
        unset = self.finger_sign == 0
        if unset.any():
            self.finger_sign[unset] = 1.0
            _, qa, _ = self._grasp_frame(valve_pos, valve_quat, kinds, torch.zeros(n, device=dev))
            self.finger_sign[unset] = -1.0
            _, qb, _ = self._grasp_frame(valve_pos, valve_quat, kinds, torch.zeros(n, device=dev))
            da = math_utils.quat_error_magnitude(qa, ee_quat)
            db = math_utils.quat_error_magnitude(qb, ee_quat)
            self.finger_sign[unset] = torch.where(da <= db, 1.0, -1.0)[unset]

        # desired valve angle: hold at the current angle until the turn, then ramp down
        turning = self.phase == PHASES.index("turn")
        self.q_des = torch.where(turning, torch.maximum(self.q_des - self.turn_step, torch.full_like(q, c.turn_target)), q)
        center, jaw_quat, z_v = self._grasp_frame(valve_pos, valve_quat, kinds, self.q_des - q)

        standoff = torch.zeros(n, device=dev)
        standoff = torch.where(self.phase == PHASES.index("pregrasp"), torch.full_like(standoff, c.pregrasp_standoff), standoff)
        standoff = torch.where(self.phase >= PHASES.index("approach"), torch.full_like(standoff, c.grasp_offset), standoff)
        release = self.phase >= PHASES.index("release")
        standoff = torch.where(release, torch.full_like(standoff, c.pregrasp_standoff), standoff)
        tcp_target = center + z_v * standoff[:, None]
        jaw_target = tcp_target - math_utils.quat_apply(jaw_quat, self.tcp_offset)

        # end-effector delta in the ROV base frame, clipped
        dpos_w = jaw_target - ee_pos
        drot_w = math_utils.axis_angle_from_quat(math_utils.quat_mul(jaw_quat, math_utils.quat_conjugate(ee_quat)))
        tcp_err = (ee_pos + math_utils.quat_apply(ee_quat, self.tcp_offset) - tcp_target).norm(dim=-1)
        rot_err = drot_w.norm(dim=-1)
        step_pos = torch.where(self.phase == PHASES.index("approach"), torch.full_like(rot_err, self.max_step_pos / 2),
                               torch.full_like(rot_err, self.max_step_pos))
        dpos_w = dpos_w * torch.clamp(step_pos / (dpos_w.norm(dim=-1) + 1e-9), max=1.0)[:, None]
        drot_w = drot_w * torch.clamp(self.max_step_rot / (rot_err + 1e-9), max=1.0)[:, None]
        root_inv = math_utils.quat_conjugate(root_quat)
        dpos_b = math_utils.quat_apply(root_inv, dpos_w)
        drot_b = math_utils.quat_apply(root_inv, drot_w)
        if (self.phase == PHASES.index("done")).any():
            done = self.phase == PHASES.index("done")
            dpos_b[done] = 0.0
            drot_b[done] = 0.0

        # gripper: open until closing, closed through the turn, open on release (relative command, scale 0.01 m)
        finger_target = as_torch(self.robot.data.joint_pos_target)[:, self.finger_ids].mean(dim=1)
        closing = (self.phase == PHASES.index("close")) | turning
        open_w = torch.tensor([c.gripper_open[k] for k in kinds], device=dev)
        desired_open = torch.where(closing, torch.zeros_like(finger_target), open_w)
        grip = torch.clamp((desired_open - finger_target) / 0.01, -0.5, 0.5)

        # labels: the same clean command as an absolute jaw pose (base frame) and absolute opening
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos, root_quat, ee_pos, ee_quat)
        self.label_pose_b[:, :3] = ee_pos_b + dpos_b
        self.label_pose_b[:, 3:] = math_utils.quat_mul(
            math_utils.quat_from_angle_axis(drot_b.norm(dim=-1), drot_b / (drot_b.norm(dim=-1, keepdim=True) + 1e-12)),
            ee_quat_b)
        lo, hi = c.finger_limits
        new_open = torch.clamp(finger_target + 0.01 * grip, lo, hi)
        self.label_grip[:] = 2.0 * (new_open - lo) / (hi - lo) - 1.0

        # phase transitions
        self.phase_steps += 1
        ph, st = self.phase, self.phase_steps
        reached = (tcp_err < c.pos_tol) & (rot_err < c.rot_tol)
        nxt = ph.clone()
        nxt = torch.where((ph == 0) & reached, 1, nxt)
        nxt = torch.where((ph == 1) & (tcp_err < c.pos_tol * 0.75), 2, nxt)
        nxt = torch.where((ph == 2) & (st >= self.close_steps), 3, nxt)
        nxt = torch.where((ph == 3) & (q <= -0.5 * torch.pi * 0.97), 4, nxt)
        nxt = torch.where((ph == 4) & (st >= self.release_steps), 5, nxt)
        nxt = torch.where((ph < 2) & (st >= self.timeout_steps), 5, nxt)  # give up
        changed = nxt != ph
        self.phase = nxt
        self.phase_steps = torch.where(changed, torch.zeros_like(st), st)

        action = torch.zeros(n, 13, device=dev)
        action[:, 6:9] = dpos_b
        action[:, 9:12] = drot_b
        action[:, 12] = grip
        self.last_errors = {"tcp_err": tcp_err, "rot_err": rot_err}
        return action
