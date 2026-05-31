"""ROV body velocity action term.

Accepts a 6-DOF body-frame target velocity command and converts it to a wrench
on the ROV root rigid body. OceanSim's original MHL example applies keyboard
force/torque directly; this term intentionally adds a velocity servo because the
ROV carries an arm and direct force commands tend to excite articulation motion.

The action is interpreted as signed axis commands ``[vx, vy, vz, wx, wy, wz]`` in
the ROV body frame. Held axes ramp the target velocity up to the configured
maximum; released axes reset to zero. The resulting wrench is rotated to world
frame before being applied through Isaac Lab's permanent wrench composer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ROVVelocityAction(ActionTerm):
    """Applies a 6-DOF body-frame velocity servo to the ROV base body.

    The action vector is ``(vx, vy, vz, wx, wy, wz)``. Components define signed
    axis commands. While an axis is held, the corresponding target velocity
    ramps toward ``cfg.lin_vel_scale`` or ``cfg.ang_vel_scale``. A proportional
    velocity servo then computes a force/torque command, which is clamped to
    ``[-force_limit, force_limit]`` / ``[-torque_limit, torque_limit]``.

    .. note::
        The target asset must be an :class:`Articulation` that contains the ROV
        base body. The action is applied through Isaac Lab's permanent wrench
        composer, which forwards a per-step external wrench to PhysX. Make sure
        the ROV's ``RigidBodyAPI`` has ``disableGravity = True`` if you want
        neutrally buoyant behavior.
    """

    cfg: ROVVelocityActionCfg
    _asset: Articulation

    def __init__(self, cfg: ROVVelocityActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(f"Expected one match for the ROV body name: {self.cfg.body_name}. Found {len(body_ids)}: {body_names}.")
        self._rov_body_idx = body_ids[0]
        self._rov_body_name = body_names[0]

        control_body_name = self.cfg.control_body_name or self.cfg.body_name
        control_body_ids, control_body_names = self._asset.find_bodies(control_body_name)
        if len(control_body_ids) != 1:
            raise ValueError(
                f"Expected one match for the ROV control frame body name: {control_body_name}. "
                f"Found {len(control_body_ids)}: {control_body_names}."
            )
        self._control_body_idx = control_body_ids[0]
        self._control_body_name = control_body_names[0]

        # action buffers
        self._raw_actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._target_lin_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_ang_vel_b = torch.zeros(self.num_envs, 3, device=self.device)

        # force / torque buffers in the layout expected by the wrench composer
        self._forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._torques = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # cache scaling / limits as tensors on device for fast computation
        self._lin_vel_scale = torch.tensor(cfg.lin_vel_scale, device=self.device, dtype=torch.float32)
        self._ang_vel_scale = torch.tensor(cfg.ang_vel_scale, device=self.device, dtype=torch.float32)
        self._kp_linear = torch.tensor(cfg.kp_linear, device=self.device, dtype=torch.float32)
        self._kp_angular = torch.tensor(cfg.kp_angular, device=self.device, dtype=torch.float32)
        self._force_limit = torch.tensor(cfg.force_limit, device=self.device, dtype=torch.float32)
        self._torque_limit = torch.tensor(cfg.torque_limit, device=self.device, dtype=torch.float32)
        self._lin_vel_ramp_rate = torch.tensor(cfg.lin_vel_ramp_rate, device=self.device, dtype=torch.float32)
        self._ang_vel_ramp_rate = torch.tensor(cfg.ang_vel_ramp_rate, device=self.device, dtype=torch.float32)
        self._action_deadband = float(cfg.action_deadband)
        self._dt = float(env.step_dt)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        """
        actions: RL 에이전트나 조이스틱에서 들어오는 목표 속도 
                 [v_x, v_y, v_z, w_x, w_y, w_z] (바디 프레임 기준)
        """
        self._raw_actions[:] = actions
        
        self._target_lin_vel_b[:] = self._ramp_targets(
            current=self._target_lin_vel_b,
            commands=actions[:, 0:3],
            max_values=self._lin_vel_scale,
            ramp_rates=self._lin_vel_ramp_rate,
        )
        self._target_ang_vel_b[:] = self._ramp_targets(
            current=self._target_ang_vel_b,
            commands=actions[:, 3:6],
            max_values=self._ang_vel_scale,
            ramp_rates=self._ang_vel_ramp_rate,
        )
        
        # processed_actions 버퍼에 기록 (옵저베이션 등에서 사용될 수 있음)
        self._processed_actions[:, 0:3] = self._target_lin_vel_b
        self._processed_actions[:, 3:6] = self._target_ang_vel_b

    def _ramp_targets(
        self,
        current: torch.Tensor,
        commands: torch.Tensor,
        max_values: torch.Tensor,
        ramp_rates: torch.Tensor,
    ) -> torch.Tensor:
        """Ramp velocity targets while an action axis is held, and reset released axes to zero."""
        active = commands.abs() > self._action_deadband
        command_sign = torch.sign(commands)
        max_targets = command_sign * commands.abs().clamp(max=1.0) * max_values
        increments = ramp_rates * self._dt

        same_direction = torch.sign(current) == command_sign
        next_targets = torch.where(same_direction, current + command_sign * increments, command_sign * increments)
        next_targets = torch.clamp(next_targets, -max_targets.abs(), max_targets.abs())
        return torch.where(active, next_targets, torch.zeros_like(current))

    def apply_actions(self):
        # 1. ROV의 현재 상태 가져오기 (월드 기준)
        body_quat_w = self._asset.data.body_quat_w[:, self._rov_body_idx]
        control_quat_w = self._asset.data.body_quat_w[:, self._control_body_idx]
        lin_vel_w = self._asset.data.body_lin_vel_w[:, self._rov_body_idx]
        ang_vel_w = self._asset.data.body_ang_vel_w[:, self._rov_body_idx]

        # 2. Express current velocity in the configured control frame. The wrench can be applied
        # to a different heavy hull body while preserving the teleoperation/body-axis convention.
        control_quat_inv = math_utils.quat_conjugate(control_quat_w)
        curr_lin_vel_b = math_utils.quat_apply(control_quat_inv, lin_vel_w)
        curr_ang_vel_b = math_utils.quat_apply(control_quat_inv, ang_vel_w)

        # 3. 제어 오차(Error) 계산
        lin_vel_error = self._target_lin_vel_b - curr_lin_vel_b
        ang_vel_error = self._target_ang_vel_b - curr_ang_vel_b

        force_b = self._kp_linear * lin_vel_error
        torque_b = self._kp_angular * ang_vel_error

        # Clamp in body frame so the actuator limits stay aligned with the ROV axes.
        force_b = torch.clamp(force_b, -self._force_limit, self._force_limit)
        torque_b = torch.clamp(torque_b, -self._torque_limit, self._torque_limit)

        # 4. 계산된 바디 프레임 힘/토크를 월드 프레임으로 변환하여 PhysX에 적용
        forces_w = math_utils.quat_apply(control_quat_w, force_b)
        torques_w = math_utils.quat_apply(control_quat_w, torque_b)

        self._asset.permanent_wrench_composer.set_forces_and_torques(
            forces=forces_w.unsqueeze(1),
            torques=torques_w.unsqueeze(1),
            body_ids=[self._rov_body_idx],
            is_global=True,
        )

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            self._raw_actions[:] = 0.0
            self._processed_actions[:] = 0.0
            self._target_lin_vel_b[:] = 0.0
            self._target_ang_vel_b[:] = 0.0
            self._forces[:] = 0.0
            self._torques[:] = 0.0
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0
            self._target_lin_vel_b[env_ids] = 0.0
            self._target_ang_vel_b[env_ids] = 0.0
            self._forces[env_ids] = 0.0
            self._torques[env_ids] = 0.0


@configclass
class ROVVelocityActionCfg(ActionTermCfg):
    """Configuration for :class:`ROVVelocityAction`."""

    class_type: type[ActionTerm] = ROVVelocityAction

    body_name: str = MISSING
    """Name of the body where the external wrench is applied."""

    control_body_name: str | None = None
    """Body whose local frame defines the velocity/action axes.

    If ``None``, uses :attr:`body_name`.
    """

    lin_vel_scale: tuple[float, float, float] = (5.0, 5.0, 5.0)
    """Per-axis scale applied to the linear velocity action (m/s)."""

    ang_vel_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Per-axis scale applied to the angular velocity action (rad/s)."""

    kp_linear: tuple[float, float, float] = (25.0, 25.0, 30.0)
    """Per-axis proportional gain for linear velocity control."""

    kp_angular: tuple[float, float, float] = (12.0, 12.0, 18.0)
    """Per-axis proportional gain for angular velocity control."""

    force_limit: tuple[float, float, float] = (100.0, 100.0, 100.0)
    """Symmetric body-frame clip on each force component (Newtons)."""

    torque_limit: tuple[float, float, float] = (100.0, 100.0, 100.0)
    """Symmetric body-frame clip on each torque component (N·m)."""

    lin_vel_ramp_rate: tuple[float, float, float] = (2.0, 2.0, 2.0)
    """Per-axis linear velocity target ramp rate while an action is held (m/s^2)."""

    ang_vel_ramp_rate: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Per-axis angular velocity target ramp rate while an action is held (rad/s^2)."""

    action_deadband: float = 1.0e-4
    """Raw action magnitude below which the axis is considered released."""


class ROVRelativeIKAction(DifferentialInverseKinematicsAction):
    """Differential IK action that idles as joint hold when no arm command is active."""

    cfg: ROVRelativeIKActionCfg

    def __init__(self, cfg: ROVRelativeIKActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._inactive_deadband = float(cfg.inactive_deadband)
        self._joint_pos_hold_target = self._asset.data.joint_pos[:, self._joint_ids].clone()

    def apply_actions(self):
        if torch.all(torch.abs(self._raw_actions) <= self._inactive_deadband):
            self._asset.set_joint_position_target(self._joint_pos_hold_target, self._joint_ids)
            return

        super().apply_actions()
        self._joint_pos_hold_target[:] = self._asset.data.joint_pos_target[:, self._joint_ids]

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._joint_pos_hold_target[:] = self._asset.data.joint_pos[:, self._joint_ids]
        else:
            self._joint_pos_hold_target[env_ids] = self._asset.data.joint_pos[env_ids][:, self._joint_ids]


@configclass
class ROVRelativeIKActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configuration for :class:`ROVRelativeIKAction`."""

    class_type: type[ActionTerm] = ROVRelativeIKAction

    inactive_deadband: float = 1.0e-6
    """Raw action magnitude below which the arm idles with joint-position hold."""


class ROVRelativeJointPositionAction(RelativeJointPositionAction):
    """Relative joint action that holds the last target when no command is active."""

    cfg: ROVRelativeJointPositionActionCfg

    def __init__(self, cfg: ROVRelativeJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._inactive_deadband = float(cfg.inactive_deadband)
        self._joint_pos_hold_target = self._asset.data.joint_pos[:, self._joint_ids].clone()

    def apply_actions(self):
        if torch.all(torch.abs(self._raw_actions) <= self._inactive_deadband):
            self._asset.set_joint_position_target(self._joint_pos_hold_target, joint_ids=self._joint_ids)
            return

        current_actions = self.processed_actions + self._asset.data.joint_pos[:, self._joint_ids]
        self._asset.set_joint_position_target(current_actions, joint_ids=self._joint_ids)
        self._joint_pos_hold_target[:] = current_actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._joint_pos_hold_target[:] = self._asset.data.joint_pos[:, self._joint_ids]
        else:
            self._joint_pos_hold_target[env_ids] = self._asset.data.joint_pos[env_ids][:, self._joint_ids]


@configclass
class ROVRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """Configuration for :class:`ROVRelativeJointPositionAction`."""

    class_type: type[ActionTerm] = ROVRelativeJointPositionAction

    inactive_deadband: float = 1.0e-6
    """Raw action magnitude below which the joint action holds its last target."""
