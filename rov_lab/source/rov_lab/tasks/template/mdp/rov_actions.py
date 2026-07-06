"""ROV body velocity action term.

Accepts a body-frame motion command and converts it to a wrench on the ROV root
rigid body. OceanSim's original MHL example applies keyboard force/torque directly;
this term intentionally adds a servo because the ROV carries an arm and direct force
commands tend to excite articulation motion.

The command vector keeps 6 slots ``[vx, vy, vz, wx, wy, wz]`` for teleop/recording
pipeline compatibility, but the vehicle is driven as an **observation-class ROV in
4-DOF**: linear ``[vx, vy, vz]`` (surge/sway/heave) plus yaw-rate (slot ``wz``). The
roll/pitch command slots (``wx``, ``wy``) are intentionally ignored -- the DWTEK I90
is not actuated in roll/pitch; those axes get pure rate damping (self-leveling is
deferred to the Fossen buoyancy restoring, Stage 2). Yaw is rate-controlled (``wz``
commands a yaw rate). No attitude position-hold term -- see the NOTE in ``apply_actions``.

See ``docs/rov-base-control-design.md`` for the full design rationale (why 4-DOF, the
I90 thrust limits, and the Fossen feed-forward roadmap).
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
    """Applies a 4-DOF body-frame motion servo to the ROV base body.

    The action vector is ``(vx, vy, vz, wx, wy, wz)`` but only 4 DOF are actuated:

    * **Linear (surge/sway/heave)** -- ``vx, vy, vz`` drive a PI velocity servo. While an
      axis is held the target velocity ramps toward ``cfg.lin_vel_scale``; released axes
      reset to zero.
    * **Yaw (rate control)** -- ``wz`` commands a yaw *rate*; an angular-velocity servo tracks
      it. No heading position-hold (an attitude spring rings on translation-induced yaw).
    * **Roll/pitch (``wx, wy``)** -- ignored. The I90 has no roll/pitch thrust; these axes get
      pure rate damping. Gentle self-leveling is deferred to the Fossen buoyancy restoring
      term (Stage 2).

    Force is clamped **asymmetrically** per axis (a vectored-thruster ROV produces more
    forward than reverse surge). The resulting wrench is rotated to world frame and applied
    through Isaac Lab's permanent wrench composer.

    .. note::
        The target asset must be an :class:`Articulation` that contains the ROV base body.
        Make sure the ROV's ``RigidBodyAPI`` has ``disableGravity = True`` for neutrally
        buoyant behaviour.
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

        # Action buffers. The command keeps 6 slots for pipeline compatibility (see module
        # docstring); only linear [vx, vy, vz] and yaw-rate (slot 5) are used. Roll/pitch
        # command slots (3, 4) are ignored -- roll/pitch are held level passively.
        self._raw_actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._target_lin_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_yaw_rate = torch.zeros(self.num_envs, device=self.device)

        # Cache scaling / limits as tensors on device for fast computation.
        self._lin_vel_scale = torch.tensor(cfg.lin_vel_scale, device=self.device, dtype=torch.float32)
        self._lin_vel_ramp_rate = torch.tensor(cfg.lin_vel_ramp_rate, device=self.device, dtype=torch.float32)
        self._yaw_rate_scale = float(cfg.yaw_rate_scale)
        self._yaw_rate_ramp_rate = float(cfg.yaw_rate_ramp_rate)
        self._kp_linear = torch.tensor(cfg.kp_linear, device=self.device, dtype=torch.float32)
        # Asymmetric per-axis body-frame force clamp: force_b in [-lower, +upper].
        self._force_limit_upper = torch.tensor(cfg.force_limit_upper, device=self.device, dtype=torch.float32)
        self._force_limit_lower = torch.tensor(cfg.force_limit_lower, device=self.device, dtype=torch.float32)
        # Per-axis (roll, pitch, yaw) torque clamp.
        self._torque_limit = torch.tensor(cfg.torque_limit, device=self.device, dtype=torch.float32)
        # Angular-velocity servo gains (roll, pitch, yaw). Roll/pitch target 0 -> rate damping;
        # yaw tracks the commanded rate. No attitude position spring (rings on translation).
        self._kp_angular = torch.tensor(cfg.kp_angular, device=self.device, dtype=torch.float32)
        self._action_deadband = float(cfg.action_deadband)
        self._dt = float(env.step_dt)

        # Arm-aware center-of-mass compensation.
        # The servo only drives the ``rov_base`` body, but the arm shifts the combined
        # (base + arm) center of mass away from where the thrust is applied. A pure horizontal
        # thrust then has a moment arm about the combined COM, which pitches the whole vehicle
        # during acceleration. We cache the per-body masses so we can recompute the combined COM
        # each step (the arm configuration, and therefore the COM, changes over time).
        self._com_compensation = bool(cfg.com_compensation)
        body_masses = self._asset.data.default_mass.to(self.device)  # (num_envs, num_bodies)
        self._body_masses = body_masses.unsqueeze(-1)  # (num_envs, num_bodies, 1)
        self._total_mass = body_masses.sum(dim=1, keepdim=True).clamp_min(1.0e-9)  # (num_envs, 1)

        # Integral action on the linear velocity servo. A pure proportional servo leaves a
        # steady-state velocity error against any persistent disturbance (e.g. the arm's
        # hydrodynamic/coupling load), which shows up as a slow continuous sink while cruising.
        # The integral term drives that error to zero.
        self._ki_linear = torch.tensor(cfg.ki_linear, device=self.device, dtype=torch.float32)
        self._lin_integral_limit = torch.tensor(cfg.lin_integral_limit, device=self.device, dtype=torch.float32)
        self._lin_vel_integral = torch.zeros(self.num_envs, 3, device=self.device)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        # Keeps 6 slots for teleop/recording compatibility; roll/pitch (slots 3, 4) are unused.
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
        actions: body-frame command ``[vx, vy, vz, wx, wy, wz]``. Only ``[vx, vy, vz]`` and
                 the yaw-rate ``wz`` (slot 5) are used; ``wx, wy`` (roll/pitch) are ignored.
        """
        self._raw_actions[:] = actions

        self._target_lin_vel_b[:] = self._ramp_targets(
            current=self._target_lin_vel_b,
            commands=actions[:, 0:3],
            max_values=self._lin_vel_scale,
            ramp_rates=self._lin_vel_ramp_rate,
        )
        # Yaw-rate command (slot 5) ramps toward the configured max rate.
        self._target_yaw_rate[:] = self._ramp_scalar(
            current=self._target_yaw_rate,
            commands=actions[:, 5],
            max_value=self._yaw_rate_scale,
            ramp_rate=self._yaw_rate_ramp_rate,
        )

        # processed_actions buffer (may be consumed by observations). Roll/pitch slots stay 0.
        self._processed_actions[:, 0:3] = self._target_lin_vel_b
        self._processed_actions[:, 3:5] = 0.0
        self._processed_actions[:, 5] = self._target_yaw_rate

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

    def _ramp_scalar(
        self,
        current: torch.Tensor,
        commands: torch.Tensor,
        max_value: float,
        ramp_rate: float,
    ) -> torch.Tensor:
        """Scalar (per-env) variant of :meth:`_ramp_targets` for the yaw-rate command."""
        active = commands.abs() > self._action_deadband
        command_sign = torch.sign(commands)
        max_targets = command_sign * commands.abs().clamp(max=1.0) * max_value
        increment = ramp_rate * self._dt

        same_direction = torch.sign(current) == command_sign
        next_targets = torch.where(same_direction, current + command_sign * increment, command_sign * increment)
        next_targets = torch.clamp(next_targets, -max_targets.abs(), max_targets.abs())
        return torch.where(active, next_targets, torch.zeros_like(current))

    def apply_actions(self):
        # 1. Current state (world frame).
        control_quat_w = self._asset.data.body_quat_w[:, self._control_body_idx]
        lin_vel_w = self._asset.data.body_lin_vel_w[:, self._rov_body_idx]
        ang_vel_w = self._asset.data.body_ang_vel_w[:, self._rov_body_idx]

        # 2. Express current velocities in the control frame. The wrench can be applied to a
        # different heavy hull body while preserving the teleoperation/body-axis convention.
        control_quat_inv = math_utils.quat_conjugate(control_quat_w)
        curr_lin_vel_b = math_utils.quat_apply(control_quat_inv, lin_vel_w)
        curr_ang_vel_b = math_utils.quat_apply(control_quat_inv, ang_vel_w)

        # 3. Linear PI velocity servo (surge/sway/heave).
        lin_vel_error = self._target_lin_vel_b - curr_lin_vel_b
        # Anti-windup clamped integral so a persistent velocity error (e.g. the slow cruise sink)
        # is integrated out instead of left as a steady-state offset.
        self._lin_vel_integral += lin_vel_error * self._dt
        self._lin_vel_integral.clamp_(-self._lin_integral_limit, self._lin_integral_limit)
        # Hold the integral at zero whenever NO linear axis is commanded so the vehicle brakes to a
        # clean stop (otherwise wound-up integral shows up as forward creep after release).
        lin_cmd_active = (self._raw_actions[:, 0:3].abs() > self._action_deadband).any(dim=1, keepdim=True)
        self._lin_vel_integral.mul_(lin_cmd_active.to(self._lin_vel_integral.dtype))
        force_b = self._kp_linear * lin_vel_error + self._ki_linear * self._lin_vel_integral

        # 4. Attitude: 4-DOF angular-velocity servo (the original, proven-stable law). Roll/pitch
        # are not commanded -- their target rate is 0, so the term is pure rate damping (gentle
        # self-leveling is deferred to the Fossen buoyancy restoring, Stage 2). Yaw tracks the
        # commanded rate.
        # NOTE: an attitude POSITION spring (yaw heading-hold or roll/pitch level-hold) was tried
        # and removed -- it turns the axis into a spring-mass system that RINGS whenever translation
        # induces a small tilt/yaw through the arm's COM offset. Do NOT reintroduce a position term
        # here without sim tuning. See docs/rov-base-control-design.md 4.3 / 0.5.
        target_ang_vel_b = torch.zeros_like(curr_ang_vel_b)
        target_ang_vel_b[:, 2] = self._target_yaw_rate
        torque_b = self._kp_angular * (target_ang_vel_b - curr_ang_vel_b)

        # 5. Clamp in body frame so actuator limits stay aligned with the ROV axes. Force clamp is
        # asymmetric per axis (a vectored-thruster ROV produces more forward than reverse surge).
        force_b = torch.clamp(force_b, min=-self._force_limit_lower, max=self._force_limit_upper)
        torque_b = torch.clamp(torque_b, min=-self._torque_limit, max=self._torque_limit)

        # 6. Body -> world.
        forces_w = math_utils.quat_apply(control_quat_w, force_b)
        torques_w = math_utils.quat_apply(control_quat_w, torque_b)

        # 7. Arm-aware COM compensation (feed-forward).
        # PhysX applies the external force at the rov_base center of mass ``P``. The combined
        # (base + arm) center of mass ``C`` is offset by the arm, so the thrust produces a
        # parasitic torque ``(P - C) x F`` about ``C`` that pitches the vehicle. Add a compensating
        # couple ``(C - P) x F`` so the net wrench is equivalent to thrust acting through the true
        # combined COM. Applied after the servo clamp so the full physical correction is delivered.
        if self._com_compensation:
            com_w = self._asset.data.body_com_pos_w  # (num_envs, num_bodies, 3)
            combined_com_w = (self._body_masses * com_w).sum(dim=1) / self._total_mass  # (num_envs, 3)
            force_point_w = com_w[:, self._rov_body_idx]  # application point P
            torques_w = torques_w + torch.cross(combined_com_w - force_point_w, forces_w, dim=-1)

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
            self._target_yaw_rate[:] = 0.0
            self._lin_vel_integral[:] = 0.0
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0
            self._target_lin_vel_b[env_ids] = 0.0
            self._target_yaw_rate[env_ids] = 0.0
            self._lin_vel_integral[env_ids] = 0.0


@configclass
class ROVVelocityActionCfg(ActionTermCfg):
    """Configuration for :class:`ROVVelocityAction` (DWTEK I90, 4-DOF).

    Defaults are set from the Investigator 90 catalogue (observation-class, 4-DOF actuated).
    See ``docs/rov-base-control-design.md`` 4.4.
    """

    class_type: type[ActionTerm] = ROVVelocityAction

    body_name: str = MISSING
    """Name of the body where the external wrench is applied."""

    control_body_name: str | None = None
    """Body whose local frame defines the velocity/action axes.

    If ``None``, uses :attr:`body_name`.
    """

    lin_vel_scale: tuple[float, float, float] = (1.5, 1.0, 0.7)
    """Per-axis scale applied to the linear velocity action (m/s).

    surge ~1.5 m/s (I90 top speed 3 kn); sway/heave slower.
    """

    yaw_rate_scale: float = 0.6
    """Max yaw-rate the ``wz`` command maps to (rad/s)."""

    kp_linear: tuple[float, float, float] = (25.0, 25.0, 30.0)
    """Per-axis proportional gain for linear velocity control."""

    ki_linear: tuple[float, float, float] = (3.0, 3.0, 5.0)
    """Per-axis integral gain for linear velocity control.

    Removes the steady-state velocity error a proportional-only servo leaves against a
    persistent disturbance (notably the slow downward drift while cruising forward).
    """

    lin_integral_limit: tuple[float, float, float] = (2.0, 2.0, 2.0)
    """Symmetric anti-windup clamp on the linear velocity integral state (m, per axis)."""

    kp_angular: tuple[float, float, float] = (12.0, 12.0, 18.0)
    """Per-axis angular-velocity servo gain ``(roll, pitch, yaw)``.

    Roll/pitch are uncommanded (target rate 0 -> pure rate damping); yaw tracks the commanded
    rate. These are the original, proven-stable values. Do NOT add an attitude *position* spring
    (heading-hold / level-hold) -- it rings on translation-induced tilt/yaw via the arm COM offset
    (see docs/rov-base-control-design.md 4.3)."""

    force_limit_upper: tuple[float, float, float] = (588.0, 329.0, 304.0)
    """Positive-direction body-frame force clip ``(+surge, +sway, +heave)`` in Newtons.

    I90 thrust: forward 60 kgf, lateral 33.5 kgf, vertical 31 kgf (1 kgf = 9.80665 N).
    """

    force_limit_lower: tuple[float, float, float] = (294.0, 329.0, 304.0)
    """Negative-direction body-frame force clip magnitudes ``(-surge, -sway, -heave)`` in N.

    Surge reverse (30 kgf) is weaker than forward (60 kgf) on the vectored I90 thrusters.
    """

    torque_limit: tuple[float, float, float] = (100.0, 100.0, 100.0)
    """Symmetric body-frame torque clip ``(roll, pitch, yaw)`` in N·m."""

    lin_vel_ramp_rate: tuple[float, float, float] = (2.0, 2.0, 2.0)
    """Per-axis linear velocity target ramp rate while an action is held (m/s^2)."""

    yaw_rate_ramp_rate: float = 1.0
    """Yaw-rate target ramp rate while the yaw command is held (rad/s^2)."""

    action_deadband: float = 1.0e-4
    """Raw action magnitude below which the axis is considered released."""

    com_compensation: bool = True
    """Whether to add an arm-aware center-of-mass compensation couple.

    When enabled, the applied thrust is made equivalent to a force acting through the combined
    (base + arm) center of mass, cancelling the parasitic pitch torque that the off-COM arm load
    would otherwise induce during acceleration. Set to ``False`` to recover the legacy behaviour
    (force applied at the ``rov_base`` body only).
    """


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
