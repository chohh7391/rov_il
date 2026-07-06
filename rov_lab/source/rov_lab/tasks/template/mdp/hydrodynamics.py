"""MarineGym Fossen hydrodynamics, ported as a standalone body-frame wrench.

This is a faithful port of MarineGym's underwater force model
(`external_dependencies/MarineGym/marinegym/robots/drone/underwaterVehicle.py:203-277`) as a pure
PyTorch function plus a thin Isaac Lab ``EventTerm`` wrapper. It adds the hydrodynamic wrench
(added-mass, Coriolis, anisotropic damping, buoyancy restoring) to the ROV base via the asset's
``instantaneous_wrench_composer`` (which auto-sums with the servo's ``permanent`` wrench in
``write_data_to_sim``).

Design/rationale: see ``docs/marinegym-hydrodynamics-eventterm.md`` and
``docs/rov-base-control-design.md``.

Notes on the port (load-bearing details preserved verbatim from the source):

* 6-DOF vector order is ``[surge(x), sway(y), heave(z), roll, pitch, yaw]``.
* MarineGym works in a Fossen (NED-ish) frame while Isaac Lab's body twist is FLU, so the source
  applies an ENU/FLU<->NED conversion: flip velocity indices ``[1,2,4,5]`` on the way in, flip the
  ``[1,2,4,5]`` outputs on the way out, and flip ``pitch`` (rpy index 1) before buoyancy. These flips
  are reproduced exactly; the sign is validated in sim (surge -> opposing drag) before trusting it.
* Coefficient matrices are diagonal, so the source's off-diagonal ``maintained_body_vels`` coupling is
  multiplied by the zero off-diagonal of the (diagonal) quadratic-damping matrix and contributes
  nothing. It is ported verbatim anyway (same result), to stay byte-for-byte faithful.
* Buoyancy uses ``rho=997, g=9.8`` (MarineGym uses 9.8 here specifically, not 9.81).
* Defaults are BlueROV.yaml values -- the sim asset geometry IS BlueROV, so these are consistent and
  are the only hydro coefficients available (the I90 catalogue provides none). Re-derive for an I90 asset.
"""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils

__all__ = ["HydroParamsCfg", "build_hydro_coeffs", "compute_hydro_wrench", "apply_hydro_wrench"]


@configclass
class HydroParamsCfg:
    """Fossen hydrodynamic coefficients + per-term enable flags.

    Defaults are the BlueROV.yaml values. Each physical term is individually gated so the model can be
    brought up and validated incrementally (see ``docs/rov-base-control-design.md`` verification plan).
    """

    enable: bool = True
    """Master switch. When ``False`` the event is a no-op (legacy behaviour, zero net change)."""

    enable_damping: bool = True
    """Anisotropic linear + quadratic drag (the main useful term)."""

    enable_buoyancy_moment: bool = True
    """Roll/pitch restoring moment from the center-of-buoyancy offset (soft self-leveling spring).

    Adds NO net force, so it works under ``disable_gravity=True`` without making the vehicle rise/sink.
    """

    enable_buoyancy_force: bool = False
    """Net buoyancy force terms. OFF: under ``disable_gravity=True`` there is no weight to cancel these,
    so they would push the vehicle around. Only enable together with gravity + real mass."""

    enable_coriolis: bool = False
    """Added-mass Coriolis/centripetal coupling. Small for slow manipulation; optional."""

    enable_added_mass: bool = False
    """Added-mass reaction (needs body acceleration). OFF by default: acceleration-dependent and the
    riskiest term numerically. When enabled it uses an explicit finite-difference of the previous
    step's velocity (not the realized PhysX acceleration, which would close an algebraic loop)."""

    volume: float = 0.0113459
    """Displaced volume (m^3), for buoyancy force = rho * g * volume."""

    co_bm: float = 0.01
    """Center-of-buoyancy moment arm (m) above the center of mass -> righting moment."""

    rho: float = 997.0
    """Water density (kg/m^3)."""

    g: float = 9.8
    """Gravity used by the buoyancy term (MarineGym uses 9.8 here, not 9.81)."""

    added_mass: tuple[float, float, float, float, float, float] = (5.5, 12.7, 14.57, 0.12, 0.12, 0.12)
    """Diagonal added-mass coefficients ``[surge, sway, heave, roll, pitch, yaw]``."""

    linear_damping: tuple[float, float, float, float, float, float] = (4.03, 6.22, 5.18, 0.07, 0.07, 0.07)
    """Diagonal linear damping coefficients."""

    quadratic_damping: tuple[float, float, float, float, float, float] = (18.18, 21.66, 36.99, 1.55, 1.55, 1.55)
    """Diagonal quadratic damping coefficients (scaled by |v|)."""

    accel_alpha: float = 0.3
    """Low-pass factor for the finite-difference acceleration (added-mass only)."""


def build_hydro_coeffs(cfg: HydroParamsCfg, num_envs: int, device) -> dict[str, torch.Tensor]:
    """Precompute the per-env diagonal coefficient matrices once (shape ``(num_envs, 6, 6)``)."""

    def _diag(vec) -> torch.Tensor:
        return torch.diag(torch.tensor(vec, device=device, dtype=torch.float32)).repeat(num_envs, 1, 1)

    return {
        "A": _diag(cfg.added_mass),
        "D_lin": _diag(cfg.linear_damping),
        "D_quad": _diag(cfg.quadratic_damping),
    }


def compute_hydro_wrench(
    lin_vel_b: torch.Tensor,
    ang_vel_b: torch.Tensor,
    roll: torch.Tensor,
    pitch: torch.Tensor,
    body_acc_b: torch.Tensor | None,
    coeffs: dict[str, torch.Tensor],
    cfg: HydroParamsCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the body-frame hydrodynamic ``(force, torque)`` for the ROV base.

    Args:
        lin_vel_b: ``(N, 3)`` body-frame linear velocity (FLU).
        ang_vel_b: ``(N, 3)`` body-frame angular velocity (FLU).
        roll: ``(N,)`` world roll angle (rad).
        pitch: ``(N,)`` world pitch angle (rad).
        body_acc_b: ``(N, 6)`` body-frame acceleration, or ``None`` (only used for added-mass).
        coeffs: dict from :func:`build_hydro_coeffs` (``A``, ``D_lin``, ``D_quad``).
        cfg: :class:`HydroParamsCfg`.

    Returns:
        ``(force_b (N,3), torque_b (N,3))`` in the ROV base body frame.
    """
    n = lin_vel_b.shape[0]
    device = lin_vel_b.device

    v = torch.cat([lin_vel_b, ang_vel_b], dim=-1)  # (N, 6), fresh tensor from cat
    # ENU/FLU -> NED frame flip (MarineGym underwaterVehicle.py:212-213).
    v[:, [1, 2, 4, 5]] *= -1.0
    pitch_f = -pitch  # rpy[[1,2]] *= -1 (yaw is unused by buoyancy)

    zeros6 = torch.zeros(n, 6, device=device)

    # Anisotropic damping D(v) @ v. The off-diagonal `maintained` terms are ported verbatim but are
    # nullified by the diagonal quadratic matrix (see module docstring); the net effect is diagonal.
    if cfg.enable_damping:
        maintained = torch.diag_embed(v)
        maintained[:, 1, 5] = v[:, 5]
        maintained[:, 2, 4] = v[:, 4]
        maintained[:, 4, 2] = v[:, 2]
        maintained[:, 5, 1] = v[:, 1]
        damping_matrix = coeffs["D_lin"] + coeffs["D_quad"] * torch.abs(maintained)
        damping = (damping_matrix @ v.unsqueeze(-1)).squeeze(-1)
    else:
        damping = zeros6

    if cfg.enable_added_mass and body_acc_b is not None:
        added_mass = (coeffs["A"] @ body_acc_b.unsqueeze(-1)).squeeze(-1)
    else:
        added_mass = zeros6

    if cfg.enable_coriolis:
        ab = (coeffs["A"] @ v.unsqueeze(-1)).squeeze(-1)
        coriolis = torch.zeros(n, 6, device=device)
        coriolis[:, 0:3] = -torch.cross(ab[:, 0:3], v[:, 3:6], dim=1)
        coriolis[:, 3:6] = -(
            torch.cross(ab[:, 0:3], v[:, 0:3], dim=1) + torch.cross(ab[:, 3:6], v[:, 3:6], dim=1)
        )
    else:
        coriolis = zeros6

    hydro = -(added_mass + coriolis + damping)
    hydro[:, [1, 2, 4, 5]] *= -1.0  # output frame flip

    buoyancy = torch.zeros(n, 6, device=device)
    fb = cfg.rho * cfg.g * cfg.volume
    if cfg.enable_buoyancy_force:
        buoyancy[:, 0] = fb * torch.sin(pitch_f)
        buoyancy[:, 1] = -fb * torch.sin(roll) * torch.cos(pitch_f)
        buoyancy[:, 2] = -fb * torch.cos(roll) * torch.cos(pitch_f)
    if cfg.enable_buoyancy_moment:
        buoyancy[:, 3] = -cfg.co_bm * fb * torch.cos(pitch_f) * torch.sin(roll)
        buoyancy[:, 4] = -cfg.co_bm * fb * torch.sin(pitch_f)
    buoyancy[:, [1, 2, 4, 5]] *= -1.0  # output frame flip

    wrench = hydro + buoyancy
    return wrench[:, 0:3], wrench[:, 3:6]


def apply_hydro_wrench(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rov_base"),
    hydro_cfg: HydroParamsCfg | None = None,
) -> None:
    """EventTerm: add the Fossen hydrodynamic wrench to the ROV base each step.

    Wire as ``mode="interval", interval_range_s=(0.0, 0.0), is_global_time=True`` so it fires every env
    step (with ``env_ids=None``). Uses the ``instantaneous_wrench_composer`` (auto-reset each step, and
    auto-summed with the servo's ``permanent`` wrench) so it never clobbers the servo.
    """
    if hydro_cfg is None or not hydro_cfg.enable:
        return

    asset = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]

    # Persistent per-env state (coeffs + finite-diff buffers) stashed on env, lazily.
    if not hasattr(env, "_rov_hydro_coeffs"):
        env._rov_hydro_coeffs = build_hydro_coeffs(hydro_cfg, env.num_envs, env.device)
        env._rov_hydro_prev_vel = torch.zeros(env.num_envs, 6, device=env.device)
        env._rov_hydro_prev_acc = torch.zeros(env.num_envs, 6, device=env.device)

    quat_w = asset.data.body_quat_w[:, body_id]
    lin_w = asset.data.body_lin_vel_w[:, body_id]
    ang_w = asset.data.body_ang_vel_w[:, body_id]

    quat_inv = math_utils.quat_conjugate(quat_w)
    lin_b = math_utils.quat_apply(quat_inv, lin_w)
    ang_b = math_utils.quat_apply(quat_inv, ang_w)
    roll, pitch, _ = math_utils.euler_xyz_from_quat(quat_w)

    acc_b = None
    if hydro_cfg.enable_added_mass:
        # Explicit finite-diff of the previous step's velocity + low-pass (MarineGym calculate_acc).
        # Using the realized PhysX acceleration would close an algebraic loop; the explicit lag avoids it.
        v = torch.cat([lin_b, ang_b], dim=-1)
        raw_acc = (v - env._rov_hydro_prev_vel) / env.step_dt
        acc_b = (1.0 - hydro_cfg.accel_alpha) * env._rov_hydro_prev_acc + hydro_cfg.accel_alpha * raw_acc
        env._rov_hydro_prev_vel = v.detach().clone()
        env._rov_hydro_prev_acc = acc_b.detach().clone()

    force_b, torque_b = compute_hydro_wrench(
        lin_b, ang_b, roll, pitch, acc_b, env._rov_hydro_coeffs, hydro_cfg
    )

    asset.instantaneous_wrench_composer.add_forces_and_torques(
        forces=force_b.unsqueeze(1),
        torques=torque_b.unsqueeze(1),
        body_ids=[body_id],
        is_global=False,
    )
