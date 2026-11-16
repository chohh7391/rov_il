"""DVL measurement model (pure torch, batched). Mirrors OceanSim's ``DVLsensor`` where noted.

Kept free of simulator state so it can be unit-tested against OceanSim's formulas without launching Isaac Sim.
"""

from __future__ import annotations

import math

import torch
import isaaclab.utils.math as math_utils


def janus_noise_transform(elevation_deg: float, device: str | torch.device = "cpu") -> torch.Tensor:
    """Beam-to-body velocity transform used by OceanSim for DVL noise, shape ``(3, 4)``.

    Same matrix as ``OceanSim/isaacsim/oceansim/sensors/DVLsensor.py`` (``self._transform``): per-beam noise
    ``n`` (4,) maps to body-frame velocity noise ``T @ n``.
    """
    s = math.sin(math.radians(elevation_deg))
    c = math.cos(math.radians(elevation_deg))
    return torch.tensor(
        [
            [1 / (2 * s), 0.0, -1 / (2 * s), 0.0],
            [0.0, 1 / (2 * s), 0.0, -1 / (2 * s)],
            [1 / (4 * c), 1 / (4 * c), 1 / (4 * c), 1 / (4 * c)],
        ],
        device=device,
        dtype=torch.float32,
    )


def dvl_body_velocity(
    lin_vel_w: torch.Tensor,
    quat_w: torch.Tensor,
    ang_vel_w: torch.Tensor | None = None,
    offset_b: torch.Tensor | None = None,
) -> torch.Tensor:
    """Velocity over ground expressed in the body frame, shape ``(N, 3)``.

    Args:
        lin_vel_w: Linear velocity of the reference point in the world frame ``(N, 3)``.
        quat_w: Body orientation, xyzw (Isaac Lab 3.0) ``(N, 4)``.
        ang_vel_w: Body angular velocity in the world frame ``(N, 3)``; needed with ``offset_b``.
        offset_b: Sensor position relative to the reference point, body frame ``(3,)`` or ``(N, 3)``. When given,
            the lever-arm term ``omega x r`` is added (velocity of the sensor point). OceanSim omits it: it reports
            ``R^T v`` of the rigid body (its center of mass), regardless of where the DVL is mounted.
    """
    vel_w = lin_vel_w
    if offset_b is not None:
        if ang_vel_w is None:
            raise ValueError("ang_vel_w is required for the lever-arm term")
        lever_w = math_utils.quat_apply(quat_w, offset_b.expand_as(lin_vel_w))
        vel_w = lin_vel_w + torch.linalg.cross(ang_vel_w, lever_w)
    return math_utils.quat_apply_inverse(quat_w, vel_w)


def bottom_lock(beam_hit: torch.Tensor, num_beams_out_range_threshold: int = 2) -> torch.Tensor:
    """OceanSim dropout rule: the velocity is dropped (zero) when ``#missed beams >= threshold``. Shape ``(N, 1)``."""
    missed = (~beam_hit.bool()).sum(dim=-1, keepdim=True)
    return missed < num_beams_out_range_threshold


def dvl_ranges_from_hits(
    hits_w: torch.Tensor, origin_w: torch.Tensor, min_range: float, max_range: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-beam range (m) and hit mask from ray hit points, shape ``(N, B)``. Range is NaN where there is no hit.

    ``hits_w`` is ``(N, B, 3)`` (Isaac Lab reports ``inf`` for rays with no hit), ``origin_w`` is ``(N, 3)``.
    """
    rng = torch.linalg.vector_norm(hits_w - origin_w.unsqueeze(1), dim=-1)
    hit = torch.isfinite(rng) & (rng >= min_range) & (rng <= max_range)
    return torch.where(hit, rng, torch.full_like(rng, float("nan"))), hit


def dvl_ray_origins_w(pos_w: torch.Tensor, quat_w: torch.Tensor, offset_pos: tuple[float, float, float]) -> torch.Tensor:
    """World position the DVL beams start from, shape ``(N, 3)``.

    Isaac Lab 3.0's ``RayCaster`` applies ``cfg.offset`` to the local ray pattern, so the rays leave from
    ``pos_w + R(quat_w) * offset``, while ``data.pos_w`` / ``data.quat_w`` report the sensor prim frame *without*
    that offset. Ranges must be measured from this origin, not from ``data.pos_w``.
    """
    off = torch.tensor(offset_pos, device=pos_w.device, dtype=pos_w.dtype).expand_as(pos_w)
    return pos_w + math_utils.quat_apply(quat_w, off)
