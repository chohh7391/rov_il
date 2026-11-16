"""OceanSim underwater image model applied to rendered RGB + depth (torch, batched)."""

from __future__ import annotations

import torch

from .uw_camera_cfg import UWCameraCfg


def uw_render(rgb: torch.Tensor, distance: torch.Tensor, cfg: UWCameraCfg) -> torch.Tensor:
    """Apply attenuation and backscatter to a batch of rendered images.

    Args:
        rgb: Clean color images, shape ``(N, H, W, 3|4)``, uint8 or float in [0, 255].
        distance: Per-pixel distance to the camera in meters, shape ``(N, H, W)`` or ``(N, H, W, 1)``.
            ``inf``/``nan`` (no hit) are treated as :attr:`UWCameraCfg.max_render_distance`.
        cfg: Camera configuration holding the water parameters.

    Returns:
        Underwater images, shape ``(N, H, W, 3)``, uint8.
    """
    far = float(cfg.max_render_distance)
    d = distance if distance.dim() == 4 else distance.unsqueeze(-1)
    d = torch.nan_to_num(d.float(), nan=far, posinf=far, neginf=far).clamp(0.0, far)
    device = rgb.device
    atten = torch.tensor(cfg.atten_coeff, device=device, dtype=torch.float32)
    back_val = torch.tensor(cfg.backscatter_value, device=device, dtype=torch.float32)
    back_coeff = torch.tensor(cfg.backscatter_coeff, device=device, dtype=torch.float32)
    clean = rgb[..., :3].float()
    out = clean * torch.exp(-d * atten) + 255.0 * back_val * (1.0 - torch.exp(-d * back_coeff))
    return out.clamp(0.0, 255.0).to(torch.uint8)
