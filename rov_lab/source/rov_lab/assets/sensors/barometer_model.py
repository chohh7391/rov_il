"""Barometer (pressure sensor) measurement model, same as OceanSim's ``BarometerSensor`` (pure torch)."""

from __future__ import annotations

import torch


def barometer_pressure_model(
    z_w: torch.Tensor,
    water_surface_z: float = 0.0,
    water_density: float = 1000.0,
    gravity: float = 9.81,
    atmosphere_pressure: float = 101325.0,
    noise_cov: float = 0.0,
) -> torch.Tensor:
    """Absolute pressure (Pa) at height ``z_w`` (any shape).

    ``p_atm + rho * g * max(surface - z, 0)`` plus Gaussian noise of variance ``noise_cov`` (Pa^2); defaults are
    OceanSim's (fresh water, 1 atm).
    """
    depth = (water_surface_z - z_w).clamp_min(0.0)
    pressure = atmosphere_pressure + water_density * gravity * depth
    if noise_cov > 0.0:
        pressure = pressure + noise_cov**0.5 * torch.randn_like(pressure)
    return pressure
