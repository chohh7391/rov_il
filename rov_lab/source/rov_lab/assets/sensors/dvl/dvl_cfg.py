"""Doppler velocity log (DVL) configuration (Isaac Lab 3.0)."""

from __future__ import annotations

from typing import Literal

from isaaclab.sensors.ray_caster.ray_caster_cfg import RayCasterCfg
from isaaclab.utils import configclass

from .dvl_pattern_cfg import DVLJanusPatternCfg


@configclass
class DVLCfg(RayCasterCfg):
    """Stock Isaac Lab ray caster with a 4-beam Janus pattern plus OceanSim's DVL parameters.

    The ray caster measures the beam ranges to the seabed. The DVL outputs are computed by observation terms in
    ``rov_lab.tasks.template.mdp.rov_observations`` using :mod:`rov_lab.assets.sensors.dvl.dvl_model`:

    * ``dvl_linear_velocity``: velocity over ground in the vehicle body frame, with per-beam noise mapped through
      the Janus transform (as OceanSim), zeroed on dropout (``num_beams_out_range_threshold`` beams lost).
    * ``dvl_depth``: per-beam range to the seabed with per-beam noise; ``miss_value`` on beams with no return.
    * ``dvl_beam_hit``: per-beam valid flag.

    Parameter names and defaults follow ``OceanSim/isaacsim/oceansim/sensors/DVLsensor.py``. Deviations from
    OceanSim are opt-in and listed on the fields (``velocity_point``, ``miss_value``). In Isaac Lab 2.x this was a
    custom ``RayCaster`` subclass; it was replaced because 3.0 turned ``RayCaster`` into a backend factory and
    changed the physics-view and buffer APIs it relied on.
    """

    pattern_cfg: DVLJanusPatternCfg = DVLJanusPatternCfg()

    body_name: str = "rov_base"
    """Robot body the DVL is mounted on (used for the velocity measurement). ``prim_path`` should be this body:
    the mounting position is given by :attr:`offset`, which Isaac Lab 3.0 applies to the ray pattern."""

    elevation: float = 22.5
    """Beam tilt from the vertical (deg). OceanSim default."""

    rotation: float = 45.0
    """Beam pattern rotation about the sensor z axis (deg). OceanSim default (Janus X configuration)."""

    min_range: float = 0.1
    """Minimum valid beam range (m)."""

    max_range: float = 100.0
    """Maximum valid beam range (m). Forwarded to :attr:`RayCasterCfg.max_distance`."""

    num_beams_out_range_threshold: int = 2
    """Velocity drops out (zero) when at least this many beams have no return. OceanSim default."""

    vel_cov: float = 0.0
    """Per-beam velocity noise variance ((m/s)^2), mapped to body axes by the Janus transform. OceanSim ``vel_cov``."""

    depth_cov: float = 0.0
    """Per-beam range noise variance (m^2). OceanSim ``depth_cov``."""

    velocity_point: Literal["com", "sensor"] = "sensor"
    """``"com"``: body center-of-mass velocity, as OceanSim. ``"sensor"``: velocity of the DVL mounting point
    (adds the lever-arm term for :attr:`offset`), which is what a real DVL measures."""

    miss_value: float | None = None
    """Range reported for beams with no return. OceanSim returns NaN; ``None`` reports ``max_range`` instead,
    which keeps policy inputs finite."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.pattern_cfg = DVLJanusPatternCfg(elevation=self.elevation, rotation=self.rotation)
        self.max_distance = self.max_range
        self.ray_alignment = "base"
