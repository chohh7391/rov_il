"""Underwater sensors for rov_lab (Isaac Lab 3.0).

The underwater camera and the DVL are stock Isaac Lab sensors configured with underwater parameters. The
processing (OceanSim image model, DVL velocity and ranges, barometric depth) lives in observation terms in
``rov_lab.tasks.template.mdp.rov_observations``.

The imaging sonar (``imaging_sonar/``) has not been ported to Isaac Lab 3.0 yet and is not exported.
"""

from .barometer_model import barometer_pressure_model
from .dvl import DVLCfg, DVLJanusPatternCfg
from .uw_camera import UWCameraCfg, uw_render

__all__ = ["DVLCfg", "DVLJanusPatternCfg", "UWCameraCfg", "barometer_pressure_model", "uw_render"]
