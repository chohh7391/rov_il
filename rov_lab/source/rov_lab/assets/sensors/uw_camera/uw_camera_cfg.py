"""Underwater camera configuration (Isaac Lab 3.0)."""

from __future__ import annotations

from isaaclab.sensors.camera.camera_cfg import CameraCfg
from isaaclab.utils import configclass

UW_REQUIRED_DATA_TYPES: tuple[str, ...] = ("rgb", "distance_to_camera")


@configclass
class UWCameraCfg(CameraCfg):
    """Stock Isaac Lab camera plus the water parameters of the OceanSim underwater image model.

    The camera renders ``rgb`` and ``distance_to_camera``. The underwater appearance is applied afterwards by
    :func:`rov_lab.assets.sensors.uw_camera.uw_render` (used by the ``uw_image`` observation term)::

        I = J * exp(-atten_coeff * d) + 255 * backscatter_value * (1 - exp(-backscatter_coeff * d))

    with ``J`` the clean color and ``d`` the per-pixel distance in meters. Defaults are OceanSim's coastal-water
    values. In Isaac Lab 2.x this was a custom ``Camera`` subclass running a Warp kernel; it was replaced because
    3.0 cameras are renderer-backed and no longer expose the render-product internals it relied on.
    """

    data_types: list[str] = list(UW_REQUIRED_DATA_TYPES)

    backscatter_value: tuple[float, float, float] = (0.0, 0.31, 0.24)
    """Ambient backscatter color (RGB, 0-1)."""

    atten_coeff: tuple[float, float, float] = (0.05, 0.05, 0.05)
    """Per-channel attenuation coefficient (1/m). Larger values mean more turbid water."""

    backscatter_coeff: tuple[float, float, float] = (0.05, 0.05, 0.2)
    """Per-channel rate (1/m) at which backscatter builds up with distance."""

    max_render_distance: float = 1.0e6
    """Distance (m) used for pixels with no hit (open water) or invalid depth.

    OceanSim feeds the raw ``inf`` depth to its kernel, so open water renders as the pure backscatter color. The
    large default reproduces that (``exp(-coeff * d) = 0``) while staying finite when a coefficient is zero.
    """

    oceansim_lighting: bool = True
    """Render without RTX's uniform ambient light (``rtx.sceneDb.ambientLightIntensity = 0``), as OceanSim does.

    Isaac Lab 3.0's rendering experience sets the ambient light to 1.0, which brightens and flattens the image. In
    the shared OceanSim comparison scene (``rov_lab/scripts/tools/oceansim_compare``) turning it off raises the
    clean-RGB PSNR against OceanSim from 17 dB to 25 dB (underwater image: 19 -> 27 dB). This is a global RTX
    setting, so it applies to every camera in the stage.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        for data_type in UW_REQUIRED_DATA_TYPES:
            if data_type not in self.data_types:
                self.data_types.append(data_type)
        if self.oceansim_lighting:
            from isaaclab_physx.renderers import IsaacRtxRendererCfg

            if isinstance(self.renderer_cfg, IsaacRtxRendererCfg):
                self.renderer_cfg.global_settings = self.renderer_cfg.global_settings.replace(ambient_light_intensity=0.0)
