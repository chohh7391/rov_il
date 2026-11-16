"""Layout of the subsea intervention work site (pure Python, shared by the asset builder and the env configs).

Site frame: origin on the (levelled) seabed at the work site, Z up, the ROV approaches along +X. The skid frame
(DAVE ``electrical_panel``) stands at the origin; the valve panel (UUV Simulator ``rov_bop_panel``) is bolted to a
mounting plate on the frame's -X side with its valve axes pointing at the ROV (-X).

``rov_lab/scripts/tools/build_subsea_assets.py`` bakes the static parts (terrain, frame, mount, rocks, tools) into
``scenes/<scene>/<scene>_work_site.usda``; the panel is spawned by the env as an articulation at ``PANEL_POS`` /
``PANEL_QUAT_WXYZ``.
"""

from __future__ import annotations

import math

WORK_SITE_SCENES: tuple[str, ...] = ("shipwreck_site", "portuguese_ledge")

PAD_RADII: tuple[float, float] = (3.5, 6.0)
"""Seabed levelled to z = 0 inside the first radius, blended back to the natural terrain at the second (m)."""

FRAME_POS: tuple[float, float, float] = (0.0, 0.0, 0.0)
FRAME_FRONT_X: float = -0.95
"""-X face of the skid frame (its footprint is 1.9 m x 2.92 m)."""

PANEL_CENTER_Y: float = 0.6
PANEL_CENTER_Z: float = 0.8
PANEL_STANDOFF: float = 0.035
"""Mounting-plate thickness between the frame face and the panel back (m)."""

PANEL_POS: tuple[float, float, float] = (FRAME_FRONT_X - PANEL_STANDOFF, PANEL_CENTER_Y, PANEL_CENTER_Z)
"""Panel frame origin (centre of its back face)."""

# panel frame -> site frame: panel z (valve axes) -> -X, panel x -> -Y, panel y -> +Z (valve rows horizontal,
# carrying handles at the top and bottom). Rotation matrix columns: x_p=(0,-1,0), y_p=(0,0,1), z_p=(-1,0,0).
PANEL_QUAT_WXYZ: tuple[float, float, float, float] = (0.5, 0.5, -0.5, -0.5)

MOUNT_PLATE_SIZE: tuple[float, float, float] = (PANEL_STANDOFF, 0.66, 0.96)
MOUNT_POST_SIZE: tuple[float, float] = (0.06, 0.06)
MOUNT_POST_Y_OFFSET: float = 0.36

ROV_STATION_POS: tuple[float, float, float] = (-2.27, PANEL_CENTER_Y, 0.85)
"""ROV hull centre while working on the panel (facing +X). At the Bravo home pose the jaw tips stop ~10 cm short of
the panel face; the paddles are then 1.05-1.08 m ahead of the hull centre (0.53-0.56 m ahead of the bow)."""

# (rock id 1-7, x, y, yaw deg, size m) for the scattered Ocean Rock models, clear of the approach corridor
ROCKS: tuple[tuple[int, float, float, float, float], ...] = (
    (1, 3.6, -3.1, 20.0, 1.6),
    (4, -3.8, -4.6, 75.0, 2.2),
    (2, 5.2, 2.7, 140.0, 1.2),
    (6, -6.8, 3.9, 210.0, 2.8),
    (3, 1.6, 5.6, 300.0, 1.8),
    (5, -1.2, -7.2, 15.0, 3.2),
    (7, 7.6, -1.2, 260.0, 2.5),
    (2, 1.9, -2.2, 95.0, 0.5),
    (1, -2.7, -2.4, 180.0, 0.6),
)
ROCK_EMBED: float = 0.12
"""Fraction of each rock's height pushed below the seabed."""

# (object, x, y, yaw deg) for tools lying on the frame deck (z from the deck surface)
DECK_TOOLS: tuple[tuple[str, float, float, float], ...] = (
    ("hot_stab", 0.35, -1.0, 80.0),
    ("torque_tool", -0.35, -0.95, 100.0),
)


def panel_rotation_matrix() -> tuple[tuple[float, float, float], ...]:
    """Rows of the panel->site rotation matrix (for checks)."""
    w, x, y, z = PANEL_QUAT_WXYZ
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )


def panel_point_in_site(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """Transform a point from the panel frame to the site frame."""
    r = panel_rotation_matrix()
    return tuple(PANEL_POS[i] + sum(r[i][k] * p[k] for k in range(3)) for i in range(3))  # type: ignore[return-value]


def smoothstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


assert abs(math.sqrt(sum(c * c for c in PANEL_QUAT_WXYZ)) - 1.0) < 1e-9
