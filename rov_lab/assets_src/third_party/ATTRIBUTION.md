# Third-party asset attribution

The subsea work-site assets (`rov_lab/assets/objects/*`, `rov_lab/assets/scenes/shipwreck_site`,
`rov_lab/assets/scenes/portuguese_ledge`) are built from the sources below by
`rov_lab/scripts/tools/build_subsea_assets.py`. Download URLs, versions and pinned commits are listed in
`manifest.json`. The licenses were checked on 2026-11-16 against the Gazebo Fuel API, each repository's LICENSE file
and the Poly Haven API. Apache-2.0 notices are in `NOTICE`.

## Creative Commons Attribution 4.0 (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/)

| Asset | Creator | Source | Changes made by rov_lab |
|---|---|---|---|
| OceanFloorShipwreck (v2) | Cole Biesemeyer and Rafi Abdullah, Open Robotics | https://app.gazebosim.org/OpenRobotics/fuel/models/OceanFloorShipwreck | Cropped to a 180 m × 180 m area around a work site, recentred, and converted to USD. Seabed subdivided, given small ripples, levelled under the work site, and retextured with CC0 sand. Corals have no collision. |
| Ocean Rock 01 – 07 (v1) | Cole Biesemeyer, Open Robotics | https://app.gazebosim.org/Cole/fuel/models/Ocean%20Rock%2001 (02 – 07 alike) | Converted to USD, rescaled so the largest extent is 1 m, then placed and rescaled in the work-site layout. |

## CC0 1.0 (public domain dedication)

| Asset | Creator | Source | Use |
|---|---|---|---|
| Portuguese Ledge (v3) | Open Robotics. Bathymetry by Dr. Dave Caress and the MBARI Seafloor Mapping Lab | https://app.gazebosim.org/OpenRobotics/fuel/models/Portuguese%20Ledge | 250 m patch of tile 1, meshed, subdivided and levelled under the work site |
| Coast Sand 01 | Rob Tuytel, Poly Haven | https://polyhaven.com/a/coast_sand_01 | seabed (shipwreck site) |
| Coral Mud 01 | Rob Tuytel, Poly Haven | https://polyhaven.com/a/coral_mud_01 | seabed (Portuguese Ledge) |
| Rock Boulder Dry | Dimitrios Savva and Rico Cilliers, Poly Haven | https://polyhaven.com/a/rock_boulder_dry | reserved for rocks |
| Rusty Painted Metal | Amal Kumar, Poly Haven | https://polyhaven.com/a/rusty_painted_metal | valve panel |

CC0 needs no attribution. It is listed here for provenance.

## Apache License 2.0

UUV Simulator (`rov_bop_panel`, `hot_stab`, `torque_tool`) and DAVE (`electrical_panel` frame and socket, `f_to_m_plug`).
See `NOTICE` and `LICENSE-Apache-2.0.txt`.

## Not used

These are excluded because of their licenses:

- Reach Robotics description files. The SDK license allows use only with Reach products.
- Girona 500 description. It has no open-source license.
- DAVE AxialSeamountPatch. The source data is CC BY-NC-SA.
- DAVE Santorini. The terms of the EMODnet source data were not confirmed.
- Sketchfab NC/ND models.
- HoloOcean worlds.
- The UUV Simulator texturelib texture.

The DAVE `niskin` model points to a mesh on Gazebo Fuel (`cole/niskin`) that has not been fetched yet.
