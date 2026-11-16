# ROV Lab Assets

Large USD scenes, robot meshes, textures, and prepared OceanSim-derived assets
are not tracked in Git. They are published separately as a Hugging Face dataset:

```text
chohh7391/rov_il-rov_lab-assets
```

From the repository root, restore them with:

```bash
huggingface-cli download chohh7391/rov_il-rov_lab-assets \
  --repo-type dataset \
  --local-dir rov_lab/assets
```

Expected local layout:

```text
rov_lab/assets/
  objects/
    collected_rock/
    rov_bop_panel/          valve panel articulation (UUV Simulator, Apache-2.0)
    hot_stab/ torque_tool/  ROV tools (UUV Simulator, Apache-2.0)
    subsea_frame/ wet_mate_plug/   skid frame, socket, plug (DAVE, Apache-2.0)
    ocean_rock_01..07/      boulders (Gazebo Fuel, CC-BY 4.0)
  robots/
    blue_rov_single_arm.usd
    bravo7_class/
    so101_follower.usd
  scenes/
    collected_MHL/
    shipwreck_site/         OceanFloorShipwreck crop + work-site layout (Gazebo Fuel, CC-BY 4.0)
    portuguese_ledge/       MBARI bathymetry patch + work-site layout (Gazebo Fuel, CC0)
  third_party/              raw downloads (fetch_third_party_assets.py); not needed at runtime
```

The subsea work-site assets are rebuilt from public sources with:

```bash
python rov_lab/scripts/tools/fetch_third_party_assets.py
<isaaclab-3.0-python> rov_lab/scripts/tools/build_subsea_assets.py
```

Their licenses and required notices are in `rov_lab/assets_src/third_party/` (`manifest.json`, `NOTICE`,
`ATTRIBUTION.md`). Keep those files next to the assets whenever they are redistributed (for example, copy them
into the Hugging Face dataset).

Some assets in `objects/`, `robots/`, and `scenes/` were prepared from
OceanSim assets under `external_dependencies/OceanSim`. Keep the prepared
assets in the Hugging Face dataset, not in this Git repository.

Do not commit generated or third-party binary asset dumps directly to this
repository.
