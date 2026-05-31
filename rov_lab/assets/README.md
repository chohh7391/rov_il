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
  robots/
    blue_rov_single_arm.usd
    so101_follower.usd
  scenes/
    collected_MHL/
```

Some assets in `objects/`, `robots/`, and `scenes/` were prepared from
OceanSim assets under `external_dependencies/OceanSim`. Keep the prepared
assets in the Hugging Face dataset, not in this Git repository.

Do not commit generated or third-party binary asset dumps directly to this
repository.
