# ROV Lab Assets

Large USD scenes, robot meshes, textures, and collected OceanSim assets are not tracked in Git.
They are published separately as a Hugging Face dataset:

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
  so101_follower.usd -> robots/so101_follower.usd
  robots/
    blue_rov_single_arm.usd
    so101_follower.usd
  OceanSim_assets/
    Bluerov/
    collected_MHL/
    collected_rock/
```

Keep the root-level `so101_follower.usd` compatibility link or copy. The
current `blue_rov_single_arm.usd` composes the arm payload from that relative
location.

`OceanSim_assets/` contains assets saved from the OceanSim source under
`external_dependencies/OceanSim`; it is distributed through the Hugging Face
dataset instead of being committed to this repository.

These files are large enough to exceed normal GitHub repository limits. Keep them local, restore them from the project asset archive, or manage them through a dedicated artifact store/Git LFS repository if needed.

Do not commit generated or third-party binary asset dumps directly to this repository.
