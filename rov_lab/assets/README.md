# ROV Lab Assets

Large USD scenes, robot meshes, textures, and collected OceanSim assets are not tracked in Git.

Expected local layout:

```text
rov_lab/assets/
  robots/
    blue_rov/
    so101_follower.usd
  scenes/
    OceanSim_assets/
```

These files are large enough to exceed normal GitHub repository limits. Keep them local, restore them from the project asset archive, or manage them through a dedicated artifact store/Git LFS repository if needed.

Do not commit generated or third-party binary asset dumps directly to this repository.
