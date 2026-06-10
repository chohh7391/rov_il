# Pick Stone LeRobot Dataset

This package converts ROV Lab pick-stone Isaac Lab HDF5 demonstrations into a
LeRobot Dataset v3 directory for LeRobot `0.4.4` training.

Default HDF5 mapping:

- `actions` -> `action`
- `obs/joint_pos` -> `observation.state`
- `obs/front` -> `observation.images.front`
- `obs/wrist` -> `observation.images.wrist`
- `task/target_color_id` -> per-episode language `task`

For the colored-cube task, every recorded episode must include a constant integer
target id at `task/target_color_id`:

- `0` -> `pick up the red cube`
- `1` -> `pick up the green cube`
- `2` -> `pick up the blue cube`

The converter reads that id once per episode and passes the corresponding natural
language string as `frame["task"]` for every frame in that episode. LeRobot stores
the distinct recorded instructions in `meta/tasks.parquet` and writes each frame's
`task_index` into the data parquet. Omit `--task` for normal color-conditioned
data. Use `--fixed-task-fallback --task "..."` only for legacy HDF5 files that do
not contain `task/target_color_id`.

Example:

```bash
uv --project rov_lerobot run rov-pick-stone-to-lerobot \
  --repo-id rov_il/pick_stone \
  --output-root datasets/lerobot/pick_stone \
  --hdf5-files datasets/hdf5/dataset.hdf5 \
  --camera-keys front wrist \
  --overwrite
```

If the local `datasets/hdf5/dataset.hdf5` file is locked by another process, run
with `HDF5_USE_FILE_LOCKING=FALSE`.

Train an ACT baseline after conversion:

```bash
uv --project rov_lerobot run lerobot-train \
  --dataset.repo_id=rov_il/pick_stone \
  --dataset.root=datasets/lerobot/pick_stone \
  --policy.type=act \
  --output_dir=checkpoints/act_pick_stone \
  --job_name=act_pick_stone \
  --policy.device=cuda \
  --wandb.enable=false
```
