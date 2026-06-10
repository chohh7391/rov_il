# rov_lerobot

`rov_lerobot` owns the LeRobot side of the ROV imitation-learning pipeline:

```text
rov_lab HDF5 demonstrations
  -> rov_lerobot HDF5 to LeRobot Dataset v3 conversion
  -> LeRobot training with built-in policies such as ACT or Diffusion Policy
  -> checkpoints for later rov_sim deployment
```

It is intentionally a separate uv project from `rov_lab`. `rov_lab` depends on
Isaac Sim / Isaac Lab packages, while `rov_lerobot` depends on `lerobot==0.4.4`
and its learning stack. Keeping them separate avoids dependency conflicts.

## Project Layout

```text
rov_lerobot/
  pyproject.toml                    # uv project metadata and dependencies
  uv.lock                           # rov_lerobot-only dependency lockfile
  lerobot_datasets/
    pick_stone/                     # pick-stone HDF5 -> LeRobot v3 converter
      cli.py                        # rov-pick-stone-to-lerobot entry point
      config.py                     # conversion defaults
      converter.py                  # LeRobotDataset v3 writer
      features.py                   # feature/schema inference
      hdf5_reader.py                # Isaac Lab HDF5 reader
  lerobot_policies/                 # future custom policies if built-ins are insufficient
  lerobot_robots/                   # future real-robot LeRobot interfaces
  lerobot_teleoperators/            # future real-robot teleoperation interfaces
```

For the current offline learning workflow, `lerobot_robots/` is not required.
The dataset converter plus LeRobot built-in policies are enough.

## Environment

Run all commands from the repository root:

```bash
cd /home/home/rov_il
```

Use the `rov_lerobot` uv project explicitly:

```bash
uv --project rov_lerobot run <command>
```

If a shell or conda environment is already active, uv may print a warning like:

```text
VIRTUAL_ENV=... does not match the project environment path rov_lerobot/.venv
```

That is expected. `uv --project rov_lerobot` uses `rov_lerobot/.venv`.

## Convert Pick-Stone HDF5 to LeRobot Dataset v3

Input HDF5 demos are expected under `datasets/hdf5/`, for example:

```textㅇ
datasets/hdf5/dataset.hdf5
```

Create the local LeRobot Dataset v3:

```bash
HDF5_USE_FILE_LOCKING=FALSE PYTHONPATH=. uv --project rov_lerobot run rov-pick-stone-to-lerobot \
--repo-id rov_il/pick_stone \
--output-root datasets/lerobot_v3/pick_stone \
--hdf5-files datasets/hdf5/dataset.hdf5 \
--camera-keys front wrist sonar \
--fixed-task-fallback \
--task "Pick up the target colored cube and lift it off theseabed." \
--overwrite
```

Check that conversion produced LeRobot metadata:

```bash
ls datasets/lerobot/pick_stone/meta/info.json
```

Default feature mapping:

```text
actions       -> action
obs/joint_pos -> observation.state
obs/front     -> observation.images.front
obs/wrist     -> observation.images.wrist
obs/sonar     -> observation.images.sonar
```

The converter follows LeRobot Dataset v3 as implemented by `lerobot==0.4.4`.

> Note: `--require-success` keeps only episodes whose HDF5 group has a `success`
> attribute set to true. `rov_lab` recordings write this attribute inconsistently — some
> episodes have it, some do not, and an episode flagged successful may still be empty
> (`num_samples == 0`). Episodes without `success == True`, or that are too short, are
> dropped, so `--require-success` can leave you with very few or zero episodes (in which
> case conversion aborts with a clear error). Omit the flag (the default) to keep every
> sufficiently long episode regardless of its success flag.

## Train ACT

After conversion, train an ACT baseline:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True uv --project rov_lerobot run lerobot-train \
--dataset.repo_id=rov_il/pick_stone \
--dataset.root=datasets/lerobot_v3/pick_stone \
--policy.type=act \
--policy.push_to_hub=false \
--output_dir=checkpoints/act_pick_stone_smoke \
--job_name=act_pick_stone_smoke \
--wandb.enable=policy.device=cuda \
--wandb.enable=false \
--batch_size=1 \
--steps=1000 \
--policy.chunk_size=20 \
--policy.n_action_steps=20 \
--policy.dim_model=256 \
--policy.dim_feedforward=1024 \
--policy.n_encoder_layers=2 \
--policy.n_decoder_layers=1 \
--policy.n_vae_encoder_layers=2
```

Use `--policy.push_to_hub=false` for local-only training. If pushing the trained
policy to Hugging Face Hub, set `--policy.repo_id=<user_or_org>/act_pick_stone`
instead.

This is also not valid because `--help` becomes a separate shell command:

```bash
uv --project rov_lerobot run lerobot-train --help
```
