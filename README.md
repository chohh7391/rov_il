# rov_il

`rov_il` is an underwater ROV imitation-learning project. The repository is organized around collecting teleoperation demonstrations in Isaac Lab, converting them to LeRobot datasets, training imitation-learning policies, and evaluating trained policies in simulation.

## Project Layout

- `rov_lab/`: Isaac Lab-based ROV demonstration collection. This currently owns the underwater ROV task, keyboard teleoperation, OceanSim-derived sensors, HDF5 recording, and runtime diagnostics.
- `rov_lerobot/`: LeRobot dataset conversion, policy training, and real-robot LeRobot integration code.
- `rov_sim/`: OceanSim-based Isaac Sim policy evaluation code.
- `external_dependencies/`: External source dependencies managed as Git submodules.
- `datasets/`, `checkpoints/`, `runs/`: Local generated artifacts. These are ignored by Git.

## External Dependencies

The project expects these submodules:

| Path | Repository | Version |
| --- | --- | --- |
| `external_dependencies/IsaacLab` | `https://github.com/isaac-sim/IsaacLab` | `v2.3.2` |
| `external_dependencies/OceanSim` | `https://github.com/umfieldrobotics/OceanSim` | current pinned commit |
| `external_dependencies/leisaac` | `https://github.com/lightwheelai/leisaac` | current pinned commit |
| `external_dependencies/lerobot` | `https://github.com/huggingface/lerobot` | `v0.4.4` |

Do not initialize submodules recursively by default. `leisaac` contains its own IsaacLab submodule under `dependencies/`, but this project uses `external_dependencies/IsaacLab` pinned to `v2.3.2`.

Recommended clone/update flow:

```bash
git clone git@github.com:chohh7391/rov_il.git
cd rov_il
git submodule update --init external_dependencies/IsaacLab
git submodule update --init external_dependencies/OceanSim
git submodule update --init external_dependencies/leisaac
git submodule update --init external_dependencies/lerobot
git submodule update --init external_dependencies/MarineGym
```

Avoid:

```bash
git submodule update --init --recursive
```

## Environment

Python version:

```bash
python --version  # 3.11
```

ROV Lab environment:

```bash
UV_PROJECT_ENVIRONMENT=.venv-rov_lab uv sync --extra lab
source .venv-rov_lab/bin/activate
```

Isaac Sim and PyTorch packages are large and may need to be installed explicitly in the `rov_lab` environment:

```bash
uv pip install --python .venv-rov_lab/bin/python "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
uv pip install --python .venv-rov_lab/bin/python -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

LeRobot version target:

```bash
LeRobot == 0.4.4
```

## Keyboard Teleoperation

Currently, `rov_lab` supports keyboard teleoperation only.

## Assets

Large simulation assets are not tracked in Git. They are published separately as a Hugging Face dataset:

```text
chohh7391/rov_il-rov_lab-assets
```

From the repository root, download them into the path expected by `rov_lab`:

```bash
huggingface-cli download chohh7391/rov_il-rov_lab-assets \
  --repo-type dataset \
  --local-dir rov_lab/assets
```

After download, the local layout should be:

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
OceanSim assets under `external_dependencies/OceanSim`. The prepared asset
archive is distributed through the Hugging Face dataset instead of being
committed to this repository.

The code resolves this directory by default:

```text
<repo-root>/rov_lab/assets
```

You can override the asset root with:

```bash
export ROV_LAB_ASSETS_ROOT=/path/to/rov_lab/assets
```

Example:

```bash
source .venv-rov_lab/bin/activate
cd rov_lab
python scripts/teleop_se3_agent.py \
  --task ROVLab-SO101-PickStone-v0 \
  --teleop_device keyboard \
  --num_envs 1 \
  --device cuda \
  --enable_cameras \
  --record \
  --dataset_file /home/home/rov_il/datasets/hdf5/dataset.hdf5
```

Useful debug options:

```bash
--print --print_sensor_every 30
--visualize
```

## Data Flow

```text
rov_lab teleoperation
  -> raw HDF5 demonstrations
  -> rov_lerobot conversion
  -> LeRobot Dataset
  -> ACT / Diffusion Policy training
  -> PyTorch checkpoint
  -> rov_sim policy evaluation
```

## Artifact Policy

Do not commit generated datasets, checkpoints, rollout logs, videos, or local virtual environments. These are ignored by `.gitignore`.
