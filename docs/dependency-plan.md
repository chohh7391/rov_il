# Dependency Plan

This document records the dependency constraints found in `external_dependencies/` and the current `uv` packaging direction.

## Target

- Package manager: `uv`
- Project Python version: `3.11`
- Primary simulator stack: Isaac Sim 5.1.0 + Isaac Lab
- Naming convention:
  - Distribution/project names use hyphens: `rov-il`, `rov-lab`.
  - Python import package names use underscores: `rov_lab`, `rov_lerobot`, `rov_sim`.
  - The repository/project identity is `rov_il`.

## External Dependency Findings

### Isaac Lab

Source: `external_dependencies/IsaacLab/README.md`, `external_dependencies/IsaacLab/source/*/setup.py`

- README advertises Python 3.11 and Isaac Sim 5.1.0 support.
- Current branch supports Isaac Sim 4.5 / 5.0 / 5.1.
- Python requirement in setup files: `>=3.10`.
- Important dependency pins:
  - `numpy<2`
  - `torch>=2.7`
  - `gymnasium==1.2.1`
  - `pillow==11.3.0`
  - `transformers==4.57.6`
  - `protobuf>=4.25.8,!=5.26.0` in `isaaclab_tasks` / `isaaclab_rl`
  - PyTorch wheel index is CUDA 12.8: `https://download.pytorch.org/whl/cu128`

Conclusion: Isaac Lab is compatible with the project Python 3.11 target.

### leisaac

Source: `external_dependencies/leisaac/README.md`, `external_dependencies/leisaac/source/leisaac/pyproject.toml`

- Python requirement: `>=3.10`.
- Classifier says Python 3.11 and Isaac Sim 5.1.0.
- Base dependencies:
  - `deepdiff`
  - `feetech-servo-sdk`
  - `psutil`
  - `pygame>=2.5.1,<2.7.0`
  - `pyserial`
- Optional dependency pins:
  - `isaaclab[isaacsim,all]==2.3.0`
  - leisaac declares `lerobot==0.4.2` as an optional extra, but this project will use `lerobot==0.4.4`.
  - `pydantic==2.10.6` for `gr00t`
  - `grpcio==1.74.0`, `protobuf==6.32.0` for `lerobot-async`

Conclusion: leisaac itself is compatible with Python 3.11. Avoid installing broad extras blindly. In particular, `lerobot-async` has a protobuf pin that may conflict with Isaac Lab protobuf expectations.

### LeRobot

Source: `external_dependencies/lerobot/pyproject.toml`, `external_dependencies/lerobot/uv.lock`

- Current local checkout version: `0.5.2`.
- Python requirement: `>=3.12`.
- Base dependencies include:
  - `torch>=2.7,<2.13.0`
  - `torchvision>=0.22.0,<0.28.0`
  - `numpy>=2.0.0,<2.3.0`
  - `gymnasium>=1.1.1,<2.0.0`
  - `draccus==0.10.0`
  - `huggingface-hub>=1.0.0,<2.0.0`

Conflicts with the project Python 3.11 / Isaac Lab environment:

- Python conflict: local LeRobot `0.5.2` requires Python `>=3.12`.
- NumPy conflict: LeRobot `0.5.2` requires `numpy>=2.0.0`, Isaac Lab requires `numpy<2`.
- Gymnasium is probably compatible by range because Isaac Lab pins `gymnasium==1.2.1`, which satisfies LeRobot `>=1.1.1,<2.0.0`.

Conclusion: the current `external_dependencies/lerobot` checkout cannot be used in the Python 3.11 Isaac Lab environment. For Python 3.11, use LeRobot `0.4.4` from PyPI or check out a compatible local LeRobot `0.4.4` source tree.

### OceanSim

Source: `external_dependencies/OceanSim/docs/README.md`, `external_dependencies/OceanSim/docs/subsections/installation.md`, `external_dependencies/OceanSim/config/extension.toml`

- OceanSim is an Isaac Sim extension, not a standard Python package with `pyproject.toml`.
- README advertises Isaac Sim 5.0 compatibility.
- Installation docs say OceanSim has no additional prerequisites beyond Isaac Sim.
- Extension dependencies are Omniverse/Isaac extension names:
  - `omni.kit.uiapp`
  - `omni.isaac.ui`
  - `omni.isaac.core`

Conclusion: OceanSim should not be managed as a normal uv Python package. It should be configured as an Isaac Sim extension path/runtime dependency.

## Main Conflict Summary

The blocking conflict has two layers:

```text
Python target: 3.11
Isaac Lab: numpy<2, python>=3.10
LeRobot local checkout: python>=3.12, numpy>=2
LeRobot 0.4.4: pulls rerun-sdk>=0.24,<0.27, which requires numpy>=2
ROV Lab visualization: use rerun-sdk>=0.22,<0.23 because 0.22.1 supports numpy>=1.23, while 0.23.x requires numpy>=2
```

Therefore, do not include `external_dependencies/lerobot` as a uv workspace member or editable dependency while targeting Python 3.11. Also do not install `lerobot==0.4.4` into the same environment as Isaac Lab unless the NumPy conflict is resolved upstream.

## Recommended uv Strategy

Use one Python 3.11 uv extra for simulator-facing packages:

```text
uv sync --extra lab

contents:
rov_lab
h5py
isaacsim[all,extscache]==5.1.0
torch==2.7.0
torchvision==0.22.0
external_dependencies/leisaac
external_dependencies/IsaacLab/source/isaaclab
external_dependencies/IsaacLab/source/isaaclab_tasks
external_dependencies/IsaacLab/source/isaaclab_mimic
```

Use a separate Python 3.11 uv extra for learning:

```text
uv sync --extra learning

contents:
lerobot==0.4.4
```

The `lab` and `learning` extras are declared as conflicting in `pyproject.toml`, so `uv` should not install them into the same environment.

Recommended commands:

```bash
UV_PROJECT_ENVIRONMENT=.venv-rov_lab uv sync --extra lab
UV_PROJECT_ENVIRONMENT=.venv-rov_lerobot uv sync --extra learning
```

Use `.venv-rov_lab` for `rov_lab` / Isaac Lab demo collection work and `.venv-rov_lerobot` for `rov_lerobot` / LeRobot training work. Running both extras into one `.venv` is intentionally blocked.

The `lab` extra includes Isaac Sim and CUDA PyTorch directly. The required package indexes are configured in `pyproject.toml`:

```bash
https://pypi.nvidia.com
https://download.pytorch.org/whl/cu128
```

Do not manually install Isaac Sim / CUDA PyTorch with `uv pip install` after `uv sync`; keep these packages in `pyproject.toml`. `uv sync` is exact by default and removes packages that are not declared in the lockfile.

`h5py` is explicitly included in the `lab` extra because Isaac Lab's dataset recorder imports it at runtime but does not currently declare it in the package dependencies.

Do not use the current local `external_dependencies/lerobot` checkout unless the project switches to Python 3.12 and resolves the Isaac Lab `numpy<2` conflict by separating environments.

## Environment Split Option

If LeRobot `0.5.x` is required later, split environments:

```text
uv env: rov_lab / rov_sim
  Python 3.11
  Isaac Lab
  leisaac
  numpy<2

uv env: rov_lerobot
  Python 3.12+
  LeRobot 0.5.x
  numpy>=2
```

This split is not the preferred starting point because the user requested Python 3.11.

## Immediate Next Steps

1. Keep `.python-version` at `3.11`.
2. Update project-owned packages to require Python `>=3.11,<3.12`.
3. Create a root `pyproject.toml` for uv with:
   - preferred: `lerobot==0.4.4`
   - alternative: a local LeRobot checkout pinned to a Python 3.11-compatible `0.4.4` tag/commit
4. Generate `uv.lock` after the LeRobot version decision.
