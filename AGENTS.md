# ROV_IL Project Agent Instructions

This file provides guidance to AI agents when working in the `rov_il` repository.

## Project Purpose

`rov_il` is a research and development project for building an imitation learning pipeline for underwater ROV manipulation.

The project target is to:

- Design an imitation learning environment from underwater robot pilot operation data.
- Build a simulator-based underwater robot imitation learning framework.
- Define simulation environments and data schemas for teleoperation data collection.
- Validate whether imitation learning from simulated ROV operation data is feasible.

Near-term milestone:

- In 2026, run a fixed ROV with a single robot arm in Isaac Sim, collect teleoperation demonstrations, train an imitation learning policy, and demonstrate policy inference in simulation.

Longer-term milestone:

- In 2027, transfer the workflow toward real water-tank experiments.

## System Architecture

The intended end-to-end data flow is:

```text
rov_lab
  -> Isaac Lab-based underwater demo collection
  -> raw HDF5 demonstrations
  -> rov_lerobot conversion
  -> LeRobot Dataset
  -> LeRobot training (ACT / Diffusion Policy)
  -> PyTorch checkpoint
  -> rov_sim policy deployment and evaluation in OceanSim-based Isaac Sim
```

Core assumptions:

- Simulator: Isaac Sim 5.1.0.
- Underwater demo collection: `rov_lab` ports OceanSim's Isaac Sim underwater functionality into Isaac Lab and uses leisaac code as an imported dependency where useful.
- Environment framework for demo collection: Isaac Lab `ManagerBasedEnv`.
- High-fidelity simulation evaluation: `rov_sim` uses an OceanSim-based Isaac Sim environment.
- Robot configuration: fixed ROV base plus one manipulator arm, initially FR3 unless a task explicitly specifies another arm.
- Teleoperation: VR, haptic, or master-slave control paths may be used for demonstration collection.
- Dataset bridge: HDF5 demonstrations are converted into LeRobot-compatible datasets.
- Training: LeRobot policies, primarily ACT or Diffusion Policy, in `rov_lerobot`.
- Deployment: learned LeRobot policies are loaded through `rov_sim` and evaluated in an OceanSim-based Isaac Sim environment.

## Codebase Map

- `rov_lab/`: Isaac Lab-based underwater demonstration generation code. This is currently implemented by merging changes into `leisaac`; the target direction is to keep project-owned code here and import leisaac functionality as needed. Put OceanSim-to-IsaacLab underwater ports, Isaac Lab environments/tasks, robot and sensor configs, teleoperation, observation collection, HDF5 writing, and demo replay/validation here.
- `rov_lerobot/`: Project-owned LeRobot integration code for real-robot and learning workflows. Put LeRobot-format policy, robot, and teleoperation implementations here, along with dataset conversion, dataset validation, training wrappers/configs, and checkpoint management.
- `rov_sim/`: OceanSim-based Isaac Sim evaluation code. Put simulation environments used to test LeRobot-trained policies, policy loading/runtime adapters, rollout scripts, metrics, logs, and evaluation visualization here.
- `shared/` or `rov_common/`: Optional project-owned shared contracts. Put HDF5 schemas, observation/action names, task ids, metadata definitions, and common typed config objects here if the same definitions are needed across `rov_lab`, `rov_lerobot`, and `rov_sim`.
- `datasets/`: Local datasets and generated demonstration data. Do not commit large generated datasets unless explicitly requested.
- `checkpoints/`: Local trained model checkpoints. Do not commit large checkpoint files unless explicitly requested.
- `runs/`: Local rollout/evaluation outputs. Do not commit large generated logs, videos, or rollout artifacts unless explicitly requested.
- `external_dependencies/OceanSim/`: Vendored or reference OceanSim source. Prefer importing or wrapping it rather than copying large pieces into project code.
- `external_dependencies/leisaac/`: Vendored or reference leisaac source. Treat as dependency code; `rov_lab` should import it rather than modifying it.
- `external_dependencies/IsaacLab/`: Vendored Isaac Lab source. Avoid modifying unless the task explicitly requires an upstream patch.
- `external_dependencies/lerobot/`: Vendored LeRobot source. It has its own `AGENTS.md`; follow that file when editing inside this subtree.

## Development Workflow

- Prefer project-owned code in `rov_lab/`, `rov_lerobot/`, and `rov_sim/` over editing vendored dependencies.
- Keep dependency-specific glue isolated behind small adapters so Isaac Lab, OceanSim, and LeRobot can be upgraded independently.
- Use Python type hints for all new function signatures, class attributes, dataclasses, and non-trivial variables.
- Add focused tests for conversion, schema validation, and policy adapter behavior when those surfaces are changed.
- Do not run broad auto-formatters across vendored or unrelated files. Match local style unless the user explicitly asks for formatting.
- Use `rg`/`rg --files` for code search.

## Environment Notes

The final environment names may change, but use these conventions unless the repository later defines a stricter setup:

```bash
conda activate rov_il
```

For Isaac Sim / Isaac Lab commands, make sure the Isaac Sim and Isaac Lab environment variables are configured according to the local install before running simulator scripts.

For LeRobot training or dataset conversion, prefer running from the repository root so relative dataset, checkpoint, and config paths resolve consistently.

## Data Collection Runbook

The collection stage belongs to `rov_lab` and should be implemented around Isaac Lab `ManagerBasedEnv` tasks.

Expected responsibilities:

- Load the fixed ROV plus manipulator robot model.
- Port or adapt OceanSim underwater dynamics, rendering, and sensor logic into the Isaac Lab environment where needed.
- Import leisaac functionality instead of duplicating it when the interface is stable enough.
- Support teleoperation inputs such as VR, haptic devices, or master-slave controls.
- Record observations, actions, rewards/events if available, sensor streams, episode metadata, and timing information.
- Save raw demonstrations to HDF5.

HDF5 demonstration files should preserve enough metadata to convert deterministically into a LeRobot dataset:

- Task name and version.
- Robot asset/config version.
- Control mode and action space.
- Observation feature schema.
- Camera names, image sizes, and frame rate.
- Episode start/end indices and success/failure labels when available.
- Simulator, Isaac Lab, OceanSim, and project commit/version metadata when available.

## Dataset Conversion Rules

Implement HDF5 to LeRobot conversion in project-owned code, normally under `rov_lerobot/`.

Conversion requirements:

- Keep the raw HDF5 data immutable.
- Validate feature shapes and dtypes before writing LeRobot output.
- Preserve episode boundaries.
- Preserve timestamps or frame indices.
- Write LeRobot-compatible metadata.
- Make conversion deterministic and resumable when practical.
- Include a small validation mode that loads the generated LeRobot dataset and checks at least one episode.

Use names such as `isaaclab2lerobot.py` only for command entry points or explicit converters. Keep reusable logic in importable modules.

## Training Rules

LeRobot training and real-robot LeRobot interfaces belong to `rov_lerobot`.

`rov_lerobot` should implement:

- LeRobot-format policies.
- LeRobot-format robot interfaces for the real ROV/manipulator system.
- LeRobot-format teleoperation interfaces for real-robot demonstration collection.
- Dataset conversion from `rov_lab` HDF5 demonstrations to LeRobot datasets.
- Training and validation entry points.

LeRobot training should use standard LeRobot policy implementations unless the task explicitly calls for custom policy research.

Preferred initial policies:

- ACT for sequence prediction baselines.
- Diffusion Policy for higher-capacity manipulation behavior.

Training outputs should be organized with:

- Dataset identifier or local dataset path.
- Task name.
- Policy type.
- Date or run id.
- Checkpoint path.
- Training config snapshot.

Do not hardcode absolute user-specific paths into reusable modules. Put local paths in CLI arguments, config files, or environment variables.

## Deployment Rules

Simulation policy deployment belongs to `rov_sim`.

`rov_sim` should use an OceanSim-based Isaac Sim environment to evaluate policies trained through `rov_lerobot`. It does not own demonstration collection and should not duplicate `rov_lab` teleoperation/recording logic.

Deployment code should:

- Load a LeRobot-trained PyTorch checkpoint.
- Reconstruct the observation preprocessing expected by the policy.
- Convert OceanSim/Isaac Sim observations into the policy input schema.
- Convert policy outputs into the simulated robot action/control schema.
- Keep inference timing explicit.
- Log policy inputs, outputs, episode status, and failures enough to debug behavior.

Avoid mixing training-only dependencies into simulator runtime code unless necessary.

## Core Conventions

1. **Project-owned changes first:** Put new code in `rov_lab/`, `rov_lerobot/`, or `rov_sim/` unless there is a clear reason to patch `external_dependencies/`.
2. **Typed Python:** New Python code must use type hints. Prefer dataclasses or typed config objects for schemas.
3. **LeRobot compatibility:** Dataset and policy interfaces must follow LeRobot feature/schema conventions.
4. **Simulator reproducibility:** Task configs, robot configs, and dataset schemas should be versioned or recorded in metadata.
5. **No large artifacts by default:** Do not commit generated HDF5 datasets, LeRobot datasets, logs, videos, or checkpoints unless the user explicitly asks.
6. **Module boundaries:** Keep `rov_lab` for Isaac Lab demo generation, `rov_lerobot` for LeRobot learning/real-robot interfaces, and `rov_sim` for OceanSim-based Isaac Sim policy evaluation.
7. **Vendored code caution:** Changes under `external_dependencies/` should be minimal, clearly justified, and easy to separate from project code.
8. **Shared contracts:** If observation/action/HDF5 definitions are used by more than one top-level package, move them into `rov_common` or another shared contract layer instead of redefining them independently.

## Suggested Initial Implementation Order

1. Define the first HDF5 demonstration schema and observation/action naming contract.
2. Implement the ROV + manipulator underwater task skeleton in `rov_lab` using Isaac Lab and imported leisaac functionality where useful.
3. Port the needed OceanSim Isaac Sim underwater behavior into the Isaac Lab collection environment.
4. Add a `rov_lab` teleoperation recording entry point that writes one valid HDF5 episode.
5. Implement `rov_lerobot` conversion from that HDF5 schema to a LeRobot dataset.
6. Implement or adapt LeRobot policy, robot, and teleoperation interfaces in `rov_lerobot` for real-robot data collection and training.
7. Train a small ACT baseline on a toy or smoke-test dataset.
8. Implement `rov_sim` OceanSim-based policy evaluation that loads the trained LeRobot checkpoint and performs rollout inference.
9. Expand to richer sensors, haptic/VR inputs, task success metrics, and real water-tank readiness.
