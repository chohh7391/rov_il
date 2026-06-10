# Pick Stone Colored-Cube Natural-Language Plan

## Summary / recommended approach

Convert `ROVLab-SO101-PickStone-v0` from the current single USD `rock` placeholder into a single-arm task with three procedurally spawned rigid cubes: `red_cube`, `green_cube`, and `blue_cube`. At reset, a project-owned Isaac Lab event samples one target color per env, stores the numeric target id on the env, records each cube's spawn height, and exposes a deterministic instruction string such as `pick up the red cube`.

Use HDF5 as the canonical data path for demonstrations. Add a small project-owned recorder term that writes numeric task metadata into HDF5 as tensors, especially `task/target_color_id` and optionally `task/instruction_id`. The offline converter in `rov_lerobot/lerobot_datasets/pick_stone/` then reads the per-episode target id, derives the instruction string, and passes it as `frame["task"]` on every `LeRobotDataset.add_frame(...)` call. In LeRobot `0.4.4` / Dataset v3, `add_frame` buffers the natural-language `task`, `save_episode` converts those strings into `task_index`, updates `meta/tasks.parquet`, and stores episode task lists in `meta/episodes/...`.

Keep direct in-recording LeRobot export compatible by changing `ROVSingleArmTaskEnvCfg.build_lerobot_frame(...)` to use the same recorded target metadata from `episode_data`, falling back to `self.task_description` only for tasks without per-episode language.

## 1. rov_lab task changes

### Current state

- [rov_lab/source/rov_lab/tasks/pick_stone/pick_stone_env_cfg.py](/home/home/rov_il/rov_lab/source/rov_lab/tasks/pick_stone/pick_stone_env_cfg.py) defines `PickStoneSceneCfg.rock` as a single `RigidObjectCfg` spawned from `OBJECT_DIR / "collected_rock" / "rock.usd"`.
- `ObservationsCfg.subtask_terms` only exposes `dummy`.
- `TerminationsCfg.success` uses `mdp.dummy_done`.
- [observations.py](/home/home/rov_il/rov_lab/source/rov_lab/tasks/pick_stone/mdp/observations.py) already has `stone_grasped(...)`, but its default `object_cfg` name is stale (`Orange001`).
- [terminations.py](/home/home/rov_il/rov_lab/source/rov_lab/tasks/pick_stone/mdp/terminations.py) has a plate-based `task_done(...)` plus `dummy_done(...)`; the new task needs target cube grasp+lift instead.
- The reusable template in [rov_single_arm_env_cfg.py](/home/home/rov_il/rov_lab/source/rov_lab/tasks/template/rov_single_arm_env_cfg.py) provides the robot, cameras, sensors, reset event, recorder config, and `build_lerobot_frame(...)`.

### Scene: replace the USD rock with three cubes

Modify `PickStoneSceneCfg`:

- Remove the `rock` asset and the unused `OBJECT_DIR` / `UsdFileCfg` imports.
- Add three `RigidObjectCfg`s:
  - `red_cube`: `prim_path="{ENV_REGEX_NS}/red_cube"`, initial position around `(1.00, -0.05, -1.50)`.
  - `green_cube`: `prim_path="{ENV_REGEX_NS}/green_cube"`, initial position around `(1.00, 0.05, -1.50)`.
  - `blue_cube`: `prim_path="{ENV_REGEX_NS}/blue_cube"`, initial position around `(1.00, 0.15, -1.50)`.
- Use `sim_utils.CuboidCfg(size=(0.03, 0.03, 0.03), ...)`.
- Use per-cube `sim_utils.PreviewSurfaceCfg(diffuse_color=...)`:
  - red: `(0.9, 0.05, 0.03)`
  - green: `(0.05, 0.75, 0.10)`
  - blue: `(0.05, 0.18, 0.9)`
- Keep underwater-style rigid settings consistent with the current rock:
  - `disable_gravity=True`
  - `linear_damping=10.0`
  - `angular_damping=10.0`
  - `max_linear_velocity=5.0`
  - `max_angular_velocity=5.0`
  - `collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True)`
- Add `mass_props=sim_utils.MassPropertiesCfg(mass=0.02)` as a starting point. This is an assumption to tune later because the current USD rock does not expose an equivalent mass in this config.
- Use identity rotations `(1.0, 0.0, 0.0, 0.0)`.

Proposed shared constants in `pick_stone_env_cfg.py`:

```python
CUBE_COLORS: tuple[str, ...] = ("red", "green", "blue")
CUBE_ASSET_NAMES: dict[str, str] = {
    "red": "red_cube",
    "green": "green_cube",
    "blue": "blue_cube",
}
TASK_INSTRUCTIONS: tuple[str, ...] = (
    "pick up the red cube",
    "pick up the green cube",
    "pick up the blue cube",
)
```

### Per-episode target selection

Add a new project-owned event function, preferably in a new file [rov_lab/source/rov_lab/tasks/pick_stone/mdp/events.py](/home/home/rov_il/rov_lab/source/rov_lab/tasks/pick_stone/mdp/events.py):

```python
def reset_target_color(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    cube_cfgs: list[SceneEntityCfg],
    spawn_positions: torch.Tensor,
    min_separation: float = 0.06,
) -> None:
    ...
```

Responsibilities:

- Sample `target_color_id` with `torch.randint(0, 3, (len(env_ids),), device=env.device)`.
- Store or update `env.pick_stone_target_color_id`, shape `[env.num_envs]`, dtype `torch.long`.
- Store `env.pick_stone_cube_spawn_heights`, shape `[env.num_envs, 3]`, dtype `torch.float32`. This should be the reset-time world `z` per cube, after adding `env.scene.env_origins`.
- Optionally store `env.pick_stone_cube_names = ("red_cube", "green_cube", "blue_cube")` and `env.pick_stone_instructions = TASK_INSTRUCTIONS` for introspection.
- Reset cube poses and velocities for the selected `env_ids`. Start deterministic and simple: three fixed non-overlapping positions relative to each env origin. Add random jitter only after the base task works.
- Write each cube state with `RigidObject.write_root_pose_to_sim(...)` and zero velocities with `write_root_velocity_to_sim(...)`.

Add a task-specific event config in `pick_stone_env_cfg.py`, replacing or extending `SingleArmEventCfg`:

```python
@configclass
class EventCfg(SingleArmEventCfg):
    reset_target = EventTerm(
        func=mdp.reset_target_color,
        mode="reset",
        params={
            "cube_cfgs": [
                SceneEntityCfg("red_cube"),
                SceneEntityCfg("green_cube"),
                SceneEntityCfg("blue_cube"),
            ],
            "spawn_positions": ...,
        },
    )
```

The success termination reads `env.pick_stone_target_color_id`. The recorder term also reads `env.pick_stone_target_color_id`. That gives one source of truth for success and language metadata.

### Instruction string generation

Add helpers in [rov_lab/source/rov_lab/tasks/pick_stone/mdp/task_metadata.py](/home/home/rov_il/rov_lab/source/rov_lab/tasks/pick_stone/mdp/task_metadata.py) or in `events.py` if kept small:

```python
TARGET_COLOR_NAMES: tuple[str, ...] = ("red", "green", "blue")
TARGET_INSTRUCTIONS: tuple[str, ...] = (
    "pick up the red cube",
    "pick up the green cube",
    "pick up the blue cube",
)

def instruction_from_target_id(target_color_id: int) -> str:
    return TARGET_INSTRUCTIONS[target_color_id]
```

For direct LeRobot recording, `build_lerobot_frame(...)` cannot receive `env` directly, but it receives `episode_data`. Therefore the target id must also be present in `episode_data._data["task"]["target_color_id"]`. The frame builder should read the latest id from `episode_data`, derive the instruction string, and set `"task": instruction`.

For HDF5 conversion, the offline converter reads `/data/demo_N/task/target_color_id` and applies the same mapping.

### Observations / subtask terms

Add subtask observation terms in `ObservationsCfg`:

- `red_cube_grasped = ObsTerm(func=mdp.object_grasped, params={"object_cfg": SceneEntityCfg("red_cube")})`
- `green_cube_grasped = ...`
- `blue_cube_grasped = ...`
- `target_cube_grasped = ObsTerm(func=mdp.target_cube_grasped, params={"cube_cfgs": [...]})`

Rename or generalize `stone_grasped(...)` to `object_grasped(...)` in [observations.py](/home/home/rov_il/rov_lab/source/rov_lab/tasks/pick_stone/mdp/observations.py). Keep the existing grasp heuristic:

- object root position near `ee_frame.data.target_pos_w[:, 1, :]`
- gripper joint position below `grasp_threshold`

Add `target_cube_grasped(...)` that computes all three cube grasp flags, stacks them `[num_envs, 3]`, and gathers using `env.pick_stone_target_color_id`.

Do not add target color id to the `policy` observation group. The intended conditioning path is natural-language `task` in LeRobot, not a numeric shortcut in state.

### Success termination keyed to the target cube

Replace `dummy_done` with a target-specific grasp+lift termination in [terminations.py](/home/home/rov_il/rov_lab/source/rov_lab/tasks/pick_stone/mdp/terminations.py):

```python
def target_cube_grasped_and_lifted(
    env: ManagerBasedRLEnv | DirectRLEnv,
    cube_cfgs: list[SceneEntityCfg],
    min_lift: float = 0.10,
    diff_threshold: float = 0.05,
    grasp_threshold: float = 0.60,
) -> torch.Tensor:
    ...
```

Logic:

- Require `target_cube_grasped(...)`.
- Gather the target cube current height from the three `RigidObject.data.root_pos_w[:, 2]` tensors.
- Gather the target cube spawn height from `env.pick_stone_cube_spawn_heights`.
- Return `target_grasped & ((target_z - target_spawn_z) >= min_lift)`.
- Wrong-color cube grasp/lift does not satisfy success because only the target cube is gathered.

Configure:

```python
success = DoneTerm(
    func=mdp.target_cube_grasped_and_lifted,
    params={
        "cube_cfgs": [
            SceneEntityCfg("red_cube"),
            SceneEntityCfg("green_cube"),
            SceneEntityCfg("blue_cube"),
        ],
        "min_lift": 0.10,
    },
)
```

## 2. HDF5 schema for the instruction

### Actual current recorder flow

Default teleop HDF5 route:

1. [rov_lab/scripts/teleop_se3_agent.py](/home/home/rov_il/rov_lab/scripts/teleop_se3_agent.py) uses `StreamingRecorderManager` unless `--use-lerobot-recorder` is passed.
2. `ROVSingleArmTaskEnvCfg.recorders` is currently `ActionStateRecorderManagerCfg`.
3. Isaac Lab recorder terms write:
   - `actions`
   - `states`
   - `obs`
   - `processed_actions`
   - `initial_state`
4. `StreamingRecorderManager` writes through `StreamingHDF5DatasetFileHandler`.
5. `StreamingHDF5DatasetFileHandler` recursively writes tensor data from `EpisodeData` into `/data/demo_N/...`. It writes `success`, `seed`, and `num_samples` as attrs.

Important constraint: the current HDF5 writer assumes non-dict values have `.cpu().numpy()`. It is not safe to put Python strings directly into `EpisodeData`. Therefore the canonical HDF5 schema should store numeric task metadata and derive strings in code.

Optional direct LeRobot route:

1. `teleop_se3_agent.py --use-lerobot-recorder` constructs `LeRobotDatasetCfg`.
2. `LeRobotRecorderManager.record_post_step()` calls `env.cfg.build_lerobot_frame(self._episodes[0], self._dataset_cfg)`.
3. [rov_single_arm_env_cfg.py](/home/home/rov_il/rov_lab/source/rov_lab/tasks/template/rov_single_arm_env_cfg.py) currently sets `"task": self.task_description`.
4. `LeRobotDatasetHandler.add_frame(...)` calls LeRobot `add_frame(frame=frame)`.

### Canonical HDF5 schema

Add a task-specific recorder term that records numeric metadata on every step:

HDF5 layout:

```text
/data/demo_000000/task/target_color_id      int64 [T]
/data/demo_000000/task/instruction_id       int64 [T]  # optional alias, same as target_color_id
```

Episode-level interpretation:

- Values must be constant within an episode.
- Converter reads the first value after `skip_first_frames`, or validates that all values are identical and then uses the first value.
- `0 -> "pick up the red cube"`
- `1 -> "pick up the green cube"`
- `2 -> "pick up the blue cube"`

Do not rely on `cfg.task` for this task after color conditioning lands. `cfg.task` should become a fallback only, for old HDF5 files or non-language-conditioned smoke tests.

### Recorder implementation plan

Add [rov_lab/source/rov_lab/tasks/pick_stone/mdp/recorders.py](/home/home/rov_il/rov_lab/source/rov_lab/tasks/pick_stone/mdp/recorders.py):

```python
class PickStoneTaskMetadataRecorder(RecorderTerm):
    def record_pre_step(self) -> tuple[str, dict[str, torch.Tensor]]:
        target_color_id = env.pick_stone_target_color_id.to(dtype=torch.long)
        return "task", {"target_color_id": target_color_id}
```

Add [rov_lab/source/rov_lab/tasks/pick_stone/mdp/recorders_cfg.py](/home/home/rov_il/rov_lab/source/rov_lab/tasks/pick_stone/mdp/recorders_cfg.py):

```python
@configclass
class PickStoneTaskMetadataRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = recorders.PickStoneTaskMetadataRecorder

@configclass
class PickStoneRecorderManagerCfg(ActionStateRecorderManagerCfg):
    record_pick_stone_task_metadata = PickStoneTaskMetadataRecorderCfg()
```

Then set `PickStoneEnvCfg.recorders = PickStoneRecorderManagerCfg()`.

This preserves the HDF5 writer's tensor-only contract and keeps target metadata outside `obs`.

## 3. rov_lerobot converter changes

### Current converter behavior

[rov_lerobot/lerobot_datasets/pick_stone/converter.py](/home/home/rov_il/rov_lerobot/lerobot_datasets/pick_stone/converter.py) currently writes every frame with:

```python
"task": cfg.task,
```

The CLI in [cli.py](/home/home/rov_il/rov_lerobot/lerobot_datasets/pick_stone/cli.py) accepts `--task` and stores a single fixed `cfg.task`.

### Required behavior

Add config fields in [config.py](/home/home/rov_il/rov_lerobot/lerobot_datasets/pick_stone/config.py):

```python
target_color_key: str = "task/target_color_id"
instruction_templates: tuple[str, ...] = (
    "pick up the red cube",
    "pick up the green cube",
    "pick up the blue cube",
)
allow_fixed_task_fallback: bool = False
```

Add `Hdf5Episode.read_scalar_constant(key: str) -> int | float | str` or a helper in `converter.py`:

- Read `episode.read(cfg.target_color_key)`.
- Flatten to 1D.
- Validate non-empty.
- Validate all values are the same; if not, raise a clear `ValueError`.
- Convert to `int`.
- Validate `0 <= target_color_id < len(cfg.instruction_templates)`.
- Return `cfg.instruction_templates[target_color_id]`.

Use this once per episode in `_add_episode_frames(...)`:

```python
episode_task = _episode_task_from_hdf5(episode, cfg)
...
frame = {
    "action": actions[frame_idx],
    "observation.state": states[frame_idx],
    "task": episode_task,
}
dataset.add_frame(frame=frame)
```

Keep `cfg.task` only as fallback for older datasets if `allow_fixed_task_fallback=True`. The default for the new task should be strict so missing target metadata fails early.

Update `cli.py`:

- Add `--target-color-key`, default `task/target_color_id`.
- Replace `--task` semantics with either:
  - `--fixed-task-fallback`, storing `allow_fixed_task_fallback=True`, or
  - keep `--task` but document it only applies when fallback is enabled.

Update [README.md](/home/home/rov_il/rov_lerobot/lerobot_datasets/pick_stone/README.md):

- Document HDF5 schema.
- Document that the dataset contains three tasks in `meta/tasks.parquet`, one per instruction actually recorded.
- Update example command to omit `--task` for color-conditioned data.

### LeRobot v0.4.4 task semantics

In `external_dependencies/lerobot/src/lerobot/datasets/lerobot_dataset.py`:

- `LeRobotDataset.add_frame(frame)` validates normal features, then pops `frame["task"]` and appends the natural-language string into `episode_buffer["task"]`.
- `LeRobotDataset.save_episode(...)` pops buffered tasks, computes unique `episode_tasks`, calls `self.meta.save_episode_tasks(episode_tasks)`, maps each frame's task string to `task_index`, and stores `task_index` in the data parquet.
- `LeRobotDatasetMetadata.save_episode_tasks(...)` writes new task strings to `meta/tasks.parquet`.
- Episode metadata stores the list of natural-language tasks for the episode.

For this task, every frame in an episode should have the same instruction. That creates one `task_index` per episode and supports language-conditioned policies that use the dataset `task` / `task_index` path.

## 4. Files touched

### rov_lab

- Modified: `rov_lab/source/rov_lab/tasks/pick_stone/pick_stone_env_cfg.py`
  - Replace `rock` with `red_cube`, `green_cube`, `blue_cube`.
  - Add constants for cube colors and instructions.
  - Add task-specific events, observations, terminations, and recorder manager config.
  - Update `task_description` to a generic fallback such as `"pick up the target colored cube"`.
- Modified: `rov_lab/source/rov_lab/tasks/pick_stone/mdp/__init__.py`
  - Export new event, metadata, termination, observation, and recorder helpers.
- Modified: `rov_lab/source/rov_lab/tasks/pick_stone/mdp/observations.py`
  - Generalize `stone_grasped` to `object_grasped`.
  - Add `target_cube_grasped`.
- Modified: `rov_lab/source/rov_lab/tasks/pick_stone/mdp/terminations.py`
  - Add `target_cube_grasped_and_lifted`.
  - Leave old plate-based `task_done` only if still referenced elsewhere; otherwise remove in a later cleanup command.
- New: `rov_lab/source/rov_lab/tasks/pick_stone/mdp/events.py`
  - Add reset target/color/cube pose event.
- New: `rov_lab/source/rov_lab/tasks/pick_stone/mdp/task_metadata.py`
  - Centralize target id to color/instruction mapping.
- New: `rov_lab/source/rov_lab/tasks/pick_stone/mdp/recorders.py`
  - Add `PickStoneTaskMetadataRecorder`.
- New: `rov_lab/source/rov_lab/tasks/pick_stone/mdp/recorders_cfg.py`
  - Add `PickStoneRecorderManagerCfg`.
- Modified: `rov_lab/source/rov_lab/tasks/template/rov_single_arm_env_cfg.py`
  - Update `build_lerobot_frame(...)` to prefer per-episode task metadata from `episode_data._data["task"]["target_color_id"]` when present and fall back to `self.task_description`.

### rov_lerobot

- Modified: `rov_lerobot/lerobot_datasets/pick_stone/config.py`
  - Add target metadata key, instruction templates, and fallback behavior.
- Modified: `rov_lerobot/lerobot_datasets/pick_stone/hdf5_reader.py`
  - Add a helper for constant scalar task metadata, or keep the helper in `converter.py`.
- Modified: `rov_lerobot/lerobot_datasets/pick_stone/converter.py`
  - Resolve a per-episode instruction from HDF5.
  - Pass the per-episode task into every `dataset.add_frame(...)`.
  - Validate missing/invalid/non-constant target metadata clearly.
- Modified: `rov_lerobot/lerobot_datasets/pick_stone/cli.py`
  - Add `--target-color-key`.
  - Make fixed `--task` a fallback mode, not the normal color-conditioned route.
- Modified: `rov_lerobot/lerobot_datasets/pick_stone/README.md`
  - Document HDF5 metadata and language-conditioned conversion.
- Modified or new tests, if the repo has a test location for this package:
  - Unit test constant target id to instruction conversion.
  - Unit test converter rejects missing target metadata unless fallback is enabled.
  - Unit test converter calls `add_frame` with different episode tasks for red/green/blue.

## 5. Open questions / risks / assumptions

- Operator visibility: the teleop UI must surface the current instruction. The keyboard teleop currently prints controls but does not display per-episode task instructions. A later implementation should print the sampled instruction after `env.reset()` and ideally expose it in the viewport/HUD.
- Reset ordering: `reset_target_color` must run after `reset_all`, otherwise `reset_scene_to_default` may overwrite cube poses or velocities.
- Recorder timing: if the metadata recorder records in `record_pre_step`, the first recorded action frame should already see the reset-selected target. This matches the current action/obs recorder cadence.
- HDF5 strings: do not store instruction strings in `EpisodeData` with the current writer. It only handles tensors. Use numeric id in HDF5 and derive strings in the converter.
- Direct LeRobot export: the canonical route should be HDF5 offline conversion, but `build_lerobot_frame(...)` should still use the same per-episode target metadata so `--use-lerobot-recorder` is not silently wrong.
- Small-cube physics: `0.03 m` cubes may be sensitive to collision margins, gripper clearance, damping, and disabled gravity. Start with the locked size, but expect simulator tuning.
- Spawn layout: fixed cube positions are easier to validate statically. Random position jitter should come after success and recording are correct.
- Target state storage on `env`: Manager-based event/termination functions can attach attributes dynamically, but this is implicit. A future custom env subclass would be cleaner, but would broaden scope.
- Wrong-color handling: the proposed success condition ignores wrong-color grasp/lift. It does not terminate on wrong-color grasp. If negative demonstrations or failure termination are desired, that should be a separate command.
- The registered `ROVLab-SO101-PickStone-Mimic-v0` points to `pick_stone_mimic_env_cfg:PickStoneMimicEnvCfg`, which is not present in the current tree. This plan does not address mimic.
- Front-mounted arm and bi-arm variants remain future work by locked decision.

## 6. Phased steps

1. Add shared task metadata constants and pure helper tests.
   - Add `task_metadata.py`.
   - Add converter-side helper for target id to instruction.
   - Unit-test mapping and validation without Isaac Sim.

2. Update `rov_lerobot` converter for per-episode tasks.
   - Add `target_color_key` and strict/fallback config.
   - Read `/task/target_color_id`.
   - Pass per-episode instruction to `dataset.add_frame`.
   - Test with synthetic HDF5 files containing two episodes with different target ids.

3. Add `rov_lab` recorder metadata path.
   - Add `PickStoneTaskMetadataRecorder`.
   - Add `PickStoneRecorderManagerCfg`.
   - Statically verify the returned value is tensor-only and HDF5-compatible.

4. Add reset target selection and cube spawning config.
   - Replace `rock` with three `CuboidCfg` assets.
   - Add reset event to initialize target ids, cube poses, spawn heights, and velocities.
   - Use `python -m py_compile` for changed `rov_lab` files; do not run simulator imports as a correctness gate.

5. Add grasp/lift observations and termination.
   - Generalize `stone_grasped`.
   - Add per-cube and target subtask terms.
   - Add `target_cube_grasped_and_lifted` and wire `TerminationsCfg.success`.
   - Static-check shape/device consistency.

6. Update direct LeRobot frame builder and teleop visibility.
   - `build_lerobot_frame(...)` reads recorded `task/target_color_id` if present.
   - `teleop_se3_agent.py` or task utilities print/display current instruction after reset.
   - Keep HDF5 offline conversion as the primary documented workflow.

7. Final static verification pass.
   - `rg "rock|Orange001|dummy_done|dummy_obs"` in `pick_stone` to catch stale wiring.
   - `python -m py_compile` on changed non-Isaac-runtime modules where possible.
   - For Isaac-dependent files, treat missing Isaac imports as expected and review syntax/consistency statically.
