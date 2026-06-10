# STATUS.md — Status Board

Shared board between the main agent and the sub agent. The sub agent updates only
`## Latest Report`. The main agent reviews the report, records a verdict under `## Review`,
and when moving on, replaces `## Current Command` in `docs/SUBAGENTS.md`.

---

## Latest Report

- **Command ID:** CMD-008
- **Status:** done
- **Deliverable:** `rov_lab/source/rov_lab/devices/keyboard/rov_keyboard.py`
- **What was done:** Added four ROV keyboard actions using the existing 14-D `np.asarray(...) * self.rov_sensitivity` pattern: `rov_up` sets index 2 to `+1.0`, `rov_down` sets index 2 to `-1.0`, `rov_yaw_left` sets index 5 to `+1.0`, and `rov_yaw_right` sets index 5 to `-1.0`. Bound `HOME`, `END`, `PAGE_UP`, and `PAGE_DOWN` to those actions. Added the same four rows to the device control display table. Confirmed the Python carb enum member names that `event.input.name` uses are `HOME`, `END`, `PAGE_UP`, and `PAGE_DOWN`; the C++ display-name table separately lists `Home`, `End`, `PageUp`, and `PageDown`, but the installed Python enum `.name` values are the uppercase/underscore strings used in the mapping.
- **Self-checks:** `.venv-rov_lab/bin/python -m py_compile rov_lab/source/rov_lab/devices/keyboard/rov_keyboard.py` passed with no output. `rg -n "rov_up|rov_down|rov_yaw_left|rov_yaw_right|HOME|END|PAGE_UP|PAGE_DOWN" rov_lab/source/rov_lab/devices/keyboard/rov_keyboard.py` showed all four control-table rows at lines 42-45, all four action vectors at lines 137-148, and all four key bindings at lines 193-196. Key-name confirmation command: `LD_LIBRARY_PATH=/home/home/rov_il/.venv-rov_lab/lib/python3.11/site-packages/isaacsim/kit:/home/home/rov_il/.venv-rov_lab/lib/python3.11/site-packages/isaacsim/kit/kernel/plugins .venv-rov_lab/bin/python - <<'PY' ...` printed `KeyboardInput.HOME HOME 61`, `KeyboardInput.END END 62`, `KeyboardInput.PAGE_UP PAGE_UP 59`, and `KeyboardInput.PAGE_DOWN PAGE_DOWN 60`. Static source cross-check: `rg -n "(eHome|eEnd|ePageUp|ePageDown|HOME|END|PAGE_UP|PAGE_DOWN|PageUp|PageDown)" .venv-rov_lab/lib/python3.11/site-packages/isaacsim/kit/kernel/py/carb/input.pyi .venv-rov_lab/lib/python3.11/site-packages/isaacsim/kit/dev/include/carb/input/InputTypes.h .venv-rov_lab/lib/python3.11/site-packages/isaacsim/kit/dev/include/carb/input/detail/KeyboardInputName.h` found Python enum members `PAGE_UP`, `PAGE_DOWN`, `HOME`, `END` and C++ enum/display-name entries.
- **Out of scope (left untouched):** No simulator task, recorder, template, dataset converter, vendored dependency, generated data, or checkpoint files were changed. Sign convention kept as requested: `+index2` is up and `+index5` is yaw-left; direction feel can be adjusted later if operator testing finds it inverted.
- **Questions / blockers:** none

---

## Review (written by main agent)

- **CMD-008:** APPROVED. Verified in source: `rov_up` (idx2 +1), `rov_down` (idx2 −1),
  `rov_yaw_left` (idx5 +1), `rov_yaw_right` (idx5 −1), all × `rov_sensitivity`; HOME/END/
  PAGE_UP/PAGE_DOWN bound correctly; four control-table rows added next to the arrow ROV rows.
  Good that Codex empirically confirmed the carb `event.input.name` strings (HOME=61, END=62,
  PAGE_UP=59, PAGE_DOWN=60) rather than guessing — a wrong name would be a silent no-op.
  py_compile passes. Direction signs (Home=up, PageUp=yaw-left) are defaults to confirm in-sim;
  flip if inverted.

- **RUNTIME FIX #2 (manager load before reset):** `gym.make` → `ObservationManager._prepare_terms`
  probes each obs term once at load, BEFORE any reset, so `env.pick_stone_target_color_id` (set
  by the reset event) didn't exist yet → `AttributeError` in `target_cube_grasped`. Fixed by a
  read helper `observations.resolve_target_color_id(env)` that returns a zeros `[num_envs]` long
  tensor when the attr is absent (shape-probe only; real values set on reset); used by both
  `target_cube_grasped` (obs) and `target_cube_grasped_and_lifted` (termination). The reset event
  keeps lazily creating/persisting the real tensor; the recorder already guarded with
  `getattr(..., None)`. py_compile passes; no unguarded reads remain.

- **RUNTIME FIX (first Isaac Sim launch):** `parse_env_cfg` raised
  `AttributeError: module 'leisaac.enhance.envs.mdp.recorders' has no attribute
  'PreStepTargetColorRecorder'`. Root cause: `pick_stone/mdp/__init__.py` does
  `from leisaac.enhance.envs.mdp import *`, which binds a `recorders` name into the mdp package
  namespace and shadowed the sibling `mdp/recorders.py`, so `recorders_cfg.py`'s
  `from . import recorders` resolved to leisaac's module. Fixed by importing the class directly:
  `from .recorders import PreStepTargetColorRecorder` (submodule-path import is immune to the
  namespace shadowing). A static-only gap — neither py_compile nor the audit caught it because
  it depends on the leisaac star-import populating the namespace at runtime. py_compile passes.

- **CMD-007:** APPROVED — pipeline complete. My own Phase 7 sweep (all rov_lab files
  py_compile, 0 stale refs, `mdp/__init__.py` exports events/observations/terminations,
  9 converter tests pass) plus Codex's fresh-eyes end-to-end audit both come back clean: the
  per-episode color→instruction contract holds on the offline route (event→recorder
  `task/target_color_id` [num_envs,1]→HDF5 [T,1]→converter→LeRobot `task`) and the in-recording
  route (build_lerobot_frame via `task_instructions`), obs/success gather by the same target id,
  and the three instruction sources agree. 6b teleop instruction print verified (generic, no-op
  for other tasks). CAVEAT: rov_lab simulator code is statically verified only (no Isaac
  runtime here) — runtime behavior (cube spawn/physics, grasp thresholds, lift margin) must be
  validated in the user's Isaac Sim environment.

- **CMD-006:** APPROVED + main-agent follow-up applied. Read Codex's template
  `build_lerobot_frame` change: correct and task-agnostic — `getattr(self,
  "task_instructions", None)`, safe `_data.get("task", {}).get("target_color_id")`, truthiness
  guard, robust `int(...[-1].reshape(-1)[0].item())`, bounds check, `task_description` fallback,
  rest unchanged. Part 1 investigation (the `--use-lerobot-recorder` path runs recorder terms
  so `_data["task"]` is populated at frame-build time; traced
  `manager_based_rl_env.py:175/213` + `lerobot_recorder_manager.py:56-64`) is a useful,
  verified finding. Part 2 recorder review = no blocker, matching my own read of the
  RecorderTerm contract. ACCEPTED Codex's activation-gap recommendation: I (main) added
  `task_instructions: tuple[str, ...] = TARGET_INSTRUCTIONS` to `PickStoneEnvCfg` (+ import) so
  the in-recording route emits color-specific tasks; py_compile passes. Phases 3 & 6 complete.

- **CMD-005:** APPROVED. Verified: `template/__init__.py` now imports `SingleArmEventCfg` and
  lists it in `__all__`, so the pick_stone `from ..template import SingleArmEventCfg` resolves
  (blocker fixed). `pick_stone_env_cfg.py` now builds a fresh `SceneEntityCfg(...)` per term and
  a `_cube_cfgs()` helper returns fresh instances per call — no shared module-level term cfgs
  left. `DEFAULT_TASK_DESCRIPTION` updated to the cube/lift fallback string. 9 tests pass.
  Phases 4–5 are complete and the pick_stone task module should now import cleanly.

- **CMD-004:** CHANGES REQUESTED. Part A config decoupling is correct (import-time
  cross-package load removed; `DEFAULT_INSTRUCTION_TEMPLATES` defined with sync comment;
  9 tests still pass). The new MDP logic is correct on inspection: `reset_target_color`
  lazily creates `env.pick_stone_target_color_id`; `target_cube_grasped` stacks `[N, 3]` and
  gathers by target id; `target_cube_grasped_and_lifted` uses each cube's
  `default_root_state[:, 2]` as the lift baseline (world-frame, origin cancels) with the TODO
  note; cubes/colors/props and the EventCfg ordering (reset_all before reset_target) are good;
  dead code + broken Mimic registration removed.
  BLOCKER: `pick_stone_env_cfg.py` does `from ..template import (... SingleArmEventCfg ...)`,
  but `template/__init__.py` does NOT export `SingleArmEventCfg` → runtime `ImportError` when
  the task loads. `py_compile` cannot catch this (it does not resolve imports), which is why
  it slipped through. MINOR: module-level `RED/GREEN/BLUE_CUBE_CFG` / `CUBE_CFGS`
  `SceneEntityCfg` instances are shared across multiple observation and termination terms;
  each term `.resolve()`s its cfg, so fresh per-term instances (the leisaac idiom) are safer.
  MINOR: `config.DEFAULT_TASK_DESCRIPTION` still reads "Pick three stones … plate …" (stale
  fallback string). → CMD-005 fixes these.

- **CMD-003:** APPROVED WITH ONE FIX. Read every changed file. `task_metadata.py` (pure
  Python, validated helpers), `hdf5_reader.read_constant_int` (empty/non-constant raise),
  `converter._resolve_episode_task` (per-episode task; KeyError→fallback or hard error;
  ValueError on non-constant/invalid id propagates), `cli.py` (`--task` is fallback-only,
  `--target-color-key`/`--fixed-task-fallback` added), and the tests (synthetic HDF5 + stub
  dataset, no real lerobot needed; covers red/blue per-episode tasks, missing-metadata raise,
  and fallback) are all correct. 9 tests pass.
  ISSUE: `config.py` loads `rov_lab/.../task_metadata.py` via a file-path `importlib` call at
  **module import time** (`_TASK_METADATA = _load_task_metadata()`). This couples rov_lerobot
  to rov_lab's source layout and does disk I/O + `exec` on every converter import; it breaks
  if rov_lerobot is imported without the rov_lab tree present. FIX to apply next: define
  `DEFAULT_INSTRUCTION_TEMPLATES` directly in `config.py` (the 3 strings) with a comment to
  keep it in sync with `task_metadata.py::TARGET_INSTRUCTIONS`, and remove the import-time
  cross-package file load. (A proper shared `rov_common` contract is the long-term option —
  noted as future work, out of scope now.)

- **CMD-002:** APPROVED. Plan is concrete and technically grounded. Verified the two
  load-bearing claims against source: the streaming HDF5 writer is tensor-only
  (`leisaac/.../hdf5_dataset_file_handler.py:64-69` → `value.cpu().numpy()` unless dict),
  so the numeric `target_color_id` design is required and correct; and
  `RecorderTerm.record_pre_step() -> tuple[str, tensor|dict]`
  (`IsaacLab/.../recorder_manager.py:112`) matches the proposed recorder term, so the
  `episode_data._data["task"]` path is valid. Adopting the plan with two refinements:
  (a) fold the dead-code/broken-state cleanup (remove `dummy_obs`/`dummy_done`, plate-based
  `put_stone_to_plate`/`task_done`, stale `Orange001` defaults, and the broken
  `ROVLab-SO101-PickStone-Mimic-v0` registration) into the implementation phases instead of
  deferring; (b) implement phase-by-phase via successive commands, starting with the
  pure-Python, fully unit-testable Phases 1–2 (task-metadata mapping + converter) before the
  simulator-only rov_lab changes.

---

## History (managed by main agent)

- (empty)
