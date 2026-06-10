# SUBAGENTS.md — Sub Agent Work Orders

This document is the **command sheet the main agent issues to the sub agent**.
The sub agent reads this file, performs *only* the work in the `## Current Command`
block below, reports the result to `docs/STATUS.md`, and exits.

---

## 1. Role and Scope

- You are the **execution sub agent**. Design ownership and final review are the main
  agent's job.
- Your task is to carry out the work in `## Current Command` **exactly, and only within
  that scope**.
- Do **not** perform refactors, feature additions, or file creation that the command
  does not ask for. If the scope is ambiguous, do not guess and expand — write the
  question under `Questions / blockers` in STATUS.md and stop.
- This is a feedback loop: the main agent reviews your report, may send corrections, and
  only adopts your work when satisfied. Optimize for a clear, reviewable deliverable.

## 2. Fixed Rules (from the project `AGENTS.md`)

1. **Project-owned code first:** Make changes inside `rov_lab/`, `rov_lerobot/`, or
   `rov_sim/`. Touch `external_dependencies/` only when the command explicitly requires it.
2. **Typed Python:** Add type hints to new function signatures, class attributes,
   dataclasses, and non-trivial variables (also in code snippets inside a plan).
3. **Match local style:** Follow the existing naming, comment density, and idioms of the
   file/directory you edit. Do not run broad auto-formatters over unrelated files.
4. **No large artifacts:** Do not commit or generate HDF5 datasets, checkpoints, videos,
   or logs. Keep raw HDF5 recordings immutable.
5. **Search with `rg`.**
6. **Assume the simulator cannot run:** This environment has no Isaac Sim / Isaac Lab
   runtime. Code that imports `isaaclab` **cannot be verified by running it**. Verify with
   `python -m py_compile` (syntax) plus static consistency review. If you attempt a runtime
   import and it fails because the simulator is missing, do **not** report that as a
   failure — judge correctness from syntax and consistency instead.
   Because `py_compile` does NOT resolve imports, whenever you add a new `import` or
   `from ... import X`, statically confirm the symbol is actually defined/exported at the
   target (e.g. `rg` the target module and its `__all__`). A missing export is a real
   runtime `ImportError` that `py_compile` will never catch.

## 3. Work Contract (follow every time)

1. Read this whole document and the `## Current Command`.
2. Read the files named in the command **and the surrounding code** first to build context.
3. Do the work allowed by the command's **Type** (see below), only within scope.
4. Run whatever self-checks are possible (`python -m py_compile <changed files>`,
   `rg` for references, etc.).
5. Overwrite the `## Latest Report` section of `docs/STATUS.md` using the format in
   section 4. You may overwrite the previous report directly; the main agent manages
   history.
6. Exit. Do not start another cycle on your own.

### Command Types

Each command declares a **Type**:

- **Planning** — The deliverable is a plan/design document at the path named in the command.
  **Do not modify any source** under `rov_lab/`, `rov_lerobot/`, or `rov_sim/`. Read code,
  investigate the real data flow, and write the plan document only. Code may appear inside
  the plan as proposed snippets, but nothing is applied to source.
- **Implementation** — Modify source per the command's scope, following section 2.

## 4. STATUS.md Report Format

Fill the `## Latest Report` section of `docs/STATUS.md` with:

```
## Latest Report

- **Command ID:** (the ID from the current command block)
- **Status:** done / partial / blocked
- **Deliverable:** for Planning, the plan doc path; for Implementation, the files changed
- **What was done:** 3–8 line summary of the core result
- **Self-checks:** commands run and their results (e.g. `py_compile` passed / `rg` hits)
- **Out of scope (left untouched):** what you deliberately did not change
- **Questions / blockers:** "none" if there are none
```

---

## Current Command

- **Command ID:** CMD-008
- **Type:** Implementation
- **Title:** Add ROV heave (up/down) and yaw (turn left/right) to the keyboard teleop.

Currently `ROVKeyboard` (`rov_lab/source/rov_lab/devices/keyboard/rov_keyboard.py`) only maps
ROV surge (index 0, UP/DOWN arrows) and sway (index 1, LEFT/RIGHT arrows). The 14-D action's
ROV block is `_delta_action[:6] = [vx, vy, vz, wx, wy, wz]`, so **index 2 (vz = heave)** and
**index 5 (wz = yaw)** are currently unmapped. Add keyboard control for both.

Edit ONLY `rov_lab/source/rov_lab/devices/keyboard/rov_keyboard.py`:

1. In `_ACTION_DELTA_MAPPING`, add four entries (all scaled by `self.rov_sensitivity`, matching
   the existing `rov_*` entries; 14-element vectors):
   - `"rov_up"`:   index 2 = **+1.0**  (the rest 0)
   - `"rov_down"`: index 2 = **-1.0**
   - `"rov_yaw_left"`:  index 5 = **+1.0**
   - `"rov_yaw_right"`: index 5 = **-1.0**

2. In `_INPUT_KEY_MAPPING`, bind:
   - `"HOME"` → `"rov_up"`, `"END"` → `"rov_down"`
   - `"PAGE_UP"` → `"rov_yaw_left"`, `"PAGE_DOWN"` → `"rov_yaw_right"`

   IMPORTANT (rule 6 spirit — verify, don't guess): the dict keys must match the exact
   `carb.input.KeyboardInput` member name strings that `event.input.name` produces. Confirm the
   exact spelling for Home/End/PageUp/PageDown (likely `"HOME"`, `"END"`, `"PAGE_UP"`,
   `"PAGE_DOWN"`) against the carb enum / isaaclab keyboard device usage before finalizing — a
   wrong name is a SILENT no-op key, not an error. If a name differs, use the correct one and
   note it in the report.

3. In `_add_device_control_description`, add four `self._display_controls_table.add_row([...])`
   rows so the on-screen controls list documents the new keys, e.g.
   `["HOME", "rov_up"]`, `["END", "rov_down"]`, `["PAGE_UP", "rov_yaw_left"]`,
   `["PAGE_DOWN", "rov_yaw_right"]`. Match the placement/style of the existing ROV rows.

### Constraints / notes
- Typed/style-consistent with the existing entries (copy the exact `np.asarray([...]) *
  self.rov_sensitivity` pattern). No other behavior change. Only this one file.
- Sign convention: +index2 = up, +index5 = yaw-left are the intended defaults; the operator
  can report if a direction feels inverted and we flip the sign later — note this in the report.

### Verification (run and report)
- `.venv-rov_lab/bin/python -m py_compile rov_lab/source/rov_lab/devices/keyboard/rov_keyboard.py`
- `rg -n "rov_up|rov_down|rov_yaw_left|rov_yaw_right|HOME|END|PAGE_UP|PAGE_DOWN" rov_lab/source/rov_lab/devices/keyboard/rov_keyboard.py`
  to confirm all four actions, four key bindings, and four control-table rows are present.
- State the carb key-name confirmation you did (Part 2 of step 2).
- Report to STATUS.md per section 4.

This command has TWO parts. The work is split with the main agent to avoid file overlap:
the **main agent has already implemented Phase 3 (the recorder)**; you implement Phase 6 in
the **template only**, and you review the main agent's recorder.

These rov_lab files import `isaaclab` and cannot run here — verify with `python -m py_compile`
plus static review, and (per rule 6) statically confirm any new import is actually exported.

### Part 1 — Implement Phase 6 (template only)
Context: the main agent added a recorder term that writes the per-episode target color id to
HDF5 under `task/target_color_id` (key `"task/target_color_id"`, value `[num_envs, 1]` long,
recorded every step). In `EpisodeData` this lands as
`episode_data._data["task"]["target_color_id"]` — a list of per-step `[1]` long tensors.

The offline converter route is already complete (CMD-003). Phase 6 is the SECONDARY
in-recording LeRobot route: `ROVSingleArmTaskEnvCfg.build_lerobot_frame(...)` in
`rov_lab/source/rov_lab/tasks/template/rov_single_arm_env_cfg.py` currently hardcodes
`"task": self.task_description`.

Do this, **editing only the template file**:
1. First INVESTIGATE: when `teleop_se3_agent.py --use-lerobot-recorder` is used, does the
   LeRobot recorder path actually run the env's configured `recorders` terms so that
   `self._episodes[0]._data["task"]["target_color_id"]` is populated at
   `build_lerobot_frame` time? Trace `LeRobotRecorderManager` / the leisaac handler. Report
   what you find. If it is NOT populated in that path, do not fake it — keep the
   `self.task_description` fallback and clearly document the limitation in your report.
2. Make `build_lerobot_frame` resolve the per-episode task string generically, WITHOUT
   hardcoding pick-stone specifics (the template must stay task-agnostic):
   - Read an OPTIONAL instruction source via `getattr(self, "task_instructions", None)`
     (a `tuple[str, ...] | None`).
   - If `task_instructions` is set AND `episode_data._data` contains
     `["task"]["target_color_id"]`, take the last recorded id
     (`int(episode_data._data["task"]["target_color_id"][-1].reshape(-1)[0].item())`),
     bounds-check it against `task_instructions`, and use `task_instructions[id]` as `"task"`.
   - Otherwise fall back to `self.task_description`.
   - Keep all other `build_lerobot_frame` behavior (action_align, image keys) unchanged.
3. Do NOT edit `pick_stone_env_cfg.py` or any pick_stone file. If wiring on the pick-stone
   side is required to activate this (e.g. a `task_instructions` attribute on
   `PickStoneEnvCfg`), SPECIFY exactly what the main agent should add in your report's
   `Questions / blockers` — do not add it yourself (that file is the main agent's).

### Part 2 — Review the main agent's Phase 3 recorder
Read and critique these main-agent-authored files; do NOT modify them:
- `rov_lab/source/rov_lab/tasks/pick_stone/mdp/recorders.py`
  (`PreStepTargetColorRecorder`)
- `rov_lab/source/rov_lab/tasks/pick_stone/mdp/recorders_cfg.py`
  (`PreStepTargetColorRecorderCfg`, `PickStoneRecorderManagerCfg`)
- the `recorders` wiring in `rov_lab/source/rov_lab/tasks/pick_stone/pick_stone_env_cfg.py`

Check for real correctness/robustness issues only (not style nits): the `RecorderTerm` /
`record_pre_step` contract and return shape vs how `add_to_episodes` indexes `value[env_id]`
and how the streaming HDF5 handler writes it; the `RecorderTermCfg.class_type` pattern; the
manager subclassing of `ActionStateRecorderManagerCfg`; the guard when
`pick_stone_target_color_id` is absent; and whether the resulting HDF5 key matches the
converter's `target_color_key="task/target_color_id"`. Put concrete findings (with file:line
and a suggested fix) in your report so the main agent can evaluate them.

### Verification (run and report)
- `python -m py_compile` on the template file you changed.
- Static import check on any new import you add (rule 6).
- Report to STATUS.md per section 4. Use `What was done` for Part 1, and put your Part 2
  review findings under a clear sub-list in `What was done` (or `Questions / blockers` for the
  pick-stone wiring you recommend). Include the Part 1 investigation conclusion.
