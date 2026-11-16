# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025, LeIsaac Project (Apache-2.0); adapted for rov_lab.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard-driven teleoperation device base for rov_lab.

Ported from ``leisaac.devices.device_base.Device`` so rov_lab no longer depends on leisaac.
The base handles the session keys shared by every rov_lab device:

* ``B``: start control
* ``R``: reset and mark the episode as failed
* ``N``: reset and mark the episode as successful

Subclasses implement :meth:`get_device_state` and :meth:`_add_device_control_description`.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import Any

import carb
import numpy as np
import omni
import torch
from prettytable import PrettyTable


class Device:
    """Base class for rov_lab teleoperation devices that listen to the Omniverse keyboard."""

    def __init__(self, env: Any, device_type: str) -> None:
        self.env = env
        self.device_type: str = device_type

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # weakref so the device can be garbage-collected while the subscription is alive
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

        self._started: bool = False
        self._reset_state: bool = False
        self._additional_callbacks: dict[str, Callable[[], None]] = {}

        self._display_controls_table = PrettyTable()
        self._display_controls_table.title = f"Teleoperation Controls for {self.device_type}"
        self._display_controls_table.field_names = ["Key", "Description"]
        self._display_controls_table.align["Description"] = "l"
        self._display_controls_table.add_row(["B", "start control"])
        self._display_controls_table.add_row(["R", "reset simulation and set task success to False"])
        self._display_controls_table.add_row(["N", "reset simulation and set task success to True"])
        self._display_controls_table.add_row(["Control+C", "quit"])
        self._display_controls_table.add_row(["==========", "=========="])
        self._add_device_control_description()

    def __del__(self) -> None:
        self._stop_keyboard_listener()

    # ------------------------------------------------------------------ subclass API
    def get_device_state(self) -> np.ndarray | dict[str, Any]:
        raise NotImplementedError

    def _add_device_control_description(self) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------ public API
    def input2action(self) -> dict[str, Any]:
        """Return the raw device state plus session flags."""
        if self._reset_state:
            self._reset_state = False
            return {"reset": True, "started": self._started}
        return {
            "reset": False,
            "started": self._started,
            self.device_type: True,
            "joint_state": self.get_device_state(),
        }

    def advance(self) -> torch.Tensor | dict[str, Any] | None:
        """Return the action tensor, a reset request dict, or ``None`` while control has not started."""
        action = self.input2action()
        if not action["started"]:
            return None
        if action["reset"]:
            return action
        for key, value in action.items():
            if isinstance(value, np.ndarray):
                action[key] = torch.tensor(value, device=self.env.device, dtype=torch.float32)
        return self.env.cfg.preprocess_device_action(action, self)

    def reset(self) -> None:
        pass

    def add_callback(self, key: str, func: Callable[[], None]) -> None:
        self._additional_callbacks[key] = func

    def display_controls(self) -> None:
        print(self._display_controls_table)

    @property
    def started(self) -> bool:
        return self._started

    @property
    def reset_state(self) -> bool:
        return self._reset_state

    @reset_state.setter
    def reset_state(self, reset_state: bool) -> None:
        self._reset_state = reset_state

    # ------------------------------------------------------------------ keyboard handling
    def _on_keyboard_event(self, event: Any, *args: Any, **kwargs: Any) -> None:
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return
        name = event.input.name
        if name == "B":
            self._started = True
            self._reset_state = False
        elif name in ("R", "N"):
            self._started = False
            self._reset_state = True
            if name in self._additional_callbacks:
                self._additional_callbacks[name]()

    def _stop_keyboard_listener(self) -> None:
        if getattr(self, "_keyboard_sub", None) is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None
