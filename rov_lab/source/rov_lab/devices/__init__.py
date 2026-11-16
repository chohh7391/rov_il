"""Teleoperation devices for ROV Lab.

Device classes need the Omniverse app (``carb``/``omni``), so they are imported lazily: task configurations can
import ``rov_lab.devices.action_process`` without a running simulator (Isaac Lab 3.0 builds configs before launch).
"""

from __future__ import annotations

from typing import Any

__all__ = ["ROVKeyboard"]


def __getattr__(name: str) -> Any:
    if name == "ROVKeyboard":
        from .keyboard import ROVKeyboard

        return ROVKeyboard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
