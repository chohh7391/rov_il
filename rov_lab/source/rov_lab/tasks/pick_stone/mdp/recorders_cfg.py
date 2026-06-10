"""Recorder manager configuration for the colored pick-stone task."""

from __future__ import annotations

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

# Import the class directly (not `from . import recorders`): the task's `mdp/__init__.py` does
# `from leisaac.enhance.envs.mdp import *`, which binds a `recorders` name into the mdp package
# namespace and would otherwise shadow this sibling module.
from .recorders import PreStepTargetColorRecorder


@configclass
class PreStepTargetColorRecorderCfg(RecorderTermCfg):
    """Configuration for the target color recorder term."""

    class_type: type[RecorderTerm] = PreStepTargetColorRecorder


@configclass
class PickStoneRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Action/state recorders plus the per-episode target color metadata recorder."""

    record_pre_step_target_color = PreStepTargetColorRecorderCfg()
