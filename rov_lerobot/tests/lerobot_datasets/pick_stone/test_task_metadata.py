from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


def _load_task_metadata():
    repo_root = Path(__file__).resolve().parents[4]
    metadata_path = repo_root / "rov_lab/source/rov_lab/tasks/pick_stone/mdp/task_metadata.py"
    spec = spec_from_file_location("rov_pick_stone_task_metadata_test", metadata_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_target_id_mappings() -> None:
    metadata = _load_task_metadata()

    assert metadata.color_name_from_target_id(0) == "red"
    assert metadata.color_name_from_target_id(1) == "green"
    assert metadata.color_name_from_target_id(2) == "blue"
    assert metadata.instruction_from_target_id(0) == "pick up the red cube"
    assert metadata.instruction_from_target_id(2) == "pick up the blue cube"
    assert metadata.cube_asset_name_from_target_id(1) == "green_cube"


@pytest.mark.parametrize("target_color_id", [-1, 3])
def test_target_id_out_of_range_raises(target_color_id: int) -> None:
    metadata = _load_task_metadata()

    with pytest.raises(ValueError, match="target_color_id must be in"):
        metadata.instruction_from_target_id(target_color_id)
