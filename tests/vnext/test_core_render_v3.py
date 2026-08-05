from __future__ import annotations

from pathlib import Path

from mub.vnext.contracts import Split
from mub.vnext.contracts.v3.common import object_identity
from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.generation import render_core_v3
from mub.vnext.generation.core import GenerationContext
from mub.vnext.generation.core_config import load_core_config
from mub.vnext.generation.family_a import generate_core_family_a_cores
from mub.vnext.io import semantic_task_hash_v3
from mub.vnext.validation.replay_v3 import replay_task_v3


ROOT = Path(__file__).resolve().parents[2]
CORE_CONFIG_PATH = ROOT / "configs" / "vnext" / "core.yaml"


def test_render_core_v3_promotes_one_a_core_across_all_four_surfaces() -> None:
    config = load_core_config(CORE_CONFIG_PATH)
    context = GenerationContext(config=config, code_revision="task-636-render-v3")
    core = generate_core_family_a_cores(config)[0]

    tasks = tuple(
        render_core_v3(
            core,
            split=Split.TRAIN,
            surface_variant=surface_variant,
            context=context,
        )
        for surface_variant in range(4)
    )

    assert all(isinstance(task, MemUpdateTaskV3) for task in tasks)
    assert {task.schema_version for task in tasks} == {"3.0.0"}
    assert len({task.task_id for task in tasks}) == 4
    assert len({task.source.raw_hash for task in tasks}) == 4
    assert len({semantic_task_hash_v3(task) for task in tasks}) == 1
    assert all(not replay_task_v3(task).issues for task in tasks)
    assert all(
        tuple(object_identity(key) for key in task.target_objects)
        == tuple(object_identity(key) for key in tasks[0].target_objects)
        for task in tasks
    )
