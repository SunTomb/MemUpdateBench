from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from mub.vnext.contracts import (
    ActionScope,
    AnswerDisposition,
    Operation,
    ReferenceResolutionStatus,
)
from mub.vnext.contracts.task import GoldAction
from mub.vnext.generation import family_c, family_d
from mub.vnext.generation.config import load_pilot_config
from mub.vnext.generation.core_config import load_core_config
from mub.vnext.validation.replay import replay_actions


ROOT = Path(__file__).resolve().parents[2]
CORE_CONFIG_PATH = ROOT / "configs" / "vnext" / "core.yaml"
PILOT_CONFIG_PATH = ROOT / "configs" / "vnext" / "pilot.yaml"
CORE_CELLS = {
    (entity_condition, attribute_condition)
    for entity_condition in ("distinct", "same_name", "alias", "namespace_collision")
    for attribute_condition in ("exact", "paraphrase", "near_name")
}
CORE_TRAPS = (
    "transient",
    "hypothetical",
    "negated",
    "uncertain",
    "semantic_near_miss",
    "duplicate_current",
    "unsupported_inference",
)
CORE_DENSITIES = (0.25, 0.50, 0.75)


def _json_digest(cores) -> str:
    payload = [core.model_dump(mode="json") for core in cores]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _actions(events):
    return [
        GoldAction(
            action_id=f"a-{index}",
            event_id=f"e-{index}",
            operation=event.operation,
            scope=(
                ActionScope.OBJECT
                if event.operation is Operation.NOOP
                else ActionScope.ATTRIBUTE
            ),
            target_object_keys=list(event.object_keys),
            value=event.value,
        )
        for index, event in enumerate(events)
    ]


def _replay(events):
    return replay_actions(_actions(events))


def test_core_family_c_has_exact_balanced_grid_and_resolution_totals():
    config = load_core_config(CORE_CONFIG_PATH)
    cores = family_c.generate_core_family_c_cores(config)

    cells = Counter(
        (
            core.stratification["entity_condition"],
            core.stratification["attribute_condition"],
        )
        for core in cores
    )
    statuses = Counter(core.canonical_answer.resolution_status for core in cores)

    assert len(cores) == 420
    assert [core.core_index for core in cores] == list(range(420))
    assert cells == Counter({cell: 35 for cell in CORE_CELLS})
    assert statuses == Counter(
        {
            ReferenceResolutionStatus.UNIQUE: 140,
            ReferenceResolutionStatus.AMBIGUOUS: 140,
            ReferenceResolutionStatus.NO_MATCH: 140,
        }
    )
    assert len({core.core_id for core in cores}) == 420
    assert len({core.trajectory_id for core in cores}) == 420


def test_core_family_c_configured_pairwise_marginals_have_imbalance_at_most_one():
    config = load_core_config(CORE_CONFIG_PATH)
    cores = family_c.generate_core_family_c_cores(config)
    cells = Counter(
        (
            core.stratification["entity_condition"],
            core.stratification["attribute_condition"],
        )
        for core in cores
    )

    family = config.families.entity_attribute_grounding
    for entity_condition in family.entity_conditions:
        marginal = [
            cells[(entity_condition, attribute_condition)]
            for attribute_condition in family.attribute_conditions
        ]
        assert max(marginal) - min(marginal) <= 1
    for attribute_condition in family.attribute_conditions:
        marginal = [
            cells[(entity_condition, attribute_condition)]
            for entity_condition in family.entity_conditions
        ]
        assert max(marginal) - min(marginal) <= 1


def test_core_family_c_uses_typed_answered_and_valid_abstained_dispositions():
    config = load_core_config(CORE_CONFIG_PATH)
    cores = family_c.generate_core_family_c_cores(config)

    for core in cores:
        canonical = core.canonical_answer
        assert canonical is not None
        assert canonical.disposition is not None
        assert canonical.resolution_status is not None
        if canonical.resolution_status is ReferenceResolutionStatus.UNIQUE:
            assert canonical.disposition is AnswerDisposition.ANSWERED
            assert len(canonical.selected_candidate_ids) == 1
            assert isinstance(canonical.value, str) and canonical.value
            assert canonical.abstention_reason is None
        else:
            assert canonical.disposition is AnswerDisposition.ABSTAINED
            assert canonical.selected_candidate_ids == ()
            assert canonical.value is None
            assert canonical.abstention_reason
            assert canonical.abstention_reason.casefold() not in {
                "null",
                "none",
                "unknown",
                "guessed",
            }


def test_core_family_c_is_deterministic_and_does_not_mutate_config():
    config = load_core_config(CORE_CONFIG_PATH)
    before = config.model_dump(mode="json")
    first = family_c.generate_core_family_c_cores(config)
    second = family_c.generate_core_family_c_cores(config)

    assert [core.model_dump(mode="json") for core in first] == [
        core.model_dump(mode="json") for core in second
    ]
    assert config.model_dump(mode="json") == before


def test_core_family_d_is_exact_seven_by_three_by_twenty_cartesian_product():
    config = load_core_config(CORE_CONFIG_PATH)
    cores = family_d.generate_core_family_d_cores(config)
    cells = Counter(
        (
            core.stratification["trap_type"],
            core.stratification["configured_noop_density"],
        )
        for core in cores
    )

    assert len(cores) == 420
    assert [core.core_index for core in cores] == list(range(420))
    assert cells == Counter(
        {(trap, density): 20 for trap in CORE_TRAPS for density in CORE_DENSITIES}
    )
    assert len({core.core_id for core in cores}) == 420
    assert len({core.trajectory_id for core in cores}) == 420


def test_core_family_d_traps_are_reviewed_semantic_noops_with_canonical_identity():
    config = load_core_config(CORE_CONFIG_PATH)
    cores = family_d.generate_core_family_d_cores(config)

    for core in cores:
        trap_type = core.stratification["trap_type"]
        traps = [
            event for event in core.events if event.metadata.get("trap_type") == trap_type
        ]
        assert len(traps) == 1
        trap = traps[0]
        target = core.query_targets[0]
        assert trap.operation is Operation.NOOP
        assert trap.object_keys == ()
        assert trap.value is None
        assert trap.metadata["semantic_effect"] == "noop"
        assert trap.metadata["review_status"] == "reviewed"
        assert trap.metadata["wording_style"] == "deterministic_reviewed_v1"
        assert trap.metadata["referenced_object_identity"] == {
            "namespace": target.namespace,
            "entity": target.entity,
            "attribute": target.attribute,
            "subkey": target.subkey,
        }
        statement = trap.metadata["surface_statement"]
        assert isinstance(statement, str) and statement.strip()
        assert target.entity.replace("_", " ") in statement
        assert target.attribute.replace("_", " ") in statement


def test_core_family_d_every_trap_is_replay_proven_non_mutation():
    config = load_core_config(CORE_CONFIG_PATH)
    cores = family_d.generate_core_family_d_cores(config)

    for core in cores:
        trap_type = core.stratification["trap_type"]
        trap_index = next(
            index
            for index, event in enumerate(core.events)
            if event.metadata.get("trap_type") == trap_type
        )
        before = _replay(core.events[:trap_index])
        after = _replay(core.events[: trap_index + 1])
        assert after.mutation_count == before.mutation_count
        assert dict(after.final_state) == dict(before.final_state)
        assert dict(after.version_history) == dict(before.version_history)
        assert after.final_state[core.query_targets[0].canonical_id] == core.expected_answer


def test_core_family_d_density_counts_are_exact_and_generation_is_deterministic():
    config = load_core_config(CORE_CONFIG_PATH)
    before = config.model_dump(mode="json")
    first = family_d.generate_core_family_d_cores(config)
    second = family_d.generate_core_family_d_cores(config)

    for core in first:
        noop_count = sum(event.operation is Operation.NOOP for event in core.events)
        assert len(core.events) == 12
        assert noop_count == int(
            len(core.events) * core.stratification["configured_noop_density"]
        )
        assert core.stratification["noop_count"] == noop_count
    assert [core.model_dump(mode="json") for core in first] == [
        core.model_dump(mode="json") for core in second
    ]
    assert config.model_dump(mode="json") == before


def test_pilot_family_c_and_d_outputs_remain_byte_stable():
    config = load_pilot_config(PILOT_CONFIG_PATH)

    assert _json_digest(family_c.generate_family_c_cores(config)) == (
        "0b84352dd254663f8525bc050100942e5836ea6d9b481bdf3abe4758aba41a14"
    )
    assert _json_digest(family_d.generate_family_d_cores(config)) == (
        "5baaa1853db2a1928f25660f6172b4ad1ccea8309c0a0a7dc47d90db3d1de5fc"
    )
