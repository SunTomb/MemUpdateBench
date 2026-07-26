from __future__ import annotations

from decimal import Decimal
import json
import os
import subprocess
import sys
from types import MappingProxyType

import pytest
from pydantic import ValidationError

from mub.vnext.contracts import Difficulty
from mub.vnext.contracts.common import FrozenDict, thaw_json
from mub.vnext.profiles import (
    CANONICAL_PROFILE_DEFAULTS,
    ProfileSpec,
    build_generic_profile,
    challenge_profile,
    easy_profile,
    hard_profile,
    medium_profile,
    resolve_profile,
)


class HostileJsonStr(str):
    __hash__ = None

    def __str__(self):
        raise RuntimeError("hostile string conversion must not be used")

    def strip(self, *args, **kwargs):
        raise RuntimeError("hostile strip must not be used")


EXPECTED_DEFAULTS = {
    "easy": {
        "update_depth": 1,
        "active_object_count": 1,
        "entity_ambiguity": "none",
        "attribute_ambiguity": "none",
        "noop_density": 0.0,
        "cross_slot_interleaving": 0.0,
        "stale_count": 0,
        "context_length": 4,
        "context_order": "chronological",
        "version_metadata": "latest_outdated",
        "query_type": "current_state",
        "source_naturalness": "synthetic_direct",
    },
    "medium": {
        "update_depth": 4,
        "active_object_count": 4,
        "entity_ambiguity": "moderate",
        "attribute_ambiguity": "moderate",
        "noop_density": 0.1,
        "cross_slot_interleaving": 0.5,
        "stale_count": 3,
        "context_length": 16,
        "context_order": "retrieval_score",
        "version_metadata": "event_index",
        "query_type": "current_state",
        "source_naturalness": "mixed_template",
    },
    "hard": {
        "update_depth": 16,
        "active_object_count": 8,
        "entity_ambiguity": "high",
        "attribute_ambiguity": "high",
        "noop_density": 0.25,
        "cross_slot_interleaving": 0.75,
        "stale_count": 15,
        "context_length": 64,
        "context_order": "reverse_chronological",
        "version_metadata": "none",
        "query_type": "current_state",
        "source_naturalness": "semi_natural",
    },
    "challenge": {
        "update_depth": 32,
        "active_object_count": 16,
        "entity_ambiguity": "compositional",
        "attribute_ambiguity": "compositional",
        "noop_density": 0.4,
        "cross_slot_interleaving": 1.0,
        "stale_count": 31,
        "context_length": 128,
        "context_order": "adversarial",
        "version_metadata": "none",
        "query_type": "mixed",
        "source_naturalness": "natural",
    },
}


@pytest.mark.parametrize(
    ("difficulty", "builder"),
    [
        (Difficulty.EASY, easy_profile),
        (Difficulty.MEDIUM, medium_profile),
        (Difficulty.HARD, hard_profile),
        (Difficulty.CHALLENGE, challenge_profile),
    ],
)
def test_all_four_generic_profiles_have_explicit_phase0_defaults(difficulty, builder):
    profile = builder("future_family")

    assert profile == build_generic_profile(difficulty, "future_family")
    assert profile.name == difficulty.value
    assert profile.task_family == "future_family"
    assert thaw_json(profile.parameters) == EXPECTED_DEFAULTS[difficulty.value]
    assert tuple(profile.allowed_overrides) == tuple(EXPECTED_DEFAULTS[difficulty.value])


def test_hard_and_challenge_defaults_are_regression_locked():
    assert thaw_json(CANONICAL_PROFILE_DEFAULTS[Difficulty.HARD.value]) == EXPECTED_DEFAULTS["hard"]
    assert thaw_json(CANONICAL_PROFILE_DEFAULTS[Difficulty.CHALLENGE.value]) == EXPECTED_DEFAULTS["challenge"]


def test_allowed_override_is_resolved_with_labels_and_bucket():
    resolved = resolve_profile(hard_profile("family-a"), {"update_depth": 7, "context_length": 40})

    assert resolved["update_depth"] == 7
    assert resolved["context_length"] == 40
    assert resolved["update_depth_bucket"] == "4-7"
    assert {key: resolved[key] for key in ("task_family", "difficulty", "profile_name", "profile_version")} == {
        "task_family": "family-a",
        "difficulty": "hard",
        "profile_name": "hard",
        "profile_version": "1.0.0",
    }


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"unknown": 1}, "unknown override"),
        ({"stale_count": 1}, "not allowed"),
        ({"task_family": "evil"}, "protected profile label"),
        ({"difficulty": "easy"}, "protected profile label"),
        ({"profile_name": "easy"}, "protected profile label"),
        ({"profile_version": "evil"}, "protected profile label"),
        ({"update_depth_bucket": "1"}, "derived profile label"),
    ],
)
def test_resolver_rejects_unknown_disallowed_and_label_overrides(overrides, match):
    profile = ProfileSpec(
        name="hard",
        version="1.0.0",
        task_family="future",
        difficulty=Difficulty.HARD,
        parameters=EXPECTED_DEFAULTS["hard"],
        allowed_overrides=("update_depth",),
    )
    with pytest.raises(ValueError, match=match):
        resolve_profile(profile, overrides)


def test_malicious_constructed_spec_cannot_enable_label_overrides():
    profile = ProfileSpec.model_construct(
        name="hard",
        version="1.0.0",
        task_family="future",
        difficulty=Difficulty.HARD,
        parameters=FrozenDict(EXPECTED_DEFAULTS["hard"]),
        allowed_overrides=("task_family", "update_depth_bucket"),
    )
    with pytest.raises(ValueError, match="protected profile label"):
        resolve_profile(profile, {"task_family": "evil"})


def test_profile_name_must_always_equal_difficulty_value():
    with pytest.raises(ValidationError, match="profile name must equal difficulty.value"):
        ProfileSpec(
            name="custom",
            version="1.0.0",
            task_family="future",
            difficulty=Difficulty.EASY,
            parameters=EXPECTED_DEFAULTS["easy"],
            allowed_overrides=(),
        )


def test_resolver_revalidates_constructed_custom_profile_name():
    profile = ProfileSpec.model_construct(
        name="custom",
        version="1.0.0",
        task_family="future",
        difficulty=Difficulty.EASY,
        parameters=FrozenDict(EXPECTED_DEFAULTS["easy"]),
        allowed_overrides=(),
    )
    with pytest.raises(ValidationError, match="profile name must equal difficulty.value"):
        resolve_profile(profile, {})


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"parameters": {**EXPECTED_DEFAULTS["easy"], "mystery_axis": 1}}, "unknown parameter"),
        ({"allowed_overrides": ("mystery_axis",)}, "parameter keys"),
        ({"allowed_overrides": ("update_depth", "update_depth")}, "unique"),
        ({"name": "hard"}, "profile name"),
        ({"task_family": "   "}, "task_family"),
        ({"version": ""}, "version"),
    ],
)
def test_profile_spec_rejects_unknown_parameters_bad_overrides_and_blank_labels(changes, match):
    data = {
        "name": "easy",
        "version": "1.0.0",
        "task_family": "future",
        "difficulty": Difficulty.EASY,
        "parameters": EXPECTED_DEFAULTS["easy"],
        "allowed_overrides": ("update_depth",),
    }
    data.update(changes)
    with pytest.raises(ValidationError, match=match):
        ProfileSpec(**data)


@pytest.mark.parametrize(
    "override,match",
    [
        ({"update_depth": True}, "type-compatible"),
        ({"update_depth": 0}, "positive"),
        ({"stale_count": -1}, "nonnegative"),
        ({"active_object_count": 1.0}, "type-compatible"),
        ({"context_length": 0}, "positive"),
        ({"noop_density": float("nan")}, "finite"),
        ({"noop_density": 1.01}, "between 0 and 1"),
        ({"cross_slot_interleaving": -0.1}, "between 0 and 1"),
        ({"context_order": "  "}, "nonblank"),
    ],
)
def test_resolver_rejects_type_and_semantic_bounds(override, match):
    profile = ProfileSpec(
        name="hard",
        version="1.0.0",
        task_family="future",
        difficulty=Difficulty.HARD,
        parameters=EXPECTED_DEFAULTS["hard"],
        allowed_overrides=tuple(EXPECTED_DEFAULTS["hard"]),
    )
    with pytest.raises((TypeError, ValueError), match=match):
        resolve_profile(profile, override)


@pytest.mark.parametrize(
    ("depth", "bucket"),
    [(1, "1"), (2, "2-3"), (3, "2-3"), (4, "4-7"), (7, "4-7"), (8, "8-15"), (15, "8-15"), (16, "16+"), (32, "16+")],
)
def test_update_depth_buckets_are_fixed(depth, bucket):
    assert resolve_profile(easy_profile("future"), {"update_depth": depth})["update_depth_bucket"] == bucket


def test_future_family_specific_registered_axis_is_supported():
    parameters = {**EXPECTED_DEFAULTS["easy"], "reasoning_depth": 2}
    profile = ProfileSpec(
        name="easy",
        version="2.0.0",
        task_family="future_family_v2",
        difficulty=Difficulty.EASY,
        parameters=parameters,
        allowed_overrides=("reasoning_depth",),
    )
    assert resolve_profile(profile, {"reasoning_depth": 3})["reasoning_depth"] == 3


def test_resolution_is_deterministic_for_parameter_and_override_insertion_order():
    forward = dict(EXPECTED_DEFAULTS["medium"])
    reverse = dict(reversed(tuple(forward.items())))
    first = ProfileSpec(
        name="medium",
        version="1.0.0",
        task_family="future",
        difficulty=Difficulty.MEDIUM,
        parameters=forward,
        allowed_overrides=("context_length", "update_depth"),
    )
    second = ProfileSpec(
        name="medium",
        version="1.0.0",
        task_family="future",
        difficulty=Difficulty.MEDIUM,
        parameters=reverse,
        allowed_overrides=("update_depth", "context_length"),
    )

    first_resolved = resolve_profile(first, {"context_length": 20, "update_depth": 3})
    second_resolved = resolve_profile(second, {"update_depth": 3, "context_length": 20})

    assert tuple(first_resolved) == tuple(second_resolved)
    assert json.dumps(thaw_json(first_resolved), separators=(",", ":")) == json.dumps(
        thaw_json(second_resolved), separators=(",", ":")
    )


def test_nested_profile_mappings_are_recursively_canonical_without_reordering_lists():
    left_nested = {
        "z": {"beta": 2, "alpha": 1},
        "a": [{"d": 4, "c": 3}, "keep-order"],
    }
    right_nested = {
        "a": [{"c": 3, "d": 4}, "keep-order"],
        "z": {"alpha": 1, "beta": 2},
    }
    base = {**EXPECTED_DEFAULTS["easy"], "alias_namespace_condition": left_nested}
    reverse = {
        **dict(reversed(tuple(EXPECTED_DEFAULTS["easy"].items()))),
        "alias_namespace_condition": right_nested,
    }
    first = ProfileSpec(
        name="easy",
        version="1",
        task_family="future",
        difficulty=Difficulty.EASY,
        parameters=base,
        allowed_overrides=("alias_namespace_condition",),
    )
    second = ProfileSpec(
        name="easy",
        version="1",
        task_family="future",
        difficulty=Difficulty.EASY,
        parameters=reverse,
        allowed_overrides=("alias_namespace_condition",),
    )
    first_resolved = resolve_profile(first, {"alias_namespace_condition": left_nested})
    second_resolved = resolve_profile(second, {"alias_namespace_condition": right_nested})

    assert tuple(first.parameters["alias_namespace_condition"]) == ("a", "z")
    assert tuple(first.parameters["alias_namespace_condition"]["z"]) == ("alpha", "beta")
    assert tuple(first.parameters["alias_namespace_condition"]["a"][0]) == ("c", "d")
    assert first.parameters["alias_namespace_condition"]["a"][1] == "keep-order"
    assert json.dumps(thaw_json(first_resolved)) == json.dumps(thaw_json(second_resolved))
    assert thaw_json(first_resolved) == thaw_json(second_resolved)


@pytest.mark.parametrize(
    "bad_nested",
    [
        {1: "non-string"},
        {"outer": {2: "non-string"}},
        {"outer": [{3: "non-string"}]},
    ],
)
def test_profile_rejects_nonstring_nested_mapping_keys_locally(bad_nested):
    parameters = {**EXPECTED_DEFAULTS["easy"], "alias_namespace_condition": bad_nested}
    with pytest.raises((TypeError, ValueError, ValidationError), match="string"):
        ProfileSpec(
            name="easy",
            version="1",
            task_family="future",
            difficulty=Difficulty.EASY,
            parameters=parameters,
            allowed_overrides=("alias_namespace_condition",),
        )


def test_resolver_rejects_nonstring_nested_override_keys_without_mutation():
    parameters = {
        **EXPECTED_DEFAULTS["easy"],
        "alias_namespace_condition": {"outer": {"key": "value"}},
    }
    profile = ProfileSpec(
        name="easy",
        version="1",
        task_family="future",
        difficulty=Difficulty.EASY,
        parameters=parameters,
        allowed_overrides=("alias_namespace_condition",),
    )
    override = {"alias_namespace_condition": {"outer": {1: "bad"}}}
    before = repr(override)
    with pytest.raises((TypeError, ValueError), match="string"):
        resolve_profile(profile, override)
    assert repr(override) == before


def test_profiles_and_resolved_output_are_recursively_immutable_and_do_not_alias_inputs():
    parameters = {**EXPECTED_DEFAULTS["easy"], "alias_namespace_condition": {"aliases": ["a"]}}
    overrides = {"alias_namespace_condition": {"aliases": ["b"]}}
    profile = ProfileSpec(
        name="easy",
        version="1.0.0",
        task_family="future",
        difficulty=Difficulty.EASY,
        parameters=parameters,
        allowed_overrides=("alias_namespace_condition",),
    )
    resolved = resolve_profile(profile, overrides)
    parameters["alias_namespace_condition"]["aliases"].append("mutated")
    overrides["alias_namespace_condition"]["aliases"].append("mutated")

    assert resolved["alias_namespace_condition"]["aliases"] == ("b",)
    with pytest.raises(TypeError):
        resolved["update_depth"] = 99
    with pytest.raises(TypeError):
        resolved["alias_namespace_condition"]["aliases"][0] = "x"
    dumped = profile.model_dump(mode="json")
    assert isinstance(dumped["parameters"], dict)
    assert isinstance(dumped["parameters"]["alias_namespace_condition"]["aliases"], list)
    assert json.loads(json.dumps(thaw_json(resolved))) == thaw_json(resolved)
    assert not isinstance(resolved, MappingProxyType)


@pytest.mark.parametrize(
    "unsafe",
    [
        {"nested"},
        frozenset({"nested"}),
        b"bytes",
        bytearray(b"bytes"),
        range(3),
        Decimal("1.5"),
        1 + 2j,
        object(),
        HostileJsonStr("hostile"),
        float("nan"),
        float("inf"),
    ],
)
def test_profile_parameters_reject_every_noncanonical_json_leaf_or_container(unsafe):
    parameters = {
        **EXPECTED_DEFAULTS["easy"],
        "alias_namespace_condition": {"nested": unsafe},
    }
    with pytest.raises((TypeError, ValueError, ValidationError), match="canonical JSON|finite"):
        ProfileSpec(
            name="easy",
            version="1",
            task_family="future",
            difficulty=Difficulty.EASY,
            parameters=parameters,
            allowed_overrides=("alias_namespace_condition",),
        )


def test_empty_sequence_default_cannot_admit_unsafe_nested_override():
    profile = ProfileSpec(
        name="easy",
        version="1",
        task_family="future",
        difficulty=Difficulty.EASY,
        parameters={
            **EXPECTED_DEFAULTS["easy"],
            "alias_namespace_condition": [],
        },
        allowed_overrides=("alias_namespace_condition",),
    )
    for unsafe in ({"unordered"}, object(), b"bytes", range(2)):
        with pytest.raises((TypeError, ValueError), match="canonical JSON"):
            resolve_profile(profile, {"alias_namespace_condition": [{"nested": unsafe}]})


def test_valid_nested_lists_preserve_order_freeze_and_dump_as_ordinary_json():
    source = {
        **EXPECTED_DEFAULTS["easy"],
        "alias_namespace_condition": [{"z": 1, "a": ["second", "first"]}],
    }
    profile = ProfileSpec(
        name="easy",
        version="1",
        task_family="future",
        difficulty=Difficulty.EASY,
        parameters=source,
        allowed_overrides=("alias_namespace_condition",),
    )
    source["alias_namespace_condition"][0]["a"].append("mutated")
    assert profile.parameters["alias_namespace_condition"][0]["a"] == (
        "second",
        "first",
    )
    assert json.loads(json.dumps(thaw_json(profile.parameters))) == thaw_json(
        profile.parameters
    )


@pytest.mark.parametrize(
    "bad_source",
    [
        {"update_depth"},
        frozenset({"update_depth"}),
        {"update_depth": True},
        "update_depth",
        range(1),
        (item for item in ("update_depth",)),
    ],
)
def test_allowed_overrides_rejects_unordered_or_odd_source_shapes(bad_source):
    with pytest.raises((TypeError, ValueError, ValidationError), match="ordered list or tuple"):
        ProfileSpec(
            name="easy",
            version="1",
            task_family="future",
            difficulty=Difficulty.EASY,
            parameters=EXPECTED_DEFAULTS["easy"],
            allowed_overrides=bad_source,
        )


@pytest.mark.parametrize(
    "bad_entry",
    [
        b"update_depth",
        bytearray(b"update_depth"),
        HostileJsonStr("update_depth"),
        object(),
        " update_depth ",
        "",
    ],
)
def test_allowed_overrides_rejects_noncanonical_entries_before_coercion(bad_entry):
    with pytest.raises(
        (TypeError, ValueError, ValidationError), match="canonical exact strings"
    ):
        ProfileSpec(
            name="easy",
            version="1",
            task_family="future",
            difficulty=Difficulty.EASY,
            parameters=EXPECTED_DEFAULTS["easy"],
            allowed_overrides=[bad_entry],
        )


@pytest.mark.parametrize(
    "bad_entry",
    [b"update_depth", bytearray(b"update_depth"), HostileJsonStr("update_depth"), object()],
)
def test_resolver_revalidation_rejects_constructed_bad_allowed_override_entry(bad_entry):
    profile = ProfileSpec.model_construct(
        name="easy",
        version="1",
        task_family="future",
        difficulty=Difficulty.EASY,
        parameters=EXPECTED_DEFAULTS["easy"],
        allowed_overrides=(bad_entry,),
    )
    with pytest.raises((TypeError, ValueError, ValidationError), match="canonical exact strings"):
        resolve_profile(profile)


def test_allowed_overrides_accepts_list_and_tuple_without_reordering():
    expected = ("context_length", "update_depth")
    for source in (list(expected), expected):
        profile = ProfileSpec(
            name="easy",
            version="1",
            task_family="future",
            difficulty=Difficulty.EASY,
            parameters=EXPECTED_DEFAULTS["easy"],
            allowed_overrides=source,
        )
        assert profile.allowed_overrides == expected
        assert json.loads(json.dumps(profile.model_dump(mode="json")))[
            "allowed_overrides"
        ] == list(expected)


def test_allowed_override_set_rejection_is_identical_across_hash_seeds():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    script = r'''
from mub.vnext.contracts import Difficulty
from mub.vnext.profiles import ProfileSpec
from tests.vnext.test_profiles import EXPECTED_DEFAULTS
try:
    ProfileSpec(name="easy", version="1", task_family="future", difficulty=Difficulty.EASY, parameters=EXPECTED_DEFAULTS["easy"], allowed_overrides={"update_depth", "context_length"})
except Exception as exc:
    print(type(exc).__name__, "ordered list or tuple" in str(exc))
'''
    outputs = []
    for seed in ("1", "777"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        env["PYTHONPATH"] = root
        outputs.append(
            subprocess.run(
                [sys.executable, "-c", script],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            ).stdout.strip()
        )
    assert outputs == ["TypeError True", "TypeError True"]


@pytest.mark.parametrize(
    "parameters,match",
    [
        ({key: value for key, value in EXPECTED_DEFAULTS["easy"].items() if key != "update_depth"}, "update_depth"),
        ({**EXPECTED_DEFAULTS["easy"], "update_depth": 0}, "positive"),
        ({**EXPECTED_DEFAULTS["easy"], "noop_density": float("nan")}, "finite"),
        ({**EXPECTED_DEFAULTS["easy"], "context_order": "  "}, "nonblank"),
        ({**EXPECTED_DEFAULTS["easy"], "cross_slot_interleaving": 1.1}, "between 0 and 1"),
        ({**EXPECTED_DEFAULTS["easy"], "alias_namespace_condition": {"nested": float("nan")}}, "finite"),
    ],
)
def test_profile_construction_validates_required_depth_and_every_default(parameters, match):
    with pytest.raises(ValidationError, match=match):
        ProfileSpec(
            name="easy",
            version="1",
            task_family="future",
            difficulty=Difficulty.EASY,
            parameters=parameters,
            allowed_overrides=(),
        )


@pytest.mark.parametrize(
    "parameters",
    [
        {key: value for key, value in EXPECTED_DEFAULTS["easy"].items() if key != "update_depth"},
        {**EXPECTED_DEFAULTS["easy"], "update_depth": 0},
        {**EXPECTED_DEFAULTS["easy"], "alias_namespace_condition": {"nested": object()}},
    ],
)
def test_resolver_rejects_model_construct_parameter_bypass(parameters):
    profile = ProfileSpec.model_construct(
        name="easy",
        version="1",
        task_family="future",
        difficulty=Difficulty.EASY,
        parameters=parameters,
        allowed_overrides=(),
    )
    with pytest.raises((TypeError, ValueError, ValidationError)):
        resolve_profile(profile)
