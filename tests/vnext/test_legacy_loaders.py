from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Callable, get_type_hints

import pytest

from mub.vnext import legacy
from mub.vnext.legacy import (
    load_csv_rows,
    load_evomemory_dataset,
    load_evomemory_results,
    load_json_summary,
    parse_legacy_bool,
)
from mub.vnext.legacy import loaders


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "legacy"
P84_SYNTHETIC_PROMPTS = {
    "synthetic-api-1": "P84 synthetic prompt: chronological_none; stale_count=16; gold=Suzhou",
    "synthetic-api-2": "P84 synthetic prompt: reverse_chronological_none; stale_count=16; gold=Suzhou",
    "synthetic-api-3": "P84 synthetic prompt: reverse_chronological_none; stale_count=16; gold=Suzhou",
}


def _snapshot(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return hashlib.sha256(path.read_bytes()).hexdigest(), stat.st_mtime_ns, stat.st_size


@pytest.mark.parametrize(
    ("filename", "loader"),
    [
        ("p63_dataset_minimal.json", load_evomemory_dataset),
        ("evomemory_results_old.json", load_evomemory_results),
        ("evomemory_results_traced.json", load_evomemory_results),
        ("p65_prompt_summary_minimal.json", load_json_summary),
        ("p83_conflict_rows.csv", load_csv_rows),
        ("p83_synthetic_dose_rows.csv", load_csv_rows),
        ("p84_api_rows.csv", load_csv_rows),
    ],
)
def test_fixture_bytes_hash_and_mtime_are_unchanged(
    filename: str, loader: Callable[[Path], Any]
) -> None:
    path = FIXTURE_DIR / filename
    before = _snapshot(path)

    loader(path)

    assert _snapshot(path) == before


@pytest.mark.parametrize(
    "filename",
    [
        "p63_dataset_minimal.json",
        "evomemory_results_old.json",
        "evomemory_results_traced.json",
        "p65_prompt_summary_minimal.json",
        "p83_conflict_rows.csv",
        "p83_synthetic_dose_rows.csv",
        "p84_api_rows.csv",
    ],
)
def test_fixture_encoding_and_newline_policy(filename: str) -> None:
    raw = (FIXTURE_DIR / filename).read_bytes()

    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    raw.decode("utf-8", errors="strict")


def test_fixture_git_attributes_lock_lf_endings() -> None:
    attributes = (FIXTURE_DIR.parents[3] / ".gitattributes").read_text(encoding="utf-8")

    assert "tests/vnext/fixtures/legacy/*.json text eol=lf\n" in attributes
    assert "tests/vnext/fixtures/legacy/*.csv text eol=lf\n" in attributes


def test_dataset_fixture_preserves_exact_lifecycle_and_raw_presence_semantics() -> None:
    rows = load_evomemory_dataset(FIXTURE_DIR / "p63_dataset_minimal.json")

    assert [row["episode_id"] for row in rows] == ["synthetic-p63-k1", "synthetic-p63-k2"]
    assert [row["num_events"] for row in rows] == [3, 4]
    assert [row["num_target_updates"] for row in rows] == [1, 2]
    assert [row["num_updates"] for row in rows] == [3, 4]
    assert rows[0]["events"] == [
        "User says: my friend Alex is considering a trip to Paris.",
        "User says: my manager Alex lives in Berlin.",
        "User says: my friend Alex lives in Suzhou.",
    ]
    assert rows[0]["latest_event_idx"] == 2
    assert rows[1]["events"][0] == "User says: my friend Alex lives in Wuxi."
    assert rows[1]["events"][-1] == "User says: my friend Alex moved to Suzhou."
    assert rows[1]["latest_event_idx"] == 3
    assert rows[1]["answer"] == "Suzhou"
    assert rows[0]["same_name_distractor"]["entity"] == "manager_alex"
    assert rows[0]["same_name_distractor"]["surface_name"] == "Alex"
    assert rows[0]["semantic_near_miss"]["attribute"] == "location"
    assert rows[0]["explicit_zero"] == 0
    assert rows[0]["explicit_false"] is False
    assert rows[0]["explicit_null"] is None
    assert "optional_absent" not in rows[0]
    assert rows[0]["provenance"]["kind"] == "LegacyProvenance"
    assert rows[0]["provenance"]["synthetic"] is True
    assert "source_type" not in rows[0]


def test_old_results_preserve_legacy_dialect_and_omit_later_answer_fields() -> None:
    payload = load_evomemory_results(FIXTURE_DIR / "evomemory_results_old.json")
    summary = payload["summary"]
    row = payload["results"][0]

    assert set(payload) == {"summary", "results"}
    assert summary["legacy_analysis_metadata"]["legacy_phase"] == "P6.3"
    assert row["example_id"] == 0
    assert row["gold_answer"] == "Suzhou"
    assert row["predicted"] == "Wuxi"
    assert row["em"] == 0.0
    assert row["f1"] == 0.0
    assert row["value_em"] is False
    assert row["answer_value_present"] is False
    assert row["legacy_note"] is None
    assert "answer_topk" not in summary
    assert "context_order" not in summary
    assert "answer_trace" not in row


def test_traced_results_preserve_realistic_retrieval_trace_and_source_event_ids() -> None:
    payload = load_evomemory_results(FIXTURE_DIR / "evomemory_results_traced.json")
    assert set(payload) == {"summary", "results"}
    assert payload["summary"]["answer_topk"] == 2
    assert payload["summary"]["context_order"] == "reverse_chronological"
    trace = payload["results"][0]["answer_trace"]

    assert trace["answer_topk"] == 2
    assert trace["gold_value_in_retrieved"] is True
    assert trace["stale_same_slot_in_retrieved"] is True
    assert trace["stale_same_slot_values"] == ["Wuxi"]
    assert trace["source_event_ids"] == [
        "synthetic-p63-k2:event:3",
        "synthetic-p63-k2:event:0",
    ]
    assert trace["retrieved_entries"] == [
        {
            "rank": 1,
            "id": "synthetic-entry-current",
            "score": 0.91,
            "content": "User says: my friend Alex moved to Suzhou.",
            "slot": {
                "entity": "friend_alex",
                "attribute": "location",
                "value": "Suzhou",
                "event_idx": 3,
            },
            "source_event_id": "synthetic-p63-k2:event:3",
            "has_gold_value": True,
            "stale_same_slot": False,
        },
        {
            "rank": 2,
            "id": "synthetic-entry-stale",
            "score": 0.83,
            "content": "User says: my friend Alex lives in Wuxi.",
            "slot": {
                "entity": "friend_alex",
                "attribute": "location",
                "value": "Wuxi",
                "event_idx": 0,
            },
            "source_event_id": "synthetic-p63-k2:event:0",
            "has_gold_value": False,
            "stale_same_slot": True,
        },
    ]


def test_prompt_summary_is_summary_only_and_preserves_retrieval_rates() -> None:
    payload = load_json_summary(FIXTURE_DIR / "p65_prompt_summary_minimal.json")

    assert payload["analysis_metadata"]["legacy_phase"] == "P6.5"
    assert payload["rows"] == [
        {
            "method": "synthetic_raw_add",
            "update_depth": 1,
            "gold_retrieval_rate": 1.0,
            "stale_retrieval_rate": 0.0,
        },
        {
            "method": "synthetic_raw_add",
            "update_depth": 2,
            "gold_retrieval_rate": 0.5,
            "stale_retrieval_rate": 1.0,
        },
    ]
    assert "task_results" not in payload
    assert "answer_traces" not in payload


def test_csv_loaders_preserve_raw_probe_dialects_and_boolean_spellings() -> None:
    conflict = load_csv_rows(FIXTURE_DIR / "p83_conflict_rows.csv")
    dose = load_csv_rows(FIXTURE_DIR / "p83_synthetic_dose_rows.csv")

    assert list(conflict[0]) == [
        "condition",
        "example_id",
        "attribute",
        "entity",
        "gold_answer",
        "distractor_count",
        "predicted",
        "em",
        "f1",
        "value_em",
        "answer_value_present",
        "stale_value_copied",
        "prompt",
        "provenance",
    ]
    assert conflict[0]["condition"] == "stale_same_slot"
    assert [row["answer_value_present"] for row in conflict] == ["False", "True"]
    assert [row["stale_value_copied"] for row in conflict] == ["True", "False"]
    assert [row["value_em"] for row in conflict] == ["0.0", "1.0"]
    assert list(dose[0]) == [
        "condition",
        "example_id",
        "attribute",
        "stale_count",
        "value_policy",
        "context_order",
        "context_annotation",
        "gold_answer",
        "predicted",
        "em",
        "f1",
        "answer_value_present",
        "prompt",
        "provenance",
    ]
    assert [row["context_annotation"] for row in dose] == [
        "none",
        "latest_outdated_label",
    ]
    assert [row["em"] for row in dose] == ["0.0", "1.0"]
    assert "latest_label_present" not in dose[0]
    assert parse_legacy_bool(conflict[0]["answer_value_present"]) is False
    assert parse_legacy_bool(conflict[1]["answer_value_present"]) is True
    assert parse_legacy_bool(conflict[0]["stale_value_copied"]) is True
    assert parse_legacy_bool(conflict[1]["stale_value_copied"]) is False


def test_p84_fixture_preserves_probe_fields_and_failure_metadata() -> None:
    rows = load_csv_rows(FIXTURE_DIR / "p84_api_rows.csv")

    assert list(rows[0]) == [
        "example_id",
        "condition",
        "stale_count",
        "model",
        "prompt_sha256",
        "gold",
        "prediction",
        "raw_response",
        "em",
        "stale_copied",
        "latency_seconds",
        "row_status",
        "caveat",
        "capacity_failed",
        "provenance",
    ]
    assert [row["condition"] for row in rows] == [
        "chronological_none",
        "reverse_chronological_none",
        "reverse_chronological_none",
    ]
    assert [row["stale_count"] for row in rows] == ["16", "16", "16"]
    for row in rows:
        expected_prompt_hash = hashlib.sha256(
            P84_SYNTHETIC_PROMPTS[row["example_id"]].encode("utf-8")
        ).hexdigest()[:16]
        assert row["prompt_sha256"] == expected_prompt_hash
    assert [row["row_status"] for row in rows] == [
        "clean",
        "empty_truncated_response_caveat",
        "capacity_failed",
    ]
    assert rows[0]["prediction"] == "Suzhou"
    assert rows[0]["raw_response"] == "Suzhou"
    assert rows[1]["prediction"] == ""
    assert rows[1]["raw_response"] == ""
    assert rows[1]["caveat"] == "empty_or_truncated_response"
    assert rows[2]["capacity_failed"] == "1.0"
    assert all(row["provenance"] == "handwritten_synthetic" for row in rows)


@pytest.mark.parametrize(
    ("loader", "payload", "message"),
    [
        (load_evomemory_dataset, {}, "top-level JSON array"),
        (load_evomemory_dataset, [1], "record at index 0"),
        (load_evomemory_dataset, [], "must not be empty"),
        (load_evomemory_results, [], "top-level JSON object"),
        (load_json_summary, [], "top-level JSON object"),
    ],
)
def test_json_loaders_reject_invalid_top_level_shapes(
    tmp_path: Path,
    loader: Callable[[Path], Any],
    payload: Any,
    message: str,
) -> None:
    path = tmp_path / "wrong-shape.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message) as exc_info:
        loader(path)

    assert path.name in str(exc_info.value)


def test_json_loaders_reject_duplicate_keys_with_chained_context(tmp_path: Path) -> None:
    path = tmp_path / "duplicates.json"
    path.write_text('{"rows": [], "rows": [1]}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key") as exc_info:
        load_json_summary(path)

    assert path.name in str(exc_info.value)
    assert exc_info.value.__cause__ is not None


def test_duplicate_sensitive_json_key_diagnostics_are_bounded_and_redacted(
    tmp_path: Path,
) -> None:
    raw_key = "authorization-Bearer-LOADER-DUPLICATE-CANARY"
    path = tmp_path / "duplicate-sensitive.json"
    path.write_text(
        '{"' + raw_key + '":1,"' + raw_key + '":2}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key") as exc_info:
        load_json_summary(path)

    message = str(exc_info.value)
    assert raw_key not in message
    assert "LOADER-DUPLICATE-CANARY" not in message
    assert path.name in message
    assert len(message) < 800
    assert exc_info.value.__cause__ is not None


@pytest.mark.parametrize("loader", [load_evomemory_dataset, load_evomemory_results, load_json_summary])
def test_json_loaders_reject_malformed_json_with_chained_context(
    tmp_path: Path, loader: Callable[[Path], Any]
) -> None:
    path = tmp_path / "malformed.json"
    path.write_text('{"unfinished":', encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON") as exc_info:
        loader(path)

    assert path.name in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)


@pytest.mark.parametrize(
    "payload",
    [
        '{"value": NaN}',
        '{"value": Infinity}',
        '{"value": -Infinity}',
        '{"value": 1e400}',
        '{"nested": [{"value": NaN}]}',
    ],
)
def test_json_loaders_reject_nonfinite_numbers_without_touching_source(
    tmp_path: Path, payload: str
) -> None:
    path = tmp_path / "nonfinite.json"
    path.write_text(payload, encoding="utf-8")
    before = _snapshot(path)

    with pytest.raises(ValueError, match="invalid JSON number") as exc_info:
        load_json_summary(path)

    assert path.name in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert _snapshot(path) == before


def test_json_loader_wraps_integer_conversion_limit_with_path_and_cause(
    tmp_path: Path,
) -> None:
    path = tmp_path / "huge-int.json"
    path.write_text('{"value": ' + "9" * 5000 + "}", encoding="utf-8")
    before = _snapshot(path)

    with pytest.raises(ValueError, match="invalid JSON") as exc_info:
        load_json_summary(path)

    assert path.name in str(exc_info.value)
    assert type(exc_info.value.__cause__) is ValueError
    assert _snapshot(path) == before


def test_json_loader_wraps_recursion_error_with_path_and_cause(tmp_path: Path) -> None:
    path = tmp_path / "deep.json"
    path.write_text('{"value":' + "[" * 20000 + "0" + "]" * 20000 + "}", encoding="utf-8")
    before = _snapshot(path)

    with pytest.raises(ValueError, match="invalid JSON") as exc_info:
        load_json_summary(path)

    assert path.name in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RecursionError)
    assert _snapshot(path) == before


@pytest.mark.parametrize("loader", [load_evomemory_dataset, load_evomemory_results, load_json_summary, load_csv_rows])
def test_loaders_reject_malformed_utf8_with_chained_path_context(
    tmp_path: Path, loader: Callable[[Path], Any]
) -> None:
    suffix = ".csv" if loader is load_csv_rows else ".json"
    path = tmp_path / f"bad-utf8{suffix}"
    path.write_bytes(b"\xff\xfe")

    with pytest.raises(ValueError, match="UTF-8") as exc_info:
        loader(path)

    assert path.name in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, UnicodeDecodeError)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "must not be empty"),
        ("\n", "blank CSV header"),
        ("\r\n", "blank CSV header"),
        ("a,b\n", "at least one data row"),
        ("a,,b\n1,2,3\n", "blank CSV header"),
        ("a,b,a\n1,2,3\n", "duplicate CSV header"),
        ("a,b\n1\n", "row 2 has 1 fields; expected 2"),
        ("a,b\n1,2,3\n", "row 2 has 3 fields; expected 2"),
        ('a,b\n"unterminated,b\n', "invalid CSV"),
    ],
)
def test_csv_loader_rejects_empty_or_malformed_tables(
    tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "malformed.csv"
    path.write_text(content, encoding="utf-8", newline="")

    with pytest.raises(ValueError, match=message) as exc_info:
        load_csv_rows(path)

    assert path.name in str(exc_info.value)


def test_csv_loader_preserves_headers_empty_cells_and_embedded_newlines(tmp_path: Path) -> None:
    path = tmp_path / "newline.csv"
    path.write_bytes(b' Name,value\r\n"alpha\r\nbeta",\r\n')

    rows = load_csv_rows(path)

    assert rows == [{" Name": "alpha\r\nbeta", "value": ""}]


@pytest.mark.parametrize("loader", [load_evomemory_dataset, load_evomemory_results, load_json_summary, load_csv_rows])
def test_loaders_reject_missing_files_with_precise_path_context(
    tmp_path: Path, loader: Callable[[Path], Any]
) -> None:
    path = tmp_path / "missing.artifact"

    with pytest.raises(FileNotFoundError) as exc_info:
        loader(path)

    assert str(path) in str(exc_info.value)


@pytest.mark.parametrize("loader", [load_evomemory_dataset, load_evomemory_results, load_json_summary, load_csv_rows])
def test_loaders_reject_directories_with_precise_path_context(
    tmp_path: Path, loader: Callable[[Path], Any]
) -> None:
    with pytest.raises(IsADirectoryError) as exc_info:
        loader(tmp_path)

    assert str(tmp_path) in str(exc_info.value)


def test_loader_detects_a_changed_source_digest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "mutable.json"
    path.write_text("[]", encoding="utf-8")
    real_sha256_file = loaders._sha256_file
    calls = 0

    def report_changed_digest(candidate: Path) -> str:
        nonlocal calls
        calls += 1
        digest = real_sha256_file(candidate)
        return digest if calls == 1 else "0" * 64

    monkeypatch.setattr(loaders, "_sha256_file", report_changed_digest)

    with pytest.raises(RuntimeError, match="changed while being loaded") as exc_info:
        load_evomemory_dataset(path)

    assert path.name in str(exc_info.value)
    assert calls == 2


@pytest.mark.parametrize(
    ("loader", "content"),
    [
        (load_evomemory_dataset, "[1]"),
        (load_evomemory_results, "[]"),
        (load_json_summary, '{"x": 1, "x": 2}'),
        (load_csv_rows, "a,b\n1\n"),
    ],
)
def test_validation_failures_do_not_change_source_bytes_hash_or_mtime(
    tmp_path: Path,
    loader: Callable[[Path], Any],
    content: str,
) -> None:
    path = tmp_path / "invalid-artifact"
    path.write_text(content, encoding="utf-8", newline="")
    before = _snapshot(path)

    with pytest.raises(ValueError):
        loader(path)

    assert _snapshot(path) == before


@pytest.mark.parametrize(
    ("loader", "path"),
    [
        (load_evomemory_dataset, FIXTURE_DIR / "p63_dataset_minimal.json"),
        (load_evomemory_results, FIXTURE_DIR / "evomemory_results_traced.json"),
        (load_json_summary, FIXTURE_DIR / "p65_prompt_summary_minimal.json"),
        (load_csv_rows, FIXTURE_DIR / "p83_conflict_rows.csv"),
    ],
)
def test_repeated_loads_return_fresh_non_aliased_payloads(
    loader: Callable[[Path], Any], path: Path
) -> None:
    first = loader(path)
    second = loader(path)

    assert first == second
    assert first is not second
    if isinstance(first, list):
        assert first[0] is not second[0]
        first[0][next(iter(first[0]))] = "mutated"
    else:
        first[next(iter(first))] = "mutated"
    assert loader(path) == second


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("   \t", None),
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        (1.0, True),
        (0.0, False),
        ("True", True),
        (" false ", False),
        ("\tFaLsE\r\n", False),
        ("TRUE", True),
        ("0", False),
        ("1", True),
        ("0.0", False),
        (" 1.0 ", True),
    ],
)
def test_parse_legacy_bool_accepts_only_documented_forms(
    value: str | bool | int | float | None, expected: bool | None
) -> None:
    assert parse_legacy_bool(value) is expected


@pytest.mark.parametrize(
    "value",
    [
        2,
        -1,
        0.5,
        -0.0,
        float("nan"),
        float("inf"),
        float("-inf"),
        "yes",
        "00",
        "1e0",
        "nan",
        "inf",
        "-0.0",
        "falſe",
        "Ｆａｌｓｅ",
        "true ",
        " ",
    ],
)
def test_parse_legacy_bool_rejects_unsupported_values(value: Any) -> None:
    with pytest.raises(ValueError):
        parse_legacy_bool(value)


def test_parse_legacy_bool_rejects_huge_int_with_bounded_message() -> None:
    with pytest.raises(ValueError) as exc_info:
        parse_legacy_bool(10**5000)

    assert str(exc_info.value) == "Unsupported legacy boolean integer value"


def test_parse_legacy_bool_rejects_subclasses_enums_and_coercive_objects() -> None:
    class IntSubclass(int):
        pass

    class FloatSubclass(float):
        pass

    class StringSubclass(str):
        pass

    class PlainEnum(Enum):
        TRUE = 1

    class IntegerEnum(IntEnum):
        TRUE = 1

    class Coercive:
        calls = 0

        def __bool__(self) -> bool:
            self.calls += 1
            return True

        def __int__(self) -> int:
            self.calls += 1
            return 1

        def __float__(self) -> float:
            self.calls += 1
            return 1.0

    coercive = Coercive()
    unsupported = [
        IntSubclass(1),
        FloatSubclass(1.0),
        StringSubclass("true"),
        Decimal("1"),
        PlainEnum.TRUE,
        IntegerEnum.TRUE,
        b"true",
        [],
        {},
        coercive,
    ]
    for value in unsupported:
        with pytest.raises(TypeError, match="^Unsupported legacy boolean type$"):
            parse_legacy_bool(value)  # type: ignore[arg-type]
    assert coercive.calls == 0


def test_parse_legacy_bool_does_not_trigger_hostile_type_name_hooks() -> None:
    hook_calls: list[str] = []

    class HostileMeta(type):
        def __getattribute__(cls, name: str) -> Any:
            if name == "__name__":
                hook_calls.append(name)
                raise AssertionError("type name hook must not run")
            return super().__getattribute__(name)

    class Hostile(metaclass=HostileMeta):
        pass

    with pytest.raises(TypeError) as exc_info:
        parse_legacy_bool(Hostile())  # type: ignore[arg-type]

    assert str(exc_info.value) == "Unsupported legacy boolean type"
    assert hook_calls == []


def test_legacy_public_api_exports_and_type_hints_are_exact() -> None:
    assert legacy.__all__ == [
        "LEGACY_CAVEATS",
        "LEGACY_NAMESPACES",
        "compile_legacy_episode",
        "import_evomemory_results",
        "legacy_namespace",
        "load_csv_rows",
        "load_evomemory_dataset",
        "load_evomemory_results",
        "load_json_summary",
        "parse_legacy_bool",
        "parse_legacy_run_name",
    ]
    assert get_type_hints(load_evomemory_dataset) == {
        "path": Path,
        "return": list[dict[str, Any]],
    }
    assert get_type_hints(load_evomemory_results) == {
        "path": Path,
        "return": dict[str, Any],
    }
    assert get_type_hints(load_json_summary) == {
        "path": Path,
        "return": dict[str, Any],
    }
    assert get_type_hints(load_csv_rows) == {
        "path": Path,
        "return": list[dict[str, str]],
    }
    assert get_type_hints(parse_legacy_bool) == {
        "value": str | bool | int | float | None,
        "return": bool | None,
    }
