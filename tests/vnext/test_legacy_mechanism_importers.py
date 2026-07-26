from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mub.vnext.legacy import mechanisms
from mub.vnext.legacy.mechanisms import (
    ApiProbeCell,
    import_api_probe,
    import_conflict_probe,
    import_stale_removal_trace,
    import_synthetic_dose,
    select_clean_api_cells,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "legacy"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_task13_api_is_public_from_mechanisms_module() -> None:
    assert set(mechanisms.__all__) == {
        "ApiProbeCell",
        "ConflictProbeCell",
        "StaleRemovalTraceCell",
        "SyntheticDoseCell",
        "import_api_probe",
        "import_conflict_probe",
        "import_stale_removal_trace",
        "import_synthetic_dose",
        "select_clean_api_cells",
    }


def test_p83_condition_order_annotation_and_stale_count_axes_remain_separate() -> None:
    conflict_path = FIXTURE_DIR / "p83_conflict_rows.csv"
    dose_path = FIXTURE_DIR / "p83_synthetic_dose_rows.csv"

    conflict = import_conflict_probe(conflict_path)
    dose = import_synthetic_dose(dose_path)

    assert [cell.surface_condition for cell in conflict] == [
        "stale_same_slot",
        "unrelated_distractors",
    ]
    assert [cell.value_policy for cell in dose] == ["conflict", "conflict"]
    assert [cell.context_order for cell in dose] == [
        "reverse_chronological",
        "reverse_chronological",
    ]
    assert [cell.context_annotation for cell in dose] == [
        "none",
        "latest_outdated_label",
    ]
    assert [cell.stale_count for cell in dose] == [8, 16]
    assert dose[0].surface_condition == "conflict_stale8_reverse_chronological_none"
    assert dose[0].config_sha256 != dose[1].config_sha256
    assert all(cell.legacy_namespace == "legacy_p83" for cell in [*conflict, *dose])
    assert all(cell.source_sha256 == _sha256(dose_path) for cell in dose)


def test_stale_removal_original_em_is_trace_composition_not_answer_rerun() -> None:
    cells = import_stale_removal_trace(
        FIXTURE_DIR / "p83_stale_removal_rows.csv"
    )

    assert [cell.surface_condition for cell in cells] == [
        "normal",
        "remove_stale_same_slot",
    ]
    assert all(cell.original_em_avg == 0.13 for cell in cells)
    assert all(
        cell.original_score_semantics == "trace_composition_not_answer_rerun"
        for cell in cells
    )
    assert all(
        "original_scores_are_trace_composition_not_answer_rerun" in cell.caveats
        for cell in cells
    )


def test_p84_preserves_prompt_sha_and_raw_response_as_distinct_fields() -> None:
    cells = import_api_probe(FIXTURE_DIR / "p84_api_state_rows.csv")
    gpt = next(cell for cell in cells if cell.model == "gpt-5.5")

    assert gpt.prompt_sha256 == "0123456789abcdef"
    assert gpt.raw_response == "Suzhou"
    assert gpt.prompt_sha256 != gpt.raw_response
    assert gpt.raw_response_path is None
    assert gpt.raw_response_sha256 is None
    assert gpt.source_sha256 == _sha256(FIXTURE_DIR / "p84_api_state_rows.csv")
    assert gpt.legacy_namespace == "legacy_p84"


def test_gemini_25_flash_is_imported_with_explicit_format_caveat() -> None:
    cells = import_api_probe(FIXTURE_DIR / "p84_api_state_rows.csv")
    flash = next(cell for cell in cells if cell.model == "gemini-2.5-flash")

    assert flash.status == "format_caveat"
    assert flash.is_completed is False
    assert "empty_or_truncated_response" in flash.caveats


def test_capacity_failed_gemini_25_pro_cannot_become_completed() -> None:
    cells = import_api_probe(FIXTURE_DIR / "p84_api_state_rows.csv")
    pro = next(cell for cell in cells if cell.model == "gemini-2.5-pro")

    assert pro.status == "capacity_failed"
    assert pro.is_completed is False
    assert pro.em is None
    with pytest.raises(ValidationError, match="capacity_failed"):
        pro.validated_replace(status="completed", is_completed=True)


def test_clean_api_selection_is_a_non_destructive_filtered_view() -> None:
    cells = import_api_probe(FIXTURE_DIR / "p84_api_state_rows.csv")
    original = tuple(cells)

    selected = select_clean_api_cells(cells)

    assert [cell.model for cell in selected] == [
        "gpt-5.5",
        "gemini-3.1-flash-lite-preview",
    ]
    assert tuple(cells) == original
    assert any(cell.model == "gemini-2.5-flash" for cell in cells)
    assert all(cell.status == "completed" for cell in selected)
    selected.clear()
    assert tuple(cells) == original


def test_mechanism_cells_are_analysis_records_not_invented_canonical_tasks() -> None:
    cells = [
        *import_conflict_probe(FIXTURE_DIR / "p83_conflict_rows.csv"),
        *import_synthetic_dose(FIXTURE_DIR / "p83_synthetic_dose_rows.csv"),
        *import_stale_removal_trace(FIXTURE_DIR / "p83_stale_removal_rows.csv"),
        *import_api_probe(FIXTURE_DIR / "p84_api_state_rows.csv"),
    ]

    for cell in cells:
        assert cell.compatibility_analysis_only is True
        assert "source_type" not in type(cell).model_fields
        assert "task_family" not in type(cell).model_fields
        with pytest.raises(ValidationError):
            cell.source_path = "changed.csv"


def test_api_statuses_include_pending_and_model_unavailable_without_completion(
    tmp_path: Path,
) -> None:
    path = tmp_path / "states.csv"
    path.write_text(
        "model,condition,stale_count,n,row_status,prompt_sha256,em,stale_copied,raw_response\n"
        "model-a,chronological_none,1,2,pending,,,,\n"
        "model-b,chronological_none,1,2,model_unavailable,,,,\n",
        encoding="utf-8",
    )

    cells = import_api_probe(path)

    assert [(cell.status, cell.is_completed) for cell in cells] == [
        ("pending", False),
        ("model_unavailable", False),
    ]


def test_importers_reject_coercive_integer_and_nonfinite_json(tmp_path: Path) -> None:
    csv_path = tmp_path / "coercive.csv"
    csv_path.write_text(
        "condition,n,stale_count,value_policy,context_order,context_annotation,em\n"
        "conflict,1,1e0,conflict,chronological,none,1.0\n",
        encoding="utf-8",
    )
    json_path = tmp_path / "nonfinite.json"
    json_path.write_text(
        '[{"condition":"final_only","n":1,"em":NaN}]',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="stale_count"):
        import_synthetic_dose(csv_path)
    with pytest.raises(ValueError, match="invalid JSON number"):
        import_conflict_probe(json_path)


def test_p83_preserves_safe_paired_raw_response_references_without_inline_payload(
    tmp_path: Path,
) -> None:
    raw_hash = "a" * 64
    cases = [
        (
            "conflict.csv",
            import_conflict_probe,
            "condition,n,em,raw_response,raw_response_path,raw_response_sha256\n"
            f"final_only,1,1.0,synthetic private payload,artifacts/raw.json,{raw_hash}\n",
        ),
        (
            "dose.csv",
            import_synthetic_dose,
            "value_policy,context_order,context_annotation,stale_count,n,em,raw_response,raw_response_path,raw_response_sha256\n"
            f"conflict,chronological,none,1,1,1.0,synthetic private payload,artifacts/raw.json,{raw_hash}\n",
        ),
        (
            "removal.csv",
            import_stale_removal_trace,
            "intervention,n,original_em_avg,raw_response,raw_response_path,raw_response_sha256\n"
            f"normal,1,0.5,synthetic private payload,artifacts/raw.json,{raw_hash}\n",
        ),
    ]

    for filename, importer, content in cases:
        path = tmp_path / filename
        path.write_text(content, encoding="utf-8")
        cell = importer(path)[0]
        assert cell.raw_response_path == "artifacts/raw.json"
        assert cell.raw_response_sha256 == raw_hash
        assert cell.raw_response is None

@pytest.mark.parametrize(
    "raw_path,raw_hash,message",
    [
        ("artifacts/raw.json", "", "supplied together"),
        ("../raw.json", "b" * 64, "repository-relative"),
        ("private/raw.json", "b" * 64, "private"),
        ("artifacts/raw.json", "B" * 64, "SHA-256"),
    ],
)
def test_p83_rejects_unpaired_unsafe_private_or_noncanonical_raw_references(
    tmp_path: Path,
    raw_path: str,
    raw_hash: str,
    message: str,
) -> None:
    path = tmp_path / "conflict.csv"
    path.write_text(
        "condition,n,em,raw_response_path,raw_response_sha256\n"
        f"final_only,1,1.0,{raw_path},{raw_hash}\n",
        encoding="utf-8",
    )

    with pytest.raises((ValueError, ValidationError), match=message):
        import_conflict_probe(path)



@pytest.mark.parametrize(
    "raw_path",
    [
        "private./raw.json",
        "artifacts/CON",
        "artifacts/aux.txt",
        "artifacts/LPT1.json",
        "artifacts/trailing. /raw.json",
        "artifacts/trailing-space /raw.json",
    ],
)
def test_raw_response_paths_reject_win32_ambiguous_segments(
    tmp_path: Path,
    raw_path: str,
) -> None:
    path = tmp_path / "conflict.csv"
    path.write_text(
        "condition,n,em,raw_response_path,raw_response_sha256\n"
        f"final_only,1,1.0,{raw_path},{'c' * 64}\n",
        encoding="utf-8",
    )

    with pytest.raises((ValueError, ValidationError), match="Win32-ambiguous"):
        import_conflict_probe(path)


def test_synthetic_dose_accepts_matching_or_missing_condition_and_rejects_contradiction(
    tmp_path: Path,
) -> None:
    derived = "conflict_stale8_reverse_chronological_none"
    matching = tmp_path / "matching.json"
    missing = tmp_path / "missing.json"
    contradictory = tmp_path / "contradictory.json"
    base = {
        "value_policy": "conflict",
        "context_order": "reverse_chronological",
        "context_annotation": "none",
        "stale_count": 8,
        "n": 1,
        "em": 0.0,
    }
    matching.write_text(
        json.dumps([{**base, "condition": derived}]), encoding="utf-8"
    )
    missing.write_text(json.dumps([base]), encoding="utf-8")
    contradictory.write_text(
        json.dumps([{**base, "condition": "conflict_stale4_chronological_none"}]),
        encoding="utf-8",
    )

    assert import_synthetic_dose(matching)[0].surface_condition == derived
    assert import_synthetic_dose(missing)[0].surface_condition == derived
    with pytest.raises(ValueError, match="contradicts writer-derived"):
        import_synthetic_dose(contradictory)


def test_explicit_blank_or_whitespace_gpt_raw_response_is_format_caveat_and_excluded(
    tmp_path: Path,
) -> None:
    path = tmp_path / "blank-api.csv"
    path.write_text(
        "model,condition,stale_count,n,row_status,prompt_sha256,em,stale_copied,raw_response\n"
        "gpt-5.5,chronological_none,16,1,clean,0123456789abcdef,1.0,0.0,\n"
        'gpt-5.4,chronological_none,16,1,clean,1111111111111111,1.0,0.0,"   "\n',
        encoding="utf-8",
    )

    cells = import_api_probe(path)

    assert [
        (
            cell.raw_response_present,
            cell.raw_response,
            cell.status,
            cell.is_completed,
        )
        for cell in cells
    ] == [
        (True, "", "format_caveat", False),
        (True, "   ", "format_caveat", False),
    ]
    assert select_clean_api_cells(cells) == []
    assert all("empty_or_truncated_response" in cell.caveats for cell in cells)


def test_absent_aggregate_raw_response_uses_metrics_while_nonblank_gpt_stays_clean(
    tmp_path: Path,
) -> None:
    aggregate = tmp_path / "aggregate.json"
    aggregate.write_text(
        json.dumps(
            [
                {
                    "model": "gpt-5.5",
                    "condition": "chronological_none",
                    "stale_count": 16,
                    "n": 8,
                    "em": 1.0,
                    "stale_copied": 0.0,
                }
            ]
        ),
        encoding="utf-8",
    )

    aggregate_cell = import_api_probe(aggregate)[0]
    nonblank_cell = next(
        cell
        for cell in import_api_probe(FIXTURE_DIR / "p84_api_state_rows.csv")
        if cell.model == "gpt-5.5"
    )

    assert aggregate_cell.raw_response_present is False
    assert aggregate_cell.raw_response is None
    assert aggregate_cell.status == "completed"
    assert select_clean_api_cells([aggregate_cell]) == [aggregate_cell]
    assert nonblank_cell.raw_response_present is True
    assert nonblank_cell.raw_response == "Suzhou"
    assert nonblank_cell.status == "completed"
    assert select_clean_api_cells([nonblank_cell]) == [nonblank_cell]


def test_explicit_null_api_response_preserves_evidence_and_is_not_clean(
    tmp_path: Path,
) -> None:
    path = tmp_path / "null-response.json"
    path.write_text(
        json.dumps(
            [
                {
                    "model": "gpt-5.5",
                    "condition": "chronological_none",
                    "n": 1,
                    "row_status": "clean",
                    "em": 1.0,
                    "stale_copied": 0.0,
                    "raw_response": None,
                }
            ]
        ),
        encoding="utf-8",
    )

    cell = import_api_probe(path)[0]

    assert cell.raw_response_present is True
    assert cell.raw_response is None
    assert cell.status == "format_caveat"
    assert cell.is_completed is False
    assert select_clean_api_cells([cell]) == []


def test_api_model_rejects_completed_blank_null_or_fabricated_response_evidence(
    tmp_path: Path,
) -> None:
    aggregate = tmp_path / "aggregate.json"
    aggregate.write_text(
        json.dumps(
            [
                {
                    "model": "gpt-5.5",
                    "condition": "chronological_none",
                    "n": 1,
                    "em": 1.0,
                    "stale_copied": 0.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    aggregate_cell = import_api_probe(aggregate)[0]
    nonblank_cell = next(
        cell
        for cell in import_api_probe(FIXTURE_DIR / "p84_api_state_rows.csv")
        if cell.model == "gpt-5.5"
    )

    for replacement in ({"raw_response": ""}, {"raw_response": "   "}, {"raw_response": None}):
        with pytest.raises(ValidationError, match="completed"):
            nonblank_cell.validated_replace(**replacement)
    with pytest.raises(ValidationError):
        nonblank_cell.validated_replace(
            raw_response_present=False,
            raw_response=None,
        )
    with pytest.raises(ValidationError):
        aggregate_cell.validated_replace(
            raw_response_present=True,
            raw_response="",
        )


def test_api_cell_rejects_unpaired_or_unsafe_raw_response_reference() -> None:
    data = {
        "source_path": "probe.csv",
        "source_sha256": "0" * 64,
        "legacy_namespace": "legacy_p84",
        "surface_condition": "chronological_none",
        "sample_count": 1,
        "config_sha256": "1" * 64,
        "prompt_sha256": "0123456789abcdef",
        "raw_response": "Suzhou",
        "raw_response_present": True,
        "raw_response_path": "raw/response.json",
        "raw_response_sha256": None,
        "caveats": ("p84_answer_layer_only",),
        "compatibility_analysis_only": True,
        "model": "gpt-5.5",
        "stale_count": 1,
        "status": "completed",
        "is_completed": True,
        "em": 1.0,
        "stale_copied": 0.0,
    }

    with pytest.raises(ValidationError, match="raw_response_path"):
        ApiProbeCell.model_validate(data)

    data["raw_response_sha256"] = "2" * 64
    data["raw_response_path"] = "../private/raw.json"
    with pytest.raises(ValidationError, match="repository-relative"):
        ApiProbeCell.model_validate(data)


def test_source_snapshot_cannot_return_swapped_parse_with_original_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "conflict.csv"
    original = "condition,n,em\nfinal_only,1,1.0\n"
    swapped = b"condition,n,em\nstale_same_slot,1,0.0\n"
    path.write_bytes(original.encode("utf-8"))
    expected_hash = hashlib.sha256(original.encode("utf-8")).hexdigest()

    def parse_swapped(_path: Path, parser):
        return parser(swapped, path)

    monkeypatch.setattr(mechanisms, "_load_stable", parse_swapped, raising=False)

    cell = import_conflict_probe(path)[0]

    assert cell.surface_condition == "final_only"
    assert cell.source_sha256 == expected_hash


def test_source_size_cap_is_enforced_on_descriptor_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "large.csv"
    path.write_text(
        "condition,n,em\nfinal_only,1,1.0\n",
        encoding="utf-8",
    )
    real_open = Path.open

    class InflatedReader:
        def __init__(self, candidate: Path) -> None:
            self.candidate = candidate
            self.handle = None

        def __enter__(self):
            self.handle = real_open(self.candidate, "rb")
            return self

        def __exit__(self, *args):
            assert self.handle is not None
            self.handle.close()

        def fileno(self):
            assert self.handle is not None
            return self.handle.fileno()

        def read(self, size: int):
            return b"x" * (size + 1)

    def inflated_open(candidate: Path, *args, **kwargs):
        if candidate == path and args == ("rb",):
            return InflatedReader(candidate)
        return real_open(candidate, *args, **kwargs)

    monkeypatch.setattr(mechanisms, "_MAX_SOURCE_BYTES", 64)
    monkeypatch.setattr(Path, "open", inflated_open)

    with pytest.raises(ValueError, match="source byte cap"):
        import_conflict_probe(path)


def test_mechanism_model_identity_invariants_survive_validated_replace() -> None:
    conflict = import_conflict_probe(FIXTURE_DIR / "p83_conflict_rows.csv")[0]
    dose = import_synthetic_dose(FIXTURE_DIR / "p83_synthetic_dose_rows.csv")[0]
    removal = import_stale_removal_trace(
        FIXTURE_DIR / "p83_stale_removal_rows.csv"
    )[0]
    api = next(
        cell
        for cell in import_api_probe(FIXTURE_DIR / "p84_api_state_rows.csv")
        if cell.model == "gpt-5.5"
    )

    with pytest.raises(ValidationError, match="config_sha256"):
        conflict.validated_replace(distractor_count=99)
    with pytest.raises(ValidationError, match="writer-derived"):
        dose.validated_replace(stale_count=99)
    with pytest.raises(ValidationError, match="surface_condition|config_sha256"):
        removal.validated_replace(intervention="remove_unrelated")
    with pytest.raises(ValidationError, match="config_sha256"):
        api.validated_replace(model="gpt-5.4")


def test_em_drop_from_final_only_is_bounded_in_model_and_importer(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid.json"
    invalid = tmp_path / "invalid.json"
    valid.write_text(
        json.dumps(
            [
                {"condition": "low", "n": 1, "em": 0.0, "em_drop_from_final_only": -1.0},
                {"condition": "high", "n": 1, "em": 1.0, "em_drop_from_final_only": 1.0},
            ]
        ),
        encoding="utf-8",
    )
    invalid.write_text(
        json.dumps(
            [{"condition": "bad", "n": 1, "em": 0.0, "em_drop_from_final_only": 1.0001}]
        ),
        encoding="utf-8",
    )

    cells = import_conflict_probe(valid)
    assert [cell.em_drop_from_final_only for cell in cells] == [-1.0, 1.0]
    with pytest.raises(ValueError, match="em_drop_from_final_only"):
        import_conflict_probe(invalid)
    with pytest.raises(ValidationError):
        cells[0].validated_replace(em_drop_from_final_only=-1.0001)


@pytest.mark.parametrize(
    "row_status,status",
    [
        ("clean", "capacity_failed"),
        ("clean", "model_unavailable"),
        ("clean", "pending"),
        ("clean", "format_caveat"),
    ],
)
def test_api_status_alias_conflicts_fail_closed(
    tmp_path: Path,
    row_status: str,
    status: str,
) -> None:
    path = tmp_path / "aliases.csv"
    path.write_text(
        "model,condition,n,row_status,status,em,stale_copied,raw_response\n"
        f"gpt-5.5,chronological_none,1,{row_status},{status},1.0,0.0,Suzhou\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="alias conflict"):
        import_api_probe(path)


def test_api_status_aliases_accept_agreeing_completion_evidence(tmp_path: Path) -> None:
    path = tmp_path / "aliases.csv"
    path.write_text(
        "model,condition,n,row_status,status,em,stale_copied,raw_response\n"
        "gpt-5.5,chronological_none,1,clean,completed,1.0,0.0,Suzhou\n",
        encoding="utf-8",
    )

    assert import_api_probe(path)[0].status == "completed"


@pytest.mark.parametrize("reserved", ["model", "condition", "stale_count"])
def test_nested_api_metrics_cannot_overwrite_outer_identity(
    tmp_path: Path,
    reserved: str,
) -> None:
    path = tmp_path / "nested.json"
    metrics = {"n": 1, "em": 1.0, "stale_copied": 0.0, reserved: "evil"}
    path.write_text(
        json.dumps(
            {
                "model": "gpt-5.5",
                "by_condition_and_stale_count": {
                    "chronological_none": {"16": metrics}
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reserved nested metric key"):
        import_api_probe(path)


@pytest.mark.parametrize(
    "credential_key",
    ["api_key", "metadata.access_token", "ｍｅｔａｄａｔａ.ａｐｉ＿ｋｅｙ"],
)
def test_recursive_credential_scan_rejects_nested_and_unicode_aliases_without_leak(
    tmp_path: Path,
    credential_key: str,
) -> None:
    path = tmp_path / "credentials.json"
    segments = credential_key.split(".")
    secret_value = "DO-NOT-LEAK-THIS-VALUE"
    credential: object = secret_value
    for segment in reversed(segments):
        credential = {segment: credential}
    payload = {
        "model": "gpt-5.5",
        "by_condition": {
            "final_only": {"n": 1, "em": 1.0, "stale_copied": 0.0}
        },
        "metadata": credential,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="sensitive credential") as exc_info:
        import_api_probe(path)

    assert secret_value not in str(exc_info.value)


def test_recursive_scan_allows_narrow_token_telemetry_and_config(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.json"
    path.write_text(
        json.dumps(
            {
                "model": "gpt-5.5",
                "by_condition": {
                    "final_only": {"n": 1, "em": 1.0, "stale_copied": 0.0}
                },
                "metadata": {
                    "prompt_tokens": 12,
                    "completion_tokens": 3,
                    "max_tokens": 100,
                    "tokenizer": "synthetic",
                },
            }
        ),
        encoding="utf-8",
    )

    assert import_api_probe(path)[0].status == "completed"


def test_clean_api_selection_uses_exact_immutable_model_allowlist(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            [
                {
                    "model": model,
                    "condition": "chronological_none",
                    "n": 1,
                    "em": 1.0,
                    "stale_copied": 0.0,
                }
                for model in [
                    "gpt-5.5",
                    "gpt-5.4",
                    "gpt-5.4-mini",
                    "gemini-3.1-flash-lite-preview",
                    "gpt-6",
                    "gpt-custom",
                ]
            ]
        ),
        encoding="utf-8",
    )

    cells = import_api_probe(path)
    selected = select_clean_api_cells(cells)

    assert [cell.model for cell in selected] == [
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gemini-3.1-flash-lite-preview",
    ]
    assert len(cells) == 6


@pytest.mark.parametrize("digest_length", [16, 64])
@pytest.mark.parametrize("kind", ["p83", "p84"])
def test_inline_unicode_prompt_authenticates_supplied_digest(
    tmp_path: Path,
    digest_length: int,
    kind: str,
) -> None:
    prompt = "请回答：朋友住在苏州"
    full_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    supplied = full_digest if digest_length == 64 else full_digest[:16]
    path = tmp_path / f"{kind}-{digest_length}.json"
    if kind == "p83":
        payload = [
            {
                "condition": "final_only",
                "n": 1,
                "em": 1.0,
                "prompt": prompt,
                "prompt_sha256": supplied,
            }
        ]
        importer = import_conflict_probe
    else:
        payload = [
            {
                "model": "gpt-5.5",
                "condition": "chronological_none",
                "n": 1,
                "em": 1.0,
                "stale_copied": 0.0,
                "raw_response": "苏州",
                "prompt": prompt,
                "prompt_sha256": supplied,
            }
        ]
        importer = import_api_probe
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    assert importer(path)[0].prompt_sha256 == supplied


@pytest.mark.parametrize("digest_length", [16, 64])
@pytest.mark.parametrize("kind", ["p83", "p84"])
def test_inline_prompt_digest_mismatch_fails_without_prompt_leak(
    tmp_path: Path,
    digest_length: int,
    kind: str,
) -> None:
    prompt = "PRIVATE SYNTHETIC PROMPT MUST NOT LEAK"
    supplied = "0" * digest_length
    path = tmp_path / f"mismatch-{kind}-{digest_length}.json"
    base = {
        "condition": "final_only" if kind == "p83" else "chronological_none",
        "n": 1,
        "em": 1.0,
        "prompt": prompt,
        "prompt_sha256": supplied,
    }
    if kind == "p84":
        base.update(
            {
                "model": "gpt-5.5",
                "stale_copied": 0.0,
                "raw_response": "Suzhou",
            }
        )
    path.write_text(json.dumps([base]), encoding="utf-8")
    importer = import_conflict_probe if kind == "p83" else import_api_probe

    with pytest.raises(ValueError, match="does not authenticate inline prompt") as exc_info:
        importer(path)

    assert prompt not in str(exc_info.value)


@pytest.mark.parametrize("kind", ["p83", "p84"])
@pytest.mark.parametrize("prompt_value", [None, "", "   \t"])
@pytest.mark.parametrize("with_digest", [False, True])
def test_explicit_invalid_prompt_fails_closed_regardless_of_digest(
    tmp_path: Path,
    kind: str,
    prompt_value,
    with_digest: bool,
) -> None:
    path = tmp_path / f"invalid-prompt-{kind}.json"
    row = {
        "condition": "final_only" if kind == "p83" else "chronological_none",
        "n": 1,
        "em": 1.0,
        "prompt": prompt_value,
    }
    if with_digest:
        row["prompt_sha256"] = "0" * 16
    if kind == "p84":
        row.update(
            {
                "model": "gpt-5.5",
                "stale_copied": 0.0,
                "raw_response": "Suzhou",
            }
        )
    path.write_text(json.dumps([row]), encoding="utf-8")
    importer = import_conflict_probe if kind == "p83" else import_api_probe

    with pytest.raises(ValueError, match="prompt.*non-blank") as exc_info:
        importer(path)

    assert "prompt_sha256='" not in str(exc_info.value)


@pytest.mark.parametrize("kind", ["p83", "p84"])
@pytest.mark.parametrize("digest_length", [16, 64])
def test_absent_prompt_preserves_standalone_valid_digest(
    tmp_path: Path,
    kind: str,
    digest_length: int,
) -> None:
    digest = "a" * digest_length
    path = tmp_path / f"standalone-{kind}-{digest_length}.json"
    row = {
        "condition": "final_only" if kind == "p83" else "chronological_none",
        "n": 1,
        "em": 1.0,
        "prompt_sha256": digest,
    }
    if kind == "p84":
        row.update(
            {
                "model": "gpt-5.5",
                "stale_copied": 0.0,
                "raw_response": "Suzhou",
            }
        )
    path.write_text(json.dumps([row]), encoding="utf-8")
    importer = import_conflict_probe if kind == "p83" else import_api_probe

    assert importer(path)[0].prompt_sha256 == digest


@pytest.mark.parametrize("kind", ["p83", "p84"])
def test_absent_prompt_and_digest_remain_absent(
    tmp_path: Path,
    kind: str,
) -> None:
    path = tmp_path / f"absent-{kind}.json"
    row = {
        "condition": "final_only" if kind == "p83" else "chronological_none",
        "n": 1,
        "em": 1.0,
    }
    if kind == "p84":
        row.update(
            {
                "model": "gpt-5.5",
                "stale_copied": 0.0,
                "raw_response": "Suzhou",
            }
        )
    path.write_text(json.dumps([row]), encoding="utf-8")
    importer = import_conflict_probe if kind == "p83" else import_api_probe

    assert importer(path)[0].prompt_sha256 is None


def test_prompt_digest_is_derived_only_when_absent(tmp_path: Path) -> None:
    prompt = "Unicode prompt: 杭州"
    path = tmp_path / "derived.json"
    path.write_text(
        json.dumps(
            [{"condition": "final_only", "n": 1, "em": 1.0, "prompt": prompt}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert import_conflict_probe(path)[0].prompt_sha256 == hashlib.sha256(
        prompt.encode("utf-8")
    ).hexdigest()


@pytest.mark.parametrize(
    "importer,fixture",
    [
        (import_conflict_probe, "p83_conflict_rows.csv"),
        (import_synthetic_dose, "p83_synthetic_dose_rows.csv"),
        (import_stale_removal_trace, "p83_stale_removal_rows.csv"),
    ],
)
def test_p83_models_reject_inline_raw_response_directly_and_on_replace(
    importer,
    fixture: str,
) -> None:
    cell = importer(FIXTURE_DIR / fixture)[0]
    data = cell.model_dump(mode="python")
    data["raw_response"] = "synthetic private payload"

    with pytest.raises(ValidationError, match="P8.3.*inline raw_response"):
        type(cell).model_validate(data)
    with pytest.raises(ValidationError, match="P8.3.*inline raw_response"):
        cell.validated_replace(raw_response="synthetic private payload")


@pytest.mark.parametrize("control", ["\x00", "\n", "\t", "\x7f"])
def test_raw_response_paths_reject_c0_and_del_without_echoing_path(
    tmp_path: Path,
    control: str,
) -> None:
    sensitive_path = f"artifacts/{control}PRIVATE-PATH.json"
    path = tmp_path / "control.json"
    path.write_text(
        json.dumps(
            [
                {
                    "condition": "final_only",
                    "n": 1,
                    "em": 1.0,
                    "raw_response_path": sensitive_path,
                    "raw_response_sha256": "d" * 64,
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="control character") as exc_info:
        import_conflict_probe(path)

    assert sensitive_path not in str(exc_info.value)


def test_valid_unicode_raw_response_path_remains_portable(tmp_path: Path) -> None:
    raw_path = "artifacts/响应/苏州.json"
    path = tmp_path / "unicode-path.json"
    path.write_text(
        json.dumps(
            [
                {
                    "condition": "final_only",
                    "n": 1,
                    "em": 1.0,
                    "raw_response_path": raw_path,
                    "raw_response_sha256": "e" * 64,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert import_conflict_probe(path)[0].raw_response_path == raw_path


def test_mechanism_import_rejects_credential_columns(tmp_path: Path) -> None:
    path = tmp_path / "credential.csv"
    path.write_text(
        "condition,n,em,api_key\nfinal_only,1,1.0,not-a-real-key\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sensitive credential"):
        import_conflict_probe(path)
