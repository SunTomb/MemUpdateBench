from pathlib import Path

from mub.vnext.generation.post_core_audit import select_main_track_audit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "data" / "vnext" / "main_track_v1"


def test_audit_selection_is_stratified_and_bound() -> None:
    selection = select_main_track_audit(CANDIDATE)
    assert selection["release_id"] == "main_track_v1"
    assert selection["selected_semantic_core_count"] == 60
    assert selection["selected_task_count"] == 240
    assert set(selection["family_counts"]) == {
        "interleaved_multi_slot_update",
        "entity_attribute_grounding",
        "noop_write_discipline",
    }
    assert set(selection["domain_counts"]) == {
        "personal", "work", "community", "services", "education", "travel",
        "household", "software", "finance", "health", "media", "civic",
    }
    assert set(selection["attribute_counts"]) == {
        "location", "company", "preference", "language", "timezone", "hobby",
        "instrument", "project", "role", "status", "priority", "contact_method",
    }
    assert set(selection["surface_counts"]) == {
        "en-US/explicit_canonical", "en-US/concise_natural",
        "es-ES/concise_natural", "ja-JP/concise_natural",
    }
    assert set(selection["split_counts"]) == {"train", "dev", "test"}
    assert selection["coverage"]["family_b"]["active_object_counts"] == [2, 4, 8, 12]
    assert len(selection["coverage"]["family_b"]["interleaving_patterns"]) == 3
    assert set(selection["coverage"]["family_c"]["entity_conditions"]) == {
        "distinct", "alias", "same_name", "namespace_collision",
    }
    assert set(selection["coverage"]["family_c"]["attribute_conditions"]) == {
        "exact", "paraphrase", "near_name",
    }
    assert set(selection["coverage"]["family_d"]["trap_types"]) == {
        "transient", "hypothetical", "negated", "uncertain", "semantic_near_miss",
        "duplicate_current", "unsupported_inference",
    }
    assert set(selection["coverage"]["family_d"]["noop_densities"]) == {0.25, 0.5, 0.75}
    assert len(selection["selected_task_hashes"]) == 240


def test_audit_selection_is_deterministic() -> None:
    assert select_main_track_audit(CANDIDATE) == select_main_track_audit(CANDIDATE)


def test_audit_selection_rejects_tampered_candidate(tmp_path: Path) -> None:
    for source in CANDIDATE.iterdir():
        target = tmp_path / source.name
        target.write_bytes(source.read_bytes())
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_bytes(tasks.read_bytes().replace(b"ADD", b"BAD", 1))
    import pytest
    with pytest.raises(ValueError):
        select_main_track_audit(tmp_path)
