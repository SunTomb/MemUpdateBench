from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from mub.vnext.release.task14_contracts import TASK14_ARTIFACT_PATHS
from mub.vnext.release.task14_publish import (
    build_task14_publication_v1,
    publish_task14_review_v1,
    verify_task14_root_v1,
)
from mub.vnext.release.task14_sources import Task14SourcePathsV1, load_task14_sources_v1
from mub.vnext.statistics.task13_v3 import Task13RuntimeBindingV1


REPOSITORY = Path(__file__).resolve().parents[2]
REMOTE_STAGE = "/NAS/yesh/MemUpdateBench/results/vnext/.mub-task13-stage-1a791f4cbfdd471aa6a8bd45ab6432d4"


@pytest.fixture(autouse=True)
def fixed_runtime(monkeypatch) -> None:
    import mub.vnext.release.task14_publish as module

    monkeypatch.setattr(
        module,
        "current_clean_task13_runtime_v3",
        lambda root: Task13RuntimeBindingV1("e" * 40, "a" * 64),
    )


@pytest.fixture(scope="module")
def loaded_sources():
    return load_task14_sources_v1(
        Task14SourcePathsV1(
            core_root=REPOSITORY / "data/vnext/core/v3",
            evidence_root=REPOSITORY / "results/vnext/core_task14_evidence",
            task13_root=REPOSITORY / "results/vnext/core_task13_bc82566_v1",
            task13_audit_path=REPOSITORY / "results/vnext/core_task13_bc82566_v1_audit.json",
            repository_root=REPOSITORY,
            remote_task13_staging_path=REMOTE_STAGE,
        )
    )


def test_publication_hash_chain_is_acyclic_and_ready(loaded_sources) -> None:
    publication = build_task14_publication_v1(
        loaded_sources,
        review_id="task14-publication-test",
        trusted_source_revision="e" * 40,
        trusted_source_tree_sha256="a" * 64,
    )
    assert publication.report.status == "READY_FOR_VERIFICATION"
    assert publication.attestation.final_approval_at_verification is True
    assert tuple(publication.artifact_bytes) == TASK14_ARTIFACT_PATHS
    assert tuple(item.path for item in publication.manifest.artifacts) == TASK14_ARTIFACT_PATHS[:3]
    assert tuple(item.path for item in publication.index.artifacts) == TASK14_ARTIFACT_PATHS[:4]
    assert TASK14_ARTIFACT_PATHS[4] not in {item.path for item in publication.index.artifacts}


def test_atomic_publication_reopens_exact_five_file_final(loaded_sources, tmp_path: Path) -> None:
    output = tmp_path / "core-task14-final"
    result = publish_task14_review_v1(
        loaded_sources,
        review_id="task14-publication-test",
        trusted_source_revision="e" * 40,
        trusted_source_tree_sha256="a" * 64,
        output_root=output,
    )
    assert result.final_approved
    assert result.output_root == output.resolve()
    assert sorted(item.name for item in output.iterdir()) == sorted(TASK14_ARTIFACT_PATHS)
    reopened = verify_task14_root_v1(output)
    assert reopened.final_approved
    assert result.index_sha256 == hashlib.sha256((output / TASK14_ARTIFACT_PATHS[4]).read_bytes()).hexdigest()


def test_existing_output_root_is_never_overwritten(loaded_sources, tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    (output / "foreign").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        publish_task14_review_v1(
            loaded_sources,
            review_id="existing",
            trusted_source_revision="e" * 40,
        trusted_source_tree_sha256="a" * 64,
            output_root=output,
        )
    assert (output / "foreign").read_text(encoding="utf-8") == "keep"


def test_source_change_in_pre_publish_leaves_no_final(
    loaded_sources, tmp_path: Path, monkeypatch
) -> None:
    import mub.vnext.release.task14_publish as publication_module

    calls = 0

    def changing(_loaded):
        nonlocal calls
        calls += 1
        return calls < 3

    monkeypatch.setattr(publication_module, "revalidate_task14_sources_v1", changing)
    output = tmp_path / "rejected"
    with pytest.raises(RuntimeError, match="source"):
        publish_task14_review_v1(
            loaded_sources,
            review_id="source-race",
            trusted_source_revision="e" * 40,
        trusted_source_tree_sha256="a" * 64,
            output_root=output,
        )
    assert not output.exists()
    assert not tuple(tmp_path.glob(".mub-task14-stage-*"))


def test_reopen_rejects_index_tamper(loaded_sources, tmp_path: Path) -> None:
    output = tmp_path / "tampered"
    publish_task14_review_v1(
        loaded_sources,
        review_id="tamper",
        trusted_source_revision="e" * 40,
        trusted_source_tree_sha256="a" * 64,
        output_root=output,
    )
    index = output / TASK14_ARTIFACT_PATHS[4]
    raw = index.read_bytes()
    index.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(Exception):
        verify_task14_root_v1(output)
