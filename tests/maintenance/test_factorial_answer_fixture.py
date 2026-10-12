from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts import vnext_plan_main_track_factorial as planner
from tests.vnext import test_main_track_factorial_answer as answer_tests
from tests.vnext import test_main_track_factorial_manager as manager_tests


@pytest.mark.parametrize("published", [False, True])
def test_answer_fixture_shares_manifest_with_imported_manager_and_restores(
    tmp_path, monkeypatch, published
):
    path = tmp_path / "published.json"
    original_manager = tmp_path / "manager-original.json"
    if published:
        path.write_bytes(b'{"preserve":"original"}')
    monkeypatch.setattr(answer_tests, "MANIFEST", path)
    monkeypatch.setattr(manager_tests, "MANIFEST", original_manager)
    calls = []

    def build(candidate, audit):
        calls.append((candidate, audit))
        return {"fixture": "test-only"}

    monkeypatch.setattr(planner, "build_factorial_manifest", build)
    factory = SimpleNamespace(mktemp=lambda name: tmp_path)
    fixture = answer_tests._use_answer_test_manifest.__wrapped__(factory)
    next(fixture)
    try:
        assert manager_tests.MANIFEST == answer_tests.MANIFEST
        assert answer_tests.MANIFEST.is_file()
        if published:
            assert path.read_bytes() == b'{"preserve":"original"}'
            assert not calls
        else:
            assert calls == [(answer_tests.CANDIDATE, answer_tests.AUDIT)]
            assert not path.exists()
    finally:
        fixture.close()
    assert answer_tests.MANIFEST == path
    assert manager_tests.MANIFEST == original_manager
