from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tests.vnext.factories import (
    build_task,
    build_task_run,
    make_object_key as build_object_key,
    make_run_manifest as build_run_manifest,
    make_score_record as build_score_record,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def fixture_root() -> Path:
    return FIXTURE_ROOT


@pytest.fixture
def make_task():
    return build_task


@pytest.fixture
def make_object_key():
    return build_object_key


@pytest.fixture
def make_task_run():
    return build_task_run


@pytest.fixture
def make_score_record():
    return build_score_record


@pytest.fixture
def make_run_manifest():
    return build_run_manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
