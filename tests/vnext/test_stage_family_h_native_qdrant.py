from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts import stage_family_h_native_qdrant as stage


def test_stage_copies_release_manifest_without_replacing_it(tmp_path):
    root = Path(__file__).resolve().parents[2]
    source = root / "data/vnext/family_h_independent/v1_published"
    output = tmp_path / "stage"
    result = stage.stage(output, source_root=source, project_root=root)
    copied = output / "release" / "release_manifest.json"
    assert copied.read_bytes() == (source / "release_manifest.json").read_bytes()
    assert result["release_manifest_sha256"] == hashlib.sha256(copied.read_bytes()).hexdigest()
    assert json.loads(copied.read_bytes())["schema"] != "memupdatebench.family-h.native-source-manifest.v1"
    assert (output / "release" / "release_index.json").is_file()


def test_stage_rejects_existing_output_without_replacing(tmp_path):
    root = Path(__file__).resolve().parents[2]
    source = root / "data/vnext/family_h_independent/v1_published"
    output = tmp_path / "stage"
    output.mkdir()
    (output / "sentinel").write_bytes(b"foreign")
    try:
        stage.stage(output, source_root=source, project_root=root)
    except FileExistsError:
        pass
    else:
        raise AssertionError("stage must be no-replace")
    assert (output / "sentinel").read_bytes() == b"foreign"
