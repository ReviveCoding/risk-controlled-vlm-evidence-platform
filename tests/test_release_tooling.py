from __future__ import annotations

import gzip
import importlib.util
import io
import tarfile
from pathlib import Path


def _load(name: str):
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_tar(path: Path, timestamp: int):
    with (
        path.open("wb") as raw,
        gzip.GzipFile(filename=path.name, mode="wb", fileobj=raw, mtime=timestamp) as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as archive,
    ):
        payload = b"payload"
        member = tarfile.TarInfo("package/file.txt")
        member.size = len(payload)
        member.mtime = timestamp
        archive.addfile(member, io.BytesIO(payload))


def test_normalized_sdist_is_reproducible(tmp_path):
    normalizer = _load("normalize_sdist")
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    _write_tar(first, 1)
    _write_tar(second, 2)
    first_out = tmp_path / "first-normalized.tar.gz"
    second_out = tmp_path / "second-normalized.tar.gz"
    normalizer.normalize_sdist(first, first_out)
    normalizer.normalize_sdist(second, second_out)
    assert first_out.read_bytes() == second_out.read_bytes()


def test_release_manifest_is_sorted_and_self_excluding(tmp_path):
    builder = _load("build_release_manifest")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    output = tmp_path / "reports" / "release_manifest.json"
    first = builder.build_manifest(tmp_path, output)
    first_bytes = output.read_bytes()
    second = builder.build_manifest(tmp_path, output)
    assert first == second
    assert output.read_bytes() == first_bytes
    assert [item["path"] for item in first["files"]] == ["a.txt", "b.txt"]


def test_release_zip_is_byte_reproducible(tmp_path):
    zipper = _load("create_release_zip")
    root = tmp_path / "root"
    root.mkdir()
    (root / "b.txt").write_text("b", encoding="utf-8")
    (root / "a.txt").write_text("a", encoding="utf-8")
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    zipper.create_zip(root, first, "project")
    zipper.create_zip(root, second, "project")
    assert first.read_bytes() == second.read_bytes()


def test_ci_actions_are_full_sha_pinned_and_docker_is_non_root():
    import re

    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    action_refs = re.findall(r"uses:\s*[^@\s]+@([^\s#]+)", workflow)
    assert action_refs
    assert all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in action_refs)
    assert "control-evidence version" in workflow
    assert "python -m mypy src/control_evidence" in workflow

    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    assert "USER appuser" in dockerfile
    assert "CONTROL_EVIDENCE_STATE_DIR=/app/state" in dockerfile
    assert "CONTROL_EVIDENCE_IDEMPOTENCY_RETENTION_SECONDS=2592000" in dockerfile
    assert "CONTROL_EVIDENCE_IDEMPOTENCY_MAX_RECORDS=100000" in dockerfile
    assert 'VOLUME ["/app/state"]' in dockerfile
    assert "http://127.0.0.1:8000/ready" in dockerfile
    assert "control_evidence.api:create_app_from_env" in dockerfile
    assert "docker build --tag risk-controlled-vlm-evidence-platform:ci ." in workflow


def test_handoff_fingerprints_and_diff_are_deterministic(tmp_path):
    builder = _load("build_release_candidate_handoff")
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (baseline / "a.txt").write_text("before", encoding="utf-8")
    (candidate / "a.txt").write_text("after", encoding="utf-8")
    (candidate / "b.txt").write_text("added", encoding="utf-8")

    first = builder.diff_entries(baseline, candidate)
    second = builder.diff_entries(baseline, candidate)

    assert first == second
    assert [item["status"] for item in first] == ["modified", "added"]
    assert builder.fingerprint(builder.source_entries(candidate)) == builder.fingerprint(
        builder.source_entries(candidate)
    )


def test_handoff_source_fingerprint_excludes_its_own_output(tmp_path):
    builder = _load("build_release_candidate_handoff")
    (tmp_path / "src.py").write_text("value = 1\n", encoding="utf-8")
    before = builder.source_entries(tmp_path)
    (tmp_path / "release_candidate_handoff.json").write_text("{}\n", encoding="utf-8")
    after = builder.source_entries(tmp_path)
    assert before == after


def test_handoff_source_fingerprint_excludes_coverage_artifact(tmp_path):
    builder = _load("build_release_candidate_handoff")
    (tmp_path / "src.py").write_text("value = 1\n", encoding="utf-8")
    before = builder.source_entries(tmp_path)
    (tmp_path / ".coverage").write_bytes(b"runtime coverage database")
    after = builder.source_entries(tmp_path)
    assert before == after
