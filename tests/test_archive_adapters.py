from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from control_evidence.archive_adapters import ArchiveSafetyError, inspect_funsd


def test_funsd_adapter_counts_entities_and_relations(tmp_path: Path):
    archive_path = tmp_path / "funsd.zip"
    payload = {"form": [{"linking": [[1, 2]]}, {"linking": []}]}
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("dataset/training_data/annotations/sample.json", json.dumps(payload))
    report = inspect_funsd(archive_path)
    assert report["annotation_files"] == 1
    assert report["entities"] == 2
    assert report["relations"] == 1


def test_archive_path_traversal_is_rejected(tmp_path: Path):
    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.json", "{}")
    with pytest.raises(ArchiveSafetyError, match="traversal"):
        inspect_funsd(archive_path)


def test_funsd_adapter_ignores_macos_resource_forks(tmp_path: Path):
    archive_path = tmp_path / "funsd-macos.zip"
    payload = {"form": [{"linking": []}]}
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("dataset/training_data/annotations/sample.json", json.dumps(payload))
        archive.writestr("__MACOSX/dataset/training_data/annotations/._sample.json", b"\x00\x05binary")
    report = inspect_funsd(archive_path)
    assert report["annotation_files"] == 1
    assert report["entities"] == 1


def test_docvqa_adapter_reads_parquet_metadata_when_dataset_extra_is_installed(tmp_path: Path):
    py7zr = pytest.importorskip("py7zr")
    pyarrow = pytest.importorskip("pyarrow")
    import pyarrow.parquet as parquet

    from control_evidence.archive_adapters import inspect_docvqa

    shard = tmp_path / "test-00000-of-00001.parquet"
    parquet.write_table(pyarrow.table({"question": ["a", "b"], "image": ["x", "y"]}), shard)
    archive_path = tmp_path / "docvqa.7z"
    with py7zr.SevenZipFile(archive_path, "w") as archive:
        archive.write(shard, arcname=shard.name)
    report = inspect_docvqa(archive_path)
    assert report["shard_count"] == 1
    assert report["rows"] == 2
    assert "no scored QA claim" in report["evaluation_caveat"]


def test_windows_style_archive_traversal_is_rejected(tmp_path: Path):
    archive_path = tmp_path / "windows-traversal.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(r"..\escape.json", "{}")
    with pytest.raises(ArchiveSafetyError, match="traversal"):
        inspect_funsd(archive_path)


def test_duplicate_paths_after_separator_normalization_are_rejected(tmp_path: Path):
    archive_path = tmp_path / "duplicates.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(r"dataset\annotations\sample.json", "{}")
        archive.writestr("dataset/annotations/sample.json", "{}")
    with pytest.raises(ArchiveSafetyError, match="duplicate"):
        inspect_funsd(archive_path)


def test_kleister_nested_xz_expansion_is_bounded(tmp_path: Path, monkeypatch):
    import lzma

    from control_evidence import archive_adapters

    archive_path = tmp_path / "kleister-bomb.zip"
    compressed = lzma.compress(b"a" * 1024)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("dataset/dev-0/in.tsv.xz", compressed)
        archive.writestr("dataset/dev-0/expected.tsv", "row\n")

    original = archive_adapters._decompress_xz_limited

    def tiny_limit(payload, *, max_bytes):
        return original(payload, max_bytes=32)

    monkeypatch.setattr(archive_adapters, "_decompress_xz_limited", tiny_limit)
    with pytest.raises(ArchiveSafetyError, match="XZ expansion"):
        archive_adapters.inspect_kleister(archive_path)
