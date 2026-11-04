from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from control_evidence.provenance import sha256_file


@pytest.mark.parametrize("size", [0, 19, 2 * 1024 * 1024 + 17])
@pytest.mark.parametrize("hasher", [sha256_file])
def test_file_digest_streams_without_reading_entire_file(tmp_path, monkeypatch, size, hasher):
    payload = (b"streamed-digest\x00\xff" * (size // 17 + 1))[:size]
    path = tmp_path / "fixture.bin"
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    def reject_whole_file_read(*args, **kwargs):
        raise AssertionError("file hashing must use bounded memory")

    monkeypatch.setattr(Path, "read_bytes", reject_whole_file_read)
    assert hasher(path) == expected
