from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from control_evidence.publication import PublicationError, TransactionalPublisher


def test_failed_publication_never_moves_latest(tmp_path):
    publisher = TransactionalPublisher(tmp_path)
    first = publisher.publish({"a.json": {"value": 1}}, run_id="first")
    assert publisher.latest() == first
    with pytest.raises(PublicationError, match="injected"):
        publisher.publish({"a.json": {"value": 2}, "b.json": {"value": 3}}, run_id="failed", fail_after=1)
    assert publisher.latest() == first
    assert not (tmp_path / "runs" / "failed").exists()


def test_manifest_hash_detects_tampering(tmp_path):
    publisher = TransactionalPublisher(tmp_path)
    run = publisher.publish({"artifact.json": {"ok": True}}, run_id="run-a")
    (run / "artifact.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(PublicationError, match="hash mismatch"):
        publisher.validate_run(run)


def test_same_run_id_is_idempotent_for_identical_artifacts(tmp_path):
    publisher = TransactionalPublisher(tmp_path)
    first = publisher.publish({"artifact.json": {"ok": True}}, run_id="stable")
    second = publisher.publish({"artifact.json": {"ok": True}}, run_id="stable")
    assert first == second


def test_same_run_id_rejects_different_artifacts(tmp_path):
    publisher = TransactionalPublisher(tmp_path)
    publisher.publish({"artifact.json": {"value": 1}}, run_id="stable")
    with pytest.raises(PublicationError, match="different artifacts"):
        publisher.publish({"artifact.json": {"value": 2}}, run_id="stable")


def test_concurrent_publishers_do_not_interleave(tmp_path):
    publisher = TransactionalPublisher(tmp_path)

    def publish(index: int):
        return publisher.publish({"artifact.json": {"index": index}}, run_id=f"run-{index}")

    with ThreadPoolExecutor(max_workers=4) as executor:
        runs = list(executor.map(publish, range(8)))
    assert len(set(runs)) == 8
    latest = publisher.latest()
    payload = json.loads((latest / "artifact.json").read_text(encoding="utf-8"))
    assert payload["index"] in range(8)
    assert not list((tmp_path / ".staging").iterdir())


def test_idempotent_retry_leaves_no_orphan_staging_directory(tmp_path):
    publisher = TransactionalPublisher(tmp_path)
    publisher.publish({"artifact.json": {"ok": True}}, run_id="same")
    publisher.publish({"artifact.json": {"ok": True}}, run_id="same")
    assert not list((tmp_path / ".staging").iterdir())


def test_lock_cleanup_retries_transient_permission_error(tmp_path, monkeypatch):
    import control_evidence.publication as publication

    lock = tmp_path / "lock"
    lock.write_text("owned", encoding="utf-8")
    original_unlink = Path.unlink
    attempts = 0

    def flaky_unlink(path, *args, **kwargs):
        nonlocal attempts
        if path == lock and attempts == 0:
            attempts += 1
            raise PermissionError("simulated transient Windows sharing violation")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)
    publication._unlink_lock_with_retry(lock, timeout_seconds=0.1)

    assert attempts == 1
    assert not lock.exists()


def test_stale_lock_is_recovered(tmp_path):
    import os
    import time

    publisher = TransactionalPublisher(tmp_path)
    publisher.lock_path.write_text("stale", encoding="utf-8")
    old = time.time() - 60
    os.utime(publisher.lock_path, (old, old))
    run = publisher.publish({"artifact.json": {"ok": True}}, run_id="recovered")
    assert run.is_dir()
    assert not publisher.lock_path.exists()


def test_manifest_rejects_unlisted_extra_file(tmp_path):
    publisher = TransactionalPublisher(tmp_path)
    run = publisher.publish({"artifact.json": {"ok": True}}, run_id="complete")
    (run / "unlisted.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(PublicationError, match="exactly cover"):
        publisher.validate_run(run)


def test_latest_pointer_cannot_escape_runs_root(tmp_path):
    publisher = TransactionalPublisher(tmp_path)
    publisher.latest_path.write_text("../outside\n", encoding="utf-8")
    with pytest.raises(PublicationError, match="LATEST pointer"):
        publisher.latest()


@pytest.mark.parametrize("run_id", ["", ".", "..", "../escape", "slash/name", "space name"])
def test_invalid_run_ids_are_rejected(tmp_path, run_id):
    publisher = TransactionalPublisher(tmp_path)
    with pytest.raises(ValueError, match="simple name"):
        publisher.publish({"artifact.json": {"ok": True}}, run_id=run_id)


def test_windows_lock_acquire_retries_transient_permission_error(monkeypatch, tmp_path):
    import errno
    import json
    import os
    import socket
    import time
    from pathlib import Path

    if os.name != "nt":
        pytest.skip("Windows-specific transient lock-acquisition behavior")

    from control_evidence import publication

    lock = tmp_path / "lock"
    lock.write_text(
        json.dumps(
            {
                "pid": -1,
                "hostname": socket.gethostname(),
                "created_at": 0,
            }
        ),
        encoding="utf-8",
    )
    old = time.time() - 3600
    os.utime(lock, (old, old))

    original_open = publication.os.open
    attempts = 0

    def transient_open(*args, **kwargs):
        nonlocal attempts

        path = Path(args[0])

        if path == lock and attempts == 0:
            attempts += 1
            raise PermissionError(
                errno.EACCES,
                "simulated transient Windows sharing violation",
                str(path),
            )

        return original_open(*args, **kwargs)

    monkeypatch.setattr(publication.os, "open", transient_open)

    with publication._exclusive_lock(
        lock,
        timeout_seconds=0.5,
        stale_after_seconds=0.01,
    ):
        assert lock.exists()

    assert attempts == 1
    assert not lock.exists()


def test_old_lock_owned_by_live_process_is_not_removed(tmp_path):
    import os
    import socket
    import time

    from control_evidence.publication import _exclusive_lock

    lock = tmp_path / "lock"
    lock.write_text(
        json.dumps({"pid": os.getpid(), "hostname": socket.gethostname(), "created_at": 0}),
        encoding="utf-8",
    )
    old = time.time() - 3600
    os.utime(lock, (old, old))
    with (
        pytest.raises(PublicationError, match="timed out"),
        _exclusive_lock(lock, timeout_seconds=0.03, stale_after_seconds=0.01),
    ):
        pass
    assert lock.exists()


def test_windows_pid_liveness_does_not_terminate_current_process():
    import os

    if os.name != "nt":
        pytest.skip("Windows-specific PID liveness behavior")

    from control_evidence.publication import _pid_alive

    assert _pid_alive(os.getpid()) is True


@pytest.mark.parametrize("run_id", ["CON", "nul.txt", "run.", "run ", "bad:name"])
def test_windows_unsafe_run_ids_are_rejected(tmp_path, run_id):
    publisher = TransactionalPublisher(tmp_path)
    with pytest.raises(ValueError, match="cross-platform"):
        publisher.publish({"artifact.json": {"ok": True}}, run_id=run_id)


@pytest.mark.parametrize("path", ["CON/file.json", "folder/NUL.txt", "folder/trailing. ", "bad:name.json"])
def test_windows_unsafe_artifact_paths_are_rejected(tmp_path, path):
    publisher = TransactionalPublisher(tmp_path)
    with pytest.raises(PublicationError, match="cross-platform"):
        publisher.publish({path: {"ok": True}}, run_id="portable")


def test_idempotent_retry_repairs_latest_pointer_after_commit_only_crash(tmp_path):
    publisher = TransactionalPublisher(tmp_path)
    previous = publisher.publish({"artifact.json": {"value": 1}}, run_id="previous")
    committed = publisher.publish({"artifact.json": {"value": 2}}, run_id="candidate")
    assert publisher.latest() == committed

    # Simulate a crash window after the committed run directory exists but
    # before the LATEST pointer durably refers to it.
    publisher.latest_path.write_text("previous\n", encoding="utf-8")
    assert publisher.latest() == previous

    retried = publisher.publish({"artifact.json": {"value": 2}}, run_id="candidate")

    assert retried == committed
    assert publisher.latest() == committed


@pytest.mark.parametrize(
    "failure",
    [OSError(28, "simulated no space left on device"), PermissionError("simulated permission denied")],
)
def test_write_failure_preserves_previous_latest_and_cleans_staging(tmp_path, monkeypatch, failure):
    import control_evidence.publication as publication

    publisher = TransactionalPublisher(tmp_path)
    previous = publisher.publish({"artifact.json": {"value": 1}}, run_id="previous")
    original = publication._write_bytes

    def fail_candidate_write(path, payload):
        if path.name == "artifact.txt" and ".staging" in path.parts:
            raise failure
        return original(path, payload)

    monkeypatch.setattr(publication, "_write_bytes", fail_candidate_write)
    with pytest.raises(type(failure)):
        publisher.publish({"artifact.txt": "candidate"}, run_id="candidate")

    assert publisher.latest() == previous
    assert not (tmp_path / "runs" / "candidate").exists()
    assert not list((tmp_path / ".staging").iterdir())
